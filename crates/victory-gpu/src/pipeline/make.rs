//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    build: Mutex<BuildState>,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
    prebuilt: OnceLock<bool>,
}

#[derive(Default)]
struct BuildState {
    phase: BuildPhase,
    retried: bool,
}

#[derive(Default)]
enum BuildPhase {
    #[default]
    Idle,
    Queued(Work),
    Building,
    RetryAfter(Instant),
    Complete,
}

const RETRY_DELAY: Duration = Duration::from_secs(2);
const RETRY_HEADROOM: usize = 2 << 30;
const MEMORY_POLL: Duration = Duration::from_millis(200);
static COMMIT_HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Work {
    CachedDraw,
    Draw,
    Retry,
    CachedEarly,
    Early,
}

const PRIORITY: [Work; 5] = [Work::CachedDraw, Work::Draw, Work::Retry, Work::CachedEarly, Work::Early];

impl Work {
    fn for_request(urgent: bool, prebuilt: bool, retry: bool) -> Self {
        match (urgent, prebuilt, retry) {
            (_, _, true) => Self::Retry,
            (true, true, false) => Self::CachedDraw,
            (true, false, false) => Self::Draw,
            (false, true, false) => Self::CachedEarly,
            (false, false, false) => Self::Early,
        }
    }

    fn early(self) -> bool {
        matches!(self, Self::CachedEarly | Self::Early)
    }

    fn can_start(self, headroom: usize) -> bool {
        headroom >= match self {
            Self::CachedDraw => 768 << 20,
            Self::Draw => 1536 << 20,
            Self::Retry | Self::CachedEarly => RETRY_HEADROOM,
            Self::Early => 3 << 30,
        }
    }
}

pub(crate) fn note_commit_headroom(available: usize) {
    COMMIT_HEADROOM.store(available as u64, Ordering::Relaxed);
}

/// Frames can be seconds apart while the title loads. Workers also sample
/// headroom while the queues are active or waiting for memory.
fn sample_headroom() -> usize {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_none_or(|last| last.elapsed() >= MEMORY_POLL) {
        *last = Some(Instant::now());
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: a valid output buffer of the advertised size.
        if unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0 {
            note_commit_headroom(info.CommitLimit.saturating_sub(info.CommitTotal).saturating_mul(info.PageSize));
        }
    }
    COMMIT_HEADROOM.load(Ordering::Relaxed) as usize
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        Pending {
            key,
            shaders,
            graphics,
            build: Mutex::default(),
            made: OnceLock::new(),
            used: AtomicBool::new(false),
            prebuilt: OnceLock::new(),
        }
    }

    fn make(&self) {
        let (started, cpu) = (Instant::now(), super::thread_cpu());
        let mut out_of_memory = false;
        let made = HostPipeline::build(&self.shaders, self.graphics.as_deref(), |code| {
            out_of_memory = code == windows::Win32::Foundation::E_OUTOFMEMORY;
        });
        #[cfg(feature = "diagnostics")]
        log_built(&self.key);
        let (user, kernel) = super::thread_cpu_since(cpu);
        tracing::debug!(
            shaders = ?self.shaders,
            ms = started.elapsed().as_millis() as u64,
            user_ms = user,
            kernel_ms = kernel,
            "pipeline made"
        );
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut state = self.build.lock().unwrap_or_else(|e| e.into_inner());
        if out_of_memory && !state.retried {
            state.phase = BuildPhase::RetryAfter(Instant::now() + RETRY_DELAY);
        } else {
            let _ = self.made.set(made);
            state.phase = BuildPhase::Complete;
        }
        done.notify_all();
    }

    fn claim(&self, work: Work) -> bool {
        let mut state = self.build.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(state.phase, BuildPhase::Queued(expected) if expected == work) {
            state.phase = BuildPhase::Building;
            true
        } else {
            false
        }
    }

    fn queued_for(&self, work: Work) -> bool {
        let state = self.build.lock().unwrap_or_else(|e| e.into_inner());
        matches!(state.phase, BuildPhase::Queued(expected) if expected == work)
    }

    fn will_progress(&self, headroom: usize) -> bool {
        let state = self.build.lock().unwrap_or_else(|e| e.into_inner());
        match state.phase {
            BuildPhase::Building => true,
            BuildPhase::Queued(work) => work.can_start(headroom),
            _ => false,
        }
    }
}

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None if matches!(
            p.build.lock().unwrap_or_else(|e| e.into_inner()).phase,
            BuildPhase::RetryAfter(_)
        ) => Some("out of memory, retry pending".into()),
        None if matches!(
            p.build.lock().unwrap_or_else(|e| e.into_inner()).phase,
            BuildPhase::Queued(work) if !work.can_start(COMMIT_HEADROOM.load(Ordering::Relaxed) as usize)
        ) => Some("waiting for commit headroom".into()),
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    if let Some(work) = enqueue(p, now) {
        queue(p.clone(), work);
    }
}

/// The queue for this request. Don't initialize the disk library on the
/// recording thread just to classify work before its first successful load.
fn enqueue(p: &Pending, now: bool) -> Option<Work> {
    let prebuilt = super::library::LOADED.load(Ordering::Relaxed) != 0
        && *p.prebuilt.get_or_init(|| super::library::prebuilt(&p.key));
    enqueue_with_headroom(p, now, COMMIT_HEADROOM.load(Ordering::Relaxed) as usize, prebuilt)
}

fn enqueue_with_headroom(p: &Pending, now: bool, headroom: usize, prebuilt: bool) -> Option<Work> {
    let mut state = p.build.lock().unwrap_or_else(|e| e.into_inner());
    let first = match state.phase {
        BuildPhase::Idle => true,
        BuildPhase::Queued(work) if work.early() && now => false,
        BuildPhase::RetryAfter(deadline)
            if now && headroom >= RETRY_HEADROOM && Instant::now() >= deadline =>
        {
            state.retried = true;
            true
        }
        _ => return None,
    };
    let work = Work::for_request(now, prebuilt, state.retried);
    state.phase = BuildPhase::Queued(work);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        if state.retried {
            tracing::info!(shaders = ?p.shaders, "retrying host pipeline after allocation failure");
        }
    }
    Some(work)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = BUDGET.saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.will_progress(COMMIT_HEADROOM.load(Ordering::Relaxed) as usize) {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest.min(MEMORY_POLL)).unwrap_or_else(|e| e.into_inner()).0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Pipelines for the workers: those draws wait for, then those started
/// early.
type PipelineQueues = [VecDeque<Arc<Pending>>; PRIORITY.len()];
static QUEUES: (Mutex<PipelineQueues>, Condvar) = (
    Mutex::new([const { VecDeque::new() }; PRIORITY.len()]),
    Condvar::new(),
);

fn next_work(queues: &mut PipelineQueues, headroom: usize) -> Option<(Work, Arc<Pending>)> {
    PRIORITY
        .into_iter()
        .filter(|work| work.can_start(headroom))
        .find_map(|work| queues[work as usize].pop_front().map(|p| (work, p)))
}

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>, job: Work) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let default = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        // Parallel driver compiles can exhaust commit during world loading.
        let n = std::env::var("VICTORY_PIPELINE_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&n| (1..=8).contains(&n))
            .unwrap_or(default);
        tracing::info!(workers = n, "starting host pipeline workers");
        (0..n)
            .filter(|i| {
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn(work);
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if p.claim(job) {
            if !job.can_start(sample_headroom()) {
                let mut state = p.build.lock().unwrap_or_else(|e| e.into_inner());
                state.phase = if state.retried {
                    state.retried = false;
                    BuildPhase::RetryAfter(Instant::now())
                } else {
                    BuildPhase::Idle
                };
                WAITING.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    lock.lock().unwrap_or_else(|e| e.into_inner())[job as usize].push_back(p);
    work.notify_one();
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before).
fn work() {
    // Below the title's own threads (loading, recording).
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    let (lock, work) = &QUEUES;
    loop {
        let (job, p) = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(next) = next_work(&mut queues, sample_headroom()) {
                    break next;
                }
                // Promoted jobs still have an old early entry. Drop that
                // reference even if memory pressure keeps early work paused.
                for job in PRIORITY {
                    queues[job as usize].retain(|p| p.queued_for(job));
                }
                let blocked = queues.iter().any(|queue| !queue.is_empty());
                let timeout = if blocked { MEMORY_POLL } else { Duration::from_secs(20) };
                let (held, waited) = work
                    .wait_timeout(queues, timeout)
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() && !blocked {
                    drop(queues);
                    super::library::save(false);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        if !p.claim(job) {
            continue; // its other queue entry
        }
        p.make();
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An early start is queued once; a draw waiting for it puts it ahead,
    /// once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let enqueue = |p: &Pending, now| enqueue_with_headroom(p, now, usize::MAX, false);
        let p = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&p, false), Some(Work::Early));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(Work::Draw));
        assert!(!p.claim(Work::Early), "the old background entry is stale");
        assert!(p.claim(Work::Draw));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&q, false), Some(Work::Early));
        assert!(q.claim(Work::Early));
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&r, true), Some(Work::Draw));
        assert_eq!(enqueue(&r, false), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(3, Ordering::Relaxed);
    }

    #[test]
    fn allocation_failure_can_requeue_once_on_demand() {
        let p = Pending::new(Vec::new(), None);
        {
            let mut state = p.build.lock().unwrap();
            state.phase = BuildPhase::RetryAfter(Instant::now() + RETRY_DELAY);
        }
        assert!(!p.will_progress(usize::MAX));
        assert_eq!(enqueue(&p, true), None);
        {
            let mut state = p.build.lock().unwrap();
            state.phase = BuildPhase::RetryAfter(Instant::now());
        }
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue_with_headroom(&p, true, RETRY_HEADROOM - 1, false), None);
        assert_eq!(enqueue_with_headroom(&p, true, RETRY_HEADROOM, false), Some(Work::Retry));
        assert_eq!(enqueue(&p, true), None);
        assert!(p.claim(Work::Retry));
        assert!(p.build.lock().unwrap().retried);
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }

    #[test]
    fn workers_pause_cold_work_and_resume_in_priority_order() {
        let mut queues: PipelineQueues = Default::default();
        for job in PRIORITY {
            queues[job as usize].push_back(Arc::new(Pending::new(Vec::new(), None)));
        }
        assert!(next_work(&mut queues, 512 << 20).is_none());
        assert_eq!(next_work(&mut queues, 1 << 30).unwrap().0, Work::CachedDraw);
        assert!(next_work(&mut queues, 1 << 30).is_none());
        assert_eq!(next_work(&mut queues, 2 << 30).unwrap().0, Work::Draw);
        assert_eq!(next_work(&mut queues, 2 << 30).unwrap().0, Work::Retry);
        assert_eq!(next_work(&mut queues, 2 << 30).unwrap().0, Work::CachedEarly);
        assert!(next_work(&mut queues, 2 << 30).is_none());
        assert_eq!(next_work(&mut queues, 3 << 30).unwrap().0, Work::Early);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }
}
