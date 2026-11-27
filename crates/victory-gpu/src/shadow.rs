//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    #[cfg(feature = "diagnostics")]
    buffer_bytes: u64,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
}

#[cfg(feature = "diagnostics")]
static SHADOW_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn buffer_usage() -> u64 {
    SHADOW_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Before-images and armed pages for one copy in a single submission.
#[derive(Default)]
pub(crate) struct Submission {
    pending: std::collections::HashSet<(usize, usize)>,
    before: Snapshot,
    armed: bool,
}

#[derive(Default)]
struct Snapshot {
    /// Adjacent before-images stay separate to avoid recopying old bytes.
    segments: BTreeMap<usize, Vec<u8>>,
    /// Union of the segments, for fast coverage and gap queries.
    covered: Vec<(usize, usize)>,
    /// Bytes already retired or excluded due to a missing shader input.
    compared: Vec<(usize, usize)>,
}

/// Gaps in a sorted union of ranges, clipped to `[from, to)`.
fn gaps(ranges: &[(usize, usize)], from: usize, to: usize) -> impl Iterator<Item = (usize, usize)> {
    let first = ranges.partition_point(|&(_, end)| end <= from);
    let mut at = from;
    ranges[first..]
        .iter()
        .copied()
        .take_while(move |&(start, _)| start < to)
        .chain(std::iter::once((to, to)))
        .filter_map(move |(start, end)| {
            let gap = (at < start).then_some((at, start));
            at = at.max(end);
            gap
        })
}

impl Snapshot {
    /// Start at the segment covering `from`, if there is one.
    fn first_segment(&self, from: usize) -> usize {
        self.segments
            .range(..=from)
            .next_back()
            .filter(|(at, bytes)| **at + bytes.len() > from)
            .map_or(from, |(&at, _)| at)
    }

    fn capture(
        &mut self,
        from: usize,
        to: usize,
        mut read: impl FnMut(usize, usize, &mut Vec<u8>),
    ) -> usize {
        let mut captured = 0;
        for (start, end) in gaps(&self.covered, from, to) {
            let mut bytes = pooled(end - start);
            read(start, end, &mut bytes);
            self.segments.insert(start, bytes);
            captured += end - start;
        }
        if captured != 0 {
            crate::host::add_range(&mut self.covered, from, to);
        }
        captured
    }

    fn covers(&self, from: usize, to: usize) -> bool {
        let first = self.covered.partition_point(|&(_, end)| end <= from);
        self.covered
            .get(first)
            .is_some_and(|&(start, end)| start <= from && end >= to)
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
struct SnapshotPool {
    buffers: Vec<Vec<u8>>,
    budget: usize,
}

static POOL: Mutex<SnapshotPool> = Mutex::new(SnapshotPool {
    buffers: Vec::new(),
    budget: POOL_BYTES,
});

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    if n < MIN_POOLED_BYTES {
        return Vec::with_capacity(n);
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool.buffers
        .iter()
        .enumerate()
        .filter(|(_, v)| (n..=n.saturating_mul(2)).contains(&v.capacity()))
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.buffers.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    if v.capacity() < MIN_POOLED_BYTES {
        return;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if v.capacity() > pool.budget {
        return;
    }
    pool.buffers.push(v);
    pool.buffers.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let budget = pool.budget;
    let mut total = 0;
    pool.buffers.retain(|v| {
        if v.capacity() > budget - total {
            return false;
        }
        total += v.capacity();
        true
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;
// Many small gaps can be captured per frame. Keep them out of the pool's
// search and eviction lists, which are for the expensive large snapshots.
const MIN_POOLED_BYTES: usize = 1 << 20;

/// Inactive snapshots can give their commit back immediately. Restore the
/// pool after pressure subsides, with a gap to avoid toggling each sample.
pub(crate) fn limit_snapshot_pool(available_commit: usize) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if available_commit < 1 << 30 {
        pool.budget = 0;
        pool.buffers.clear();
    } else if available_commit >= 2 << 30 {
        pool.budget = POOL_BYTES;
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn snapshot_pool_usage() -> (usize, usize) {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    (pool.buffers.len(), pool.buffers.iter().map(Vec::capacity).sum())
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// Log destination ranges before the kernel writes them, for checking
/// crash addresses against the write-back trace.
#[cfg(feature = "diagnostics")]
fn trace_copy_back(start: usize, bytes: usize, path: &'static str) {
    tracing::debug!(
        target: "victory_gpu::writeback",
        start = format_args!("{start:#x}"),
        bytes,
        path,
        "GPU copy-back to title memory"
    );
}

#[cfg(feature = "diagnostics")]
pub(crate) mod writeback_audit {
    use std::cell::Cell;
    use std::io::{BufWriter, Seek, SeekFrom, Write};
    use std::sync::{Mutex, OnceLock};

    const RECORDS: u64 = 65_536;
    struct Journal {
        file: BufWriter<std::fs::File>,
        sequence: u64,
    }
    static JOURNAL: OnceLock<Option<Mutex<Journal>>> = OnceLock::new();
    thread_local! { static FENCE: Cell<u64> = const { Cell::new(0) }; }

    pub(crate) struct Retirement;

    impl Retirement {
        pub(crate) fn begin(fence: u64) -> Self {
            JOURNAL.get_or_init(|| {
                let path = std::env::var_os("VICTORY_WRITEBACK_AUDIT")?;
                match std::fs::File::create(path) {
                    Ok(file) => Some(Mutex::new(Journal { file: BufWriter::new(file), sequence: 0 })),
                    Err(error) => {
                        tracing::warn!(%error, "could not open write-back audit");
                        None
                    }
                }
            });
            FENCE.set(fence);
            Self
        }
    }

    impl Drop for Retirement {
        fn drop(&mut self) {
            FENCE.set(0);
            if let Some(Some(journal)) = JOURNAL.get()
                && let Err(error) = journal.lock().unwrap_or_else(|e| e.into_inner()).file.flush()
            {
                tracing::warn!(%error, "could not flush write-back audit");
            }
        }
    }

    /// Fixed 32-byte records: sequence, fence, address, byte count, path.
    /// Path 1 is a native compare/write piece that may contain untouched bytes.
    /// Path 2 is an exact byte run.
    pub(super) fn record(address: usize, len: usize, path: u32) {
        let Some(Some(journal)) = JOURNAL.get() else { return };
        let fence = FENCE.get();
        if fence == 0 {
            return;
        }
        let mut journal = journal.lock().unwrap_or_else(|e| e.into_inner());
        let sequence = journal.sequence + 1;
        let mut row = [0; 32];
        row[..8].copy_from_slice(&sequence.to_le_bytes());
        row[8..16].copy_from_slice(&fence.to_le_bytes());
        row[16..24].copy_from_slice(&(address as u64).to_le_bytes());
        row[24..28].copy_from_slice(&(len as u32).to_le_bytes());
        row[28..].copy_from_slice(&path.to_le_bytes());
        let result = (|| {
            if journal.sequence != 0 && journal.sequence % RECORDS == 0 {
                journal.file.seek(SeekFrom::Start(0))?;
            }
            journal.file.write_all(&row)
        })();
        match result {
            Ok(()) => journal.sequence = sequence,
            Err(error) => tracing::warn!(%error, "could not record write-back audit"),
        }
    }
}

/// Merge the GPU delta into title memory. Adjacent GPU changes are
/// written together; gaps are never written from the CPU snapshot.
/// Retirement still gives the GPU precedence for overlapping changes.
fn write_gpu_delta(address: usize, before: &[u8], host: &mut [u8], title: &[u8]) -> usize {
    let _writing = crate::watch::writing(address, host.len());
    #[cfg(target_env = "msvc")]
    {
        // The diagnostic trace reports exact write ranges, so it still
        // uses the kernel-copy path.
        #[cfg(feature = "diagnostics")]
        let trace = tracing::enabled!(target: "victory_gpu::writeback", tracing::Level::DEBUG);
        #[cfg(not(feature = "diagnostics"))]
        let trace = false;
        if !trace {
            return write_gpu_delta_pages(address, before, host, title);
        }
    }
    write_gpu_delta_runs(address, before, host, title)
}

/// On MSVC, check the write to each page inside a native exception handler.
/// A page can be unmapped between the successful CPU read and this write.
#[cfg(target_env = "msvc")]
fn write_gpu_delta_pages(address: usize, before: &[u8], host: &mut [u8], title: &[u8]) -> usize {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    unsafe extern "C" {
        fn victory_write_gpu_delta(
            destination: *mut u8,
            before: *const u8,
            host: *mut u8,
            title: *const u8,
            length: usize,
        ) -> usize;
    }
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let page = *PAGE_SIZE.get_or_init(|| {
        let mut info = SYSTEM_INFO::default();
        // SAFETY: GetSystemInfo fills this structure.
        unsafe { GetSystemInfo(&mut info) };
        info.dwPageSize as usize
    });
    let mut offset = 0;
    let mut written = 0;
    while offset < host.len() {
        let at = address + offset;
        let len = (page - at % page).min(host.len() - offset);
        let range = offset..offset + len;
        let (before, host, title) = (&before[range.clone()], &mut host[range.clone()], &title[range]);
        if host == before {
            host.copy_from_slice(title);
        } else {
            // SAFETY: the three slices cover len bytes, and the native
            // routine handles faults on the destination within this page.
            let bytes = unsafe {
                victory_write_gpu_delta(at as *mut u8, before.as_ptr(), host.as_mut_ptr(), title.as_ptr(), len)
            };
            #[cfg(feature = "diagnostics")]
            if bytes != 0 {
                writeback_audit::record(at, len, 1);
            }
            written += bytes;
        }
        offset += len;
    }
    written
}

fn write_gpu_delta_runs(address: usize, before: &[u8], host: &mut [u8], title: &[u8]) -> usize {
    let mut written = 0;
    let mut run = None;
    let mut flush = |range: std::ops::Range<usize>, bytes: &[u8]| {
        let start = address + range.start;
        let len = range.len();
        #[cfg(feature = "diagnostics")]
        trace_copy_back(start, len, "delta");
        crate::host::write_mapped(start as *mut u8, bytes.as_ptr(), len);
        #[cfg(feature = "diagnostics")]
        writeback_audit::record(start, len, 2);
        written += len;
    };
    for i in 0..host.len() {
        if host[i] != before[i] && host[i] != title[i] {
            run.get_or_insert(i);
        } else {
            if let Some(start) = run.take() {
                flush(start..i, &host[start..i]);
            }
            host[i] = title[i];
        }
    }
    if let Some(start) = run {
        flush(start..host.len(), &host[start..]);
    }
    written
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        if self.tracked.as_ref().is_some_and(|t| t.changed(page / PAGE)) {
            return 0;
        }
        // The pre-write title bytes are our baseline for a clean page.
        // Merge only the differences so a CPU store elsewhere on the page
        // can survive. Keep the page dirty for the next refresh.
        write_gpu_delta(from, &title, host, &title)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        let tracked = crate::watch::track(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        #[cfg(feature = "diagnostics")]
        let buffer_bytes = {
            // SAFETY: the live buffer was allocated for this copy.
            let size = unsafe { buffer.GetDesc() }.Width;
            SHADOW_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            size
        };
        let shadow = Arc::new(Shadow {
            buffer,
            #[cfg(feature = "diagnostics")]
            buffer_bytes,
            data,
            gpu,
            start,
            len,
            tracked,
            marks,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    pub(crate) fn arm_view_writes(&self, from: usize, to: usize) {
        if let Some(tracked) = &self.tracked {
            tracked.arm(from, to);
        }
    }

    pub(crate) fn disarm_view_writes(&self) {
        if let Some(tracked) = &self.tracked {
            tracked.disarm();
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::prepare_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, each byte is captured when a declaration
    /// first covers it. Gaps and already-saved bytes are left alone.
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(
        &self,
        submission: &mut Submission,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        submission.pending.insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            submission.armed = true;
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let captured = submission.before.capture(from, to, |a, b, bytes| {
            bytes.extend_from_slice(host(a, b));
        });
        #[cfg(feature = "diagnostics")]
        if captured != 0 {
            crate::copystat::add(crate::copystat::C::SnapshotBytes, captured as u64);
            crate::copystat::add(crate::copystat::C::SnapshotUs, started.elapsed().as_micros() as u64);
            crate::copystat::snapshot(self.start, self.len, captured as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = captured;
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(
        &self,
        submission: &mut Submission,
        address: usize,
        len: usize,
    ) -> usize {
        self.copy_back_or_not(submission, address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, submission: &mut Submission, address: usize, len: usize) {
        self.invalidate(address, len);
        self.copy_back_or_not(submission, address, len, false);
    }

    /// Exclude bytes produced by a list that accessed missing GPU memory,
    /// even if another list declared a partly overlapping writable view.
    pub(crate) fn withhold(&self, submission: &mut Submission, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        if from < to {
            crate::host::add_range(&mut submission.before.compared, from, to);
        }
    }

    /// Discard unreleased GPU output when the copy is next refreshed.
    pub(crate) fn invalidate(&self, address: usize, len: usize) {
        if let Some(tracked) = &self.tracked {
            tracked.invalidate(address, address.saturating_add(len));
        }
    }

    fn copy_back_or_not(
        &self,
        submission: &mut Submission,
        address: usize,
        len: usize,
        back: bool,
    ) -> usize {
        let Submission {
            pending,
            before,
            armed,
        } = submission;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        if back && from < to && before.covers(from, to) {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            let mut scratch = Vec::new();
            for (a, b) in gaps(&before.compared, from, to) {
                written += self.compare_snapshot(before, a, b, &mut scratch);
            }
            // A later declaration must not replay bytes already merged into title memory.
            crate::host::add_range(&mut before.compared, from, to);
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            before.covered.clear();
            before.compared.clear();
            for (_, bytes) in std::mem::take(&mut before.segments) {
                give_back(bytes);
            }
            if *armed {
                if let Some(t) = &self.tracked {
                    t.disarm();
                }
                *armed = false;
            }
        }
        written
    }

    fn compare_snapshot(
        &self,
        snapshot: &Snapshot,
        from: usize,
        to: usize,
        scratch: &mut Vec<u8>,
    ) -> usize {
        let start = self.start as usize;
        let mut written = 0;
        for (&offset, bytes) in snapshot.segments.range(snapshot.first_segment(from)..to) {
            let mut at = from.max(offset);
            let end = to.min(offset + bytes.len());
            while at < end {
                let address = start + at;
                let n = (PAGE - address % PAGE).min(end - at);
                // SAFETY: this segment is within the host mapping. The GPU
                // has finished writing it and only retirement changes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &bytes[at - offset..at - offset + n];
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old {
                    if scratch.is_empty() {
                        scratch.resize(PAGE, 0);
                    }
                    let title = &mut scratch[..n];
                    if crate::host::read_into(address as *const u8, title) {
                        written += write_gpu_delta(address, old, host, title);
                    }
                }
                at += n;
            }
        }
        written
    }

    /// Capture a marked slice's baseline before any copy-back can open a
    /// shared page. If the CPU changed an unarmed page, keep its bytes
    /// because the GPU's before-image is unavailable.
    fn prepare_marked(self: &Arc<Self>, page: usize, scratch: &mut [u8]) -> Option<MarkedSlice> {
        let t = self.tracked.as_ref()?;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return None;
        }
        // Writes from this batch must not save a new image for this slice.
        t.unarm(page / PAGE);
        let saved = t.saved(page / PAGE);
        let n = to - from;
        let before = match &saved {
            Some(before) => &before[from - page..to - page],
            None => {
                let title = &mut scratch[..n];
                if t.changed(page / PAGE)
                    || !crate::host::read_into(from as *const u8, title)
                    || t.changed(page / PAGE)
                {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::MarkedKept, 1);
                    return None;
                }
                title
            }
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BytesCompared, n as u64);
        // SAFETY: inside the mapping. The GPU has completed this submission.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        (host != before).then(|| MarkedSlice {
            copy: self.clone(),
            from,
            before: before.to_vec(),
        })
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(
        &self,
        address: usize,
        len: usize,
        submission: Option<&mut Submission>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if let Some(submission) = submission {
            let mut busy: Vec<_> = submission
                .pending
                .iter()
                .copied()
                .filter(|&(a, e)| a < to && e > from)
                .collect();
            if busy.is_empty() {
                return self.update_free(from, to);
            }
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them, what the title
            // wrote since it was taken (`update_busy`).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_free(at, a);
                }
                if e > a.max(at) {
                    bytes += self.update_busy(submission, a.max(at), e);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` where a list submitted earlier
    /// declared that shaders write. A submission waits for the one before it,
    /// so none of this submission's shaders has run yet: tracked dirty pages,
    /// or the whole untracked range, are copied as anywhere else. Skipped, the
    /// title's constants there stayed stale for the whole submission (a
    /// dispatch read its sizes as zeros and looped until the device was
    /// removed). The same bytes go into the snapshot `writing` took, so
    /// copy-back neither takes them for the GPU's nor puts them back over a
    /// later title write.
    fn update_busy(&self, submission: &mut Submission, from: usize, to: usize) -> usize {
        let untracked = [(from, to)];
        let tracked = self.tracked.as_ref().map(|t| t.take(from, to));
        let runs = tracked.as_deref().unwrap_or(&untracked);
        let start = self.start as usize;
        let mut copied = 0;
        for &(a, b) in runs {
            copied += self.copy(a, b);
            let (a, b) = (a - start, b - start);
            let first = submission.before.first_segment(a);
            for (&at, bytes) in submission.before.segments.range_mut(first..b) {
                let (s, e) = (at.max(a), (at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - at..e - at].copy_from_slice(host);
                }
            }
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let runs = t.take(from, to);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let copied: usize = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if let Some(tracked) = &self.tracked {
            // SAFETY: this range lies within the mapped host copy.
            tracked.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        SHADOW_BYTES.fetch_sub(self.buffer_bytes, std::sync::atomic::Ordering::Relaxed);
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

fn bitmap() -> Option<&'static Bitmap> {
    static BITMAP: std::sync::OnceLock<Option<Bitmap>> = std::sync::OnceLock::new();
    BITMAP
        .get_or_init(|| {
            let bytes = u64::from(victory_shader::emit::VIEW_WRITE_BITS / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes.
pub(crate) fn view_writes() -> Option<u64> {
    bitmap().map(|b| b.gpu)
}

/// After a submission: every page its shaders marked as written through
/// views goes back, but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared); the bitmap is cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
/// Exclusive CPU lease of the view-write bitmap for a synchronous
/// submission. Its GPU address is fixed in the recorded root arguments.
pub(crate) struct ViewMarks {
    bitmap: &'static Bitmap,
}

static MARK_LEASE: Mutex<bool> = Mutex::new(false);
static MARK_FREE: std::sync::Condvar = std::sync::Condvar::new();

impl ViewMarks {
    pub(crate) fn acquire() -> Option<Self> {
        let bitmap = bitmap()?;
        let mut active = MARK_FREE
            .wait_while(
                MARK_LEASE.lock().unwrap_or_else(|e| e.into_inner()),
                |active| *active,
            )
            .unwrap_or_else(|e| e.into_inner());
        *active = true;
        Some(Self { bitmap })
    }

    pub(crate) fn copy_back(&mut self, withheld: &[(Arc<Shadow>, usize, usize)]) {
        copy_back_marks(self.bitmap, withheld);
    }
}

impl Drop for ViewMarks {
    fn drop(&mut self) {
        *MARK_LEASE.lock().unwrap_or_else(|e| e.into_inner()) = false;
        MARK_FREE.notify_one();
    }
}

struct MarkedSlice {
    copy: Arc<Shadow>,
    from: usize,
    before: Vec<u8>,
}

impl MarkedSlice {
    fn commit(self, scratch: &mut [u8]) -> usize {
        let title = &mut scratch[..self.before.len()];
        // Read the latest title bytes for the merge. Other slices may
        // have written this page since the batch captured its baselines.
        if !crate::host::read_into(self.from as *const u8, title) {
            return 0;
        }
        let offset = self.from - self.copy.start as usize;
        // SAFETY: this slice was clipped to the mapping in prepare_marked.
        // Only retirement accesses it until this submission is finished.
        let host = unsafe {
            std::slice::from_raw_parts_mut(self.copy.data.add(offset), title.len())
        };
        if host == title {
            return 0;
        }
        // Keep the page dirty for the next refresh. A CPU write may land
        // between the read and the GPU delta and leave the host copy stale.
        write_gpu_delta(self.from, &self.before, host, title)
    }
}

fn copy_back_marks(b: &Bitmap, withheld: &[(Arc<Shadow>, usize, usize)]) {
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
    // SAFETY: the mapped bitmap; the GPU is done with it (one submission
    // runs at a time).
    let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
    // Marks of offsets past the bitmap land in its last bit, no copy's.
    bits[words - 1] &= u32::MAX >> 1;
    let mut marked = Vec::new();
    let used = (marks().end as usize).div_ceil(32).min(words);
    for (w, word) in bits[..used].iter_mut().enumerate() {
        let mut x = std::mem::take(word);
        while x != 0 {
            marked.push((w * 32) as u32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked
            .iter()
            .filter_map(|&n| {
                let (&base, copy) = m.owners.range(..=n).next_back()?;
                let copy = copy.upgrade()?;
                let k = n - base;
                (k < pages(copy.start, copy.len)).then(|| {
                    let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
                    (copy, page)
                })
            })
            .collect()
    };
    let mut scratch = vec![0; PAGE];
    let mut staged = Vec::new();
    for (copy, start, end) in coalesce_marked(pages) {
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(
                crate::copystat::C::DirtyPages,
                ((end - start) / PAGE) as u64,
            );
            crate::copystat::add(crate::copystat::C::CoalescedRuns, 1);
        }
        for page in (start..end).step_by(PAGE) {
            if withheld.iter().any(|(c, a, n)| {
                Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a.saturating_add(*n)
            }) {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::FallbackComparisons, 1);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            if let Some(slice) = copy.prepare_marked(page, &mut scratch) {
                staged.push(slice);
            }
        }
    }
    // Opening a page dirties every tracker of it, including trackers of
    // disjoint suballocations and other virtual mappings of that page.
    // Capture every baseline before the first such write.
    for slice in staged {
        let written = slice.commit(&mut scratch);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

/// Merge adjacent dirty pages for the same copy, without including gaps.
fn coalesce_marked(mut pages: Vec<(Arc<Shadow>, usize)>) -> Vec<(Arc<Shadow>, usize, usize)> {
    pages.sort_unstable_by_key(|(copy, page)| (Arc::as_ptr(copy) as usize, *page));
    let mut runs: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
    for (copy, page) in pages {
        if let Some((last_copy, _, end)) = runs.last_mut()
            && Arc::ptr_eq(last_copy, &copy)
            && page <= *end
        {
            *end = (*end).max(page.saturating_add(PAGE));
        } else {
            runs.push((copy, page, page.saturating_add(PAGE)));
        }
    }
    runs
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
const WINDOW: u64 = 4 << 20;

fn loose_window_size() -> u64 {
    static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let bytes = std::env::var("VICTORY_WINDOW_KB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .and_then(|kb| kb.checked_mul(1024))
            .filter(|&bytes| bytes.is_power_of_two() && (PAGE as u64..=WINDOW).contains(&bytes))
            .unwrap_or(WINDOW);
        tracing::info!(window_kib = bytes / 1024, "loose shadow window size");
        bytes
    })
}

fn aligned_window(address: u64, end: u64, size: u64) -> Option<(u64, u64)> {
    let start = address & !(size - 1);
    let len = end.checked_next_multiple_of(size)?.checked_sub(start)?;
    Some((start, len))
}

enum LooseCopy {
    Live(Arc<Shadow>),
    /// Remember the range for indirect shader access after a remap. A
    /// descriptor or a recorded list can still own the original copy.
    Unmapped(Weak<Shadow>),
}

impl LooseCopy {
    fn live(&self) -> Option<&Arc<Shadow>> {
        match self {
            Self::Live(shadow) => Some(shadow),
            Self::Unmapped(_) => None,
        }
    }
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<LooseWindows<LooseCopy>> = Mutex::new(LooseWindows {
    entries: BTreeMap::new(),
    longest: 0,
});

struct LooseWindows<T> {
    entries: BTreeMap<(u64, u64), T>,
    longest: u64,
}

impl<T> LooseWindows<T> {
    fn insert(&mut self, key: (u64, u64), value: T) {
        self.longest = self.longest.max(key.1);
        self.entries.insert(key, value);
    }

    /// Reuse the smallest existing window covering the whole requested range.
    fn covering(&self, start: u64, len: u64) -> Option<(&(u64, u64), &T)> {
        let end = start.checked_add(len)?;
        if len > self.longest {
            return None;
        }
        if let Some(exact) = self.entries.get_key_value(&(start, len)) {
            return Some(exact);
        }
        // A window starting earlier than this cannot reach the requested end.
        let first = end.saturating_sub(self.longest);
        self.entries
            .range((first, 0)..=(start, u64::MAX))
            .filter(|((base, size), _)| base.checked_add(*size).is_some_and(|last| last >= end))
            .min_by_key(|((_, size), _)| *size)
    }
}

/// A batch of title mapping changes, bounded by one enclosing interval.
/// This lock is independent of the copy cache, page tables and GPU queue.
static LOOSE_MAPPING_CHANGES: Mutex<Option<(u64, u64)>> = Mutex::new(None);

pub(crate) fn note_mapping_change(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    let mut changes = LOOSE_MAPPING_CHANGES.lock().unwrap_or_else(|e| e.into_inner());
    *changes = Some(match *changes {
        Some((first, last)) => (start.min(first), end.max(last)),
        None => (start, end),
    });
}

/// A window is reclaimable only if every byte is reserved or free.
/// In particular a guarded or inaccessible committed page still counts.
fn fully_unmapped(start: u64, len: u64) -> Option<bool> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = usize::try_from(start).ok()?;
    let end = at.checked_add(usize::try_from(len).ok()?)?;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: VirtualQuery fills a valid output buffer and never
        // dereferences the queried address in user space.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n != size_of_val(&info) {
            return None;
        }
        match info.State {
            MEM_COMMIT => return Some(false),
            MEM_RESERVE | MEM_FREE => {}
            _ => return None,
        }
        let next = (info.BaseAddress as usize).checked_add(info.RegionSize)?;
        if next <= at {
            return None;
        }
        at = next;
    }
    Some(true)
}

/// Called under the submission lock after draining earlier GPU work.
/// Rebuild any page tables holding evicted copies before releasing our
/// last reference. Recorded commands and descriptors keep their own refs.
pub(crate) fn reconcile_loose_mappings() {
    let changes = LOOSE_MAPPING_CHANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let Some((first, last)) = changes else { return };
    let mut evicted = Vec::new();
    {
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        let earliest = first.saturating_sub(map.longest);
        for (&(start, len), copy) in map.entries.range_mut((earliest, 0)..(last, 0)) {
            if start + len <= first {
                continue;
            }
            if let Some(old) = reconcile_window(copy, start, len) {
                evicted.push(old);
            }
        }
    }
    if !evicted.is_empty() {
        crate::memory::forget_loose_copies(&evicted);
        #[cfg(feature = "diagnostics")]
        {
            let released = evicted.iter().filter(|copy| Arc::strong_count(copy) == 1);
            let reclaimed_bytes = released.map(|copy| copy.len).sum::<u64>();
            tracing::debug!(windows = evicted.len(), reclaimed_bytes, "unmapped loose host memory copies");
            LOOSE_STATS[4].fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed);
            LOOSE_STATS[5].fetch_add(reclaimed_bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn reconcile_window(copy: &mut LooseCopy, start: u64, len: u64) -> Option<Arc<Shadow>> {
    match (fully_unmapped(start, len), &*copy) {
        (Some(true), LooseCopy::Live(old)) => {
            let old = old.clone();
            *copy = LooseCopy::Unmapped(Arc::downgrade(&old));
            changed();
            Some(old)
        }
        (Some(false), LooseCopy::Unmapped(old)) => {
            if let Some(shadow) = old.upgrade().or_else(|| new_loose_copy(start, len)) {
                *copy = LooseCopy::Live(shadow);
                changed();
            } else {
                // Retry when allocation pressure may have subsided.
                note_mapping_change(start as usize, len as usize);
            }
            None
        }
        _ => None,
    }
}

#[cfg(feature = "diagnostics")]
static LOOSE_STATS: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let (start, len) = aligned_window(address, end, loose_window_size())?;
    let (page_start, page_len) = aligned_window(address, end, PAGE as u64)?;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let cached = map.covering(start, len)
        .filter(|(_, copy)| copy.live().is_some())
        .or_else(|| map.covering(page_start, page_len));
    if let Some((&key, copy)) = cached {
        if let LooseCopy::Unmapped(old) = copy {
            if let Some(shadow) = old.upgrade().or_else(|| new_loose_copy(key.0, key.1)) {
                map.insert(key, LooseCopy::Live(shadow.clone()));
                changed();
                return Some(shadow);
            }
        } else if let Some(shadow) = copy.live() {
            #[cfg(feature = "diagnostics")]
            {
                use std::sync::atomic::Ordering::Relaxed;
                if shadow.start == start && shadow.len == len {
                    LOOSE_STATS[0].fetch_add(1, Relaxed);
                } else {
                    LOOSE_STATS[1].fetch_add(1, Relaxed);
                    LOOSE_STATS[2].fetch_add(len, Relaxed);
                }
            }
            return Some(shadow.clone());
        }
    }
    // A full cache window can fail to allocate even when the requested
    // pages still fit. Keep the fallback page-aligned for indirect access.
    let (key, shadow) = allocate_loose_window((start, len), (page_start, page_len), new_loose_copy)?;
    map.insert(key, LooseCopy::Live(shadow.clone()));
    changed();
    Some(shadow)
}

fn allocate_loose_window<T>(
    window: (u64, u64),
    pages: (u64, u64),
    mut allocate: impl FnMut(u64, u64) -> Option<T>,
) -> Option<((u64, u64), T)> {
    if let Some(copy) = allocate(window.0, window.1) {
        return Some((window, copy));
    }
    if window == pages {
        return None;
    }
    allocate(pages.0, pages.1).map(|copy| (pages, copy))
}

fn new_loose_copy(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    #[cfg(feature = "diagnostics")]
    LOOSE_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [exact, covering, avoided_bytes, created, evicted, reclaimed_bytes] =
        LOOSE_STATS.each_ref().map(|s| s.swap(0, Relaxed));
    let (windows, bytes, unmapped) = {
        let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        let (windows, bytes) = map.entries.values().filter_map(LooseCopy::live)
            .fold((0, 0), |(n, bytes), copy| (n + 1, bytes + copy.len));
        (windows, bytes, map.entries.len() - windows)
    };
    tracing::info!(exact_reuses = exact, covering_reuses = covering, avoided_bytes,
        created, live_windows = windows, live_bytes = bytes, unmapped_windows = unmapped,
        evicted, reclaimed_bytes, "loose host memory copies");
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, address, len);
    map.insert((address, len), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).entries.values().filter_map(LooseCopy::live).cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .values()
            .filter_map(|copy| match copy {
                LooseCopy::Live(s) => Some((s.gpu, s.start, s.len)),
                LooseCopy::Unmapped(s) => s.upgrade().map(|s| (s.gpu, s.start, s.len)),
            }),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    #[test]
    fn small_windows_preserve_the_full_requested_range() {
        let size = 256 << 10;
        assert_eq!(super::aligned_window(size - 4, size, size), Some((0, size)));
        assert_eq!(super::aligned_window(size - 4, size + 4, size), Some((0, 2 * size)));
        assert_eq!(super::aligned_window(size + 4, 3 * size, size), Some((size, 2 * size)));
    }

    #[test]
    fn failed_loose_window_allocation_retries_only_the_requested_pages() {
        let window = (0, super::WINDOW);
        let pages = (0x10000, 0x20000);
        let mut attempts = Vec::new();
        let result = super::allocate_loose_window(window, pages, |start, len| {
            attempts.push((start, len));
            ((start, len) == pages).then_some("allocated")
        });
        assert_eq!(attempts, [window, pages]);
        assert_eq!(result, Some((pages, "allocated")));
    }

    #[test]
    fn covering_window_reuses_the_smallest_complete_copy() {
        let mut map = LooseWindows {
            entries: BTreeMap::new(),
            longest: 0,
        };
        map.insert((0, 16), "large");
        map.insert((4, 8), "small");
        map.insert((20, 4), "separate");
        assert_eq!(map.covering(6, 4), Some((&(4, 8), &"small")));
        assert_eq!(map.covering(3, 10), Some((&(0, 16), &"large")));
        assert_eq!(map.covering(10, 8), None);
        assert_eq!(map.covering(u64::MAX, 2), None);
    }

    #[test]
    fn bounded_window_lookup_matches_a_full_scan() {
        let mut map = LooseWindows {
            entries: BTreeMap::new(),
            longest: 0,
        };
        for i in 0..2048 {
            map.insert((i * 16, 8 + (i % 7) * 16), i);
        }
        for start in (0..32800).step_by(13) {
            for len in [4, 24, 64, 104, 120] {
                let old = map.entries
                    .range(..=(start, u64::MAX))
                    .filter(|((base, size), _)| base + size >= start + len)
                    .min_by_key(|((_, size), _)| *size);
                assert_eq!(map.covering(start, len), old, "{start}+{len}");
            }
        }
    }
    use super::*;

    #[test]
    fn unmapped_window_preserves_held_copy_and_rebuilds_released_copy_on_remap() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree,
        };

        let Some(_) = host() else { return };
        // SAFETY: fresh reservation of two pages, owned by this test.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        let start = base as u64;
        let len = (2 * PAGE) as u64;
        // SAFETY: changes only the test's own reservation.
        assert_eq!(unsafe { VirtualAlloc(base, PAGE, MEM_COMMIT, PAGE_NOACCESS) }, base);
        let original = new_loose_copy(start, len).unwrap();
        let mut cached = LooseCopy::Live(original.clone());
        assert!(reconcile_window(&mut cached, start, len).is_none());
        assert!(cached.live().is_some(), "one committed page keeps the window");

        // SAFETY: this page is committed in the test's reservation.
        assert_eq!(unsafe { VirtualFree(base, PAGE, MEM_DECOMMIT) }, 1);
        let evicted = reconcile_window(&mut cached, start, len).unwrap();
        assert!(Arc::ptr_eq(&original, &evicted));
        assert!(cached.live().is_none());
        drop(evicted);
        // SAFETY: changes only the test's own reservation.
        assert_eq!(unsafe { VirtualAlloc(base, PAGE, MEM_COMMIT, PAGE_READWRITE) }, base);
        assert!(reconcile_window(&mut cached, start, len).is_none());
        assert!(Arc::ptr_eq(cached.live().unwrap(), &original));

        // SAFETY: this page is committed in the test's reservation.
        assert_eq!(unsafe { VirtualFree(base, PAGE, MEM_DECOMMIT) }, 1);
        drop(reconcile_window(&mut cached, start, len).unwrap());
        let weak = Arc::downgrade(&original);
        drop(original);
        assert!(weak.upgrade().is_none(), "the cache only holds a weak reference");
        // SAFETY: commits the page again and then releases the reservation.
        assert_eq!(unsafe { VirtualAlloc(base, PAGE, MEM_COMMIT, PAGE_READWRITE) }, base);
        reconcile_window(&mut cached, start, len);
        assert_eq!(cached.live().unwrap().range(), (start, len));
        assert_eq!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 1);
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        let mut submission = Submission::default();
        copy.update(p, len, None);
        // An earlier list declares all of it as written by shaders.
        copy.writing(&mut submission, p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(
            copy.update(p + PAGE + 16, 16, Some(&mut submission)) > 0,
            "the title's write is copied"
        );
        assert_eq!(
            host(PAGE + 20),
            0x55,
            "the title's constant reaches the copy"
        );
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(&mut submission, p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    #[test]
    fn busy_untracked_refresh_updates_the_writeback_baseline() {
        let mut title = Box::new([0u8; 16]);
        let p = title.as_mut_ptr() as usize;
        register(p, p as u64, 16);
        let Some(copy) = find(p as u64, 16) else {
            unregister(p, p as u64);
            return;
        };
        assert!(copy.tracked.is_none());
        let mut submission = Submission::default();
        copy.update(p, 16, None);
        copy.writing(&mut submission, p, 16, false);
        title[1] = 0x55;
        assert_eq!(copy.update(p, 16, Some(&mut submission)), 16);
        assert_eq!(copy.read_host(p as u64, 1)[0].to_le_bytes()[1], 0x55);
        title[1] = 0x66;
        copy.write_host(p as u64, &[0xaa]);
        assert_eq!(copy.copy_back(&mut submission, p, 16), 1);
        assert_eq!(&title[..2], &[0xaa, 0x66]);
        unregister(p, p as u64);
    }

    #[test]
    fn growing_snapshot_captures_each_byte_only_once() {
        let mut snapshot = Snapshot::default();
        let mut captured = 0;
        let mut read = |from, to, bytes: &mut Vec<u8>| {
            bytes.resize(to - from, from as u8);
            captured += to - from;
        };
        for end in 1..=4096 {
            assert_eq!(snapshot.capture(0, end, &mut read), 1);
        }
        assert_eq!(snapshot.capture(0, 4096, &mut read), 0);
        assert_eq!(snapshot.capture(8192, 8193, &mut read), 1);
        assert!(!snapshot.covers(0, 8193));
        assert_eq!(snapshot.capture(4095, 8193, &mut read), 4096);
        assert!(snapshot.covers(0, 8193));
        assert_eq!(captured, 8193);
        assert_eq!(snapshot.segments[&7], [7]);
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn overlapping_snapshot_cannot_publish_withheld_bytes() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        if let Some(copy) = find(p as u64, PAGE as u64) {
            let mut submission = Submission::default();
            copy.update(p, PAGE, None);
            copy.writing(&mut submission, p, 4, false);
            copy.writing(&mut submission, p + 2, 4, false);
            copy.write_host(p as u64, &[1, 2, 3, 4, 5, 6]);
            copy.withhold(&mut submission, p + 2, 4);

            assert_eq!(copy.copy_back(&mut submission, p, 4), 2);
            copy.keep_writes(&mut submission, p + 2, 4);
            // SAFETY: this test owns the live pool.
            assert_eq!(unsafe { std::slice::from_raw_parts(p as *const u8, 6) }, &[1, 2, 0, 0, 0, 0]);
            copy.update(p, PAGE, None);
            assert_eq!(copy.read_host(p as u64, 1), [0x0201]);
        }
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut submission = Submission::default();
        copy.update(p, len, None);
        let bytes = |s: &Submission| s.before.segments.values().map(Vec::len).sum::<usize>();
        copy.writing(&mut submission, p + 16, 64, false);
        copy.writing(&mut submission, p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&submission), 164, "two segments");
        copy.writing(&mut submission, p + PAGE, 2 * PAGE, false);
        copy.writing(&mut submission, p + PAGE, PAGE, false);
        copy.writing(&mut submission, p + 80, 8, false);
        copy.writing(&mut submission, p + 18, 4, false);
        assert_eq!(
            bytes(&submission),
            72 + 2 * PAGE + 100,
            "each covered byte is retained once"
        );
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(
            copy.copy_back(&mut submission, p + 16, 64),
            3,
            "only the GPU's changed bytes"
        );
        // The title replaces an already retired byte before another
        // overlapping declaration is retired.
        // SAFETY: the byte is inside the live pool.
        unsafe { ((p + 20) as *mut u8).write_volatile(42) };
        assert_eq!(copy.copy_back(&mut submission, p + 18, 4), 0);
        assert_eq!(
            copy.copy_back(&mut submission, p + PAGE, 2 * PAGE),
            1,
            "only the GPU's changed byte"
        );
        assert_eq!(
            copy.copy_back(&mut submission, p + PAGE, PAGE),
            0,
            "compared once"
        );
        assert_eq!(copy.copy_back(&mut submission, p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(&mut submission, p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!(
            [
                title(p + 20),
                title(p + 21),
                title(p + 22),
                title(p + PAGE + 5)
            ],
            [42, 2, 3, 9]
        );
        assert_eq!(
            bytes(&submission),
            0,
            "the snapshot goes with the last range"
        );
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(mut marks)) = (find(p as u64, len as u64), ViewMarks::acquire())
        else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        let mut submission = Submission::default();
        copy.update(p, len, None);
        copy.writing(&mut submission, p, len, true);
        assert!(submission.before.segments.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = marks.bitmap.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        marks.copy_back(&[]);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(
            copy.copy_back(&mut submission, p, len),
            0,
            "nothing kept to compare"
        );
        unregister(p, p as u64);
    }

    #[test]
    fn dirty_runs_join_adjacent_pages_without_including_gaps() {
        let p = crate::watch::tests::pool(4);
        register(p, p as u64, (4 * PAGE) as u64);
        let Some(copy) = find(p as u64, (4 * PAGE) as u64) else {
            unregister(p, p as u64);
            return;
        };
        let runs = coalesce_marked(vec![
            (copy.clone(), p + 3 * PAGE),
            (copy.clone(), p + PAGE),
            (copy.clone(), p),
            (copy.clone(), p + PAGE),
        ]);
        assert_eq!(
            runs.iter().map(|(_, a, b)| (*a, *b)).collect::<Vec<_>>(),
            [(p, p + 2 * PAGE), (p + 3 * PAGE, p + 4 * PAGE)]
        );
        unregister(p, p as u64);
    }

    #[test]
    fn marked_suballocations_sharing_a_page_both_write_back() {
        const LEN: usize = 4096;
        let p = crate::watch::tests::pool(1);
        let second = p + LEN;
        register(p, p as u64, LEN as u64);
        register(second, second as u64, LEN as u64);
        if let (Some(a), Some(b), Some(mut marks)) = (
            find(p as u64, LEN as u64),
            find(second as u64, LEN as u64),
            ViewMarks::acquire(),
        ) {
            let bits = marks.bitmap.data as *mut u32;
            let mark = |copy: &Shadow| {
                let n = copy.marks.expect("tracked suballocation") as usize;
                // SAFETY: a dword of the leased bitmap. No GPU work uses it.
                unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
            };
            for byte in [11, 22] {
                assert_eq!(a.update(p, LEN, None), LEN);
                assert_eq!(b.update(second, LEN, None), LEN);
                a.write_host(p as u64, &[byte]);
                b.write_host(second as u64, &[byte + 1]);
                mark(&a);
                mark(&b);
                marks.copy_back(&[]);
                // SAFETY: both bytes are inside the live pool.
                let title = unsafe { (*(p as *const u8), *(second as *const u8)) };
                assert_eq!(title, (byte, byte + 1));
            }
        }
        unregister(p, p as u64);
        unregister(second, second as u64);
        crate::watch::tests::free(p);
    }

    #[test]
    fn heap_armed_write_keeps_cpu_bytes_written_through_an_alias() {
        let (p, alias) = crate::watch::tests::twin_views(1);
        register(p, p as u64, PAGE as u64);
        if let Some(copy) = find(p as u64, PAGE as u64) {
            copy.arm_view_writes(p, p + PAGE);
            copy.update(p, PAGE, None);
            copy.write_host((p + 1) as u64, &[19, 20]);
            // SAFETY: both bytes belong to the alias of the tracked page.
            unsafe {
                ((alias + 2) as *mut u8).write_volatile(9);
                ((alias + 3) as *mut u8).write_volatile(8);
            }
            let mut scratch = vec![0; PAGE];
            let slice = copy.prepare_marked(p, &mut scratch).expect("armed heap write");
            assert_eq!(slice.commit(&mut scratch), 2);
            // SAFETY: the alias is live and the GPU merge has completed.
            assert_eq!(unsafe { std::slice::from_raw_parts((alias + 1) as *const u8, 3) }, &[19, 20, 8]);
            copy.disarm_view_writes();
        }
        unregister(p, p as u64);
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for address in [p, alias] {
            crate::watch::title_memory_remapped(address, PAGE);
            // SAFETY: these are the two views returned by twin_views.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: address as *mut _ }) };
        }
    }

    #[test]
    fn partial_refresh_updates_the_marked_before_image() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        if let Some(copy) = find(p as u64, PAGE as u64) {
            let mut submission = Submission::default();
            copy.update(p, PAGE, None);
            copy.writing(&mut submission, p, PAGE, true);
            let byte = |at: usize| (p + at) as *mut u8;
            // SAFETY: these bytes are inside the live pool.
            unsafe {
                byte(5).write_volatile(1);
                byte(6).write_volatile(2);
            }
            assert_eq!(copy.update(p + 5, 2, Some(&mut submission)), 2);
            // The GPU writes a byte back to the pre-refresh value, and
            // the CPU replaces the other byte after preparation.
            copy.write_host((p + 6) as u64, &[0]);
            // SAFETY: byte in the live pool.
            unsafe { byte(5).write_volatile(3) };
            let mut scratch = vec![0; PAGE];
            let slice = copy.prepare_marked(p, &mut scratch).expect("GPU changed the baseline");
            assert_eq!(slice.commit(&mut scratch), 1);
            // SAFETY: both bytes are inside the live pool.
            assert_eq!(unsafe { (*byte(5), *byte(6)) }, (3, 0));
            copy.copy_back(&mut submission, p, PAGE);
        }
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    #[test]
    fn unchanged_marked_overlap_refreshes_after_the_outer_copy_writes() {
        let p = crate::watch::tests::pool(1);
        let inner = p + 16;
        register(p, p as u64, PAGE as u64);
        register(inner, inner as u64, 16);
        if let (Some(outer), Some(copy)) = (find(p as u64, PAGE as u64), find(inner as u64, 16)) {
            outer.update(p, PAGE, None);
            copy.update(inner, 16, None);
            outer.write_host(inner as u64, &[27]);
            let mut scratch = vec![0; PAGE];
            let write = outer.prepare_marked(p, &mut scratch).expect("outer copy changed");
            assert!(copy.prepare_marked(p, &mut scratch).is_none());
            assert_eq!(write.commit(&mut scratch), 1);
            assert_eq!(copy.update(inner, 16, None), 16);
            // SAFETY: the first byte of the live, refreshed host copy.
            assert_eq!(unsafe { *copy.data }, 27);
        }
        unregister(inner, inner as u64);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    #[test]
    fn delta_write_does_not_replay_an_older_cpu_read_between_gpu_changes() {
        let p = crate::watch::tests::pool(1);
        let before = [0; 5];
        let old_title = [0, 0, 0, 0, 0];
        let mut host = [0, 1, 0, 2, 0];
        // The CPU changes a byte after the title buffer above was read.
        // SAFETY: five bytes at the start of the test pool.
        unsafe { (p as *mut u8).add(2).write_volatile(42) };
        assert_eq!(write_gpu_delta(p, &before, &mut host, &old_title), 2);
        // SAFETY: five bytes at the start of the test pool.
        assert_eq!(unsafe { std::slice::from_raw_parts(p as *const u8, 5) }, [0, 1, 42, 2, 0]);
    }

    #[test]
    #[cfg(target_env = "msvc")]
    fn delta_write_continues_after_an_unmapped_destination_page() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

        let mut info = SYSTEM_INFO::default();
        // SAFETY: the system call fills this structure.
        unsafe { GetSystemInfo(&mut info) };
        let page = info.dwPageSize as usize;
        // SAFETY: reserve and commit three pages, then decommit the middle.
        let dst = unsafe { VirtualAlloc(std::ptr::null(), 3 * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!dst.is_null());
        assert_ne!(unsafe { VirtualFree(dst.cast::<u8>().add(page).cast(), page, MEM_DECOMMIT) }, 0);

        let before = [0; 3];
        let mut gpu = [1, 2, 3];
        let title = [0; 3];
        // Split the cases so only the first byte of each system page is marked.
        let mut host = vec![0; 2 * page + 1];
        for (i, value) in gpu.iter().enumerate() {
            host[i * page] = *value;
        }
        let zero = vec![0; host.len()];
        let written = write_gpu_delta_pages(dst as usize, &zero, &mut host, &zero);
        // SAFETY: the first and last pages stayed committed.
        let actual = unsafe { [dst.cast::<u8>().read(), dst.cast::<u8>().add(2 * page).read()] };
        assert_ne!(unsafe { VirtualFree(dst, 0, MEM_RELEASE) }, 0);
        assert_eq!((written, actual), (2, [1, 3]));
        // An ordinary mapped range with exact gaps produces the same counts.
        let mut memory = [0u8; 3];
        assert_eq!(write_gpu_delta_pages(memory.as_mut_ptr() as usize, &before, &mut gpu, &title), 3);
        assert_eq!(memory, [1, 2, 3]);
    }

    /// CPU-only and GPU-only bytes survive a snapshot merge. If both
    /// change the same byte, the existing merge gives the GPU's value
    /// precedence when the submission is retired.
    #[test]
    fn snapshot_merge_keeps_cpu_only_bytes_and_gpu_wins_overlap() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return;
        };
        let mut submission = Submission::default();
        copy.update(p, PAGE, None);
        copy.writing(&mut submission, p + 8, 8, false);
        copy.write_host((p + 8) as u64, &[0xaa, 0xbb]);
        // SAFETY: these bytes are within the test pool.
        unsafe {
            (p as *mut u8).add(9).write_volatile(0xcc);
            (p as *mut u8).add(10).write_volatile(0xdd);
        }
        copy.copy_back(&mut submission, p + 8, 8);
        // SAFETY: these bytes are within the test pool.
        unsafe {
            assert_eq!((p as *const u8).add(8).read_volatile(), 0xaa);
            assert_eq!((p as *const u8).add(9).read_volatile(), 0xbb);
            assert_eq!((p as *const u8).add(10).read_volatile(), 0xdd);
        }
        unregister(p, p as u64);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
}
