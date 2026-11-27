//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Title submissions with CPU-visible effects retire synchronously. The
//! opt-in queue can overlap GPU-only title lists.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    memory_adapter: OnceLock<Option<IDXGIAdapter3>>,
    shared_buffer_retry: Mutex<SharedBufferRetry>,
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    submissions: Mutex<Submissions>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under the submission lock.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// Finish queued GPU-only work before publishing title completion or exiting.
pub(crate) fn drain_pending() {
    if max_in_flight() > 1
        && let Some(Some(host)) = HOST.get()
    {
        host.drain();
    }
}

/// Stop accepting asynchronous work and retire the queue at title shutdown.
pub fn finish_pending() {
    if max_in_flight() > 1
        && let Some(Some(host)) = HOST.get()
    {
        let mut submissions = host.submissions.lock().unwrap_or_else(|e| e.into_inner());
        submissions.finishing = true;
        host.drain_through(&mut submissions, u64::MAX);
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, (&raw mut options).cast(), size_of_val(&options) as u32)
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match (victory_shader::emit::host::Host {
            device: device.clone(),
        })
        .fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            memory_adapter: OnceLock::new(),
            shared_buffer_retry: Mutex::default(),
            depth_bounds_supported,
            queue,
            fence,
            submissions: Mutex::new(Submissions::default()),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shared_buffer_oom_retries_one_at_a_time_and_recovers() {
        let now = Instant::now();
        let retry_at = now + SharedBufferRetry::DELAY;
        let mut retry = SharedBufferRetry::default();
        let large = SharedBufferRetry::bucket(8 << 20);
        let small = SharedBufferRetry::bucket(256);
        assert_eq!(SharedBufferRetry::bucket(1 << 16), small);
        assert_eq!(SharedBufferRetry::bucket(u64::MAX), 48);
        assert_eq!(retry.allow(large, now), Some(false));
        assert!(retry.finish(large, false, true, now));
        for _ in 0..10_000 {
            assert_eq!(retry.allow(large, now), None);
        }
        assert_eq!(retry.allow(small, now), Some(false));
        assert_eq!(retry.allow(large, retry_at), Some(true));
        assert_eq!(retry.allow(large, retry_at), None);
        assert!(!retry.finish(large, true, true, retry_at));
        assert_eq!(retry.allow(large, retry_at), None);
        let retry_at = retry_at + SharedBufferRetry::DELAY;
        assert_eq!(retry.allow(large, retry_at), Some(true));
        assert!(!retry.finish(large, true, false, retry_at));
        assert_eq!(retry.allow(large, retry_at), Some(false));
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    #[test]
    fn retirement_stops_at_the_requested_fence_in_order() {
        let mut pending = VecDeque::from([1, 2, 3]);
        let mut retired = Vec::new();
        assert_eq!(
            retire_through(&mut pending, |&fence| fence <= 2, |work| retired.push(work)),
            2
        );
        assert_eq!(retired, [1, 2]);
        assert_eq!(pending, [3]);
    }

    #[test]
    fn allocator_and_resources_stay_owned_until_retirement() {
        let allocator = Arc::new(());
        let upload = Arc::new(());
        let weak_allocator = Arc::downgrade(&allocator);
        let weak_upload = Arc::downgrade(&upload);
        let mut pending = VecDeque::from([(7, allocator, upload)]);
        retire_through(&mut pending, |work| work.0 <= 6, |_| unreachable!());
        assert!(weak_allocator.upgrade().is_some());
        assert!(weak_upload.upgrade().is_some());
        retire_through(
            &mut pending,
            |work| work.0 <= 7,
            |work| {
                assert!(weak_allocator.upgrade().is_some(), "during fence wait");
                assert!(weak_upload.upgrade().is_some(), "during fence wait");
                drop(work);
            },
        );
        assert!(weak_allocator.upgrade().is_none());
        assert!(weak_upload.upgrade().is_none());
    }

    #[test]
    fn depth_two_retires_oldest_before_accepting_a_third_submission() {
        let mut pending = VecDeque::from([1]);
        assert_eq!(oldest_when_full(&pending, 2), None);
        pending.push_back(2);
        let oldest = *oldest_when_full(&pending, 2).expect("two submissions fill the queue");
        let mut retired = Vec::new();
        retire_through(
            &mut pending,
            |&fence| fence <= oldest,
            |work| retired.push(work),
        );
        pending.push_back(3);
        assert_eq!(retired, [1]);
        assert_eq!(pending, [2, 3]);
    }

    #[test]
    fn queued_recording_holds_upload_and_defers_list_reuse() {
        let Ok(host) = create() else { return };
        let owners: Vec<_> = (0..3)
            .map(|_| {
                let list = host.new_list().expect("command list");
                let (_, upload) = list.upload(&host, 256).expect("upload buffer");
                static SOURCE: [u8; 4] = [1, 2, 3, 4];
                list.defer_fill(upload, SOURCE.as_ptr(), SOURCE.len());
                list.close();
                Arc::new(crate::list::ListStorage::test_recording(list))
            })
            .collect();
        let first = Arc::downgrade(&owners[0]);
        let second = Arc::downgrade(&owners[1]);
        for owner in &owners {
            host.execute_title_at_depth(std::slice::from_ref(owner), 2);
        }
        // The third list retires the first; it leaves the second in place
        // even if the GPU has already signalled both fences.
        let replay = owners[2].clone();
        drop(owners);
        assert!(first.upgrade().is_none());
        assert!(second.upgrade().is_some());
        let list = replay.host().unwrap();
        let fence = list.queued_until.load(Ordering::Acquire);
        assert!(fence > 0);
        host.wait_for_list(list);
        // SAFETY: the test owns a live queue and fence.
        assert!(unsafe { host.fence.GetCompletedValue() } >= fence);
        assert!(second.upgrade().is_none());
        assert!(host.submissions.lock().unwrap().pending.is_empty());
        list.reset();
        assert!(list.deferred().uploads.is_empty());
        list.close();
        // SAFETY: this test owns the event and all waiters have returned.
        unsafe { windows::Win32::Foundation::CloseHandle(host.event) }.unwrap();
    }

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 2);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    #[test]
    fn breadcrumb_report_keeps_the_stop_in_a_long_barrier_interval() {
        let mut history = vec![D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER];
        let mut markers = Vec::new();
        for i in 0..40 {
            markers.push((history.len() as u32, format!("dispatch {i}")));
            history.push(D3D12_AUTO_BREADCRUMB_OP_SETMARKER);
            history.push(D3D12_AUTO_BREADCRUMB_OP_DISPATCH);
        }
        history.push(D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER);
        let report = in_flight(&history, &markers, 62);
        assert!(report.contains("command 62 (op 6): dispatch 30"));
        assert!(report.contains("command 60 (op 6): dispatch 29"));
        assert!(report.contains("command 64 (op 6): dispatch 31"));
        assert_eq!(report.matches("(op 6)").count(), 12);
        assert!(!report.contains("command 2 (op 6)"));
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    allocator: ID3D12CommandAllocator,
    queued_until: AtomicU64,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The clamped indirect records and their count, reused within this list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

fn update_memory_pressure() {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    let mut system = PERFORMANCE_INFORMATION { cb: std::mem::size_of::<PERFORMANCE_INFORMATION>() as u32, ..Default::default() };
    // SAFETY: the output buffer is correctly sized.
    if unsafe { K32GetPerformanceInfo(&mut system, system.cb) } == 0 {
        return;
    }
    let available = system.CommitLimit.saturating_sub(system.CommitTotal).saturating_mul(system.PageSize);
    crate::shadow::limit_snapshot_pool(available);
    crate::texture::trim_scratch(available);
    crate::pipeline::note_commit_headroom(available);
    #[cfg(feature = "diagnostics")]
    {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let mut process = PROCESS_MEMORY_COUNTERS { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
        // SAFETY: a valid output buffer and the current process handle.
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut process, process.cb) } == 0 {
            return;
        }
        const MIB: usize = 1024 * 1024;
        let (snapshot_buffers, snapshot_bytes) = crate::shadow::snapshot_pool_usage();
        let (staging_bytes, scratch_bytes) = crate::texture::staging_usage();
        let (bitmap_count, bitmap_bytes, page_table_bytes) = crate::memory::tracking_usage();
        tracing::info!(system_commit_mib = system.CommitTotal.saturating_mul(system.PageSize) / MIB,
            system_limit_mib = system.CommitLimit.saturating_mul(system.PageSize) / MIB,
            process_commit_mib = process.PagefileUsage / MIB,
            process_working_set_mib = process.WorkingSetSize / MIB,
            snapshot_pool_mib = snapshot_bytes / MIB, snapshot_buffers,
            shadow_buffers_mib = crate::shadow::buffer_usage() / MIB as u64,
            texture_staging_mib = staging_bytes / MIB as u64,
            texture_scratch_mib = scratch_bytes / MIB,
            bitmap_count, bitmap_backing_mib = bitmap_bytes / MIB as u64,
            page_tables_mib = page_table_bytes / MIB as u64,
            "memory pressure");
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>, bool),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(Arc<crate::memory::RecordedMap>),
}

enum After {
    Memory(Arc<crate::memory::RecordedMap>),
    Readback {
        title: usize,
        mapped: usize,
        region: ReadbackRegion,
    },
    #[cfg(feature = "diagnostics")]
    Work(Box<dyn Fn() + Send>),
}

/// Byte ranges written by a GPU copy. Padding and neighboring texels
/// stay in title memory so separate regions can share a destination.
pub(crate) struct ReadbackRegion {
    pub first: usize,
    pub row_pitch: usize,
    pub slice_pitch: usize,
    pub row_bytes: usize,
    pub rows: usize,
    pub slices: usize,
    pub packed: Option<u32>,
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Last GPU source copied into each texture backed by aliased title memory.
    texture_aliases: std::collections::HashMap<usize, usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<After>,
    view_marks: bool,
}

#[derive(Default)]
struct Submissions {
    last: u64,
    pending: VecDeque<InFlightSubmission>,
    finishing: bool,
}

struct PreparedSubmission {
    commands: Vec<Option<ID3D12CommandList>>,
    firsts: Vec<usize>,
    batch: Batch,
    remap_version: u64,
}

/// Owns the recorded commands and their title storage through the fence.
/// The storage holds allocators, scratch/uploads, resource references and
/// descriptor leases. Synchronous callers also keep their borrowed lists
/// alive until retirement.
struct InFlightSubmission {
    fence: u64,
    prepared: PreparedSubmission,
    owners: Vec<Arc<crate::list::ListStorage>>,
}

fn max_in_flight() -> usize {
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        if cfg!(feature = "diagnostics")
            && std::env::var("VICTORY_MAX_IN_FLIGHT").as_deref() == Ok("2")
        {
            2
        } else {
            1
        }
    })
}

fn oldest_when_full<T>(pending: &VecDeque<T>, depth: usize) -> Option<&T> {
    (pending.len() >= depth).then(|| pending.front()).flatten()
}

/// Drain in fence order, keeping the oldest entry and its resources alive
/// during the wait. Returns how many entries were retired.
fn retire_through<T>(
    pending: &mut VecDeque<T>,
    mut next: impl FnMut(&T) -> bool,
    mut retire: impl FnMut(T),
) -> usize {
    let mut retired = 0;
    while pending.front().is_some_and(&mut next) {
        retire(pending.pop_front().expect("checked front"));
        retired += 1;
    }
    retired
}

impl Host {
    /// A device that can run wave32 but cannot run wave64.
    pub(crate) fn wave32_only(&self) -> bool {
        static WAVE32_ONLY: OnceLock<bool> = OnceLock::new();
        *WAVE32_ONLY.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the feature query writes the given structure.
            let queried = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            let wave32_only = queried.is_ok()
                && options.WaveOps.as_bool()
                && options.WaveLaneCountMin <= 32
                && (32..64).contains(&options.WaveLaneCountMax);
            tracing::info!(
                min = options.WaveLaneCountMin,
                max = options.WaveLaneCountMax,
                wave32_only,
                "host wave sizes"
            );
            wave32_only
        })
    }

    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                allocator,
                queued_until: AtomicU64::new(0),
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Internal work may expose a readback to its caller immediately.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.execute_inner(runs, Vec::new(), 1);
    }

    pub(crate) fn execute_title(&self, owners: Vec<Arc<crate::list::ListStorage>>) {
        self.execute_title_at_depth(&owners, max_in_flight());
    }

    fn execute_title_at_depth(&self, owners: &[Arc<crate::list::ListStorage>], depth: usize) {
        let runs: Vec<_> = owners.iter().filter_map(|owner| owner.host()).collect();
        self.execute_inner(&runs, owners.to_vec(), depth);
    }

    fn execute_inner(
        &self,
        runs: &[&HostList],
        owners: Vec<Arc<crate::list::ListStorage>>,
        depth: usize,
    ) {
        let mut submissions = self.submissions.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(not(feature = "diagnostics"))]
        let traced = false;
        #[cfg(feature = "diagnostics")]
        let timed = timing();
        #[cfg(not(feature = "diagnostics"))]
        let timed = false;

        let queue = depth > 1
            && !submissions.finishing
            && !traced
            && !timed
            && !owners.is_empty()
            && runs.iter().all(|list| list.can_queue());
        if queue {
            // Replaying a list can reset its prelude allocator. Wait for its
            // last execution even when there is room in the queue.
            let last_use = runs
                .iter()
                .map(|l| l.queued_until.load(Ordering::Relaxed))
                .max()
                .unwrap_or(0);
            self.drain_through(&mut submissions, last_use);
            if let Some(oldest) = oldest_when_full(&submissions.pending, depth) {
                let fence = oldest.fence;
                self.drain_through(&mut submissions, fence);
            }
            let preparing = Instant::now();
            crate::descriptor::sync_heaps();
            let (prepared, _) = self.prepare(runs, &[], submissions.last + 1, traced, preparing);
            let fence = self
                .submit(&mut submissions, &prepared.commands)
                .unwrap_or_else(|err| fail(format_args!("could not signal host GPU fence: {err}")));
            for list in runs {
                list.queued_until.store(fence, Ordering::Release);
            }
            submissions.pending.push_back(InFlightSubmission {
                fence,
                prepared,
                owners,
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(
                crate::copystat::C::QueueDepthTwo,
                u64::from(submissions.pending.len() == 2),
            );
            return;
        }

        self.drain_through(&mut submissions, u64::MAX);
        let preparing = Instant::now();
        crate::descriptor::sync_heaps();
        crate::shadow::reconcile_loose_mappings();
        // A shared upload list is leased only for this synchronous submission.
        let mut seen_heaps = std::collections::HashSet::new();
        let heaps: Vec<_> = runs
            .iter()
            .flat_map(|list| {
                list.deferred()
                    .copies
                    .iter()
                    .filter_map(|step| match step {
                        Step::Heap(viewed, _) if seen_heaps.insert(Arc::as_ptr(viewed)) => {
                            Some(viewed.clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let (prepared, uploads) = self.prepare(runs, &heaps, submissions.last + 1, traced, preparing);
        let runs: Vec<_> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let waiting = Instant::now();
        let mut removed_in = None;
        let completed = if timed {
            let mut completed = true;
            for (i, (&first, end)) in prepared
                .firsts
                .iter()
                .zip(
                    prepared
                        .firsts
                        .iter()
                        .copied()
                        .skip(1)
                        .chain([prepared.commands.len()]),
                )
                .enumerate()
            {
                let started = Instant::now();
                completed = self
                    .submit(&mut submissions, &prepared.commands[first..end])
                    .is_ok_and(|fence| self.wait_for_fence(fence));
                if !completed {
                    removed_in = Some(i);
                    break;
                }
                #[cfg(feature = "diagnostics")]
                if started.elapsed() > Duration::from_millis(50) {
                    // SAFETY: a live queue, and this list's fence has passed.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let commands = runs[i]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(
                        ms = started.elapsed().as_millis(),
                        commands,
                        "slow host list"
                    );
                    for (ms, op) in runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
                let _ = started;
            }
            completed
        } else {
            self.submit(&mut submissions, &prepared.commands)
                .is_ok_and(|fence| self.wait_for_fence(fence))
        };
        let gpu = waiting.elapsed();
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        if gpu > Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| {
                    l.pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            tracing::warn!(
                ms = gpu.as_millis(),
                lists = runs.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // SAFETY: a plain query.
        if !completed || unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(&runs),
                last_notes(&runs),
                removed_during(&runs, removed_in)
            ));
        }
        let fence = submissions.last;
        self.retire(
            InFlightSubmission {
                fence,
                prepared,
                owners,
            },
            &runs,
        );
        release_retired();
        self.log_messages();
        #[cfg(feature = "diagnostics")]
        if traced && !crate::capture::compact() {
            drop(submissions);
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(fence, &targets);
        }
    }

    fn prepare(
        &self,
        runs: &[&HostList],
        heaps: &[Arc<crate::bindless::Viewed>],
        next_fence: u64,
        traced: bool,
        started: Instant,
    ) -> (PreparedSubmission, Option<crate::bindless::Uploads>) {
        let remap_version = crate::watch::remap_version();
        #[cfg(feature = "diagnostics")]
        let _audit = crate::shadow::writeback_audit::Retirement::begin(next_fence);
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (next_fence, traced);
        let mut batch = Batch::default();
        batch.view_marks = runs
            .iter()
            .any(|list| list.deferred().view_marks)
            .then(crate::shadow::ViewMarks::acquire)
            .flatten();
        for list in runs {
            list.run_deferred(&mut batch);
        }
        batch.flush();
        // A title copy in this batch may have changed texture backing.
        // Record the upload after those CPU copies, then execute it first.
        let uploads = {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(Instant::now(), crate::copystat::C::UploadsUs);
            (!heaps.is_empty())
                .then(|| crate::bindless::uploads(heaps))
                .flatten()
        };
        let count = runs.len() + usize::from(uploads.is_some());
        let ordered = uploads
            .as_ref()
            .map(crate::bindless::Uploads::list)
            .into_iter()
            .chain(runs.iter().copied());
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in ordered.clone().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        next_fence,
                        count
                    );
                }
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * count);
        for table in batch.memory.values() {
            match table.card_upload() {
                Ok(Some(upload)) => lists.push(Some(upload)),
                Ok(None) => {}
                Err(error) => fail(format_args!("a VRAM page table upload did not record ({error})")),
            }
        }
        // Where each run's lists start in the command array.
        let mut firsts = Vec::with_capacity(count);
        for list in ordered {
            firsts.push(if firsts.is_empty() { 0 } else { lists.len() });
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.untracked_writes = crate::memory::UntrackedWrites::capture(batch.memory.values());
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        SUBMISSION_US[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        (
            PreparedSubmission {
                commands: lists,
                firsts,
                batch,
                remap_version,
            },
            uploads,
        )
    }

    fn submit(
        &self,
        submissions: &mut Submissions,
        commands: &[Option<ID3D12CommandList>],
    ) -> windows::core::Result<u64> {
        let fence = submissions
            .last
            .checked_add(1)
            .expect("host fence overflow");
        // SAFETY: closed command lists and resources held by the prepared
        // submission or its recording, under the submission lock.
        unsafe {
            self.queue.ExecuteCommandLists(commands);
            self.queue.Signal(&self.fence, fence)?;
        }
        submissions.last = fence;
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, fence, 0);
        Ok(fence)
    }

    fn wait_for_fence(&self, fence: u64) -> bool {
        #[cfg(feature = "diagnostics")]
        let _waited = crate::watch::Timed(Instant::now(), crate::copystat::C::FenceWaitUs);
        // SAFETY: a live fence and event, exclusively waited under the
        // submission lock. Device removal reports u64::MAX.
        let completed = unsafe {
            let value = self.fence.GetCompletedValue();
            if value < fence {
                if self.fence.SetEventOnCompletion(fence, self.event).is_err()
                    || WaitForSingleObject(self.event, INFINITE) != WAIT_OBJECT_0
                {
                    return false;
                }
                self.fence.GetCompletedValue()
            } else {
                value
            }
        };
        if completed < fence || completed == u64::MAX {
            return false;
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, fence, 0);
        true
    }

    fn wait_for_list(&self, list: &HostList) {
        let mut submissions = self.submissions.lock().unwrap_or_else(|e| e.into_inner());
        let fence = list.queued_until.swap(0, Ordering::AcqRel);
        self.drain_through(&mut submissions, fence);
    }

    fn drain(&self) {
        let mut submissions = self.submissions.lock().unwrap_or_else(|e| e.into_inner());
        self.drain_through(&mut submissions, u64::MAX);
    }

    fn drain_through(&self, submissions: &mut Submissions, fence: u64) {
        let retired = retire_through(
            &mut submissions.pending,
            |work| work.fence <= fence,
            |work| {
                let waiting = Instant::now();
                if !self.wait_for_fence(work.fence) {
                    let runs: Vec<_> = work
                        .owners
                        .iter()
                        .filter_map(|owner| owner.host())
                        .collect();
                    fail(format_args!(
                        "host GPU did not complete fence {}{}{}",
                        work.fence,
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                SUBMISSION_US[1].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                self.retire(work, &[]);
            },
        );
        if retired != 0 && submissions.pending.is_empty() {
            release_retired();
            self.log_messages();
        }
    }

    fn retire(&self, work: InFlightSubmission, runs: &[&HostList]) {
        #[cfg(feature = "diagnostics")]
        let _retiring = crate::watch::Timed(Instant::now(), crate::copystat::C::RetirementUs);
        let InFlightSubmission {
            mut prepared,
            owners,
            ..
        } = work;
        if runs.is_empty() {
            // A detached submission has only GPU work. Dropping the
            // recording can now return descriptor slots and GPU objects.
            drop(prepared);
            drop(owners);
            return;
        }
        #[cfg(feature = "diagnostics")]
        let _audit = crate::shadow::writeback_audit::Retirement::begin(work.fence);
        #[cfg(feature = "diagnostics")]
        let _writeback = tracing::enabled!(target: "victory_gpu::writeback", tracing::Level::DEBUG)
            .then(|| {
                let bvh = runs.iter().any(|list| {
                    let pipelines = list.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                    pipelines.contains("[compute caa78b8ca62bc3d6]")
                        || pipelines.contains("[compute 28918b7150a7aad5]")
                });
                tracing::debug_span!(
                    target: "victory_gpu::writeback",
                    "retire",
                    fence = work.fence,
                    bvh
                ).entered()
            });
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        let mapping = crate::watch::WritebackMapping::begin(prepared.remap_version);
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        let mut list_misses = Vec::with_capacity(runs.len());
        for list in runs {
            let deferred = list.deferred();
            // Shaders that read memory with no copy computed something else
            // than the console's: it stays out of title memory, where it broke
            // the title's own data (runs c1, c4).
            let missed = deferred.after.iter().any(|work| match work {
                After::Memory(map) => prepared
                    .batch
                    .memory
                    .get(&map.key())
                    .is_some_and(|table| map.missed(table)),
                After::Readback { .. } => false,
                #[cfg(feature = "diagnostics")]
                After::Work(_) => false,
            });
            list_misses.push(missed);
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= !missed,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, !missed));
                        }
                    }
                }
            }
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        for (shadow, address, len) in &withheld {
            let submission = prepared
                .batch
                .shadows
                .get_mut(&(Arc::as_ptr(shadow) as usize))
                .expect("prepared write");
            shadow.withhold(submission, *address, *len);
        }
        if let Some(marks) = prepared.batch.view_marks.as_mut() {
            marks.copy_back(&withheld);
        }
        for (shadow, address, len, goes_back) in &back {
            let (address, len) = (*address, *len);
            if *goes_back {
                let written = shadow.copy_back(
                    prepared
                        .batch
                        .shadows
                        .get_mut(&(Arc::as_ptr(shadow) as usize))
                        .expect("prepared write"),
                    address,
                    len,
                );
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(
                    prepared
                        .batch
                        .shadows
                        .get_mut(&(Arc::as_ptr(shadow) as usize))
                        .expect("prepared write"),
                    address,
                    len,
                );
                crate::memory::note_withheld(len as u64);
            }
        }
        for (list, missed) in runs.iter().zip(list_misses) {
            let deferred = list.deferred();
            for work in &deferred.after {
                match work {
                    After::Memory(map) => {
                        if let Some(table) = prepared.batch.memory.get(&map.key()) {
                            map.copy_back(table, missed, &mut prepared.batch.untracked_writes);
                        }
                    }
                    After::Readback { title, mapped, region } => {
                        let mut packed_row = Vec::new();
                        if region.packed.is_some() {
                            if packed_row.try_reserve_exact(region.row_bytes).is_err() {
                                tracing::warn!(bytes = region.row_bytes, "no memory for a packed readback row");
                                continue;
                            }
                            packed_row.resize(region.row_bytes, 0u8);
                        }
                        for slice in 0..region.slices {
                            for row in 0..region.rows {
                                let at = region.first + slice * region.slice_pitch
                                    + row * region.row_pitch;
                                let dst = title + at;
                                if crate::watch::stale_gpu_destination(dst, region.row_bytes) {
                                    continue;
                                }
                                let source = if let Some(format) = region.packed {
                                    // SAFETY: the host pitch and texel width were both doubled.
                                    let rgba = unsafe { std::slice::from_raw_parts(
                                        (mapped + at * 2) as *const u8, region.row_bytes * 2,
                                    ) };
                                    crate::texture::pack_row(format, rgba, &mut packed_row);
                                    packed_row.as_ptr()
                                } else {
                                    (mapped + at) as *const u8
                                };
                                let _writing = crate::watch::writing(dst, region.row_bytes);
                                write_mapped(dst as *mut u8, source, region.row_bytes);
                                #[cfg(feature = "diagnostics")]
                                crate::shadow::writeback_audit::record(dst, region.row_bytes, 3);
                            }
                        }
                        let bytes = region.row_bytes * region.rows * region.slices;
                        COPY_BACK[1].fetch_add(bytes as u64, Ordering::Relaxed);
                    }
                    #[cfg(feature = "diagnostics")]
                    After::Work(work) => work(),
                }
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                if crate::watch::stale_gpu_destination(at, bytes) {
                    continue;
                }
                what.write(at as *mut u8, bytes);
                #[cfg(feature = "diagnostics")]
                crate::shadow::writeback_audit::record(at, bytes, 5);
            }
        }
        drop(mapping);
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e})"
            ));
        }
    }

    /// Detached lists may fill recording-owned upload buffers. They must
    /// have no shader bindings, mutable tracking or CPU-visible writes.
    fn can_queue(&self) -> bool {
        let deferred = self.deferred();
        deferred
            .copies
            .iter()
            .all(|step| matches!(step, Step::Fill(..)))
            && deferred.after.is_empty()
            && deferred.writes.is_empty()
            && !deferred.view_marks
            && self
                .pipelines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
    }

    pub(crate) fn wait_for_completion(&self) {
        if self.queued_until.load(Ordering::Acquire) != 0
            && let Some(host) = host()
        {
            host.wait_for_list(self);
        }
    }

    /// Reuse the recorded storage only after its last fence retires.
    pub(crate) fn reset(&self) {
        self.wait_for_completion();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        if copy.packed().is_some() && wanted != 0x400 {
            crate::texture::upload(self, resource, texture);
        }
        self.transition_copy(&copy, wanted).map(|before| (texture, before))
    }

    /// Tracks the pinned host texture and returns its previous state when
    /// this list needs a barrier.
    pub(crate) fn transition_copy(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<u32> {
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, copy, twin, wanted)
        {
            return None;
        }
        self.track(copy, legal_state(&copy.texture, wanted))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        if barriers.is_empty() && copies.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = legal_state(&t.texture, wanted);
                        (real != wanted).then(|| {
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32))
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        let markers = dred_on();
        if markers || timing() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            if markers {
                let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
                const UNICODE: u32 = 0;
                // SAFETY: an open list; the marker is copied when recorded.
                unsafe {
                    self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
                };
            }
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>, writes: bool) {
        let mut deferred = self.deferred();
        // A writing draw after a read-only draw must still arm the UAVs.
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, usize::from(writes));
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone(), writes));
        }
    }

    /// Retain the recorded bitmap and prepare its table before execution.
    pub(crate) fn defer_memory(&self, map: Arc<crate::memory::RecordedMap>) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Memory(map.clone()));
        deferred.after.push(After::Memory(map));
    }

    pub(crate) fn defer_view_marks(&self) {
        self.deferred().view_marks = true;
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(After::Work(Box::new(work)));
    }

    /// Publishes the GPU-written rows after the fence and before title
    /// completion signals. The list keeps the buffer that owns `mapped`.
    pub(crate) fn defer_readback(
        &self,
        title: *mut u8,
        mapped: *const u8,
        region: ReadbackRegion,
    ) {
        self.deferred().after.push(After::Readback {
            title: title as usize,
            mapped: mapped as usize,
            region,
        });
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { _buffer: buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.gpu + at, (chunk.mapped + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    pub(crate) fn indirect_args(&self, size: u64) -> Option<ID3D12Resource> {
        let mut scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, capacity)) = &*scratch
            && *capacity >= size
        {
            return Some(buffer.clone());
        }
        let capacity = size.max(4096).checked_next_power_of_two()?;
        let (buffer, _) = host()?.shared_buffer(capacity)?;
        if let Some((previous, _)) = scratch.replace((buffer.clone(), capacity)) {
            // Earlier commands in this list still refer to the old buffer.
            self.keep(&previous);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Reuse an alias copy while the source is still in the copy-source
    /// state. Leaving that state allows the title to write the source again.
    pub(crate) fn alias_copy_current(
        &self,
        from: &Arc<crate::resource::HostTexture>,
        to: &Arc<crate::resource::HostTexture>,
    ) -> bool {
        let (from, to) = (Arc::as_ptr(from) as usize, Arc::as_ptr(to) as usize);
        if self.deferred().texture_aliases.get(&to) != Some(&from) {
            return false;
        }
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&from)
            .is_some_and(|(_, state, _)| *state == D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)
    }

    pub(crate) fn note_alias_copy(
        &self,
        from: &Arc<crate::resource::HostTexture>,
        to: &Arc<crate::resource::HostTexture>,
    ) {
        self.deferred()
            .texture_aliases
            .insert(Arc::as_ptr(to) as usize, Arc::as_ptr(from) as usize);
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Run(_) | Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(_) => C::StepTextureUs,
                    Step::Heap(..) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    if len == 0 {
                        continue;
                    }
                    batch.flush();
                    let (mapping, ranges) = crate::watch::write_ranges(dst, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    #[cfg(feature = "diagnostics")]
                    crate::shadow::writeback_audit::record(dst, len, 4);
                    if mapping == crate::watch::mapping_version() {
                        batch.title_copied(&ranges);
                    } else {
                        batch.title_written();
                    }
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked && !shadow.is_shared() {
                        batch.flush();
                    }
                    let state = batch
                        .shadows
                        .entry(Arc::as_ptr(shadow) as usize)
                        .or_default();
                    shadow.writing(state, *address, *len, *marked);
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
                Step::Heap(viewed, writes) => {
                    let arm = *writes && batch.view_marks.is_some();
                    let key = (Arc::as_ptr(viewed) as usize, arm);
                    let previous = batch.refreshed_heaps.get(&key).copied();
                    if let Some(revision) = crate::bindless::refresh(
                        viewed,
                        previous,
                        |shadow, address, len, uav| {
                            batch.add(shadow, address, len);
                            if arm && uav && shadow.marked() {
                                batch.arm_heap_write(shadow, address, len);
                            }
                        },
                    ) {
                        batch.refreshed_heaps.insert(key, revision);
                        if arm {
                            batch.refreshed_heaps.insert((key.0, false), revision);
                        }
                    }
                    continue;
                }
                Step::Memory(map) => {
                    let key = map.key();
                    let mut table = batch.memory.remove(&key).unwrap_or_else(|| map.prepare());
                    // Repeated lists use the exact same layout. Its ranges
                    // are already pending or current until a title write.
                    if batch.refreshed_tables.insert(key) {
                        table.refresh(|s, a, l| batch.add(s, a, l));
                    }
                    table.keep(map);
                    batch.memory.insert(key, table);
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged where ranges touch:
/// done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
#[derive(Default)]
struct Batch {
    shadows: std::collections::HashMap<usize, crate::shadow::Submission>,
    memory: std::collections::HashMap<usize, crate::memory::PreparedTable>,
    untracked_writes: crate::memory::UntrackedWrites,
    refreshed_tables: std::collections::HashSet<usize>,
    refreshed_heaps: std::collections::HashMap<(usize, bool), u64>,
    view_marks: Option<crate::shadow::ViewMarks>,
    heap_writes: std::collections::HashMap<usize, CopyRanges>,
    refreshes: Vec<CopyRanges>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The ranges of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: std::collections::HashMap<usize, CopyRanges>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

struct CopyRanges {
    shadow: Arc<crate::shadow::Shadow>,
    ranges: Vec<(usize, usize)>,
}

/// Keep sorted, disjoint half-open ranges, joining neighbors without
/// turning the gaps between them into memory copies.
pub(crate) fn add_range(ranges: &mut Vec<(usize, usize)>, mut from: usize, mut to: usize) {
    let first = ranges.partition_point(|&(_, end)| end < from);
    let last = ranges.partition_point(|&(start, _)| start <= to);
    if first == last {
        ranges.insert(first, (from, to));
    } else {
        from = from.min(ranges[first].0);
        to = to.max(ranges[last - 1].1);
        ranges[first] = (from, to);
        ranges.drain(first + 1..last);
    }
}

impl Batch {
    fn arm_heap_write(&mut self, shadow: &Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address.saturating_add(len);
        let armed = self.heap_writes.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| CopyRanges {
            shadow: shadow.clone(),
            ranges: Vec::new(),
        });
        let at = armed.ranges.partition_point(|&(_, to)| to <= address);
        if !armed.ranges.get(at).is_some_and(|&(from, to)| from <= address && end <= to) {
            shadow.arm_view_writes(address, end);
            add_range(&mut armed.ranges, address, end);
        }
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        if shadow.is_shared() {
            return;
        }
        let key = Arc::as_ptr(shadow) as usize;
        let end = address.saturating_add(len);
        if address >= end {
            return;
        }
        if let Some(current) = self.current.get(&key) {
            let at = current.ranges.partition_point(|&(_, to)| to <= address);
            if current.ranges.get(at).is_some_and(|&(from, to)| from <= address && end <= to) {
                return;
            }
        }
        match self.pending.get(&key) {
            Some(&i) => {
                add_range(&mut self.refreshes[i].ranges, address, end);
            }
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push(CopyRanges {
                    shadow: shadow.clone(),
                    ranges: vec![(address, end)],
                });
            }
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        self.refreshed_tables.clear();
        self.refreshed_heaps.clear();
        self.current.clear();
    }

    /// A recorded copy names its destination. Keep refresh coverage for
    /// all other pages, including other ranges of a partially hit shadow.
    fn title_copied(&mut self, written: &[(usize, usize)]) {
        self.textures.clear();
        self.refreshed_tables.clear();
        self.refreshed_heaps.clear();
        self.current.retain(|_, current| {
            let (start, len) = current.shadow.range();
            let end = start + len;
            let first = written.partition_point(|&(_, to)| to as u64 <= start);
            for &(from, to) in &written[first..] {
                if from as u64 >= end {
                    break;
                }
                let ranges = &mut current.ranges;
                let first = ranges.partition_point(|&(_, end)| end <= from);
                let last = ranges.partition_point(|&(start, _)| start < to);
                if first < last {
                    let (start, end) = (ranges[first].0, ranges[last - 1].1);
                    let left = (start < from).then_some((start, from));
                    let right = (end > to).then_some((to, end));
                    ranges.splice(first..last, left.into_iter().chain(right));
                }
            }
            !current.ranges.is_empty()
        });
    }

    /// Does the refreshes; returns the bytes copied.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let mut total = 0;
        for CopyRanges { shadow, ranges } in self.refreshes.drain(..) {
            let key = Arc::as_ptr(&shadow) as usize;
            let current = self.current.entry(key).or_insert_with(|| CopyRanges {
                shadow: shadow.clone(),
                ranges: Vec::new(),
            });
            for (from, to) in ranges {
                let mut refresh = |from, to| {
                    let started = std::time::Instant::now();
                    let copied = shadow.update(from, to - from, self.shadows.get_mut(&key));
                    let us = started.elapsed().as_micros() as u64;
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::range(from, to - from, us, 0);
                    PREPARING[0].fetch_add(us, Ordering::Relaxed);
                    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                    total += copied;
                };
                let mut at = from;
                let first = current.ranges.partition_point(|&(_, end)| end <= at);
                for &(start, end) in &current.ranges[first..] {
                    if start >= to {
                        break;
                    }
                    if at < start {
                        refresh(at, start);
                    }
                    at = at.max(end);
                }
                if at < to {
                    refresh(at, to);
                }
                add_range(&mut current.ranges, from, to);
            }
        }
        total
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        for armed in self.heap_writes.values() {
            armed.shadow.disarm_view_writes();
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let copied = read_prefix(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if copied == len {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst {
        copy_sparse_to_host(dst.wrapping_add(copied), src.wrapping_add(copied), len - copied);
        return;
    }
    copy_in_pieces(dst, src, len);
}

/// After the fast bulk read fails, skip holes in sparse title allocations
/// a region at a time. The destination is a host buffer for the entire range.
fn copy_sparse_to_host(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };

    let start = src as usize;
    let end = start + len;
    let mut at = start;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: the query fills a valid buffer and doesn't dereference `at`.
        let queried = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        let to = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        let out = dst.wrapping_add(at - start);
        if queried != size_of_val(&info) || to <= at {
            copy_in_pieces(out, at as *const u8, end - at);
            return;
        }
        let bytes = to - at;
        if matches!(info.State, MEM_FREE | MEM_RESERVE) {
            zero_unreadable(out, bytes);
        } else {
            let copied = read_prefix(at as *const u8, out, bytes);
            if copied < bytes {
                // Mappings and protections can change after the query.
                copy_in_pieces(out.wrapping_add(copied), (at + copied) as *const u8, bytes - copied);
            }
        }
        at = to;
    }
}

/// Fall back to 64 KiB and then 4 KiB reads where a larger read fails.
fn copy_in_pieces(dst: *mut u8, src: *const u8, len: usize) {
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    zero_unreadable(to, n);
                }
            });
        }
    });
}

fn zero_unreadable(dst: *mut u8, len: usize) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Unmapped, len as u64);
    // SAFETY: inside the mapped destination range.
    unsafe { dst.write_bytes(0, len) };
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    read_prefix(src, dst, len) == len
}

/// How much of the contiguous prefix the kernel copied before a hole.
fn read_prefix(src: *const u8, dst: *mut u8, len: usize) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    if unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) } != 0 {
        len
    } else {
        done.min(len)
    }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn a_sparse_read_zeroes_only_the_hole_and_respects_unaligned_ends() {
        let from = holed(0x67);
        let mut to = vec![0xee; 3 * BLOCK];
        // SAFETY: offsets are inside the test reservation and output buffer.
        copy_mapped(unsafe { to.as_mut_ptr().add(17) }, unsafe { from.add(17) }, 3 * BLOCK - 46, false);
        assert!(to[..17].iter().all(|&b| b == 0xee));
        assert!(to[17..BLOCK].iter().all(|&b| b == 0x67));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..3 * BLOCK - 29].iter().all(|&b| b == 0x67));
        assert!(to[3 * BLOCK - 29..].iter().all(|&b| b == 0xee));
        release(from);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// Local memory budget and current usage for the adapter executing our lists.
    pub(crate) fn local_video_memory(&self) -> Option<(u64, u64)> {
        let adapter = self.memory_adapter.get_or_init(|| {
            // SAFETY: plain DXGI queries on the live device's adapter.
            unsafe {
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(self.device.GetAdapterLuid()).ok()?;
                (adapter.GetDesc1().ok()?.DedicatedVideoMemory != 0).then_some(adapter)
            }
        }).as_ref()?;
        let mut memory = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a live adapter and writable output for its local segment.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut memory).ok()? };
        Some((memory.Budget, memory.CurrentUsage))
    }

    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_in_state(size, D3D12_RESOURCE_STATE_COPY_DEST)
    }

    pub(crate) fn default_buffer_in_state(
        &self,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    composer: Option<crate::compose::Composer>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let root_views = crate::draw::take_root_view_slots() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            update_memory_pressure();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let deferred = crate::resource::DEFERRED_TEXTURES.load(Ordering::Relaxed);
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 deferred host allocation for {deferred}; copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                let copy_frames = std::mem::take(&mut self.copy_stats_frames) as f64;
                crate::copystat::log(copy_frames);
                crate::bindless::log(copy_frames);
                crate::descriptor::host::log_heap_overflow();
                crate::shadow::log_views(copy_frames);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        // SAFETY: live resources supplied by the title's queue.
        let desc = unsafe { texture.GetDesc() };
        let planes = overlay.and_then(|(overlay, overlay_state)| {
            let overlay_desc = unsafe { overlay.GetDesc() };
            Some((
                overlay,
                D3D12_RESOURCE_STATES(overlay_state as i32),
                crate::compose::view(&desc)?,
                crate::compose::view(&overlay_desc)?,
            ))
        });
        let format = if planes.is_some() {
            DXGI_FORMAT_R8G8B8A8_UNORM
        } else if let Some(format) = swap_format(desc.Format) {
            format
        } else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        if planes.is_some() && p.composer.is_none() {
            match crate::compose::Composer::new(self, format) {
                Ok(composer) => {
                    p.composer = Some(composer);
                    tracing::info!(frame, width, height, "compositing two display planes");
                }
                Err(error) => {
                    tracing::warn!(%error, "display compositor unavailable");
                    drop(presentation);
                    return self.present(texture, state, None, frame);
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            let copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
            let present = D3D12_RESOURCE_STATE_PRESENT;
            let (shown, back_state) = if let (
                Some((over, over_state, base_fmt, over_fmt)),
                Some(composer),
            ) = (planes, p.composer.as_mut())
            {
                let shader_read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
                let sources = [(texture, state), (over, over_state)];
                let mut barriers = Vec::with_capacity(3);
                barriers.extend(sources.iter().filter(|(_, state)| *state != shader_read)
                    .map(|(source, state)| transition(source, *state, shader_read)));
                barriers.push(transition(&back, present, target));
                list.ResourceBarrier(&barriers);
                composer.record(
                    self,
                    list,
                    [(texture, base_fmt), (over, over_fmt)],
                    &back,
                    (width, height),
                );
                barriers.clear();
                barriers.extend(sources.iter().filter(|(_, state)| *state != shader_read)
                    .map(|(source, state)| transition(source, shader_read, *state)));
                barriers.push(transition(&back, target, copy_source));
                list.ResourceBarrier(&barriers);
                (&back, copy_source)
            } else {
                let dest = D3D12_RESOURCE_STATE_COPY_DEST;
                let barriers = [
                    transition(&back, present, dest),
                    transition(texture, state, copy_source),
                ];
                list.ResourceBarrier(&barriers[..if state == copy_source { 1 } else { 2 }]);
                list.CopyResource(&back, texture);
                (texture, dest)
            };
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, shown))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, shown);
            if back_state == D3D12_RESOURCE_STATE_COPY_DEST && state != copy_source {
                list.ResourceBarrier(&[transition(shown, copy_source, state)]);
            }
            list.ResourceBarrier(&[transition(&back, back_state, present)]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                composer: None,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let bucket = SharedBufferRetry::bucket(size);
        let probe = self.shared_buffer_retry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .allow(bucket, Instant::now())?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        let out_of_memory = made.as_ref().is_err_and(|e| e.code() == windows::Win32::Foundation::E_OUTOFMEMORY);
        let log_oom = (probe || out_of_memory) && self.shared_buffer_retry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish(bucket, probe, out_of_memory, Instant::now());
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                if !out_of_memory || log_oom {
                    tracing::warn!(error = %e, size, "no shared buffer");
                }
                None
            }
        }
    }
}

/// Avoid repeated driver allocation and logging when many draws need the
/// same size of shared buffer. Smaller requests can still try their own class.
struct SharedBufferRetry {
    classes: [SharedBufferPressure; 49],
}

#[derive(Clone, Copy)]
enum SharedBufferPressure {
    Ready,
    Cooling(Instant),
    Probing,
}

impl Default for SharedBufferRetry {
    fn default() -> Self {
        Self { classes: [SharedBufferPressure::Ready; 49] }
    }
}

impl SharedBufferRetry {
    const DELAY: Duration = Duration::from_millis(250);

    fn bucket(size: u64) -> usize {
        (u64::BITS - (size.max(1 << 16) - 1).leading_zeros() - 16) as usize
    }

    /// None while blocked, otherwise whether the admitted call is a probe.
    fn allow(&mut self, bucket: usize, now: Instant) -> Option<bool> {
        match self.classes[bucket] {
            SharedBufferPressure::Ready => Some(false),
            SharedBufferPressure::Cooling(until) if now >= until => {
                // Admit one probe while other callers may still be recording.
                self.classes[bucket] = SharedBufferPressure::Probing;
                Some(true)
            }
            _ => None,
        }
    }

    /// Returns true when this failure starts a period of memory pressure.
    fn finish(&mut self, bucket: usize, probe: bool, out_of_memory: bool, now: Instant) -> bool {
        let class = &mut self.classes[bucket];
        if probe {
            *class = if out_of_memory {
                SharedBufferPressure::Cooling(now + Self::DELAY)
            } else {
                SharedBufferPressure::Ready
            };
        } else if out_of_memory && matches!(class, SharedBufferPressure::Ready) {
            *class = SharedBufferPressure::Cooling(now + Self::DELAY);
            return true;
        }
        false
    }
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(resource);
}

/// After a submission has run, with none running: the retired resources go.
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    drop(retired);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds, the first
/// 500.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let hashes = HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    });
    if hashes.iter().any(|h| what.contains(h.as_str())) && LOGGED.fetch_add(1, Ordering::Relaxed) < 500 {
        tracing::info!("traced command: {what}");
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; recorded tail of list {i} of {} ({} notes): {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    // Keep the stopping point visible even in a long barrier interval.
    let is_command = |i: &usize| history[*i] != D3D12_AUTO_BREADCRUMB_OP_SETMARKER;
    let start = (from..done).rev().filter(is_command).nth(5).unwrap_or(from);
    for i in (start..to).filter(is_command).take(12) {
        let op = history[i];
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    #[test]
    fn a_known_copy_refreshes_aliased_pages_and_keeps_unrelated_coverage() {
        use crate::watch::{self, PAGE};
        let (p, alias) = watch::tests::twin_views(3);
        let len = 3 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        if let Some(copy) = crate::shadow::find(p as u64, len as u64) {
            let mut batch = Batch::default();
            batch.add(&copy, p, len);
            assert_eq!(batch.flush(), len);
            let dst = alias + PAGE + 16;
            let (_, written) = watch::write_ranges(dst, 4);
            let value = 77u32.to_le_bytes();
            super::copy_mapped(dst as *mut u8, value.as_ptr(), value.len(), true);
            batch.title_copied(&written);
            assert_eq!(batch.current.values().next().unwrap().ranges,
                [(p, p + PAGE), (p + 2 * PAGE, p + len)]);
            batch.add(&copy, p, len);
            assert_eq!(batch.flush(), PAGE);
            assert_eq!(copy.read_host((p + PAGE + 16) as u64, 1), [77]);
        }
        crate::shadow::unregister(p, p as u64);
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for address in [p, alias] {
            watch::title_memory_remapped(address, len);
            // SAFETY: this test owns the two views from twin_views.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: address as *mut _ }) };
        }
    }

    #[test]
    fn disjoint_refreshes_skip_the_gap_until_it_is_requested() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(5);
        let len = 5 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            crate::watch::tests::free(p);
            return;
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, PAGE);
        batch.add(&copy, p + 4 * PAGE, PAGE);
        assert_eq!(batch.flush(), 2 * PAGE, "copy only the two requested pages");
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 3 * PAGE, "fill the gap without copying the ends again");
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0);
        crate::shadow::unregister(p, p as u64);
        drop(batch);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}
