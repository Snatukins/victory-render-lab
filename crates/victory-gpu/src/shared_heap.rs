//! Experimental in-place backing for the title's low 174 MiB graphics
//! buffer. Based on the gleaf fork's renderer port 1 (e4a531f).

use std::collections::BTreeMap;
use std::os::windows::io::{AsRawHandle, OwnedHandle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
};

const PAGE: u64 = 64 << 10;
const BUFFER_BYTES: u64 = 174 << 20;
const LOW_END: u64 = 1 << 32;
static MAPPINGS: Mutex<BTreeMap<u64, Mapping>> = Mutex::new(BTreeMap::new());

struct Mapping {
    end: u64,
    section: Arc<OwnedHandle>,
    offset: u64,
    current: Arc<AtomicBool>,
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_SHARED_GRAPHICS_HEAP"))
}

/// Register a successfully committed XMem view of a pagefile-backed section.
pub fn mapped(start: usize, len: usize, section: &Arc<OwnedHandle>, offset: u64) {
    let start = start as u64;
    let Some(end) = start.checked_add(len as u64) else { return };
    if enabled() && start < end && end <= LOW_END {
        MAPPINGS.lock().unwrap_or_else(|e| e.into_inner()).insert(start, Mapping {
            end,
            section: section.clone(),
            offset,
            current: Arc::new(AtomicBool::new(true)),
        });
    }
}

pub(crate) fn unmapped(start: usize, len: usize) {
    if !enabled() || len == 0 || start as u64 >= LOW_END {
        return;
    }
    let start = start as u64;
    let end = start.saturating_add(len as u64);
    MAPPINGS.lock().unwrap_or_else(|e| e.into_inner()).retain(|&base, mapping| {
        let keep = mapping.end <= start || base >= end;
        if !keep {
            mapping.current.store(false, Ordering::Release);
        }
        keep
    });
}

struct SectionView(MEMORY_MAPPED_VIEW_ADDRESS);

// SAFETY: The view is only unmapped when the GPU buffer's last owner drops.
unsafe impl Send for SectionView {}
unsafe impl Sync for SectionView {}

impl Drop for SectionView {
    fn drop(&mut self) {
        // SAFETY: This is our independent view of the section.
        unsafe { UnmapViewOfFile(self.0) };
    }
}

pub(crate) struct Backing {
    // The resource is held by Shadow and drops before these fields.
    _heap: ID3D12Heap,
    view: SectionView,
    mappings: Vec<Arc<AtomicBool>>,
}

impl Backing {
    pub(crate) fn data(&self) -> *mut u8 {
        self.view.0.Value.cast()
    }

    pub(crate) fn is_current(&self) -> bool {
        self.mappings.iter().all(|m| m.load(Ordering::Acquire))
    }
}

/// Open a second view so the driver pins our address. Sections with holes,
/// reordered slices, or mappings in different sections keep the copy path.
pub(crate) fn place(start: u64, len: u64) -> Option<(ID3D12Resource, Backing)> {
    if !enabled() || len != BUFFER_BYTES || start >= LOW_END || !start.is_multiple_of(PAGE) {
        return None;
    }
    let Some((section, offset, mappings)) = section_range(start, start + len) else {
        tracing::info!(start, len, "shared graphics heap unavailable: title range is not one section run");
        return None;
    };
    match open(&crate::host::host()?.device, &section, offset, len, mappings) {
        Ok((buffer, backing)) if backing.is_current() => {
            tracing::info!(start, len, "shared graphics heap placed in the title's pages");
            Some((buffer, backing))
        }
        Ok(_) => None,
        Err(error) => {
            tracing::warn!(start, len, %error, "shared graphics heap refused; using a copy");
            None
        }
    }
}

fn section_range(start: u64, end: u64) -> Option<(Arc<OwnedHandle>, u64, Vec<Arc<AtomicBool>>)> {
    let maps = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
    let (&base, first) = maps.range(..=start).next_back()?;
    if first.end <= start {
        return None;
    }
    let offset = first.offset + start - base;
    let mut current = vec![first.current.clone()];
    let mut next = first.end;
    while next < end {
        let item = maps.get(&next)?;
        if !Arc::ptr_eq(&first.section, &item.section) || item.offset != offset + next - start {
            return None;
        }
        current.push(item.current.clone());
        next = item.end;
    }
    Some((first.section.clone(), offset, current))
}

fn open(
    device: &ID3D12Device,
    section: &OwnedHandle,
    offset: u64,
    len: u64,
    mappings: Vec<Arc<AtomicBool>>,
) -> anyhow::Result<(ID3D12Resource, Backing)> {
    let device13: ID3D12Device13 = device.cast()?;
    // SAFETY: The registered views cover committed pages of this live section.
    let view = unsafe {
        MapViewOfFile(section.as_raw_handle(), FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, len as usize)
    };
    if view.Value.is_null() {
        return Err(std::io::Error::last_os_error().into());
    }
    let view = SectionView(view);
    // SAFETY: The view covers the requested size and outlives the heap.
    let heap: ID3D12Heap = unsafe { device13.OpenExistingHeapFromAddress1(view.0.Value, len as usize)? };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer = None;
    // SAFETY: The buffer fits at offset zero in the opened heap.
    unsafe { device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)? };
    Ok((buffer.ok_or_else(|| anyhow::anyhow!("no placed buffer"))?, Backing { _heap: heap, view, mappings }))
}
