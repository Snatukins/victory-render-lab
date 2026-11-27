//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};
#[cfg(feature = "diagnostics")]
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// shortcut: GPU copies between different packed layouts need a conversion pass.
pub(crate) fn incompatible_copy(src: Option<u32>, dst: Option<u32>) -> bool {
    if src == dst {
        return false;
    }
    once!(?src, ?dst, "GPU texture copy or resolve skipped: packed stand-in layouts differ");
    true
}

/// Expands one row of packed guest texels into the host format.
pub(crate) fn expand_row(format: u32, packed: &[u8], rgba: &mut [u8]) {
    if format == 67 {
        for (src, dst) in packed.as_chunks::<4>().0.iter().zip(rgba.as_chunks_mut::<8>().0) {
            let bits = u32::from_le_bytes(*src);
            let exponent = bits >> 27;
            for (i, channel) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let half = if i == 3 {
                    0x3c00
                } else {
                    let units = u64::from((bits >> (i * 9)) & 511) << exponent;
                    // Both formats have subnormal units of 2^-24. The
                    // nine-bit mantissa fits exactly in a half's significand.
                    if units < 1024 {
                        units as u16
                    } else {
                        let shift = units.ilog2() - 10;
                        ((shift as u16 + 1) << 10) | ((units >> shift) as u16 & 1023)
                    }
                };
                *channel = half.to_le_bytes();
            }
        }
        return;
    }
    for (p, c) in packed.as_chunks::<2>().0.iter().zip(rgba.as_chunks_mut::<4>().0.iter_mut()) {
        let bits = u16::from_le_bytes(*p);
        let channel = |shift: u32, max: u16| (((bits >> shift & max) * 255 + max / 2) / max) as u8;
        let color = match format {
            86 => [channel(10, 31), channel(5, 31), channel(0, 31), channel(15, 1)],
            115 => [channel(8, 15), channel(4, 15), channel(0, 15), channel(12, 15)],
            _ => unreachable!(),
        };
        *c = color;
    }
}

pub(crate) fn pack_row(format: u32, rgba: &[u8], packed: &mut [u8]) {
    if format == 67 {
        for (src, dst) in rgba.as_chunks::<8>().0.iter().zip(packed.as_chunks_mut::<4>().0) {
            let rgb = std::array::from_fn::<_, 3, _>(|i| {
                let half = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let exponent = (half >> 10) & 31;
                let mantissa = u64::from(half & 1023);
                const MAX: u64 = 511 << 31;
                match (half >> 15, exponent) {
                    (1, _) => 0,
                    (0, 31) if mantissa != 0 => 0,
                    (0, 31) => MAX,
                    (0, 0) => mantissa,
                    _ => ((1024 + mantissa) << (exponent - 1)).min(MAX),
                }
            });
            let max = rgb.into_iter().max().unwrap_or(0);
            let mut shift = max.checked_ilog2().unwrap_or(0).saturating_sub(8);
            let quantize = |value: u64, shift: u32| (value + ((1u64 << shift) >> 1)) >> shift;
            if quantize(max, shift) == 512 && shift < 31 {
                shift += 1;
            }
            let [r, g, b] = rgb.map(|value| quantize(value, shift).min(511) as u32);
            *dst = (r | g << 9 | b << 18 | shift << 27).to_le_bytes();
        }
        return;
    }
    for (c, p) in rgba.as_chunks::<4>().0.iter().zip(packed.as_chunks_mut::<2>().0.iter_mut()) {
        let channel = |i: usize, max: u16| (u16::from(c[i]) * max + 127) / 255;
        let bits = match format {
            86 => channel(3, 1) << 15 | channel(0, 31) << 10 | channel(1, 31) << 5 | channel(2, 31),
            115 => channel(3, 15) << 12 | channel(0, 15) << 8 | channel(1, 15) << 4 | channel(2, 15),
            _ => unreachable!(),
        };
        *p = bits.to_le_bytes();
    }
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    packed: Option<u32>,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    refresh: Mutex<RefreshState>,
}

#[derive(Default)]
struct RefreshState {
    last_hash: Option<blake3::Hash>,
    /// New staging data that the bindless upload list has yet to copy.
    pending_upload: bool,
}

/// Tiled bytes are needed only while detiling, never by the recorded GPU copy.
static TILED_SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());
const MAX_TILED_SCRATCH: usize = 16 << 20;
#[cfg(feature = "diagnostics")]
static STAGING_BYTES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn trim_scratch(available: usize) {
    if available < 2 << 30 && let Ok(mut scratch) = TILED_SCRATCH.try_lock() {
        *scratch = Vec::new();
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn staging_usage() -> (u64, usize) {
    let scratch = TILED_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    (STAGING_BYTES.load(Ordering::Relaxed), scratch.capacity())
}

// SAFETY: the staging mapping is written only at submission, under
// `refresh`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records a copy from a GPU-written alias into `to` when its source changes:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), so sampling and copying
/// read what the GPU wrote. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, resource: This, to: &Arc<crate::resource::HostTexture>) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(from), Some(host)) = (crate::resource::alias_source(resource), host()) else {
        return;
    };
    if incompatible_copy(from.packed(), to.packed()) || list.alias_copy_current(&from, to) {
        return;
    }
    let (a, b) = (&from.texture, &to.texture);
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    if da.SampleDesc.Count != 1 || db.SampleDesc.Count != 1 {
        return;
    }
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut row_bytes = vec![0u64; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device.GetCopyableFootprints(
                d,
                0,
                n,
                0,
                Some(f.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                Some(row_bytes.as_mut_ptr()),
                Some(&mut total),
            );
        }
        (f, rows, row_bytes, total)
    };
    let (fa, ra, sa, total) = footprints(&da, count(&da));
    let (fb, rb, sb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset
                && x.Footprint.RowPitch == y.Footprint.RowPitch
                && ra[i] == rb[i]
                && sa[i] == sb[i]
                && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    to.host_written
        .store(true, std::sync::atomic::Ordering::Relaxed);
    crate::list::ensure_copy_state(list, &from, COPY_SOURCE);
    crate::list::ensure_copy_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    list.note_alias_copy(&from, to);
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    let pair = (a.as_raw() as usize, b.as_raw() as usize);
    if logged.len() < 64 && !logged.contains(&pair) {
        logged.push(pair);
        tracing::info!(from = ?a.as_raw(), to = ?resource, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    record(list, resource, texture, &staging);
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    let Some(staging) = staging(resource, texture) else {
        return false;
    };
    if !staging.refresh_staging(true) {
        return false;
    }
    record(list, resource, texture, &staging);
    list.keep(&staging.buffer);
    true
}

/// Records the copy of every subresource of `staging` into `texture`,
/// moving it to the copy destination state first.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        self.refresh_staging(false)
    }

    fn refresh_staging(&self, claim_upload: bool) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let mut state = self.refresh.lock().unwrap_or_else(|e| e.into_inner());
        let (read, changed) = self.refresh_now(&mut state.last_hash);
        // An explicit list can refresh the shared staging buffer before
        // the upload list records. Its GPU copy may run later.
        state.pending_upload |= changed;
        let needed = if claim_upload {
            std::mem::take(&mut state.pending_upload)
        } else {
            changed
        };
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        needed
    }

    /// Returns the bytes of title memory read, and whether it detiled.
    fn refresh_now(&self, last: &mut Option<blake3::Hash>) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        if self.tracked.as_ref().is_some_and(|t| !t.has_changes()) {
            return (0, false);
        }
        let mut pooled;
        let mut large = Vec::new();
        let tiled = if size <= MAX_TILED_SCRATCH {
            pooled = TILED_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            &mut *pooled
        } else {
            &mut large
        };
        // Reserve before taking dirty pages so an allocation failure
        // leaves the texture due for refresh.
        if tiled.try_reserve_exact(size.saturating_sub(tiled.len())).is_err() {
            once!(size, "could not allocate texture detile scratch");
            return (0, false);
        }
        tiled.resize(size, 0);
        if let Some(t) = &self.tracked {
            if t.take(start, start + size).is_empty() {
                return (0, false);
            }
            // Missing pages of tracked, sparse textures read as zeros.
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            self.detile(tiled);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        // The title can unmap these pages while a texture is being read.
        // Keep the previous staging data if the read cannot finish.
        if !crate::host::read_into(self.memory, tiled) {
            return (0, false);
        }
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if *last == Some(hash) {
            return (size, false);
        }
        *last = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe { std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len) };
            let detile = |dst: &mut [u8], pitch| self.layout.detile(tiled, mip, slice, dst, pitch, pitch * rows as usize);
            let result = if let Some(format) = self.packed {
                let row = f.Width as usize * self.layout.desc.element_bytes as usize;
                let mut packed = vec![0; row * rows as usize * f.Depth as usize];
                detile(&mut packed, row).map(|()| {
                    for (src, dst) in packed.chunks_exact(row).zip(linear.chunks_mut(pitch)) {
                        expand_row(format, src, dst);
                    }
                })
            } else {
                detile(linear, pitch)
            };
            if let Err(e) = result {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for Staging {
    fn drop(&mut self) {
        STAGING_BYTES.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
        packed: copy.packed(),
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        refresh: Mutex::default(),
    });
    #[cfg(feature = "diagnostics")]
    STAGING_BYTES.fetch_add(total, Ordering::Relaxed);
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_staging_refreshes_after_guest_pixels_change() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let Some(host) = host() else { return };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 1,
            height: 1,
            depth: 1,
            mips: 1,
            element_bytes: 2,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        }).unwrap();
        let address = crate::watch::tests::pool(1);
        let (buffer, data) = host.upload_buffer(256).unwrap();
        let tracked = crate::watch::track(address, layout.size as usize, true).unwrap();
        #[cfg(feature = "diagnostics")]
        STAGING_BYTES.fetch_add(256, Ordering::Relaxed);
        let staging = Staging {
            buffer, size: 256, data, packed: Some(115),
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Width: 1, Height: 1, Depth: 1, RowPitch: 256, ..Default::default()
                },
            }],
            rows: vec![1], mips: 1, layout, memory: address as *const u8,
            tracked: Some(tracked),
            refresh: Mutex::default(),
        };
        for (word, expected) in [(0xff00u16, [255, 0, 0, 255]), (0x800f, [0, 0, 255, 136])] {
            // SAFETY: the fixture owns this guest texel; the tracked write
            // sees the same guard fault a title writer does.
            unsafe { (address as *mut u16).write_volatile(word.to_le()) };
            assert!(staging.refresh());
            // SAFETY: this test alone uses the mapped upload buffer.
            assert_eq!(unsafe { data.cast::<[u8; 4]>().read() }, expected);
            assert!(!staging.refresh());
        }
        drop(staging);
        crate::watch::tests::free(address);
    }

    #[test]
    fn packed_hdr_expands_and_round_trips_without_clipping() {
        for bits in [1u32, 16 << 27 | 256 | 128 << 9 | 64 << 18, u32::MAX] {
            let mut host = [0; 8];
            let mut guest = [0; 4];
            expand_row(67, &bits.to_le_bytes(), &mut host);
            assert_eq!(&host[6..], &0x3c00u16.to_le_bytes());
            pack_row(67, &host, &mut guest);
            assert_eq!(u32::from_le_bytes(guest), bits);
        }
        let mut host = [0; 8];
        expand_row(67, &(18u32 << 27 | 256).to_le_bytes(), &mut host);
        assert_eq!(&host[..2], &0x4400u16.to_le_bytes()); // red = 4.0
    }

    #[test]
    fn packed_colors_round_trip_with_their_alpha() {
        for (format, word, expected) in [
            (86, 0x8443u16, [8, 16, 25, 255]),
            (115, 0xa123, [17, 34, 51, 170]),
        ] {
            let mut rgba = [0; 4];
            let mut packed = [0; 2];
            expand_row(format, &word.to_le_bytes(), &mut rgba);
            assert_eq!(rgba, expected);
            for word in 0..=u16::MAX {
                expand_row(format, &word.to_le_bytes(), &mut rgba);
                pack_row(format, &rgba, &mut packed);
                assert_eq!(u16::from_le_bytes(packed), word);
            }
        }
        for (alpha, high) in [(127, 0), (128, 0x80)] {
            let mut packed = [0; 2];
            pack_row(86, &[0, 0, 0, alpha], &mut packed);
            assert_eq!(packed, [0, high]);
        }
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
