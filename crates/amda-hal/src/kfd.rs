//! Linux `/dev/kfd` ioctl ABI definitions, `/sys/class/kfd` topology parser,
//! GPUVM virtual address management, hardware signal layout, and native Linux KFD backend.

use crate::device::{DeviceInfo, HardwareFeatures};
use crate::error::HalError;
use amda_elf::IsaTarget;
use core::mem::size_of;
use core::sync::atomic::AtomicI64;
use std::collections::BTreeMap;

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;

const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;

const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> u32 {
    (dir << IOC_DIRSHIFT)
        | ((ty as u32) << IOC_TYPESHIFT)
        | ((nr as u32) << IOC_NRSHIFT)
        | ((size as u32) << IOC_SIZESHIFT)
}

const fn iow<T>(ty: u8, nr: u8) -> u32 {
    ioc(IOC_WRITE, ty, nr, size_of::<T>())
}

const fn ior<T>(ty: u8, nr: u8) -> u32 {
    ioc(IOC_READ, ty, nr, size_of::<T>())
}

const fn iowr<T>(ty: u8, nr: u8) -> u32 {
    ioc(IOC_READ | IOC_WRITE, ty, nr, size_of::<T>())
}

pub const KFD_IOCTL_BASE: u8 = b'K';

// KFD memory allocation flags (`uapi/linux/kfd_ioctl.h`)
pub const KFD_IOC_ALLOC_MEM_FLAGS_VRAM: u32 = 1 << 0;
pub const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
pub const KFD_IOC_ALLOC_MEM_FLAGS_USERPTR: u32 = 1 << 2;
pub const KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL: u32 = 1 << 3;
pub const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;
pub const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
pub const KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC: u32 = 1 << 29;
pub const KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE: u32 = 1 << 28;
pub const KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;
pub const KFD_IOC_ALLOC_MEM_FLAGS_COHERENT: u32 = 1 << 26;
pub const KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED: u32 = 1 << 25;

// KFD queue types
pub const KFD_IOC_QUEUE_TYPE_COMPUTE: u32 = 0;
pub const KFD_IOC_QUEUE_TYPE_SDMA: u32 = 1;
pub const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 2;

/// `AMD_SIGNAL_KIND_USER` (HSA user-mode coherent signal).
pub const AMD_SIGNAL_KIND_USER: i64 = 1;

/// Spec-accurate 64-byte HSA signal structure read/decremented by the GPU Command Processor.
#[repr(C, align(64))]
pub struct HsaSignalStruct {
    pub kind: i64,
    pub value: AtomicI64,
    pub event_mailbox_ptr: u64,
    pub event_id: u32,
    pub reserved1: u32,
    pub start_ts: u64,
    pub end_ts: u64,
    pub queue_ptr: u64,
    pub reserved3: [u32; 2],
}

const _: () = assert!(size_of::<HsaSignalStruct>() == 64);
const _: () = assert!(core::mem::align_of::<HsaSignalStruct>() == 64);

/// `kfd_ioctl_get_version_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlGetVersionArgs {
    pub major_version: u32,
    pub minor_version: u32,
}

/// `kfd_ioctl_create_queue_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlCreateQueueArgs {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,
    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,
    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
}

/// `kfd_ioctl_destroy_queue_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlDestroyQueueArgs {
    pub queue_id: u32,
    pub pad: u32,
}

/// `kfd_ioctl_create_event_args` (32 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlCreateEventArgs {
    pub event_page_offset: u64,
    pub event_slot_index: u32,
    pub event_type: u32,
    pub auto_reset: u32,
    pub node_id: u32,
    pub event_id: u32,
    pub event_trigger_data: u32,
}

/// `kfd_ioctl_destroy_event_args` (8 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlDestroyEventArgs {
    pub event_id: u32,
    pub pad: u32,
}

/// `kfd_ioctl_set_event_args` (8 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlSetEventArgs {
    pub event_id: u32,
    pub pad: u32,
}

/// `kfd_event_data` (24 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdEventData {
    pub data: [u64; 2],
    pub kfd_event_data_ext: u64,
}

/// `kfd_ioctl_wait_events_args` (24 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlWaitEventsArgs {
    pub events_ptr: u64,
    pub num_events: u32,
    pub wait_for_all: u32,
    pub timeout: u32,
    pub wait_result: u32,
}

/// `kfd_process_device_apertures` (56 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdProcessDeviceApertures {
    pub lds_base: u64,
    pub lds_limit: u64,
    pub scratch_base: u64,
    pub scratch_limit: u64,
    pub gpuvm_base: u64,
    pub gpuvm_limit: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

/// `kfd_ioctl_get_process_apertures_new_args` (16 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlGetProcessAperturesNewArgs {
    pub kfd_process_device_apertures_ptr: u64,
    pub num_of_nodes: u32,
    pub pad: u32,
}

/// `kfd_ioctl_acquire_vm_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlAcquireVmArgs {
    pub drm_fd: u32,
    pub gpu_id: u32,
}

/// `kfd_ioctl_alloc_memory_of_gpu_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlAllocMemoryOfGpuArgs {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

/// `kfd_ioctl_free_memory_of_gpu_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlFreeMemoryOfGpuArgs {
    pub handle: u64,
}

/// `kfd_ioctl_map_memory_to_gpu_args`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlMapMemoryToGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

/// `kfd_ioctl_unmap_memory_from_gpu_args` (24 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct KfdIoctlUnmapMemoryFromGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

const _: () = assert!(size_of::<KfdIoctlCreateEventArgs>() == 32);
const _: () = assert!(size_of::<KfdIoctlDestroyEventArgs>() == 8);
const _: () = assert!(size_of::<KfdIoctlSetEventArgs>() == 8);
const _: () = assert!(size_of::<KfdEventData>() == 24);
const _: () = assert!(size_of::<KfdIoctlWaitEventsArgs>() == 24);
const _: () = assert!(size_of::<KfdProcessDeviceApertures>() == 56);
const _: () = assert!(size_of::<KfdIoctlGetProcessAperturesNewArgs>() == 16);
const _: () = assert!(size_of::<KfdIoctlUnmapMemoryFromGpuArgs>() == 24);

pub const AMDKFD_IOC_GET_VERSION: u32 = ior::<KfdIoctlGetVersionArgs>(KFD_IOCTL_BASE, 0x01);
pub const AMDKFD_IOC_CREATE_QUEUE: u32 = iowr::<KfdIoctlCreateQueueArgs>(KFD_IOCTL_BASE, 0x02);
pub const AMDKFD_IOC_DESTROY_QUEUE: u32 = iowr::<KfdIoctlDestroyQueueArgs>(KFD_IOCTL_BASE, 0x03);
pub const AMDKFD_IOC_CREATE_EVENT: u32 = iowr::<KfdIoctlCreateEventArgs>(KFD_IOCTL_BASE, 0x08);
pub const AMDKFD_IOC_DESTROY_EVENT: u32 = iow::<KfdIoctlDestroyEventArgs>(KFD_IOCTL_BASE, 0x09);
pub const AMDKFD_IOC_SET_EVENT: u32 = iow::<KfdIoctlSetEventArgs>(KFD_IOCTL_BASE, 0x0A);
pub const AMDKFD_IOC_WAIT_EVENTS: u32 = iowr::<KfdIoctlWaitEventsArgs>(KFD_IOCTL_BASE, 0x0C);
pub const AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: u32 =
    iowr::<KfdIoctlGetProcessAperturesNewArgs>(KFD_IOCTL_BASE, 0x14);
pub const AMDKFD_IOC_ACQUIRE_VM: u32 = iow::<KfdIoctlAcquireVmArgs>(KFD_IOCTL_BASE, 0x15);
pub const AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: u32 =
    iowr::<KfdIoctlAllocMemoryOfGpuArgs>(KFD_IOCTL_BASE, 0x16);
pub const AMDKFD_IOC_FREE_MEMORY_OF_GPU: u32 =
    iow::<KfdIoctlFreeMemoryOfGpuArgs>(KFD_IOCTL_BASE, 0x17);
pub const AMDKFD_IOC_MAP_MEMORY_TO_GPU: u32 =
    iowr::<KfdIoctlMapMemoryToGpuArgs>(KFD_IOCTL_BASE, 0x18);
pub const AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: u32 =
    iowr::<KfdIoctlUnmapMemoryFromGpuArgs>(KFD_IOCTL_BASE, 0x19);

// Keep IOC_NONE referenced without dead-code warnings across targets.
const _: u32 = ioc(IOC_NONE, KFD_IOCTL_BASE, 0, 0);

/// Selects `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` flags for APU vs discrete GPU and coherent/executable uses.
pub const fn select_kfd_alloc_flags(is_apu: bool, executable: bool, host_visible: bool) -> u32 {
    let mut flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
    if is_apu || host_visible {
        flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    } else {
        flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
    }
    if executable {
        flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
    }
    flags
}

/// Patches a 64-byte `AmdhsaKernelDescriptor` so `kernel_code_entry_byte_offset` (`[16..24]`)
/// points from `kd_gpu_va` directly to `text_entry_gpu_va` in GPU virtual memory.
pub fn patch_kernel_descriptor_entry_offset(
    kd_bytes: &mut [u8; 64],
    kd_gpu_va: u64,
    text_entry_gpu_va: u64,
) -> Result<(), HalError> {
    let diff = i128::from(text_entry_gpu_va) - i128::from(kd_gpu_va);
    let off = i64::try_from(diff).map_err(|_| {
        HalError::InvalidCodeObject("kernel code entry offset overflows i64".to_owned())
    })?;
    kd_bytes[16..24].copy_from_slice(&off.to_le_bytes());
    Ok(())
}

/// Page-aligned GPUVM virtual address allocator operating inside a KFD process aperture `[base, limit)`.
#[derive(Debug, Clone)]
pub struct KfdGpuVaAllocator {
    base: u64,
    limit: u64,
    cursor: u64,
    free_blocks: BTreeMap<u64, u64>,
}

impl KfdGpuVaAllocator {
    /// Creates an allocator over `[base, limit)` aligned to 4096-byte pages.
    pub fn new(base: u64, limit: u64) -> Self {
        let aligned_base = (base.max(0x1_0000)).div_ceil(4096) * 4096;
        let aligned_limit = (limit / 4096) * 4096;
        Self {
            base: aligned_base,
            limit: aligned_limit.max(aligned_base),
            cursor: aligned_base,
            free_blocks: BTreeMap::new(),
        }
    }

    /// Allocates at least `size` bytes (rounded up to 4096B pages) aligned to `align.max(4096)`.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<u64, HalError> {
        let alloc_size = u64::try_from(size.max(1).div_ceil(4096) * 4096)
            .map_err(|_| HalError::OutOfMemory("GPUVM allocation size overflow".to_owned()))?;
        let req_align = u64::try_from(align.max(4096).next_power_of_two())
            .map_err(|_| HalError::OutOfMemory("GPUVM alignment overflow".to_owned()))?;

        // Check reusable freed blocks first.
        let mut chosen: Option<(u64, u64, u64)> = None;
        for (&blk_start, &blk_len) in &self.free_blocks {
            let aligned_va = blk_start.div_ceil(req_align) * req_align;
            let prefix = aligned_va - blk_start;
            if blk_len >= prefix.saturating_add(alloc_size) {
                chosen = Some((blk_start, blk_len, aligned_va));
                break;
            }
        }
        if let Some((blk_start, blk_len, aligned_va)) = chosen {
            self.free_blocks.remove(&blk_start);
            let prefix = aligned_va - blk_start;
            if prefix > 0 {
                self.free_blocks.insert(blk_start, prefix);
            }
            let suffix_start = aligned_va + alloc_size;
            let suffix_len = (blk_start + blk_len) - suffix_start;
            if suffix_len > 0 {
                self.free_blocks.insert(suffix_start, suffix_len);
            }
            return Ok(aligned_va);
        }

        // Bump from cursor.
        let aligned_cursor = self.cursor.div_ceil(req_align) * req_align;
        let end = aligned_cursor
            .checked_add(alloc_size)
            .ok_or_else(|| HalError::OutOfMemory("GPUVM aperture address overflow".to_owned()))?;
        if end > self.limit || aligned_cursor < self.base {
            return Err(HalError::OutOfMemory(format!(
                "GPUVM aperture exhausted (requested {alloc_size}B in [0x{:x}..0x{:x}))",
                self.base, self.limit
            )));
        }
        if aligned_cursor > self.cursor {
            self.free_blocks
                .insert(self.cursor, aligned_cursor - self.cursor);
        }
        self.cursor = end;
        Ok(aligned_cursor)
    }

    /// Returns a previously allocated GPUVM block to the free list and coalesces neighbors.
    pub fn free(&mut self, va: u64, size: usize) {
        let free_size = (size.max(1).div_ceil(4096) * 4096) as u64;
        let mut start = va;
        let mut len = free_size;

        // Merge with preceding neighbor if adjacent.
        if let Some((&prev_start, &prev_len)) = self.free_blocks.range(..start).next_back() {
            if prev_start + prev_len == start {
                self.free_blocks.remove(&prev_start);
                start = prev_start;
                len += prev_len;
            }
        }
        // Merge with following neighbor if adjacent.
        let next_key = start + len;
        if let Some(next_len) = self.free_blocks.remove(&next_key) {
            len += next_len;
        }
        self.free_blocks.insert(start, len);
    }
}

/// Parsed GPU properties from `/sys/class/kfd/kfd/topology/nodes/<n>/properties`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KfdNodeProps {
    pub gpu_id: u32,
    pub simd_count: u32,
    pub simd_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub lds_size_in_kb: u32,
    pub gfx_target_version: u32,
    pub capability: u32,
    pub drm_render_minor: u32,
    pub num_xcc: u32,
    pub vram_bytes: u64,
    pub name: String,
}

/// Decodes Linux KFD `gfx_target_version` (`major * 10000 + minor * 100 + stepping`) into [`IsaTarget`].
pub fn decode_gfx_target_version(ver: u32) -> Result<IsaTarget, HalError> {
    match ver {
        90008 => Ok(IsaTarget::Gfx908),
        90010 => Ok(IsaTarget::Gfx90a),
        90402 => Ok(IsaTarget::Gfx942),
        100300 => Ok(IsaTarget::Gfx1030),
        110000 => Ok(IsaTarget::Gfx1100),
        110001 => Ok(IsaTarget::Gfx1101),
        110002 => Ok(IsaTarget::Gfx1102),
        110003 => Ok(IsaTarget::Gfx1103),
        120000 => Ok(IsaTarget::Gfx1200),
        120100 => Ok(IsaTarget::Gfx1201),
        other => Err(HalError::KfdError(format!(
            "unrecognized KFD gfx_target_version {other}"
        ))),
    }
}

/// Parses a KFD sysfs `properties` text block. Returns `None` for CPU-only topology nodes.
pub fn parse_kfd_topology_properties(text: &str) -> Option<KfdNodeProps> {
    let mut gpu_id = 0u32;
    let mut simd_count = 0u32;
    let mut simd_per_cu = 2u32;
    let mut max_waves_per_simd = 16u32;
    let mut lds_size_in_kb = 64u32;
    let mut gfx_target_version = 0u32;
    let mut capability = 0u32;
    let mut drm_render_minor = 0u32;
    let mut num_xcc = 1u32;
    let mut vram_bytes = 0u64;
    let mut marketing_name = None::<String>;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let key = parts.next()?.trim();
        let val_str = parts.next().unwrap_or("").trim();

        match key {
            "gpu_id" => gpu_id = val_str.parse().unwrap_or(0),
            "simd_count" => simd_count = val_str.parse().unwrap_or(0),
            "simd_per_cu" => simd_per_cu = val_str.parse().unwrap_or(2).max(1),
            "max_waves_per_simd" => max_waves_per_simd = val_str.parse().unwrap_or(16),
            "lds_size_in_kb" => lds_size_in_kb = val_str.parse().unwrap_or(64),
            "gfx_target_version" => gfx_target_version = val_str.parse().unwrap_or(0),
            "capability" => capability = val_str.parse().unwrap_or(0),
            "drm_render_minor" => drm_render_minor = val_str.parse().unwrap_or(0),
            "num_xcc" => num_xcc = val_str.parse().unwrap_or(1).max(1),
            "local_mem_size" => vram_bytes = val_str.parse().unwrap_or(0),
            "marketing_name" if !val_str.is_empty() => {
                marketing_name = Some(val_str.to_owned());
            }
            _ => {}
        }
    }

    if simd_count == 0 || gfx_target_version == 0 {
        return None;
    }

    let isa_name = decode_gfx_target_version(gfx_target_version)
        .map(|i| i.as_str().to_owned())
        .unwrap_or_else(|_| format!("gfx{gfx_target_version}"));

    Some(KfdNodeProps {
        gpu_id,
        simd_count,
        simd_per_cu,
        max_waves_per_simd,
        lds_size_in_kb,
        gfx_target_version,
        capability,
        drm_render_minor,
        num_xcc,
        vram_bytes,
        name: marketing_name.unwrap_or_else(|| format!("AMD GPU ({isa_name})")),
    })
}

/// Parses a KFD `mem_banks/<n>/properties` file and returns device-local VRAM bytes
/// (`heap_type` 1 = `FB_PUBLIC` or 2 = `FB_PRIVATE`), returning `0` for system/LDS/scratch banks.
pub fn parse_kfd_mem_bank_properties(text: &str) -> u64 {
    let mut heap_type: Option<u32> = None;
    let mut size_in_bytes: u64 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let Some(key) = parts.next().map(str::trim) else {
            continue;
        };
        let val_str = parts.next().unwrap_or("").trim();
        match key {
            "heap_type" => heap_type = val_str.parse().ok(),
            "size_in_bytes" => size_in_bytes = val_str.parse().unwrap_or(0),
            _ => {}
        }
    }

    match heap_type {
        Some(1 | 2) => size_in_bytes,
        _ => 0,
    }
}

/// Hardware queue auxiliary buffer sizes required by KFD/MEC across `num_xcc` XCCs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KfdQueueBufferSizes {
    pub eop_bytes: usize,
    pub ctl_stack_bytes: usize,
    pub ctx_save_restore_bytes: usize,
}

/// Computes multi-XCC page-aligned EOP, control-stack, and CWSR context save/restore sizes.
pub fn compute_kfd_queue_buffer_sizes(
    cu_count: u32,
    max_waves_per_cu: u32,
    num_xcc: u32,
) -> KfdQueueBufferSizes {
    const KFD_PAGE: usize = 4096;
    let xccs = (num_xcc as usize).max(1);
    let page_per_xcc = KFD_PAGE * xccs;
    let align_up_xcc = |n: usize| n.max(page_per_xcc).div_ceil(page_per_xcc) * page_per_xcc;

    let eop_bytes = align_up_xcc(KFD_PAGE * xccs);
    let wave_slots = (cu_count as usize).max(1) * (max_waves_per_cu as usize).max(1);
    let ctl_stack_bytes = align_up_xcc(wave_slots * 16 + KFD_PAGE * xccs);
    // Reserve ~1152 KiB per CU (covers CDNA3 VGPR+AGPR+SGPR+64KiB LDS save state) plus control stack.
    let cwsr_save_bytes = (cu_count as usize).max(1) * 1152 * 1024;
    let ctx_save_restore_bytes = align_up_xcc(ctl_stack_bytes + cwsr_save_bytes);

    KfdQueueBufferSizes {
        eop_bytes,
        ctl_stack_bytes,
        ctx_save_restore_bytes,
    }
}

/// Resolves a kernel descriptor's `kernel_code_entry_byte_offset` into a `.text`-relative byte
/// offset for both raw `ld.lld` HSACOs (where `.rodata` precedes `.text` and offset is VA delta)
/// and repacked AMDA HSACOs (where `kernel_code_entry_byte_offset` is already `.text`-relative).
pub fn resolve_kernel_text_entry_offset(
    rodata_sh_addr: u64,
    text_sh_addr: u64,
    text_len: u64,
    kd_off: usize,
    orig_entry_off: i64,
) -> u64 {
    if text_sh_addr > 0 && rodata_sh_addr > 0 {
        let cand =
            i128::from(rodata_sh_addr) + i128::from(kd_off as u64) + i128::from(orig_entry_off);
        let t_start = i128::from(text_sh_addr);
        let t_end = t_start + i128::from(text_len);
        if cand >= t_start && cand < t_end {
            return (cand - t_start) as u64;
        }
    }
    let rel = orig_entry_off.max(0) as u64;
    if text_len > 0 && rel >= text_len {
        0
    } else {
        rel
    }
}

/// Converts parsed KFD node properties into a capability-driven [`DeviceInfo`].
pub fn device_info_from_kfd(index: usize, props: &KfdNodeProps) -> Result<DeviceInfo, HalError> {
    let isa = decode_gfx_target_version(props.gfx_target_version)?;
    let cus = (props.simd_count / props.simd_per_cu.max(1)).max(1);
    let unified_memory = (props.capability & 0x200) != 0 || isa == IsaTarget::Gfx1103;
    let mut features = HardwareFeatures::from_isa(isa, cus, unified_memory);
    features.lds_bytes_per_cu = props.lds_size_in_kb.saturating_mul(1024);
    features.max_waves_per_cu = props.max_waves_per_simd.saturating_mul(props.simd_per_cu);

    Ok(DeviceInfo {
        index,
        name: props.name.clone(),
        isa,
        vram_bytes: if props.vram_bytes > 0 {
            props.vram_bytes
        } else {
            8 * 1024 * 1024 * 1024
        },
        features,
        backend_name: "kfd",
    })
}

#[cfg(target_os = "linux")]
pub mod linux {
    use super::*;
    use crate::backend::{CodeObjectId, DeviceBackend, KernelHandle, QueueId, SignalId};
    use amda_elf::{parse_hsaco, KernelMetadata};
    use core::sync::atomic::{fence, AtomicU16, AtomicU64, Ordering};
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::fs;
    use std::os::raw::{c_char, c_int, c_long, c_ulong, c_void};
    use std::path::Path;
    use std::sync::{Mutex, RwLock};
    use std::time::{Duration, Instant};

    const O_RDWR: c_int = 2;
    const O_CLOEXEC: c_int = 0o2000000;
    const PROT_NONE: c_int = 0;
    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_SHARED: c_int = 1;
    const MAP_PRIVATE: c_int = 2;
    const MAP_FIXED: c_int = 0x10;
    const MAP_ANONYMOUS: c_int = 0x20;
    const MAP_FAILED: usize = usize::MAX;
    const PAGE_SIZE: usize = 4096;

    extern "C" {
        fn open(pathname: *const c_char, flags: c_int) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: c_long,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, length: usize) -> c_int;
    }

    #[derive(Clone)]
    struct KfdDeviceRuntime {
        gpu_id: u32,
        drm_fd: c_int,
        is_apu: bool,
        num_xcc: u32,
        cu_count: u32,
        max_waves_per_cu: u32,
    }

    #[derive(Clone, Copy)]
    struct KfdAllocation {
        device: usize,
        gpu_va: u64,
        map_len: usize,
        va_reserve_len: usize,
        kfd_handle: u64,
        host_addr: usize,
        has_gpuvm_reservation: bool,
    }

    struct KfdQueue {
        gpu_va_ring: u64,
        gpu_va_ptrs: u64,
        gpu_va_eop: u64,
        gpu_va_ctx: u64,
        kfd_queue_id: u32,
        ring_cap: usize,
        ring_host: usize,
        wptr_host: usize,
        doorbell_page_host: usize,
        doorbell_offset_in_page: usize,
        write_idx: Mutex<u64>,
    }

    struct KfdSignal {
        gpu_va: u64,
        host_addr: usize,
    }

    struct KfdCodeObject {
        gpu_va: u64,
        kernel_ids: Vec<u64>,
    }

    struct KfdKernelEntry {
        kd_gpu_va: u64,
        metadata: KernelMetadata,
    }

    /// Direct Linux `/dev/kfd` + DRM + user-space AQL queue + doorbell backend.
    pub struct KfdBackend {
        kfd_fd: c_int,
        devices: Vec<DeviceInfo>,
        dev_rt: Vec<KfdDeviceRuntime>,
        va_allocs: Vec<Mutex<KfdGpuVaAllocator>>,
        allocations: RwLock<BTreeMap<u64, KfdAllocation>>,
        queues: RwLock<HashMap<u64, KfdQueue>>,
        signals: RwLock<HashMap<u64, KfdSignal>>,
        code_objects: RwLock<HashMap<u64, KfdCodeObject>>,
        kernels: RwLock<HashMap<u64, KfdKernelEntry>>,
        next_id: AtomicU64,
        watchdog_ms: u64,
    }

    impl KfdBackend {
        /// Opens `/dev/kfd`, probes GPU topology, acquires per-device DRM VMs, and fetches apertures.
        pub fn open_system() -> Result<Self, HalError> {
            let path = CString::new("/dev/kfd").expect("static path");
            let fd = unsafe { open(path.as_ptr(), O_RDWR | O_CLOEXEC) };
            if fd < 0 {
                let err = std::io::Error::last_os_error();
                return Err(HalError::KfdError(format!(
                    "failed to open /dev/kfd ({err})"
                )));
            }

            let mut ver = KfdIoctlGetVersionArgs::default();
            let rc = unsafe {
                ioctl(
                    fd,
                    c_ulong::from(AMDKFD_IOC_GET_VERSION),
                    &mut ver as *mut KfdIoctlGetVersionArgs as *mut c_void,
                )
            };
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                unsafe { close(fd) };
                return Err(HalError::KfdError(format!(
                    "AMDKFD_IOC_GET_VERSION ioctl failed ({err})"
                )));
            }

            let mut node_props = Vec::new();
            let mut devices = Vec::new();
            let topo_root = Path::new("/sys/class/kfd/kfd/topology/nodes");
            if let Ok(entries) = fs::read_dir(topo_root) {
                let mut sorted_paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
                sorted_paths.sort();
                for node_path in sorted_paths {
                    let prop_path = node_path.join("properties");
                    if let Ok(text) = fs::read_to_string(&prop_path) {
                        if let Some(mut props) = parse_kfd_topology_properties(&text) {
                            if props.vram_bytes == 0 {
                                let mut banked_vram = 0u64;
                                if let Ok(banks) = fs::read_dir(node_path.join("mem_banks")) {
                                    for bank in banks.flatten() {
                                        let bprop = bank.path().join("properties");
                                        if let Ok(btext) = fs::read_to_string(&bprop) {
                                            banked_vram = banked_vram.saturating_add(
                                                parse_kfd_mem_bank_properties(&btext),
                                            );
                                        }
                                    }
                                }
                                if banked_vram > 0 {
                                    props.vram_bytes = banked_vram;
                                }
                            }
                            if let Ok(info) = device_info_from_kfd(devices.len(), &props) {
                                devices.push(info);
                                node_props.push(props);
                            }
                        }
                    }
                }
            }

            if devices.is_empty() {
                unsafe { close(fd) };
                return Err(HalError::KfdError(
                    "no AMD GPU nodes found in /sys/class/kfd/kfd/topology/nodes".to_owned(),
                ));
            }

            // Open /dev/dri/renderD<N> per GPU and bind KFD VM context.
            let mut dev_rt: Vec<KfdDeviceRuntime> = Vec::with_capacity(node_props.len());
            for (idx, props) in node_props.iter().enumerate() {
                let render_minor = if props.drm_render_minor >= 128 {
                    props.drm_render_minor
                } else {
                    128 + idx as u32
                };
                let render_path = CString::new(format!("/dev/dri/renderD{render_minor}"))
                    .map_err(|_| HalError::KfdError("invalid renderD path".to_owned()))?;
                let drm_fd = unsafe { open(render_path.as_ptr(), O_RDWR | O_CLOEXEC) };
                if drm_fd < 0 {
                    let err = std::io::Error::last_os_error();
                    for rt in &dev_rt {
                        unsafe { close(rt.drm_fd) };
                    }
                    unsafe { close(fd) };
                    return Err(HalError::KfdError(format!(
                        "failed to open /dev/dri/renderD{render_minor} ({err})"
                    )));
                }

                let acq = KfdIoctlAcquireVmArgs {
                    drm_fd: drm_fd as u32,
                    gpu_id: props.gpu_id,
                };
                let acq_rc = unsafe {
                    ioctl(
                        fd,
                        c_ulong::from(AMDKFD_IOC_ACQUIRE_VM),
                        &acq as *const KfdIoctlAcquireVmArgs as *mut c_void,
                    )
                };
                if acq_rc < 0 {
                    let err = std::io::Error::last_os_error();
                    unsafe { close(drm_fd) };
                    for rt in &dev_rt {
                        unsafe { close(rt.drm_fd) };
                    }
                    unsafe { close(fd) };
                    return Err(HalError::KfdError(format!(
                        "AMDKFD_IOC_ACQUIRE_VM failed for gpu_id {} ({err})",
                        props.gpu_id
                    )));
                }

                dev_rt.push(KfdDeviceRuntime {
                    gpu_id: props.gpu_id,
                    drm_fd,
                    is_apu: devices[idx].features.unified_memory,
                    num_xcc: props.num_xcc.max(1),
                    cu_count: devices[idx].features.compute_units,
                    max_waves_per_cu: devices[idx].features.max_waves_per_cu,
                });
            }

            // Query process GPUVM apertures for each GPU node.
            let mut apertures = vec![KfdProcessDeviceApertures::default(); dev_rt.len().max(8)];
            let mut ap_args = KfdIoctlGetProcessAperturesNewArgs {
                kfd_process_device_apertures_ptr: apertures.as_mut_ptr() as u64,
                num_of_nodes: apertures.len() as u32,
                pad: 0,
            };
            let ap_rc = unsafe {
                ioctl(
                    fd,
                    c_ulong::from(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW),
                    &mut ap_args as *mut KfdIoctlGetProcessAperturesNewArgs as *mut c_void,
                )
            };

            let mut va_allocs = Vec::with_capacity(dev_rt.len());
            for rt in &dev_rt {
                let mut found = None;
                if ap_rc >= 0 {
                    let count = (ap_args.num_of_nodes as usize).min(apertures.len());
                    for ap in &apertures[..count] {
                        if ap.gpu_id == rt.gpu_id && ap.gpuvm_limit > ap.gpuvm_base {
                            found = Some((ap.gpuvm_base, ap.gpuvm_limit));
                            break;
                        }
                    }
                }
                let (base, limit) = found.unwrap_or((0x0000_0010_0000_0000, 0x0000_0F00_0000_0000));
                va_allocs.push(Mutex::new(KfdGpuVaAllocator::new(base, limit)));
            }

            let watchdog_ms = std::env::var("AMDA_WATCHDOG_MS")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(5000);

            Ok(Self {
                kfd_fd: fd,
                devices,
                dev_rt,
                va_allocs,
                allocations: RwLock::new(BTreeMap::new()),
                queues: RwLock::new(HashMap::new()),
                signals: RwLock::new(HashMap::new()),
                code_objects: RwLock::new(HashMap::new()),
                kernels: RwLock::new(HashMap::new()),
                next_id: AtomicU64::new(1),
                watchdog_ms,
            })
        }

        fn next_handle(&self) -> u64 {
            self.next_id.fetch_add(1, Ordering::Relaxed)
        }

        fn check_device(&self, device: usize) -> Result<&KfdDeviceRuntime, HalError> {
            self.dev_rt
                .get(device)
                .ok_or(HalError::DeviceNotFound(device))
        }

        fn alloc_kfd_buffer(
            &self,
            device: usize,
            size: usize,
            align: usize,
            executable: bool,
            host_visible: bool,
        ) -> Result<KfdAllocation, HalError> {
            self.alloc_kfd_buffer_with_flags(
                device,
                size,
                align,
                executable,
                host_visible,
                0,
                1,
            )
        }

        fn alloc_kfd_buffer_with_flags(
            &self,
            device: usize,
            size: usize,
            align: usize,
            executable: bool,
            host_visible: bool,
            extra_flags: u32,
            va_reserve_multiplier: usize,
        ) -> Result<KfdAllocation, HalError> {
            let rt = self.check_device(device)?;
            let map_len = size.max(1).div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let va_reserve_len = map_len
                .checked_mul(va_reserve_multiplier.max(1))
                .ok_or_else(|| HalError::OutOfMemory("KFD VA reservation size overflow".to_owned()))?;
            let gpu_va = self.va_allocs[device]
                .lock()
                .expect("va_alloc lock")
                .alloc(va_reserve_len, align)?;

            // Reserve virtual address space with PROT_NONE before binding KFD backing store.
            let reserved = unsafe {
                mmap(
                    gpu_va as *mut c_void,
                    va_reserve_len,
                    PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                    -1,
                    0,
                )
            };
            if reserved as usize == MAP_FAILED {
                let err = std::io::Error::last_os_error();
                self.va_allocs[device]
                    .lock()
                    .expect("va_alloc lock")
                    .free(gpu_va, va_reserve_len);
                return Err(HalError::OutOfMemory(format!(
                    "mmap(PROT_NONE) reservation failed for GPUVM ({err})"
                )));
            }

            let flags = select_kfd_alloc_flags(rt.is_apu, executable, host_visible) | extra_flags;
            let mut alloc_args = KfdIoctlAllocMemoryOfGpuArgs {
                va_addr: gpu_va,
                size: map_len as u64,
                handle: 0,
                mmap_offset: 0,
                gpu_id: rt.gpu_id,
                flags,
            };
            let rc = unsafe {
                ioctl(
                    self.kfd_fd,
                    c_ulong::from(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU),
                    &mut alloc_args as *mut KfdIoctlAllocMemoryOfGpuArgs as *mut c_void,
                )
            };
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                unsafe { munmap(gpu_va as *mut c_void, va_reserve_len) };
                self.va_allocs[device]
                    .lock()
                    .expect("va_alloc lock")
                    .free(gpu_va, va_reserve_len);
                return Err(HalError::KfdError(format!(
                    "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU failed (flags=0x{flags:08x}: {err})"
                )));
            }

            let mut dev_id = rt.gpu_id;
            let mut map_args = KfdIoctlMapMemoryToGpuArgs {
                handle: alloc_args.handle,
                device_ids_array_ptr: (&mut dev_id as *mut u32) as u64,
                n_devices: 1,
                n_success: 0,
            };
            let map_rc = unsafe {
                ioctl(
                    self.kfd_fd,
                    c_ulong::from(AMDKFD_IOC_MAP_MEMORY_TO_GPU),
                    &mut map_args as *mut KfdIoctlMapMemoryToGpuArgs as *mut c_void,
                )
            };
            if map_rc < 0 {
                let err = std::io::Error::last_os_error();
                let free_args = KfdIoctlFreeMemoryOfGpuArgs {
                    handle: alloc_args.handle,
                };
                unsafe {
                    ioctl(
                        self.kfd_fd,
                        c_ulong::from(AMDKFD_IOC_FREE_MEMORY_OF_GPU),
                        &free_args as *const KfdIoctlFreeMemoryOfGpuArgs as *mut c_void,
                    );
                    munmap(gpu_va as *mut c_void, va_reserve_len);
                }
                self.va_allocs[device]
                    .lock()
                    .expect("va_alloc lock")
                    .free(gpu_va, va_reserve_len);
                return Err(HalError::KfdError(format!(
                    "AMDKFD_IOC_MAP_MEMORY_TO_GPU failed ({err})"
                )));
            }

            // Map host pointer over fixed GPUVM reservation for SVM / direct copies.
            let map_fd = if (flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) != 0 {
                rt.drm_fd
            } else {
                self.kfd_fd
            };
            let mut host_ptr = unsafe {
                mmap(
                    gpu_va as *mut c_void,
                    map_len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    map_fd,
                    alloc_args.mmap_offset as c_long,
                )
            };
            if host_ptr as usize == MAP_FAILED {
                host_ptr = unsafe {
                    mmap(
                        gpu_va as *mut c_void,
                        map_len,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED,
                        self.kfd_fd,
                        alloc_args.mmap_offset as c_long,
                    )
                };
            }
            let host_addr = if host_ptr as usize == MAP_FAILED {
                0
            } else {
                host_ptr as usize
            };

            let rec = KfdAllocation {
                device,
                gpu_va,
                map_len,
                va_reserve_len,
                kfd_handle: alloc_args.handle,
                host_addr,
                has_gpuvm_reservation: true,
            };
            self.allocations
                .write()
                .expect("allocations lock")
                .insert(gpu_va, rec);
            Ok(rec)
        }

        fn free_kfd_buffer(&self, gpu_va: u64) -> Result<(), HalError> {
            let rec = self
                .allocations
                .write()
                .expect("allocations lock")
                .remove(&gpu_va)
                .ok_or(HalError::InvalidHandle("KFD allocation"))?;

            if let Some(rt) = self.dev_rt.get(rec.device) {
                let mut dev_id = rt.gpu_id;
                let mut unmap_args = KfdIoctlUnmapMemoryFromGpuArgs {
                    handle: rec.kfd_handle,
                    device_ids_array_ptr: (&mut dev_id as *mut u32) as u64,
                    n_devices: 1,
                    n_success: 0,
                };
                unsafe {
                    ioctl(
                        self.kfd_fd,
                        c_ulong::from(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU),
                        &mut unmap_args as *mut KfdIoctlUnmapMemoryFromGpuArgs as *mut c_void,
                    );
                }
            }
            let free_args = KfdIoctlFreeMemoryOfGpuArgs {
                handle: rec.kfd_handle,
            };
            unsafe {
                ioctl(
                    self.kfd_fd,
                    c_ulong::from(AMDKFD_IOC_FREE_MEMORY_OF_GPU),
                    &free_args as *const KfdIoctlFreeMemoryOfGpuArgs as *mut c_void,
                );
                if rec.has_gpuvm_reservation {
                    munmap(rec.gpu_va as *mut c_void, rec.va_reserve_len);
                }
            }
            if let Some(va_alloc) = self.va_allocs.get(rec.device) {
                va_alloc
                    .lock()
                    .expect("va_alloc lock")
                    .free(rec.gpu_va, rec.va_reserve_len);
            }
            Ok(())
        }

        fn host_slice_mut<'a>(&self, addr: u64, len: usize) -> Result<(*mut u8, usize), HalError> {
            let allocs = self.allocations.read().expect("allocations read lock");
            let (&base, rec) = allocs
                .range(..=addr)
                .next_back()
                .ok_or_else(|| HalError::MemoryAccessFault {
                    addr,
                    size: len,
                    reason: "unmapped KFD GPU address".to_owned(),
                })?;
            let off = (addr - base) as usize;
            let end = off.checked_add(len).ok_or_else(|| HalError::MemoryAccessFault {
                addr,
                size: len,
                reason: "KFD copy offset overflow".to_owned(),
            })?;
            if end > rec.map_len {
                return Err(HalError::MemoryAccessFault {
                    addr,
                    size: len,
                    reason: "KFD access exceeds allocation bounds".to_owned(),
                });
            }
            if rec.host_addr == 0 {
                return Err(HalError::KfdError(
                    "GPU allocation lacks host mmap access".to_owned(),
                ));
            }
            Ok(((rec.host_addr + off) as *mut u8, off))
        }
    }

    impl Drop for KfdBackend {
        fn drop(&mut self) {
            let queue_ids: Vec<u64> = self
                .queues
                .read()
                .expect("queues lock")
                .keys()
                .copied()
                .collect();
            for qid in queue_ids {
                let _ = self.destroy_queue(QueueId(qid));
            }

            let sig_ids: Vec<u64> = self
                .signals
                .read()
                .expect("signals lock")
                .keys()
                .copied()
                .collect();
            for sid in sig_ids {
                let _ = self.destroy_signal(SignalId(sid));
            }

            let obj_ids: Vec<u64> = self
                .code_objects
                .read()
                .expect("code_objects lock")
                .keys()
                .copied()
                .collect();
            for oid in obj_ids {
                let _ = self.unload_code_object(CodeObjectId(oid));
            }

            let remaining: Vec<u64> = self
                .allocations
                .read()
                .expect("allocations lock")
                .keys()
                .copied()
                .collect();
            for ptr in remaining {
                let _ = self.free_kfd_buffer(ptr);
            }

            for rt in &self.dev_rt {
                if rt.drm_fd >= 0 {
                    unsafe { close(rt.drm_fd) };
                }
            }
            if self.kfd_fd >= 0 {
                unsafe { close(self.kfd_fd) };
            }
        }
    }

    impl DeviceBackend for KfdBackend {
        fn devices(&self) -> &[DeviceInfo] {
            &self.devices
        }

        fn alloc_vram(&self, device: usize, size: usize, align: usize) -> Result<u64, HalError> {
            let rec = self.alloc_kfd_buffer(device, size, align, false, true)?;
            Ok(rec.gpu_va)
        }

        fn free_vram(&self, device: usize, ptr: u64) -> Result<(), HalError> {
            self.check_device(device)?;
            self.free_kfd_buffer(ptr)
        }

        fn copy_htod(&self, device: usize, dst: u64, src: &[u8]) -> Result<(), HalError> {
            self.check_device(device)?;
            if src.is_empty() {
                return Ok(());
            }
            let (host_dst, _) = self.host_slice_mut(dst, src.len())?;
            unsafe {
                core::ptr::copy_nonoverlapping(src.as_ptr(), host_dst, src.len());
            }
            Ok(())
        }

        fn copy_dtoh(&self, device: usize, dst: &mut [u8], src: u64) -> Result<(), HalError> {
            self.check_device(device)?;
            if dst.is_empty() {
                return Ok(());
            }
            let (host_src, _) = self.host_slice_mut(src, dst.len())?;
            unsafe {
                core::ptr::copy_nonoverlapping(host_src as *const u8, dst.as_mut_ptr(), dst.len());
            }
            Ok(())
        }

        fn copy_dtod(&self, device: usize, dst: u64, src: u64, len: usize) -> Result<(), HalError> {
            self.check_device(device)?;
            if len == 0 {
                return Ok(());
            }
            let (host_src, _) = self.host_slice_mut(src, len)?;
            let mut tmp = vec![0u8; len];
            unsafe {
                core::ptr::copy_nonoverlapping(host_src as *const u8, tmp.as_mut_ptr(), len);
            }
            let (host_dst, _) = self.host_slice_mut(dst, len)?;
            unsafe {
                core::ptr::copy_nonoverlapping(tmp.as_ptr(), host_dst, len);
            }
            Ok(())
        }

        fn create_queue(&self, device: usize, ring_size: usize) -> Result<QueueId, HalError> {
            let rt = self.check_device(device)?.clone();
            let ring_cap = ring_size.max(64).next_power_of_two();
            let ring_bytes = (ring_cap * 64).max(PAGE_SIZE);
            let q_sizes =
                compute_kfd_queue_buffer_sizes(rt.cu_count, rt.max_waves_per_cu, rt.num_xcc);

            let ring_alloc = self
                .alloc_kfd_buffer_with_flags(
                    device,
                    ring_bytes,
                    PAGE_SIZE,
                    false,
                    true,
                    KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM,
                    2,
                )
                .or_else(|_| self.alloc_kfd_buffer(device, ring_bytes, PAGE_SIZE, false, true))?;
            let ptrs_alloc = match self.alloc_kfd_buffer(device, PAGE_SIZE, PAGE_SIZE, false, true)
            {
                Ok(a) => a,
                Err(e) => {
                    let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                    return Err(e);
                }
            };
            let eop_alloc = match self.alloc_kfd_buffer(
                device,
                q_sizes.eop_bytes,
                PAGE_SIZE,
                false,
                true,
            ) {
                Ok(a) => a,
                Err(e) => {
                    let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                    let _ = self.free_kfd_buffer(ptrs_alloc.gpu_va);
                    return Err(e);
                }
            };
            let ctx_alloc = match self.alloc_kfd_buffer(
                device,
                q_sizes.ctx_save_restore_bytes,
                PAGE_SIZE,
                false,
                true,
            ) {
                Ok(a) => a,
                Err(e) => {
                    let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                    let _ = self.free_kfd_buffer(ptrs_alloc.gpu_va);
                    let _ = self.free_kfd_buffer(eop_alloc.gpu_va);
                    return Err(e);
                }
            };

            if ring_alloc.host_addr == 0 || ptrs_alloc.host_addr == 0 {
                let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ptrs_alloc.gpu_va);
                let _ = self.free_kfd_buffer(eop_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ctx_alloc.gpu_va);
                return Err(HalError::KfdError(
                    "failed to host-map KFD AQL queue ring or pointers".to_owned(),
                ));
            }
            unsafe {
                core::ptr::write_bytes(ring_alloc.host_addr as *mut u8, 0, ring_bytes);
                core::ptr::write_bytes(ptrs_alloc.host_addr as *mut u8, 0, PAGE_SIZE);
            }

            let wptr_va = ptrs_alloc.gpu_va;
            let rptr_va = ptrs_alloc.gpu_va + 64;
            let mut create_args = KfdIoctlCreateQueueArgs {
                ring_base_address: ring_alloc.gpu_va,
                write_pointer_address: wptr_va,
                read_pointer_address: rptr_va,
                doorbell_offset: 0,
                ring_size: ring_bytes as u32,
                gpu_id: rt.gpu_id,
                queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
                queue_percentage: 100,
                queue_priority: 7,
                queue_id: 0,
                eop_buffer_address: eop_alloc.gpu_va,
                eop_buffer_size: eop_alloc.map_len as u64,
                ctx_save_restore_address: ctx_alloc.gpu_va,
                ctx_save_restore_size: ctx_alloc.map_len as u32,
                ctl_stack_size: q_sizes.ctl_stack_bytes as u32,
            };

            let rc = unsafe {
                ioctl(
                    self.kfd_fd,
                    c_ulong::from(AMDKFD_IOC_CREATE_QUEUE),
                    &mut create_args as *mut KfdIoctlCreateQueueArgs as *mut c_void,
                )
            };
            if rc < 0 {
                let err = std::io::Error::last_os_error();
                let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ptrs_alloc.gpu_va);
                let _ = self.free_kfd_buffer(eop_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ctx_alloc.gpu_va);
                return Err(HalError::KfdError(format!(
                    "AMDKFD_IOC_CREATE_QUEUE failed ({err})"
                )));
            }

            let page_off = (create_args.doorbell_offset & !((PAGE_SIZE as u64) - 1)) as c_long;
            let in_page_off = (create_args.doorbell_offset & ((PAGE_SIZE as u64) - 1)) as usize;
            let db_ptr = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    PAGE_SIZE,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    self.kfd_fd,
                    page_off,
                )
            };
            if db_ptr as usize == MAP_FAILED {
                let err = std::io::Error::last_os_error();
                let d_args = KfdIoctlDestroyQueueArgs {
                    queue_id: create_args.queue_id,
                    pad: 0,
                };
                unsafe {
                    ioctl(
                        self.kfd_fd,
                        c_ulong::from(AMDKFD_IOC_DESTROY_QUEUE),
                        &d_args as *const KfdIoctlDestroyQueueArgs as *mut c_void,
                    );
                }
                let _ = self.free_kfd_buffer(ring_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ptrs_alloc.gpu_va);
                let _ = self.free_kfd_buffer(eop_alloc.gpu_va);
                let _ = self.free_kfd_buffer(ctx_alloc.gpu_va);
                return Err(HalError::KfdError(format!(
                    "mmap for KFD doorbell failed ({err})"
                )));
            }

            let id = self.next_handle();
            self.queues.write().expect("queues write lock").insert(
                id,
                KfdQueue {
                    gpu_va_ring: ring_alloc.gpu_va,
                    gpu_va_ptrs: ptrs_alloc.gpu_va,
                    gpu_va_eop: eop_alloc.gpu_va,
                    gpu_va_ctx: ctx_alloc.gpu_va,
                    kfd_queue_id: create_args.queue_id,
                    ring_cap,
                    ring_host: ring_alloc.host_addr,
                    wptr_host: ptrs_alloc.host_addr,
                    doorbell_page_host: db_ptr as usize,
                    doorbell_offset_in_page: in_page_off,
                    write_idx: Mutex::new(0),
                },
            );
            Ok(QueueId(id))
        }

        fn destroy_queue(&self, queue: QueueId) -> Result<(), HalError> {
            let q = self
                .queues
                .write()
                .expect("queues write lock")
                .remove(&queue.0)
                .ok_or(HalError::InvalidHandle("KFD queue"))?;

            let d_args = KfdIoctlDestroyQueueArgs {
                queue_id: q.kfd_queue_id,
                pad: 0,
            };
            unsafe {
                ioctl(
                    self.kfd_fd,
                    c_ulong::from(AMDKFD_IOC_DESTROY_QUEUE),
                    &d_args as *const KfdIoctlDestroyQueueArgs as *mut c_void,
                );
                if q.doorbell_page_host != 0 {
                    munmap(q.doorbell_page_host as *mut c_void, PAGE_SIZE);
                }
            }
            let _ = self.free_kfd_buffer(q.gpu_va_ring);
            let _ = self.free_kfd_buffer(q.gpu_va_ptrs);
            let _ = self.free_kfd_buffer(q.gpu_va_eop);
            let _ = self.free_kfd_buffer(q.gpu_va_ctx);
            Ok(())
        }

        fn create_signal(&self, initial_value: i64) -> Result<SignalId, HalError> {
            let alloc = self.alloc_kfd_buffer(0, PAGE_SIZE, 64, false, true)?;
            if alloc.host_addr == 0 {
                let _ = self.free_kfd_buffer(alloc.gpu_va);
                return Err(HalError::KfdError(
                    "failed to host-map coherent HSA signal memory".to_owned(),
                ));
            }
            unsafe {
                let sig = alloc.host_addr as *mut HsaSignalStruct;
                core::ptr::write_bytes(sig as *mut u8, 0, size_of::<HsaSignalStruct>());
                (*sig).kind = AMD_SIGNAL_KIND_USER;
                (*sig).value.store(initial_value, Ordering::Release);
            }
            let sig_va = alloc.gpu_va;
            self.signals.write().expect("signals write lock").insert(
                sig_va,
                KfdSignal {
                    gpu_va: sig_va,
                    host_addr: alloc.host_addr,
                },
            );
            Ok(SignalId(sig_va))
        }

        fn destroy_signal(&self, signal: SignalId) -> Result<(), HalError> {
            let s = self
                .signals
                .write()
                .expect("signals write lock")
                .remove(&signal.0)
                .ok_or(HalError::InvalidHandle("KFD signal"))?;
            self.free_kfd_buffer(s.gpu_va)
        }

        fn signal_store(&self, signal: SignalId, value: i64) -> Result<(), HalError> {
            let guard = self.signals.read().expect("signals read lock");
            let s = guard
                .get(&signal.0)
                .ok_or(HalError::InvalidHandle("KFD signal"))?;
            unsafe {
                let ptr = s.host_addr as *const HsaSignalStruct;
                (*ptr).value.store(value, Ordering::Release);
            }
            Ok(())
        }

        fn signal_load(&self, signal: SignalId) -> Result<i64, HalError> {
            let guard = self.signals.read().expect("signals read lock");
            let s = guard
                .get(&signal.0)
                .ok_or(HalError::InvalidHandle("KFD signal"))?;
            let val = unsafe {
                let ptr = s.host_addr as *const HsaSignalStruct;
                (*ptr).value.load(Ordering::Acquire)
            };
            Ok(val)
        }

        fn signal_wait(&self, signal: SignalId, expected: i64) -> Result<(), HalError> {
            let host_addr = {
                let guard = self.signals.read().expect("signals read lock");
                guard
                    .get(&signal.0)
                    .ok_or(HalError::InvalidHandle("KFD signal"))?
                    .host_addr
            };
            let start = Instant::now();
            let timeout = Duration::from_millis(self.watchdog_ms);
            let mut spins = 0u32;
            loop {
                let cur = unsafe {
                    let ptr = host_addr as *const HsaSignalStruct;
                    (*ptr).value.load(Ordering::Acquire)
                };
                if cur == expected {
                    return Ok(());
                }
                if start.elapsed() >= timeout {
                    return Err(HalError::ExecutionError(format!(
                        "GPU watchdog timeout after {}ms waiting for signal 0x{:x} (observed {cur}, expected {expected})",
                        self.watchdog_ms, signal.0
                    )));
                }
                spins += 1;
                if spins < 256 {
                    core::hint::spin_loop();
                } else if spins < 1024 {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(Duration::from_micros(50));
                }
            }
        }

        fn load_code_object(&self, device: usize, hsaco: &[u8]) -> Result<CodeObjectId, HalError> {
            self.check_device(device)?;
            let image = parse_hsaco(hsaco)
                .map_err(|e| HalError::InvalidCodeObject(format!("HSACO parse error: {e}")))?;

            let rodata_sec = image.sections.iter().find(|s| s.name == ".rodata");
            let text_sec = image.sections.iter().find(|s| s.name == ".text");

            let rodata_bytes = rodata_sec.map(|s| s.data.as_slice()).unwrap_or(&[]);
            let text_bytes = text_sec.map(|s| s.data.as_slice()).unwrap_or(&[]);
            if text_bytes.is_empty() {
                return Err(HalError::InvalidCodeObject(
                    "HSACO lacks .text machine code section".to_owned(),
                ));
            }

            let rodata_padded = if rodata_bytes.is_empty() {
                64 * image.kernels.len().max(1)
            } else {
                rodata_bytes.len().div_ceil(256) * 256
            };
            let total_size = rodata_padded + text_bytes.len();
            let alloc = self.alloc_kfd_buffer(device, total_size, PAGE_SIZE, true, true)?;
            if alloc.host_addr == 0 {
                let _ = self.free_kfd_buffer(alloc.gpu_va);
                return Err(HalError::KfdError(
                    "failed to host-map code object memory".to_owned(),
                ));
            }

            let text_gpu_va = alloc.gpu_va + rodata_padded as u64;
            let host_base = alloc.host_addr as *mut u8;
            unsafe {
                core::ptr::write_bytes(host_base, 0, rodata_padded);
                if !rodata_bytes.is_empty() {
                    core::ptr::copy_nonoverlapping(rodata_bytes.as_ptr(), host_base, rodata_bytes.len());
                }
                core::ptr::copy_nonoverlapping(
                    text_bytes.as_ptr(),
                    host_base.add(rodata_padded),
                    text_bytes.len(),
                );
            }

            let text_sh_addr = text_sec.map(|s| s.addr).unwrap_or(0);
            let rodata_sh_addr = rodata_sec.map(|s| s.addr).unwrap_or(0);
            let text_len_u64 = text_bytes.len() as u64;

            let kd_count = (rodata_bytes.len() / 64).max(image.kernels.len());
            let mut patched_entries = Vec::with_capacity(image.kernels.len());

            for (idx, meta) in image.kernels.iter().enumerate() {
                let kd_idx = idx.min(kd_count.saturating_sub(1));
                let kd_off = kd_idx * 64;
                let kd_gpu_va = alloc.gpu_va + kd_off as u64;

                if kd_off + 64 <= rodata_padded {
                    let mut kd_buf = [0u8; 64];
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            host_base.add(kd_off),
                            kd_buf.as_mut_ptr(),
                            64,
                        );
                    }
                    let orig_entry_off = i64::from_le_bytes([
                        kd_buf[16], kd_buf[17], kd_buf[18], kd_buf[19], kd_buf[20], kd_buf[21],
                        kd_buf[22], kd_buf[23],
                    ]);
                    let rel_in_text = resolve_kernel_text_entry_offset(
                        rodata_sh_addr,
                        text_sh_addr,
                        text_len_u64,
                        kd_off,
                        orig_entry_off,
                    );
                    let entry_gpu_va = text_gpu_va + rel_in_text;
                    if let Err(e) =
                        patch_kernel_descriptor_entry_offset(&mut kd_buf, kd_gpu_va, entry_gpu_va)
                    {
                        let _ = self.free_kfd_buffer(alloc.gpu_va);
                        return Err(e);
                    }
                    unsafe {
                        core::ptr::copy_nonoverlapping(kd_buf.as_ptr(), host_base.add(kd_off), 64);
                    }
                }

                patched_entries.push((kd_gpu_va, meta.clone()));
            }

            let mut kernel_ids = Vec::with_capacity(patched_entries.len());
            let mut kguard = self.kernels.write().expect("kernels write lock");
            for (kd_gpu_va, metadata) in patched_entries {
                let kid = self.next_handle();
                kguard.insert(
                    kid,
                    KfdKernelEntry {
                        kd_gpu_va,
                        metadata,
                    },
                );
                kernel_ids.push(kid);
            }
            drop(kguard);

            fence(Ordering::SeqCst);
            let obj_id = self.next_handle();
            self.code_objects
                .write()
                .expect("code_objects write lock")
                .insert(
                    obj_id,
                    KfdCodeObject {
                        gpu_va: alloc.gpu_va,
                        kernel_ids,
                    },
                );
            Ok(CodeObjectId(obj_id))
        }

        fn unload_code_object(&self, obj: CodeObjectId) -> Result<(), HalError> {
            let co = self
                .code_objects
                .write()
                .expect("code_objects write lock")
                .remove(&obj.0)
                .ok_or(HalError::InvalidHandle("KFD code object"))?;
            let mut kguard = self.kernels.write().expect("kernels write lock");
            for kid in co.kernel_ids {
                kguard.remove(&kid);
            }
            drop(kguard);
            self.free_kfd_buffer(co.gpu_va)
        }

        fn get_kernel(
            &self,
            obj: CodeObjectId,
            name: &str,
        ) -> Result<(KernelHandle, KernelMetadata), HalError> {
            let co_kids = self
                .code_objects
                .read()
                .expect("code_objects read lock")
                .get(&obj.0)
                .map(|c| c.kernel_ids.clone())
                .ok_or(HalError::InvalidHandle("KFD code object"))?;

            let kguard = self.kernels.read().expect("kernels read lock");
            for kid in co_kids {
                if let Some(entry) = kguard.get(&kid) {
                    if entry.metadata.name == name {
                        return Ok((KernelHandle(entry.kd_gpu_va), entry.metadata.clone()));
                    }
                }
            }
            Err(HalError::KernelNotFound(name.to_owned()))
        }

        fn submit_aql_packet(&self, queue: QueueId, packet: &[u8; 64]) -> Result<(), HalError> {
            let guard = self.queues.read().expect("queues read lock");
            let q = guard
                .get(&queue.0)
                .ok_or(HalError::InvalidHandle("KFD queue"))?;
            if q.ring_host == 0 || q.wptr_host == 0 || q.doorbell_page_host == 0 {
                return Err(HalError::KfdError(
                    "KFD queue pointers not mapped".to_owned(),
                ));
            }

            let mut w_guard = q.write_idx.lock().expect("queue wptr lock");
            let idx = *w_guard;
            let slot_idx = (idx as usize) & (q.ring_cap - 1);
            let slot_ptr = unsafe { (q.ring_host as *mut u8).add(slot_idx * 64) };

            unsafe {
                // Publish payload bytes [2..64] first, then Release-store 16-bit header at [0..2].
                core::ptr::copy_nonoverlapping(packet[2..64].as_ptr(), slot_ptr.add(2), 62);
                let header = u16::from_le_bytes([packet[0], packet[1]]);
                let hdr_atom = &*(slot_ptr as *const AtomicU16);
                hdr_atom.store(header, Ordering::Release);

                // Publish updated write_index and ring the 64-bit KFD doorbell.
                let new_idx = idx + 1;
                let wptr_atom = &*(q.wptr_host as *const AtomicU64);
                wptr_atom.store(new_idx, Ordering::Release);
                fence(Ordering::SeqCst);

                let doorbell_ptr = (q.doorbell_page_host + q.doorbell_offset_in_page) as *mut u64;
                core::ptr::write_volatile(doorbell_ptr, idx);
            }
            *w_guard = idx + 1;
            Ok(())
        }
    }
}

/// Attempts to open the native Linux `/dev/kfd` backend. Returns [`HalError::UnsupportedOnOs`] on non-Linux hosts.
pub fn try_open_kfd_backend() -> Result<Box<dyn crate::backend::DeviceBackend>, HalError> {
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(linux::KfdBackend::open_system()?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(HalError::UnsupportedOnOs(
            "/dev/kfd is only available on Linux hosts",
        ))
    }
}
