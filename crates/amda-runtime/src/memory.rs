//! Caching power-of-two slab allocator and Virtual Memory Management (VMM) registry.

use crate::error::RuntimeError;
use amda_hal::DeviceBackend;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

const MIN_BUCKET_SIZE: usize = 256;
const MAX_POOL_BUCKET_SIZE: usize = 16 * 1024 * 1024; // 16 MiB

/// Allocation statistics exposed by [`MemoryPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub active_allocations: usize,
    pub active_bytes: usize,
    pub pooled_bytes: usize,
}

#[derive(Default)]
struct PoolState {
    live: HashMap<u64, usize>,
    free_buckets: HashMap<usize, Vec<u64>>,
    stats: MemoryStats,
}

/// Caching power-of-two slab sub-allocator layered over `DeviceBackend::alloc_vram`.
pub struct MemoryPool {
    state: Mutex<PoolState>,
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool {
    /// Creates an empty device memory pool.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(PoolState::default()),
        }
    }

    /// Allocates at least `size` bytes, reusing a pooled slab when available.
    pub fn alloc(
        &self,
        backend: &dyn DeviceBackend,
        device: usize,
        size: usize,
        align: usize,
    ) -> Result<u64, RuntimeError> {
        let bucket = if size == 0 {
            MIN_BUCKET_SIZE
        } else if size <= MAX_POOL_BUCKET_SIZE {
            size.max(MIN_BUCKET_SIZE).next_power_of_two()
        } else {
            size
        };

        if bucket <= MAX_POOL_BUCKET_SIZE && align <= MIN_BUCKET_SIZE {
            let mut st = self.state.lock().expect("memory pool lock");
            if let Some(list) = st.free_buckets.get_mut(&bucket) {
                if let Some(ptr) = list.pop() {
                    st.stats.pool_hits += 1;
                    st.stats.pooled_bytes = st.stats.pooled_bytes.saturating_sub(bucket);
                    st.stats.active_allocations += 1;
                    st.stats.active_bytes += bucket;
                    st.live.insert(ptr, bucket);
                    return Ok(ptr);
                }
            }
        }

        let ptr = backend.alloc_vram(device, bucket, align.max(MIN_BUCKET_SIZE))?;
        let mut st = self.state.lock().expect("memory pool lock");
        st.stats.pool_misses += 1;
        st.stats.active_allocations += 1;
        st.stats.active_bytes += bucket;
        st.live.insert(ptr, bucket);
        Ok(ptr)
    }

    /// Returns a buffer to the slab pool (or frees directly if larger than 16 MiB).
    pub fn free(
        &self,
        backend: &dyn DeviceBackend,
        device: usize,
        ptr: u64,
    ) -> Result<(), RuntimeError> {
        if ptr == 0 {
            return Ok(());
        }
        let bucket = {
            let mut st = self.state.lock().expect("memory pool lock");
            let bucket = st
                .live
                .remove(&ptr)
                .ok_or(RuntimeError::InvalidHandle("device pointer"))?;
            st.stats.active_allocations = st.stats.active_allocations.saturating_sub(1);
            st.stats.active_bytes = st.stats.active_bytes.saturating_sub(bucket);
            if bucket <= MAX_POOL_BUCKET_SIZE {
                st.stats.pooled_bytes += bucket;
                st.free_buckets.entry(bucket).or_default().push(ptr);
                return Ok(());
            }
            bucket
        };

        let _ = bucket;
        backend.free_vram(device, ptr)?;
        Ok(())
    }

    /// Releases all cached free slabs back to the HAL backend.
    pub fn trim(&self, backend: &dyn DeviceBackend, device: usize) -> Result<(), RuntimeError> {
        let to_free: Vec<u64> = {
            let mut st = self.state.lock().expect("memory pool lock");
            st.stats.pooled_bytes = 0;
            st.free_buckets.drain().flat_map(|(_, v)| v).collect()
        };
        for ptr in to_free {
            backend.free_vram(device, ptr)?;
        }
        Ok(())
    }

    /// Returns a snapshot of allocator statistics.
    pub fn stats(&self) -> MemoryStats {
        self.state.lock().expect("memory pool lock").stats
    }

    /// Checks whether `ptr` falls inside any live allocation, returning `(base, size)`.
    pub fn contains_ptr(&self, ptr: u64) -> Option<(u64, usize)> {
        if ptr == 0 {
            return None;
        }
        let st = self.state.lock().expect("memory pool lock");
        for (&base, &size) in &st.live {
            if ptr >= base && ptr < base.saturating_add(size as u64) {
                return Some((base, size));
            }
        }
        None
    }
}

/// Reserved virtual address range entry for CUDA-style VMM (`cuMemAddressReserve` / `cuMemMap`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmmReservation {
    pub va_base: u64,
    pub size: usize,
    pub mapped_backing: Option<u64>,
}

/// Virtual Memory Management address reservation registry (foundation for Milestone 3 VMM).
pub struct VmmRegistry {
    next_va: u64,
    reservations: BTreeMap<u64, VmmReservation>,
}

impl Default for VmmRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VmmRegistry {
    pub fn new() -> Self {
        Self {
            next_va: 0x0000_4000_0000_0000,
            reservations: BTreeMap::new(),
        }
    }

    /// Reserves a contiguous virtual address range aligned to `align`.
    pub fn reserve_va(&mut self, size: usize, align: usize) -> Result<u64, RuntimeError> {
        if size == 0 {
            return Err(RuntimeError::InvalidArgument(
                "VMM reservation size must be non-zero".to_owned(),
            ));
        }
        let align = align.max(4096).next_power_of_two() as u64;
        let va = (self.next_va + align - 1) & !(align - 1);
        self.next_va = va
            .checked_add(size as u64)
            .and_then(|v| v.checked_add(4096))
            .ok_or_else(|| RuntimeError::InvalidArgument("VMM address overflow".to_owned()))?;
        self.reservations.insert(
            va,
            VmmReservation {
                va_base: va,
                size,
                mapped_backing: None,
            },
        );
        Ok(va)
    }

    /// Associates a physical allocation handle/pointer with a reserved VA range.
    pub fn map_va(&mut self, va: u64, backing_ptr: u64) -> Result<(), RuntimeError> {
        let entry = self
            .reservations
            .get_mut(&va)
            .ok_or(RuntimeError::InvalidHandle("VMM virtual address"))?;
        entry.mapped_backing = Some(backing_ptr);
        Ok(())
    }

    /// Clears any backing mapping from `va`.
    pub fn unmap_va(&mut self, va: u64) -> Result<(), RuntimeError> {
        let entry = self
            .reservations
            .get_mut(&va)
            .ok_or(RuntimeError::InvalidHandle("VMM virtual address"))?;
        entry.mapped_backing = None;
        Ok(())
    }

    /// Releases a virtual address reservation.
    pub fn free_va(&mut self, va: u64) -> Result<(), RuntimeError> {
        if self.reservations.remove(&va).is_none() {
            return Err(RuntimeError::InvalidHandle("VMM virtual address"));
        }
        Ok(())
    }

    /// Returns a reservation record by base address.
    pub fn get(&self, va: u64) -> Option<&VmmReservation> {
        self.reservations.get(&va)
    }
}
