//! Pluggable hardware submission backend trait (`DeviceBackend`).

use crate::device::DeviceInfo;
use crate::error::HalError;
use amda_elf::KernelMetadata;

/// Opaque handle to a hardware AQL queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueId(pub u64);

/// Opaque handle to an HSA hardware signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignalId(pub u64);

/// Opaque handle to a loaded HSACO code object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodeObjectId(pub u64);

/// Opaque handle to an executable kernel object inside a loaded code object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelHandle(pub u64);

/// Trait implemented by physical (`/dev/kfd`) and virtual (`MockBackend`) GPU backends.
pub trait DeviceBackend: Send + Sync {
    /// Returns all discovered GPU devices managed by this backend.
    fn devices(&self) -> &[DeviceInfo];

    /// Allocates `size` bytes of device VRAM aligned to `align`, returning a GPU virtual address.
    fn alloc_vram(&self, device: usize, size: usize, align: usize) -> Result<u64, HalError>;

    /// Frees a previously allocated VRAM buffer.
    fn free_vram(&self, device: usize, ptr: u64) -> Result<(), HalError>;

    /// Copies bytes from host memory into device VRAM.
    fn copy_htod(&self, device: usize, dst: u64, src: &[u8]) -> Result<(), HalError>;

    /// Copies bytes from device VRAM into host memory.
    fn copy_dtoh(&self, device: usize, dst: &mut [u8], src: u64) -> Result<(), HalError>;

    /// Copies `len` bytes between two device VRAM buffers.
    fn copy_dtod(&self, device: usize, dst: u64, src: u64, len: usize) -> Result<(), HalError>;

    /// Creates a user-mode AQL queue with at least `ring_size` packet slots.
    fn create_queue(&self, device: usize, ring_size: usize) -> Result<QueueId, HalError>;

    /// Destroys a previously created AQL queue.
    fn destroy_queue(&self, queue: QueueId) -> Result<(), HalError>;

    /// Creates a hardware synchronization signal initialized to `initial_value`.
    fn create_signal(&self, initial_value: i64) -> Result<SignalId, HalError>;

    /// Destroys a signal handle.
    fn destroy_signal(&self, signal: SignalId) -> Result<(), HalError>;

    /// Sets a signal's atomic value.
    fn signal_store(&self, signal: SignalId, value: i64) -> Result<(), HalError>;

    /// Reads a signal's current atomic value.
    fn signal_load(&self, signal: SignalId) -> Result<i64, HalError>;

    /// Blocks until the signal reaches `expected`.
    fn signal_wait(&self, signal: SignalId, expected: i64) -> Result<(), HalError>;

    /// Loads an HSACO ELF64 code object onto `device`.
    fn load_code_object(&self, device: usize, hsaco: &[u8]) -> Result<CodeObjectId, HalError>;

    /// Unloads a loaded HSACO code object.
    fn unload_code_object(&self, obj: CodeObjectId) -> Result<(), HalError>;

    /// Looks up a kernel entry point and its AMDGPU metadata inside `obj`.
    fn get_kernel(&self, obj: CodeObjectId, name: &str) -> Result<(KernelHandle, KernelMetadata), HalError>;

    /// Publishes a 64-byte AQL packet into `queue`'s ring buffer and rings its doorbell.
    fn submit_aql_packet(&self, queue: QueueId, packet: &[u8; 64]) -> Result<(), HalError>;
}
