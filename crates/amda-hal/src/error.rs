//! Error types for the AMDA Hardware Abstraction Layer.

use core::fmt;

/// Errors returned by HAL backends (`/dev/kfd` and `MockBackend`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HalError {
    /// Requested device index does not exist.
    DeviceNotFound(usize),
    /// An invalid queue, signal, code object, or kernel handle was passed.
    InvalidHandle(&'static str),
    /// Device or virtual MMU allocation failed or requested invalid size.
    OutOfMemory(String),
    /// GPU memory access violated virtual MMU allocation bounds or alignment.
    MemoryAccessFault { addr: u64, size: usize, reason: String },
    /// AQL ring buffer is full.
    QueueFull,
    /// Named kernel entry point was not found in the loaded code object.
    KernelNotFound(String),
    /// HSACO code object failed validation or lacks executable IR/ISA payload.
    InvalidCodeObject(String),
    /// Kernel or packet execution error in the command processor.
    ExecutionError(String),
    /// Requested backend or operation is unsupported on the current OS.
    UnsupportedOnOs(&'static str),
    /// Linux `/dev/kfd` ioctl or topology error.
    KfdError(String),
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(idx) => write!(f, "AMD GPU device index {idx} not found"),
            Self::InvalidHandle(kind) => write!(f, "invalid {kind} handle"),
            Self::OutOfMemory(msg) => write!(f, "GPU out of memory: {msg}"),
            Self::MemoryAccessFault { addr, size, reason } => {
                write!(f, "GPU memory fault at 0x{addr:016x} (size {size}): {reason}")
            }
            Self::QueueFull => write!(f, "AQL hardware queue ring buffer is full"),
            Self::KernelNotFound(name) => write!(f, "kernel '{name}' not found in code object"),
            Self::InvalidCodeObject(msg) => write!(f, "invalid HSACO code object: {msg}"),
            Self::ExecutionError(msg) => write!(f, "GPU execution error: {msg}"),
            Self::UnsupportedOnOs(msg) => write!(f, "unsupported on host OS: {msg}"),
            Self::KfdError(msg) => write!(f, "Linux KFD error: {msg}"),
        }
    }
}

impl std::error::Error for HalError {}
