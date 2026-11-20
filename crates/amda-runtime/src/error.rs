//! Error types for the AMDA core execution runtime.

use amda_compiler::CompileError;
use amda_elf::ElfError;
use amda_hal::HalError;
use core::fmt;

/// Errors returned by `amda-runtime`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Hal(HalError),
    Compile(CompileError),
    Elf(ElfError),
    InvalidArgument(String),
    InvalidHandle(&'static str),
    KernelNotFound(String),
    NotReady,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hal(e) => write!(f, "HAL error: {e}"),
            Self::Compile(e) => write!(f, "compiler error: {e}"),
            Self::Elf(e) => write!(f, "ELF error: {e}"),
            Self::InvalidArgument(msg) => write!(f, "invalid runtime argument: {msg}"),
            Self::InvalidHandle(kind) => write!(f, "invalid runtime {kind} handle"),
            Self::KernelNotFound(name) => write!(f, "kernel '{name}' not found"),
            Self::NotReady => write!(f, "stream or event work not yet complete"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<HalError> for RuntimeError {
    fn from(e: HalError) -> Self {
        Self::Hal(e)
    }
}

impl From<CompileError> for RuntimeError {
    fn from(e: CompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<ElfError> for RuntimeError {
    fn from(e: ElfError) -> Self {
        Self::Elf(e)
    }
}
