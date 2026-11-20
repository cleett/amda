//! Error types for AMDGPU HSACO ELF64 and MessagePack metadata operations.

use core::fmt;

/// Errors returned when parsing or validating AMDGPU HSACO ELF64 binaries and metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElfError {
    /// File header does not begin with `\x7FELF`.
    InvalidMagic,
    /// Binary is not ELF64 (`EI_CLASS != 2`) or not little-endian (`EI_DATA != 1`).
    UnsupportedClass,
    /// Binary machine is not `EM_AMDGPU` (`224`).
    UnsupportedMachine(u16),
    /// ELF header flags specify an unrecognized AMDGPU `EF_AMDGPU_MACH_*` target.
    UnsupportedIsa(u32),
    /// Input buffer ended prematurely while parsing headers, sections, or notes.
    Truncated,
    /// Section or program header offsets/sizes fall outside the ELF byte slice or overflow.
    OutOfBounds,
    /// MessagePack payload in `.note` could not be decoded or violates schema expectations.
    InvalidMsgPack(String),
    /// ELF image did not contain an `NT_AMDGPU_METADATA` (`32`) note with `amdhsa.kernels`.
    MissingMetadata,
}

impl fmt::Display for ElfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid ELF magic (expected \\x7FELF)"),
            Self::UnsupportedClass => write!(f, "unsupported ELF class or endianness (expected ELF64 LSB)"),
            Self::UnsupportedMachine(m) => write!(f, "unsupported ELF machine {m} (expected EM_AMDGPU = 224)"),
            Self::UnsupportedIsa(flags) => write!(f, "unsupported AMDGPU ISA ELF flags 0x{flags:08x}"),
            Self::Truncated => write!(f, "truncated ELF or metadata buffer"),
            Self::OutOfBounds => write!(f, "ELF section or segment exceeds buffer bounds"),
            Self::InvalidMsgPack(msg) => write!(f, "invalid AMDGPU MsgPack metadata: {msg}"),
            Self::MissingMetadata => write!(f, "missing NT_AMDGPU_METADATA note in HSACO image"),
        }
    }
}

impl std::error::Error for ElfError {}
