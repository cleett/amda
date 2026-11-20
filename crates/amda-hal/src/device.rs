//! ISA-driven hardware feature discovery and device metadata (no SKU allowlists).

use amda_elf::IsaTarget;

/// Hardware capabilities derived directly from ISA architecture and KFD topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareFeatures {
    pub wave32: bool,
    pub wave64: bool,
    pub wmma: bool,
    pub mfma: bool,
    pub fp8: bool,
    pub bf16: bool,
    pub unified_memory: bool,
    pub lds_bytes_per_cu: u32,
    pub max_waves_per_cu: u32,
    pub compute_units: u32,
}

impl HardwareFeatures {
    /// Derives architectural hardware features from `isa` and topology counts.
    pub const fn from_isa(isa: IsaTarget, compute_units: u32, unified_memory: bool) -> Self {
        match isa {
            IsaTarget::Gfx908 => Self {
                wave32: false,
                wave64: true,
                wmma: false,
                mfma: true,
                fp8: false,
                bf16: true,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 40,
                compute_units,
            },
            IsaTarget::Gfx90a => Self {
                wave32: false,
                wave64: true,
                wmma: false,
                mfma: true,
                fp8: false,
                bf16: true,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 32,
                compute_units,
            },
            IsaTarget::Gfx942 => Self {
                wave32: false,
                wave64: true,
                wmma: false,
                mfma: true,
                fp8: true,
                bf16: true,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 32,
                compute_units,
            },
            IsaTarget::Gfx1030 => Self {
                wave32: true,
                wave64: true,
                wmma: false,
                mfma: false,
                fp8: false,
                bf16: false,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 32,
                compute_units,
            },
            IsaTarget::Gfx1100
            | IsaTarget::Gfx1101
            | IsaTarget::Gfx1102
            | IsaTarget::Gfx1103 => Self {
                wave32: true,
                wave64: true,
                wmma: true,
                mfma: false,
                fp8: false,
                bf16: true,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 32,
                compute_units,
            },
            IsaTarget::Gfx1200 | IsaTarget::Gfx1201 => Self {
                wave32: true,
                wave64: true,
                wmma: true,
                mfma: false,
                fp8: true,
                bf16: true,
                unified_memory,
                lds_bytes_per_cu: 65_536,
                max_waves_per_cu: 32,
                compute_units,
            },
        }
    }
}

/// Discovered AMD GPU device description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub isa: IsaTarget,
    pub vram_bytes: u64,
    pub features: HardwareFeatures,
    pub backend_name: &'static str,
}
