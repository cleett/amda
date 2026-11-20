//! Tiered JIT compilation policy and tier selection heuristics.

use crate::ir::{IrModule, WaveMode};

/// Compilation backend tier used to emit an HSACO artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileTier {
    /// BarraCUDA-inspired zero-LLVM direct AMDGPU machine-code backend (sub-ms cold compile).
    Tier0Fast,
    /// Full optimizing AMDA Tile/LLVM IR pipeline (for hot kernels, fusion graphs, and Wave64 emulation).
    Tier1Optimizing,
}

impl CompileTier {
    /// Deterministic integer tag included in cache keys.
    pub const fn as_tag(self) -> u8 {
        match self {
            Self::Tier0Fast => 0,
            Self::Tier1Optimizing => 1,
        }
    }

    /// Human-readable tier name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tier0Fast => "tier0-fast-native",
            Self::Tier1Optimizing => "tier1-optimizing-llvm",
        }
    }
}

/// User or runtime policy governing tier selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierPolicy {
    /// Start cold kernels at Tier 0 and promote to Tier 1 when hot, complex, or fusible.
    Auto,
    /// Force Tier 0 zero-LLVM direct native ISA emission.
    ForceTier0,
    /// Force Tier 1 optimizing LLVM AMDGPU IR pipeline.
    ForceTier1,
}

impl TierPolicy {
    /// Reads `AMDA_TIER` (`auto`, `0`/`tier0`/`fast`, or `1`/`tier1`/`opt`). Defaults to `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("AMDA_TIER")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "0" | "tier0" | "fast" => Self::ForceTier0,
            "1" | "tier1" | "opt" => Self::ForceTier1,
            _ => Self::Auto,
        }
    }
}

/// Reads `AMDA_PROMOTE_THRESHOLD` (number of launches before Tier 0 promotes to Tier 1).
pub fn promote_threshold_from_env() -> u64 {
    std::env::var("AMDA_PROMOTE_THRESHOLD")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16)
}

/// Maximum IR instruction count kept on Tier 0 before escalating directly to Tier 1.
const TIER0_MAX_INSTS: usize = 256;

/// Selects the appropriate [`CompileTier`] for an [`IrModule`] given runtime signals.
pub fn select_tier(
    ir: &IrModule,
    policy: TierPolicy,
    launch_count: u64,
    in_fusion_graph: bool,
    promote_threshold: u64,
) -> CompileTier {
    match policy {
        TierPolicy::ForceTier0 => CompileTier::Tier0Fast,
        TierPolicy::ForceTier1 => CompileTier::Tier1Optimizing,
        TierPolicy::Auto => {
            if in_fusion_graph || launch_count >= promote_threshold {
                return CompileTier::Tier1Optimizing;
            }
            for k in &ir.kernels {
                if k.wave_mode == WaveMode::EmulatedWave32OnWave64 || k.insts.len() > TIER0_MAX_INSTS
                {
                    return CompileTier::Tier1Optimizing;
                }
            }
            CompileTier::Tier0Fast
        }
    }
}
