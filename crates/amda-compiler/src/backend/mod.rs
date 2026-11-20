//! Tier 0 pure-Rust native AMDGPU codegen backends (zero LLVM dependency).

pub mod amdgpu_isa;
pub mod regalloc;

pub use amdgpu_isa::{
    compile_tier0_native, verify_encoded_machine_code, DecodedInst, NativeKernelCode,
    NativeModuleCode,
};
pub use regalloc::{
    allocate_registers, analyze_divergence, Divergence, KernelAllocation, PhysReg, RegAllocSummary,
};
