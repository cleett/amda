//! Divergence-aware SGPR/VGPR register allocation (BarraCUDA-inspired).
//!
//! Performs fixed-point uniformity/divergence analysis over AMDA IR so wave-uniform
//! values (kernel parameters, `%ctaid`, `%ntid`) live in scalar `SGPR`s while
//! lane-divergent values (`%tid`, element indices, loaded data) live in vector `VGPR`s.

use crate::error::CompileError;
use crate::ir::{AddressSpace, IrKernel, IrOp, IrOperand, IrType, SpecialReg};
use amda_elf::IsaTarget;
use std::collections::HashMap;

/// Wavefront uniformity classification of an IR register or operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Divergence {
    /// Identical across every active lane in a wavefront (maps to `SGPR`).
    Uniform,
    /// Varies across lanes in a wavefront (maps to `VGPR`).
    Divergent,
}

/// Physical AMDGPU register assigned to an IR virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysReg {
    Sgpr(u8),
    SgprPair(u8),
    Vgpr(u8),
    VgprPair(u8),
    Vcc,
}

impl PhysReg {
    /// Formats this physical register in AMDGPU assembly notation.
    pub fn fmt_asm(self) -> String {
        match self {
            Self::Sgpr(n) => format!("s{n}"),
            Self::SgprPair(n) => format!("s[{n}:{}]", n + 1),
            Self::Vgpr(n) => format!("v{n}"),
            Self::VgprPair(n) => format!("v[{n}:{}]", n + 1),
            Self::Vcc => "vcc".to_owned(),
        }
    }

    /// Returns the base register index (`0..255`).
    pub const fn base_idx(self) -> u8 {
        match self {
            Self::Sgpr(n) | Self::SgprPair(n) | Self::Vgpr(n) | Self::VgprPair(n) => n,
            Self::Vcc => 106,
        }
    }
}

/// Compact register allocation & occupancy summary attached to compiled artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegAllocSummary {
    pub kernel_name: String,
    pub sgpr_count: u32,
    pub vgpr_count: u32,
    pub uniform_ir_regs: usize,
    pub divergent_ir_regs: usize,
    pub code_bytes: usize,
    pub estimated_waves_per_simd: u32,
}

/// Full register allocation result for one kernel.
#[derive(Debug, Clone)]
pub struct KernelAllocation {
    pub divergence: Vec<Divergence>,
    pub reg_map: Vec<PhysReg>,
    pub sgpr_count: u32,
    pub vgpr_count: u32,
    pub uniform_ir_regs: usize,
    pub divergent_ir_regs: usize,
    pub estimated_waves_per_simd: u32,
}

/// Runs fixed-point divergence analysis across `kernel.insts`.
pub fn analyze_divergence(kernel: &IrKernel) -> Vec<Divergence> {
    let n = kernel.reg_types.len();
    let mut div = vec![Divergence::Uniform; n];

    let mut changed = true;
    while changed {
        changed = false;
        for inst in &kernel.insts {
            let pred_div = match inst.pred {
                Some(pg) => div.get(pg.reg as usize).copied().unwrap_or(Divergence::Uniform),
                None => Divergence::Uniform,
            };

            let (dst_opt, rhs_div) = match &inst.op {
                IrOp::LoadParam { dst, .. } => (Some(*dst), Divergence::Uniform),
                IrOp::Load {
                    dst, space, base, ..
                } => {
                    let base_div = operand_div(base, &div);
                    // Shared memory loads or loads with divergent bases produce divergent results;
                    // uniform global loads stay uniform only if predicate & base are uniform.
                    let op_div = if *space == AddressSpace::Shared || base_div == Divergence::Divergent
                    {
                        Divergence::Divergent
                    } else {
                        Divergence::Uniform
                    };
                    (Some(*dst), op_div)
                }
                IrOp::CvtaToGlobal { dst, src } | IrOp::CvtaToShared { dst, src } => {
                    (Some(*dst), operand_div(src, &div))
                }
                IrOp::Mov { dst, src, .. }
                | IrOp::Cvt { dst, src, .. }
                | IrOp::Neg { dst, src, .. }
                | IrOp::Abs { dst, src, .. }
                | IrOp::UnaryMath { dst, src, .. } => (Some(*dst), operand_div(src, &div)),
                IrOp::Bin { dst, lhs, rhs, .. } | IrOp::Setp { dst, lhs, rhs, .. } => {
                    let d = combine_div(operand_div(lhs, &div), operand_div(rhs, &div));
                    (Some(*dst), d)
                }
                IrOp::Mad { dst, a, b, c, .. } => {
                    let d = combine_div(
                        operand_div(a, &div),
                        combine_div(operand_div(b, &div), operand_div(c, &div)),
                    );
                    (Some(*dst), d)
                }
                IrOp::Bfe { dst, src, pos, len, .. } => {
                    let d = combine_div(
                        operand_div(src, &div),
                        combine_div(operand_div(pos, &div), operand_div(len, &div)),
                    );
                    (Some(*dst), d)
                }
                IrOp::Bfi { dst, insert, base, pos, len } => {
                    let d = combine_div(
                        operand_div(insert, &div),
                        combine_div(
                            operand_div(base, &div),
                            combine_div(operand_div(pos, &div), operand_div(len, &div)),
                        ),
                    );
                    (Some(*dst), d)
                }
                IrOp::Prmt { dst, a, b, sel } => {
                    let d = combine_div(
                        operand_div(a, &div),
                        combine_div(operand_div(b, &div), operand_div(sel, &div)),
                    );
                    (Some(*dst), d)
                }
                IrOp::Dp4a { dst, a, b, c, .. } => {
                    let d = combine_div(
                        operand_div(a, &div),
                        combine_div(operand_div(b, &div), operand_div(c, &div)),
                    );
                    (Some(*dst), d)
                }
                IrOp::Atomic { dst, .. } => (*dst, Divergence::Divergent),
                IrOp::Selp { dst, a, b, pred, .. } => {
                    let p_div = div.get(*pred as usize).copied().unwrap_or(Divergence::Uniform);
                    let d = combine_div(
                        p_div,
                        combine_div(operand_div(a, &div), operand_div(b, &div)),
                    );
                    (Some(*dst), d)
                }
                IrOp::WarpShuffle { dst, .. } | IrOp::WarpVote { dst, .. } => {
                    (Some(*dst), Divergence::Divergent)
                }
                IrOp::Store { .. }
                | IrOp::Label(_)
                | IrOp::Branch(_)
                | IrOp::Barrier
                | IrOp::MemBar
                | IrOp::Return => (None, Divergence::Uniform),
            };

            if let Some(dst) = dst_opt {
                let final_div = combine_div(pred_div, rhs_div);
                if let Some(slot) = div.get_mut(dst as usize) {
                    if *slot != final_div && final_div == Divergence::Divergent {
                        *slot = Divergence::Divergent;
                        changed = true;
                    }
                }
            }
        }
    }

    div
}

fn operand_div(op: &IrOperand, reg_div: &[Divergence]) -> Divergence {
    match op {
        IrOperand::Reg(r) => reg_div.get(*r as usize).copied().unwrap_or(Divergence::Uniform),
        IrOperand::Special(sp) => match sp {
            SpecialReg::TidX | SpecialReg::TidY | SpecialReg::TidZ | SpecialReg::LaneId => {
                Divergence::Divergent
            }
            SpecialReg::CtaIdX
            | SpecialReg::CtaIdY
            | SpecialReg::CtaIdZ
            | SpecialReg::NtidX
            | SpecialReg::NtidY
            | SpecialReg::NtidZ
            | SpecialReg::NctaIdX
            | SpecialReg::NctaIdY
            | SpecialReg::NctaIdZ => Divergence::Uniform,
        },
        IrOperand::ImmU64(_)
        | IrOperand::ImmI64(_)
        | IrOperand::ImmF64(_)
        | IrOperand::Param(_)
        | IrOperand::SharedOffset(_) => Divergence::Uniform,
        IrOperand::LocalOffset(_) => Divergence::Divergent,
    }
}

const fn combine_div(a: Divergence, b: Divergence) -> Divergence {
    match (a, b) {
        (Divergence::Uniform, Divergence::Uniform) => Divergence::Uniform,
        _ => Divergence::Divergent,
    }
}

/// Allocates physical AMDGPU SGPRs and VGPRs for `kernel` using linear-scan liveness.
pub fn allocate_registers(
    kernel: &IrKernel,
    target: IsaTarget,
) -> Result<KernelAllocation, CompileError> {
    let divergence = analyze_divergence(kernel);
    let n_regs = kernel.reg_types.len();

    let mut first_use = vec![usize::MAX; n_regs];
    let mut last_use = vec![0usize; n_regs];

    let mut record = |reg: u32, pc: usize| {
        let idx = reg as usize;
        if idx < n_regs {
            first_use[idx] = first_use[idx].min(pc);
            last_use[idx] = last_use[idx].max(pc);
        }
    };

    let mut label_pcs = HashMap::new();
    for (pc, inst) in kernel.insts.iter().enumerate() {
        if let IrOp::Label(name) = &inst.op {
            label_pcs.insert(name.as_str(), pc);
        }
    }

    let mut back_edges = Vec::new();

    for (pc, inst) in kernel.insts.iter().enumerate() {
        if let Some(pg) = inst.pred {
            record(pg.reg, pc);
        }
        match &inst.op {
            IrOp::LoadParam { dst, .. } => record(*dst, pc),
            IrOp::Load { dst, base, .. } => {
                record(*dst, pc);
                record_op(base, pc, &mut record);
            }
            IrOp::Store { base, src, .. } => {
                record_op(base, pc, &mut record);
                record_op(src, pc, &mut record);
            }
            IrOp::CvtaToGlobal { dst, src }
            | IrOp::CvtaToShared { dst, src }
            | IrOp::Mov { dst, src, .. }
            | IrOp::Cvt { dst, src, .. }
            | IrOp::Neg { dst, src, .. }
            | IrOp::Abs { dst, src, .. }
            | IrOp::UnaryMath { dst, src, .. } => {
                record(*dst, pc);
                record_op(src, pc, &mut record);
            }
            IrOp::Bin { dst, lhs, rhs, .. } | IrOp::Setp { dst, lhs, rhs, .. } => {
                record(*dst, pc);
                record_op(lhs, pc, &mut record);
                record_op(rhs, pc, &mut record);
            }
            IrOp::Mad { dst, a, b, c, .. }
            | IrOp::Prmt { dst, a, b, sel: c }
            | IrOp::Dp4a { dst, a, b, c, .. } => {
                record(*dst, pc);
                record_op(a, pc, &mut record);
                record_op(b, pc, &mut record);
                record_op(c, pc, &mut record);
            }
            IrOp::Bfe { dst, src, pos, len, .. } => {
                record(*dst, pc);
                record_op(src, pc, &mut record);
                record_op(pos, pc, &mut record);
                record_op(len, pc, &mut record);
            }
            IrOp::Bfi {
                dst,
                insert,
                base,
                pos,
                len,
            } => {
                record(*dst, pc);
                record_op(insert, pc, &mut record);
                record_op(base, pc, &mut record);
                record_op(pos, pc, &mut record);
                record_op(len, pc, &mut record);
            }
            IrOp::Atomic {
                dst, base, val, cmp, ..
            } => {
                if let Some(d) = dst {
                    record(*d, pc);
                }
                record_op(base, pc, &mut record);
                record_op(val, pc, &mut record);
                if let Some(c) = cmp {
                    record_op(c, pc, &mut record);
                }
            }
            IrOp::Selp { dst, a, b, pred, .. } => {
                record(*dst, pc);
                record_op(a, pc, &mut record);
                record_op(b, pc, &mut record);
                record(*pred, pc);
            }
            IrOp::WarpShuffle {
                dst,
                src,
                lane_delta,
                ..
            } => {
                record(*dst, pc);
                record_op(src, pc, &mut record);
                record_op(lane_delta, pc, &mut record);
            }
            IrOp::WarpVote { dst, pred, .. } => {
                record(*dst, pc);
                record(*pred, pc);
            }
            IrOp::Branch(lbl) => {
                if let Some(&target_pc) = label_pcs.get(lbl.as_str()) {
                    if target_pc < pc {
                        back_edges.push((target_pc, pc));
                    }
                }
            }
            IrOp::Label(_) | IrOp::Barrier | IrOp::MemBar | IrOp::Return => {}
        }
    }

    for (target_pc, branch_pc) in back_edges {
        for idx in 0..n_regs {
            if first_use[idx] <= branch_pc && last_use[idx] >= target_pc {
                last_use[idx] = last_use[idx].max(branch_pc);
            }
        }
    }

    // Build sorted intervals (start, end, vreg_idx)
    let mut intervals: Vec<(usize, usize, usize)> = (0..n_regs)
        .map(|idx| {
            let s = if first_use[idx] == usize::MAX {
                0
            } else {
                first_use[idx]
            };
            let e = last_use[idx].max(s);
            (s, e, idx)
        })
        .collect();
    intervals.sort_by_key(|&(s, e, idx)| (s, e, idx));

    // ABI reservations:
    // SGPRs: s[0:1] dispatch_ptr, s[2:3] kernarg, s4..s6 workgroup_id_{x,y,z}, s7 unused, s[8:9] scratch
    // Allocatable SGPR pool starts at s10 (max 101 so VCC/EXEC remain untouched)
    // VGPRs: v0..v2 workitem_id_{x,y,z}, v3 unused (keeps scratch pair even-aligned), v4..v5 scratch
    // Allocatable VGPR pool starts at v6 (max 251)
    let mut sgpr_free_until = [0usize; 102];
    let mut vgpr_free_until = [0usize; 252];
    let mut reg_map = vec![PhysReg::Sgpr(10); n_regs];

    let mut max_sgpr_used: u8 = 9;
    let mut max_vgpr_used: u8 = 5;

    for (start, end, vreg) in intervals {
        let ir_ty = kernel.reg_types[vreg];
        let is_64 = matches!(ir_ty, IrType::U64 | IrType::I64 | IrType::F64);
        let is_pred = ir_ty == IrType::Pred;
        let is_divergent = divergence[vreg] == Divergence::Divergent;

        if is_pred && is_divergent {
            // Divergent predicates allocate an SGPR pair on Wave64 (or SGPR on Wave32) to store lane masks
            if target.native_wave_size() == 64 {
                let s = alloc_pair(&mut sgpr_free_until, 10, start, end + 1)?;
                max_sgpr_used = max_sgpr_used.max(s + 1);
                reg_map[vreg] = PhysReg::SgprPair(s);
            } else {
                let s = alloc_single(&mut sgpr_free_until, 10, start, end + 1)?;
                max_sgpr_used = max_sgpr_used.max(s);
                reg_map[vreg] = PhysReg::Sgpr(s);
            }
        } else if !is_divergent {
            if is_64 {
                let s = alloc_pair(&mut sgpr_free_until, 10, start, end + 1)?;
                max_sgpr_used = max_sgpr_used.max(s + 1);
                reg_map[vreg] = PhysReg::SgprPair(s);
            } else {
                let s = alloc_single(&mut sgpr_free_until, 10, start, end + 1)?;
                max_sgpr_used = max_sgpr_used.max(s);
                reg_map[vreg] = PhysReg::Sgpr(s);
            }
        } else if is_64 {
            let v = alloc_pair(&mut vgpr_free_until, 6, start, end + 1)?;
            max_vgpr_used = max_vgpr_used.max(v + 1);
            reg_map[vreg] = PhysReg::VgprPair(v);
        } else {
            let v = alloc_single(&mut vgpr_free_until, 6, start, end + 1)?;
            max_vgpr_used = max_vgpr_used.max(v);
            reg_map[vreg] = PhysReg::Vgpr(v);
        }
    }

    let sgpr_count = u32::from(max_sgpr_used) + 1;
    let vgpr_count = u32::from(max_vgpr_used) + 1;

    let uniform_ir_regs = divergence
        .iter()
        .filter(|&&d| d == Divergence::Uniform)
        .count();
    let divergent_ir_regs = divergence.len() - uniform_ir_regs;

    let estimated_waves_per_simd = if target.is_rdna() {
        let gran = vgpr_count.max(1).div_ceil(8) * 8;
        (1536 / gran).clamp(1, 16)
    } else {
        let gran = vgpr_count.max(1).div_ceil(4) * 4;
        (512 / gran).clamp(1, 8)
    };

    Ok(KernelAllocation {
        divergence,
        reg_map,
        sgpr_count,
        vgpr_count,
        uniform_ir_regs,
        divergent_ir_regs,
        estimated_waves_per_simd,
    })
}

fn record_op(op: &IrOperand, pc: usize, record: &mut impl FnMut(u32, usize)) {
    if let IrOperand::Reg(r) = op {
        record(*r, pc);
    }
}

fn alloc_single(
    free_until: &mut [usize],
    start_idx: usize,
    start_pc: usize,
    free_at: usize,
) -> Result<u8, CompileError> {
    for (idx, slot) in free_until.iter_mut().enumerate().skip(start_idx) {
        if *slot <= start_pc {
            *slot = free_at;
            return Ok(idx as u8);
        }
    }
    Err(CompileError::LoweringError(
        "Tier 0 linear-scan allocator ran out of physical registers".to_owned(),
    ))
}

fn alloc_pair(
    free_until: &mut [usize],
    start_idx: usize,
    start_pc: usize,
    free_at: usize,
) -> Result<u8, CompileError> {
    let first_even = if start_idx.is_multiple_of(2) {
        start_idx
    } else {
        start_idx + 1
    };
    let mut idx = first_even;
    while idx + 1 < free_until.len() {
        if free_until[idx] <= start_pc && free_until[idx + 1] <= start_pc {
            free_until[idx] = free_at;
            free_until[idx + 1] = free_at;
            return Ok(idx as u8);
        }
        idx += 2;
    }
    Err(CompileError::LoweringError(
        "Tier 0 linear-scan allocator ran out of physical register pairs".to_owned(),
    ))
}
