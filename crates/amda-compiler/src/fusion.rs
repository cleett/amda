//! Real cross-kernel IR fusion pass eliminating intermediate global memory round-trips.

use crate::error::CompileError;
use crate::ir::{
    AddressSpace, BinOp, IrInst, IrKernel, IrOp, IrOperand, IrParam, IrPredGuard, IrType,
    SpecialReg, WaveMode,
};
use amda_elf::{KernelArgMetadata, KernelMetadata, ValueKind};
use std::collections::{HashMap, HashSet};

/// Origin parameter index in the producer or consumer kernel before fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedParamSource {
    ProducerParam(u32),
    ConsumerParam(u32),
}

/// Complete output of fusing a producer elementwise kernel with a consumer elementwise kernel.
#[derive(Debug, Clone)]
pub struct FusedKernelPlan {
    pub kernel: IrKernel,
    pub param_sources: Vec<FusedParamSource>,
    pub eliminated_global_stores: usize,
    pub eliminated_global_loads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum AddrExpr {
    Unknown,
    Imm(i64),
    Param(u32),
    Special(u8),
    Add(Box<AddrExpr>, Box<AddrExpr>),
    Sub(Box<AddrExpr>, Box<AddrExpr>),
    Mul(Box<AddrExpr>, Box<AddrExpr>),
}

const fn is_integer_ir_ty(ty: IrType) -> bool {
    matches!(
        ty,
        IrType::U8
            | IrType::I8
            | IrType::U16
            | IrType::I16
            | IrType::U32
            | IrType::I32
            | IrType::U64
            | IrType::I64
    )
}

const fn special_tag(s: SpecialReg) -> u8 {
    s as u8
}

fn norm_add(a: AddrExpr, b: AddrExpr) -> AddrExpr {
    match (a, b) {
        (AddrExpr::Unknown, _) | (_, AddrExpr::Unknown) => AddrExpr::Unknown,
        (AddrExpr::Imm(0), x) | (x, AddrExpr::Imm(0)) => x,
        (AddrExpr::Imm(x), AddrExpr::Imm(y)) => AddrExpr::Imm(x.wrapping_add(y)),
        (l, r) => {
            if l <= r {
                AddrExpr::Add(Box::new(l), Box::new(r))
            } else {
                AddrExpr::Add(Box::new(r), Box::new(l))
            }
        }
    }
}

fn norm_sub(a: AddrExpr, b: AddrExpr) -> AddrExpr {
    match (a, b) {
        (AddrExpr::Unknown, _) | (_, AddrExpr::Unknown) => AddrExpr::Unknown,
        (x, AddrExpr::Imm(0)) => x,
        (AddrExpr::Imm(x), AddrExpr::Imm(y)) => AddrExpr::Imm(x.wrapping_sub(y)),
        (l, r) => AddrExpr::Sub(Box::new(l), Box::new(r)),
    }
}

fn norm_mul(a: AddrExpr, b: AddrExpr) -> AddrExpr {
    match (a, b) {
        (AddrExpr::Unknown, _) | (_, AddrExpr::Unknown) => AddrExpr::Unknown,
        (AddrExpr::Imm(0), _) | (_, AddrExpr::Imm(0)) => AddrExpr::Imm(0),
        (AddrExpr::Imm(1), x) | (x, AddrExpr::Imm(1)) => x,
        (AddrExpr::Imm(x), AddrExpr::Imm(y)) => AddrExpr::Imm(x.wrapping_mul(y)),
        (l, r) => {
            if l <= r {
                AddrExpr::Mul(Box::new(l), Box::new(r))
            } else {
                AddrExpr::Mul(Box::new(r), Box::new(l))
            }
        }
    }
}

fn eval_operand(op: &IrOperand, reg_expr: &HashMap<u32, AddrExpr>) -> AddrExpr {
    match op {
        IrOperand::Reg(r) => reg_expr.get(r).cloned().unwrap_or(AddrExpr::Unknown),
        IrOperand::Param(p) => AddrExpr::Param(*p),
        IrOperand::ImmU64(v) => AddrExpr::Imm(*v as i64),
        IrOperand::ImmI64(v) => AddrExpr::Imm(*v),
        IrOperand::Special(s) => AddrExpr::Special(special_tag(*s)),
        _ => AddrExpr::Unknown,
    }
}

fn contains_any_param(expr: &AddrExpr) -> bool {
    match expr {
        AddrExpr::Param(_) => true,
        AddrExpr::Add(a, b) | AddrExpr::Sub(a, b) | AddrExpr::Mul(a, b) => {
            contains_any_param(a) || contains_any_param(b)
        }
        AddrExpr::Unknown | AddrExpr::Imm(_) | AddrExpr::Special(_) => false,
    }
}

fn contains_special(expr: &AddrExpr) -> bool {
    match expr {
        AddrExpr::Special(_) => true,
        AddrExpr::Add(a, b) | AddrExpr::Sub(a, b) | AddrExpr::Mul(a, b) => {
            contains_special(a) || contains_special(b)
        }
        AddrExpr::Unknown | AddrExpr::Imm(_) | AddrExpr::Param(_) => false,
    }
}

fn strip_param(expr: &AddrExpr, target_param: u32) -> Option<AddrExpr> {
    match expr {
        AddrExpr::Unknown => None,
        AddrExpr::Param(p) if *p == target_param => Some(AddrExpr::Imm(0)),
        AddrExpr::Add(l, r) => {
            if !contains_any_param(r) && **r != AddrExpr::Unknown {
                strip_param(l, target_param).map(|rest| norm_add(rest, (**r).clone()))
            } else if !contains_any_param(l) && **l != AddrExpr::Unknown {
                strip_param(r, target_param).map(|rest| norm_add((**l).clone(), rest))
            } else {
                None
            }
        }
        AddrExpr::Sub(l, r) => {
            if !contains_any_param(r) && **r != AddrExpr::Unknown {
                strip_param(l, target_param).map(|rest| norm_sub(rest, (**r).clone()))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_relative_offset(addr: &AddrExpr, target_param: u32, const_off: i64) -> Option<AddrExpr> {
    let stripped = strip_param(addr, target_param)?;
    let rel = norm_add(stripped, AddrExpr::Imm(const_off));
    if rel == AddrExpr::Unknown || !contains_special(&rel) {
        None
    } else {
        Some(rel)
    }
}

#[derive(Debug, Clone)]
struct KernelAccessSummary {
    stores_by_param: HashMap<u32, Vec<(usize, IrType, Option<AddrExpr>)>>,
    loads_by_param: HashMap<u32, Vec<(usize, IrType, Option<AddrExpr>)>>,
    disallowed_ops: bool,
}

fn origin_of_operand(op: &IrOperand, reg_origin: &HashMap<u32, Option<u32>>) -> Option<u32> {
    match op {
        IrOperand::Reg(r) => reg_origin.get(r).copied().flatten(),
        IrOperand::Param(p) => Some(*p),
        _ => None,
    }
}

fn analyze_kernel_accesses(kernel: &IrKernel) -> KernelAccessSummary {
    let mut reg_origin: HashMap<u32, Option<u32>> = HashMap::new();
    let mut reg_expr: HashMap<u32, AddrExpr> = HashMap::new();
    let mut seen_labels: HashSet<String> = HashSet::new();
    let mut stores_by_param: HashMap<u32, Vec<(usize, IrType, Option<AddrExpr>)>> = HashMap::new();
    let mut loads_by_param: HashMap<u32, Vec<(usize, IrType, Option<AddrExpr>)>> = HashMap::new();
    let mut disallowed_ops = kernel.shared_bytes > 0 || kernel.private_bytes > 0;

    for (idx, inst) in kernel.insts.iter().enumerate() {
        match &inst.op {
            IrOp::LoadParam { dst, param_idx, .. } => {
                reg_origin.insert(*dst, Some(*param_idx));
                let expr = if kernel
                    .params
                    .get(*param_idx as usize)
                    .is_some_and(|p| p.is_global_ptr)
                {
                    AddrExpr::Param(*param_idx)
                } else {
                    AddrExpr::Unknown
                };
                reg_expr.insert(*dst, expr);
            }
            IrOp::CvtaToGlobal { dst, src } | IrOp::Mov { dst, src, .. } => {
                let orig = origin_of_operand(src, &reg_origin);
                reg_origin.insert(*dst, orig);
                let expr = eval_operand(src, &reg_expr);
                reg_expr.insert(*dst, expr);
            }
            IrOp::Cvt {
                dst,
                dst_ty,
                src_ty,
                src,
            } => {
                reg_origin.remove(dst);
                if is_integer_ir_ty(*dst_ty) && is_integer_ir_ty(*src_ty) {
                    let expr = eval_operand(src, &reg_expr);
                    reg_expr.insert(*dst, expr);
                } else {
                    reg_expr.insert(*dst, AddrExpr::Unknown);
                }
            }
            IrOp::Bin {
                op,
                dst,
                lhs,
                rhs,
                ..
            } => {
                let orig = if matches!(op, BinOp::Add | BinOp::Sub) {
                    match (
                        origin_of_operand(lhs, &reg_origin),
                        origin_of_operand(rhs, &reg_origin),
                    ) {
                        (Some(p), None) | (None, Some(p)) => Some(p),
                        (Some(a), Some(b)) if a == b => Some(a),
                        _ => None,
                    }
                } else {
                    None
                };
                reg_origin.insert(*dst, orig);

                let l = eval_operand(lhs, &reg_expr);
                let r = eval_operand(rhs, &reg_expr);
                let expr = match op {
                    BinOp::Add => norm_add(l, r),
                    BinOp::Sub => norm_sub(l, r),
                    BinOp::Mul | BinOp::MulWide => norm_mul(l, r),
                    BinOp::Shl => {
                        if let AddrExpr::Imm(k) = r {
                            if (0..63).contains(&k) {
                                norm_mul(l, AddrExpr::Imm(1i64 << k))
                            } else {
                                AddrExpr::Unknown
                            }
                        } else {
                            AddrExpr::Unknown
                        }
                    }
                    _ => AddrExpr::Unknown,
                };
                reg_expr.insert(*dst, expr);
            }
            IrOp::Mad { dst, a, b, c, .. } => {
                reg_origin.remove(dst);
                let ea = eval_operand(a, &reg_expr);
                let eb = eval_operand(b, &reg_expr);
                let ec = eval_operand(c, &reg_expr);
                reg_expr.insert(*dst, norm_add(norm_mul(ea, eb), ec));
            }
            IrOp::Selp { dst, a, b, .. } => {
                let orig = match (
                    origin_of_operand(a, &reg_origin),
                    origin_of_operand(b, &reg_origin),
                ) {
                    (Some(pa), Some(pb)) if pa == pb => Some(pa),
                    _ => None,
                };
                reg_origin.insert(*dst, orig);
                reg_expr.insert(*dst, AddrExpr::Unknown);
            }
            IrOp::Load {
                dst,
                ty,
                space,
                base,
                offset,
            } => {
                if *space == AddressSpace::Shared || *space == AddressSpace::Private {
                    disallowed_ops = true;
                }
                if matches!(space, AddressSpace::Global | AddressSpace::Generic) {
                    if let Some(pidx) = origin_of_operand(base, &reg_origin) {
                        let base_expr = eval_operand(base, &reg_expr);
                        let rel = extract_relative_offset(&base_expr, pidx, *offset);
                        loads_by_param
                            .entry(pidx)
                            .or_default()
                            .push((idx, *ty, rel));
                    }
                }
                reg_origin.remove(dst);
                reg_expr.insert(*dst, AddrExpr::Unknown);
            }
            IrOp::Store {
                ty,
                space,
                base,
                offset,
                ..
            } => {
                if *space == AddressSpace::Shared || *space == AddressSpace::Private {
                    disallowed_ops = true;
                }
                if matches!(space, AddressSpace::Global | AddressSpace::Generic) {
                    if let Some(pidx) = origin_of_operand(base, &reg_origin) {
                        let base_expr = eval_operand(base, &reg_expr);
                        let rel = extract_relative_offset(&base_expr, pidx, *offset);
                        stores_by_param
                            .entry(pidx)
                            .or_default()
                            .push((idx, *ty, rel));
                    }
                }
            }
            IrOp::Label(lbl) => {
                seen_labels.insert(lbl.clone());
            }
            IrOp::Branch(lbl) => {
                if seen_labels.contains(lbl) {
                    disallowed_ops = true;
                }
            }
            IrOp::Barrier
            | IrOp::Atomic { .. }
            | IrOp::WarpShuffle { .. }
            | IrOp::WarpVote { .. } => {
                disallowed_ops = true;
            }
            IrOp::CvtaToShared { dst, .. } => {
                disallowed_ops = true;
                reg_origin.remove(dst);
                reg_expr.insert(*dst, AddrExpr::Unknown);
            }
            IrOp::Neg { dst, .. }
            | IrOp::Abs { dst, .. }
            | IrOp::Setp { dst, .. }
            | IrOp::UnaryMath { dst, .. }
            | IrOp::Bfe { dst, .. }
            | IrOp::Bfi { dst, .. }
            | IrOp::Prmt { dst, .. }
            | IrOp::Dp4a { dst, .. } => {
                reg_origin.remove(dst);
                reg_expr.insert(*dst, AddrExpr::Unknown);
            }
            IrOp::MemBar | IrOp::Return => {}
        }
    }

    KernelAccessSummary {
        stores_by_param,
        loads_by_param,
        disallowed_ops,
    }
}

fn read_global_ptr_from_kernargs(kernel: &IrKernel, param_idx: u32, kernargs: &[u8]) -> Option<u64> {
    let p = kernel.params.get(param_idx as usize)?;
    if !p.is_global_ptr || p.size != 8 {
        return None;
    }
    let start = p.offset as usize;
    let slice = kernargs.get(start..start + 8)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn has_other_aliased_param(
    kernel: &IrKernel,
    kernargs: &[u8],
    exclude_param: u32,
    target_ptr: u64,
) -> bool {
    kernel.params.iter().enumerate().any(|(i, p)| {
        let idx = i as u32;
        idx != exclude_param
            && p.is_global_ptr
            && read_global_ptr_from_kernargs(kernel, idx, kernargs) == Some(target_ptr)
    })
}

/// Identifies a single intermediate global buffer `(prod_out_param, cons_in_param, device_ptr)`
/// suitable for eliminating its store/load round-trip via cross-kernel fusion.
pub fn find_fusible_buffer_params(
    producer: &IrKernel,
    consumer: &IrKernel,
    prod_kernargs: &[u8],
    cons_kernargs: &[u8],
) -> Option<(u32, u32, u64)> {
    let prod_acc = analyze_kernel_accesses(producer);
    let cons_acc = analyze_kernel_accesses(consumer);
    if prod_acc.disallowed_ops || cons_acc.disallowed_ops {
        return None;
    }

    for (pidx, p_param) in producer.params.iter().enumerate() {
        if !p_param.is_global_ptr {
            continue;
        }
        let pidx_u32 = pidx as u32;
        let Some(stores) = prod_acc.stores_by_param.get(&pidx_u32) else {
            continue;
        };
        if stores.len() != 1 || prod_acc.loads_by_param.contains_key(&pidx_u32) {
            continue;
        }
        let Some(ref prod_off) = stores[0].2 else {
            continue;
        };
        let Some(ptr) = read_global_ptr_from_kernargs(producer, pidx_u32, prod_kernargs) else {
            continue;
        };
        if ptr == 0 || has_other_aliased_param(producer, prod_kernargs, pidx_u32, ptr) {
            continue;
        }

        for (cidx, c_param) in consumer.params.iter().enumerate() {
            if !c_param.is_global_ptr {
                continue;
            }
            let cidx_u32 = cidx as u32;
            let Some(c_ptr) = read_global_ptr_from_kernargs(consumer, cidx_u32, cons_kernargs)
            else {
                continue;
            };
            if c_ptr != ptr || has_other_aliased_param(consumer, cons_kernargs, cidx_u32, ptr) {
                continue;
            }
            let Some(loads) = cons_acc.loads_by_param.get(&cidx_u32) else {
                continue;
            };
            if loads.is_empty() || cons_acc.stores_by_param.contains_key(&cidx_u32) {
                continue;
            }
            if loads
                .iter()
                .all(|(_, ty, off)| *ty == stores[0].1 && off.as_ref() == Some(prod_off))
            {
                return Some((pidx_u32, cidx_u32, ptr));
            }
        }
    }
    None
}

fn shift_operand_regs(op: &IrOperand, reg_offset: u32, param_remap: &HashMap<u32, u32>) -> IrOperand {
    match op {
        IrOperand::Reg(r) => IrOperand::Reg(*r + reg_offset),
        IrOperand::Param(p) => IrOperand::Param(*param_remap.get(p).unwrap_or(p)),
        IrOperand::ImmU64(v) => IrOperand::ImmU64(*v),
        IrOperand::ImmI64(v) => IrOperand::ImmI64(*v),
        IrOperand::ImmF64(v) => IrOperand::ImmF64(*v),
        IrOperand::Special(s) => IrOperand::Special(*s),
        IrOperand::SharedOffset(o) => IrOperand::SharedOffset(*o),
        IrOperand::LocalOffset(o) => IrOperand::LocalOffset(*o),
    }
}

fn remap_inst(
    inst: &IrInst,
    reg_offset: u32,
    param_remap: &HashMap<u32, u32>,
    label_prefix: &str,
    return_target: &str,
) -> IrInst {
    let pred = inst.pred.map(|pg| IrPredGuard {
        reg: pg.reg + reg_offset,
        negated: pg.negated,
    });
    let op = match &inst.op {
        IrOp::LoadParam {
            dst,
            ty,
            param_idx,
            offset,
        } => {
            if let Some(&mapped) = param_remap.get(param_idx) {
                IrOp::LoadParam {
                    dst: *dst + reg_offset,
                    ty: *ty,
                    param_idx: mapped,
                    offset: *offset,
                }
            } else {
                IrOp::Mov {
                    dst: *dst + reg_offset,
                    ty: *ty,
                    src: IrOperand::ImmU64(0),
                }
            }
        }
        IrOp::Load {
            dst,
            ty,
            space,
            base,
            offset,
        } => IrOp::Load {
            dst: *dst + reg_offset,
            ty: *ty,
            space: *space,
            base: shift_operand_regs(base, reg_offset, param_remap),
            offset: *offset,
        },
        IrOp::Store {
            ty,
            space,
            base,
            offset,
            src,
        } => IrOp::Store {
            ty: *ty,
            space: *space,
            base: shift_operand_regs(base, reg_offset, param_remap),
            offset: *offset,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::CvtaToGlobal { dst, src } => IrOp::CvtaToGlobal {
            dst: *dst + reg_offset,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::CvtaToShared { dst, src } => IrOp::CvtaToShared {
            dst: *dst + reg_offset,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Mov { dst, ty, src } => IrOp::Mov {
            dst: *dst + reg_offset,
            ty: *ty,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Cvt {
            dst,
            dst_ty,
            src_ty,
            src,
        } => IrOp::Cvt {
            dst: *dst + reg_offset,
            dst_ty: *dst_ty,
            src_ty: *src_ty,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Neg { dst, ty, src } => IrOp::Neg {
            dst: *dst + reg_offset,
            ty: *ty,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Abs { dst, ty, src } => IrOp::Abs {
            dst: *dst + reg_offset,
            ty: *ty,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Bin {
            op,
            dst,
            ty,
            lhs,
            rhs,
        } => IrOp::Bin {
            op: *op,
            dst: *dst + reg_offset,
            ty: *ty,
            lhs: shift_operand_regs(lhs, reg_offset, param_remap),
            rhs: shift_operand_regs(rhs, reg_offset, param_remap),
        },
        IrOp::Mad { dst, ty, a, b, c } => IrOp::Mad {
            dst: *dst + reg_offset,
            ty: *ty,
            a: shift_operand_regs(a, reg_offset, param_remap),
            b: shift_operand_regs(b, reg_offset, param_remap),
            c: shift_operand_regs(c, reg_offset, param_remap),
        },
        IrOp::Setp {
            cmp,
            dst,
            ty,
            lhs,
            rhs,
        } => IrOp::Setp {
            cmp: *cmp,
            dst: *dst + reg_offset,
            ty: *ty,
            lhs: shift_operand_regs(lhs, reg_offset, param_remap),
            rhs: shift_operand_regs(rhs, reg_offset, param_remap),
        },
        IrOp::Selp {
            dst,
            ty,
            a,
            b,
            pred,
        } => IrOp::Selp {
            dst: *dst + reg_offset,
            ty: *ty,
            a: shift_operand_regs(a, reg_offset, param_remap),
            b: shift_operand_regs(b, reg_offset, param_remap),
            pred: *pred + reg_offset,
        },
        IrOp::Label(lbl) => IrOp::Label(format!("{label_prefix}{lbl}")),
        IrOp::Branch(lbl) => IrOp::Branch(format!("{label_prefix}{lbl}")),
        IrOp::Barrier => IrOp::Barrier,
        IrOp::MemBar => IrOp::MemBar,
        IrOp::UnaryMath { op, dst, ty, src } => IrOp::UnaryMath {
            op: *op,
            dst: *dst + reg_offset,
            ty: *ty,
            src: shift_operand_regs(src, reg_offset, param_remap),
        },
        IrOp::Bfe {
            dst,
            signed,
            src,
            pos,
            len,
        } => IrOp::Bfe {
            dst: *dst + reg_offset,
            signed: *signed,
            src: shift_operand_regs(src, reg_offset, param_remap),
            pos: shift_operand_regs(pos, reg_offset, param_remap),
            len: shift_operand_regs(len, reg_offset, param_remap),
        },
        IrOp::Bfi {
            dst,
            insert,
            base,
            pos,
            len,
        } => IrOp::Bfi {
            dst: *dst + reg_offset,
            insert: shift_operand_regs(insert, reg_offset, param_remap),
            base: shift_operand_regs(base, reg_offset, param_remap),
            pos: shift_operand_regs(pos, reg_offset, param_remap),
            len: shift_operand_regs(len, reg_offset, param_remap),
        },
        IrOp::Prmt { dst, a, b, sel } => IrOp::Prmt {
            dst: *dst + reg_offset,
            a: shift_operand_regs(a, reg_offset, param_remap),
            b: shift_operand_regs(b, reg_offset, param_remap),
            sel: shift_operand_regs(sel, reg_offset, param_remap),
        },
        IrOp::Dp4a {
            dst,
            a_signed,
            b_signed,
            a,
            b,
            c,
        } => IrOp::Dp4a {
            dst: *dst + reg_offset,
            a_signed: *a_signed,
            b_signed: *b_signed,
            a: shift_operand_regs(a, reg_offset, param_remap),
            b: shift_operand_regs(b, reg_offset, param_remap),
            c: shift_operand_regs(c, reg_offset, param_remap),
        },
        IrOp::Atomic {
            op,
            dst,
            ty,
            space,
            base,
            offset,
            val,
            cmp,
        } => IrOp::Atomic {
            op: *op,
            dst: dst.map(|d| d + reg_offset),
            ty: *ty,
            space: *space,
            base: shift_operand_regs(base, reg_offset, param_remap),
            offset: *offset,
            val: shift_operand_regs(val, reg_offset, param_remap),
            cmp: cmp
                .as_ref()
                .map(|c| shift_operand_regs(c, reg_offset, param_remap)),
        },
        IrOp::WarpShuffle {
            dst,
            mode,
            src,
            lane_delta,
            width,
        } => IrOp::WarpShuffle {
            dst: *dst + reg_offset,
            mode: *mode,
            src: shift_operand_regs(src, reg_offset, param_remap),
            lane_delta: shift_operand_regs(lane_delta, reg_offset, param_remap),
            width: *width,
        },
        IrOp::WarpVote {
            mode,
            dst,
            pred: vpred,
            negated,
        } => IrOp::WarpVote {
            mode: *mode,
            dst: *dst + reg_offset,
            pred: *vpred + reg_offset,
            negated: *negated,
        },
        IrOp::Return => IrOp::Branch(return_target.to_owned()),
    };
    IrInst { pred, op }
}

const fn align_up(val: u32, align: u32) -> u32 {
    if align <= 1 {
        val
    } else {
        (val + align - 1) & !(align - 1)
    }
}

/// Fuses two compatible elementwise kernels (`producer` -> `consumer`), eliminating
/// the intermediate global memory store/load round-trip via an SSA bridge register.
pub fn fuse_elementwise_kernels(
    producer: &IrKernel,
    consumer: &IrKernel,
    producer_out_param: u32,
    consumer_in_param: u32,
) -> Result<FusedKernelPlan, CompileError> {
    if (producer_out_param as usize) >= producer.params.len() {
        return Err(CompileError::LoweringError(
            "producer_out_param out of range".to_owned(),
        ));
    }
    if (consumer_in_param as usize) >= consumer.params.len() {
        return Err(CompileError::LoweringError(
            "consumer_in_param out of range".to_owned(),
        ));
    }

    let prod_acc = analyze_kernel_accesses(producer);
    let cons_acc = analyze_kernel_accesses(consumer);
    if prod_acc.disallowed_ops || cons_acc.disallowed_ops {
        return Err(CompileError::LoweringError(
            "fusion rejected: kernels with shared/local/barrier/atomic/warp ops are not fusible"
                .to_owned(),
        ));
    }
    if prod_acc.loads_by_param.contains_key(&producer_out_param) {
        return Err(CompileError::LoweringError(
            "fusion rejected: producer reads from producer_out_param".to_owned(),
        ));
    }
    let prod_stores = prod_acc
        .stores_by_param
        .get(&producer_out_param)
        .ok_or_else(|| {
            CompileError::LoweringError(
                "fusion rejected: producer never stores to producer_out_param".to_owned(),
            )
        })?;
    if prod_stores.len() != 1 {
        return Err(CompileError::LoweringError(
            "fusion rejected: producer must have exactly one store to producer_out_param"
                .to_owned(),
        ));
    }

    if cons_acc.stores_by_param.contains_key(&consumer_in_param) {
        return Err(CompileError::LoweringError(
            "fusion rejected: consumer writes back to consumer_in_param".to_owned(),
        ));
    }
    let cons_loads = cons_acc
        .loads_by_param
        .get(&consumer_in_param)
        .ok_or_else(|| {
            CompileError::LoweringError(
                "fusion rejected: consumer never loads from consumer_in_param".to_owned(),
            )
        })?;

    let (prod_store_idx, prod_val_ty, prod_off_opt) = &prod_stores[0];
    let prod_store_idx = *prod_store_idx;
    let prod_val_ty = *prod_val_ty;
    let prod_off = prod_off_opt.as_ref().ok_or_else(|| {
        CompileError::LoweringError(
            "fusion rejected: producer store address is not statically thread-indexed".to_owned(),
        )
    })?;
    if cons_loads
        .iter()
        .any(|(_, lt, loff)| *lt != prod_val_ty || loff.as_ref() != Some(prod_off))
    {
        return Err(CompileError::LoweringError(
            "fusion rejected: producer store and consumer load types or index expressions differ"
                .to_owned(),
        ));
    }

    let mut prod_param_remap = HashMap::new();
    let mut cons_param_remap = HashMap::new();
    let mut fused_params = Vec::new();
    let mut arg_metas = Vec::new();
    let mut param_sources = Vec::new();
    let mut offset = 0u32;

    for (idx, p) in producer.params.iter().enumerate() {
        let idx_u32 = idx as u32;
        if idx_u32 == producer_out_param {
            continue;
        }
        let fused_idx = fused_params.len() as u32;
        prod_param_remap.insert(idx_u32, fused_idx);
        param_sources.push(FusedParamSource::ProducerParam(idx_u32));

        let align = p.size.clamp(1, 8);
        offset = align_up(offset, align);
        let name = format!("p{idx_u32}_{}", p.name);
        fused_params.push(IrParam {
            name: name.clone(),
            ty: p.ty,
            offset,
            size: p.size,
            is_global_ptr: p.is_global_ptr,
        });
        arg_metas.push(KernelArgMetadata {
            name,
            offset,
            size: p.size,
            value_kind: if p.is_global_ptr {
                ValueKind::GlobalBuffer
            } else {
                ValueKind::ByValue
            },
        });
        offset += p.size;
    }

    for (idx, p) in consumer.params.iter().enumerate() {
        let idx_u32 = idx as u32;
        if idx_u32 == consumer_in_param {
            continue;
        }
        let fused_idx = fused_params.len() as u32;
        cons_param_remap.insert(idx_u32, fused_idx);
        param_sources.push(FusedParamSource::ConsumerParam(idx_u32));

        let align = p.size.clamp(1, 8);
        offset = align_up(offset, align);
        let name = format!("c{idx_u32}_{}", p.name);
        fused_params.push(IrParam {
            name: name.clone(),
            ty: p.ty,
            offset,
            size: p.size,
            is_global_ptr: p.is_global_ptr,
        });
        arg_metas.push(KernelArgMetadata {
            name,
            offset,
            size: p.size,
            value_kind: if p.is_global_ptr {
                ValueKind::GlobalBuffer
            } else {
                ValueKind::ByValue
            },
        });
        offset += p.size;
    }

    let kernarg_segment_size = align_up(offset, 8);
    let fused_name = format!("{}_fused_{}", producer.name, consumer.name);

    // Virtual registers: [producer regs] + [v_bridge, v_prod_wrote] + [consumer regs shifted]
    let v_bridge = producer.reg_types.len() as u32;
    let v_prod_wrote = v_bridge + 1;
    let cons_reg_offset = v_bridge + 2;

    let mut reg_types = producer.reg_types.clone();
    reg_types.push(prod_val_ty);
    reg_types.push(IrType::Pred);
    reg_types.extend_from_slice(&consumer.reg_types);

    let cons_load_set: HashMap<usize, IrType> = cons_loads
        .iter()
        .map(|(idx, ty, _)| (*idx, *ty))
        .collect();
    let mut fused_insts =
        Vec::with_capacity(producer.insts.len() + consumer.insts.len() + 8);

    fused_insts.push(IrInst {
        pred: None,
        op: IrOp::Mov {
            dst: v_prod_wrote,
            ty: IrType::Pred,
            src: IrOperand::ImmU64(0),
        },
    });

    for (idx, inst) in producer.insts.iter().enumerate() {
        if idx == prod_store_idx {
            if let IrOp::Store { ty, src, .. } = &inst.op {
                let mapped_src = shift_operand_regs(src, 0, &prod_param_remap);
                fused_insts.push(IrInst {
                    pred: inst.pred,
                    op: IrOp::Mov {
                        dst: v_bridge,
                        ty: *ty,
                        src: mapped_src,
                    },
                });
                fused_insts.push(IrInst {
                    pred: inst.pred,
                    op: IrOp::Mov {
                        dst: v_prod_wrote,
                        ty: IrType::Pred,
                        src: IrOperand::ImmU64(1),
                    },
                });
                continue;
            }
        }
        fused_insts.push(remap_inst(inst, 0, &prod_param_remap, "__p_", "__fuse_join"));
    }

    fused_insts.push(IrInst {
        pred: None,
        op: IrOp::Label("__fuse_join".to_owned()),
    });
    fused_insts.push(IrInst {
        pred: Some(IrPredGuard {
            reg: v_prod_wrote,
            negated: true,
        }),
        op: IrOp::Branch("__fuse_exit".to_owned()),
    });

    for (idx, inst) in consumer.insts.iter().enumerate() {
        if let Some(&load_ty) = cons_load_set.get(&idx) {
            if let IrOp::Load { dst, .. } = &inst.op {
                fused_insts.push(IrInst {
                    pred: inst.pred.map(|pg| IrPredGuard {
                        reg: pg.reg + cons_reg_offset,
                        negated: pg.negated,
                    }),
                    op: IrOp::Mov {
                        dst: *dst + cons_reg_offset,
                        ty: load_ty,
                        src: IrOperand::Reg(v_bridge),
                    },
                });
                continue;
            }
        }
        fused_insts.push(remap_inst(
            inst,
            cons_reg_offset,
            &cons_param_remap,
            "__c_",
            "__fuse_exit",
        ));
    }

    fused_insts.push(IrInst {
        pred: None,
        op: IrOp::Label("__fuse_exit".to_owned()),
    });
    fused_insts.push(IrInst {
        pred: None,
        op: IrOp::Return,
    });

    let wave_mode = if producer.wave_mode == WaveMode::EmulatedWave32OnWave64
        || consumer.wave_mode == WaveMode::EmulatedWave32OnWave64
    {
        WaveMode::EmulatedWave32OnWave64
    } else {
        producer.wave_mode
    };

    let metadata = KernelMetadata {
        name: fused_name.clone(),
        symbol: format!("{fused_name}.kd"),
        kernarg_segment_size,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        wavefront_size: producer.metadata.wavefront_size,
        max_flat_workgroup_size: producer
            .metadata
            .max_flat_workgroup_size
            .min(consumer.metadata.max_flat_workgroup_size),
        args: arg_metas,
    };

    Ok(FusedKernelPlan {
        kernel: IrKernel {
            name: fused_name,
            params: fused_params,
            reg_types,
            shared_bytes: 0,
            private_bytes: 0,
            wave_mode,
            insts: fused_insts,
            metadata,
        },
        param_sources,
        eliminated_global_stores: 1,
        eliminated_global_loads: cons_loads.len(),
    })
}

/// Packs merged kernarg bytes for a fused kernel from the original producer & consumer kernargs.
pub fn pack_fused_kernargs(
    plan: &FusedKernelPlan,
    producer: &IrKernel,
    consumer: &IrKernel,
    prod_kernargs: &[u8],
    cons_kernargs: &[u8],
) -> Result<Vec<u8>, CompileError> {
    let mut buf = vec![0u8; plan.kernel.metadata.kernarg_segment_size as usize];
    for (fused_idx, &src) in plan.param_sources.iter().enumerate() {
        let dst_arg = plan.kernel.metadata.args.get(fused_idx).ok_or_else(|| {
            CompileError::LoweringError("fused metadata arg count mismatch".to_owned())
        })?;
        let (src_param, src_bytes) = match src {
            FusedParamSource::ProducerParam(i) => (
                producer.params.get(i as usize).ok_or_else(|| {
                    CompileError::LoweringError("producer param index out of bounds".to_owned())
                })?,
                prod_kernargs,
            ),
            FusedParamSource::ConsumerParam(i) => (
                consumer.params.get(i as usize).ok_or_else(|| {
                    CompileError::LoweringError("consumer param index out of bounds".to_owned())
                })?,
                cons_kernargs,
            ),
        };
        let s_start = src_param.offset as usize;
        let s_end = s_start + src_param.size as usize;
        let slice = src_bytes.get(s_start..s_end).ok_or_else(|| {
            CompileError::LoweringError("source kernarg slice out of bounds".to_owned())
        })?;
        let d_start = dst_arg.offset as usize;
        let d_end = d_start + dst_arg.size as usize;
        buf[d_start..d_end].copy_from_slice(slice);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compile_tier0_native, emit_llvm_ir, lower_ptx_to_ir, parse_ptx, validate_llvm_ir,
        verify_encoded_machine_code, IrModule,
    };
    use amda_elf::IsaTarget;

    const VADD_PTX: &str = r#"
        .version 7.8
        .target sm_80
        .address_size 64
        .visible .entry vadd(
            .param .u64 a,
            .param .u64 b,
            .param .u64 tmp,
            .param .u32 n
        ) {
            .reg .pred %p<2>;
            .reg .f32 %f<4>;
            .reg .b32 %r<6>;
            .reg .b64 %rd<10>;
            ld.param.u64 %rd1, [a];
            ld.param.u64 %rd2, [b];
            ld.param.u64 %rd3, [tmp];
            ld.param.u32 %r1, [n];
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra $L_EXIT;
            cvta.to.global.u64 %rd4, %rd1;
            cvta.to.global.u64 %rd5, %rd2;
            cvta.to.global.u64 %rd6, %rd3;
            mul.wide.s32 %rd7, %r5, 4;
            add.s64 %rd8, %rd4, %rd7;
            add.s64 %rd9, %rd5, %rd7;
            ld.global.f32 %f1, [%rd8];
            ld.global.f32 %f2, [%rd9];
            add.f32 %f3, %f1, %f2;
            add.s64 %rd8, %rd6, %rd7;
            st.global.f32 [%rd8], %f3;
        $L_EXIT:
            ret;
        }
    "#;

    const SCALE_BIAS_PTX: &str = r#"
        .version 7.8
        .target sm_80
        .address_size 64
        .visible .entry scale_bias(
            .param .u64 tmp,
            .param .u64 out,
            .param .f32 scale,
            .param .f32 bias,
            .param .u32 n
        ) {
            .reg .pred %p<2>;
            .reg .f32 %f<5>;
            .reg .b32 %r<6>;
            .reg .b64 %rd<8>;
            ld.param.u64 %rd1, [tmp];
            ld.param.u64 %rd2, [out];
            ld.param.f32 %f1, [scale];
            ld.param.f32 %f2, [bias];
            ld.param.u32 %r1, [n];
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra $L_EXIT;
            cvta.to.global.u64 %rd3, %rd1;
            cvta.to.global.u64 %rd4, %rd2;
            mul.wide.s32 %rd5, %r5, 4;
            add.s64 %rd6, %rd3, %rd5;
            ld.global.f32 %f3, [%rd6];
            fma.rn.f32 %f4, %f3, %f1, %f2;
            add.s64 %rd7, %rd4, %rd5;
            st.global.f32 [%rd7], %f4;
        $L_EXIT:
            ret;
        }
    "#;

    const SHARED_PTX: &str = r#"
        .version 7.8
        .target sm_80
        .address_size 64
        .visible .entry uses_smem(.param .u64 a, .param .u64 b) {
            .shared .b8 sm[64];
            .reg .f32 %f<2>;
            .reg .b32 %r<2>;
            .reg .b64 %rd<3>;
            ld.param.u64 %rd1, [a];
            cvta.to.global.u64 %rd2, %rd1;
            ld.global.f32 %f1, [%rd2];
            mov.u32 %r1, sm;
            st.shared.f32 [%r1], %f1;
            bar.sync 0;
            ret;
        }
    "#;

    #[test]
    fn fuse_elementwise_kernels_compiles_for_all_targets() {
        let p_mod = parse_ptx(VADD_PTX).expect("parse producer");
        let c_mod = parse_ptx(SCALE_BIAS_PTX).expect("parse consumer");

        for target in [IsaTarget::Gfx1100, IsaTarget::Gfx1103, IsaTarget::Gfx942] {
            let p_ir = lower_ptx_to_ir(&p_mod, target).expect("lower producer");
            let c_ir = lower_ptx_to_ir(&c_mod, target).expect("lower consumer");

            let plan = fuse_elementwise_kernels(&p_ir.kernels[0], &c_ir.kernels[0], 2, 0)
                .expect("fusion should succeed");
            assert_eq!(plan.eliminated_global_stores, 1);
            assert_eq!(plan.eliminated_global_loads, 1);
            assert_eq!(plan.kernel.params.len(), 7);

            let fused_mod = IrModule {
                target,
                kernels: vec![plan.kernel.clone()],
            };
            let llvm = emit_llvm_ir(&fused_mod, target);
            validate_llvm_ir(&llvm).expect("fused LLVM IR validates");

            let native = compile_tier0_native(&fused_mod, target).expect("fused Tier 0 compiles");
            let decoded = verify_encoded_machine_code(&native.kernels[0].machine_code, target)
                .expect("fused ISA decodes");
            assert!(!decoded.is_empty());
        }
    }

    #[test]
    fn rejects_non_fusible_shared_memory_kernels() {
        let s_mod = parse_ptx(SHARED_PTX).expect("parse shared");
        let c_mod = parse_ptx(SCALE_BIAS_PTX).expect("parse consumer");
        let s_ir = lower_ptx_to_ir(&s_mod, IsaTarget::Gfx1100).expect("lower shared");
        let c_ir = lower_ptx_to_ir(&c_mod, IsaTarget::Gfx1100).expect("lower consumer");
        assert!(fuse_elementwise_kernels(&s_ir.kernels[0], &c_ir.kernels[0], 1, 0).is_err());
    }

    #[test]
    fn rejects_looped_kernels() {
        const LOOPED_CONSUMER_PTX: &str = r#"
            .version 7.8
            .target sm_80
            .address_size 64
            .visible .entry looped_cons(
                .param .u64 tmp,
                .param .u64 out,
                .param .u32 n
            ) {
                .reg .pred %p<2>;
                .reg .f32 %f<3>;
                .reg .b32 %r<5>;
                .reg .b64 %rd<7>;
                ld.param.u64 %rd1, [tmp];
                ld.param.u64 %rd2, [out];
                ld.param.u32 %r1, [n];
                mov.u32 %r2, %ctaid.x;
                mov.u32 %r3, %ntid.x;
                mov.u32 %r4, %tid.x;
                mad.lo.s32 %r0, %r2, %r3, %r4;
                setp.ge.s32 %p1, %r0, %r1;
                @%p1 bra $L_EXIT;
                cvta.to.global.u64 %rd3, %rd1;
                cvta.to.global.u64 %rd4, %rd2;
                mul.wide.s32 %rd5, %r0, 4;
                add.s64 %rd6, %rd3, %rd5;
                mov.f32 %f1, 0f00000000;
            $L_LOOP:
                ld.global.f32 %f2, [%rd6];
                add.f32 %f1, %f1, %f2;
                sub.s32 %r1, %r1, 1;
                setp.gt.s32 %p1, %r1, 0;
                @%p1 bra $L_LOOP;
                add.s64 %rd6, %rd4, %rd5;
                st.global.f32 [%rd6], %f1;
            $L_EXIT:
                ret;
            }
        "#;
        let p_mod = parse_ptx(VADD_PTX).expect("parse producer");
        let l_mod = parse_ptx(LOOPED_CONSUMER_PTX).expect("parse looped");
        let p_ir = lower_ptx_to_ir(&p_mod, IsaTarget::Gfx1100).expect("lower producer");
        let l_ir = lower_ptx_to_ir(&l_mod, IsaTarget::Gfx1100).expect("lower looped");
        assert!(fuse_elementwise_kernels(&p_ir.kernels[0], &l_ir.kernels[0], 2, 0).is_err());
    }

    #[test]
    fn rejects_shifted_stencil_offset_consumer() {
        const STENCIL_CONSUMER_PTX: &str = r#"
            .version 7.8
            .target sm_80
            .address_size 64
            .visible .entry stencil_cons(
                .param .u64 tmp,
                .param .u64 out,
                .param .f32 scale,
                .param .f32 bias,
                .param .u32 n
            ) {
                .reg .pred %p<2>;
                .reg .f32 %f<5>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<8>;
                ld.param.u64 %rd1, [tmp];
                ld.param.u64 %rd2, [out];
                ld.param.f32 %f1, [scale];
                ld.param.f32 %f2, [bias];
                ld.param.u32 %r1, [n];
                mov.u32 %r2, %ctaid.x;
                mov.u32 %r3, %ntid.x;
                mov.u32 %r4, %tid.x;
                mad.lo.s32 %r5, %r2, %r3, %r4;
                setp.ge.s32 %p1, %r5, %r1;
                @%p1 bra $L_EXIT;
                cvta.to.global.u64 %rd3, %rd1;
                cvta.to.global.u64 %rd4, %rd2;
                mul.wide.s32 %rd5, %r5, 4;
                add.s64 %rd6, %rd3, %rd5;
                ld.global.f32 %f3, [%rd6 + 4];
                fma.rn.f32 %f4, %f3, %f1, %f2;
                add.s64 %rd7, %rd4, %rd5;
                st.global.f32 [%rd7], %f4;
            $L_EXIT:
                ret;
            }
        "#;
        let p_mod = parse_ptx(VADD_PTX).expect("parse producer");
        let c_mod = parse_ptx(STENCIL_CONSUMER_PTX).expect("parse stencil");
        let p_ir = lower_ptx_to_ir(&p_mod, IsaTarget::Gfx1100).expect("lower producer");
        let c_ir = lower_ptx_to_ir(&c_mod, IsaTarget::Gfx1100).expect("lower stencil");
        assert!(fuse_elementwise_kernels(&p_ir.kernels[0], &c_ir.kernels[0], 2, 0).is_err());
    }

    #[test]
    fn rejects_aliased_in_place_pointer_params() {
        let p_mod = parse_ptx(VADD_PTX).expect("parse producer");
        let c_mod = parse_ptx(SCALE_BIAS_PTX).expect("parse consumer");
        let p_ir = lower_ptx_to_ir(&p_mod, IsaTarget::Gfx1100).expect("lower producer");
        let c_ir = lower_ptx_to_ir(&c_mod, IsaTarget::Gfx1100).expect("lower consumer");
        let prod = &p_ir.kernels[0];
        let cons = &c_ir.kernels[0];

        let mut prod_ka = vec![0u8; prod.metadata.kernarg_segment_size as usize];
        let mut cons_ka = vec![0u8; cons.metadata.kernarg_segment_size as usize];
        let ptr_a = 0x1000_0000u64;
        let ptr_b = 0x2000_0000u64;
        let ptr_tmp = 0x3000_0000u64;
        let ptr_out = 0x4000_0000u64;

        let write_u64 = |buf: &mut [u8], off: u32, val: u64| {
            let s = off as usize;
            buf[s..s + 8].copy_from_slice(&val.to_le_bytes());
        };

        // Case 1: producer input `a` aliases `tmp` -> must refuse fusion.
        write_u64(&mut prod_ka, prod.params[0].offset, ptr_tmp);
        write_u64(&mut prod_ka, prod.params[1].offset, ptr_b);
        write_u64(&mut prod_ka, prod.params[2].offset, ptr_tmp);
        write_u64(&mut cons_ka, cons.params[0].offset, ptr_tmp);
        write_u64(&mut cons_ka, cons.params[1].offset, ptr_out);
        assert_eq!(
            find_fusible_buffer_params(prod, cons, &prod_ka, &cons_ka),
            None
        );

        // Case 2: producer disjoint, consumer output `out` aliases `tmp` -> must refuse fusion.
        write_u64(&mut prod_ka, prod.params[0].offset, ptr_a);
        write_u64(&mut cons_ka, cons.params[1].offset, ptr_tmp);
        assert_eq!(
            find_fusible_buffer_params(prod, cons, &prod_ka, &cons_ka),
            None
        );

        // Case 3: disjoint -> allowed.
        write_u64(&mut cons_ka, cons.params[1].offset, ptr_out);
        assert_eq!(
            find_fusible_buffer_params(prod, cons, &prod_ka, &cons_ka),
            Some((2, 0, ptr_tmp))
        );
    }
}
