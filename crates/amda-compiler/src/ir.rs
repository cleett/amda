//! AMDA Core / Tile IR, PTX lowering, legalization passes, LLVM AMDGPU IR emitter,
//! and binary `.amda.ir` serialization.

use crate::error::CompileError;
use crate::ptx::ast::{
    PtxFunc, PtxInstruction, PtxKernel, PtxModule, PtxOperand, PtxStatement, PtxType,
};
use amda_elf::{IsaTarget, KernelArgMetadata, KernelMetadata, ValueKind};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

const IR_BINARY_MAGIC: [u8; 8] = *b"AMDAIR02";

/// Strongly typed scalar values in AMDA IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IrType {
    U32 = 1,
    I32 = 2,
    U64 = 3,
    I64 = 4,
    F32 = 5,
    F64 = 6,
    Pred = 7,
    U8 = 8,
    I8 = 9,
    U16 = 10,
    I16 = 11,
    F16 = 12,
    Bf16 = 13,
}

impl IrType {
    /// Converts a PTX type into its corresponding AMDA IR register type.
    pub const fn from_ptx(ty: PtxType) -> Self {
        match ty {
            PtxType::Pred => Self::Pred,
            PtxType::B8 | PtxType::U8 => Self::U8,
            PtxType::S8 => Self::I8,
            PtxType::B16 | PtxType::U16 => Self::U16,
            PtxType::S16 => Self::I16,
            PtxType::F16 => Self::F16,
            PtxType::Bf16 => Self::Bf16,
            PtxType::B32 | PtxType::U32 | PtxType::F16x2 => Self::U32,
            PtxType::S32 => Self::I32,
            PtxType::B64 | PtxType::U64 => Self::U64,
            PtxType::S64 => Self::I64,
            PtxType::F32 => Self::F32,
            PtxType::F64 => Self::F64,
        }
    }

    /// Size of this IR scalar type in bytes.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Pred | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    const fn llvm_ty(self) -> &'static str {
        match self {
            Self::U8 | Self::I8 => "i8",
            Self::U16 | Self::I16 => "i16",
            Self::F16 => "half",
            Self::Bf16 => "bfloat",
            Self::U32 | Self::I32 => "i32",
            Self::U64 | Self::I64 => "i64",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Pred => "i1",
        }
    }

    const fn is_signed_int_ty(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            1 => Ok(Self::U32),
            2 => Ok(Self::I32),
            3 => Ok(Self::U64),
            4 => Ok(Self::I64),
            5 => Ok(Self::F32),
            6 => Ok(Self::F64),
            7 => Ok(Self::Pred),
            8 => Ok(Self::U8),
            9 => Ok(Self::I8),
            10 => Ok(Self::U16),
            11 => Ok(Self::I16),
            12 => Ok(Self::F16),
            13 => Ok(Self::Bf16),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown IrType tag {tag}"
            ))),
        }
    }
}

/// AMDGPU memory address spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AddressSpace {
    /// Flat generic pointer (`addrspace(0)`).
    Generic = 0,
    /// Global VRAM buffer (`addrspace(1)`).
    Global = 1,
    /// Workgroup shared LDS (`addrspace(3)`).
    Shared = 3,
    /// Constant memory (`addrspace(4)`).
    Constant = 4,
    /// Per-lane private scratch (`addrspace(5)`).
    Private = 5,
}

impl AddressSpace {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Generic),
            1 => Ok(Self::Global),
            3 => Ok(Self::Shared),
            4 => Ok(Self::Constant),
            5 => Ok(Self::Private),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown AddressSpace tag {tag}"
            ))),
        }
    }
}

/// Legalized wavefront execution mode targeting RDNA vs CDNA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WaveMode {
    /// RDNA native Wave32 (1:1 mapping for CUDA 32-lane warps).
    NativeWave32 = 1,
    /// CDNA Wave64 emulating two 32-lane virtual CUDA warps per hardware wavefront.
    EmulatedWave32OnWave64 = 2,
    /// CDNA native Wave64 (no 32-lane warp assumptions detected).
    NativeWave64 = 3,
}

impl WaveMode {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            1 => Ok(Self::NativeWave32),
            2 => Ok(Self::EmulatedWave32OnWave64),
            3 => Ok(Self::NativeWave64),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown WaveMode tag {tag}"
            ))),
        }
    }
}

/// Thread/workgroup identification special registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SpecialReg {
    TidX = 0,
    TidY = 1,
    TidZ = 2,
    CtaIdX = 3,
    CtaIdY = 4,
    CtaIdZ = 5,
    NtidX = 6,
    NtidY = 7,
    NtidZ = 8,
    NctaIdX = 9,
    NctaIdY = 10,
    NctaIdZ = 11,
    LaneId = 12,
}

impl SpecialReg {
    fn from_name(s: &str) -> Option<Self> {
        match s {
            "%tid.x" => Some(Self::TidX),
            "%tid.y" => Some(Self::TidY),
            "%tid.z" => Some(Self::TidZ),
            "%ctaid.x" => Some(Self::CtaIdX),
            "%ctaid.y" => Some(Self::CtaIdY),
            "%ctaid.z" => Some(Self::CtaIdZ),
            "%ntid.x" => Some(Self::NtidX),
            "%ntid.y" => Some(Self::NtidY),
            "%ntid.z" => Some(Self::NtidZ),
            "%nctaid.x" => Some(Self::NctaIdX),
            "%nctaid.y" => Some(Self::NctaIdY),
            "%nctaid.z" => Some(Self::NctaIdZ),
            "%laneid" => Some(Self::LaneId),
            _ => None,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::TidX),
            1 => Ok(Self::TidY),
            2 => Ok(Self::TidZ),
            3 => Ok(Self::CtaIdX),
            4 => Ok(Self::CtaIdY),
            5 => Ok(Self::CtaIdZ),
            6 => Ok(Self::NtidX),
            7 => Ok(Self::NtidY),
            8 => Ok(Self::NtidZ),
            9 => Ok(Self::NctaIdX),
            10 => Ok(Self::NctaIdY),
            11 => Ok(Self::NctaIdZ),
            12 => Ok(Self::LaneId),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown SpecialReg tag {tag}"
            ))),
        }
    }
}

/// Comparison predicates for `Setp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CmpOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

impl CmpOp {
    fn from_modifiers(mods: &[String]) -> Option<Self> {
        for m in mods {
            match m.as_str() {
                "eq" | "equ" => return Some(Self::Eq),
                "ne" | "neu" => return Some(Self::Ne),
                "lt" | "ltu" | "lo" => return Some(Self::Lt),
                "le" | "leu" | "ls" => return Some(Self::Le),
                "gt" | "gtu" | "hi" => return Some(Self::Gt),
                "ge" | "geu" | "hs" => return Some(Self::Ge),
                _ => {}
            }
        }
        None
    }

    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Eq),
            1 => Ok(Self::Ne),
            2 => Ok(Self::Lt),
            3 => Ok(Self::Le),
            4 => Ok(Self::Gt),
            5 => Ok(Self::Ge),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown CmpOp tag {tag}"
            ))),
        }
    }
}

/// Binary arithmetic and bitwise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BinOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    MulWide = 3,
    Div = 4,
    Shl = 5,
    Shr = 6,
    And = 7,
    Or = 8,
    Xor = 9,
    Min = 10,
    Max = 11,
    Rem = 12,
}

impl BinOp {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Mul),
            3 => Ok(Self::MulWide),
            4 => Ok(Self::Div),
            5 => Ok(Self::Shl),
            6 => Ok(Self::Shr),
            7 => Ok(Self::And),
            8 => Ok(Self::Or),
            9 => Ok(Self::Xor),
            10 => Ok(Self::Min),
            11 => Ok(Self::Max),
            12 => Ok(Self::Rem),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown BinOp tag {tag}"
            ))),
        }
    }
}

/// Unary transcendental and reciprocal math operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UnaryMathOp {
    Rcp = 0,
    Sqrt = 1,
    Rsqrt = 2,
    Ex2 = 3,
    Lg2 = 4,
    Sin = 5,
    Cos = 6,
    Tanh = 7,
}

impl UnaryMathOp {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Rcp),
            1 => Ok(Self::Sqrt),
            2 => Ok(Self::Rsqrt),
            3 => Ok(Self::Ex2),
            4 => Ok(Self::Lg2),
            5 => Ok(Self::Sin),
            6 => Ok(Self::Cos),
            7 => Ok(Self::Tanh),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown UnaryMathOp tag {tag}"
            ))),
        }
    }
}

/// Warp shuffle direction mode (`shfl.sync`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShuffleMode {
    Down = 0,
    Up = 1,
    Bfly = 2,
    Idx = 3,
}

impl ShuffleMode {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Down),
            1 => Ok(Self::Up),
            2 => Ok(Self::Bfly),
            3 => Ok(Self::Idx),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown ShuffleMode tag {tag}"
            ))),
        }
    }
}

/// Warp predicate vote mode (`vote.sync`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VoteMode {
    Ballot = 0,
    Any = 1,
    All = 2,
}

impl VoteMode {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Ballot),
            1 => Ok(Self::Any),
            2 => Ok(Self::All),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown VoteMode tag {tag}"
            ))),
        }
    }
}

/// Atomic read-modify-write operation (`atom` / `red`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AtomicOp {
    Add = 0,
    Min = 1,
    Max = 2,
    And = 3,
    Or = 4,
    Xor = 5,
    Exch = 6,
    Cas = 7,
}

impl AtomicOp {
    fn from_tag(tag: u8) -> Result<Self, CompileError> {
        match tag {
            0 => Ok(Self::Add),
            1 => Ok(Self::Min),
            2 => Ok(Self::Max),
            3 => Ok(Self::And),
            4 => Ok(Self::Or),
            5 => Ok(Self::Xor),
            6 => Ok(Self::Exch),
            7 => Ok(Self::Cas),
            _ => Err(CompileError::InvalidIrPayload(format!(
                "unknown AtomicOp tag {tag}"
            ))),
        }
    }
}

/// Operands in AMDA IR instructions.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOperand {
    /// Virtual register index.
    Reg(u32),
    /// Unsigned integer immediate.
    ImmU64(u64),
    /// Signed integer immediate.
    ImmI64(i64),
    /// Floating-point immediate.
    ImmF64(f64),
    /// Built-in hardware thread/block/lane special register.
    Special(SpecialReg),
    /// Kernel parameter index.
    Param(u32),
    /// Byte offset of a `.shared` LDS symbol.
    SharedOffset(u32),
    /// Byte offset of a `.local` per-thread scratch symbol.
    LocalOffset(u32),
}

/// Optional instruction execution predicate guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrPredGuard {
    pub reg: u32,
    pub negated: bool,
}

/// Single AMDA IR instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct IrInst {
    pub pred: Option<IrPredGuard>,
    pub op: IrOp,
}

/// AMDA IR operation variants.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOp {
    LoadParam {
        dst: u32,
        ty: IrType,
        param_idx: u32,
        offset: u32,
    },
    Load {
        dst: u32,
        ty: IrType,
        space: AddressSpace,
        base: IrOperand,
        offset: i64,
    },
    Store {
        ty: IrType,
        space: AddressSpace,
        base: IrOperand,
        offset: i64,
        src: IrOperand,
    },
    CvtaToGlobal {
        dst: u32,
        src: IrOperand,
    },
    CvtaToShared {
        dst: u32,
        src: IrOperand,
    },
    Mov {
        dst: u32,
        ty: IrType,
        src: IrOperand,
    },
    Cvt {
        dst: u32,
        dst_ty: IrType,
        src_ty: IrType,
        src: IrOperand,
    },
    Neg {
        dst: u32,
        ty: IrType,
        src: IrOperand,
    },
    Abs {
        dst: u32,
        ty: IrType,
        src: IrOperand,
    },
    Bin {
        op: BinOp,
        dst: u32,
        ty: IrType,
        lhs: IrOperand,
        rhs: IrOperand,
    },
    Mad {
        dst: u32,
        ty: IrType,
        a: IrOperand,
        b: IrOperand,
        c: IrOperand,
    },
    Setp {
        cmp: CmpOp,
        dst: u32,
        ty: IrType,
        lhs: IrOperand,
        rhs: IrOperand,
    },
    Selp {
        dst: u32,
        ty: IrType,
        a: IrOperand,
        b: IrOperand,
        pred: u32,
    },
    Label(String),
    Branch(String),
    Barrier,
    MemBar,
    UnaryMath {
        op: UnaryMathOp,
        dst: u32,
        ty: IrType,
        src: IrOperand,
    },
    Bfe {
        dst: u32,
        signed: bool,
        src: IrOperand,
        pos: IrOperand,
        len: IrOperand,
    },
    Bfi {
        dst: u32,
        insert: IrOperand,
        base: IrOperand,
        pos: IrOperand,
        len: IrOperand,
    },
    Prmt {
        dst: u32,
        a: IrOperand,
        b: IrOperand,
        sel: IrOperand,
    },
    Dp4a {
        dst: u32,
        a_signed: bool,
        b_signed: bool,
        a: IrOperand,
        b: IrOperand,
        c: IrOperand,
    },
    Atomic {
        op: AtomicOp,
        dst: Option<u32>,
        ty: IrType,
        space: AddressSpace,
        base: IrOperand,
        offset: i64,
        val: IrOperand,
        cmp: Option<IrOperand>,
    },
    WarpShuffle {
        dst: u32,
        mode: ShuffleMode,
        src: IrOperand,
        lane_delta: IrOperand,
        width: u32,
    },
    WarpVote {
        mode: VoteMode,
        dst: u32,
        pred: u32,
        negated: bool,
    },
    Return,
}

/// Kernel parameter descriptor in AMDA IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrParam {
    pub name: String,
    pub ty: IrType,
    pub offset: u32,
    pub size: u32,
    pub is_global_ptr: bool,
}

/// Lowered AMDA IR kernel ready for LLVM IR emission or virtual GPU execution.
#[derive(Debug, Clone, PartialEq)]
pub struct IrKernel {
    pub name: String,
    pub params: Vec<IrParam>,
    /// Register table storing the IR type of each `Reg(idx)`.
    pub reg_types: Vec<IrType>,
    /// Static LDS shared memory required per workgroup in bytes.
    pub shared_bytes: u32,
    /// Static `.local` private scratch memory required per work-item in bytes.
    pub private_bytes: u32,
    /// Legalized wavefront execution mode.
    pub wave_mode: WaveMode,
    /// Instruction stream.
    pub insts: Vec<IrInst>,
    /// Computed HSACO metadata.
    pub metadata: KernelMetadata,
}

/// Lowered AMDA IR module containing one or more kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct IrModule {
    pub target: IsaTarget,
    pub kernels: Vec<IrKernel>,
}

/// Fusion opportunity detected between adjacent kernel launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    pub kernel_names: Vec<String>,
    pub fusion_kind: String,
    pub estimated_bandwidth_saved_pct: u32,
}

/// Analyzes adjacent kernels for cross-boundary producer-consumer fusion opportunities.
pub fn analyze_kernel_fusion(kernels: &[&IrKernel]) -> Vec<FusionCandidate> {
    let mut out = Vec::new();
    for pair in kernels.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let a_writes_global = a
            .insts
            .iter()
            .any(|i| matches!(i.op, IrOp::Store { space: AddressSpace::Global, .. }));
        let b_reads_global = b
            .insts
            .iter()
            .any(|i| matches!(i.op, IrOp::Load { space: AddressSpace::Global, .. }));
        if a_writes_global && b_reads_global && a.shared_bytes == 0 && b.shared_bytes == 0 {
            out.push(FusionCandidate {
                kernel_names: vec![a.name.clone(), b.name.clone()],
                fusion_kind: "ElementwiseProducerConsumer".to_owned(),
                estimated_bandwidth_saved_pct: 33,
            });
        }
    }
    out
}

/// Lowers a parsed [`PtxModule`] into an AMDA [`IrModule`] legalized for `target`.
pub fn lower_ptx_to_ir(module: &PtxModule, target: IsaTarget) -> Result<IrModule, CompileError> {
    let mut kernels = Vec::with_capacity(module.kernels.len());
    for ptx_k in &module.kernels {
        kernels.push(lower_kernel(module, ptx_k, target)?);
    }
    Ok(IrModule { target, kernels })
}

fn lower_kernel(
    module: &PtxModule,
    ptx_k: &PtxKernel,
    target: IsaTarget,
) -> Result<IrKernel, CompileError> {
    let func_map: HashMap<&str, &PtxFunc> =
        module.funcs.iter().map(|f| (f.name.as_str(), f)).collect();

    // 1. Register allocation map
    let mut reg_map: HashMap<String, u32> = HashMap::new();
    let mut reg_types: Vec<IrType> = Vec::new();

    for decl in &ptx_k.regs {
        let ir_ty = IrType::from_ptx(decl.ty);
        match decl.count {
            Some(n) => {
                for idx in 0..n {
                    let name = format!("{}{idx}", decl.prefix);
                    if let std::collections::hash_map::Entry::Vacant(e) = reg_map.entry(name) {
                        let id = reg_types.len() as u32;
                        e.insert(id);
                        reg_types.push(ir_ty);
                    }
                }
            }
            None => {
                if !reg_map.contains_key(&decl.prefix) {
                    let id = reg_types.len() as u32;
                    reg_map.insert(decl.prefix.clone(), id);
                    reg_types.push(ir_ty);
                }
            }
        }
    }

    // 2. Shared memory symbol table (module-level + kernel-level, incl. `.extern .shared`)
    let mut shared_offsets: HashMap<String, u32> = HashMap::new();
    let mut shared_bytes = 0u32;
    for s in module
        .shared_decls
        .iter()
        .chain(ptx_k.shared_decls.iter())
        .filter(|s| !s.is_extern)
    {
        let align = s.align.max(s.ty.byte_size()).max(1);
        shared_bytes = align_up_u32(shared_bytes, align);
        shared_offsets.insert(s.name.clone(), shared_bytes);
        shared_bytes += s.ty.byte_size() * s.count.max(1);
    }
    shared_bytes = align_up_u32(shared_bytes, 16);
    for s in module
        .shared_decls
        .iter()
        .chain(ptx_k.shared_decls.iter())
        .filter(|s| s.is_extern)
    {
        let align = s.align.max(s.ty.byte_size()).max(1);
        let ext_off = align_up_u32(shared_bytes, align);
        shared_offsets.insert(s.name.clone(), ext_off);
    }

    // 2b. Per-thread `.local` private scratch allocations
    let mut local_offsets: HashMap<String, u32> = HashMap::new();
    let mut private_bytes = 0u32;
    for l in &ptx_k.local_decls {
        let align = l.align.max(l.ty.byte_size()).max(1);
        private_bytes = align_up_u32(private_bytes, align);
        local_offsets.insert(l.name.clone(), private_bytes);
        private_bytes += l.ty.byte_size() * l.count.max(1);
    }
    private_bytes = align_up_u32(private_bytes, 16);

    // 3. Determine which params are used as global pointers via cvta/global ld/st
    let mut param_name_to_idx: HashMap<String, u32> = HashMap::new();
    for (idx, p) in ptx_k.params.iter().enumerate() {
        param_name_to_idx.insert(p.name.clone(), idx as u32);
    }

    // Track register address spaces for legalization
    let mut reg_spaces: HashMap<u32, AddressSpace> = HashMap::new();
    let mut ptr_params: HashSet<u32> = HashSet::new();
    let mut uses_warp_ops = false;

    // Pre-scan for `ld.param` -> `cvta.to.global` provenance and warp instructions (kernel + helpers)
    let mut param_dest_regs: HashMap<u32, u32> = HashMap::new();
    for stmt in ptx_k
        .statements
        .iter()
        .chain(module.funcs.iter().flat_map(|f| f.statements.iter()))
    {
        if let PtxStatement::Instruction(inst) = stmt {
            if matches!(inst.opcode.as_str(), "shfl" | "vote") {
                uses_warp_ops = true;
            }
            if inst.opcode == "ld" && inst.modifiers.iter().any(|m| m == "param") {
                if let (Some(PtxOperand::Reg(dst)), Some(PtxOperand::Mem { base, .. })) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    if let (Some(&reg_id), Some(&pidx)) =
                        (reg_map.get(dst), param_name_to_idx.get(base))
                    {
                        param_dest_regs.insert(reg_id, pidx);
                    }
                }
            }
            if inst.opcode == "cvta" && inst.modifiers.iter().any(|m| m == "global") {
                if let (Some(PtxOperand::Reg(dst)), Some(PtxOperand::Reg(src))) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    if let Some(&src_id) = reg_map.get(src) {
                        if let Some(&pidx) = param_dest_regs.get(&src_id) {
                            ptr_params.insert(pidx);
                        }
                    }
                    if let Some(&dst_id) = reg_map.get(dst) {
                        reg_spaces.insert(dst_id, AddressSpace::Global);
                    }
                }
            }
            for op in &inst.operands {
                if let PtxOperand::Reg(r) = op {
                    if r == "%laneid" {
                        uses_warp_ops = true;
                    }
                }
            }
        }
    }

    // 4. Compute kernarg offsets & metadata
    let mut ir_params = Vec::with_capacity(ptx_k.params.len());
    let mut arg_metas = Vec::with_capacity(ptx_k.params.len());
    let mut offset = 0u32;

    for (idx, p) in ptx_k.params.iter().enumerate() {
        let ir_ty = IrType::from_ptx(p.ty);
        let size = p.ty.byte_size() * p.count.max(1);
        let align = size.clamp(1, 8);
        offset = align_up_u32(offset, align);

        let is_global_ptr =
            p.is_ptr || ptr_params.contains(&(idx as u32)) || matches!(p.ty, PtxType::U64 | PtxType::B64);

        ir_params.push(IrParam {
            name: p.name.clone(),
            ty: ir_ty,
            offset,
            size,
            is_global_ptr,
        });
        arg_metas.push(KernelArgMetadata {
            name: p.name.clone(),
            offset,
            size,
            value_kind: if is_global_ptr {
                ValueKind::GlobalBuffer
            } else {
                ValueKind::ByValue
            },
        });
        offset += size;
    }
    let kernarg_segment_size = align_up_u32(offset, 8);

    // 5. Lower instructions with address space inference and `.func` inlining
    let mut insts = Vec::with_capacity(ptx_k.statements.len());
    let mut inline_ctx = LowerCtx {
        func_map: &func_map,
        reg_map: &mut reg_map,
        reg_types: &mut reg_types,
        reg_spaces: &mut reg_spaces,
        param_map: &param_name_to_idx,
        shared_offsets: &shared_offsets,
        local_offsets: &mut local_offsets,
        private_bytes: &mut private_bytes,
        inline_counter: 0,
    };

    for stmt in &ptx_k.statements {
        match stmt {
            PtxStatement::Label(lbl) => insts.push(IrInst {
                pred: None,
                op: IrOp::Label(lbl.clone()),
            }),
            PtxStatement::Instruction(inst) => {
                lower_instruction(inst, &mut inline_ctx, &HashMap::new(), &HashMap::new(), None, &mut insts, 0)?;
            }
        }
    }

    let wave_mode = if target.is_rdna() {
        WaveMode::NativeWave32
    } else if uses_warp_ops {
        WaveMode::EmulatedWave32OnWave64
    } else {
        WaveMode::NativeWave64
    };

    let metadata = KernelMetadata {
        name: ptx_k.name.clone(),
        symbol: format!("{}.kd", ptx_k.name),
        kernarg_segment_size,
        group_segment_fixed_size: shared_bytes,
        private_segment_fixed_size: private_bytes,
        wavefront_size: target.native_wave_size(),
        max_flat_workgroup_size: 1024,
        args: arg_metas,
    };

    Ok(IrKernel {
        name: ptx_k.name.clone(),
        params: ir_params,
        reg_types,
        shared_bytes,
        private_bytes,
        wave_mode,
        insts,
        metadata,
    })
}

struct LowerCtx<'a> {
    func_map: &'a HashMap<&'a str, &'a PtxFunc>,
    reg_map: &'a mut HashMap<String, u32>,
    reg_types: &'a mut Vec<IrType>,
    reg_spaces: &'a mut HashMap<u32, AddressSpace>,
    param_map: &'a HashMap<String, u32>,
    shared_offsets: &'a HashMap<String, u32>,
    local_offsets: &'a mut HashMap<String, u32>,
    private_bytes: &'a mut u32,
    inline_counter: u32,
}

#[allow(clippy::too_many_arguments)]
fn lower_instruction(
    inst: &PtxInstruction,
    ctx: &mut LowerCtx<'_>,
    local_in_params: &HashMap<String, u32>,
    local_ret_params: &HashMap<String, u32>,
    callee_ret_label: Option<&str>,
    out: &mut Vec<IrInst>,
    inline_depth: usize,
) -> Result<(), CompileError> {
    let pred = match &inst.pred {
        Some(p) => {
            let reg = resolve_reg(ctx.reg_map, ctx.reg_types, &p.reg, IrType::Pred);
            Some(IrPredGuard {
                reg,
                negated: p.negated,
            })
        }
        None => None,
    };

    let inst_ty = infer_inst_type(&inst.modifiers).unwrap_or(IrType::U32);
    let vec_len: usize = if inst.modifiers.iter().any(|m| m == "v4") {
        4
    } else if inst.modifiers.iter().any(|m| m == "v2") {
        2
    } else {
        1
    };

    match inst.opcode.as_str() {
        "call" => {
            return inline_ptx_call(inst, pred, ctx, out, inline_depth);
        }
        "ld" => {
            let (base_name, offset) = expect_mem_operand(&inst.operands, 1, "ld src")?;
            if inst.modifiers.iter().any(|m| m == "param") {
                if let Some(&param_vreg) = local_in_params.get(base_name) {
                    let dst_name = expect_reg_operand(&inst.operands, 0, "ld.param inline dst")?;
                    let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
                    out.push(IrInst {
                        pred,
                        op: IrOp::Mov {
                            dst,
                            ty: inst_ty,
                            src: IrOperand::Reg(param_vreg),
                        },
                    });
                    return Ok(());
                }
                let param_idx = *ctx.param_map.get(base_name).ok_or_else(|| {
                    CompileError::LoweringError(format!("unknown parameter '{base_name}' in ld.param"))
                })?;
                if vec_len > 1 {
                    let dst_names = extract_reg_list(&inst.operands[0], vec_len, "ld.param vector dst")?;
                    let stride = inst_ty.byte_size();
                    for (i, dname) in dst_names.iter().enumerate() {
                        let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dname, inst_ty);
                        let off_u32 = u32::try_from(offset + (i as i64) * i64::from(stride))
                            .map_err(|_| CompileError::LoweringError("negative ld.param offset".to_owned()))?;
                        out.push(IrInst {
                            pred,
                            op: IrOp::LoadParam {
                                dst,
                                ty: inst_ty,
                                param_idx,
                                offset: off_u32,
                            },
                        });
                    }
                } else {
                    let dst_name = expect_reg_operand(&inst.operands, 0, "ld dst")?;
                    let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
                    let off_u32 = u32::try_from(offset).map_err(|_| {
                        CompileError::LoweringError("negative ld.param offset".to_owned())
                    })?;
                    out.push(IrInst {
                        pred,
                        op: IrOp::LoadParam {
                            dst,
                            ty: inst_ty,
                            param_idx,
                            offset: off_u32,
                        },
                    });
                }
                return Ok(());
            }

            let explicit_space = space_from_modifiers(&inst.modifiers);
            let (base_op, inferred_space) = resolve_addr_base(
                base_name,
                ctx.reg_map,
                ctx.reg_types,
                ctx.reg_spaces,
                ctx.shared_offsets,
                ctx.local_offsets,
            )?;
            let space = explicit_space.unwrap_or(inferred_space);

            if vec_len > 1 {
                let dst_names = extract_reg_list(&inst.operands[0], vec_len, "ld vector dst")?;
                let stride = i64::from(inst_ty.byte_size());
                for (i, dname) in dst_names.iter().enumerate() {
                    let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dname, inst_ty);
                    out.push(IrInst {
                        pred,
                        op: IrOp::Load {
                            dst,
                            ty: inst_ty,
                            space,
                            base: base_op.clone(),
                            offset: offset + (i as i64) * stride,
                        },
                    });
                }
            } else {
                let dst_name = expect_reg_operand(&inst.operands, 0, "ld dst")?;
                let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
                out.push(IrInst {
                    pred,
                    op: IrOp::Load {
                        dst,
                        ty: inst_ty,
                        space,
                        base: base_op,
                        offset,
                    },
                });
            }
            return Ok(());
        }
        "st" => {
            let (base_name, offset) = expect_mem_operand(&inst.operands, 0, "st dst")?;
            if inst.modifiers.iter().any(|m| m == "param") {
                if let Some(&ret_vreg) = local_ret_params.get(base_name) {
                    let src_op = inst.operands.get(1).ok_or_else(|| {
                        CompileError::LoweringError("st.param missing src".to_owned())
                    })?;
                    let src = lower_operand(
                        src_op,
                        ctx.reg_map,
                        ctx.reg_types,
                        ctx.shared_offsets,
                        ctx.local_offsets,
                        inst_ty,
                    )?;
                    out.push(IrInst {
                        pred,
                        op: IrOp::Mov {
                            dst: ret_vreg,
                            ty: inst_ty,
                            src,
                        },
                    });
                    return Ok(());
                }
            }
            let explicit_space = space_from_modifiers(&inst.modifiers);
            let (base_op, inferred_space) = resolve_addr_base(
                base_name,
                ctx.reg_map,
                ctx.reg_types,
                ctx.reg_spaces,
                ctx.shared_offsets,
                ctx.local_offsets,
            )?;
            let space = explicit_space.unwrap_or(inferred_space);

            if vec_len > 1 {
                let src_op = inst.operands.get(1).ok_or_else(|| {
                    CompileError::LoweringError("st.v* missing tuple operand".to_owned())
                })?;
                let items = match src_op {
                    PtxOperand::Tuple(v) if v.len() >= vec_len => v,
                    _ => {
                        return Err(CompileError::LoweringError(
                            "expected vector tuple operand for st.v2/v4".to_owned(),
                        ))
                    }
                };
                let stride = i64::from(inst_ty.byte_size());
                for (i, item) in items.iter().take(vec_len).enumerate() {
                    let src = lower_operand(
                        item,
                        ctx.reg_map,
                        ctx.reg_types,
                        ctx.shared_offsets,
                        ctx.local_offsets,
                        inst_ty,
                    )?;
                    out.push(IrInst {
                        pred,
                        op: IrOp::Store {
                            ty: inst_ty,
                            space,
                            base: base_op.clone(),
                            offset: offset + (i as i64) * stride,
                            src,
                        },
                    });
                }
            } else {
                let src_op = inst.operands.get(1).ok_or_else(|| {
                    CompileError::LoweringError("st missing source operand".to_owned())
                })?;
                let src = lower_operand(
                    src_op,
                    ctx.reg_map,
                    ctx.reg_types,
                    ctx.shared_offsets,
                    ctx.local_offsets,
                    inst_ty,
                )?;
                out.push(IrInst {
                    pred,
                    op: IrOp::Store {
                        ty: inst_ty,
                        space,
                        base: base_op,
                        offset,
                        src,
                    },
                });
            }
            return Ok(());
        }
        _ => {}
    }

    let op = match inst.opcode.as_str() {
        "cvta" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "cvta dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, IrType::U64);
            let src_op = inst.operands.get(1).ok_or_else(|| {
                CompileError::LoweringError("cvta missing source operand".to_owned())
            })?;
            let src = lower_operand(
                src_op,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U64,
            )?;

            if inst.modifiers.iter().any(|m| m == "shared") {
                ctx.reg_spaces.insert(dst, AddressSpace::Shared);
                IrOp::CvtaToShared { dst, src }
            } else if inst.modifiers.iter().any(|m| m == "local") {
                ctx.reg_spaces.insert(dst, AddressSpace::Private);
                IrOp::Mov {
                    dst,
                    ty: IrType::U64,
                    src,
                }
            } else {
                ctx.reg_spaces.insert(dst, AddressSpace::Global);
                IrOp::CvtaToGlobal { dst, src }
            }
        }
        "mov" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "mov dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src_op = inst.operands.get(1).ok_or_else(|| {
                CompileError::LoweringError("mov missing source operand".to_owned())
            })?;
            let src = lower_operand(
                src_op,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            if matches!(src, IrOperand::SharedOffset(_)) {
                ctx.reg_spaces.insert(dst, AddressSpace::Shared);
            } else if matches!(src, IrOperand::LocalOffset(_)) {
                ctx.reg_spaces.insert(dst, AddressSpace::Private);
            }
            IrOp::Mov {
                dst,
                ty: inst_ty,
                src,
            }
        }
        "cvt" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "cvt dst")?;
            let (dst_ty, src_ty) = infer_cvt_types(&inst.modifiers);
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, dst_ty);
            let src_op = inst.operands.get(1).ok_or_else(|| {
                CompileError::LoweringError("cvt missing source operand".to_owned())
            })?;
            let src = lower_operand(
                src_op,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                src_ty,
            )?;
            IrOp::Cvt {
                dst,
                dst_ty,
                src_ty,
                src,
            }
        }
        "neg" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "neg dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("neg missing src".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            IrOp::Neg {
                dst,
                ty: inst_ty,
                src,
            }
        }
        "abs" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "abs dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("abs missing src".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            IrOp::Abs {
                dst,
                ty: inst_ty,
                src,
            }
        }
        "rcp" | "sqrt" | "rsqrt" | "ex2" | "lg2" | "sin" | "cos" | "tanh" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "unary math dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("unary math missing src".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let op = match inst.opcode.as_str() {
                "rcp" => UnaryMathOp::Rcp,
                "sqrt" => UnaryMathOp::Sqrt,
                "rsqrt" => UnaryMathOp::Rsqrt,
                "ex2" => UnaryMathOp::Ex2,
                "lg2" => UnaryMathOp::Lg2,
                "sin" => UnaryMathOp::Sin,
                "cos" => UnaryMathOp::Cos,
                "tanh" => UnaryMathOp::Tanh,
                _ => unreachable!(),
            };
            IrOp::UnaryMath {
                op,
                dst,
                ty: inst_ty,
                src,
            }
        }
        "add" | "sub" | "mul" | "div" | "rem" | "shl" | "shr" | "and" | "or" | "xor" | "min"
        | "max" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "binop dst")?;
            let is_wide = inst.opcode == "mul" && inst.modifiers.iter().any(|m| m == "wide");
            let dst_ty = if is_wide {
                match inst_ty {
                    IrType::I32 => IrType::I64,
                    _ => IrType::U64,
                }
            } else {
                inst_ty
            };
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, dst_ty);

            let lhs_op = inst.operands.get(1).ok_or_else(|| {
                CompileError::LoweringError(format!("{} missing lhs", inst.opcode))
            })?;
            let rhs_op = inst.operands.get(2).ok_or_else(|| {
                CompileError::LoweringError(format!("{} missing rhs", inst.opcode))
            })?;

            let lhs = lower_operand(
                lhs_op,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let rhs = lower_operand(
                rhs_op,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;

            if inst.opcode == "add" {
                if let IrOperand::Reg(lhs_reg) = lhs {
                    if let Some(&sp) = ctx.reg_spaces.get(&lhs_reg) {
                        ctx.reg_spaces.insert(dst, sp);
                    }
                }
            }

            let op = match inst.opcode.as_str() {
                "add" => BinOp::Add,
                "sub" => BinOp::Sub,
                "mul" if is_wide => BinOp::MulWide,
                "mul" => BinOp::Mul,
                "div" => BinOp::Div,
                "rem" => BinOp::Rem,
                "shl" => BinOp::Shl,
                "shr" => BinOp::Shr,
                "and" => BinOp::And,
                "or" => BinOp::Or,
                "xor" => BinOp::Xor,
                "min" => BinOp::Min,
                "max" => BinOp::Max,
                _ => unreachable!(),
            };

            IrOp::Bin {
                op,
                dst,
                ty: inst_ty,
                lhs,
                rhs,
            }
        }
        "mad" | "fma" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "mad/fma dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let a = lower_operand(
                inst.operands.get(1).ok_or_else(|| {
                    CompileError::LoweringError("mad/fma missing operand a".to_owned())
                })?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let b = lower_operand(
                inst.operands.get(2).ok_or_else(|| {
                    CompileError::LoweringError("mad/fma missing operand b".to_owned())
                })?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let c = lower_operand(
                inst.operands.get(3).ok_or_else(|| {
                    CompileError::LoweringError("mad/fma missing operand c".to_owned())
                })?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            IrOp::Mad {
                dst,
                ty: inst_ty,
                a,
                b,
                c,
            }
        }
        "bfe" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "bfe dst")?;
            let signed = inst_ty == IrType::I32;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("bfe missing src".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let pos = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("bfe missing pos".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let len = lower_operand(
                inst.operands
                    .get(3)
                    .ok_or_else(|| CompileError::LoweringError("bfe missing len".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            IrOp::Bfe {
                dst,
                signed,
                src,
                pos,
                len,
            }
        }
        "bfi" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "bfi dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, IrType::U32);
            let insert = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("bfi missing insert".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let base = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("bfi missing base".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let pos = lower_operand(
                inst.operands
                    .get(3)
                    .ok_or_else(|| CompileError::LoweringError("bfi missing pos".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let len = lower_operand(
                inst.operands
                    .get(4)
                    .ok_or_else(|| CompileError::LoweringError("bfi missing len".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            IrOp::Bfi {
                dst,
                insert,
                base,
                pos,
                len,
            }
        }
        "prmt" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "prmt dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, IrType::U32);
            let a = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("prmt missing a".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let b = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("prmt missing b".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let sel = lower_operand(
                inst.operands
                    .get(3)
                    .ok_or_else(|| CompileError::LoweringError("prmt missing sel".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            IrOp::Prmt { dst, a, b, sel }
        }
        "dp4a" => {
            let mut ptys = Vec::new();
            for m in &inst.modifiers {
                if let Some(pt) = PtxType::from_modifier(m) {
                    ptys.push(pt);
                }
            }
            let a_signed = ptys.first().map(|p| p.is_signed()).unwrap_or(true);
            let b_signed = ptys.get(1).copied().or(ptys.first().copied()).map(|p| p.is_signed()).unwrap_or(true);
            let dst_name = expect_reg_operand(&inst.operands, 0, "dp4a dst")?;
            let dst_ty = if a_signed || b_signed { IrType::I32 } else { IrType::U32 };
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, dst_ty);
            let a = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("dp4a missing a".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let b = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("dp4a missing b".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            let c = lower_operand(
                inst.operands
                    .get(3)
                    .ok_or_else(|| CompileError::LoweringError("dp4a missing c".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                dst_ty,
            )?;
            IrOp::Dp4a {
                dst,
                a_signed,
                b_signed,
                a,
                b,
                c,
            }
        }
        "atom" | "red" => {
            let aop = inst
                .modifiers
                .iter()
                .find_map(|m| match m.as_str() {
                    "add" | "inc" => Some(AtomicOp::Add),
                    "min" => Some(AtomicOp::Min),
                    "max" => Some(AtomicOp::Max),
                    "and" => Some(AtomicOp::And),
                    "or" => Some(AtomicOp::Or),
                    "xor" => Some(AtomicOp::Xor),
                    "exch" => Some(AtomicOp::Exch),
                    "cas" => Some(AtomicOp::Cas),
                    _ => None,
                })
                .ok_or_else(|| {
                    CompileError::LoweringError(format!(
                        "unsupported atomic op modifiers: {:?}",
                        inst.modifiers
                    ))
                })?;

            let (dst, mem_idx) = match inst.operands.first() {
                Some(PtxOperand::Mem { .. }) => (None, 0usize),
                Some(PtxOperand::Ident(s)) if s == "_" => (None, 1usize),
                Some(PtxOperand::Reg(d)) => {
                    (Some(resolve_reg(ctx.reg_map, ctx.reg_types, d, inst_ty)), 1usize)
                }
                _ => {
                    return Err(CompileError::LoweringError(
                        "atom/red missing target or address".to_owned(),
                    ))
                }
            };
            let (base_name, offset) = expect_mem_operand(&inst.operands, mem_idx, "atom mem")?;
            let explicit_space = space_from_modifiers(&inst.modifiers);
            let (base, inferred_space) = resolve_addr_base(
                base_name,
                ctx.reg_map,
                ctx.reg_types,
                ctx.reg_spaces,
                ctx.shared_offsets,
                ctx.local_offsets,
            )?;
            let space = explicit_space.unwrap_or(inferred_space);

            let (val_op, cmp_op) = if aop == AtomicOp::Cas {
                let cmp_raw = inst.operands.get(mem_idx + 1).ok_or_else(|| {
                    CompileError::LoweringError("atom.cas missing cmp".to_owned())
                })?;
                let val_raw = inst.operands.get(mem_idx + 2).ok_or_else(|| {
                    CompileError::LoweringError("atom.cas missing new value".to_owned())
                })?;
                (
                    lower_operand(
                        val_raw,
                        ctx.reg_map,
                        ctx.reg_types,
                        ctx.shared_offsets,
                        ctx.local_offsets,
                        inst_ty,
                    )?,
                    Some(lower_operand(
                        cmp_raw,
                        ctx.reg_map,
                        ctx.reg_types,
                        ctx.shared_offsets,
                        ctx.local_offsets,
                        inst_ty,
                    )?),
                )
            } else {
                let val_raw = inst.operands.get(mem_idx + 1).ok_or_else(|| {
                    CompileError::LoweringError("atom/red missing value operand".to_owned())
                })?;
                (
                    lower_operand(
                        val_raw,
                        ctx.reg_map,
                        ctx.reg_types,
                        ctx.shared_offsets,
                        ctx.local_offsets,
                        inst_ty,
                    )?,
                    None,
                )
            };

            IrOp::Atomic {
                op: aop,
                dst,
                ty: inst_ty,
                space,
                base,
                offset,
                val: val_op,
                cmp: cmp_op,
            }
        }
        "setp" => {
            let cmp = CmpOp::from_modifiers(&inst.modifiers).ok_or_else(|| {
                CompileError::LoweringError(format!(
                    "unsupported setp comparison modifiers: {:?}",
                    inst.modifiers
                ))
            })?;
            let dst_name = expect_reg_operand(&inst.operands, 0, "setp dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, IrType::Pred);
            let lhs = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("setp missing lhs".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let rhs = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("setp missing rhs".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            IrOp::Setp {
                cmp,
                dst,
                ty: inst_ty,
                lhs,
                rhs,
            }
        }
        "selp" => {
            let dst_name = expect_reg_operand(&inst.operands, 0, "selp dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let a = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("selp missing a".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let b = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("selp missing b".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let pred_reg = match inst.operands.get(3) {
                Some(PtxOperand::Reg(name)) => {
                    resolve_reg(ctx.reg_map, ctx.reg_types, name, IrType::Pred)
                }
                _ => return Err(CompileError::LoweringError("expected register operand for selp predicate".to_owned())),
            };
            IrOp::Selp {
                dst,
                ty: inst_ty,
                a,
                b,
                pred: pred_reg,
            }
        }
        "bra" => {
            let target = match inst.operands.first() {
                Some(PtxOperand::Ident(lbl)) => lbl.clone(),
                _ => {
                    return Err(CompileError::LoweringError(
                        "bra expects a label identifier".to_owned(),
                    ))
                }
            };
            IrOp::Branch(target)
        }
        "bar" | "barrier" => IrOp::Barrier,
        "membar" | "fence" => IrOp::MemBar,
        "shfl" => {
            let mode = inst
                .modifiers
                .iter()
                .find_map(|m| match m.as_str() {
                    "down" => Some(ShuffleMode::Down),
                    "up" => Some(ShuffleMode::Up),
                    "bfly" => Some(ShuffleMode::Bfly),
                    "idx" => Some(ShuffleMode::Idx),
                    _ => None,
                })
                .unwrap_or(ShuffleMode::Down);

            let dst_name = match inst.operands.first() {
                Some(PtxOperand::Reg(r)) => r.as_str(),
                Some(PtxOperand::Tuple(items)) => match items.first() {
                    Some(PtxOperand::Reg(r)) => r.as_str(),
                    _ => {
                        return Err(CompileError::LoweringError(
                            "shfl destination tuple missing register".to_owned(),
                        ))
                    }
                },
                _ => {
                    return Err(CompileError::LoweringError(
                        "shfl missing destination register".to_owned(),
                    ))
                }
            };
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, inst_ty);
            let src = lower_operand(
                inst.operands
                    .get(1)
                    .ok_or_else(|| CompileError::LoweringError("shfl missing src".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                inst_ty,
            )?;
            let lane_delta = lower_operand(
                inst.operands
                    .get(2)
                    .ok_or_else(|| CompileError::LoweringError("shfl missing delta".to_owned()))?,
                ctx.reg_map,
                ctx.reg_types,
                ctx.shared_offsets,
                ctx.local_offsets,
                IrType::U32,
            )?;
            IrOp::WarpShuffle {
                dst,
                mode,
                src,
                lane_delta,
                width: 32,
            }
        }
        "vote" => {
            let mode = inst
                .modifiers
                .iter()
                .find_map(|m| match m.as_str() {
                    "ballot" => Some(VoteMode::Ballot),
                    "any" => Some(VoteMode::Any),
                    "all" => Some(VoteMode::All),
                    _ => None,
                })
                .unwrap_or(VoteMode::Any);
            let dst_ty = if mode == VoteMode::Ballot {
                IrType::U32
            } else {
                IrType::Pred
            };
            let dst_name = expect_reg_operand(&inst.operands, 0, "vote dst")?;
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, dst_name, dst_ty);
            let (pred_reg, negated) = match inst.operands.get(1) {
                Some(PtxOperand::Reg(r)) => (resolve_reg(ctx.reg_map, ctx.reg_types, r, IrType::Pred), false),
                Some(PtxOperand::NegatedReg(r)) => (resolve_reg(ctx.reg_map, ctx.reg_types, r, IrType::Pred), true),
                Some(PtxOperand::ImmInt(v)) => {
                    let tmp = ctx.reg_types.len() as u32;
                    ctx.reg_types.push(IrType::Pred);
                    out.push(IrInst {
                        pred,
                        op: IrOp::Mov {
                            dst: tmp,
                            ty: IrType::Pred,
                            src: IrOperand::ImmU64(u64::from(*v != 0)),
                        },
                    });
                    (tmp, false)
                }
                _ => {
                    return Err(CompileError::LoweringError(
                        "vote missing predicate operand".to_owned(),
                    ))
                }
            };
            IrOp::WarpVote {
                mode,
                dst,
                pred: pred_reg,
                negated,
            }
        }
        "ret" | "exit" => {
            if let Some(lbl) = callee_ret_label {
                IrOp::Branch(lbl.to_owned())
            } else {
                IrOp::Return
            }
        }
        other => {
            return Err(CompileError::LoweringError(format!(
                "unsupported PTX instruction '{other}'"
            )))
        }
    };

    out.push(IrInst { pred, op });
    Ok(())
}

fn extract_reg_list<'a>(
    op: &'a PtxOperand,
    expected: usize,
    ctx_msg: &str,
) -> Result<Vec<&'a str>, CompileError> {
    match op {
        PtxOperand::Tuple(items) if items.len() >= expected => {
            let mut names = Vec::with_capacity(expected);
            for item in items.iter().take(expected) {
                match item {
                    PtxOperand::Reg(r) => names.push(r.as_str()),
                    _ => {
                        return Err(CompileError::LoweringError(format!(
                            "expected register in {ctx_msg}"
                        )))
                    }
                }
            }
            Ok(names)
        }
        _ => Err(CompileError::LoweringError(format!(
            "expected {expected}-element register tuple for {ctx_msg}"
        ))),
    }
}

fn inline_ptx_call(
    inst: &PtxInstruction,
    pred: Option<IrPredGuard>,
    ctx: &mut LowerCtx<'_>,
    out: &mut Vec<IrInst>,
    inline_depth: usize,
) -> Result<(), CompileError> {
    if inline_depth >= 8 {
        return Err(CompileError::LoweringError(
            "max PTX .func inline depth exceeded (recursion not supported)".to_owned(),
        ));
    }

    let (ret_ops, func_name, arg_ops): (Vec<&PtxOperand>, &str, Vec<&PtxOperand>) =
        match inst.operands.as_slice() {
            [PtxOperand::Tuple(rets), PtxOperand::Ident(fname), PtxOperand::Tuple(args), ..] => (
                rets.iter().collect(),
                fname.as_str(),
                args.iter().collect(),
            ),
            [PtxOperand::Ident(fname), PtxOperand::Tuple(args), ..] => {
                (Vec::new(), fname.as_str(), args.iter().collect())
            }
            [PtxOperand::Ident(fname)] => (Vec::new(), fname.as_str(), Vec::new()),
            [PtxOperand::Reg(ret_r), PtxOperand::Ident(fname), PtxOperand::Tuple(args), ..] => {
                let _ = ret_r;
                (
                    vec![&inst.operands[0]],
                    fname.as_str(),
                    args.iter().collect(),
                )
            }
            _ => {
                return Err(CompileError::LoweringError(
                    "malformed PTX call instruction operands".to_owned(),
                ))
            }
        };

    let func = *ctx.func_map.get(func_name).ok_or_else(|| {
        CompileError::LoweringError(format!("unknown PTX .func '{func_name}' in call"))
    })?;

    ctx.inline_counter += 1;
    let id = ctx.inline_counter;
    let prefix = format!("__inline_{id}_{}_", sanitize_llvm_ident(func_name));
    let ret_label = format!("{prefix}ret");
    let skip_label = format!("{prefix}skip");

    // Predicated call guard
    if let Some(pg) = pred {
        out.push(IrInst {
            pred: Some(IrPredGuard {
                reg: pg.reg,
                negated: !pg.negated,
            }),
            op: IrOp::Branch(skip_label.clone()),
        });
    }

    let mut callee_reg_map: HashMap<String, u32> = HashMap::new();
    let mut local_in_params: HashMap<String, u32> = HashMap::new();
    let mut local_ret_params: HashMap<String, u32> = HashMap::new();

    for decl in &func.regs {
        let ir_ty = IrType::from_ptx(decl.ty);
        match decl.count {
            Some(n) => {
                for ridx in 0..n {
                    let cname = format!("{}{ridx}", decl.prefix);
                    let vreg = ctx.reg_types.len() as u32;
                    ctx.reg_types.push(ir_ty);
                    callee_reg_map.insert(cname, vreg);
                }
            }
            None => {
                let vreg = ctx.reg_types.len() as u32;
                ctx.reg_types.push(ir_ty);
                callee_reg_map.insert(decl.prefix.clone(), vreg);
            }
        }
    }

    // Bind input parameters into callee virtual registers
    for (p, caller_arg) in func.in_params.iter().zip(arg_ops.iter()) {
        let pty = IrType::from_ptx(p.ty);
        let arg_val = lower_operand(
            caller_arg,
            ctx.reg_map,
            ctx.reg_types,
            ctx.shared_offsets,
            ctx.local_offsets,
            pty,
        )?;
        let pvreg = ctx.reg_types.len() as u32;
        ctx.reg_types.push(pty);
        out.push(IrInst {
            pred: None,
            op: IrOp::Mov {
                dst: pvreg,
                ty: pty,
                src: arg_val,
            },
        });
        local_in_params.insert(p.name.clone(), pvreg);
        if p.name.starts_with('%') {
            callee_reg_map.insert(p.name.clone(), pvreg);
        }
    }

    // Allocate callee return parameter registers
    let mut ret_vregs = Vec::new();
    for p in &func.ret_params {
        let pty = IrType::from_ptx(p.ty);
        let rvreg = ctx.reg_types.len() as u32;
        ctx.reg_types.push(pty);
        local_ret_params.insert(p.name.clone(), rvreg);
        if p.name.starts_with('%') {
            callee_reg_map.insert(p.name.clone(), rvreg);
        }
        ret_vregs.push((rvreg, pty));
    }

    // Allocate callee `.local` scratch if present
    let mut callee_locals = ctx.local_offsets.clone();
    for l in &func.local_decls {
        let align = l.align.max(l.ty.byte_size()).max(1);
        *ctx.private_bytes = align_up_u32(*ctx.private_bytes, align);
        callee_locals.insert(l.name.clone(), *ctx.private_bytes);
        *ctx.private_bytes += l.ty.byte_size() * l.count.max(1);
    }
    *ctx.private_bytes = align_up_u32(*ctx.private_bytes, 16);

    let mut callee_spaces: HashMap<u32, AddressSpace> = HashMap::new();
    let empty_pmap: HashMap<String, u32> = HashMap::new();
    let mut sub_ctx = LowerCtx {
        func_map: ctx.func_map,
        reg_map: &mut callee_reg_map,
        reg_types: ctx.reg_types,
        reg_spaces: &mut callee_spaces,
        param_map: &empty_pmap,
        shared_offsets: ctx.shared_offsets,
        local_offsets: &mut callee_locals,
        private_bytes: ctx.private_bytes,
        inline_counter: ctx.inline_counter,
    };

    for stmt in &func.statements {
        match stmt {
            PtxStatement::Label(lbl) => out.push(IrInst {
                pred: None,
                op: IrOp::Label(format!("{prefix}{lbl}")),
            }),
            PtxStatement::Instruction(cinst) => {
                // Rewrite internal branch targets cleanly with inline prefix
                let mut cloned = cinst.clone();
                if cloned.opcode == "bra" {
                    if let Some(PtxOperand::Ident(lbl)) = cloned.operands.first_mut() {
                        *lbl = format!("{prefix}{lbl}");
                    }
                }
                lower_instruction(
                    &cloned,
                    &mut sub_ctx,
                    &local_in_params,
                    &local_ret_params,
                    Some(&ret_label),
                    out,
                    inline_depth + 1,
                )?;
            }
        }
    }
    ctx.inline_counter = sub_ctx.inline_counter;

    out.push(IrInst {
        pred: None,
        op: IrOp::Label(ret_label),
    });

    // Copy return registers back into caller destinations
    for (caller_ret, (rvreg, rty)) in ret_ops.iter().zip(ret_vregs.iter()) {
        if let PtxOperand::Reg(cname) = caller_ret {
            let dst = resolve_reg(ctx.reg_map, ctx.reg_types, cname, *rty);
            out.push(IrInst {
                pred: None,
                op: IrOp::Mov {
                    dst,
                    ty: *rty,
                    src: IrOperand::Reg(*rvreg),
                },
            });
        }
    }

    if pred.is_some() {
        out.push(IrInst {
            pred: None,
            op: IrOp::Label(skip_label),
        });
    }

    Ok(())
}

fn resolve_reg(
    reg_map: &mut HashMap<String, u32>,
    reg_types: &mut Vec<IrType>,
    name: &str,
    default_ty: IrType,
) -> u32 {
    if let Some(&id) = reg_map.get(name) {
        return id;
    }
    let id = reg_types.len() as u32;
    reg_map.insert(name.to_owned(), id);
    reg_types.push(default_ty);
    id
}

fn resolve_addr_base(
    base_name: &str,
    reg_map: &mut HashMap<String, u32>,
    reg_types: &mut Vec<IrType>,
    reg_spaces: &HashMap<u32, AddressSpace>,
    shared_offsets: &HashMap<String, u32>,
    local_offsets: &HashMap<String, u32>,
) -> Result<(IrOperand, AddressSpace), CompileError> {
    if let Some(&off) = shared_offsets.get(base_name) {
        return Ok((IrOperand::SharedOffset(off), AddressSpace::Shared));
    }
    if let Some(&off) = local_offsets.get(base_name) {
        return Ok((IrOperand::LocalOffset(off), AddressSpace::Private));
    }
    if base_name.starts_with('%') {
        let reg_id = resolve_reg(reg_map, reg_types, base_name, IrType::U64);
        let space = reg_spaces
            .get(&reg_id)
            .copied()
            .unwrap_or(AddressSpace::Generic);
        return Ok((IrOperand::Reg(reg_id), space));
    }
    Err(CompileError::LoweringError(format!(
        "unknown memory base '{base_name}'"
    )))
}

fn lower_operand(
    op: &PtxOperand,
    reg_map: &mut HashMap<String, u32>,
    reg_types: &mut Vec<IrType>,
    shared_offsets: &HashMap<String, u32>,
    local_offsets: &HashMap<String, u32>,
    hint_ty: IrType,
) -> Result<IrOperand, CompileError> {
    match op {
        PtxOperand::Reg(name) => {
            if let Some(sp) = SpecialReg::from_name(name) {
                Ok(IrOperand::Special(sp))
            } else {
                Ok(IrOperand::Reg(resolve_reg(
                    reg_map, reg_types, name, hint_ty,
                )))
            }
        }
        PtxOperand::ImmInt(v) => {
            if matches!(hint_ty, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64) || *v < 0 {
                Ok(IrOperand::ImmI64(*v as i64))
            } else {
                Ok(IrOperand::ImmU64(*v as u64))
            }
        }
        PtxOperand::ImmFloat(v) => Ok(IrOperand::ImmF64(*v)),
        PtxOperand::Ident(sym) => {
            if let Some(&off) = shared_offsets.get(sym) {
                Ok(IrOperand::SharedOffset(off))
            } else if let Some(&off) = local_offsets.get(sym) {
                Ok(IrOperand::LocalOffset(off))
            } else {
                Err(CompileError::LoweringError(format!(
                    "unresolved symbol operand '{sym}'"
                )))
            }
        }
        PtxOperand::NegatedReg(_) | PtxOperand::Mem { .. } | PtxOperand::Tuple(_) => Err(
            CompileError::LoweringError("unexpected negated/memory/tuple operand".to_owned()),
        ),
    }
}

fn expect_reg_operand<'a>(
    operands: &'a [PtxOperand],
    idx: usize,
    ctx: &str,
) -> Result<&'a str, CompileError> {
    match operands.get(idx) {
        Some(PtxOperand::Reg(r)) => Ok(r.as_str()),
        _ => Err(CompileError::LoweringError(format!(
            "expected register operand for {ctx}"
        ))),
    }
}

fn expect_mem_operand<'a>(
    operands: &'a [PtxOperand],
    idx: usize,
    ctx: &str,
) -> Result<(&'a str, i64), CompileError> {
    match operands.get(idx) {
        Some(PtxOperand::Mem { base, offset }) => Ok((base.as_str(), *offset)),
        _ => Err(CompileError::LoweringError(format!(
            "expected memory operand for {ctx}"
        ))),
    }
}

fn space_from_modifiers(mods: &[String]) -> Option<AddressSpace> {
    for m in mods {
        match m.as_str() {
            "global" => return Some(AddressSpace::Global),
            "shared" => return Some(AddressSpace::Shared),
            "const" => return Some(AddressSpace::Constant),
            "local" => return Some(AddressSpace::Private),
            _ => {}
        }
    }
    None
}

fn infer_inst_type(mods: &[String]) -> Option<IrType> {
    for m in mods.iter().rev() {
        if let Some(pt) = PtxType::from_modifier(m) {
            return Some(IrType::from_ptx(pt));
        }
    }
    None
}

fn infer_cvt_types(mods: &[String]) -> (IrType, IrType) {
    let mut found = Vec::new();
    for m in mods {
        if let Some(pt) = PtxType::from_modifier(m) {
            found.push(IrType::from_ptx(pt));
        }
    }
    match found.as_slice() {
        [dst, src, ..] => (*dst, *src),
        [only] => (*only, *only),
        [] => (IrType::U32, IrType::U32),
    }
}

const fn align_up_u32(val: u32, align: u32) -> u32 {
    if align <= 1 {
        val
    } else {
        (val + align - 1) & !(align - 1)
    }
}

/// Emits complete, opaque-pointer SSA LLVM IR targeting `amdgcn-amd-amdhsa` from an [`IrModule`].
pub fn emit_llvm_ir(module: &IrModule, target: IsaTarget) -> String {
    let mut out = String::with_capacity(8192);
    let _ = writeln!(out, "; ModuleID = 'amda_module'");
    let _ = writeln!(
        out,
        "target datalayout = \"e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9\""
    );
    let _ = writeln!(out, "target triple = \"amdgcn-amd-amdhsa\"");
    let _ = writeln!(out);

    // Intrinsic declarations
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.x() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.y() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.z() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.x() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.y() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.z() #1");
    let _ = writeln!(out, "declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.amdgcn.ballot.i32(i1) #1");
    let _ = writeln!(out, "declare i64 @llvm.amdgcn.ballot.i64(i1) #1");
    let _ = writeln!(out, "declare void @llvm.amdgcn.s.barrier() #2");
    let _ = writeln!(out, "declare float @llvm.fma.f32(float, float, float) #1");
    let _ = writeln!(out, "declare double @llvm.fma.f64(double, double, double) #1");
    let _ = writeln!(out, "declare float @llvm.fabs.f32(float) #1");
    let _ = writeln!(out, "declare double @llvm.fabs.f64(double) #1");
    let _ = writeln!(out, "declare float @llvm.sqrt.f32(float) #1");
    let _ = writeln!(out, "declare double @llvm.sqrt.f64(double) #1");
    let _ = writeln!(out, "declare float @llvm.exp2.f32(float) #1");
    let _ = writeln!(out, "declare float @llvm.log2.f32(float) #1");
    let _ = writeln!(out, "declare float @llvm.sin.f32(float) #1");
    let _ = writeln!(out, "declare float @llvm.cos.f32(float) #1");
    let _ = writeln!(out, "declare float @llvm.minnum.f32(float, float) #1");
    let _ = writeln!(out, "declare double @llvm.minnum.f64(double, double) #1");
    let _ = writeln!(out, "declare float @llvm.maxnum.f32(float, float) #1");
    let _ = writeln!(out, "declare double @llvm.maxnum.f64(double, double) #1");
    let _ = writeln!(out, "declare i32 @llvm.smin.i32(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.smax.i32(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.umin.i32(i32, i32) #1");
    let _ = writeln!(out, "declare i32 @llvm.umax.i32(i32, i32) #1");
    let _ = writeln!(out);

    // Self-contained AMDA math & quantization bitcode helpers
    emit_llvm_math_helpers(&mut out);

    for kernel in &module.kernels {
        emit_llvm_kernel(&mut out, kernel, target);
    }

    let wave_feat = if target.native_wave_size() == 32 {
        "+wavefrontsize32,-wavefrontsize64"
    } else {
        "-wavefrontsize32,+wavefrontsize64"
    };

    let _ = writeln!(
        out,
        "attributes #0 = {{ \"amdgpu-flat-work-group-size\"=\"1,1024\" \"target-cpu\"=\"{}\" \"target-features\"=\"{}\" \"uniform-work-group-size\"=\"true\" }}",
        target.as_str(),
        wave_feat
    );
    let _ = writeln!(out, "attributes #1 = {{ nocallback nofree nosync nounwind speculatable willreturn memory(none) }}");
    let _ = writeln!(out, "attributes #2 = {{ convergent nocallback nofree nounwind willreturn }}");

    out
}

struct LlvmEmitter<'a> {
    out: &'a mut String,
    kernel: &'a IrKernel,
    target: IsaTarget,
    tmp_id: usize,
    bb_id: usize,
    block_terminated: bool,
}

impl<'a> LlvmEmitter<'a> {
    fn new(out: &'a mut String, kernel: &'a IrKernel, target: IsaTarget) -> Self {
        Self {
            out,
            kernel,
            target,
            tmp_id: 0,
            bb_id: 0,
            block_terminated: false,
        }
    }

    fn next_tmp(&mut self) -> String {
        self.tmp_id += 1;
        format!("%t{}", self.tmp_id)
    }

    fn next_bb(&mut self, prefix: &str) -> String {
        self.bb_id += 1;
        format!("{prefix}_{}", self.bb_id)
    }

    fn ensure_open_block(&mut self) {
        if self.block_terminated {
            let lbl = self.next_bb("bb_dead");
            let _ = writeln!(self.out, "{lbl}:");
            self.block_terminated = false;
        }
    }

    fn reg_slot(reg: u32) -> String {
        format!("%reg_slot_{reg}")
    }

    fn reg_ty(&self, reg: u32) -> IrType {
        self.kernel
            .reg_types
            .get(reg as usize)
            .copied()
            .unwrap_or(IrType::U32)
    }

    fn load_reg(&mut self, reg: u32) -> (String, IrType) {
        let rty = self.reg_ty(reg);
        let tmp = self.next_tmp();
        let align = rty.byte_size().max(1);
        let slot = Self::reg_slot(reg);
        let _ = writeln!(
            self.out,
            "  {tmp} = load {}, ptr addrspace(5) {slot}, align {align}",
            rty.llvm_ty()
        );
        (tmp, rty)
    }

    fn store_reg(&mut self, reg: u32, val: &str, val_ty: IrType) {
        let dst_ty = self.reg_ty(reg);
        let coerced = self.coerce_val(val, val_ty, dst_ty);
        let align = dst_ty.byte_size().max(1);
        let slot = Self::reg_slot(reg);
        let _ = writeln!(
            self.out,
            "  store {} {coerced}, ptr addrspace(5) {slot}, align {align}",
            dst_ty.llvm_ty()
        );
    }

    fn coerce_val(&mut self, val: &str, from: IrType, to: IrType) -> String {
        if from == to || from.llvm_ty() == to.llvm_ty() {
            return val.to_owned();
        }
        let tmp = self.next_tmp();
        // Float <-> same-width integer bitcasts or sub-32 integer conversion
        match (from, to) {
            (IrType::F32, IrType::U32 | IrType::I32) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast float {val} to i32");
                return tmp;
            }
            (IrType::U32 | IrType::I32, IrType::F32) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast i32 {val} to float");
                return tmp;
            }
            (IrType::F64, IrType::U64 | IrType::I64) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast double {val} to i64");
                return tmp;
            }
            (IrType::U64 | IrType::I64, IrType::F64) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast i64 {val} to double");
                return tmp;
            }
            (IrType::F16, IrType::U16 | IrType::I16) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast half {val} to i16");
                return tmp;
            }
            (IrType::Bf16, IrType::U16 | IrType::I16) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast bfloat {val} to i16");
                return tmp;
            }
            (IrType::U16 | IrType::I16, IrType::F16) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast i16 {val} to half");
                return tmp;
            }
            (IrType::U16 | IrType::I16, IrType::Bf16) => {
                let _ = writeln!(self.out, "  {tmp} = bitcast i16 {val} to bfloat");
                return tmp;
            }
            (IrType::F16 | IrType::Bf16, IrType::U32 | IrType::I32) => {
                let b16 = self.next_tmp();
                let _ = writeln!(self.out, "  {b16} = bitcast {} {val} to i16", from.llvm_ty());
                let _ = writeln!(self.out, "  {tmp} = zext i16 {b16} to i32");
                return tmp;
            }
            (IrType::U32 | IrType::I32, IrType::F16 | IrType::Bf16) => {
                let tr = self.next_tmp();
                let _ = writeln!(self.out, "  {tr} = trunc i32 {val} to i16");
                let _ = writeln!(self.out, "  {tmp} = bitcast i16 {tr} to {}", to.llvm_ty());
                return tmp;
            }
            (IrType::F16, IrType::Bf16) | (IrType::Bf16, IrType::F16) => {
                let flt = self.next_tmp();
                let _ = writeln!(self.out, "  {flt} = fpext {} {val} to float", from.llvm_ty());
                let _ = writeln!(self.out, "  {tmp} = fptrunc float {flt} to {}", to.llvm_ty());
                return tmp;
            }
            _ => {}
        }

        if from == IrType::Pred && !to.is_float_ty() {
            let _ = writeln!(self.out, "  {tmp} = zext i1 {val} to {}", to.llvm_ty());
            return tmp;
        }
        if !from.is_float_ty() && to == IrType::Pred {
            let _ = writeln!(self.out, "  {tmp} = trunc {} {val} to i1", from.llvm_ty());
            return tmp;
        }
        if !from.is_float_ty() && !to.is_float_ty() {
            let fb = from.byte_size();
            let tb = to.byte_size();
            if fb < tb {
                let op = if from.is_signed_int_ty() { "sext" } else { "zext" };
                let _ = writeln!(
                    self.out,
                    "  {tmp} = {op} {} {val} to {}",
                    from.llvm_ty(),
                    to.llvm_ty()
                );
            } else {
                let _ = writeln!(
                    self.out,
                    "  {tmp} = trunc {} {val} to {}",
                    from.llvm_ty(),
                    to.llvm_ty()
                );
            }
            return tmp;
        }
        if from.is_float_ty() && to.is_float_ty() {
            let op = if from.byte_size() < to.byte_size() {
                "fpext"
            } else {
                "fptrunc"
            };
            let _ = writeln!(
                self.out,
                "  {tmp} = {op} {} {val} to {}",
                from.llvm_ty(),
                to.llvm_ty()
            );
            return tmp;
        }

        let _ = writeln!(
            self.out,
            "  ; fallback coerce {} -> {}",
            from.llvm_ty(),
            to.llvm_ty()
        );
        zero_literal(to)
    }

    fn eval_operand(&mut self, op: &IrOperand, expected_ty: IrType) -> String {
        match op {
            IrOperand::Reg(r) => {
                let (v, ty) = self.load_reg(*r);
                self.coerce_val(&v, ty, expected_ty)
            }
            IrOperand::ImmU64(v) => match expected_ty {
                IrType::F16 => {
                    let tmp = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = fptrunc float {} to half",
                        format_llvm_f32(*v as f32)
                    );
                    tmp
                }
                IrType::Bf16 => {
                    let tmp = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = fptrunc float {} to bfloat",
                        format_llvm_f32(*v as f32)
                    );
                    tmp
                }
                IrType::F32 => format_llvm_f32(*v as f32),
                IrType::F64 => format_llvm_f64(*v as f64),
                IrType::Pred => if (*v & 1) != 0 { "true" } else { "false" }.to_owned(),
                IrType::U8 | IrType::I8 => (*v as u8 as i8).to_string(),
                IrType::U16 | IrType::I16 => (*v as u16 as i16).to_string(),
                IrType::U32 | IrType::I32 => (*v as u32 as i32).to_string(),
                IrType::U64 | IrType::I64 => (*v as i64).to_string(),
            },
            IrOperand::ImmI64(v) => match expected_ty {
                IrType::F16 => {
                    let tmp = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = fptrunc float {} to half",
                        format_llvm_f32(*v as f32)
                    );
                    tmp
                }
                IrType::Bf16 => {
                    let tmp = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = fptrunc float {} to bfloat",
                        format_llvm_f32(*v as f32)
                    );
                    tmp
                }
                IrType::F32 => format_llvm_f32(*v as f32),
                IrType::F64 => format_llvm_f64(*v as f64),
                IrType::Pred => if *v != 0 { "true" } else { "false" }.to_owned(),
                IrType::U8 | IrType::I8 => (*v as i8).to_string(),
                IrType::U16 | IrType::I16 => (*v as i16).to_string(),
                IrType::U32 | IrType::I32 => (*v as i32).to_string(),
                IrType::U64 | IrType::I64 => v.to_string(),
            },
            IrOperand::ImmF64(f) => {
                let raw = match expected_ty {
                    IrType::F16 => {
                        let tmp = self.next_tmp();
                        let _ = writeln!(self.out, "  {tmp} = fptrunc float {} to half", format_llvm_f32(*f as f32));
                        return tmp;
                    }
                    IrType::Bf16 => {
                        let tmp = self.next_tmp();
                        let _ = writeln!(self.out, "  {tmp} = fptrunc float {} to bfloat", format_llvm_f32(*f as f32));
                        return tmp;
                    }
                    IrType::F32 => format_llvm_f32(*f as f32),
                    IrType::F64 => format_llvm_f64(*f),
                    IrType::U8 | IrType::I8 => (*f as i8).to_string(),
                    IrType::U16 | IrType::I16 => (*f as i16).to_string(),
                    IrType::U32 | IrType::I32 => {
                        let bits = (*f as f32).to_bits() as i32;
                        bits.to_string()
                    }
                    IrType::U64 | IrType::I64 => (f.to_bits() as i64).to_string(),
                    IrType::Pred => if *f != 0.0 { "true" } else { "false" }.to_owned(),
                };
                raw
            }
            IrOperand::Special(sp) => {
                let val_i32 = self.emit_special(*sp);
                self.coerce_val(&val_i32, IrType::U32, expected_ty)
            }
            IrOperand::Param(idx) => {
                if let Some(p) = self.kernel.params.get(*idx as usize) {
                    if p.is_global_ptr {
                        let tmp = self.next_tmp();
                        let _ = writeln!(
                            self.out,
                            "  {tmp} = ptrtoint ptr addrspace(1) %arg_{} to i64",
                            p.name
                        );
                        self.coerce_val(&tmp, IrType::U64, expected_ty)
                    } else {
                        self.coerce_val(&format!("%arg_{}", p.name), p.ty, expected_ty)
                    }
                } else {
                    zero_literal(expected_ty)
                }
            }
            IrOperand::SharedOffset(off) | IrOperand::LocalOffset(off) => {
                self.coerce_val(&off.to_string(), IrType::U32, expected_ty)
            }
        }
    }

    fn emit_special(&mut self, sp: SpecialReg) -> String {
        let tmp = self.next_tmp();
        match sp {
            SpecialReg::TidX => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workitem.id.x()");
            }
            SpecialReg::TidY => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workitem.id.y()");
            }
            SpecialReg::TidZ => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workitem.id.z()");
            }
            SpecialReg::CtaIdX => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workgroup.id.x()");
            }
            SpecialReg::CtaIdY => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workgroup.id.y()");
            }
            SpecialReg::CtaIdZ => {
                let _ = writeln!(self.out, "  {tmp} = call i32 @llvm.amdgcn.workgroup.id.z()");
            }
            SpecialReg::NtidX | SpecialReg::NtidY | SpecialReg::NtidZ => {
                let byte_off = match sp {
                    SpecialReg::NtidX => 4,
                    SpecialReg::NtidY => 6,
                    _ => 8,
                };
                let dp = self.next_tmp();
                let gep = self.next_tmp();
                let w16 = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {dp} = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()"
                );
                let _ = writeln!(
                    self.out,
                    "  {gep} = getelementptr inbounds i8, ptr addrspace(4) {dp}, i32 {byte_off}"
                );
                let _ = writeln!(
                    self.out,
                    "  {w16} = load i16, ptr addrspace(4) {gep}, align 2"
                );
                let _ = writeln!(self.out, "  {tmp} = zext i16 {w16} to i32");
            }
            SpecialReg::NctaIdX | SpecialReg::NctaIdY | SpecialReg::NctaIdZ => {
                let (grid_off, wg_off) = match sp {
                    SpecialReg::NctaIdX => (12, 4),
                    SpecialReg::NctaIdY => (16, 6),
                    _ => (20, 8),
                };
                let dp = self.next_tmp();
                let g_gep = self.next_tmp();
                let grid_sz = self.next_tmp();
                let w_gep = self.next_tmp();
                let w16 = self.next_tmp();
                let wg_sz = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {dp} = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()"
                );
                let _ = writeln!(
                    self.out,
                    "  {g_gep} = getelementptr inbounds i8, ptr addrspace(4) {dp}, i32 {grid_off}"
                );
                let _ = writeln!(
                    self.out,
                    "  {grid_sz} = load i32, ptr addrspace(4) {g_gep}, align 4"
                );
                let _ = writeln!(
                    self.out,
                    "  {w_gep} = getelementptr inbounds i8, ptr addrspace(4) {dp}, i32 {wg_off}"
                );
                let _ = writeln!(
                    self.out,
                    "  {w16} = load i16, ptr addrspace(4) {w_gep}, align 2"
                );
                let _ = writeln!(self.out, "  {wg_sz} = zext i16 {w16} to i32");
                let _ = writeln!(self.out, "  {tmp} = udiv i32 {grid_sz}, {wg_sz}");
            }
            SpecialReg::LaneId => {
                if self.target.native_wave_size() == 32 {
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"
                    );
                } else {
                    let lo = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {lo} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"
                    );
                    let _ = writeln!(
                        self.out,
                        "  {tmp} = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 {lo})"
                    );
                }
            }
        }
        tmp
    }

    fn emit_inst(&mut self, inst: &IrInst) {
        // Labels open a new block unconditionally
        if let IrOp::Label(lbl) = &inst.op {
            let clean = sanitize_llvm_ident(lbl);
            if !self.block_terminated {
                let _ = writeln!(self.out, "  br label %{clean}");
            }
            let _ = writeln!(self.out, "{clean}:");
            self.block_terminated = false;
            return;
        }

        self.ensure_open_block();

        // Branch handles its own predicate directly via `br i1`
        if let IrOp::Branch(lbl) = &inst.op {
            let clean = sanitize_llvm_ident(lbl);
            if let Some(pg) = inst.pred {
                let (p_val, p_ty) = self.load_reg(pg.reg);
                let cond = self.coerce_val(&p_val, p_ty, IrType::Pred);
                let cont = self.next_bb("bb_cont");
                if pg.negated {
                    let _ = writeln!(
                        self.out,
                        "  br i1 {cond}, label %{cont}, label %{clean}"
                    );
                } else {
                    let _ = writeln!(
                        self.out,
                        "  br i1 {cond}, label %{clean}, label %{cont}"
                    );
                }
                let _ = writeln!(self.out, "{cont}:");
                self.block_terminated = false;
            } else {
                let _ = writeln!(self.out, "  br label %{clean}");
                self.block_terminated = true;
            }
            return;
        }

        // Predicated non-branch instruction: emit conditional then/merge blocks
        let merge_lbl = if let Some(pg) = inst.pred {
            let (p_val, p_ty) = self.load_reg(pg.reg);
            let cond = self.coerce_val(&p_val, p_ty, IrType::Pred);
            let then_lbl = self.next_bb("bb_pred_then");
            let merge = self.next_bb("bb_pred_merge");
            if pg.negated {
                let _ = writeln!(
                    self.out,
                    "  br i1 {cond}, label %{merge}, label %{then_lbl}"
                );
            } else {
                let _ = writeln!(
                    self.out,
                    "  br i1 {cond}, label %{then_lbl}, label %{merge}"
                );
            }
            let _ = writeln!(self.out, "{then_lbl}:");
            Some(merge)
        } else {
            None
        };

        self.emit_op(&inst.op);

        if let Some(merge) = merge_lbl {
            if !self.block_terminated {
                let _ = writeln!(self.out, "  br label %{merge}");
            }
            let _ = writeln!(self.out, "{merge}:");
            self.block_terminated = false;
        }
    }

    fn emit_op(&mut self, op: &IrOp) {
        match op {
            IrOp::Label(_) | IrOp::Branch(_) => {}
            IrOp::LoadParam {
                dst,
                ty,
                param_idx,
                ..
            } => {
                if let Some(param) = self.kernel.params.get(*param_idx as usize) {
                    if param.is_global_ptr {
                        let tmp = self.next_tmp();
                        let _ = writeln!(
                            self.out,
                            "  {tmp} = ptrtoint ptr addrspace(1) %arg_{} to i64",
                            param.name
                        );
                        self.store_reg(*dst, &tmp, IrType::U64);
                    } else {
                        let arg_name = format!("%arg_{}", param.name);
                        let coerced = self.coerce_val(&arg_name, param.ty, *ty);
                        self.store_reg(*dst, &coerced, *ty);
                    }
                }
            }
            IrOp::CvtaToGlobal { dst, src } | IrOp::CvtaToShared { dst, src } => {
                let val = self.eval_operand(src, IrType::U64);
                self.store_reg(*dst, &val, IrType::U64);
            }
            IrOp::Mov { dst, ty, src } => {
                let val = self.eval_operand(src, *ty);
                self.store_reg(*dst, &val, *ty);
            }
            IrOp::Cvt {
                dst,
                dst_ty,
                src_ty,
                src,
            } => {
                let s = self.eval_operand(src, *src_ty);
                let res = self.emit_cvt(*dst_ty, *src_ty, &s);
                self.store_reg(*dst, &res, *dst_ty);
            }
            IrOp::Neg { dst, ty, src } => {
                let s = self.eval_operand(src, *ty);
                let tmp = self.next_tmp();
                if ty.is_float_ty() {
                    let _ = writeln!(self.out, "  {tmp} = fneg {} {s}", ty.llvm_ty());
                } else {
                    let _ = writeln!(self.out, "  {tmp} = sub {} 0, {s}", ty.llvm_ty());
                }
                self.store_reg(*dst, &tmp, *ty);
            }
            IrOp::Abs { dst, ty, src } => {
                let s = self.eval_operand(src, *ty);
                let tmp = self.next_tmp();
                match ty {
                    IrType::F32 => {
                        let _ = writeln!(self.out, "  {tmp} = call float @llvm.fabs.f32(float {s})");
                    }
                    IrType::F64 => {
                        let _ = writeln!(self.out, "  {tmp} = call double @llvm.fabs.f64(double {s})");
                    }
                    _ => {
                        let neg = self.next_tmp();
                        let cond = self.next_tmp();
                        let lt = ty.llvm_ty();
                        let _ = writeln!(self.out, "  {neg} = sub {lt} 0, {s}");
                        let _ = writeln!(self.out, "  {cond} = icmp slt {lt} {s}, 0");
                        let _ = writeln!(
                            self.out,
                            "  {tmp} = select i1 {cond}, {lt} {neg}, {lt} {s}"
                        );
                    }
                }
                self.store_reg(*dst, &tmp, *ty);
            }
            IrOp::Bin {
                op,
                dst,
                ty,
                lhs,
                rhs,
            } => {
                if *op == BinOp::MulWide {
                    let l32 = self.eval_operand(lhs, *ty);
                    let r32 = self.eval_operand(rhs, *ty);
                    let ext_op = if *ty == IrType::I32 { "sext" } else { "zext" };
                    let dst_ty = if *ty == IrType::I32 {
                        IrType::I64
                    } else {
                        IrType::U64
                    };
                    let l64 = self.next_tmp();
                    let r64 = self.next_tmp();
                    let res = self.next_tmp();
                    let _ = writeln!(self.out, "  {l64} = {ext_op} i32 {l32} to i64");
                    let _ = writeln!(self.out, "  {r64} = {ext_op} i32 {r32} to i64");
                    let _ = writeln!(self.out, "  {res} = mul i64 {l64}, {r64}");
                    self.store_reg(*dst, &res, dst_ty);
                    return;
                }

                let l = self.eval_operand(lhs, *ty);
                let r = self.eval_operand(rhs, *ty);
                let lt = ty.llvm_ty();
                let res = self.next_tmp();
                if matches!(op, BinOp::Min | BinOp::Max) {
                    let is_min = *op == BinOp::Min;
                    match ty {
                        IrType::F32 => {
                            let intr = if is_min { "@llvm.minnum.f32" } else { "@llvm.maxnum.f32" };
                            let _ = writeln!(self.out, "  {res} = call float {intr}(float {l}, float {r})");
                        }
                        IrType::F64 => {
                            let intr = if is_min { "@llvm.minnum.f64" } else { "@llvm.maxnum.f64" };
                            let _ = writeln!(self.out, "  {res} = call double {intr}(double {l}, double {r})");
                        }
                        _ => {
                            let intr = match (is_min, ty.is_signed_int_ty()) {
                                (true, true) => "@llvm.smin.i32",
                                (true, false) => "@llvm.umin.i32",
                                (false, true) => "@llvm.smax.i32",
                                (false, false) => "@llvm.umax.i32",
                            };
                            let l32 = self.coerce_val(&l, *ty, IrType::U32);
                            let r32 = self.coerce_val(&r, *ty, IrType::U32);
                            let r_i32 = self.next_tmp();
                            let _ = writeln!(self.out, "  {r_i32} = call i32 {intr}(i32 {l32}, i32 {r32})");
                            self.store_reg(*dst, &r_i32, IrType::U32);
                            return;
                        }
                    }
                    self.store_reg(*dst, &res, *ty);
                    return;
                }
                let instr = match (op, ty.is_float_ty(), ty.is_signed_int_ty()) {
                    (BinOp::Add, true, _) => "fadd",
                    (BinOp::Add, false, _) => "add",
                    (BinOp::Sub, true, _) => "fsub",
                    (BinOp::Sub, false, _) => "sub",
                    (BinOp::Mul, true, _) => "fmul",
                    (BinOp::Mul, false, _) => "mul",
                    (BinOp::Div, true, _) => "fdiv",
                    (BinOp::Div, false, true) => "sdiv",
                    (BinOp::Div, false, false) => "udiv",
                    (BinOp::Rem, true, _) => "frem",
                    (BinOp::Rem, false, true) => "srem",
                    (BinOp::Rem, false, false) => "urem",
                    (BinOp::Shl, false, _) => "shl",
                    (BinOp::Shr, false, true) => "ashr",
                    (BinOp::Shr, false, false) => "lshr",
                    (BinOp::And, false, _) => "and",
                    (BinOp::Or, false, _) => "or",
                    (BinOp::Xor, false, _) => "xor",
                    _ => "add",
                };
                let _ = writeln!(self.out, "  {res} = {instr} {lt} {l}, {r}");
                self.store_reg(*dst, &res, *ty);
            }
            IrOp::UnaryMath { op, dst, ty, src } => {
                let use_f64 = *ty == IrType::F64;
                let s = if use_f64 {
                    self.eval_operand(src, IrType::F64)
                } else {
                    self.eval_operand(src, IrType::F32)
                };
                let res = self.next_tmp();
                let out_ty = if use_f64 { IrType::F64 } else { IrType::F32 };
                match (op, use_f64) {
                    (UnaryMathOp::Rcp, false) => {
                        let _ = writeln!(self.out, "  {res} = fdiv float 1.0, {s}");
                    }
                    (UnaryMathOp::Rcp, true) => {
                        let _ = writeln!(self.out, "  {res} = fdiv double 1.0, {s}");
                    }
                    (UnaryMathOp::Sqrt, false) => {
                        let _ = writeln!(self.out, "  {res} = call float @llvm.sqrt.f32(float {s})");
                    }
                    (UnaryMathOp::Sqrt, true) => {
                        let _ = writeln!(self.out, "  {res} = call double @llvm.sqrt.f64(double {s})");
                    }
                    (UnaryMathOp::Rsqrt, false) => {
                        let _ = writeln!(self.out, "  {res} = call float @__amda_rsqrt_f32(float {s})");
                    }
                    (UnaryMathOp::Rsqrt, true) => {
                        let sq = self.next_tmp();
                        let _ = writeln!(self.out, "  {sq} = call double @llvm.sqrt.f64(double {s})");
                        let _ = writeln!(self.out, "  {res} = fdiv double 1.0, {sq}");
                    }
                    (UnaryMathOp::Ex2, _) => {
                        let sf = if use_f64 {
                            let t = self.next_tmp();
                            let _ = writeln!(self.out, "  {t} = fptrunc double {s} to float");
                            t
                        } else {
                            s
                        };
                        let _ = writeln!(self.out, "  {res} = call float @llvm.exp2.f32(float {sf})");
                        self.store_reg(*dst, &res, IrType::F32);
                        return;
                    }
                    (UnaryMathOp::Lg2, _) => {
                        let sf = if use_f64 {
                            let t = self.next_tmp();
                            let _ = writeln!(self.out, "  {t} = fptrunc double {s} to float");
                            t
                        } else {
                            s
                        };
                        let _ = writeln!(self.out, "  {res} = call float @llvm.log2.f32(float {sf})");
                        self.store_reg(*dst, &res, IrType::F32);
                        return;
                    }
                    (UnaryMathOp::Sin, _) => {
                        let _ = writeln!(self.out, "  {res} = call float @llvm.sin.f32(float {s})");
                        self.store_reg(*dst, &res, IrType::F32);
                        return;
                    }
                    (UnaryMathOp::Cos, _) => {
                        let _ = writeln!(self.out, "  {res} = call float @llvm.cos.f32(float {s})");
                        self.store_reg(*dst, &res, IrType::F32);
                        return;
                    }
                    (UnaryMathOp::Tanh, _) => {
                        let _ = writeln!(self.out, "  {res} = call float @__amda_tanh_f32(float {s})");
                        self.store_reg(*dst, &res, IrType::F32);
                        return;
                    }
                }
                self.store_reg(*dst, &res, out_ty);
            }
            IrOp::Bfe {
                dst,
                signed,
                src,
                pos,
                len,
            } => {
                let s = self.eval_operand(src, IrType::U32);
                let p = self.eval_operand(pos, IrType::U32);
                let l = self.eval_operand(len, IrType::U32);
                let shl_amt = self.next_tmp();
                let pre = self.next_tmp();
                let shr_amt = self.next_tmp();
                let sum = self.next_tmp();
                let res = self.next_tmp();
                let _ = writeln!(self.out, "  {sum} = add i32 {p}, {l}");
                let _ = writeln!(self.out, "  {shl_amt} = sub i32 32, {sum}");
                let _ = writeln!(self.out, "  {pre} = shl i32 {s}, {shl_amt}");
                let _ = writeln!(self.out, "  {shr_amt} = sub i32 32, {l}");
                let shr_op = if *signed { "ashr" } else { "lshr" };
                let _ = writeln!(self.out, "  {res} = {shr_op} i32 {pre}, {shr_amt}");
                self.store_reg(*dst, &res, IrType::U32);
            }
            IrOp::Bfi {
                dst,
                insert,
                base,
                pos,
                len,
            } => {
                let ins = self.eval_operand(insert, IrType::U32);
                let b = self.eval_operand(base, IrType::U32);
                let p = self.eval_operand(pos, IrType::U32);
                let l = self.eval_operand(len, IrType::U32);
                let m1 = self.next_tmp();
                let m2 = self.next_tmp();
                let pos_mask = self.next_tmp();
                let not_mask = self.next_tmp();
                let ins_masked = self.next_tmp();
                let ins_sh = self.next_tmp();
                let b_clr = self.next_tmp();
                let res = self.next_tmp();
                let _ = writeln!(self.out, "  {m1} = shl i32 1, {l}");
                let _ = writeln!(self.out, "  {m2} = sub i32 {m1}, 1");
                let _ = writeln!(self.out, "  {pos_mask} = shl i32 {m2}, {p}");
                let _ = writeln!(self.out, "  {not_mask} = xor i32 {pos_mask}, -1");
                let _ = writeln!(self.out, "  {b_clr} = and i32 {b}, {not_mask}");
                let _ = writeln!(self.out, "  {ins_masked} = and i32 {ins}, {m2}");
                let _ = writeln!(self.out, "  {ins_sh} = shl i32 {ins_masked}, {p}");
                let _ = writeln!(self.out, "  {res} = or i32 {b_clr}, {ins_sh}");
                self.store_reg(*dst, &res, IrType::U32);
            }
            IrOp::Prmt { dst, a, b, sel } => {
                let va = self.eval_operand(a, IrType::U32);
                let vb = self.eval_operand(b, IrType::U32);
                let vs = self.eval_operand(sel, IrType::U32);
                let res = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {res} = call i32 @__amda_prmt_b32(i32 {va}, i32 {vb}, i32 {vs})"
                );
                self.store_reg(*dst, &res, IrType::U32);
            }
            IrOp::Dp4a {
                dst,
                a_signed,
                b_signed,
                a,
                b,
                c,
            } => {
                let va = self.eval_operand(a, IrType::U32);
                let vb = self.eval_operand(b, IrType::U32);
                let vc = self.eval_operand(c, IrType::I32);
                let res = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {res} = call i32 @__amda_dp4a(i32 {va}, i32 {vb}, i32 {vc}, i1 {a_signed}, i1 {b_signed})"
                );
                self.store_reg(*dst, &res, IrType::I32);
            }
            IrOp::Mad { dst, ty, a, b, c } => {
                let va = self.eval_operand(a, *ty);
                let vb = self.eval_operand(b, *ty);
                let vc = self.eval_operand(c, *ty);
                let res = self.next_tmp();
                match ty {
                    IrType::F32 => {
                        let _ = writeln!(
                            self.out,
                            "  {res} = call float @llvm.fma.f32(float {va}, float {vb}, float {vc})"
                        );
                    }
                    IrType::F64 => {
                        let _ = writeln!(
                            self.out,
                            "  {res} = call double @llvm.fma.f64(double {va}, double {vb}, double {vc})"
                        );
                    }
                    _ => {
                        let lt = ty.llvm_ty();
                        let prod = self.next_tmp();
                        let _ = writeln!(self.out, "  {prod} = mul {lt} {va}, {vb}");
                        let _ = writeln!(self.out, "  {res} = add {lt} {prod}, {vc}");
                    }
                }
                self.store_reg(*dst, &res, *ty);
            }
            IrOp::Setp {
                cmp,
                dst,
                ty,
                lhs,
                rhs,
            } => {
                let l = self.eval_operand(lhs, *ty);
                let r = self.eval_operand(rhs, *ty);
                let lt = ty.llvm_ty();
                let res = self.next_tmp();
                if ty.is_float_ty() {
                    let pred = match cmp {
                        CmpOp::Eq => "oeq",
                        CmpOp::Ne => "one",
                        CmpOp::Lt => "olt",
                        CmpOp::Le => "ole",
                        CmpOp::Gt => "ogt",
                        CmpOp::Ge => "oge",
                    };
                    let _ = writeln!(self.out, "  {res} = fcmp {pred} {lt} {l}, {r}");
                } else {
                    let signed = ty.is_signed_int_ty();
                    let pred = match (cmp, signed) {
                        (CmpOp::Eq, _) => "eq",
                        (CmpOp::Ne, _) => "ne",
                        (CmpOp::Lt, true) => "slt",
                        (CmpOp::Lt, false) => "ult",
                        (CmpOp::Le, true) => "sle",
                        (CmpOp::Le, false) => "ule",
                        (CmpOp::Gt, true) => "sgt",
                        (CmpOp::Gt, false) => "ugt",
                        (CmpOp::Ge, true) => "sge",
                        (CmpOp::Ge, false) => "uge",
                    };
                    let _ = writeln!(self.out, "  {res} = icmp {pred} {lt} {l}, {r}");
                }
                self.store_reg(*dst, &res, IrType::Pred);
            }
            IrOp::Selp {
                dst,
                ty,
                a,
                b,
                pred,
            } => {
                let va = self.eval_operand(a, *ty);
                let vb = self.eval_operand(b, *ty);
                let (p_val, p_ty) = self.load_reg(*pred);
                let cond = self.coerce_val(&p_val, p_ty, IrType::Pred);
                let lt = ty.llvm_ty();
                let res = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {res} = select i1 {cond}, {lt} {va}, {lt} {vb}"
                );
                self.store_reg(*dst, &res, *ty);
            }
            IrOp::Load {
                dst,
                ty,
                space,
                base,
                offset,
            } => {
                let (ptr_val, ptr_ty_str) = self.emit_addr_ptr(*space, base, *offset);
                let lt = ty.llvm_ty();
                let align = ty.byte_size().max(1);
                let res = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {res} = load {lt}, {ptr_ty_str} {ptr_val}, align {align}"
                );
                self.store_reg(*dst, &res, *ty);
            }
            IrOp::Store {
                ty,
                space,
                base,
                offset,
                src,
            } => {
                let (ptr_val, ptr_ty_str) = self.emit_addr_ptr(*space, base, *offset);
                let val = self.eval_operand(src, *ty);
                let lt = ty.llvm_ty();
                let align = ty.byte_size().max(1);
                let _ = writeln!(
                    self.out,
                    "  store {lt} {val}, {ptr_ty_str} {ptr_val}, align {align}"
                );
            }
            IrOp::Barrier => {
                let _ = writeln!(self.out, "  call void @llvm.amdgcn.s.barrier()");
            }
            IrOp::MemBar => {
                let _ = writeln!(self.out, "  fence syncscope(\"agent\") seq_cst");
            }
            IrOp::Atomic {
                op,
                dst,
                ty,
                space,
                base,
                offset,
                val,
                cmp,
            } => {
                let (ptr_val, ptr_ty_str) = self.emit_addr_ptr(*space, base, *offset);
                let lt = ty.llvm_ty();
                let v = self.eval_operand(val, *ty);
                let res = self.next_tmp();
                if *op == AtomicOp::Cas {
                    let cval = cmp
                        .as_ref()
                        .map(|c| self.eval_operand(c, *ty))
                        .unwrap_or_else(|| zero_literal(*ty));
                    let pair = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {pair} = cmpxchg {ptr_ty_str} {ptr_val}, {lt} {cval}, {lt} {v} acq_rel monotonic"
                    );
                    let _ = writeln!(self.out, "  {res} = extractvalue {{ {lt}, i1 }} {pair}, 0");
                } else {
                    let rmw = match (op, ty.is_float_ty(), ty.is_signed_int_ty()) {
                        (AtomicOp::Add, true, _) => "fadd",
                        (AtomicOp::Add, false, _) => "add",
                        (AtomicOp::Min, false, true) => "min",
                        (AtomicOp::Min, false, false) => "umin",
                        (AtomicOp::Max, false, true) => "max",
                        (AtomicOp::Max, false, false) => "umax",
                        (AtomicOp::And, false, _) => "and",
                        (AtomicOp::Or, false, _) => "or",
                        (AtomicOp::Xor, false, _) => "xor",
                        (AtomicOp::Exch, _, _) => "xchg",
                        _ => "add",
                    };
                    let _ = writeln!(
                        self.out,
                        "  {res} = atomicrmw {rmw} {ptr_ty_str} {ptr_val}, {lt} {v} acq_rel"
                    );
                }
                if let Some(d) = dst {
                    self.store_reg(*d, &res, *ty);
                }
            }
            IrOp::WarpShuffle {
                dst,
                mode,
                src,
                lane_delta,
                width,
            } => {
                let val = self.eval_operand(src, IrType::U32);
                let delta = self.eval_operand(lane_delta, IrType::U32);
                let hw_lane = self.emit_special(SpecialReg::LaneId);
                let emit_target_lane = |em: &mut Self, base_lane: &str| -> String {
                    match mode {
                        ShuffleMode::Down => {
                            let t = em.next_tmp();
                            let ok = em.next_tmp();
                            let c = em.next_tmp();
                            let _ = writeln!(em.out, "  {t} = add i32 {base_lane}, {delta}");
                            let _ = writeln!(em.out, "  {ok} = icmp ult i32 {t}, {width}");
                            let _ = writeln!(em.out, "  {c} = select i1 {ok}, i32 {t}, i32 {base_lane}");
                            c
                        }
                        ShuffleMode::Up => {
                            let t = em.next_tmp();
                            let ok = em.next_tmp();
                            let c = em.next_tmp();
                            let _ = writeln!(em.out, "  {t} = sub i32 {base_lane}, {delta}");
                            let _ = writeln!(em.out, "  {ok} = icmp uge i32 {base_lane}, {delta}");
                            let _ = writeln!(em.out, "  {c} = select i1 {ok}, i32 {t}, i32 {base_lane}");
                            c
                        }
                        ShuffleMode::Bfly => {
                            let t = em.next_tmp();
                            let _ = writeln!(em.out, "  {t} = xor i32 {base_lane}, {delta}");
                            t
                        }
                        ShuffleMode::Idx => delta.clone(),
                    }
                };
                let lane_addr = if self.kernel.wave_mode == WaveMode::EmulatedWave32OnWave64 {
                    let sub_base = self.next_tmp();
                    let v_lane = self.next_tmp();
                    let _ = writeln!(self.out, "  {sub_base} = and i32 {hw_lane}, -32");
                    let _ = writeln!(self.out, "  {v_lane} = and i32 {hw_lane}, 31");
                    let chosen = emit_target_lane(self, &v_lane);
                    let phys = self.next_tmp();
                    let addr = self.next_tmp();
                    let _ = writeln!(self.out, "  {phys} = or i32 {sub_base}, {chosen}");
                    let _ = writeln!(self.out, "  {addr} = shl i32 {phys}, 2");
                    addr
                } else {
                    let chosen = emit_target_lane(self, &hw_lane);
                    let addr = self.next_tmp();
                    let _ = writeln!(self.out, "  {addr} = shl i32 {chosen}, 2");
                    addr
                };
                let res = self.next_tmp();
                let _ = writeln!(
                    self.out,
                    "  {res} = call i32 @llvm.amdgcn.ds.bpermute(i32 {lane_addr}, i32 {val})"
                );
                self.store_reg(*dst, &res, IrType::U32);
            }
            IrOp::WarpVote {
                mode,
                dst,
                pred,
                negated,
            } => {
                let (pv, pty) = self.load_reg(*pred);
                let b_raw = self.coerce_val(&pv, pty, IrType::Pred);
                let b_val = if *negated {
                    let inv = self.next_tmp();
                    let _ = writeln!(self.out, "  {inv} = xor i1 {b_raw}, true");
                    inv
                } else {
                    b_raw
                };
                let wave32 = self.kernel.metadata.wavefront_size == 32;
                let (ballot32, active_shift64) = if wave32 {
                    let b32 = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {b32} = call i32 @llvm.amdgcn.ballot.i32(i1 {b_val})"
                    );
                    (b32, None)
                } else {
                    let mask64 = self.next_tmp();
                    let _ = writeln!(
                        self.out,
                        "  {mask64} = call i64 @llvm.amdgcn.ballot.i64(i1 {b_val})"
                    );
                    if self.kernel.wave_mode == WaveMode::EmulatedWave32OnWave64 {
                        let hw_lane = self.emit_special(SpecialReg::LaneId);
                        let sh = self.next_tmp();
                        let sh64 = self.next_tmp();
                        let sft = self.next_tmp();
                        let t32 = self.next_tmp();
                        let _ = writeln!(self.out, "  {sh} = and i32 {hw_lane}, 32");
                        let _ = writeln!(self.out, "  {sh64} = zext i32 {sh} to i64");
                        let _ = writeln!(self.out, "  {sft} = lshr i64 {mask64}, {sh64}");
                        let _ = writeln!(self.out, "  {t32} = trunc i64 {sft} to i32");
                        (t32, Some(sh64))
                    } else {
                        let t32 = self.next_tmp();
                        let _ = writeln!(self.out, "  {t32} = trunc i64 {mask64} to i32");
                        (t32, None)
                    }
                };
                match mode {
                    VoteMode::Ballot => self.store_reg(*dst, &ballot32, IrType::U32),
                    VoteMode::Any => {
                        let any = self.next_tmp();
                        let _ = writeln!(self.out, "  {any} = icmp ne i32 {ballot32}, 0");
                        self.store_reg(*dst, &any, IrType::Pred);
                    }
                    VoteMode::All => {
                        let active32 = if wave32 {
                            let a32 = self.next_tmp();
                            let _ = writeln!(
                                self.out,
                                "  {a32} = call i32 @llvm.amdgcn.ballot.i32(i1 true)"
                            );
                            a32
                        } else {
                            let active64 = self.next_tmp();
                            let _ = writeln!(
                                self.out,
                                "  {active64} = call i64 @llvm.amdgcn.ballot.i64(i1 true)"
                            );
                            let shifted = if let Some(sh64) = active_shift64 {
                                let s = self.next_tmp();
                                let _ = writeln!(self.out, "  {s} = lshr i64 {active64}, {sh64}");
                                s
                            } else {
                                active64
                            };
                            let a32 = self.next_tmp();
                            let _ = writeln!(self.out, "  {a32} = trunc i64 {shifted} to i32");
                            a32
                        };
                        let all = self.next_tmp();
                        let _ = writeln!(self.out, "  {all} = icmp eq i32 {ballot32}, {active32}");
                        self.store_reg(*dst, &all, IrType::Pred);
                    }
                }
            }
            IrOp::Return => {
                let _ = writeln!(self.out, "  ret void");
                self.block_terminated = true;
            }
        }
    }

    fn emit_addr_ptr(
        &mut self,
        space: AddressSpace,
        base: &IrOperand,
        offset: i64,
    ) -> (String, &'static str) {
        if space == AddressSpace::Shared {
            let base32 = self.eval_operand(base, IrType::U32);
            let off32 = if offset != 0 {
                let t = self.next_tmp();
                let _ = writeln!(self.out, "  {t} = add i32 {base32}, {offset}");
                t
            } else {
                base32
            };
            let ptr = self.next_tmp();
            let _ = writeln!(
                self.out,
                "  {ptr} = getelementptr i8, ptr addrspace(3) @__amda_lds_{}, i32 {off32}",
                self.kernel.name
            );
            (ptr, "ptr addrspace(3)")
        } else if space == AddressSpace::Private && self.kernel.private_bytes > 0 {
            let base32 = self.eval_operand(base, IrType::U32);
            let off32 = if offset != 0 {
                let t = self.next_tmp();
                let _ = writeln!(self.out, "  {t} = add i32 {base32}, {offset}");
                t
            } else {
                base32
            };
            let ptr = self.next_tmp();
            let _ = writeln!(
                self.out,
                "  {ptr} = getelementptr i8, ptr addrspace(5) %__amda_local, i32 {off32}"
            );
            (ptr, "ptr addrspace(5)")
        } else {
            let base64 = self.eval_operand(base, IrType::U64);
            let addr64 = if offset != 0 {
                let t = self.next_tmp();
                let _ = writeln!(self.out, "  {t} = add i64 {base64}, {offset}");
                t
            } else {
                base64
            };
            let ptr = self.next_tmp();
            let ptr_ty = match space {
                AddressSpace::Global => "ptr addrspace(1)",
                AddressSpace::Constant => "ptr addrspace(4)",
                AddressSpace::Private => "ptr addrspace(5)",
                _ => "ptr",
            };
            let _ = writeln!(self.out, "  {ptr} = inttoptr i64 {addr64} to {ptr_ty}");
            (ptr, ptr_ty)
        }
    }

    fn emit_cvt(&mut self, dst_ty: IrType, src_ty: IrType, val: &str) -> String {
        if dst_ty == src_ty {
            return val.to_owned();
        }
        let st = src_ty.llvm_ty();
        let dt = dst_ty.llvm_ty();
        if st == dt {
            return val.to_owned();
        }
        let res = self.next_tmp();
        if !src_ty.is_float_ty() && dst_ty.is_float_ty() {
            let op = if src_ty.is_signed_int_ty() {
                "sitofp"
            } else {
                "uitofp"
            };
            let _ = writeln!(self.out, "  {res} = {op} {st} {val} to {dt}");
            return res;
        }
        if src_ty.is_float_ty() && !dst_ty.is_float_ty() && dst_ty != IrType::Pred {
            let op = if dst_ty.is_signed_int_ty() {
                "fptosi"
            } else {
                "fptoui"
            };
            let _ = writeln!(self.out, "  {res} = {op} {st} {val} to {dt}");
            return res;
        }
        if src_ty.is_float_ty() && dst_ty.is_float_ty() {
            if matches!((src_ty, dst_ty), (IrType::F16, IrType::Bf16) | (IrType::Bf16, IrType::F16)) {
                let flt = self.next_tmp();
                let _ = writeln!(self.out, "  {flt} = fpext {st} {val} to float");
                let _ = writeln!(self.out, "  {res} = fptrunc float {flt} to {dt}");
                return res;
            }
            let op = if src_ty.byte_size() < dst_ty.byte_size() {
                "fpext"
            } else {
                "fptrunc"
            };
            let _ = writeln!(self.out, "  {res} = {op} {st} {val} to {dt}");
            return res;
        }
        self.coerce_val(val, src_ty, dst_ty)
    }
}

impl IrType {
    const fn is_float_ty(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }
}

fn zero_literal(ty: IrType) -> String {
    match ty {
        IrType::F16 | IrType::Bf16 | IrType::F32 | IrType::F64 => "0.0".to_owned(),
        IrType::Pred => "false".to_owned(),
        _ => "0".to_owned(),
    }
}

fn format_llvm_f32(v: f32) -> String {
    let bits64 = f64::from(v).to_bits();
    format!("0x{bits64:016X}")
}

fn format_llvm_f64(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

fn emit_llvm_math_helpers(out: &mut String) {
    let _ = writeln!(out, "define internal float @__amda_rsqrt_f32(float %x) #1 {{");
    let _ = writeln!(out, "entry:");
    let _ = writeln!(out, "  %s = call float @llvm.sqrt.f32(float %x)");
    let _ = writeln!(out, "  %r = fdiv float 1.0, %s");
    let _ = writeln!(out, "  ret float %r");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    let log2_e = format_llvm_f32(std::f32::consts::LOG2_E);
    let two = format_llvm_f32(2.0);
    let _ = writeln!(out, "define internal float @__amda_tanh_f32(float %x) #1 {{");
    let _ = writeln!(out, "entry:");
    let _ = writeln!(out, "  %two = fmul float %x, {two}");
    let _ = writeln!(out, "  %k = fmul float %two, {log2_e}");
    let _ = writeln!(out, "  %e = call float @llvm.exp2.f32(float %k)");
    let _ = writeln!(out, "  %num = fsub float %e, 1.0");
    let _ = writeln!(out, "  %den = fadd float %e, 1.0");
    let _ = writeln!(out, "  %r = fdiv float %num, %den");
    let _ = writeln!(out, "  ret float %r");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    let _ = writeln!(out, "define internal i32 @__amda_prmt_b32(i32 %a, i32 %b, i32 %sel) #1 {{");
    let _ = writeln!(out, "entry:");
    let _ = writeln!(out, "  %a64 = zext i32 %a to i64");
    let _ = writeln!(out, "  %b64 = zext i32 %b to i64");
    let _ = writeln!(out, "  %bhi = shl i64 %b64, 32");
    let _ = writeln!(out, "  %ab = or i64 %a64, %bhi");
    for lane in 0..4 {
        let shift = lane * 4;
        let _ = writeln!(out, "  %shs{lane} = lshr i32 %sel, {shift}");
        let _ = writeln!(out, "  %nib{lane} = and i32 %shs{lane}, 15");
        let _ = writeln!(out, "  %idx{lane} = and i32 %nib{lane}, 7");
        let _ = writeln!(out, "  %bs{lane} = shl i32 %idx{lane}, 3");
        let _ = writeln!(out, "  %bs64_{lane} = zext i32 %bs{lane} to i64");
        let _ = writeln!(out, "  %ex{lane} = lshr i64 %ab, %bs64_{lane}");
        let _ = writeln!(out, "  %ex32_{lane} = trunc i64 %ex{lane} to i32");
        let _ = writeln!(out, "  %raw{lane} = and i32 %ex32_{lane}, 255");
        let _ = writeln!(out, "  %b8_{lane} = trunc i32 %raw{lane} to i8");
        let _ = writeln!(out, "  %sxt{lane} = sext i8 %b8_{lane} to i32");
        let _ = writeln!(out, "  %sar{lane} = ashr i32 %sxt{lane}, 7");
        let _ = writeln!(out, "  %sgnfill{lane} = and i32 %sar{lane}, 255");
        let _ = writeln!(out, "  %sflag{lane} = and i32 %nib{lane}, 8");
        let _ = writeln!(out, "  %iss{lane} = icmp ne i32 %sflag{lane}, 0");
        let _ = writeln!(out, "  %pick{lane} = select i1 %iss{lane}, i32 %sgnfill{lane}, i32 %raw{lane}");
        let osh = lane * 8;
        let _ = writeln!(out, "  %part{lane} = shl i32 %pick{lane}, {osh}");
    }
    let _ = writeln!(out, "  %p01 = or i32 %part0, %part1");
    let _ = writeln!(out, "  %p23 = or i32 %part2, %part3");
    let _ = writeln!(out, "  %out = or i32 %p01, %p23");
    let _ = writeln!(out, "  ret i32 %out");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "define internal i32 @__amda_dp4a(i32 %a, i32 %b, i32 %c, i1 %a_sgn, i1 %b_sgn) #1 {{"
    );
    let _ = writeln!(out, "entry:");
    let mut acc = "%c".to_owned();
    for lane in 0..4 {
        let sh = lane * 8;
        let _ = writeln!(out, "  %sa{lane} = lshr i32 %a, {sh}");
        let _ = writeln!(out, "  %ba{lane} = trunc i32 %sa{lane} to i8");
        let _ = writeln!(out, "  %asxt{lane} = sext i8 %ba{lane} to i32");
        let _ = writeln!(out, "  %azxt{lane} = zext i8 %ba{lane} to i32");
        let _ = writeln!(out, "  %va{lane} = select i1 %a_sgn, i32 %asxt{lane}, i32 %azxt{lane}");

        let _ = writeln!(out, "  %sb{lane} = lshr i32 %b, {sh}");
        let _ = writeln!(out, "  %bb{lane} = trunc i32 %sb{lane} to i8");
        let _ = writeln!(out, "  %bsxt{lane} = sext i8 %bb{lane} to i32");
        let _ = writeln!(out, "  %bzxt{lane} = zext i8 %bb{lane} to i32");
        let _ = writeln!(out, "  %vb{lane} = select i1 %b_sgn, i32 %bsxt{lane}, i32 %bzxt{lane}");

        let _ = writeln!(out, "  %prod{lane} = mul i32 %va{lane}, %vb{lane}");
        let next_acc = format!("%acc{lane}");
        let _ = writeln!(out, "  {next_acc} = add i32 {acc}, %prod{lane}");
        acc = next_acc;
    }
    let _ = writeln!(out, "  ret i32 {acc}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

fn emit_llvm_kernel(out: &mut String, kernel: &IrKernel, target: IsaTarget) {
    let lds_bytes = kernel.shared_bytes.max(1);
    if kernel.shared_bytes > 0
        || kernel.insts.iter().any(|i| {
            matches!(
                i.op,
                IrOp::Load {
                    space: AddressSpace::Shared,
                    ..
                } | IrOp::Store {
                    space: AddressSpace::Shared,
                    ..
                } | IrOp::Atomic {
                    space: AddressSpace::Shared,
                    ..
                }
            )
        })
    {
        let _ = writeln!(
            out,
            "@__amda_lds_{} = internal addrspace(3) global [{} x i8] undef, align 16",
            kernel.name, lds_bytes
        );
    }

    let _ = write!(out, "define amdgpu_kernel void @{}(", kernel.name);
    for (idx, p) in kernel.params.iter().enumerate() {
        if idx > 0 {
            let _ = write!(out, ", ");
        }
        if p.is_global_ptr {
            let _ = write!(out, "ptr addrspace(1) %arg_{}", p.name);
        } else {
            let _ = write!(out, "{} %arg_{}", p.ty.llvm_ty(), p.name);
        }
    }
    let _ = writeln!(out, ") #0 {{");
    let _ = writeln!(out, "entry:");
    let _ = writeln!(
        out,
        "  ; wave_mode = {:?}, kernarg_bytes = {}, shared_bytes = {}, private_bytes = {}",
        kernel.wave_mode, kernel.metadata.kernarg_segment_size, kernel.shared_bytes, kernel.private_bytes
    );

    if kernel.private_bytes > 0 {
        let _ = writeln!(
            out,
            "  %__amda_local = alloca [{} x i8], align 16, addrspace(5)",
            kernel.private_bytes
        );
    }

    // Emit addrspace(5) allocas for all virtual IR registers (promoted to SSA by LLVM mem2reg)
    for (idx, &ty) in kernel.reg_types.iter().enumerate() {
        let align = ty.byte_size().max(1);
        let lt = ty.llvm_ty();
        let slot = LlvmEmitter::reg_slot(idx as u32);
        let z = zero_literal(ty);
        let _ = writeln!(out, "  {slot} = alloca {lt}, align {align}, addrspace(5)");
        let _ = writeln!(out, "  store {lt} {z}, ptr addrspace(5) {slot}, align {align}");
    }

    let mut emitter = LlvmEmitter::new(out, kernel, target);
    for inst in &kernel.insts {
        emitter.emit_inst(inst);
    }
    if !emitter.block_terminated {
        let _ = writeln!(emitter.out, "  ret void");
    }

    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
}

/// Validates emitted LLVM IR for structural & SSA invariants:
/// - `amdgcn-amd-amdhsa` triple and `amdgpu_kernel` definitions
/// - `addrspace(5)` on all `alloca` instructions
/// - Basic block terminators with no unreachable instructions between terminator and next label
/// - Every branch target label defined within the same function
/// - Unique SSA `%name =` definitions per function
pub fn validate_llvm_ir(llvm_ir: &str) -> Result<(), String> {
    if !llvm_ir.contains("target triple = \"amdgcn-amd-amdhsa\"") {
        return Err("missing target triple amdgcn-amd-amdhsa".to_owned());
    }

    let mut in_fn = false;
    let mut fn_name = String::new();
    let mut fn_count = 0usize;
    let mut defined_labels: HashSet<String> = HashSet::new();
    let mut used_labels: Vec<String> = Vec::new();
    let mut defined_ssa: HashSet<String> = HashSet::new();
    let mut need_terminator = false;
    let mut expect_label = false;

    for (line_no, raw) in llvm_ir.lines().enumerate() {
        let line = raw.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("define ") {
            let is_internal = line.starts_with("define internal ");
            if !is_internal && !line.contains("amdgpu_kernel") {
                return Err(format!("line {}: kernel missing amdgpu_kernel calling conv", line_no + 1));
            }
            in_fn = true;
            if !is_internal {
                fn_count += 1;
            }
            fn_name = line.to_owned();
            defined_labels.clear();
            used_labels.clear();
            defined_ssa.clear();
            need_terminator = false;
            expect_label = true;
            continue;
        }
        if !in_fn {
            continue;
        }
        if line == "}" {
            if need_terminator {
                return Err(format!("function '{fn_name}' ends without basic block terminator"));
            }
            for target_lbl in &used_labels {
                if !defined_labels.contains(target_lbl) {
                    return Err(format!(
                        "function '{fn_name}' branches to undefined label '%{target_lbl}'"
                    ));
                }
            }
            in_fn = false;
            continue;
        }

        if line.ends_with(':') {
            if need_terminator {
                return Err(format!(
                    "line {}: label '{line}' follows non-terminated basic block",
                    line_no + 1
                ));
            }
            let lbl = line.trim_end_matches(':').to_owned();
            if !defined_labels.insert(lbl.clone()) {
                return Err(format!(
                    "line {}: duplicate label '{lbl}' in {fn_name}",
                    line_no + 1
                ));
            }
            need_terminator = false;
            expect_label = false;
            continue;
        }

        if expect_label {
            return Err(format!(
                "line {}: instruction after terminator without basic block label: {line}",
                line_no + 1
            ));
        }

        if line.contains(" = alloca ") && !line.contains("addrspace(5)") {
            return Err(format!(
                "line {}: alloca missing addrspace(5): {line}",
                line_no + 1
            ));
        }

        if let Some(eq_pos) = line.find(" = ") {
            let lhs = line[..eq_pos].trim();
            if lhs.starts_with('%') && !defined_ssa.insert(lhs.to_owned()) {
                return Err(format!(
                    "line {}: duplicate SSA definition '{lhs}' in {fn_name}",
                    line_no + 1
                ));
            }
        }

        if line.starts_with("br label %") {
            let target = line.trim_start_matches("br label %").trim();
            used_labels.push(target.to_owned());
            need_terminator = false;
            expect_label = true;
            continue;
        }
        if line.starts_with("br i1 ") {
            for part in line.split(',') {
                let trimmed = part.trim();
                if let Some(rest) = trimmed.strip_prefix("label %") {
                    used_labels.push(rest.trim().to_owned());
                }
            }
            need_terminator = false;
            expect_label = true;
            continue;
        }
        if line.starts_with("ret ") || line == "unreachable" {
            need_terminator = false;
            expect_label = true;
            continue;
        }

        need_terminator = true;
    }

    if fn_count == 0 {
        return Err("LLVM IR contained no amdgpu_kernel definitions".to_owned());
    }
    if in_fn {
        return Err("unclosed LLVM function body".to_owned());
    }
    Ok(())
}

fn sanitize_llvm_ident(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ============================================================================
// Binary Serialization for `.amda.ir` HSACO Section
// ============================================================================

impl IrModule {
    /// Serializes this [`IrModule`] into a compact deterministic byte stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = BinWriter::default();
        w.bytes(&IR_BINARY_MAGIC);
        w.str(self.target.as_str());
        w.u32(self.kernels.len() as u32);
        for k in &self.kernels {
            write_kernel(&mut w, k);
        }
        w.buf
    }

    /// Deserializes an [`IrModule`] previously produced by [`IrModule::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let mut r = BinReader::new(bytes);
        let magic = r.exact(8)?;
        if magic != IR_BINARY_MAGIC {
            return Err(CompileError::InvalidIrPayload(
                "invalid AMDA IR binary magic".to_owned(),
            ));
        }
        let target_str = r.str()?;
        let target = IsaTarget::from_str(&target_str)
            .map_err(|_| CompileError::InvalidIrPayload(format!("unknown ISA '{target_str}'")))?;
        let k_count = r.u32()? as usize;
        let mut kernels = Vec::with_capacity(k_count.min(256));
        for _ in 0..k_count {
            kernels.push(read_kernel(&mut r, target)?);
        }
        Ok(Self { target, kernels })
    }
}

fn write_kernel(w: &mut BinWriter, k: &IrKernel) {
    w.str(&k.name);
    w.u32(k.shared_bytes);
    w.u32(k.private_bytes);
    w.u8(k.wave_mode as u8);

    w.u32(k.params.len() as u32);
    for p in &k.params {
        w.str(&p.name);
        w.u8(p.ty as u8);
        w.u32(p.offset);
        w.u32(p.size);
        w.u8(u8::from(p.is_global_ptr));
    }

    w.u32(k.reg_types.len() as u32);
    for rt in &k.reg_types {
        w.u8(*rt as u8);
    }

    w.u32(k.insts.len() as u32);
    for inst in &k.insts {
        match inst.pred {
            Some(pg) => {
                w.u8(1);
                w.u32(pg.reg);
                w.u8(u8::from(pg.negated));
            }
            None => w.u8(0),
        }
        write_op(w, &inst.op);
    }
}

fn read_kernel(r: &mut BinReader<'_>, target: IsaTarget) -> Result<IrKernel, CompileError> {
    let name = r.str()?;
    let shared_bytes = r.u32()?;
    let private_bytes = r.u32()?;
    let wave_mode = WaveMode::from_tag(r.u8()?)?;

    let p_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(p_count.min(256));
    let mut arg_metas = Vec::with_capacity(p_count.min(256));
    let mut max_end = 0u32;

    for _ in 0..p_count {
        let pname = r.str()?;
        let ty = IrType::from_tag(r.u8()?)?;
        let offset = r.u32()?;
        let size = r.u32()?;
        let is_global_ptr = r.u8()? != 0;
        max_end = max_end.max(offset.saturating_add(size));
        arg_metas.push(KernelArgMetadata {
            name: pname.clone(),
            offset,
            size,
            value_kind: if is_global_ptr {
                ValueKind::GlobalBuffer
            } else {
                ValueKind::ByValue
            },
        });
        params.push(IrParam {
            name: pname,
            ty,
            offset,
            size,
            is_global_ptr,
        });
    }

    let r_count = r.u32()? as usize;
    let mut reg_types = Vec::with_capacity(r_count.min(4096));
    for _ in 0..r_count {
        reg_types.push(IrType::from_tag(r.u8()?)?);
    }

    let i_count = r.u32()? as usize;
    let mut insts = Vec::with_capacity(i_count.min(65536));
    for _ in 0..i_count {
        let pred = if r.u8()? != 0 {
            let reg = r.u32()?;
            let negated = r.u8()? != 0;
            Some(IrPredGuard { reg, negated })
        } else {
            None
        };
        let op = read_op(r)?;
        insts.push(IrInst { pred, op });
    }

    let metadata = KernelMetadata {
        name: name.clone(),
        symbol: format!("{name}.kd"),
        kernarg_segment_size: align_up_u32(max_end, 8),
        group_segment_fixed_size: shared_bytes,
        private_segment_fixed_size: private_bytes,
        wavefront_size: target.native_wave_size(),
        max_flat_workgroup_size: 1024,
        args: arg_metas,
    };

    Ok(IrKernel {
        name,
        params,
        reg_types,
        shared_bytes,
        private_bytes,
        wave_mode,
        insts,
        metadata,
    })
}

fn write_op(w: &mut BinWriter, op: &IrOp) {
    match op {
        IrOp::LoadParam {
            dst,
            ty,
            param_idx,
            offset,
        } => {
            w.u8(1);
            w.u32(*dst);
            w.u8(*ty as u8);
            w.u32(*param_idx);
            w.u32(*offset);
        }
        IrOp::Load {
            dst,
            ty,
            space,
            base,
            offset,
        } => {
            w.u8(2);
            w.u32(*dst);
            w.u8(*ty as u8);
            w.u8(*space as u8);
            write_operand(w, base);
            w.i64(*offset);
        }
        IrOp::Store {
            ty,
            space,
            base,
            offset,
            src,
        } => {
            w.u8(3);
            w.u8(*ty as u8);
            w.u8(*space as u8);
            write_operand(w, base);
            w.i64(*offset);
            write_operand(w, src);
        }
        IrOp::CvtaToGlobal { dst, src } => {
            w.u8(4);
            w.u32(*dst);
            write_operand(w, src);
        }
        IrOp::CvtaToShared { dst, src } => {
            w.u8(5);
            w.u32(*dst);
            write_operand(w, src);
        }
        IrOp::Mov { dst, ty, src } => {
            w.u8(6);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, src);
        }
        IrOp::Cvt {
            dst,
            dst_ty,
            src_ty,
            src,
        } => {
            w.u8(7);
            w.u32(*dst);
            w.u8(*dst_ty as u8);
            w.u8(*src_ty as u8);
            write_operand(w, src);
        }
        IrOp::Neg { dst, ty, src } => {
            w.u8(8);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, src);
        }
        IrOp::Abs { dst, ty, src } => {
            w.u8(9);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, src);
        }
        IrOp::Bin {
            op,
            dst,
            ty,
            lhs,
            rhs,
        } => {
            w.u8(10);
            w.u8(*op as u8);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, lhs);
            write_operand(w, rhs);
        }
        IrOp::Mad { dst, ty, a, b, c } => {
            w.u8(11);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, a);
            write_operand(w, b);
            write_operand(w, c);
        }
        IrOp::Setp {
            cmp,
            dst,
            ty,
            lhs,
            rhs,
        } => {
            w.u8(12);
            w.u8(*cmp as u8);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, lhs);
            write_operand(w, rhs);
        }
        IrOp::Selp {
            dst,
            ty,
            a,
            b,
            pred,
        } => {
            w.u8(13);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, a);
            write_operand(w, b);
            w.u32(*pred);
        }
        IrOp::Label(s) => {
            w.u8(14);
            w.str(s);
        }
        IrOp::Branch(s) => {
            w.u8(15);
            w.str(s);
        }
        IrOp::Barrier => w.u8(16),
        IrOp::WarpShuffle {
            dst,
            mode,
            src,
            lane_delta,
            width,
        } => {
            w.u8(17);
            w.u32(*dst);
            w.u8(*mode as u8);
            write_operand(w, src);
            write_operand(w, lane_delta);
            w.u32(*width);
        }
        IrOp::Return => w.u8(18),
        IrOp::MemBar => w.u8(19),
        IrOp::UnaryMath { op, dst, ty, src } => {
            w.u8(20);
            w.u8(*op as u8);
            w.u32(*dst);
            w.u8(*ty as u8);
            write_operand(w, src);
        }
        IrOp::Bfe {
            dst,
            signed,
            src,
            pos,
            len,
        } => {
            w.u8(21);
            w.u32(*dst);
            w.u8(u8::from(*signed));
            write_operand(w, src);
            write_operand(w, pos);
            write_operand(w, len);
        }
        IrOp::Bfi {
            dst,
            insert,
            base,
            pos,
            len,
        } => {
            w.u8(22);
            w.u32(*dst);
            write_operand(w, insert);
            write_operand(w, base);
            write_operand(w, pos);
            write_operand(w, len);
        }
        IrOp::Prmt { dst, a, b, sel } => {
            w.u8(23);
            w.u32(*dst);
            write_operand(w, a);
            write_operand(w, b);
            write_operand(w, sel);
        }
        IrOp::Dp4a {
            dst,
            a_signed,
            b_signed,
            a,
            b,
            c,
        } => {
            w.u8(24);
            w.u32(*dst);
            w.u8(u8::from(*a_signed));
            w.u8(u8::from(*b_signed));
            write_operand(w, a);
            write_operand(w, b);
            write_operand(w, c);
        }
        IrOp::Atomic {
            op,
            dst,
            ty,
            space,
            base,
            offset,
            val,
            cmp,
        } => {
            w.u8(25);
            w.u8(*op as u8);
            match dst {
                Some(d) => {
                    w.u8(1);
                    w.u32(*d);
                }
                None => w.u8(0),
            }
            w.u8(*ty as u8);
            w.u8(*space as u8);
            write_operand(w, base);
            w.i64(*offset);
            write_operand(w, val);
            match cmp {
                Some(c) => {
                    w.u8(1);
                    write_operand(w, c);
                }
                None => w.u8(0),
            }
        }
        IrOp::WarpVote {
            mode,
            dst,
            pred,
            negated,
        } => {
            w.u8(26);
            w.u8(*mode as u8);
            w.u32(*dst);
            w.u32(*pred);
            w.u8(u8::from(*negated));
        }
    }
}

fn read_op(r: &mut BinReader<'_>) -> Result<IrOp, CompileError> {
    let tag = r.u8()?;
    match tag {
        1 => Ok(IrOp::LoadParam {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            param_idx: r.u32()?,
            offset: r.u32()?,
        }),
        2 => Ok(IrOp::Load {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            space: AddressSpace::from_tag(r.u8()?)?,
            base: read_operand(r)?,
            offset: r.i64()?,
        }),
        3 => Ok(IrOp::Store {
            ty: IrType::from_tag(r.u8()?)?,
            space: AddressSpace::from_tag(r.u8()?)?,
            base: read_operand(r)?,
            offset: r.i64()?,
            src: read_operand(r)?,
        }),
        4 => Ok(IrOp::CvtaToGlobal {
            dst: r.u32()?,
            src: read_operand(r)?,
        }),
        5 => Ok(IrOp::CvtaToShared {
            dst: r.u32()?,
            src: read_operand(r)?,
        }),
        6 => Ok(IrOp::Mov {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            src: read_operand(r)?,
        }),
        7 => Ok(IrOp::Cvt {
            dst: r.u32()?,
            dst_ty: IrType::from_tag(r.u8()?)?,
            src_ty: IrType::from_tag(r.u8()?)?,
            src: read_operand(r)?,
        }),
        8 => Ok(IrOp::Neg {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            src: read_operand(r)?,
        }),
        9 => Ok(IrOp::Abs {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            src: read_operand(r)?,
        }),
        10 => Ok(IrOp::Bin {
            op: BinOp::from_tag(r.u8()?)?,
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            lhs: read_operand(r)?,
            rhs: read_operand(r)?,
        }),
        11 => Ok(IrOp::Mad {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            a: read_operand(r)?,
            b: read_operand(r)?,
            c: read_operand(r)?,
        }),
        12 => Ok(IrOp::Setp {
            cmp: CmpOp::from_tag(r.u8()?)?,
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            lhs: read_operand(r)?,
            rhs: read_operand(r)?,
        }),
        13 => Ok(IrOp::Selp {
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            a: read_operand(r)?,
            b: read_operand(r)?,
            pred: r.u32()?,
        }),
        14 => Ok(IrOp::Label(r.str()?)),
        15 => Ok(IrOp::Branch(r.str()?)),
        16 => Ok(IrOp::Barrier),
        17 => Ok(IrOp::WarpShuffle {
            dst: r.u32()?,
            mode: ShuffleMode::from_tag(r.u8()?)?,
            src: read_operand(r)?,
            lane_delta: read_operand(r)?,
            width: r.u32()?,
        }),
        18 => Ok(IrOp::Return),
        19 => Ok(IrOp::MemBar),
        20 => Ok(IrOp::UnaryMath {
            op: UnaryMathOp::from_tag(r.u8()?)?,
            dst: r.u32()?,
            ty: IrType::from_tag(r.u8()?)?,
            src: read_operand(r)?,
        }),
        21 => Ok(IrOp::Bfe {
            dst: r.u32()?,
            signed: r.u8()? != 0,
            src: read_operand(r)?,
            pos: read_operand(r)?,
            len: read_operand(r)?,
        }),
        22 => Ok(IrOp::Bfi {
            dst: r.u32()?,
            insert: read_operand(r)?,
            base: read_operand(r)?,
            pos: read_operand(r)?,
            len: read_operand(r)?,
        }),
        23 => Ok(IrOp::Prmt {
            dst: r.u32()?,
            a: read_operand(r)?,
            b: read_operand(r)?,
            sel: read_operand(r)?,
        }),
        24 => Ok(IrOp::Dp4a {
            dst: r.u32()?,
            a_signed: r.u8()? != 0,
            b_signed: r.u8()? != 0,
            a: read_operand(r)?,
            b: read_operand(r)?,
            c: read_operand(r)?,
        }),
        25 => {
            let op = AtomicOp::from_tag(r.u8()?)?;
            let dst = if r.u8()? != 0 { Some(r.u32()?) } else { None };
            let ty = IrType::from_tag(r.u8()?)?;
            let space = AddressSpace::from_tag(r.u8()?)?;
            let base = read_operand(r)?;
            let offset = r.i64()?;
            let val = read_operand(r)?;
            let cmp = if r.u8()? != 0 {
                Some(read_operand(r)?)
            } else {
                None
            };
            Ok(IrOp::Atomic {
                op,
                dst,
                ty,
                space,
                base,
                offset,
                val,
                cmp,
            })
        }
        26 => Ok(IrOp::WarpVote {
            mode: VoteMode::from_tag(r.u8()?)?,
            dst: r.u32()?,
            pred: r.u32()?,
            negated: r.u8()? != 0,
        }),
        _ => Err(CompileError::InvalidIrPayload(format!(
            "unknown IrOp tag {tag}"
        ))),
    }
}

fn write_operand(w: &mut BinWriter, op: &IrOperand) {
    match op {
        IrOperand::Reg(r) => {
            w.u8(1);
            w.u32(*r);
        }
        IrOperand::ImmU64(v) => {
            w.u8(2);
            w.u64(*v);
        }
        IrOperand::ImmI64(v) => {
            w.u8(3);
            w.i64(*v);
        }
        IrOperand::ImmF64(v) => {
            w.u8(4);
            w.u64(v.to_bits());
        }
        IrOperand::Special(s) => {
            w.u8(5);
            w.u8(*s as u8);
        }
        IrOperand::Param(idx) => {
            w.u8(6);
            w.u32(*idx);
        }
        IrOperand::SharedOffset(off) => {
            w.u8(7);
            w.u32(*off);
        }
        IrOperand::LocalOffset(off) => {
            w.u8(8);
            w.u32(*off);
        }
    }
}

fn read_operand(r: &mut BinReader<'_>) -> Result<IrOperand, CompileError> {
    match r.u8()? {
        1 => Ok(IrOperand::Reg(r.u32()?)),
        2 => Ok(IrOperand::ImmU64(r.u64()?)),
        3 => Ok(IrOperand::ImmI64(r.i64()?)),
        4 => Ok(IrOperand::ImmF64(f64::from_bits(r.u64()?))),
        5 => Ok(IrOperand::Special(SpecialReg::from_tag(r.u8()?)?)),
        6 => Ok(IrOperand::Param(r.u32()?)),
        7 => Ok(IrOperand::SharedOffset(r.u32()?)),
        8 => Ok(IrOperand::LocalOffset(r.u32()?)),
        t => Err(CompileError::InvalidIrPayload(format!(
            "unknown IrOperand tag {t}"
        ))),
    }
}

#[derive(Default)]
struct BinWriter {
    buf: Vec<u8>,
}

impl BinWriter {
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

struct BinReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn exact(&mut self, len: usize) -> Result<&'a [u8], CompileError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| CompileError::InvalidIrPayload("offset overflow".to_owned()))?;
        if end > self.data.len() {
            return Err(CompileError::InvalidIrPayload(
                "unexpected end of IR payload".to_owned(),
            ));
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, CompileError> {
        Ok(self.exact(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, CompileError> {
        let b = self.exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, CompileError> {
        let b = self.exact(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> Result<i64, CompileError> {
        let b = self.exact(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn str(&mut self) -> Result<String, CompileError> {
        let len = self.u32()? as usize;
        let bytes = self.exact(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| CompileError::InvalidIrPayload("invalid UTF-8 string in IR".to_owned()))
    }
}
