//! Abstract Syntax Tree for parsed NVIDIA PTX modules.

/// Top-level parsed PTX module.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxModule {
    /// PTX ISA version string (e.g. `"7.8"`).
    pub version: String,
    /// Target SM architecture (e.g. `"sm_80"`).
    pub target: String,
    /// Pointer width (`32` or `64`).
    pub address_size: u32,
    /// Global/module-scope `.shared` declarations.
    pub shared_decls: Vec<PtxSharedDecl>,
    /// Parsed `.func` helper device functions eligible for inlining.
    pub funcs: Vec<PtxFunc>,
    /// Parsed `.entry` kernels.
    pub kernels: Vec<PtxKernel>,
}

/// Scalar/vector type inside PTX declarations and instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    Pred,
    B8,
    B16,
    B32,
    B64,
    U8,
    U16,
    U32,
    U64,
    S8,
    S16,
    S32,
    S64,
    F16,
    Bf16,
    F16x2,
    F32,
    F64,
}

impl PtxType {
    /// Parses a PTX type modifier such as `"u64"`, `"s32"`, `"f32"`, `"f16"`, `"bf16"`, `"pred"`.
    pub fn from_modifier(s: &str) -> Option<Self> {
        match s {
            "pred" => Some(Self::Pred),
            "b8" => Some(Self::B8),
            "b16" => Some(Self::B16),
            "b32" => Some(Self::B32),
            "b64" => Some(Self::B64),
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "s8" => Some(Self::S8),
            "s16" => Some(Self::S16),
            "s32" => Some(Self::S32),
            "s64" => Some(Self::S64),
            "f16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "f16x2" => Some(Self::F16x2),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }

    /// Size of this PTX type in bytes.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Pred | Self::B8 | Self::U8 | Self::S8 => 1,
            Self::B16 | Self::U16 | Self::S16 | Self::F16 | Self::Bf16 => 2,
            Self::B32 | Self::U32 | Self::S32 | Self::F16x2 | Self::F32 => 4,
            Self::B64 | Self::U64 | Self::S64 | Self::F64 => 8,
        }
    }

    /// Returns true for signed integer types.
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    /// Returns true for floating-point types.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Bf16 | Self::F16x2 | Self::F32 | Self::F64
        )
    }
}

/// Kernel or helper function parameter declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxParam {
    pub name: String,
    pub ty: PtxType,
    /// Array count (1 for scalar/pointer params, >1 for byte arrays).
    pub count: u32,
    pub is_ptr: bool,
}

/// Register declaration inside a kernel (`.reg .u32 %r<16>;` or `.reg .f32 %f0, %f1;`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxRegDecl {
    pub ty: PtxType,
    pub prefix: String,
    /// `Some(N)` if declared as `%r<N>`, `None` if declared as an exact single register name.
    pub count: Option<u32>,
}

/// Shared memory declaration (`.shared .align 4 .b8 smem[1024];` or `.extern .shared .b8 dyn[];`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxSharedDecl {
    pub name: String,
    pub ty: PtxType,
    pub align: u32,
    pub count: u32,
    pub is_extern: bool,
}

/// Private per-thread `.local` stack array declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxLocalDecl {
    pub name: String,
    pub ty: PtxType,
    pub align: u32,
    pub count: u32,
}

/// Parsed `.func` device helper function.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxFunc {
    pub name: String,
    pub visible: bool,
    pub ret_params: Vec<PtxParam>,
    pub in_params: Vec<PtxParam>,
    pub regs: Vec<PtxRegDecl>,
    pub shared_decls: Vec<PtxSharedDecl>,
    pub local_decls: Vec<PtxLocalDecl>,
    pub statements: Vec<PtxStatement>,
}

/// Parsed `.entry` kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxKernel {
    pub name: String,
    pub visible: bool,
    pub params: Vec<PtxParam>,
    pub regs: Vec<PtxRegDecl>,
    pub shared_decls: Vec<PtxSharedDecl>,
    pub local_decls: Vec<PtxLocalDecl>,
    pub statements: Vec<PtxStatement>,
}

/// Statement inside a PTX kernel or function body.
#[derive(Debug, Clone, PartialEq)]
pub enum PtxStatement {
    Label(String),
    Instruction(PtxInstruction),
}

/// Optional execution guard (`@%p0` or `@!%p0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxPredicate {
    pub reg: String,
    pub negated: bool,
}

/// Single PTX instruction with optional predicate guard and dot modifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct PtxInstruction {
    pub pred: Option<PtxPredicate>,
    pub opcode: String,
    pub modifiers: Vec<String>,
    pub operands: Vec<PtxOperand>,
}

/// PTX instruction operand.
#[derive(Debug, Clone, PartialEq)]
pub enum PtxOperand {
    /// Register such as `%r0` or `%tid.x`.
    Reg(String),
    /// Negated register predicate such as `!%p1` (used in `vote.sync`).
    NegatedReg(String),
    /// Signed/unsigned integer literal.
    ImmInt(i128),
    /// Floating-point literal.
    ImmFloat(f64),
    /// Symbol / label identifier (e.g. `$L_BB0` or parameter name).
    Ident(String),
    /// Memory reference `[base + offset]`.
    Mem { base: String, offset: i64 },
    /// Braced `{ %r0, %r1 }`, parenthesized `(a, b)`, or piped `%r0|%p0` tuple operand.
    Tuple(Vec<PtxOperand>),
}
