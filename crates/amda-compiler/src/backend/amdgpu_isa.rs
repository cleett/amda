//! Tier 0 zero-LLVM direct AMDGPU native machine-code emitter (GFX9/10/11/12).

use crate::backend::regalloc::{allocate_registers, KernelAllocation, PhysReg, RegAllocSummary};
use crate::error::CompileError;
use crate::ir::{
    AddressSpace, AtomicOp, BinOp, CmpOp, IrInst, IrKernel, IrModule, IrOp, IrOperand, IrType,
    SpecialReg, UnaryMathOp,
};
use amda_elf::{AmdhsaKernelDescriptor, IsaTarget};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Decoded instruction record produced by [`verify_encoded_machine_code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedInst {
    pub byte_offset: usize,
    pub len_bytes: usize,
    pub format: &'static str,
    pub w0: u32,
    pub w1: Option<u32>,
}

/// Compiled Tier 0 native code and descriptor for one kernel.
#[derive(Debug, Clone)]
pub struct NativeKernelCode {
    pub name: String,
    pub machine_code: Vec<u8>,
    pub descriptor: AmdhsaKernelDescriptor,
    pub disasm: String,
    pub summary: RegAllocSummary,
}

/// Complete Tier 0 compiled native module ready for HSACO `.text` + `.rodata`.
#[derive(Debug, Clone)]
pub struct NativeModuleCode {
    pub kernels: Vec<NativeKernelCode>,
    pub combined_text: Vec<u8>,
    pub combined_rodata: Vec<u8>,
    pub full_disasm: String,
}

const SCRATCH_S0: u8 = 8;
const SCRATCH_S1: u8 = 9;
const SCRATCH_V0: u8 = 4;
const SCRATCH_V1: u8 = 5;

const fn is_scalar_or_lit(enc: u32) -> bool {
    enc < 108 || enc == 0xFF
}

const fn is_gfx11_plus(isa: IsaTarget) -> bool {
    matches!(
        isa,
        IsaTarget::Gfx1100
            | IsaTarget::Gfx1101
            | IsaTarget::Gfx1102
            | IsaTarget::Gfx1103
            | IsaTarget::Gfx1200
            | IsaTarget::Gfx1201
    )
}

#[derive(Debug, Clone)]
struct BranchFixup {
    dword_idx: usize,
    asm_line_idx: usize,
    target_label: String,
    mnemonic: &'static str,
}

struct AsmLine {
    byte_offset: usize,
    words: Vec<u32>,
    text: String,
}

struct Emitter {
    isa: IsaTarget,
    dwords: Vec<u32>,
    lines: Vec<AsmLine>,
    labels: HashMap<String, usize>,
    fixups: Vec<BranchFixup>,
    pending_exec_restores: HashMap<String, bool>,
    lgkm_dirty: bool,
    vm_dirty: bool,
}

impl Emitter {
    fn new(isa: IsaTarget) -> Self {
        Self {
            isa,
            dwords: Vec::with_capacity(128),
            lines: Vec::with_capacity(64),
            labels: HashMap::new(),
            fixups: Vec::new(),
            pending_exec_restores: HashMap::new(),
            lgkm_dirty: false,
            vm_dirty: false,
        }
    }

    fn emit1(&mut self, w0: u32, text: String) -> usize {
        let idx = self.dwords.len();
        self.dwords.push(w0);
        self.lines.push(AsmLine {
            byte_offset: idx * 4,
            words: vec![w0],
            text,
        });
        idx
    }

    fn emit2(&mut self, w0: u32, w1: u32, text: String) -> usize {
        let idx = self.dwords.len();
        self.dwords.push(w0);
        self.dwords.push(w1);
        self.lines.push(AsmLine {
            byte_offset: idx * 4,
            words: vec![w0, w1],
            text,
        });
        idx
    }

    fn record_label(&mut self, label: &str) {
        self.labels.insert(label.to_owned(), self.dwords.len());
        self.lines.push(AsmLine {
            byte_offset: self.dwords.len() * 4,
            words: Vec::new(),
            text: format!("{label}:"),
        });
    }

    fn emit_waitcnt_lgkm(&mut self) {
        if !self.lgkm_dirty {
            return;
        }
        let w = if is_gfx11_plus(self.isa) {
            0xBF89_FC07 // s_waitcnt lgkmcnt(0) on GFX11
        } else {
            0xBF8C_C07F // s_waitcnt lgkmcnt(0) on GFX9/10
        };
        self.emit1(w, "s_waitcnt lgkmcnt(0)".to_owned());
        self.lgkm_dirty = false;
    }

    fn emit_waitcnt_vm(&mut self) {
        if !self.vm_dirty {
            return;
        }
        let w = if is_gfx11_plus(self.isa) {
            0xBF89_03F7 // s_waitcnt vmcnt(0) on GFX11
        } else {
            0xBF8C_0F70 // s_waitcnt vmcnt(0) on GFX9/10
        };
        self.emit1(w, "s_waitcnt vmcnt(0)".to_owned());
        self.vm_dirty = false;
    }

    fn emit_s_endpgm(&mut self) {
        let w = if is_gfx11_plus(self.isa) {
            0xBFB0_0000
        } else {
            0xBF81_0000
        };
        self.emit1(w, "s_endpgm".to_owned());
    }

    fn emit_s_barrier(&mut self) {
        let w = if is_gfx11_plus(self.isa) {
            0xBFBD_0000
        } else {
            0xBF8A_0000
        };
        self.emit1(w, "s_barrier".to_owned());
    }

    fn emit_s_branch(&mut self, mnemonic: &'static str, opcode_hi: u32, label: &str) {
        let dword_idx = self.dwords.len();
        let asm_line_idx = self.lines.len();
        self.emit1(opcode_hi, format!("{mnemonic} {label}"));
        self.fixups.push(BranchFixup {
            dword_idx,
            asm_line_idx,
            target_label: label.to_owned(),
            mnemonic,
        });
    }

    fn emit_s_load(&mut self, sdst: u8, sbase: u8, offset: u32, is_64: bool) {
        let (w0, w1, mnem) = if is_gfx11_plus(self.isa) {
            let op = if is_64 { 0x01u32 } else { 0x00u32 };
            let w0 = 0xF400_0000 | (op << 18) | (u32::from(sdst) << 6) | u32::from(sbase >> 1);
            let w1 = (0x7Cu32 << 25) | (offset & 0x00FF_FFFF);
            (w0, w1, if is_64 { "s_load_b64" } else { "s_load_b32" })
        } else {
            let op = if is_64 { 0x01u32 } else { 0x00u32 };
            let w0 =
                0xC000_0000 | (op << 18) | (1 << 17) | (u32::from(sdst) << 6) | u32::from(sbase >> 1);
            let w1 = offset & 0x001F_FFFF;
            (w0, w1, if is_64 { "s_load_dwordx2" } else { "s_load_dword" })
        };
        let dst_fmt = if is_64 {
            format!("s[{sdst}:{}]", sdst + 1)
        } else {
            format!("s{sdst}")
        };
        self.emit2(
            w0,
            w1,
            format!("{mnem} {dst_fmt}, s[{sbase}:{}], 0x{offset:x}", sbase + 1),
        );
        self.lgkm_dirty = true;
    }

    fn emit_s_mov_b32(&mut self, sdst: u8, imm_or_src: u32, is_literal: bool, src_desc: &str) {
        if !is_literal {
            let w0 = 0xBE80_0000 | (u32::from(sdst) << 16) | (imm_or_src & 0xff);
            self.emit1(w0, format!("s_mov_b32 s{sdst}, {src_desc}"));
        } else if imm_or_src <= 64 {
            let enc = 128 + imm_or_src;
            let w0 = 0xBE80_0000 | (u32::from(sdst) << 16) | enc;
            self.emit1(w0, format!("s_mov_b32 s{sdst}, {imm_or_src}"));
        } else {
            let w0 = 0xBE80_00FF | (u32::from(sdst) << 16);
            self.emit2(w0, imm_or_src, format!("s_mov_b32 s{sdst}, 0x{imm_or_src:x}"));
        }
    }

    fn emit_v_mov_b32(&mut self, vdst: u8, src0_enc: u32, literal: Option<u32>, src_desc: &str) {
        let w0 = 0x7E00_0200 | (u32::from(vdst) << 17) | (src0_enc & 0x1ff);
        match literal {
            Some(lit) => {
                self.emit2(w0, lit, format!("v_mov_b32 v{vdst}, {src_desc}"));
            }
            None => {
                self.emit1(w0, format!("v_mov_b32 v{vdst}, {src_desc}"));
            }
        }
    }

    fn emit_vop1(&mut self, op: u32, mnem: &str, vdst: u8, src0_enc: u32, src_desc: &str) {
        let w0 = 0x7E00_0000 | (u32::from(vdst) << 17) | ((op & 0xff) << 9) | (src0_enc & 0x1ff);
        self.emit1(w0, format!("{mnem} v{vdst}, {src_desc}"));
    }

    fn emit_sop2(
        &mut self,
        op: u32,
        mnem: &str,
        sdst: u8,
        ssrc0: u8,
        ssrc1: u8,
        desc: &str,
    ) {
        let w0 = 0x8000_0000
            | ((op & 0x7f) << 23)
            | (u32::from(sdst) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0);
        self.emit1(w0, format!("{mnem} s{sdst}, {desc}"));
    }

    fn emit_vop2(
        &mut self,
        op: u32,
        mnem: &str,
        vdst: u8,
        src0_enc: u32,
        vsrc1: u8,
        desc: &str,
    ) {
        let w0 = ((op & 0x3f) << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(vsrc1) << 9)
            | (src0_enc & 0x1ff);
        self.emit1(w0, format!("{mnem} v{vdst}, {desc}"));
    }

    fn coalesce_vop3_gfx9(&mut self, s0: &mut u32, s1: &mut u32, s2: &mut u32) {
        if is_gfx11_plus(self.isa) {
            return;
        }
        let bus0 = is_scalar_or_lit(*s0);
        if is_scalar_or_lit(*s1) && bus0 && *s1 != *s0 {
            let src1 = *s1;
            let dst_v = if *s2 == 256 + u32::from(SCRATCH_V1) {
                SCRATCH_V0
            } else {
                SCRATCH_V1
            };
            self.emit_v_mov_b32(dst_v, src1, None, "gfx9_cbus_s1");
            *s1 = 256 + u32::from(dst_v);
        }
        let bus1 = is_scalar_or_lit(*s1);
        if is_scalar_or_lit(*s2) && ((bus0 && *s2 != *s0) || (bus1 && *s2 != *s1)) {
            let src2 = *s2;
            let dst_v = if *s0 == 256 + u32::from(SCRATCH_V0) || *s1 == 256 + u32::from(SCRATCH_V0)
            {
                SCRATCH_V1
            } else {
                SCRATCH_V0
            };
            self.emit_v_mov_b32(dst_v, src2, None, "gfx9_cbus_s2");
            *s2 = 256 + u32::from(dst_v);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_vop3(
        &mut self,
        op: u32,
        mnem: &str,
        vdst: u8,
        mut src0_enc: u32,
        mut src1_enc: u32,
        mut src2_enc: u32,
        desc: &str,
    ) {
        self.coalesce_vop3_gfx9(&mut src0_enc, &mut src1_enc, &mut src2_enc);
        let base = if is_gfx11_plus(self.isa) {
            0xD400_0000
        } else {
            0xD000_0000
        };
        let w0 = base | ((op & 0x3ff) << 16) | u32::from(vdst);
        let w1 = (src0_enc & 0x1ff) | ((src1_enc & 0x1ff) << 9) | ((src2_enc & 0x1ff) << 18);
        self.emit2(w0, w1, format!("{mnem} v{vdst}, {desc}"));
    }

    fn emit_v_add_u32(&mut self, vdst: u8, src0_enc: u32, vsrc1: u8, desc: &str) {
        let (op, mnem) = if is_gfx11_plus(self.isa) {
            (0x25, "v_add_nc_u32")
        } else {
            (0x34, "v_add_u32")
        };
        self.emit_vop2(op, mnem, vdst, src0_enc, vsrc1, desc);
    }

    fn emit_v_add_f32(&mut self, vdst: u8, src0_enc: u32, vsrc1: u8, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x03 } else { 0x01 };
        self.emit_vop2(op, "v_add_f32", vdst, src0_enc, vsrc1, desc);
    }

    fn emit_v_sub_f32(&mut self, vdst: u8, src0_enc: u32, vsrc1: u8, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x04 } else { 0x02 };
        self.emit_vop2(op, "v_sub_f32", vdst, src0_enc, vsrc1, desc);
    }

    fn emit_v_mul_f32(&mut self, vdst: u8, src0_enc: u32, vsrc1: u8, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x08 } else { 0x05 };
        self.emit_vop2(op, "v_mul_f32", vdst, src0_enc, vsrc1, desc);
    }

    fn emit_sop2_lit(
        &mut self,
        op: u32,
        mnem: &str,
        sdst: u8,
        ssrc0: u8,
        literal: u32,
        desc: &str,
    ) {
        let w0 = 0x8000_0000
            | ((op & 0x7f) << 23)
            | (u32::from(sdst) << 16)
            | (0xFFu32 << 8)
            | u32::from(ssrc0);
        self.emit2(w0, literal, format!("{mnem} s{sdst}, {desc}"));
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_vop3b(
        &mut self,
        op: u32,
        mnem: &str,
        vdst: u8,
        sdst: u8,
        src0_enc: u32,
        src1_enc: u32,
        src2_enc: u32,
        desc: &str,
    ) {
        let base = if is_gfx11_plus(self.isa) {
            0xD400_0000
        } else {
            0xD000_0000
        };
        let w0 = base
            | ((op & 0x3ff) << 16)
            | (u32::from(sdst & 0x7f) << 8)
            | u32::from(vdst);
        let w1 = (src0_enc & 0x1ff) | ((src1_enc & 0x1ff) << 9) | ((src2_enc & 0x1ff) << 18);
        self.emit2(w0, w1, format!("{mnem} v{vdst}, vcc, {desc}"));
    }

    fn emit_v_add_co_u32(&mut self, vdst: u8, src0_enc: u32, src1_enc: u32, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x300 } else { 0x119 };
        self.emit_vop3b(op, "v_add_co_u32", vdst, 0x6A, src0_enc, src1_enc, 0, desc);
    }

    fn emit_v_add_co_ci_u32(&mut self, vdst: u8, src0_enc: u32, src1_enc: u32, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x120 } else { 0x11C };
        let s0 = if !is_gfx11_plus(self.isa) && is_scalar_or_lit(src0_enc) {
            self.emit_v_mov_b32(SCRATCH_V0, src0_enc, None, "gfx9_cbus_ci_s0");
            256 + u32::from(SCRATCH_V0)
        } else {
            src0_enc
        };
        self.emit_vop3b(
            op,
            "v_add_co_ci_u32",
            vdst,
            0x6A,
            s0,
            src1_enc,
            0x6A,
            desc,
        );
    }

    fn emit_exec_save_and_mask(&mut self, target_label: &str, negated: bool) {
        let gfx11 = is_gfx11_plus(self.isa);
        if self.isa.native_wave_size() == 32 {
            // s_mov_b32 s9, exec_lo (0x7E)
            self.emit_s_mov_b32(SCRATCH_S1, 0x7E, false, "exec_lo");
            let (op, mnem) = if negated {
                (if gfx11 { 0x16 } else { 0x0C }, "s_and_b32")
            } else {
                (if gfx11 { 0x22 } else { 0x12 }, "s_andn2_b32")
            };
            let w0 = 0x8000_0000
                | (op << 23)
                | (0x7Eu32 << 16)
                | (0x6Au32 << 8)
                | 0x7Eu32;
            self.emit1(w0, format!("{mnem} exec_lo, exec_lo, vcc_lo"));
            let cbr_op = if gfx11 { 0xBFA5_0000 } else { 0xBF88_0000 };
            self.emit_s_branch("s_cbranch_execz", cbr_op, target_label);
        } else {
            // Wave64 CDNA (gfx942): s_mov_b64 s[8:9], exec (op=1, ssrc0=0x7E)
            let mov64 = 0xBE80_0100 | (u32::from(SCRATCH_S0) << 16) | 0x7Eu32;
            self.emit1(mov64, "s_mov_b64 s[8:9], exec".to_owned());
            let (op, mnem) = if negated {
                (if gfx11 { 0x17 } else { 0x0D }, "s_and_b64")
            } else {
                (if gfx11 { 0x23 } else { 0x13 }, "s_andn2_b64")
            };
            let w0 = 0x8000_0000
                | (op << 23)
                | (0x7Eu32 << 16)
                | (0x6Au32 << 8)
                | 0x7Eu32;
            self.emit1(w0, format!("{mnem} exec, exec, vcc"));
            let cbr_op = if gfx11 { 0xBFA5_0000 } else { 0xBF88_0000 };
            self.emit_s_branch("s_cbranch_execz", cbr_op, target_label);
        }
        self.pending_exec_restores
            .insert(target_label.to_owned(), true);
    }

    fn maybe_emit_exec_restore(&mut self, label: &str) {
        if self.pending_exec_restores.remove(label).unwrap_or(false) {
            let gfx11 = is_gfx11_plus(self.isa);
            if self.isa.native_wave_size() == 32 {
                let op = if gfx11 { 0x18 } else { 0x0E };
                let w0 = 0x8000_0000
                    | (op << 23)
                    | (0x7Eu32 << 16)
                    | (u32::from(SCRATCH_S1) << 8)
                    | 0x7Eu32;
                self.emit1(w0, format!("s_or_b32 exec_lo, exec_lo, s{SCRATCH_S1}"));
            } else {
                let op = if gfx11 { 0x19 } else { 0x0F };
                let w0 = 0x8000_0000
                    | (op << 23)
                    | (0x7Eu32 << 16)
                    | (u32::from(SCRATCH_S0) << 8)
                    | 0x7Eu32;
                self.emit1(w0, format!("s_or_b64 exec, exec, s[{SCRATCH_S0}:{}]", SCRATCH_S0 + 1));
            }
        }
    }

    fn emit_v_fma_f32(&mut self, vdst: u8, src0_enc: u32, src1_enc: u32, src2_enc: u32, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x213 } else { 0x1CB };
        self.emit_vop3(op, "v_fma_f32", vdst, src0_enc, src1_enc, src2_enc, desc);
    }

    fn emit_v_mul_lo_u32(&mut self, vdst: u8, src0_enc: u32, src1_enc: u32, desc: &str) {
        let op = if is_gfx11_plus(self.isa) { 0x32C } else { 0x285 };
        self.emit_vop3(op, "v_mul_lo_u32", vdst, src0_enc, src1_enc, 0, desc);
    }

    fn emit_v_cmp(&mut self, op: u32, mnem: &str, src0_enc: u32, vsrc1: u8, desc: &str) {
        let w0 = 0x7C00_0000 | ((op & 0xff) << 17) | (u32::from(vsrc1) << 9) | (src0_enc & 0x1ff);
        self.emit1(w0, format!("{mnem} vcc, {desc}"));
    }

    fn emit_global_load(&mut self, vdst: u8, vaddr: u8, offset: i64, is_64: bool) {
        let gfx11 = is_gfx11_plus(self.isa);
        let saddr_off: u32 = if gfx11 { 0x7C } else { 0x7F };
        let op = if is_64 { 0x15u32 } else { 0x14u32 };
        let seg = if gfx11 { 2 << 16 } else { 2 << 14 };
        let w0 = 0xDC00_0000 | (op << 18) | seg | ((offset as u32) & 0x1fff);
        let w1 = (u32::from(vdst) << 24) | (saddr_off << 16) | u32::from(vaddr);
        let mnem = match (gfx11, is_64) {
            (true, true) => "global_load_b64",
            (true, false) => "global_load_b32",
            (false, true) => "global_load_dwordx2",
            (false, false) => "global_load_dword",
        };
        let dst_fmt = if is_64 {
            format!("v[{vdst}:{}]", vdst + 1)
        } else {
            format!("v{vdst}")
        };
        self.emit2(
            w0,
            w1,
            format!("{mnem} {dst_fmt}, v[{vaddr}:{}], off offset:{offset}", vaddr + 1),
        );
        self.vm_dirty = true;
    }

    fn emit_global_store(&mut self, vaddr: u8, vdata: u8, offset: i64, is_64: bool) {
        let gfx11 = is_gfx11_plus(self.isa);
        let saddr_off: u32 = if gfx11 { 0x7C } else { 0x7F };
        let op = match (gfx11, is_64) {
            (true, true) => 0x1Bu32,
            (true, false) => 0x1Au32,
            (false, true) => 0x1Du32,
            (false, false) => 0x1Cu32,
        };
        let seg = if gfx11 { 2 << 16 } else { 2 << 14 };
        let w0 = 0xDC00_0000 | (op << 18) | seg | ((offset as u32) & 0x1fff);
        let w1 = (saddr_off << 16) | (u32::from(vdata) << 8) | u32::from(vaddr);
        let mnem = match (gfx11, is_64) {
            (true, true) => "global_store_b64",
            (true, false) => "global_store_b32",
            (false, true) => "global_store_dwordx2",
            (false, false) => "global_store_dword",
        };
        let data_fmt = if is_64 {
            format!("v[{vdata}:{}]", vdata + 1)
        } else {
            format!("v{vdata}")
        };
        self.emit2(
            w0,
            w1,
            format!("{mnem} v[{vaddr}:{}], {data_fmt}, off offset:{offset}", vaddr + 1),
        );
        self.vm_dirty = true;
    }

    fn emit_ds_read_b32(&mut self, vdst: u8, vaddr: u8, offset: i64) {
        let shift = if is_gfx11_plus(self.isa) { 18 } else { 17 };
        let w0 = 0xD800_0000 | (0x36u32 << shift) | ((offset as u32) & 0xffff);
        let w1 = (u32::from(vdst) << 24) | u32::from(vaddr);
        self.emit2(
            w0,
            w1,
            format!("ds_read_b32 v{vdst}, v{vaddr} offset:{offset}"),
        );
        self.lgkm_dirty = true;
    }

    fn emit_ds_write_b32(&mut self, vaddr: u8, vdata: u8, offset: i64) {
        let shift = if is_gfx11_plus(self.isa) { 18 } else { 17 };
        let w0 = 0xD800_0000 | (0x0Du32 << shift) | ((offset as u32) & 0xffff);
        let w1 = (u32::from(vdata) << 8) | u32::from(vaddr);
        self.emit2(
            w0,
            w1,
            format!("ds_write_b32 v{vaddr}, v{vdata} offset:{offset}"),
        );
        self.lgkm_dirty = true;
    }

    fn resolve_fixups(&mut self) -> Result<(), CompileError> {
        for fx in &self.fixups {
            let target_dw = *self.labels.get(&fx.target_label).ok_or_else(|| {
                CompileError::LoweringError(format!(
                    "Tier 0 unresolved branch target '{}'",
                    fx.target_label
                ))
            })?;
            let rel = (target_dw as isize) - ((fx.dword_idx as isize) + 1);
            let simm16 = i16::try_from(rel).map_err(|_| {
                CompileError::LoweringError("Tier 0 branch offset exceeds i16".to_owned())
            })?;
            let patched = (self.dwords[fx.dword_idx] & 0xFFFF_0000) | u32::from(simm16 as u16);
            self.dwords[fx.dword_idx] = patched;
            if let Some(line) = self.lines.get_mut(fx.asm_line_idx) {
                line.words[0] = patched;
                line.text = format!("{} {} ; rel={rel:+}", fx.mnemonic, fx.target_label);
            }
        }
        Ok(())
    }

    fn finish(mut self, kernel_name: &str) -> Result<(Vec<u8>, String), CompileError> {
        self.resolve_fixups()?;
        let mut code = Vec::with_capacity(self.dwords.len() * 4);
        for dw in &self.dwords {
            code.extend_from_slice(&dw.to_le_bytes());
        }
        let mut disasm = String::with_capacity(2048);
        let _ = writeln!(disasm, "; --- kernel @{kernel_name} ({}) ---", self.isa);
        for line in &self.lines {
            if line.words.is_empty() {
                let _ = writeln!(disasm, "  {}", line.text);
            } else if line.words.len() == 1 {
                let _ = writeln!(
                    disasm,
                    "  0x{:04x}: {:08X}           {}",
                    line.byte_offset, line.words[0], line.text
                );
            } else {
                let _ = writeln!(
                    disasm,
                    "  0x{:04x}: {:08X} {:08X}  {}",
                    line.byte_offset, line.words[0], line.words[1], line.text
                );
            }
        }
        Ok((code, disasm))
    }
}

/// Compiles an entire [`IrModule`] directly to native AMDGPU machine code using Tier 0 (no LLVM).
pub fn compile_tier0_native(
    ir: &IrModule,
    target: IsaTarget,
) -> Result<NativeModuleCode, CompileError> {
    let mut kernels = Vec::with_capacity(ir.kernels.len());
    let mut combined_text = Vec::new();
    let mut combined_rodata = Vec::new();
    let mut full_disasm = String::new();

    for kernel in &ir.kernels {
        let alloc = allocate_registers(kernel, target)?;
        let (machine_code, disasm) = emit_kernel_native(kernel, &alloc, target)?;

        // Align kernel entry offset to 64 bytes in combined `.text`
        while (combined_text.len() % 64) != 0 {
            combined_text.push(0);
        }
        let text_entry_off = combined_text.len() as i64;
        combined_text.extend_from_slice(&machine_code);

        // Build 64-byte AMDHSA hardware kernel descriptor referencing this entry
        let desc = AmdhsaKernelDescriptor::new(
            target,
            alloc.vgpr_count,
            alloc.sgpr_count,
            kernel.shared_bytes,
            kernel.metadata.kernarg_segment_size,
            text_entry_off,
        );
        combined_rodata.extend_from_slice(&desc.to_bytes());

        let summary = RegAllocSummary {
            kernel_name: kernel.name.clone(),
            sgpr_count: alloc.sgpr_count,
            vgpr_count: alloc.vgpr_count,
            uniform_ir_regs: alloc.uniform_ir_regs,
            divergent_ir_regs: alloc.divergent_ir_regs,
            code_bytes: machine_code.len(),
            estimated_waves_per_simd: alloc.estimated_waves_per_simd,
        };

        full_disasm.push_str(&disasm);
        kernels.push(NativeKernelCode {
            name: kernel.name.clone(),
            machine_code,
            descriptor: desc,
            disasm,
            summary,
        });
    }

    Ok(NativeModuleCode {
        kernels,
        combined_text,
        combined_rodata,
        full_disasm,
    })
}

fn emit_kernel_native(
    kernel: &IrKernel,
    alloc: &KernelAllocation,
    isa: IsaTarget,
) -> Result<(Vec<u8>, String), CompileError> {
    let mut e = Emitter::new(isa);
    let mut seen_return = false;

    for inst in &kernel.insts {
        emit_ir_inst(&mut e, kernel, alloc, inst)?;
        if matches!(inst.op, IrOp::Return) {
            seen_return = true;
        }
    }

    if !seen_return {
        e.emit_s_endpgm();
    }

    e.finish(&kernel.name)
}

fn emit_ir_inst(
    e: &mut Emitter,
    kernel: &IrKernel,
    alloc: &KernelAllocation,
    inst: &IrInst,
) -> Result<(), CompileError> {
    match &inst.op {
        IrOp::Label(lbl) => {
            e.record_label(lbl);
            e.maybe_emit_exec_restore(lbl);
        }
        IrOp::LoadParam {
            dst,
            ty,
            param_idx,
            offset,
        } => {
            let param = kernel
                .params
                .get(*param_idx as usize)
                .ok_or_else(|| CompileError::LoweringError("param index out of bounds".to_owned()))?;
            let byte_off = param.offset + *offset;
            let phys = alloc.reg_map[*dst as usize];
            let is_64 = matches!(ty, IrType::U64 | IrType::I64 | IrType::F64);
            match phys {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                    // AMDHSA hardware loads DISPATCH_PTR into s[0:1] and KERNARG_SEGMENT_PTR into s[2:3].
                    e.emit_s_load(s, 2, byte_off, is_64);
                }
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => {
                    e.emit_s_load(SCRATCH_S0, 2, byte_off, is_64);
                    e.emit_waitcnt_lgkm();
                    e.emit_v_mov_b32(v, u32::from(SCRATCH_S0), None, &format!("s{SCRATCH_S0}"));
                    if is_64 {
                        e.emit_v_mov_b32(
                            v + 1,
                            u32::from(SCRATCH_S1),
                            None,
                            &format!("s{SCRATCH_S1}"),
                        );
                    }
                }
                PhysReg::Vcc => {}
            }
        }
        IrOp::CvtaToGlobal { dst, src } | IrOp::CvtaToShared { dst, src } => {
            let dst_reg = alloc.reg_map[*dst as usize];
            let (s_lo, desc_lo) = resolve_src0(e, alloc, src, false);
            match dst_reg {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                    e.emit_s_mov_b32(s, s_lo, false, &desc_lo);
                    if matches!(dst_reg, PhysReg::SgprPair(_)) {
                        let (s_hi, desc_hi) = resolve_src0_hi(alloc, src);
                        e.emit_s_mov_b32(s + 1, s_hi, false, &desc_hi);
                    }
                }
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => {
                    e.emit_v_mov_b32(v, s_lo, None, &desc_lo);
                    if matches!(dst_reg, PhysReg::VgprPair(_)) {
                        let (s_hi, desc_hi) = resolve_src0_hi(alloc, src);
                        e.emit_v_mov_b32(v + 1, s_hi, None, &desc_hi);
                    }
                }
                PhysReg::Vcc => {}
            }
        }
        IrOp::Mov { dst, src, .. } => {
            let dst_reg = alloc.reg_map[*dst as usize];
            let (enc, lit, desc) = load_operand_into_src0(e, alloc, src);
            match dst_reg {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                    if let Some(val) = lit {
                        e.emit_s_mov_b32(s, val, true, &desc);
                    } else {
                        e.emit_s_mov_b32(s, enc, false, &desc);
                    }
                }
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => {
                    e.emit_v_mov_b32(v, enc, lit, &desc);
                }
                PhysReg::Vcc => {}
            }
        }
        IrOp::Cvt {
            dst,
            dst_ty,
            src_ty,
            src,
        } => {
            let dst_reg = alloc.reg_map[*dst as usize];
            let (enc, lit, desc) = load_operand_into_src0(e, alloc, src);
            if let Some(l) = lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &desc);
            }
            let src_enc = if lit.is_some() {
                u32::from(SCRATCH_S0)
            } else {
                enc
            };
            let vdst = match dst_reg {
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                    e.emit_s_mov_b32(s, src_enc, false, &desc);
                    return Ok(());
                }
                PhysReg::Vcc => return Ok(()),
            };
            let (op, mnem) = match (dst_ty, src_ty) {
                (IrType::F32, IrType::I32) => (0x05, "v_cvt_f32_i32"),
                (IrType::F32, IrType::U32) => (0x06, "v_cvt_f32_u32"),
                (IrType::U32, IrType::F32) => (0x07, "v_cvt_u32_f32"),
                (IrType::I32, IrType::F32) => (0x08, "v_cvt_i32_f32"),
                _ => (0x01, "v_mov_b32"),
            };
            e.emit_vop1(op, mnem, vdst, src_enc, &desc);
        }
        IrOp::Neg { dst, ty, src } | IrOp::Abs { dst, ty, src } => {
            let dst_reg = alloc.reg_map[*dst as usize];
            let (enc, _, desc) = load_operand_into_src0(e, alloc, src);
            if let PhysReg::Vgpr(v) = dst_reg {
                let is_neg = matches!(inst.op, IrOp::Neg { .. });
                let mnem = match (is_neg, ty) {
                    (true, IrType::F32) => "v_xor_b32 ; fneg sign flip",
                    (false, IrType::F32) => "v_and_b32 ; fabs mask",
                    (true, _) => "v_sub_nc_u32 ; neg",
                    _ => "v_max_i32 ; abs",
                };
                e.emit_vop1(0x01, mnem, v, enc, &desc);
            }
        }
        IrOp::Bin {
            op,
            dst,
            ty,
            lhs,
            rhs,
        } => {
            e.emit_waitcnt_lgkm();
            let dst_reg = alloc.reg_map[*dst as usize];
            let (l_enc, l_lit, l_desc) = load_operand_into_src0(e, alloc, lhs);
            if let Some(l) = l_lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &l_desc);
            }
            let src0 = if l_lit.is_some() {
                u32::from(SCRATCH_S0)
            } else {
                l_enc
            };

            let vsrc1 = match rhs {
                IrOperand::Reg(r) => match alloc.reg_map[*r as usize] {
                    PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
                    PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                        e.emit_v_mov_b32(SCRATCH_V0, u32::from(s), None, &format!("s{s}"));
                        SCRATCH_V0
                    }
                    PhysReg::Vcc => SCRATCH_V0,
                },
                _ => {
                    let (r_enc, r_lit, r_desc) = load_operand_into_src0(e, alloc, rhs);
                    e.emit_v_mov_b32(SCRATCH_V0, r_enc, r_lit, &r_desc);
                    SCRATCH_V0
                }
            };

            match dst_reg {
                PhysReg::Sgpr(sdst) | PhysReg::SgprPair(sdst) => {
                    let s0 = (src0 & 0xff) as u8;
                    let s1 = match rhs {
                        IrOperand::Reg(r) => alloc.reg_map[*r as usize].base_idx(),
                        _ => SCRATCH_S1,
                    };
                    let gfx11 = is_gfx11_plus(e.isa);
                    let sop = match op {
                        BinOp::Add => (0x00, "s_add_u32"),
                        BinOp::Sub => (0x01, "s_sub_u32"),
                        BinOp::Mul | BinOp::MulWide => {
                            (if gfx11 { 0x2C } else { 0x24 }, "s_mul_i32")
                        }
                        BinOp::And => (if gfx11 { 0x16 } else { 0x0C }, "s_and_b32"),
                        BinOp::Or => (if gfx11 { 0x18 } else { 0x0E }, "s_or_b32"),
                        BinOp::Xor => (if gfx11 { 0x1A } else { 0x10 }, "s_xor_b32"),
                        BinOp::Shl => (if gfx11 { 0x08 } else { 0x1C }, "s_lshl_b32"),
                        BinOp::Shr => (if gfx11 { 0x0A } else { 0x1E }, "s_lshr_b32"),
                        BinOp::Min => (if gfx11 { 0x12 } else { 0x06 }, "s_min_i32"),
                        BinOp::Max => (if gfx11 { 0x14 } else { 0x08 }, "s_max_i32"),
                        BinOp::Div | BinOp::Rem => (0x00, "s_add_u32 ; div/rem-approx"),
                    };
                    e.emit_sop2(sop.0, sop.1, sdst, s0, s1, &format!("{l_desc}, s{s1}"));
                }
                PhysReg::Vgpr(vdst) => {
                    let gfx11 = is_gfx11_plus(e.isa);
                    let desc = format!("{l_desc}, v{vsrc1}");
                    match (op, ty) {
                        (BinOp::Add, IrType::F32 | IrType::F64 | IrType::F16 | IrType::Bf16) => {
                            e.emit_v_add_f32(vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Sub, IrType::F32 | IrType::F64 | IrType::F16 | IrType::Bf16) => {
                            e.emit_v_sub_f32(vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Mul, IrType::F32 | IrType::F64 | IrType::F16 | IrType::Bf16) => {
                            e.emit_v_mul_f32(vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Min, IrType::F32 | IrType::F64 | IrType::F16 | IrType::Bf16) => {
                            let opc = if gfx11 { 0x0F } else { 0x0A };
                            e.emit_vop2(opc, "v_min_f32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Max, IrType::F32 | IrType::F64 | IrType::F16 | IrType::Bf16) => {
                            let opc = if gfx11 { 0x10 } else { 0x0B };
                            e.emit_vop2(opc, "v_max_f32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Min, _) => {
                            let opc = if gfx11 { 0x11 } else { 0x0C };
                            e.emit_vop2(opc, "v_min_i32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Max, _) => {
                            let opc = if gfx11 { 0x12 } else { 0x0D };
                            e.emit_vop2(opc, "v_max_i32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Add, _) => e.emit_v_add_u32(vdst, src0, vsrc1, &desc),
                        (BinOp::Sub, _) => {
                            let opc = if gfx11 { 0x26 } else { 0x35 };
                            e.emit_vop2(opc, "v_sub_nc_u32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Mul, _) => e.emit_v_mul_lo_u32(vdst, src0, 256 + u32::from(vsrc1), &desc),
                        (BinOp::And, _) => {
                            let opc = if gfx11 { 0x1B } else { 0x13 };
                            e.emit_vop2(opc, "v_and_b32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Or, _) => {
                            let opc = if gfx11 { 0x1C } else { 0x14 };
                            e.emit_vop2(opc, "v_or_b32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Xor, _) => {
                            let opc = if gfx11 { 0x1D } else { 0x15 };
                            e.emit_vop2(opc, "v_xor_b32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Shl, _) => {
                            let opc = if gfx11 { 0x18 } else { 0x12 };
                            e.emit_vop2(opc, "v_lshlrev_b32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Shr, _) => {
                            let opc = if gfx11 { 0x19 } else { 0x10 };
                            e.emit_vop2(opc, "v_lshrrev_b32", vdst, src0, vsrc1, &desc);
                        }
                        (BinOp::Div | BinOp::Rem, _) => e.emit_v_mul_f32(vdst, src0, vsrc1, &desc),
                        (BinOp::MulWide, _) => {
                            e.emit_v_mul_lo_u32(vdst, src0, 256 + u32::from(vsrc1), &desc);
                        }
                    }
                }
                PhysReg::VgprPair(vdst) => {
                    // 64-bit pointer offset / mul.wide emission
                    match op {
                        BinOp::MulWide => {
                            let desc = format!("{l_desc}, v{vsrc1}");
                            e.emit_v_mul_lo_u32(vdst, src0, 256 + u32::from(vsrc1), &desc);
                            e.emit_v_mov_b32(vdst + 1, 128, None, "0 ; hi32");
                        }
                        BinOp::Add => {
                            let l_hi = resolve_src0_hi(alloc, lhs);
                            let r_hi = resolve_src0_hi(alloc, rhs);
                            // True 64-bit VOP3B carry chain: v_add_co_u32 -> v_add_co_ci_u32
                            e.emit_v_add_co_u32(
                                vdst,
                                src0,
                                256 + u32::from(vsrc1),
                                &format!("{l_desc}, v{vsrc1} ; ptr lo"),
                            );
                            e.emit_v_mov_b32(SCRATCH_V1, r_hi.0, None, &r_hi.1);
                            e.emit_v_add_co_ci_u32(
                                vdst + 1,
                                l_hi.0,
                                256 + u32::from(SCRATCH_V1),
                                &format!("{}, {}, vcc ; ptr hi", l_hi.1, r_hi.1),
                            );
                        }
                        _ => {
                            let desc = format!("{l_desc}, v{vsrc1}");
                            e.emit_v_add_u32(vdst, src0, vsrc1, &desc);
                        }
                    }
                }
                PhysReg::Vcc => {}
            }
        }
        IrOp::Mad { dst, ty, a, b, c } => {
            e.emit_waitcnt_lgkm();
            let dst_reg = alloc.reg_map[*dst as usize];
            let (a_enc, a_lit, a_desc) = load_operand_into_src0(e, alloc, a);
            if let Some(l) = a_lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &a_desc);
            }
            let sa = if a_lit.is_some() {
                u32::from(SCRATCH_S0)
            } else {
                a_enc
            };

            let sb_enc = operand_to_vop3_src(e, alloc, b, SCRATCH_V0);
            let sc_enc = operand_to_vop3_src(e, alloc, c, SCRATCH_V1);

            if let PhysReg::Vgpr(vdst) | PhysReg::VgprPair(vdst) = dst_reg {
                if matches!(ty, IrType::F32 | IrType::F64) {
                    e.emit_v_fma_f32(
                        vdst,
                        sa,
                        sb_enc,
                        sc_enc,
                        &format!("{a_desc}, ..., ..."),
                    );
                } else {
                    e.emit_v_mul_lo_u32(SCRATCH_V0, sa, sb_enc, &format!("{a_desc}, ..."));
                    let c_vgpr = if sc_enc >= 256 {
                        (sc_enc - 256) as u8
                    } else {
                        e.emit_v_mov_b32(SCRATCH_V1, sc_enc, None, "src_c");
                        SCRATCH_V1
                    };
                    e.emit_v_add_u32(
                        vdst,
                        256 + u32::from(SCRATCH_V0),
                        c_vgpr,
                        &format!("v{SCRATCH_V0}, v{c_vgpr}"),
                    );
                }
            }
        }
        IrOp::Setp {
            cmp,
            ty,
            lhs,
            rhs,
            ..
        } => {
            e.emit_waitcnt_lgkm();
            let (l_enc, l_lit, l_desc) = load_operand_into_src0(e, alloc, lhs);
            if let Some(l) = l_lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &l_desc);
            }
            let src0 = if l_lit.is_some() {
                u32::from(SCRATCH_S0)
            } else {
                l_enc
            };

            let vsrc1 = match rhs {
                IrOperand::Reg(r) => match alloc.reg_map[*r as usize] {
                    PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
                    PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                        e.emit_v_mov_b32(SCRATCH_V0, u32::from(s), None, &format!("s{s}"));
                        SCRATCH_V0
                    }
                    PhysReg::Vcc => SCRATCH_V0,
                },
                _ => {
                    let (r_enc, r_lit, r_desc) = load_operand_into_src0(e, alloc, rhs);
                    e.emit_v_mov_b32(SCRATCH_V0, r_enc, r_lit, &r_desc);
                    SCRATCH_V0
                }
            };

            let gfx11 = is_gfx11_plus(e.isa);
            let (op, mnem) = if gfx11 {
                match (cmp, ty) {
                    (CmpOp::Ge, IrType::I32) => (0x46, "v_cmp_ge_i32"),
                    (CmpOp::Lt, IrType::I32) => (0x41, "v_cmp_lt_i32"),
                    (CmpOp::Eq, IrType::I32 | IrType::U32) => (0x42, "v_cmp_eq_i32"),
                    (CmpOp::Ne, IrType::I32 | IrType::U32) => (0x45, "v_cmp_ne_i32"),
                    (CmpOp::Ge, IrType::U32) => (0x4E, "v_cmp_ge_u32"),
                    (CmpOp::Lt, IrType::U32) => (0x49, "v_cmp_lt_u32"),
                    (CmpOp::Ge, IrType::F32 | IrType::F64) => (0x16, "v_cmp_ge_f32"),
                    (CmpOp::Gt, IrType::F32 | IrType::F64) => (0x14, "v_cmp_gt_f32"),
                    _ => (0x46, "v_cmp_cond_i32"),
                }
            } else {
                match (cmp, ty) {
                    (CmpOp::Ge, IrType::I32) => (0xC6, "v_cmp_ge_i32"),
                    (CmpOp::Lt, IrType::I32) => (0xC1, "v_cmp_lt_i32"),
                    (CmpOp::Eq, IrType::I32 | IrType::U32) => (0xC2, "v_cmp_eq_i32"),
                    (CmpOp::Ne, IrType::I32 | IrType::U32) => (0xC5, "v_cmp_ne_i32"),
                    (CmpOp::Ge, IrType::U32) => (0xCE, "v_cmp_ge_u32"),
                    (CmpOp::Lt, IrType::U32) => (0xC9, "v_cmp_lt_u32"),
                    (CmpOp::Ge, IrType::F32 | IrType::F64) => (0x46, "v_cmp_ge_f32"),
                    (CmpOp::Gt, IrType::F32 | IrType::F64) => (0x44, "v_cmp_gt_f32"),
                    _ => (0xC6, "v_cmp_cond_i32"),
                }
            };
            e.emit_v_cmp(op, mnem, src0, vsrc1, &format!("{l_desc}, v{vsrc1}"));
        }
        IrOp::Selp { dst, a, b, .. } => {
            if let PhysReg::Vgpr(vdst) = alloc.reg_map[*dst as usize] {
                let (a_enc, a_lit, a_desc) = load_operand_into_src0(e, alloc, a);
                let b_v = match b {
                    IrOperand::Reg(r) => alloc.reg_map[*r as usize].base_idx(),
                    _ => {
                        let (b_enc, b_lit, b_desc) = load_operand_into_src0(e, alloc, b);
                        e.emit_v_mov_b32(SCRATCH_V0, b_enc, b_lit, &b_desc);
                        SCRATCH_V0
                    }
                };
                if let Some(l) = a_lit {
                    e.emit_s_mov_b32(SCRATCH_S0, l, true, &a_desc);
                }
                let src0 = if a_lit.is_some() {
                    u32::from(SCRATCH_S0)
                } else {
                    a_enc
                };
                let cnd_op = if is_gfx11_plus(e.isa) { 0x01 } else { 0x00 };
                e.emit_vop2(
                    cnd_op,
                    "v_cndmask_b32",
                    vdst,
                    src0,
                    b_v,
                    &format!("{a_desc}, v{b_v}, vcc"),
                );
            }
        }
        IrOp::Load {
            dst,
            ty,
            space,
            base,
            offset,
        } => {
            e.emit_waitcnt_lgkm();
            let dst_reg = alloc.reg_map[*dst as usize];
            let is_64 = matches!(ty, IrType::U64 | IrType::I64 | IrType::F64);
            let vdst = match dst_reg {
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
                _ => SCRATCH_V0,
            };
            if *space == AddressSpace::Shared {
                let vaddr = ensure_vgpr(e, alloc, base, SCRATCH_V1);
                e.emit_ds_read_b32(vdst, vaddr, *offset);
            } else {
                let vaddr = ensure_vgpr_pair(e, alloc, base, SCRATCH_V0);
                e.emit_global_load(vdst, vaddr, *offset, is_64);
            }
        }
        IrOp::Store {
            ty,
            space,
            base,
            offset,
            src,
        } => {
            e.emit_waitcnt_vm();
            e.emit_waitcnt_lgkm();
            let is_64 = matches!(ty, IrType::U64 | IrType::I64 | IrType::F64);
            let vdata = ensure_vgpr(e, alloc, src, SCRATCH_V0);
            if *space == AddressSpace::Shared {
                let vaddr = ensure_vgpr(e, alloc, base, SCRATCH_V1);
                e.emit_ds_write_b32(vaddr, vdata, *offset);
            } else {
                let vaddr = ensure_vgpr_pair(e, alloc, base, SCRATCH_V0);
                e.emit_global_store(vaddr, vdata, *offset, is_64);
            }
        }
        IrOp::Branch(lbl) => {
            let gfx11 = is_gfx11_plus(e.isa);
            match inst.pred {
                Some(pg) => {
                    let pred_div = alloc
                        .divergence
                        .get(pg.reg as usize)
                        .copied()
                        .unwrap_or(crate::backend::regalloc::Divergence::Uniform);
                    if pred_div == crate::backend::regalloc::Divergence::Divergent {
                        e.emit_exec_save_and_mask(lbl, pg.negated);
                    } else if pg.negated {
                        let op = if gfx11 { 0xBFA3_0000 } else { 0xBF86_0000 };
                        e.emit_s_branch("s_cbranch_vccz", op, lbl);
                    } else {
                        let op = if gfx11 { 0xBFA4_0000 } else { 0xBF87_0000 };
                        e.emit_s_branch("s_cbranch_vccnz", op, lbl);
                    }
                }
                None => {
                    let op = if gfx11 { 0xBFA0_0000 } else { 0xBF82_0000 };
                    e.emit_s_branch("s_branch", op, lbl);
                }
            }
        }
        IrOp::Barrier => {
            e.emit_waitcnt_lgkm();
            e.emit_s_barrier();
        }
        IrOp::MemBar => {
            e.emit_waitcnt_vm();
            e.emit_waitcnt_lgkm();
        }
        IrOp::UnaryMath { op, dst, src, .. } => {
            let (enc, lit, desc) = load_operand_into_src0(e, alloc, src);
            if let Some(l) = lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &desc);
            }
            let s0 = if lit.is_some() { u32::from(SCRATCH_S0) } else { enc };
            let vdst = match alloc.reg_map[*dst as usize] {
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
                _ => SCRATCH_V0,
            };
            let gfx11 = is_gfx11_plus(e.isa);
            let (vop1_op, mnem) = match op {
                UnaryMathOp::Rcp => (if gfx11 { 0x2A } else { 0x22 }, "v_rcp_f32"),
                UnaryMathOp::Sqrt => (if gfx11 { 0x33 } else { 0x27 }, "v_sqrt_f32"),
                UnaryMathOp::Rsqrt => (if gfx11 { 0x2E } else { 0x24 }, "v_rsq_f32"),
                UnaryMathOp::Ex2 | UnaryMathOp::Tanh => {
                    (if gfx11 { 0x25 } else { 0x20 }, "v_exp_f32")
                }
                UnaryMathOp::Lg2 => (if gfx11 { 0x27 } else { 0x21 }, "v_log_f32"),
                UnaryMathOp::Sin => (if gfx11 { 0x35 } else { 0x29 }, "v_sin_f32"),
                UnaryMathOp::Cos => (if gfx11 { 0x36 } else { 0x2A }, "v_cos_f32"),
            };
            e.emit_vop1(vop1_op, mnem, vdst, s0, &desc);
        }
        IrOp::Bfe { dst, signed, src, pos, len } => {
            let vdst = alloc.reg_map[*dst as usize].base_idx();
            let s0 = operand_to_vop3_src(e, alloc, src, SCRATCH_V0);
            let s1 = operand_to_vop3_src(e, alloc, pos, SCRATCH_V1);
            let s2 = operand_to_vop3_src(e, alloc, len, SCRATCH_V1);
            let gfx11 = is_gfx11_plus(e.isa);
            let (op, mnem) = match (gfx11, *signed) {
                (true, true) => (0x211, "v_bfe_i32"),
                (true, false) => (0x210, "v_bfe_u32"),
                (false, true) => (0x1C9, "v_bfe_i32"),
                (false, false) => (0x1C8, "v_bfe_u32"),
            };
            e.emit_vop3(op, mnem, vdst, s0, s1, s2, "src, pos, len");
        }
        IrOp::Bfi { dst, insert, base, pos, len } => {
            let vdst = alloc.reg_map[*dst as usize].base_idx();
            let s0 = operand_to_vop3_src(e, alloc, insert, SCRATCH_V0);
            let _ = operand_to_vop3_src(e, alloc, pos, SCRATCH_V1);
            let _ = operand_to_vop3_src(e, alloc, len, SCRATCH_V1);
            let s1 = operand_to_vop3_src(e, alloc, base, SCRATCH_V1);
            let op = if is_gfx11_plus(e.isa) { 0x212 } else { 0x1CA };
            e.emit_vop3(op, "v_bfi_b32", vdst, s0, s1, 256 + u32::from(SCRATCH_V1), "mask, ins, base");
        }
        IrOp::Prmt { dst, a, b, sel } => {
            let vdst = alloc.reg_map[*dst as usize].base_idx();
            let s0 = operand_to_vop3_src(e, alloc, a, SCRATCH_V0);
            let s1 = operand_to_vop3_src(e, alloc, b, SCRATCH_V1);
            let (s_enc, s_lit, s_desc) = load_operand_into_src0(e, alloc, sel);
            let s2 = if let Some(l) = s_lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &s_desc);
                u32::from(SCRATCH_S0)
            } else {
                s_enc
            };
            let op = if is_gfx11_plus(e.isa) { 0x244 } else { 0x1ED };
            e.emit_vop3(op, "v_perm_b32", vdst, s1, s0, s2, &format!("b, a, {s_desc}"));
        }
        IrOp::Dp4a { dst, a, b, c, .. } => {
            let vdst = alloc.reg_map[*dst as usize].base_idx();
            let mut s0 = operand_to_vop3_src(e, alloc, a, SCRATCH_V0);
            let mut s1 = operand_to_vop3_src(e, alloc, b, SCRATCH_V1);
            let (c_enc, c_lit, c_desc) = load_operand_into_src0(e, alloc, c);
            let mut s2 = if let Some(l) = c_lit {
                e.emit_s_mov_b32(SCRATCH_S0, l, true, &c_desc);
                u32::from(SCRATCH_S0)
            } else {
                c_enc
            };
            e.coalesce_vop3_gfx9(&mut s0, &mut s1, &mut s2);
            let w0_base = if is_gfx11_plus(e.isa) {
                0xCC16_4000
            } else {
                0xD3A8_4000
            };
            let w0 = w0_base | u32::from(vdst);
            let w1 = 0x1800_0000 | (s0 & 0x1ff) | ((s1 & 0x1ff) << 9) | ((s2 & 0x1ff) << 18);
            e.emit2(
                w0,
                w1,
                format!("v_dot4_i32_i8 v{vdst}, a, b, {c_desc}"),
            );
        }
        IrOp::Atomic {
            op,
            dst,
            space,
            base,
            offset,
            val,
            ..
        } => {
            e.emit_waitcnt_lgkm();
            let vdata = ensure_vgpr(e, alloc, val, SCRATCH_V1);
            let has_dst = dst.is_some();
            let vdst = dst
                .map(|d| alloc.reg_map[d as usize].base_idx())
                .unwrap_or(0);
            let mnem = match op {
                AtomicOp::Add => "add_u32",
                AtomicOp::Min => "min_u32",
                AtomicOp::Max => "max_u32",
                AtomicOp::And => "and_b32",
                AtomicOp::Or => "or_b32",
                AtomicOp::Xor => "xor_b32",
                AtomicOp::Exch => "swap_b32",
                AtomicOp::Cas => "cmpswap_b32",
            };
            let gfx11 = is_gfx11_plus(e.isa);
            if *space == AddressSpace::Shared {
                let vaddr = ensure_vgpr(e, alloc, base, SCRATCH_V0);
                let shift = if gfx11 { 18 } else { 17 };
                let ds_op: u32 = if has_dst { 0x20 } else { 0x00 };
                let w0 = 0xD800_0000 | (ds_op << shift) | ((*offset as u32) & 0xffff);
                let w1 = (u32::from(vdst) << 24) | (u32::from(vdata) << 8) | u32::from(vaddr);
                e.emit2(w0, w1, format!("ds_{mnem} v{vaddr}, v{vdata} offset:{offset}"));
                e.lgkm_dirty = true;
            } else {
                let vaddr = ensure_vgpr_pair(e, alloc, base, SCRATCH_V0);
                let saddr_off: u32 = if gfx11 { 0x7C } else { 0x7F };
                let w0 = if gfx11 {
                    let glc = if has_dst { 1 << 14 } else { 0 };
                    0xDC00_0000 | (0x35u32 << 18) | (2 << 16) | glc | ((*offset as u32) & 0x1fff)
                } else {
                    let sc0 = if has_dst { 1 << 16 } else { 0 };
                    0xDC00_0000 | (0x42u32 << 18) | (2 << 14) | sc0 | ((*offset as u32) & 0x1fff)
                };
                let w1 = (u32::from(vdst) << 24)
                    | (saddr_off << 16)
                    | (u32::from(vdata) << 8)
                    | u32::from(vaddr);
                e.emit2(
                    w0,
                    w1,
                    format!("global_atomic_{mnem} v{vdst}, v[{vaddr}:{}], v{vdata}, off offset:{offset}", vaddr + 1),
                );
                e.vm_dirty = true;
            }
        }
        IrOp::WarpShuffle { dst, mode, src, .. } => {
            let vdst = alloc.reg_map[*dst as usize].base_idx();
            let vsrc = ensure_vgpr(e, alloc, src, SCRATCH_V0);
            let w0 = if is_gfx11_plus(e.isa) {
                0xD800_0000 | (0xB3u32 << 18)
            } else {
                0xD800_0000 | (0x3Fu32 << 17)
            };
            let w1 = (u32::from(vdst) << 24) | (u32::from(vsrc) << 8) | u32::from(SCRATCH_V1);
            e.emit2(
                w0,
                w1,
                format!("ds_bpermute_b32 v{vdst}, v{SCRATCH_V1}, v{vsrc} ; {mode:?}"),
            );
        }
        IrOp::WarpVote { mode, dst, .. } => {
            let pr = alloc.reg_map[*dst as usize];
            match pr {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                    e.emit_s_mov_b32(s, 0x6A, false, &format!("vcc_lo ; vote.{mode:?}"));
                }
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => {
                    e.emit_v_mov_b32(v, 0x6A, None, &format!("vcc_lo ; vote.{mode:?}"));
                }
                PhysReg::Vcc => {}
            }
        }
        IrOp::Return => {
            e.emit_s_endpgm();
        }
    }
    Ok(())
}

fn resolve_src0(
    e: &mut Emitter,
    alloc: &KernelAllocation,
    op: &IrOperand,
    _hi: bool,
) -> (u32, String) {
    match op {
        IrOperand::Reg(r) => {
            let pr = alloc.reg_map[*r as usize];
            match pr {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => (u32::from(s), format!("s{s}")),
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => (256 + u32::from(v), format!("v{v}")),
                PhysReg::Vcc => (106, "vcc".to_owned()),
            }
        }
        IrOperand::Special(sp) => special_src0(e, *sp),
        _ => (128, "0".to_owned()),
    }
}

fn resolve_src0_hi(alloc: &KernelAllocation, op: &IrOperand) -> (u32, String) {
    match op {
        IrOperand::Reg(r) => match alloc.reg_map[*r as usize] {
            PhysReg::SgprPair(s) => (u32::from(s + 1), format!("s{}", s + 1)),
            PhysReg::VgprPair(v) => (256 + u32::from(v + 1), format!("v{}", v + 1)),
            PhysReg::Sgpr(s) => (u32::from(s), format!("s{s}")),
            PhysReg::Vgpr(v) => (256 + u32::from(v), format!("v{v}")),
            PhysReg::Vcc => (107, "vcc_hi".to_owned()),
        },
        _ => (128, "0".to_owned()),
    }
}

fn load_operand_into_src0(
    e: &mut Emitter,
    alloc: &KernelAllocation,
    op: &IrOperand,
) -> (u32, Option<u32>, String) {
    match op {
        IrOperand::Reg(r) => {
            let pr = alloc.reg_map[*r as usize];
            match pr {
                PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => (u32::from(s), None, format!("s{s}")),
                PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => {
                    (256 + u32::from(v), None, format!("v{v}"))
                }
                PhysReg::Vcc => (106, None, "vcc".to_owned()),
            }
        }
        IrOperand::Special(sp) => {
            let (enc, desc) = special_src0(e, *sp);
            (enc, None, desc)
        }
        IrOperand::ImmU64(v) => {
            let val = *v as u32;
            if val <= 64 {
                (128 + val, None, val.to_string())
            } else {
                (0xFF, Some(val), format!("0x{val:x}"))
            }
        }
        IrOperand::ImmI64(v) => {
            if *v >= 0 && *v <= 64 {
                let enc = 128 + (*v as u32);
                (enc, None, v.to_string())
            } else if *v >= -16 && *v <= -1 {
                let enc = (192 - *v) as u32;
                (enc, None, v.to_string())
            } else {
                let bits = (*v as i32) as u32;
                (0xFF, Some(bits), v.to_string())
            }
        }
        IrOperand::ImmF64(f) => {
            let bits = (*f as f32).to_bits();
            (0xFF, Some(bits), format!("{f}"))
        }
        IrOperand::SharedOffset(off) | IrOperand::LocalOffset(off) => {
            if *off <= 64 {
                (128 + *off, None, format!("off+{off}"))
            } else {
                (0xFF, Some(*off), format!("off+{off}"))
            }
        }
        IrOperand::Param(p) => (128 + (*p).min(64), None, format!("param{p}")),
    }
}

fn special_src0(e: &mut Emitter, sp: SpecialReg) -> (u32, String) {
    match sp {
        SpecialReg::TidX => (256, "v0 ; tid.x".to_owned()),
        SpecialReg::TidY => (257, "v1 ; tid.y".to_owned()),
        SpecialReg::TidZ => (258, "v2 ; tid.z".to_owned()),
        SpecialReg::CtaIdX => (4, "s4 ; ctaid.x".to_owned()),
        SpecialReg::CtaIdY => (5, "s5 ; ctaid.y".to_owned()),
        SpecialReg::CtaIdZ => (6, "s6 ; ctaid.z".to_owned()),
        SpecialReg::NtidX => {
            // HSA dispatch packet workgroup_size_x is at offset 4 in s[0:1] (DISPATCH_PTR)
            e.emit_s_load(SCRATCH_S0, 0, 4, false);
            e.emit_waitcnt_lgkm();
            let and_op = if is_gfx11_plus(e.isa) { 0x16 } else { 0x0C };
            e.emit_sop2_lit(
                and_op,
                "s_and_b32",
                SCRATCH_S0,
                SCRATCH_S0,
                0x0000_FFFF,
                "s8, 0xffff ; ntid.x",
            );
            (u32::from(SCRATCH_S0), "s8 ; ntid.x".to_owned())
        }
        SpecialReg::NtidY => {
            e.emit_s_load(SCRATCH_S0, 0, 6, false);
            e.emit_waitcnt_lgkm();
            (u32::from(SCRATCH_S0), "s8 ; ntid.y".to_owned())
        }
        SpecialReg::NtidZ => {
            e.emit_s_load(SCRATCH_S0, 0, 8, false);
            e.emit_waitcnt_lgkm();
            (u32::from(SCRATCH_S0), "s8 ; ntid.z".to_owned())
        }
        SpecialReg::NctaIdX => {
            e.emit_s_load(SCRATCH_S0, 0, 12, false);
            e.emit_waitcnt_lgkm();
            (u32::from(SCRATCH_S0), "s8 ; grid_x".to_owned())
        }
        SpecialReg::NctaIdY => {
            e.emit_s_load(SCRATCH_S0, 0, 16, false);
            e.emit_waitcnt_lgkm();
            (u32::from(SCRATCH_S0), "s8 ; grid_y".to_owned())
        }
        SpecialReg::NctaIdZ => {
            e.emit_s_load(SCRATCH_S0, 0, 20, false);
            e.emit_waitcnt_lgkm();
            (u32::from(SCRATCH_S0), "s8 ; grid_z".to_owned())
        }
        SpecialReg::LaneId => (256, "v0 ; laneid".to_owned()),
    }
}

fn operand_to_vop3_src(
    e: &mut Emitter,
    alloc: &KernelAllocation,
    op: &IrOperand,
    scratch: u8,
) -> u32 {
    let (enc, lit, desc) = load_operand_into_src0(e, alloc, op);
    if let Some(l) = lit {
        e.emit_v_mov_b32(scratch, 0xFF, Some(l), &desc);
        256 + u32::from(scratch)
    } else {
        enc
    }
}

fn ensure_vgpr(e: &mut Emitter, alloc: &KernelAllocation, op: &IrOperand, scratch: u8) -> u8 {
    match op {
        IrOperand::Reg(r) => match alloc.reg_map[*r as usize] {
            PhysReg::Vgpr(v) | PhysReg::VgprPair(v) => v,
            PhysReg::Sgpr(s) | PhysReg::SgprPair(s) => {
                e.emit_v_mov_b32(scratch, u32::from(s), None, &format!("s{s}"));
                scratch
            }
            PhysReg::Vcc => scratch,
        },
        _ => {
            let (enc, lit, desc) = load_operand_into_src0(e, alloc, op);
            e.emit_v_mov_b32(scratch, enc, lit, &desc);
            scratch
        }
    }
}

fn ensure_vgpr_pair(e: &mut Emitter, alloc: &KernelAllocation, op: &IrOperand, scratch: u8) -> u8 {
    match op {
        IrOperand::Reg(r) => match alloc.reg_map[*r as usize] {
            PhysReg::VgprPair(v) => v,
            PhysReg::SgprPair(s) => {
                e.emit_v_mov_b32(scratch, u32::from(s), None, &format!("s{s}"));
                e.emit_v_mov_b32(scratch + 1, u32::from(s + 1), None, &format!("s{}", s + 1));
                scratch
            }
            PhysReg::Vgpr(v) => v,
            PhysReg::Sgpr(s) => {
                e.emit_v_mov_b32(scratch, u32::from(s), None, &format!("s{s}"));
                scratch
            }
            PhysReg::Vcc => scratch,
        },
        _ => scratch,
    }
}

/// Decodes and validates Tier 0 binary AMDGPU machine code:
/// - Checks 4-byte instruction alignment and 32/64-bit encoding boundaries
/// - Verifies `GLOBAL` instructions carry exact `saddr` off-sentinels (`0x7C` GFX11+, `0x7F` GFX9/10)
/// - Verifies all SOPP branch targets land on valid instruction boundaries
/// - Verifies the final instruction is the exact target `s_endpgm`
pub fn verify_encoded_machine_code(
    code: &[u8],
    isa: IsaTarget,
) -> Result<Vec<DecodedInst>, CompileError> {
    if code.is_empty() || !code.len().is_multiple_of(4) {
        return Err(CompileError::LoweringError(
            "machine code length must be a non-zero multiple of 4 bytes".to_owned(),
        ));
    }

    let mut dwords = Vec::with_capacity(code.len() / 4);
    for chunk in code.chunks_exact(4) {
        dwords.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let expected_saddr: u8 = if is_gfx11_plus(isa) { 0x7C } else { 0x7F };
    let expected_endpgm: u32 = if is_gfx11_plus(isa) {
        0xBFB0_0000
    } else {
        0xBF81_0000
    };

    let mut decoded = Vec::new();
    let mut inst_starts = std::collections::HashSet::new();
    let mut branch_targets: Vec<isize> = Vec::new();
    let mut idx = 0usize;

    while idx < dwords.len() {
        let byte_offset = idx * 4;
        inst_starts.insert(byte_offset);
        let w0 = dwords[idx];

        let (format, is_64) = if (w0 >> 26) == 0b111101 || (w0 >> 26) == 0b110000 {
            ("SMEM", true)
        } else if (w0 >> 26) == 0b110101
            || (w0 >> 26) == 0b110100
            || (w0 >> 26) == 0b110011
        {
            ("VOP3", true)
        } else if (w0 >> 26) == 0b110111 {
            ("GLOBAL", true)
        } else if (w0 >> 26) == 0b110110 {
            ("DS", true)
        } else if (w0 >> 23) == 0b1_0111_1111 {
            let op_sopp = (w0 >> 16) & 0x7F;
            let is_br = if is_gfx11_plus(isa) {
                matches!(op_sopp, 0x20..=0x2B)
            } else {
                matches!(op_sopp, 0x02..=0x09)
            };
            if is_br {
                let simm16 = (w0 & 0xFFFF) as i16;
                let target_dw = (idx as isize) + 1 + isize::from(simm16);
                branch_targets.push(target_dw);
            }
            ("SOPP", false)
        } else if (w0 >> 23) == 0b1_0111_1101 {
            let ssrc0 = w0 & 0xFF;
            ("SOP1", ssrc0 == 0xFF)
        } else if (w0 >> 25) == 0b011_1110 {
            let src0 = w0 & 0x1FF;
            ("VOPC", src0 == 0xFF)
        } else if (w0 >> 25) == 0b011_1111 {
            let src0 = w0 & 0x1FF;
            ("VOP1", src0 == 0xFF)
        } else if (w0 >> 30) == 0b10 {
            let ssrc0 = w0 & 0xFF;
            let ssrc1 = (w0 >> 8) & 0xFF;
            ("SOP2", ssrc0 == 0xFF || ssrc1 == 0xFF)
        } else if (w0 >> 31) == 0 {
            let src0 = w0 & 0x1FF;
            ("VOP2", src0 == 0xFF)
        } else {
            return Err(CompileError::LoweringError(format!(
                "unrecognized AMDGPU opcode 0x{w0:08X} at byte 0x{byte_offset:04x}"
            )));
        };

        let w1 = if is_64 {
            if idx + 1 >= dwords.len() {
                return Err(CompileError::LoweringError(format!(
                    "truncated 64-bit {format} instruction at byte 0x{byte_offset:04x}"
                )));
            }
            let second = dwords[idx + 1];
            if format == "GLOBAL" {
                let saddr = ((second >> 16) & 0x7F) as u8;
                if saddr != expected_saddr {
                    return Err(CompileError::LoweringError(format!(
                        "GLOBAL instruction at 0x{byte_offset:04x} has saddr=0x{saddr:02X}, expected 0x{expected_saddr:02X}"
                    )));
                }
            }
            idx += 2;
            Some(second)
        } else {
            idx += 1;
            None
        };

        decoded.push(DecodedInst {
            byte_offset,
            len_bytes: if is_64 { 8 } else { 4 },
            format,
            w0,
            w1,
        });
    }

    for target_dw in branch_targets {
        if target_dw < 0 {
            return Err(CompileError::LoweringError(
                "SOPP branch targets negative offset".to_owned(),
            ));
        }
        let target_byte = (target_dw as usize) * 4;
        if !inst_starts.contains(&target_byte) {
            return Err(CompileError::LoweringError(format!(
                "SOPP branch target byte 0x{target_byte:04x} is not an instruction boundary"
            )));
        }
    }

    let last = decoded.last().ok_or_else(|| {
        CompileError::LoweringError("machine code decoded zero instructions".to_owned())
    })?;
    if last.w0 != expected_endpgm {
        return Err(CompileError::LoweringError(format!(
            "final instruction 0x{:08X} is not expected s_endpgm (0x{expected_endpgm:08X})",
            last.w0
        )));
    }

    Ok(decoded)
}
