//! Spec-accurate 64-byte AMDHSA kernel descriptor (`amdhsa_kernel_descriptor_t`).

use crate::elf64::IsaTarget;
use core::mem::{offset_of, size_of};

/// `ENABLE_SGPR_DISPATCH_PTR` bit in `kernel_code_properties`.
pub const AMDHSA_PROP_ENABLE_SGPR_DISPATCH_PTR: u16 = 1 << 1;
/// `ENABLE_SGPR_KERNARG_SEGMENT_PTR` bit in `kernel_code_properties`.
pub const AMDHSA_PROP_ENABLE_SGPR_KERNARG_SEGMENT_PTR: u16 = 1 << 3;
/// `ENABLE_WAVEFRONT_SIZE32` bit in `kernel_code_properties` (RDNA Wave32).
pub const AMDHSA_PROP_ENABLE_WAVEFRONT_SIZE32: u16 = 1 << 10;

/// AMDHSA Hardware Kernel Descriptor (`amdhsa_kernel_descriptor_t`, 64 bytes).
///
/// Placed in `.rodata` (`<kernel>.kd`) and pointed to by `KernelDispatchPacket.kernel_object`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct AmdhsaKernelDescriptor {
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub kernarg_size: u32,
    pub reserved0: [u8; 4],
    pub kernel_code_entry_byte_offset: i64,
    pub reserved1: [u8; 20],
    pub compute_pgm_rsrc3: u32,
    pub compute_pgm_rsrc1: u32,
    pub compute_pgm_rsrc2: u32,
    pub kernel_code_properties: u16,
    pub kernarg_preload: u16,
    pub reserved2: [u8; 4],
}

const _: () = assert!(size_of::<AmdhsaKernelDescriptor>() == 64);
const _: () = assert!(offset_of!(AmdhsaKernelDescriptor, kernel_code_entry_byte_offset) == 16);
const _: () = assert!(offset_of!(AmdhsaKernelDescriptor, compute_pgm_rsrc3) == 44);
const _: () = assert!(offset_of!(AmdhsaKernelDescriptor, compute_pgm_rsrc1) == 48);
const _: () = assert!(offset_of!(AmdhsaKernelDescriptor, compute_pgm_rsrc2) == 52);
const _: () = assert!(offset_of!(AmdhsaKernelDescriptor, kernel_code_properties) == 56);

impl AmdhsaKernelDescriptor {
    /// Constructs a hardware kernel descriptor configured for `isa` and register/LDS counts.
    pub fn new(
        isa: IsaTarget,
        vgpr_count: u32,
        sgpr_count: u32,
        lds_bytes: u32,
        kernarg_size: u32,
        entry_byte_offset: i64,
    ) -> Self {
        // CDNA2/3 (gfx90a/gfx942) and RDNA allocate VGPRs at granularity 8; only CDNA1 (gfx908) uses granularity 4.
        let vgpr_gran_step = if matches!(isa, IsaTarget::Gfx908) {
            4
        } else {
            8
        };
        let gran_vgpr = vgpr_count
            .max(1)
            .div_ceil(vgpr_gran_step)
            .saturating_sub(1)
            .min(0x3f);
        let gran_sgpr = sgpr_count.max(1).div_ceil(8).saturating_sub(1).min(0x0f);

        // COMPUTE_PGM_RSRC1:
        //  [5:0]   = GRANULATED_WORKITEM_VGPR_COUNT
        //  [9:6]   = GRANULATED_WAVEFRONT_SGPR_COUNT (used on GFX9; 0 on GFX10/11/12)
        //  [19:16] = FLOAT_MODE (0x0F: round-to-nearest-even + denormals enabled)
        //  [21]    = DX10_CLAMP
        //  [23]    = IEEE_MODE
        //  [29]    = WGP_MODE (on GFX10/11+)
        //  [30]    = MEM_ORDERED (on GFX10/11+)
        //  [31]    = FWD_PROGRESS (on GFX10/11+)
        let mut rsrc1 = gran_vgpr | (0x0f << 16) | (1 << 21) | (1 << 23);
        if isa.is_rdna() {
            rsrc1 |= (1 << 29) | (1 << 30) | (1 << 31);
        } else {
            rsrc1 |= gran_sgpr << 6;
        }

        // COMPUTE_PGM_RSRC2:
        //  [5:1]   = USER_SGPR_COUNT = 4 (s[0:1] dispatch ptr + s[2:3] kernarg)
        //  [7]     = ENABLE_SGPR_WORKGROUP_ID_X
        //  [8]     = ENABLE_SGPR_WORKGROUP_ID_Y
        //  [9]     = ENABLE_SGPR_WORKGROUP_ID_Z
        //  [12:11] = ENABLE_VGPR_WORKITEM_ID (2 => X, Y, Z)
        //  [23:15] = LDS_SIZE (granularity 512B on RDNA, 256B on CDNA)
        let lds_gran = if isa.is_rdna() { 512 } else { 256 };
        let lds_blocks = lds_bytes.div_ceil(lds_gran).min(0x1ff);
        let rsrc2 = (4 << 1) | (1 << 7) | (1 << 8) | (1 << 9) | (2 << 11) | (lds_blocks << 15);

        let rsrc3 = match isa {
            // CDNA2/3 (gfx90a/gfx942): [5:0] ACCUM_OFFSET = ceil(arch_vgprs / 4) - 1 when AGPRs == 0
            IsaTarget::Gfx90a | IsaTarget::Gfx942 => {
                vgpr_count.max(1).div_ceil(4).saturating_sub(1).min(0x3f)
            }
            _ => 0,
        };

        let mut props =
            AMDHSA_PROP_ENABLE_SGPR_DISPATCH_PTR | AMDHSA_PROP_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
        if isa.is_rdna() {
            props |= AMDHSA_PROP_ENABLE_WAVEFRONT_SIZE32;
        }

        Self {
            group_segment_fixed_size: lds_bytes,
            private_segment_fixed_size: 0,
            kernarg_size,
            reserved0: [0; 4],
            kernel_code_entry_byte_offset: entry_byte_offset,
            reserved1: [0; 20],
            compute_pgm_rsrc3: rsrc3,
            compute_pgm_rsrc1: rsrc1,
            compute_pgm_rsrc2: rsrc2,
            kernel_code_properties: props,
            kernarg_preload: 0,
            reserved2: [0; 4],
        }
    }

    /// Serializes this 64-byte kernel descriptor in little-endian wire order.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.group_segment_fixed_size.to_le_bytes());
        out[4..8].copy_from_slice(&self.private_segment_fixed_size.to_le_bytes());
        out[8..12].copy_from_slice(&self.kernarg_size.to_le_bytes());
        out[12..16].copy_from_slice(&self.reserved0);
        out[16..24].copy_from_slice(&self.kernel_code_entry_byte_offset.to_le_bytes());
        out[24..44].copy_from_slice(&self.reserved1);
        out[44..48].copy_from_slice(&self.compute_pgm_rsrc3.to_le_bytes());
        out[48..52].copy_from_slice(&self.compute_pgm_rsrc1.to_le_bytes());
        out[52..56].copy_from_slice(&self.compute_pgm_rsrc2.to_le_bytes());
        out[56..58].copy_from_slice(&self.kernel_code_properties.to_le_bytes());
        out[58..60].copy_from_slice(&self.kernarg_preload.to_le_bytes());
        out[60..64].copy_from_slice(&self.reserved2);
        out
    }

    /// Deserializes a 64-byte slice into an [`AmdhsaKernelDescriptor`].
    pub fn from_bytes(b: &[u8; 64]) -> Self {
        let mut reserved0 = [0u8; 4];
        reserved0.copy_from_slice(&b[12..16]);
        let mut reserved1 = [0u8; 20];
        reserved1.copy_from_slice(&b[24..44]);
        let mut reserved2 = [0u8; 4];
        reserved2.copy_from_slice(&b[60..64]);

        Self {
            group_segment_fixed_size: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            private_segment_fixed_size: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            kernarg_size: u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            reserved0,
            kernel_code_entry_byte_offset: i64::from_le_bytes([
                b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23],
            ]),
            reserved1,
            compute_pgm_rsrc3: u32::from_le_bytes([b[44], b[45], b[46], b[47]]),
            compute_pgm_rsrc1: u32::from_le_bytes([b[48], b[49], b[50], b[51]]),
            compute_pgm_rsrc2: u32::from_le_bytes([b[52], b[53], b[54], b[55]]),
            kernel_code_properties: u16::from_le_bytes([b[56], b[57]]),
            kernarg_preload: u16::from_le_bytes([b[58], b[59]]),
            reserved2,
        }
    }

    /// Returns `true` if `ENABLE_WAVEFRONT_SIZE32` is set in `kernel_code_properties`.
    pub const fn is_wave32(&self) -> bool {
        (self.kernel_code_properties & AMDHSA_PROP_ENABLE_WAVEFRONT_SIZE32) != 0
    }

    /// Returns the granulated VGPR block count from `compute_pgm_rsrc1[5:0]`.
    pub const fn granulated_vgprs(&self) -> u32 {
        self.compute_pgm_rsrc1 & 0x3f
    }

    /// Returns the granulated SGPR block count from `compute_pgm_rsrc1[9:6]`.
    pub const fn granulated_sgprs(&self) -> u32 {
        (self.compute_pgm_rsrc1 >> 6) & 0x0f
    }
}
