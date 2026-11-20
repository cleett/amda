//! AMDGPU HSACO ELF64 constants, target ISA enumeration, parser, and builder.

use crate::descriptor::AmdhsaKernelDescriptor;
use crate::error::ElfError;
use crate::metadata::{decode_amdhsa_metadata, encode_amdhsa_metadata, KernelMetadata};
use core::fmt;

/// ELF identification magic (`\x7FELF`).
pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
/// ELF64 class (`EI_CLASS`).
pub const EI_CLASS_64: u8 = 2;
/// Little-endian data encoding (`EI_DATA`).
pub const EI_DATA_LSB: u8 = 1;
/// AMDGPU HSA OS ABI identifier (`ELFOSABI_AMDGPU_HSA`).
pub const ELFOSABI_AMDGPU_HSA: u8 = 64;
/// AMDGPU HSA ABI version v2 (Code Object v4/v5).
pub const ELFABIVERSION_AMDGPU_HSA_V4: u8 = 2;
/// Shared object / relocatable code object ELF type (`ET_DYN`).
pub const ET_DYN: u16 = 3;
/// ELF machine architecture for AMD GPUs (`EM_AMDGPU`).
pub const EM_AMDGPU: u16 = 224;

/// Program header type: loadable segment.
pub const PT_LOAD: u32 = 1;
/// Program header type: auxiliary note information.
pub const PT_NOTE: u32 = 4;

/// Program header flags.
pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

/// Section header types.
pub const SHT_NULL: u32 = 0;
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_NOTE: u32 = 7;

/// Section header flags.
pub const SHF_WRITE: u64 = 1;
pub const SHF_ALLOC: u64 = 2;
pub const SHF_EXECINSTR: u64 = 4;

/// Note type for MsgPack-encoded `amdhsa` metadata (`NT_AMDGPU_METADATA`).
pub const NT_AMDGPU_METADATA: u32 = 32;

/// Mask isolating `EF_AMDGPU_MACH_*` in `e_flags`.
pub const EF_AMDGPU_MACH_MASK: u32 = 0x00ff;

pub const EF_AMDGPU_MACH_AMDGCN_GFX908: u32 = 0x030;
pub const EF_AMDGPU_MACH_AMDGCN_GFX90A: u32 = 0x03f;
pub const EF_AMDGPU_MACH_AMDGCN_GFX942: u32 = 0x04c;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1030: u32 = 0x036;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1100: u32 = 0x041;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1101: u32 = 0x042;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1102: u32 = 0x043;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1103: u32 = 0x044;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1200: u32 = 0x048;
pub const EF_AMDGPU_MACH_AMDGCN_GFX1201: u32 = 0x04e;

const ELF64_EHDR_SIZE: usize = 64;
const ELF64_PHDR_SIZE: usize = 56;
const ELF64_SHDR_SIZE: usize = 64;

/// Supported AMDGPU ISA generations across CDNA and RDNA families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaTarget {
    /// CDNA1 (Instinct MI100)
    Gfx908,
    /// CDNA2 (Instinct MI200 series)
    Gfx90a,
    /// CDNA3 (Instinct MI300 series)
    Gfx942,
    /// RDNA2 (Radeon RX 6800/6900)
    Gfx1030,
    /// RDNA3 (Radeon RX 7900 XT/XTX)
    Gfx1100,
    /// RDNA3 (Radeon RX 7700/7800 XT)
    Gfx1101,
    /// RDNA3 (Radeon RX 7600)
    Gfx1102,
    /// RDNA3 APU (Radeon 780M — Phoenix / Hawk Point)
    Gfx1103,
    /// RDNA4 (Radeon RX 8000/9060 class)
    Gfx1200,
    /// RDNA4 (Radeon RX 9070 XT)
    Gfx1201,
}

impl IsaTarget {
    /// Maps an `e_flags` machine code into an [`IsaTarget`].
    pub fn from_elf_flags(flags: u32) -> Result<Self, ElfError> {
        match flags & EF_AMDGPU_MACH_MASK {
            EF_AMDGPU_MACH_AMDGCN_GFX908 => Ok(Self::Gfx908),
            EF_AMDGPU_MACH_AMDGCN_GFX90A => Ok(Self::Gfx90a),
            EF_AMDGPU_MACH_AMDGCN_GFX942 => Ok(Self::Gfx942),
            EF_AMDGPU_MACH_AMDGCN_GFX1030 => Ok(Self::Gfx1030),
            EF_AMDGPU_MACH_AMDGCN_GFX1100 => Ok(Self::Gfx1100),
            EF_AMDGPU_MACH_AMDGCN_GFX1101 => Ok(Self::Gfx1101),
            EF_AMDGPU_MACH_AMDGCN_GFX1102 => Ok(Self::Gfx1102),
            EF_AMDGPU_MACH_AMDGCN_GFX1103 => Ok(Self::Gfx1103),
            EF_AMDGPU_MACH_AMDGCN_GFX1200 => Ok(Self::Gfx1200),
            EF_AMDGPU_MACH_AMDGCN_GFX1201 => Ok(Self::Gfx1201),
            _ => Err(ElfError::UnsupportedIsa(flags)),
        }
    }

    /// Converts this target into its `EF_AMDGPU_MACH_*` ELF `e_flags` value.
    pub const fn to_elf_flags(self) -> u32 {
        match self {
            Self::Gfx908 => EF_AMDGPU_MACH_AMDGCN_GFX908,
            Self::Gfx90a => EF_AMDGPU_MACH_AMDGCN_GFX90A,
            Self::Gfx942 => EF_AMDGPU_MACH_AMDGCN_GFX942,
            Self::Gfx1030 => EF_AMDGPU_MACH_AMDGCN_GFX1030,
            Self::Gfx1100 => EF_AMDGPU_MACH_AMDGCN_GFX1100,
            Self::Gfx1101 => EF_AMDGPU_MACH_AMDGCN_GFX1101,
            Self::Gfx1102 => EF_AMDGPU_MACH_AMDGCN_GFX1102,
            Self::Gfx1103 => EF_AMDGPU_MACH_AMDGCN_GFX1103,
            Self::Gfx1200 => EF_AMDGPU_MACH_AMDGCN_GFX1200,
            Self::Gfx1201 => EF_AMDGPU_MACH_AMDGCN_GFX1201,
        }
    }

    /// Canonical LLVM/ROCm target triple CPU string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gfx908 => "gfx908",
            Self::Gfx90a => "gfx90a",
            Self::Gfx942 => "gfx942",
            Self::Gfx1030 => "gfx1030",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1101 => "gfx1101",
            Self::Gfx1102 => "gfx1102",
            Self::Gfx1103 => "gfx1103",
            Self::Gfx1200 => "gfx1200",
            Self::Gfx1201 => "gfx1201",
        }
    }

    /// Parses a canonical target string (case-insensitive) such as `"gfx1100"` or `"gfx942"`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ElfError> {
        <Self as core::str::FromStr>::from_str(s)
    }
}

impl core::str::FromStr for IsaTarget {
    type Err = ElfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gfx908" => Ok(Self::Gfx908),
            "gfx90a" => Ok(Self::Gfx90a),
            "gfx942" => Ok(Self::Gfx942),
            "gfx1030" => Ok(Self::Gfx1030),
            "gfx1100" => Ok(Self::Gfx1100),
            "gfx1101" => Ok(Self::Gfx1101),
            "gfx1102" => Ok(Self::Gfx1102),
            "gfx1103" => Ok(Self::Gfx1103),
            "gfx1200" => Ok(Self::Gfx1200),
            "gfx1201" => Ok(Self::Gfx1201),
            _ => Err(ElfError::UnsupportedIsa(0)),
        }
    }
}

impl IsaTarget {

    /// Default wavefront width: `32` on RDNA (`gfx10xx/11xx/12xx`), `64` on CDNA (`gfx9xx`).
    pub const fn native_wave_size(self) -> u32 {
        match self {
            Self::Gfx908 | Self::Gfx90a | Self::Gfx942 => 64,
            Self::Gfx1030
            | Self::Gfx1100
            | Self::Gfx1101
            | Self::Gfx1102
            | Self::Gfx1103
            | Self::Gfx1200
            | Self::Gfx1201 => 32,
        }
    }

    /// Returns `true` for CDNA datacenter targets (`gfx9xx`).
    pub const fn is_cdna(self) -> bool {
        matches!(self, Self::Gfx908 | Self::Gfx90a | Self::Gfx942)
    }

    /// Returns `true` for RDNA targets (`gfx10xx/11xx/12xx`).
    pub const fn is_rdna(self) -> bool {
        !self.is_cdna()
    }
}

impl fmt::Display for IsaTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parsed view of an ELF64 section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfSection {
    /// Section name resolved from `.shstrtab`.
    pub name: String,
    /// `sh_type`.
    pub sh_type: u32,
    /// `sh_flags`.
    pub sh_flags: u64,
    /// Virtual address (`sh_addr`).
    pub addr: u64,
    /// File offset (`sh_offset`).
    pub offset: u64,
    /// Raw section contents.
    pub data: Vec<u8>,
}

/// Parsed AMDGPU HSACO ELF64 code object image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HsacoImage {
    /// Target ISA decoded from `e_flags`.
    pub isa: IsaTarget,
    /// Decoded `amdhsa.kernels` metadata records from `PT_NOTE` / `.note`.
    pub kernels: Vec<KernelMetadata>,
    /// Decoded 64-byte `amdhsa_kernel_descriptor_t` entries from `.rodata` (if present).
    pub kernel_descriptors: Vec<AmdhsaKernelDescriptor>,
    /// Parsed ELF sections.
    pub sections: Vec<ElfSection>,
    /// Optional serialized AMDA IR payload from the `.amda.ir` section.
    pub amda_ir_payload: Option<Vec<u8>>,
}

/// Parses an AMDGPU HSACO ELF64 byte image, validating ELF headers and decoding `.note` metadata.
pub fn parse_hsaco(bytes: &[u8]) -> Result<HsacoImage, ElfError> {
    if bytes.len() < ELF64_EHDR_SIZE {
        return Err(ElfError::Truncated);
    }
    if bytes[0..4] != ELF_MAGIC {
        return Err(ElfError::InvalidMagic);
    }
    if bytes[4] != EI_CLASS_64 || bytes[5] != EI_DATA_LSB {
        return Err(ElfError::UnsupportedClass);
    }

    let machine = read_u16_le(bytes, 18)?;
    if machine != EM_AMDGPU {
        return Err(ElfError::UnsupportedMachine(machine));
    }

    let e_phoff = read_u64_le(bytes, 32)? as usize;
    let e_shoff = read_u64_le(bytes, 40)? as usize;
    let e_flags = read_u32_le(bytes, 48)?;
    let e_phentsize = usize::from(read_u16_le(bytes, 54)?);
    let e_phnum = usize::from(read_u16_le(bytes, 56)?);
    let e_shentsize = usize::from(read_u16_le(bytes, 58)?);
    let e_shnum = usize::from(read_u16_le(bytes, 60)?);
    let e_shstrndx = usize::from(read_u16_le(bytes, 62)?);

    let isa = IsaTarget::from_elf_flags(e_flags)?;

    // Parse program headers and locate PT_NOTE if present.
    let mut kernels: Option<Vec<KernelMetadata>> = None;
    if e_phnum > 0 {
        if e_phentsize < ELF64_PHDR_SIZE {
            return Err(ElfError::Truncated);
        }
        for i in 0..e_phnum {
            let off = e_phoff
                .checked_add(i.checked_mul(e_phentsize).ok_or(ElfError::OutOfBounds)?)
                .ok_or(ElfError::OutOfBounds)?;
            let end = off.checked_add(ELF64_PHDR_SIZE).ok_or(ElfError::OutOfBounds)?;
            if end > bytes.len() {
                return Err(ElfError::OutOfBounds);
            }
            let p_type = read_u32_le(bytes, off)?;
            if p_type == PT_NOTE {
                let p_offset = read_u64_le(bytes, off + 8)? as usize;
                let p_filesz = read_u64_le(bytes, off + 32)? as usize;
                let note_end = p_offset.checked_add(p_filesz).ok_or(ElfError::OutOfBounds)?;
                if note_end > bytes.len() {
                    return Err(ElfError::OutOfBounds);
                }
                if let Some(parsed) = parse_note_segment(&bytes[p_offset..note_end])? {
                    kernels = Some(parsed);
                }
            }
        }
    }

    // Parse section headers.
    let mut raw_sections = Vec::with_capacity(e_shnum);
    if e_shnum > 0 {
        if e_shentsize < ELF64_SHDR_SIZE {
            return Err(ElfError::Truncated);
        }
        for i in 0..e_shnum {
            let off = e_shoff
                .checked_add(i.checked_mul(e_shentsize).ok_or(ElfError::OutOfBounds)?)
                .ok_or(ElfError::OutOfBounds)?;
            let end = off.checked_add(ELF64_SHDR_SIZE).ok_or(ElfError::OutOfBounds)?;
            if end > bytes.len() {
                return Err(ElfError::OutOfBounds);
            }
            let sh_name = read_u32_le(bytes, off)?;
            let sh_type = read_u32_le(bytes, off + 4)?;
            let sh_flags = read_u64_le(bytes, off + 8)?;
            let sh_addr = read_u64_le(bytes, off + 16)?;
            let sh_offset = read_u64_le(bytes, off + 24)?;
            let sh_size = read_u64_le(bytes, off + 32)?;

            let data = if sh_type == SHT_NULL || sh_size == 0 {
                Vec::new()
            } else {
                let start = sh_offset as usize;
                let size = sh_size as usize;
                let sec_end = start.checked_add(size).ok_or(ElfError::OutOfBounds)?;
                if sec_end > bytes.len() {
                    return Err(ElfError::OutOfBounds);
                }
                bytes[start..sec_end].to_vec()
            };

            raw_sections.push((sh_name, sh_type, sh_flags, sh_addr, sh_offset, data));
        }
    }

    let shstrtab = raw_sections
        .get(e_shstrndx)
        .map(|s| s.5.clone())
        .unwrap_or_default();

    let mut sections = Vec::with_capacity(raw_sections.len());
    let mut amda_ir_payload = None;
    let mut kernel_descriptors = Vec::new();

    for (sh_name, sh_type, sh_flags, sh_addr, sh_offset, data) in raw_sections {
        let name = read_c_string(&shstrtab, sh_name as usize);
        if kernels.is_none() && sh_type == SHT_NOTE {
            if let Some(parsed) = parse_note_segment(&data)? {
                kernels = Some(parsed);
            }
        }
        if name == ".amda.ir" && !data.is_empty() {
            amda_ir_payload = Some(data.clone());
        }
        if name == ".rodata" && !data.is_empty() && (data.len() % 64) == 0 {
            for chunk in data.chunks_exact(64) {
                let arr: &[u8; 64] = chunk.try_into().expect("64-byte chunk");
                kernel_descriptors.push(AmdhsaKernelDescriptor::from_bytes(arr));
            }
        }
        sections.push(ElfSection {
            name,
            sh_type,
            sh_flags,
            addr: sh_addr,
            offset: sh_offset,
            data,
        });
    }

    let kernels = kernels.ok_or(ElfError::MissingMetadata)?;

    Ok(HsacoImage {
        isa,
        kernels,
        kernel_descriptors,
        sections,
        amda_ir_payload,
    })
}

/// Builds a valid little-endian ELF64 `ET_DYN`/`EM_AMDGPU` HSACO image.
///
/// Delegates to [`build_hsaco_with_rodata`] with no `.rodata` kernel descriptor section.
pub fn build_hsaco(
    isa: IsaTarget,
    kernels: &[KernelMetadata],
    code_bytes: &[u8],
    amda_ir_payload: Option<&[u8]>,
) -> Vec<u8> {
    build_hsaco_with_rodata(isa, kernels, None, code_bytes, amda_ir_payload)
}

/// Builds a valid little-endian ELF64 `ET_DYN`/`EM_AMDGPU` HSACO image with an optional
/// `.rodata` section carrying 64-byte [`AmdhsaKernelDescriptor`] records.
///
/// Keeps section headers (`e_shoff`) at the end of the image so total ELF extent equals
/// `e_shoff + e_shnum * e_shentsize`.
pub fn build_hsaco_with_rodata(
    isa: IsaTarget,
    kernels: &[KernelMetadata],
    rodata_bytes: Option<&[u8]>,
    code_bytes: &[u8],
    amda_ir_payload: Option<&[u8]>,
) -> Vec<u8> {
    let msgpack = encode_amdhsa_metadata(kernels);
    let note_bytes = build_amdgpu_note(&msgpack);

    let has_rodata = rodata_bytes.is_some_and(|b| !b.is_empty());
    let has_ir = amda_ir_payload.is_some_and(|b| !b.is_empty());

    let mut shstrtab = Vec::new();
    let idx_null = push_shstr(&mut shstrtab, "");
    let idx_text = push_shstr(&mut shstrtab, ".text");
    let idx_rodata = if has_rodata {
        Some(push_shstr(&mut shstrtab, ".rodata"))
    } else {
        None
    };
    let idx_note = push_shstr(&mut shstrtab, ".note");
    let idx_amda_ir = if has_ir {
        Some(push_shstr(&mut shstrtab, ".amda.ir"))
    } else {
        None
    };
    let idx_shstrtab = push_shstr(&mut shstrtab, ".shstrtab");

    let num_phdrs: u16 = if has_rodata { 3 } else { 2 };
    let num_shdrs: u16 = 4 + u16::from(has_rodata) + u16::from(has_ir);
    let shstrndx: u16 = num_shdrs - 1;

    let phoff = ELF64_EHDR_SIZE;
    let headers_end = phoff + (num_phdrs as usize) * ELF64_PHDR_SIZE;

    let text_offset = align_up(headers_end, 16);
    let text_vaddr: u64 = 0x1000;
    let fallback_endpgm: [u8; 4] = if isa.is_rdna() {
        0xBFB0_0000u32.to_le_bytes()
    } else {
        0xBF81_0000u32.to_le_bytes()
    };
    let text_data = if code_bytes.is_empty() {
        &fallback_endpgm[..]
    } else {
        code_bytes
    };

    let mut cursor = text_offset + text_data.len();

    let rodata_layout = if let Some(rdata) = rodata_bytes.filter(|b| !b.is_empty()) {
        let off = align_up(cursor, 64);
        let vaddr = align_up_u64(text_vaddr + (text_data.len() as u64), 0x1000);
        cursor = off + rdata.len();
        Some((off, vaddr, rdata))
    } else {
        None
    };

    let note_offset = align_up(cursor, 8);
    cursor = note_offset + note_bytes.len();

    let ir_offset = if has_ir {
        let off = align_up(cursor, 8);
        cursor = off + amda_ir_payload.unwrap().len();
        Some(off)
    } else {
        None
    };

    let shstrtab_offset = align_up(cursor, 4);
    cursor = shstrtab_offset + shstrtab.len();

    let shoff = align_up(cursor, 8);
    let total_size = shoff + (num_shdrs as usize) * ELF64_SHDR_SIZE;

    let mut out = vec![0u8; total_size];

    // ELF64 Header (64 bytes)
    out[0..4].copy_from_slice(&ELF_MAGIC);
    out[4] = EI_CLASS_64;
    out[5] = EI_DATA_LSB;
    out[6] = 1; // EV_CURRENT
    out[7] = ELFOSABI_AMDGPU_HSA;
    out[8] = ELFABIVERSION_AMDGPU_HSA_V4;
    write_u16_le(&mut out, 16, ET_DYN);
    write_u16_le(&mut out, 18, EM_AMDGPU);
    write_u32_le(&mut out, 20, 1); // EV_CURRENT
    write_u64_le(&mut out, 24, text_vaddr); // e_entry
    write_u64_le(&mut out, 32, phoff as u64);
    write_u64_le(&mut out, 40, shoff as u64);
    write_u32_le(&mut out, 48, isa.to_elf_flags());
    write_u16_le(&mut out, 52, ELF64_EHDR_SIZE as u16);
    write_u16_le(&mut out, 54, ELF64_PHDR_SIZE as u16);
    write_u16_le(&mut out, 56, num_phdrs);
    write_u16_le(&mut out, 58, ELF64_SHDR_SIZE as u16);
    write_u16_le(&mut out, 60, num_shdrs);
    write_u16_le(&mut out, 62, shstrndx);

    // Program Header 0: PT_LOAD for .text
    write_phdr(
        &mut out[phoff..phoff + ELF64_PHDR_SIZE],
        PT_LOAD,
        PF_R | PF_X,
        text_offset as u64,
        text_vaddr,
        text_data.len() as u64,
        text_data.len() as u64,
        0x1000,
    );

    let mut next_ph_idx = 1usize;
    if let Some((roff, rvaddr, rdata)) = rodata_layout {
        let ph = phoff + next_ph_idx * ELF64_PHDR_SIZE;
        write_phdr(
            &mut out[ph..ph + ELF64_PHDR_SIZE],
            PT_LOAD,
            PF_R,
            roff as u64,
            rvaddr,
            rdata.len() as u64,
            rdata.len() as u64,
            0x1000,
        );
        next_ph_idx += 1;
    }

    // Last Program Header: PT_NOTE for .note
    let ph_note = phoff + next_ph_idx * ELF64_PHDR_SIZE;
    write_phdr(
        &mut out[ph_note..ph_note + ELF64_PHDR_SIZE],
        PT_NOTE,
        PF_R,
        note_offset as u64,
        0,
        note_bytes.len() as u64,
        note_bytes.len() as u64,
        4,
    );

    // Section payloads
    out[text_offset..text_offset + text_data.len()].copy_from_slice(text_data);
    if let Some((roff, _, rdata)) = rodata_layout {
        out[roff..roff + rdata.len()].copy_from_slice(rdata);
    }
    out[note_offset..note_offset + note_bytes.len()].copy_from_slice(&note_bytes);
    if let (Some(off), Some(payload)) = (ir_offset, amda_ir_payload) {
        if !payload.is_empty() {
            out[off..off + payload.len()].copy_from_slice(payload);
        }
    }
    out[shstrtab_offset..shstrtab_offset + shstrtab.len()].copy_from_slice(&shstrtab);

    // Section Headers
    let mut sh_idx = 0usize;
    // 0: SHT_NULL
    write_shdr(
        &mut out[shoff..shoff + ELF64_SHDR_SIZE],
        idx_null,
        SHT_NULL,
        0,
        0,
        0,
        0,
        1,
    );
    sh_idx += 1;

    // 1: .text
    let off = shoff + sh_idx * ELF64_SHDR_SIZE;
    write_shdr(
        &mut out[off..off + ELF64_SHDR_SIZE],
        idx_text,
        SHT_PROGBITS,
        SHF_ALLOC | SHF_EXECINSTR,
        text_vaddr,
        text_offset as u64,
        text_data.len() as u64,
        16,
    );
    sh_idx += 1;

    // Optional .rodata
    if let (Some(name_idx), Some((roff, rvaddr, rdata))) = (idx_rodata, rodata_layout) {
        let off = shoff + sh_idx * ELF64_SHDR_SIZE;
        write_shdr(
            &mut out[off..off + ELF64_SHDR_SIZE],
            name_idx,
            SHT_PROGBITS,
            SHF_ALLOC,
            rvaddr,
            roff as u64,
            rdata.len() as u64,
            64,
        );
        sh_idx += 1;
    }

    // .note
    let off = shoff + sh_idx * ELF64_SHDR_SIZE;
    write_shdr(
        &mut out[off..off + ELF64_SHDR_SIZE],
        idx_note,
        SHT_NOTE,
        0,
        0,
        note_offset as u64,
        note_bytes.len() as u64,
        4,
    );
    sh_idx += 1;

    // Optional .amda.ir
    if let (Some(name_idx), Some(sec_off), Some(payload)) = (idx_amda_ir, ir_offset, amda_ir_payload)
    {
        let off = shoff + sh_idx * ELF64_SHDR_SIZE;
        write_shdr(
            &mut out[off..off + ELF64_SHDR_SIZE],
            name_idx,
            SHT_PROGBITS,
            0,
            0,
            sec_off as u64,
            payload.len() as u64,
            8,
        );
        sh_idx += 1;
    }

    // Last: .shstrtab
    let off = shoff + sh_idx * ELF64_SHDR_SIZE;
    write_shdr(
        &mut out[off..off + ELF64_SHDR_SIZE],
        idx_shstrtab,
        SHT_STRTAB,
        0,
        0,
        shstrtab_offset as u64,
        shstrtab.len() as u64,
        1,
    );

    out
}

fn build_amdgpu_note(desc: &[u8]) -> Vec<u8> {
    const NAME: &[u8] = b"AMDGPU\0";
    let namesz = NAME.len() as u32;
    let descsz = desc.len() as u32;
    let padded_name = align_up(NAME.len(), 4);
    let padded_desc = align_up(desc.len(), 4);

    let mut note = vec![0u8; 12 + padded_name + padded_desc];
    write_u32_le(&mut note, 0, namesz);
    write_u32_le(&mut note, 4, descsz);
    write_u32_le(&mut note, 8, NT_AMDGPU_METADATA);
    note[12..12 + NAME.len()].copy_from_slice(NAME);
    note[12 + padded_name..12 + padded_name + desc.len()].copy_from_slice(desc);
    note
}

fn parse_note_segment(note: &[u8]) -> Result<Option<Vec<KernelMetadata>>, ElfError> {
    let mut pos = 0usize;
    while pos + 12 <= note.len() {
        let namesz = read_u32_le(note, pos)? as usize;
        let descsz = read_u32_le(note, pos + 4)? as usize;
        let n_type = read_u32_le(note, pos + 8)?;
        pos += 12;

        let name_aligned = align_up(namesz, 4);
        let desc_aligned = align_up(descsz, 4);

        let name_end = pos.checked_add(namesz).ok_or(ElfError::OutOfBounds)?;
        let desc_start = pos.checked_add(name_aligned).ok_or(ElfError::OutOfBounds)?;
        let desc_end = desc_start.checked_add(descsz).ok_or(ElfError::OutOfBounds)?;
        let next_pos = desc_start
            .checked_add(desc_aligned)
            .ok_or(ElfError::OutOfBounds)?;

        if desc_end > note.len() {
            return Err(ElfError::Truncated);
        }

        let name_slice = &note[pos..name_end];
        let trimmed_name = name_slice.strip_suffix(b"\0").unwrap_or(name_slice);
        if n_type == NT_AMDGPU_METADATA && trimmed_name == b"AMDGPU" {
            let kernels = decode_amdhsa_metadata(&note[desc_start..desc_end])?;
            return Ok(Some(kernels));
        }

        pos = next_pos.min(note.len());
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn write_phdr(
    dst: &mut [u8],
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
) {
    write_u32_le(dst, 0, p_type);
    write_u32_le(dst, 4, p_flags);
    write_u64_le(dst, 8, p_offset);
    write_u64_le(dst, 16, p_vaddr);
    write_u64_le(dst, 24, p_vaddr); // p_paddr
    write_u64_le(dst, 32, p_filesz);
    write_u64_le(dst, 40, p_memsz);
    write_u64_le(dst, 48, p_align);
}

#[allow(clippy::too_many_arguments)]
fn write_shdr(
    dst: &mut [u8],
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_addralign: u64,
) {
    write_u32_le(dst, 0, sh_name);
    write_u32_le(dst, 4, sh_type);
    write_u64_le(dst, 8, sh_flags);
    write_u64_le(dst, 16, sh_addr);
    write_u64_le(dst, 24, sh_offset);
    write_u64_le(dst, 32, sh_size);
    write_u32_le(dst, 40, 0); // sh_link
    write_u32_le(dst, 44, 0); // sh_info
    write_u64_le(dst, 48, sh_addralign);
    write_u64_le(dst, 56, 0); // sh_entsize
}

fn push_shstr(tab: &mut Vec<u8>, s: &str) -> u32 {
    let idx = tab.len() as u32;
    tab.extend_from_slice(s.as_bytes());
    tab.push(0);
    idx
}

fn read_c_string(tab: &[u8], offset: usize) -> String {
    if offset >= tab.len() {
        return String::new();
    }
    let slice = &tab[offset..];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).into_owned()
}

const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

const fn align_up_u64(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

fn read_u16_le(buf: &[u8], off: usize) -> Result<u16, ElfError> {
    let slice = buf.get(off..off + 2).ok_or(ElfError::Truncated)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, ElfError> {
    let slice = buf.get(off..off + 4).ok_or(ElfError::Truncated)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(buf: &[u8], off: usize) -> Result<u64, ElfError> {
    let slice = buf.get(off..off + 8).ok_or(ElfError::Truncated)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn write_u16_le(buf: &mut [u8], off: usize, val: u16) {
    buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

fn write_u32_le(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

fn write_u64_le(buf: &mut [u8], off: usize, val: u64) {
    buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
}
