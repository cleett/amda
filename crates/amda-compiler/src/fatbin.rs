//! NVIDIA fatbin container parser and pure-Rust LZ4 block codec.

use crate::error::CompileError;

/// Magic number for an NVIDIA fatbin container (`0xBA55ED50`).
pub const FATBIN_MAGIC: u32 = 0xBA55_ED50;
/// Magic number for an NVIDIA `__fatBinC_Wrapper_t` header (`0x466243B1`).
pub const FATBIN_WRAPPER_MAGIC: u32 = 0x4662_43B1;

const FATBIN_KIND_PTX: u16 = 1;
const FATBIN_FLAG_ZLIB: u64 = 0x1000;
const FATBIN_FLAG_LZ4: u64 = 0x2000;
const FATBIN_FLAG_ZSTD: u64 = 0x8000;
const FATBIN_FLAG_COMPRESSED_MASK: u64 = FATBIN_FLAG_ZLIB | FATBIN_FLAG_LZ4 | FATBIN_FLAG_ZSTD | 0x4000;

/// Extracted payload from an NVIDIA fatbin image or raw module buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractedInput<'a> {
    /// Raw UTF-8 PTX text directly passed to `cuModuleLoadData`.
    RawPtx(&'a str),
    /// PTX source unpacked from an uncompressed or LZ4-compressed fatbin container.
    FatbinPtx(String),
    /// Native AMDGPU HSACO ELF64 binary (`\x7FELF`).
    Hsaco(&'a [u8]),
}

/// Inspects raw module bytes passed to `cuModuleLoadData` and identifies/unpacks them.
pub fn inspect_module_bytes(data: &[u8]) -> Result<ExtractedInput<'_>, CompileError> {
    if data.len() >= 4 && data[0..4] == [0x7F, b'E', b'L', b'F'] {
        return Ok(ExtractedInput::Hsaco(data));
    }

    if data.len() >= 4 {
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic == FATBIN_MAGIC {
            let ptx = extract_ptx_from_fatbin(data)?;
            return Ok(ExtractedInput::FatbinPtx(ptx));
        }
        if magic == FATBIN_WRAPPER_MAGIC {
            if let Some(pos) = find_fatbin_magic(data) {
                let ptx = extract_ptx_from_fatbin(&data[pos..])?;
                return Ok(ExtractedInput::FatbinPtx(ptx));
            }
            return Err(CompileError::InvalidFatbin(
                "fatbin wrapper does not contain an inline fatbin container".to_owned(),
            ));
        }
    }

    // Otherwise treat as UTF-8 PTX source text (trimmed of trailing NULs).
    let trimmed_bytes = match data.iter().position(|&b| b == 0) {
        Some(nul) => &data[..nul],
        None => data,
    };
    let text = std::str::from_utf8(trimmed_bytes).map_err(|_| {
        CompileError::InvalidFatbin("module payload is neither ELF, fatbin, nor UTF-8 PTX".to_owned())
    })?;
    Ok(ExtractedInput::RawPtx(text))
}

/// Decompresses a raw LZ4 block into a buffer of up to `expected_uncompressed_len` bytes.
pub fn lz4_decompress_block(
    src: &[u8],
    expected_uncompressed_len: usize,
) -> Result<Vec<u8>, CompileError> {
    if src.is_empty() && expected_uncompressed_len == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(expected_uncompressed_len.min(16 * 1024 * 1024));
    let mut pos = 0usize;

    while pos < src.len() {
        let token = src[pos];
        pos += 1;

        // 1. Literal length + literals
        let mut lit_len = usize::from(token >> 4);
        if lit_len == 15 {
            loop {
                let extra = *src
                    .get(pos)
                    .ok_or_else(|| CompileError::InvalidFatbin("truncated LZ4 literal length".to_owned()))?;
                pos += 1;
                lit_len = lit_len
                    .checked_add(usize::from(extra))
                    .ok_or_else(|| CompileError::InvalidFatbin("LZ4 literal length overflow".to_owned()))?;
                if extra != 255 {
                    break;
                }
            }
        }

        let lit_end = pos
            .checked_add(lit_len)
            .ok_or_else(|| CompileError::InvalidFatbin("LZ4 literal offset overflow".to_owned()))?;
        if lit_end > src.len() {
            return Err(CompileError::InvalidFatbin(
                "truncated LZ4 literal payload".to_owned(),
            ));
        }
        if out.len().saturating_add(lit_len) > expected_uncompressed_len {
            return Err(CompileError::InvalidFatbin(
                "LZ4 decompressed data exceeds expected uncompressed_size".to_owned(),
            ));
        }
        out.extend_from_slice(&src[pos..lit_end]);
        pos = lit_end;

        // Last LZ4 sequence ends right after literals.
        if pos == src.len() {
            break;
        }

        // 2. Match offset (u16 LE) and match length
        if pos + 2 > src.len() {
            return Err(CompileError::InvalidFatbin(
                "truncated LZ4 match offset".to_owned(),
            ));
        }
        let offset = usize::from(u16::from_le_bytes([src[pos], src[pos + 1]]));
        pos += 2;
        if offset == 0 || offset > out.len() {
            return Err(CompileError::InvalidFatbin(format!(
                "invalid LZ4 back-reference offset {offset} (output len {})",
                out.len()
            )));
        }

        let mut match_len = usize::from(token & 0x0F) + 4;
        if (token & 0x0F) == 15 {
            loop {
                let extra = *src
                    .get(pos)
                    .ok_or_else(|| CompileError::InvalidFatbin("truncated LZ4 match length".to_owned()))?;
                pos += 1;
                match_len = match_len
                    .checked_add(usize::from(extra))
                    .ok_or_else(|| CompileError::InvalidFatbin("LZ4 match length overflow".to_owned()))?;
                if extra != 255 {
                    break;
                }
            }
        }

        if out.len().saturating_add(match_len) > expected_uncompressed_len {
            return Err(CompileError::InvalidFatbin(
                "LZ4 match expansion exceeds expected uncompressed_size".to_owned(),
            ));
        }

        // Byte-by-byte copy handles overlapping RLE matches accurately.
        for _ in 0..match_len {
            let b = out[out.len() - offset];
            out.push(b);
        }
    }

    Ok(out)
}

/// Compresses `src` into a standard raw LZ4 block using a greedy hash-table matcher.
pub fn lz4_compress_block(src: &[u8]) -> Vec<u8> {
    if src.is_empty() {
        return Vec::new();
    }

    const HASH_BITS: usize = 12;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    const MIN_MATCH: usize = 4;
    const LAST_LITERALS: usize = 5;
    const MF_LIMIT: usize = 12;

    let mut out = Vec::with_capacity(src.len() + 16);
    if src.len() < MF_LIMIT {
        emit_lz4_literals_only(src, &mut out);
        return out;
    }

    let mut table = [usize::MAX; HASH_SIZE];
    let mut ip = 0usize;
    let mut anchor = 0usize;
    let match_limit = src.len() - LAST_LITERALS;

    while ip + MIN_MATCH <= match_limit {
        let seq = u32::from_le_bytes([src[ip], src[ip + 1], src[ip + 2], src[ip + 3]]);
        let h = ((seq.wrapping_mul(2654435761)) >> (32 - HASH_BITS)) as usize;
        let cand = table[h];
        table[h] = ip;

        if cand != usize::MAX
            && ip > cand
            && (ip - cand) <= 65535
            && src[cand..cand + 4] == src[ip..ip + 4]
        {
            let mut mlen = 4usize;
            while ip + mlen < match_limit && src[cand + mlen] == src[ip + mlen] {
                mlen += 1;
            }

            let lit_len = ip - anchor;
            let match_extra = mlen - MIN_MATCH;
            let token = ((lit_len.min(15) as u8) << 4) | (match_extra.min(15) as u8);
            out.push(token);
            if lit_len >= 15 {
                emit_lz4_ext_len(lit_len - 15, &mut out);
            }
            out.extend_from_slice(&src[anchor..ip]);
            out.extend_from_slice(&((ip - cand) as u16).to_le_bytes());
            if match_extra >= 15 {
                emit_lz4_ext_len(match_extra - 15, &mut out);
            }

            ip += mlen;
            anchor = ip;
        } else {
            ip += 1;
        }
    }

    if anchor < src.len() {
        emit_lz4_literals_only(&src[anchor..], &mut out);
    }

    out
}

fn emit_lz4_literals_only(lits: &[u8], out: &mut Vec<u8>) {
    let lit_len = lits.len();
    out.push((lit_len.min(15) as u8) << 4);
    if lit_len >= 15 {
        emit_lz4_ext_len(lit_len - 15, out);
    }
    out.extend_from_slice(lits);
}

fn emit_lz4_ext_len(mut rem: usize, out: &mut Vec<u8>) {
    while rem >= 255 {
        out.push(255);
        rem -= 255;
    }
    out.push(rem as u8);
}

/// Extracts uncompressed or LZ4-compressed PTX entries from an NVIDIA fatbin (`0xBA55ED50`) buffer.
pub fn extract_ptx_from_fatbin(data: &[u8]) -> Result<String, CompileError> {
    if data.len() < 16 {
        return Err(CompileError::InvalidFatbin("truncated fatbin header".to_owned()));
    }
    let magic = read_u32_le(data, 0)?;
    if magic != FATBIN_MAGIC {
        return Err(CompileError::InvalidFatbin(format!(
            "unexpected fatbin magic 0x{magic:08x}"
        )));
    }
    let header_size = usize::from(read_u16_le(data, 6)?);
    let fat_size = read_u64_le(data, 8)? as usize;

    let end = header_size
        .checked_add(fat_size)
        .ok_or_else(|| CompileError::InvalidFatbin("fatbin size overflow".to_owned()))?;
    if header_size < 16 || end > data.len() {
        return Err(CompileError::InvalidFatbin(
            "fatbin payload extends past end of buffer".to_owned(),
        ));
    }

    let mut pos = header_size;
    let mut found_unsupported_compressed = false;
    while pos + 48 <= end {
        let kind = read_u16_le(data, pos)?;
        let entry_hdr_size = read_u32_le(data, pos + 4)? as usize;
        let padded_payload_size = read_u64_le(data, pos + 8)? as usize;
        let payload_size = read_u32_le(data, pos + 16)? as usize;
        let flags = read_u64_le(data, pos + 40)?;
        let uncompressed_size = if entry_hdr_size >= 64 && pos + 64 <= end {
            read_u64_le(data, pos + 56)? as usize
        } else {
            0
        };

        if entry_hdr_size < 48 {
            return Err(CompileError::InvalidFatbin(
                "invalid fatbin entry header size".to_owned(),
            ));
        }

        let payload_start = pos
            .checked_add(entry_hdr_size)
            .ok_or_else(|| CompileError::InvalidFatbin("entry offset overflow".to_owned()))?;
        let payload_end = payload_start
            .checked_add(padded_payload_size)
            .ok_or_else(|| CompileError::InvalidFatbin("entry payload overflow".to_owned()))?;
        if payload_end > end {
            return Err(CompileError::InvalidFatbin(
                "fatbin entry extends past fatbin payload".to_owned(),
            ));
        }

        if kind == FATBIN_KIND_PTX {
            let actual_len = if payload_size > 0 {
                payload_size.min(padded_payload_size)
            } else {
                padded_payload_size
            };
            let raw = &data[payload_start..payload_start + actual_len];

            if (flags & FATBIN_FLAG_LZ4) != 0 {
                let expected = if uncompressed_size > 0 {
                    uncompressed_size
                } else {
                    actual_len.saturating_mul(16).max(4096)
                };
                let decompressed = lz4_decompress_block(raw, expected)?;
                return decode_utf8_ptx(&decompressed);
            } else if (flags & FATBIN_FLAG_COMPRESSED_MASK) != 0 {
                found_unsupported_compressed = true;
            } else {
                return decode_utf8_ptx(raw);
            }
        }

        pos = payload_end;
    }

    if found_unsupported_compressed {
        Err(CompileError::CompressedFatbinUnsupported)
    } else {
        Err(CompileError::MissingPtxInFatbin)
    }
}

fn decode_utf8_ptx(raw: &[u8]) -> Result<String, CompileError> {
    let clean = match raw.iter().position(|&b| b == 0) {
        Some(nul) => &raw[..nul],
        None => raw,
    };
    let ptx = std::str::from_utf8(clean).map_err(|_| {
        CompileError::InvalidFatbin("PTX payload in fatbin is not valid UTF-8".to_owned())
    })?;
    Ok(ptx.to_owned())
}

/// Builds an uncompressed single-PTX NVIDIA fatbin container using spec-accurate entry offsets.
pub fn build_uncompressed_ptx_fatbin(ptx: &str) -> Vec<u8> {
    let mut raw = ptx.as_bytes().to_vec();
    raw.push(0);
    build_ptx_fatbin_entry(&raw, 0, raw.len())
}

/// Builds an LZ4-compressed single-PTX NVIDIA fatbin container (`flags = 0x2000`).
pub fn build_lz4_ptx_fatbin(ptx: &str) -> Vec<u8> {
    let mut raw = ptx.as_bytes().to_vec();
    raw.push(0);
    let compressed = lz4_compress_block(&raw);
    build_ptx_fatbin_entry(&compressed, FATBIN_FLAG_LZ4, raw.len())
}

fn build_ptx_fatbin_entry(payload_bytes: &[u8], flags: u64, uncompressed_size: usize) -> Vec<u8> {
    let actual_len = payload_bytes.len();
    let mut padded = payload_bytes.to_vec();
    while !padded.len().is_multiple_of(8) {
        padded.push(0);
    }

    let entry_hdr_size = 64usize;
    let total_entry_size = entry_hdr_size + padded.len();
    let mut out = vec![0u8; 16 + total_entry_size];

    // Fatbin header (16 bytes)
    out[0..4].copy_from_slice(&FATBIN_MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&1u16.to_le_bytes()); // version
    out[6..8].copy_from_slice(&16u16.to_le_bytes()); // header_size
    out[8..16].copy_from_slice(&(total_entry_size as u64).to_le_bytes());

    // Fatbin entry header (64 bytes, NVIDIA offsets)
    let e = &mut out[16..16 + entry_hdr_size];
    e[0..2].copy_from_slice(&FATBIN_KIND_PTX.to_le_bytes());
    e[2..4].copy_from_slice(&1u16.to_le_bytes()); // version
    e[4..8].copy_from_slice(&(entry_hdr_size as u32).to_le_bytes());
    e[8..16].copy_from_slice(&(padded.len() as u64).to_le_bytes()); // padded_payload_size
    e[16..20].copy_from_slice(&(actual_len as u32).to_le_bytes()); // payload_size
    e[40..48].copy_from_slice(&flags.to_le_bytes());
    e[56..64].copy_from_slice(&(uncompressed_size as u64).to_le_bytes());

    out[16 + entry_hdr_size..].copy_from_slice(&padded);
    out
}

fn find_fatbin_magic(data: &[u8]) -> Option<usize> {
    let target = FATBIN_MAGIC.to_le_bytes();
    data.windows(4).position(|w| w == target)
}

fn read_u16_le(data: &[u8], off: usize) -> Result<u16, CompileError> {
    let slice = data
        .get(off..off + 2)
        .ok_or_else(|| CompileError::InvalidFatbin("truncated u16".to_owned()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(data: &[u8], off: usize) -> Result<u32, CompileError> {
    let slice = data
        .get(off..off + 4)
        .ok_or_else(|| CompileError::InvalidFatbin("truncated u32".to_owned()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(data: &[u8], off: usize) -> Result<u64, CompileError> {
    let slice = data
        .get(off..off + 8)
        .ok_or_else(|| CompileError::InvalidFatbin("truncated u64".to_owned()))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}
