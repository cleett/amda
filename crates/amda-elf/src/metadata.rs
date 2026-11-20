//! AMDGPU HSA code object kernel metadata and pure-Rust MessagePack codec.

use crate::error::ElfError;

/// Semantic kind of a kernel argument in AMDGPU HSA metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueKind {
    /// Pointer to a global memory buffer (`"global_buffer"`).
    GlobalBuffer,
    /// Scalar / struct passed by value in the kernarg buffer (`"by_value"`).
    ByValue,
}

impl ValueKind {
    /// Returns the AMDGPU metadata string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GlobalBuffer => "global_buffer",
            Self::ByValue => "by_value",
        }
    }

    /// Parses a metadata `.value_kind` string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ElfError> {
        <Self as core::str::FromStr>::from_str(s)
    }
}

impl core::str::FromStr for ValueKind {
    type Err = ElfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "global_buffer" => Ok(Self::GlobalBuffer),
            "by_value" => Ok(Self::ByValue),
            other => Err(ElfError::InvalidMsgPack(format!(
                "unsupported .value_kind '{other}'"
            ))),
        }
    }
}

/// Metadata describing one argument inside a kernel's `kernarg` segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArgMetadata {
    /// Argument name.
    pub name: String,
    /// Byte offset from the start of the `kernarg` buffer.
    pub offset: u32,
    /// Size of the argument in bytes.
    pub size: u32,
    /// Semantic kind (global pointer vs scalar value).
    pub value_kind: ValueKind,
}

/// AMDGPU HSA metadata for an individual kernel entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelMetadata {
    /// Source-level kernel name (e.g. `"vector_add"`).
    pub name: String,
    /// Kernel descriptor symbol name (e.g. `"vector_add.kd"`).
    pub symbol: String,
    /// Total byte size of the `kernarg` segment.
    pub kernarg_segment_size: u32,
    /// Static LDS / group segment bytes required per workgroup.
    pub group_segment_fixed_size: u32,
    /// Static scratch / private segment bytes per work-item.
    pub private_segment_fixed_size: u32,
    /// Native wavefront width (`32` on RDNA, `64` on CDNA).
    pub wavefront_size: u32,
    /// Maximum supported flat workgroup size.
    pub max_flat_workgroup_size: u32,
    /// Ordered argument layout descriptors.
    pub args: Vec<KernelArgMetadata>,
}

/// Minimal MessagePack value tree used for encoding/decoding `amdhsa` note payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MsgPackValue {
    UInt(u64),
    Int(i64),
    Str(String),
    Array(Vec<MsgPackValue>),
    Map(Vec<(String, MsgPackValue)>),
}

/// Encodes `kernels` into a standard `amdhsa.version = [1, 2]` MessagePack payload.
pub fn encode_amdhsa_metadata(kernels: &[KernelMetadata]) -> Vec<u8> {
    let kernel_values: Vec<MsgPackValue> = kernels
        .iter()
        .map(|k| {
            let args_val = k
                .args
                .iter()
                .map(|a| {
                    MsgPackValue::Map(vec![
                        (".name".to_owned(), MsgPackValue::Str(a.name.clone())),
                        (".offset".to_owned(), MsgPackValue::UInt(u64::from(a.offset))),
                        (".size".to_owned(), MsgPackValue::UInt(u64::from(a.size))),
                        (
                            ".value_kind".to_owned(),
                            MsgPackValue::Str(a.value_kind.as_str().to_owned()),
                        ),
                    ])
                })
                .collect();

            MsgPackValue::Map(vec![
                (".name".to_owned(), MsgPackValue::Str(k.name.clone())),
                (".symbol".to_owned(), MsgPackValue::Str(k.symbol.clone())),
                (
                    ".kernarg_segment_size".to_owned(),
                    MsgPackValue::UInt(u64::from(k.kernarg_segment_size)),
                ),
                (
                    ".group_segment_fixed_size".to_owned(),
                    MsgPackValue::UInt(u64::from(k.group_segment_fixed_size)),
                ),
                (
                    ".private_segment_fixed_size".to_owned(),
                    MsgPackValue::UInt(u64::from(k.private_segment_fixed_size)),
                ),
                (
                    ".wavefront_size".to_owned(),
                    MsgPackValue::UInt(u64::from(k.wavefront_size)),
                ),
                (
                    ".max_flat_workgroup_size".to_owned(),
                    MsgPackValue::UInt(u64::from(k.max_flat_workgroup_size)),
                ),
                (".args".to_owned(), MsgPackValue::Array(args_val)),
            ])
        })
        .collect();

    let root = MsgPackValue::Map(vec![
        (
            "amdhsa.version".to_owned(),
            MsgPackValue::Array(vec![MsgPackValue::UInt(1), MsgPackValue::UInt(2)]),
        ),
        ("amdhsa.kernels".to_owned(), MsgPackValue::Array(kernel_values)),
    ]);

    let mut out = Vec::with_capacity(256);
    encode_value(&root, &mut out);
    out
}

/// Decodes an `amdhsa` MessagePack payload into a list of [`KernelMetadata`] records.
pub fn decode_amdhsa_metadata(bytes: &[u8]) -> Result<Vec<KernelMetadata>, ElfError> {
    let mut reader = MsgPackReader::new(bytes);
    let root = reader.read_value()?;
    let root_map = match root {
        MsgPackValue::Map(m) => m,
        _ => {
            return Err(ElfError::InvalidMsgPack(
                "root metadata object must be a map".to_owned(),
            ))
        }
    };

    let kernels_val = find_map_entry(&root_map, "amdhsa.kernels")
        .ok_or_else(|| ElfError::InvalidMsgPack("missing 'amdhsa.kernels'".to_owned()))?;

    let kernels_arr = match kernels_val {
        MsgPackValue::Array(arr) => arr,
        _ => {
            return Err(ElfError::InvalidMsgPack(
                "'amdhsa.kernels' must be an array".to_owned(),
            ))
        }
    };

    let mut kernels = Vec::with_capacity(kernels_arr.len());
    for entry in kernels_arr {
        let map = match entry {
            MsgPackValue::Map(m) => m,
            _ => {
                return Err(ElfError::InvalidMsgPack(
                    "kernel entry must be a map".to_owned(),
                ))
            }
        };

        let name = expect_str(map, ".name")?;
        let symbol = find_str(map, ".symbol").unwrap_or_else(|| format!("{name}.kd"));
        let kernarg_segment_size = expect_u32(map, ".kernarg_segment_size")?;
        let group_segment_fixed_size = find_u32(map, ".group_segment_fixed_size")?.unwrap_or(0);
        let private_segment_fixed_size =
            find_u32(map, ".private_segment_fixed_size")?.unwrap_or(0);
        let wavefront_size = find_u32(map, ".wavefront_size")?.unwrap_or(32);
        let max_flat_workgroup_size = find_u32(map, ".max_flat_workgroup_size")?.unwrap_or(1024);

        let mut args = Vec::new();
        if let Some(args_val) = find_map_entry(map, ".args") {
            let args_arr = match args_val {
                MsgPackValue::Array(arr) => arr,
                _ => {
                    return Err(ElfError::InvalidMsgPack(
                        "kernel '.args' must be an array".to_owned(),
                    ))
                }
            };
            for arg_entry in args_arr {
                let arg_map = match arg_entry {
                    MsgPackValue::Map(m) => m,
                    _ => {
                        return Err(ElfError::InvalidMsgPack(
                            "kernel arg entry must be a map".to_owned(),
                        ))
                    }
                };
                let arg_name = find_str(arg_map, ".name").unwrap_or_default();
                let offset = expect_u32(arg_map, ".offset")?;
                let size = expect_u32(arg_map, ".size")?;
                let vk_str = expect_str(arg_map, ".value_kind")?;
                let value_kind = ValueKind::from_str(&vk_str)?;
                args.push(KernelArgMetadata {
                    name: arg_name,
                    offset,
                    size,
                    value_kind,
                });
            }
        }

        kernels.push(KernelMetadata {
            name,
            symbol,
            kernarg_segment_size,
            group_segment_fixed_size,
            private_segment_fixed_size,
            wavefront_size,
            max_flat_workgroup_size,
            args,
        });
    }

    Ok(kernels)
}

fn find_map_entry<'a>(map: &'a [(String, MsgPackValue)], key: &str) -> Option<&'a MsgPackValue> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn find_str(map: &[(String, MsgPackValue)], key: &str) -> Option<String> {
    match find_map_entry(map, key) {
        Some(MsgPackValue::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn expect_str(map: &[(String, MsgPackValue)], key: &str) -> Result<String, ElfError> {
    match find_map_entry(map, key) {
        Some(MsgPackValue::Str(s)) => Ok(s.clone()),
        _ => Err(ElfError::InvalidMsgPack(format!(
            "missing or non-string field '{key}'"
        ))),
    }
}

fn find_u32(map: &[(String, MsgPackValue)], key: &str) -> Result<Option<u32>, ElfError> {
    match find_map_entry(map, key) {
        None => Ok(None),
        Some(MsgPackValue::UInt(v)) => u32::try_from(*v)
            .map(Some)
            .map_err(|_| ElfError::InvalidMsgPack(format!("integer overflow in '{key}'"))),
        Some(MsgPackValue::Int(v)) if *v >= 0 => u32::try_from(*v)
            .map(Some)
            .map_err(|_| ElfError::InvalidMsgPack(format!("integer overflow in '{key}'"))),
        Some(_) => Err(ElfError::InvalidMsgPack(format!(
            "non-integer field '{key}'"
        ))),
    }
}

fn expect_u32(map: &[(String, MsgPackValue)], key: &str) -> Result<u32, ElfError> {
    find_u32(map, key)?
        .ok_or_else(|| ElfError::InvalidMsgPack(format!("missing integer field '{key}'")))
}

fn encode_value(val: &MsgPackValue, out: &mut Vec<u8>) {
    match val {
        MsgPackValue::UInt(v) => encode_uint(*v, out),
        MsgPackValue::Int(v) => {
            if *v >= 0 {
                encode_uint(*v as u64, out);
            } else if *v >= -32 {
                out.push(*v as i8 as u8);
            } else if *v >= i64::from(i8::MIN) {
                out.push(0xd0);
                out.push(*v as i8 as u8);
            } else if *v >= i64::from(i16::MIN) {
                out.push(0xd1);
                out.extend_from_slice(&(*v as i16).to_be_bytes());
            } else if *v >= i64::from(i32::MIN) {
                out.push(0xd2);
                out.extend_from_slice(&(*v as i32).to_be_bytes());
            } else {
                out.push(0xd3);
                out.extend_from_slice(&v.to_be_bytes());
            }
        }
        MsgPackValue::Str(s) => encode_str(s, out),
        MsgPackValue::Array(arr) => {
            let len = arr.len();
            if len <= 15 {
                out.push(0x90 | (len as u8));
            } else if len <= u16::MAX as usize {
                out.push(0xdc);
                out.extend_from_slice(&(len as u16).to_be_bytes());
            } else {
                out.push(0xdd);
                out.extend_from_slice(&(len as u32).to_be_bytes());
            }
            for item in arr {
                encode_value(item, out);
            }
        }
        MsgPackValue::Map(entries) => {
            let len = entries.len();
            if len <= 15 {
                out.push(0x80 | (len as u8));
            } else if len <= u16::MAX as usize {
                out.push(0xde);
                out.extend_from_slice(&(len as u16).to_be_bytes());
            } else {
                out.push(0xdf);
                out.extend_from_slice(&(len as u32).to_be_bytes());
            }
            for (k, v) in entries {
                encode_str(k, out);
                encode_value(v, out);
            }
        }
    }
}

fn encode_uint(v: u64, out: &mut Vec<u8>) {
    if v <= 0x7f {
        out.push(v as u8);
    } else if v <= u64::from(u8::MAX) {
        out.push(0xcc);
        out.push(v as u8);
    } else if v <= u64::from(u16::MAX) {
        out.push(0xcd);
        out.extend_from_slice(&(v as u16).to_be_bytes());
    } else if v <= u64::from(u32::MAX) {
        out.push(0xce);
        out.extend_from_slice(&(v as u32).to_be_bytes());
    } else {
        out.push(0xcf);
        out.extend_from_slice(&v.to_be_bytes());
    }
}

fn encode_str(s: &str, out: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 31 {
        out.push(0xa0 | (len as u8));
    } else if len <= u8::MAX as usize {
        out.push(0xd9);
        out.push(len as u8);
    } else if len <= u16::MAX as usize {
        out.push(0xda);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdb);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
    out.extend_from_slice(bytes);
}

struct MsgPackReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MsgPackReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, ElfError> {
        let b = *self.data.get(self.pos).ok_or(ElfError::Truncated)?;
        self.pos += 1;
        Ok(b)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], ElfError> {
        let end = self.pos.checked_add(len).ok_or(ElfError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(ElfError::Truncated);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u16_be(&mut self) -> Result<u16, ElfError> {
        let b = self.read_exact(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn read_u32_be(&mut self) -> Result<u32, ElfError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64_be(&mut self) -> Result<u64, ElfError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self, len: usize) -> Result<String, ElfError> {
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| ElfError::InvalidMsgPack("invalid UTF-8 string".to_owned()))
    }

    fn read_value(&mut self) -> Result<MsgPackValue, ElfError> {
        let tag = self.read_u8()?;
        match tag {
            0x00..=0x7f => Ok(MsgPackValue::UInt(u64::from(tag))),
            0x80..=0x8f => self.read_map(usize::from(tag & 0x0f)),
            0x90..=0x9f => self.read_array(usize::from(tag & 0x0f)),
            0xa0..=0xbf => self.read_string(usize::from(tag & 0x1f)).map(MsgPackValue::Str),
            0xcc => Ok(MsgPackValue::UInt(u64::from(self.read_u8()?))),
            0xcd => Ok(MsgPackValue::UInt(u64::from(self.read_u16_be()?))),
            0xce => Ok(MsgPackValue::UInt(u64::from(self.read_u32_be()?))),
            0xcf => Ok(MsgPackValue::UInt(self.read_u64_be()?)),
            0xd0 => {
                let v = self.read_u8()? as i8;
                Ok(MsgPackValue::Int(i64::from(v)))
            }
            0xd1 => {
                let b = self.read_exact(2)?;
                Ok(MsgPackValue::Int(i64::from(i16::from_be_bytes([b[0], b[1]]))))
            }
            0xd2 => {
                let b = self.read_exact(4)?;
                Ok(MsgPackValue::Int(i64::from(i32::from_be_bytes([
                    b[0], b[1], b[2], b[3],
                ]))))
            }
            0xd3 => {
                let b = self.read_exact(8)?;
                Ok(MsgPackValue::Int(i64::from_be_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ])))
            }
            0xd9 => {
                let len = usize::from(self.read_u8()?);
                self.read_string(len).map(MsgPackValue::Str)
            }
            0xda => {
                let len = usize::from(self.read_u16_be()?);
                self.read_string(len).map(MsgPackValue::Str)
            }
            0xdb => {
                let len = self.read_u32_be()? as usize;
                self.read_string(len).map(MsgPackValue::Str)
            }
            0xdc => {
                let len = usize::from(self.read_u16_be()?);
                self.read_array(len)
            }
            0xdd => {
                let len = self.read_u32_be()? as usize;
                self.read_array(len)
            }
            0xde => {
                let len = usize::from(self.read_u16_be()?);
                self.read_map(len)
            }
            0xdf => {
                let len = self.read_u32_be()? as usize;
                self.read_map(len)
            }
            0xe0..=0xff => Ok(MsgPackValue::Int(i64::from(tag as i8))),
            other => Err(ElfError::InvalidMsgPack(format!(
                "unsupported MsgPack tag 0x{other:02x}"
            ))),
        }
    }

    fn read_array(&mut self, len: usize) -> Result<MsgPackValue, ElfError> {
        let mut items = Vec::with_capacity(len.min(1024));
        for _ in 0..len {
            items.push(self.read_value()?);
        }
        Ok(MsgPackValue::Array(items))
    }

    fn read_map(&mut self, len: usize) -> Result<MsgPackValue, ElfError> {
        let mut entries = Vec::with_capacity(len.min(1024));
        for _ in 0..len {
            let key = match self.read_value()? {
                MsgPackValue::Str(s) => s,
                _ => {
                    return Err(ElfError::InvalidMsgPack(
                        "map key must be a UTF-8 string".to_owned(),
                    ))
                }
            };
            let val = self.read_value()?;
            entries.push((key, val));
        }
        Ok(MsgPackValue::Map(entries))
    }
}
