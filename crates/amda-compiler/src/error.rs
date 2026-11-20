//! Error types for fatbin unpacking, PTX parsing, IR lowering, and code generation.

use core::fmt;

/// Errors returned by the AMDA compiler pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// Input fatbin header or wrapper magic did not match supported formats.
    InvalidFatbin(String),
    /// Fatbin entry uses an unsupported compression format (only uncompressed and LZ4 are supported).
    CompressedFatbinUnsupported,
    /// Fatbin container held no PTX entry.
    MissingPtxInFatbin,
    /// PTX tokenization failed at the given 1-based line and column.
    LexError {
        line: usize,
        col: usize,
        message: String,
    },
    /// PTX syntax parsing failed at the given 1-based line and column.
    ParseError {
        line: usize,
        col: usize,
        message: String,
    },
    /// Semantic lowering from PTX to AMDA IR failed.
    LoweringError(String),
    /// Serialized `.amda.ir` binary payload is malformed or incompatible.
    InvalidIrPayload(String),
    /// I/O error while reading or writing cached compilation artifacts.
    CacheIo(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFatbin(msg) => write!(f, "invalid fatbin container: {msg}"),
            Self::CompressedFatbinUnsupported => {
                write!(
                    f,
                    "unsupported compressed fatbin payload (only uncompressed and LZ4 entries are supported)"
                )
            }
            Self::MissingPtxInFatbin => write!(f, "fatbin container contained no PTX payload"),
            Self::LexError { line, col, message } => {
                write!(f, "PTX lex error at {line}:{col}: {message}")
            }
            Self::ParseError { line, col, message } => {
                write!(f, "PTX parse error at {line}:{col}: {message}")
            }
            Self::LoweringError(msg) => write!(f, "IR lowering error: {msg}"),
            Self::InvalidIrPayload(msg) => write!(f, "invalid AMDA IR payload: {msg}"),
            Self::CacheIo(msg) => write!(f, "artifact cache I/O error: {msg}"),
        }
    }
}

impl std::error::Error for CompileError {}
