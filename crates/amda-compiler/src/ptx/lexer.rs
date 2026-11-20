//! Tokenizer for NVIDIA PTX assembly source.

use crate::error::CompileError;

/// Source span coordinates (1-based line and column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// Token kinds emitted by the PTX lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Directive(String),
    Register(String),
    Ident(String),
    Int(u64),
    Float(f64, String),
    StringLit(String),
    Comma,
    Semicolon,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LAngle,
    RAngle,
    Plus,
    Minus,
    At,
    Bang,
    Pipe,
    Equals,
    Eof,
}

/// Token accompanied by source location.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Tokenizes PTX source text into a stream of [`Token`]s ending with [`TokenKind::Eof`].
pub fn tokenize(src: &str) -> Result<Vec<Token>, CompileError> {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut col = 1usize;
    let mut tokens = Vec::new();

    while i < bytes.len() {
        let ch = bytes[i];

        // Whitespace
        if ch.is_ascii_whitespace() {
            if ch == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            i += 1;
            continue;
        }

        // Line comments: // ...
        if ch == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comments: /* ... */
        if ch == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start_line = line;
            let start_col = col;
            i += 2;
            col += 2;
            let mut closed = false;
            while i + 1 < bytes.len() {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    col += 2;
                    closed = true;
                    break;
                }
                if bytes[i] == b'\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
                i += 1;
            }
            if !closed {
                return Err(CompileError::LexError {
                    line: start_line,
                    col: start_col,
                    message: "unterminated block comment".to_owned(),
                });
            }
            continue;
        }

        let span = Span { line, col };

        // Single-character punctuation
        let punct = match ch {
            b',' => Some(TokenKind::Comma),
            b';' => Some(TokenKind::Semicolon),
            b':' => Some(TokenKind::Colon),
            b'(' => Some(TokenKind::LParen),
            b')' => Some(TokenKind::RParen),
            b'[' => Some(TokenKind::LBracket),
            b']' => Some(TokenKind::RBracket),
            b'{' => Some(TokenKind::LBrace),
            b'}' => Some(TokenKind::RBrace),
            b'<' => Some(TokenKind::LAngle),
            b'>' => Some(TokenKind::RAngle),
            b'+' => Some(TokenKind::Plus),
            b'-' => Some(TokenKind::Minus),
            b'@' => Some(TokenKind::At),
            b'!' => Some(TokenKind::Bang),
            b'|' => Some(TokenKind::Pipe),
            b'=' => Some(TokenKind::Equals),
            _ => None,
        };
        if let Some(kind) = punct {
            tokens.push(Token { kind, span });
            i += 1;
            col += 1;
            continue;
        }

        // String literal: "..."
        if ch == b'"' {
            i += 1;
            col += 1;
            let mut s = String::new();
            let mut closed = false;
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'"' {
                    i += 1;
                    col += 1;
                    closed = true;
                    break;
                }
                if c == b'\\' && i + 1 < bytes.len() {
                    let esc = bytes[i + 1];
                    s.push(match esc {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                    i += 2;
                    col += 2;
                    continue;
                }
                if c == b'\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
                s.push(c as char);
                i += 1;
            }
            if !closed {
                return Err(CompileError::LexError {
                    line: span.line,
                    col: span.col,
                    message: "unterminated string literal".to_owned(),
                });
            }
            tokens.push(Token {
                kind: TokenKind::StringLit(s),
                span,
            });
            continue;
        }

        // Directive: .name
        if ch == b'.' {
            let start = i;
            i += 1;
            col += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
                col += 1;
            }
            let text = &src[start..i];
            if text.len() == 1 {
                return Err(CompileError::LexError {
                    line: span.line,
                    col: span.col,
                    message: "unexpected standalone '.'".to_owned(),
                });
            }
            tokens.push(Token {
                kind: TokenKind::Directive(text.to_owned()),
                span,
            });
            continue;
        }

        // Register: %name or %tid.x
        if ch == b'%' {
            let start = i;
            i += 1;
            col += 1;
            while i < bytes.len() && (is_ident_continue(bytes[i]) || bytes[i] == b'.') {
                i += 1;
                col += 1;
            }
            let text = &src[start..i];
            if text.len() == 1 {
                return Err(CompileError::LexError {
                    line: span.line,
                    col: span.col,
                    message: "unexpected standalone '%'".to_owned(),
                });
            }
            tokens.push(Token {
                kind: TokenKind::Register(text.to_owned()),
                span,
            });
            continue;
        }

        // Numeric literals: 0x..., 0f..., 0d..., decimal integer or float
        if ch.is_ascii_digit() {
            let start = i;
            if ch == b'0' && i + 1 < bytes.len() {
                let next = bytes[i + 1];
                if next == b'x' || next == b'X' {
                    i += 2;
                    col += 2;
                    let hex_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                        i += 1;
                        col += 1;
                    }
                    if i == hex_start {
                        return Err(CompileError::LexError {
                            line: span.line,
                            col: span.col,
                            message: "expected hexadecimal digits after '0x'".to_owned(),
                        });
                    }
                    let val = u64::from_str_radix(&src[hex_start..i], 16).map_err(|e| {
                        CompileError::LexError {
                            line: span.line,
                            col: span.col,
                            message: format!("invalid hex integer: {e}"),
                        }
                    })?;
                    tokens.push(Token {
                        kind: TokenKind::Int(val),
                        span,
                    });
                    continue;
                }
                if next == b'f' || next == b'F' {
                    i += 2;
                    col += 2;
                    let hex_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                        i += 1;
                        col += 1;
                    }
                    let bits = u32::from_str_radix(&src[hex_start..i], 16).map_err(|e| {
                        CompileError::LexError {
                            line: span.line,
                            col: span.col,
                            message: format!("invalid PTX 0f float literal: {e}"),
                        }
                    })?;
                    let f = f64::from(f32::from_bits(bits));
                    tokens.push(Token {
                        kind: TokenKind::Float(f, src[start..i].to_owned()),
                        span,
                    });
                    continue;
                }
                if next == b'd' || next == b'D' {
                    i += 2;
                    col += 2;
                    let hex_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                        i += 1;
                        col += 1;
                    }
                    let bits = u64::from_str_radix(&src[hex_start..i], 16).map_err(|e| {
                        CompileError::LexError {
                            line: span.line,
                            col: span.col,
                            message: format!("invalid PTX 0d float literal: {e}"),
                        }
                    })?;
                    let f = f64::from_bits(bits);
                    tokens.push(Token {
                        kind: TokenKind::Float(f, src[start..i].to_owned()),
                        span,
                    });
                    continue;
                }
            }

            let mut is_float = false;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
                col += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                is_float = true;
                i += 1;
                col += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                is_float = true;
                i += 1;
                col += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                    col += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
            }

            let raw = &src[start..i];
            if is_float {
                let val: f64 = raw.parse().map_err(|e| CompileError::LexError {
                    line: span.line,
                    col: span.col,
                    message: format!("invalid float literal '{raw}': {e}"),
                })?;
                tokens.push(Token {
                    kind: TokenKind::Float(val, raw.to_owned()),
                    span,
                });
            } else {
                // Strip optional trailing 'U' integer suffix if present
                let val: u64 = raw.parse().map_err(|e| CompileError::LexError {
                    line: span.line,
                    col: span.col,
                    message: format!("invalid integer literal '{raw}': {e}"),
                })?;
                if i < bytes.len() && (bytes[i] == b'U' || bytes[i] == b'u') {
                    i += 1;
                    col += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Int(val),
                    span,
                });
            }
            continue;
        }

        // Identifier / opcode / label ($L_BB0, add.s32, vector_add)
        if is_ident_start(ch) {
            let start = i;
            i += 1;
            col += 1;
            while i < bytes.len() && (is_ident_continue(bytes[i]) || bytes[i] == b'.') {
                i += 1;
                col += 1;
            }
            let ident = src[start..i].to_owned();
            tokens.push(Token {
                kind: TokenKind::Ident(ident),
                span,
            });
            continue;
        }

        return Err(CompileError::LexError {
            line: span.line,
            col: span.col,
            message: format!("unexpected character '{}'", ch as char),
        });
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span { line, col },
    });
    Ok(tokens)
}

const fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'$'
}

const fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}
