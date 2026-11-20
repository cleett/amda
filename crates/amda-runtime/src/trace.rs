//! Structured runtime & ABI trace mode controlled by `AMDA_TRACE=1|true`.

use std::sync::OnceLock;

static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Parses an environment flag string (`"1"` or `"true"`, case-insensitive).
pub fn parse_trace_flag(raw: &str) -> bool {
    matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true")
}

/// Returns `true` when `AMDA_TRACE` is enabled (`1` or `true`).
pub fn trace_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("AMDA_TRACE")
            .ok()
            .map(|v| parse_trace_flag(&v))
            .unwrap_or(false)
    })
}

/// Emits a structured diagnostic line to `stderr` when `AMDA_TRACE` is enabled.
pub fn trace_log(category: &str, msg: &str) {
    if trace_enabled() {
        eprintln!("[amda:trace][{category}] {msg}");
    }
}

/// Formats a compact hexadecimal preview of the first up to `max_bytes` bytes.
pub fn format_hex_preview(bytes: &[u8], max_bytes: usize) -> String {
    let take = bytes.len().min(max_bytes);
    let mut s = String::with_capacity(take * 2 + 4);
    for &b in &bytes[..take] {
        use core::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    if bytes.len() > take {
        s.push_str("..");
    }
    s
}
