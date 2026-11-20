//! Pure-Rust SHA-256 hasher and persistent in-memory + optional disk artifact cache.

use crate::error::CompileError;
use crate::tier::CompileTier;
use amda_elf::IsaTarget;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

const COMPILER_CACHE_EPOCH: &str = "amda-compiler-v0.1.0";

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Computes the FIPS 180-4 SHA-256 digest of `input`.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(input.len() + 72);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let b = &chunk[i * 4..i * 4 + 4];
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = [0u8; 32];
    for (idx, word) in state.iter().enumerate() {
        out[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Formats a SHA-256 digest as lowercase hexadecimal.
pub fn to_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in digest {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Snapshot of artifact cache hits and misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// Thread-safe HSACO artifact cache backed by memory and optionally `AMDA_CACHE_DIR`.
pub struct ArtifactCache {
    mem: RwLock<HashMap<[u8; 32], Vec<u8>>>,
    disk_dir: Option<PathBuf>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ArtifactCache {
    /// Creates an artifact cache using `AMDA_CACHE_DIR` if set in the environment.
    pub fn from_env() -> Self {
        let disk_dir = std::env::var_os("AMDA_CACHE_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        Self::new(disk_dir)
    }

    /// Creates an artifact cache with an explicit optional disk directory.
    pub fn new(disk_dir: Option<PathBuf>) -> Self {
        Self {
            mem: RwLock::new(HashMap::new()),
            disk_dir,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Computes the deterministic cache key for `(compiler_version, target_isa, tier, source_bytes)`.
    pub fn compute_key(target: IsaTarget, tier: CompileTier, source: &[u8]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(COMPILER_CACHE_EPOCH.len() + 24 + source.len());
        buf.extend_from_slice(COMPILER_CACHE_EPOCH.as_bytes());
        buf.push(0);
        buf.extend_from_slice(target.as_str().as_bytes());
        buf.push(0);
        buf.push(tier.as_tag());
        buf.push(0);
        buf.extend_from_slice(source);
        sha256(&buf)
    }

    /// Looks up a cached HSACO artifact by key.
    pub fn get(&self, key: &[u8; 32]) -> Option<Vec<u8>> {
        if let Ok(guard) = self.mem.read() {
            if let Some(bytes) = guard.get(key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(bytes.clone());
            }
        }

        if let Some(dir) = &self.disk_dir {
            let path = dir.join(format!("{}.hsaco", to_hex(key)));
            if let Ok(bytes) = fs::read(&path) {
                if let Ok(mut guard) = self.mem.write() {
                    guard.insert(*key, bytes.clone());
                }
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(bytes);
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Stores a compiled HSACO artifact under `key` in memory and on disk (if configured).
    pub fn put(&self, key: [u8; 32], hsaco: &[u8]) -> Result<(), CompileError> {
        if let Ok(mut guard) = self.mem.write() {
            guard.insert(key, hsaco.to_vec());
        }
        if let Some(dir) = &self.disk_dir {
            fs::create_dir_all(dir)
                .map_err(|e| CompileError::CacheIo(format!("create cache dir: {e}")))?;
            let path = dir.join(format!("{}.hsaco", to_hex(&key)));
            fs::write(&path, hsaco)
                .map_err(|e| CompileError::CacheIo(format!("write {}: {e}", path.display())))?;
        }
        Ok(())
    }

    /// Returns cumulative hit/miss statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}
