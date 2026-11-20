//! External LLVM/LLD/llvm-objdump discovery and AMDGPU object/disassembly invocation.

use crate::error::CompileError;
use amda_elf::IsaTarget;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Discovered external LLVM binary paths (`opt`, `llc`, `ld.lld`, `llvm-objdump`, `llvm-mc`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlvmToolchain {
    pub opt: Option<PathBuf>,
    pub llc: Option<PathBuf>,
    pub lld: Option<PathBuf>,
    pub objdump: Option<PathBuf>,
    pub llvm_mc: Option<PathBuf>,
}

impl LlvmToolchain {
    /// Probes `AMDA_OPT`, `AMDA_LLC`, `AMDA_LLD`, `AMDA_LLVM_OBJDUMP`, `AMDA_LLVM_MC`,
    /// `llc` sibling binaries, `PATH`, and `/opt/rocm/llvm/bin`.
    pub fn discover() -> Self {
        let llc = find_tool(
            "AMDA_LLC",
            &[
                "llc", "llc-22", "llc-21", "llc-20", "llc-19", "llc-18", "llc-17",
            ],
        );
        let opt = find_opt_tool(llc.as_deref());
        Self {
            opt,
            llc,
            lld: find_tool(
                "AMDA_LLD",
                &[
                    "ld.lld",
                    "ld.lld-22",
                    "ld.lld-21",
                    "ld.lld-20",
                    "ld.lld-19",
                    "ld.lld-18",
                    "ld.lld-17",
                ],
            ),
            objdump: find_tool(
                "AMDA_LLVM_OBJDUMP",
                &[
                    "llvm-objdump",
                    "llvm-objdump-22",
                    "llvm-objdump-21",
                    "llvm-objdump-20",
                    "llvm-objdump-19",
                    "llvm-objdump-18",
                    "llvm-objdump-17",
                ],
            ),
            llvm_mc: find_tool(
                "AMDA_LLVM_MC",
                &[
                    "llvm-mc",
                    "llvm-mc-22",
                    "llvm-mc-21",
                    "llvm-mc-20",
                    "llvm-mc-19",
                    "llvm-mc-18",
                    "llvm-mc-17",
                ],
            ),
        }
    }

    /// Returns `true` when both `llc` and `ld.lld` are available to produce native ELF HSACOs.
    pub fn has_codegen_tools(&self) -> bool {
        self.llc.is_some() && self.lld.is_some()
    }
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn create(prefix: &str) -> Result<Self, CompileError> {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("amda_{prefix}_{}_{}", std::process::id(), id));
        fs::create_dir_all(&dir)
            .map_err(|e| CompileError::CacheIo(format!("create temp dir: {e}")))?;
        Ok(Self { path: dir })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Compiles `llvm_ir` to an AMDGPU HSACO ELF64 shared object via `llc` + `ld.lld`.
pub fn compile_llvm_ir_to_hsaco(llvm_ir: &str, target: IsaTarget) -> Result<Vec<u8>, CompileError> {
    let tools = LlvmToolchain::discover();
    let llc = tools.llc.ok_or_else(|| {
        CompileError::LoweringError("llc not found in AMDA_LLC or PATH".to_owned())
    })?;
    let lld = tools.lld.ok_or_else(|| {
        CompileError::LoweringError("ld.lld not found in AMDA_LLD or PATH".to_owned())
    })?;

    let tmp = TempDirGuard::create("llc")?;
    let ll_path = tmp.path().join("kernel.ll");
    let opt_bc_path = tmp.path().join("kernel.opt.bc");
    let obj_path = tmp.path().join("kernel.o");
    let hsaco_path = tmp.path().join("kernel.hsaco");

    fs::write(&ll_path, llvm_ir)
        .map_err(|e| CompileError::CacheIo(format!("write temp LLVM IR: {e}")))?;

    let llc_input = if let Some(opt) = tools.opt {
        let opt_out = Command::new(&opt)
            .args(["-O2"])
            .arg(&ll_path)
            .arg("-o")
            .arg(&opt_bc_path)
            .output()
            .map_err(|e| CompileError::LoweringError(format!("failed to execute opt: {e}")))?;
        if !opt_out.status.success() {
            return Err(CompileError::LoweringError(format!(
                "opt failed ({}): {}",
                opt_out.status,
                String::from_utf8_lossy(&opt_out.stderr)
            )));
        }
        opt_bc_path
    } else {
        ll_path
    };

    let mcpu = format!("-mcpu={}", target.as_str());
    let llc_out = Command::new(&llc)
        .args(["-march=amdgcn", &mcpu, "-filetype=obj", "-O2"])
        .arg(&llc_input)
        .arg("-o")
        .arg(&obj_path)
        .output()
        .map_err(|e| CompileError::LoweringError(format!("failed to execute llc: {e}")))?;
    if !llc_out.status.success() {
        return Err(CompileError::LoweringError(format!(
            "llc failed ({}): {}",
            llc_out.status,
            String::from_utf8_lossy(&llc_out.stderr)
        )));
    }

    let lld_out = Command::new(&lld)
        .args(["-shared"])
        .arg(&obj_path)
        .arg("-o")
        .arg(&hsaco_path)
        .output()
        .map_err(|e| CompileError::LoweringError(format!("failed to execute ld.lld: {e}")))?;
    if !lld_out.status.success() {
        return Err(CompileError::LoweringError(format!(
            "ld.lld failed ({}): {}",
            lld_out.status,
            String::from_utf8_lossy(&lld_out.stderr)
        )));
    }

    fs::read(&hsaco_path)
        .map_err(|e| CompileError::CacheIo(format!("read linked HSACO: {e}")))
}

/// Disassembles an HSACO binary with `llvm-objdump` if installed on the host.
/// Returns `Ok(None)` when `llvm-objdump` is not present.
pub fn disassemble_hsaco_with_objdump(
    hsaco: &[u8],
    target: IsaTarget,
) -> Result<Option<String>, CompileError> {
    let tools = LlvmToolchain::discover();
    let Some(objdump) = tools.objdump else {
        return Ok(None);
    };

    let tmp = TempDirGuard::create("objdump")?;
    let hsaco_path = tmp.path().join("kernel.hsaco");
    fs::write(&hsaco_path, hsaco)
        .map_err(|e| CompileError::CacheIo(format!("write temp HSACO: {e}")))?;

    let mcpu = format!("--mcpu={}", target.as_str());
    let out = Command::new(&objdump)
        .args(["-d", "--triple=amdgcn-amd-amdhsa", &mcpu])
        .arg(&hsaco_path)
        .output()
        .map_err(|e| CompileError::LoweringError(format!("failed to execute llvm-objdump: {e}")))?;
    if !out.status.success() {
        return Err(CompileError::LoweringError(format!(
            "llvm-objdump failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    if !text.contains("Disassembly of section .text:")
        || text.contains("<invalid>")
        || text.contains("invalid instruction")
        || text.contains(".long 0x")
        || text.contains("v_illegal")
    {
        return Err(CompileError::LoweringError(format!(
            "llvm-objdump reported invalid or undecodable machine instruction:\n{text}"
        )));
    }
    Ok(Some(text))
}

fn find_opt_tool(llc: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Some(p) = env::var_os("AMDA_OPT").filter(|v| !v.is_empty()) {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Some(llc_path) = llc {
        if let Some(fname) = llc_path.file_name().and_then(|s| s.to_str()) {
            if let Some(suffix) = fname.strip_prefix("llc") {
                let sibling_versioned = llc_path.with_file_name(format!("opt{suffix}"));
                if sibling_versioned.is_file() {
                    return Some(sibling_versioned);
                }
            }
        }
        let sibling_opt = llc_path.with_file_name("opt");
        if sibling_opt.is_file() {
            return Some(sibling_opt);
        }
    }
    find_tool(
        "AMDA_OPT",
        &[
            "opt", "opt-22", "opt-21", "opt-20", "opt-19", "opt-18", "opt-17",
        ],
    )
}

fn find_tool(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    if let Some(p) = env::var_os(env_var).filter(|v| !v.is_empty()) {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let mut dirs: Vec<PathBuf> = env::var_os("PATH")
        .map(|path_env| env::split_paths(&path_env).collect())
        .unwrap_or_default();
    let rocm_llvm_bin = PathBuf::from("/opt/rocm/llvm/bin");
    if !dirs.iter().any(|d| d == &rocm_llvm_bin) {
        dirs.push(rocm_llvm_bin);
    }
    for dir in dirs {
        for name in candidates {
            let full = dir.join(name);
            if full.is_file() {
                return Some(full);
            }
        }
    }
    None
}
