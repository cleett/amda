//! Dual-backend differential execution oracle comparing numerical agreement across GPUs/backends.

use crate::context::Runtime;
use crate::error::RuntimeError;
use amda_hal::DeviceBackend;
use std::sync::Arc;

/// Absolute and relative comparison tolerances for differential validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffTolerance {
    pub abs_tol: f32,
    pub rel_tol: f32,
}

impl Default for DiffTolerance {
    fn default() -> Self {
        Self {
            abs_tol: 1e-4,
            rel_tol: 1e-4,
        }
    }
}

/// Individual element mismatch recorded by [`compare_f32_slices`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffMismatch {
    pub index: usize,
    pub lhs: f32,
    pub rhs: f32,
    pub abs_err: f32,
    pub rel_err: f32,
}

/// Structured numerical comparison summary between two backend runs.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffReport {
    pub workload: String,
    pub lhs_backend: String,
    pub rhs_backend: String,
    pub elements: usize,
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    pub mismatches: Vec<DiffMismatch>,
    pub passed: bool,
}

/// Typed argument descriptor used by [`execute_ptx_f32_on_backend`].
#[derive(Debug, Clone, Copy)]
pub enum DiffKernelArg<'a> {
    InputBytes(&'a [u8]),
    InOutF32(&'a [f32]),
    OutputF32(usize),
    ScalarU32(u32),
    ScalarF32(f32),
}

/// Compares two `f32` slices element-by-element under `tol`, detecting NaNs, length drift, and error peaks.
pub fn compare_f32_slices(
    workload: &str,
    lhs_backend: &str,
    rhs_backend: &str,
    lhs: &[f32],
    rhs: &[f32],
    tol: DiffTolerance,
) -> DiffReport {
    let min_len = lhs.len().min(rhs.len());
    let max_len = lhs.len().max(rhs.len());
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut mismatches = Vec::new();

    for i in 0..min_len {
        let a = lhs[i];
        let b = rhs[i];
        let abs_err = if a.is_nan() || b.is_nan() {
            f32::INFINITY
        } else {
            (a - b).abs()
        };
        let scale = a.abs().max(b.abs()).max(1e-6);
        let rel_err = if abs_err.is_infinite() {
            f32::INFINITY
        } else {
            abs_err / scale
        };

        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }

        if !(abs_err <= tol.abs_tol || rel_err <= tol.rel_tol) && mismatches.len() < 16 {
            mismatches.push(DiffMismatch {
                index: i,
                lhs: a,
                rhs: b,
                abs_err,
                rel_err,
            });
        }
    }

    let length_match = lhs.len() == rhs.len();
    let passed = length_match && mismatches.is_empty();

    DiffReport {
        workload: workload.to_owned(),
        lhs_backend: lhs_backend.to_owned(),
        rhs_backend: rhs_backend.to_owned(),
        elements: max_len,
        max_abs_err,
        max_rel_err,
        mismatches,
        passed,
    }
}

/// Executes a single PTX kernel on `backend` (device 0) and downloads the designated `OutputF32` buffer.
#[allow(clippy::too_many_arguments)]
pub fn execute_ptx_f32_on_backend(
    backend: Arc<dyn DeviceBackend>,
    ptx: &str,
    kernel_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    args: &[DiffKernelArg<'_>],
) -> Result<Vec<f32>, RuntimeError> {
    let rt = Runtime::with_backend(backend);
    let ctx = rt.create_context(0)?;
    let module = ctx.load_module(ptx.as_bytes())?;
    let func = module.get_function(kernel_name)?;

    let mut allocs: Vec<u64> = Vec::new();
    let mut output_slot: Option<(u64, usize)> = None;
    let mut raw_arg_storage: Vec<[u8; 8]> = Vec::with_capacity(args.len());
    let mut raw_arg_lens: Vec<usize> = Vec::with_capacity(args.len());

    let res = (|| -> Result<Vec<f32>, RuntimeError> {
        for arg in args {
            match *arg {
                DiffKernelArg::InputBytes(bytes) => {
                    let alloc_len = bytes.len().max(4);
                    let ptr = ctx.alloc(alloc_len)?;
                    allocs.push(ptr);
                    if !bytes.is_empty() {
                        ctx.copy_htod(ptr, bytes)?;
                    }
                    raw_arg_storage.push(ptr.to_le_bytes());
                    raw_arg_lens.push(8);
                }
                DiffKernelArg::InOutF32(initial) => {
                    let bytes_len = initial.len().saturating_mul(4).max(4);
                    let ptr = ctx.alloc(bytes_len)?;
                    allocs.push(ptr);
                    if !initial.is_empty() {
                        let raw: Vec<u8> =
                            initial.iter().flat_map(|v| v.to_le_bytes()).collect();
                        ctx.copy_htod(ptr, &raw)?;
                    }
                    if output_slot.is_none() {
                        output_slot = Some((ptr, initial.len()));
                    }
                    raw_arg_storage.push(ptr.to_le_bytes());
                    raw_arg_lens.push(8);
                }
                DiffKernelArg::OutputF32(elems) => {
                    let bytes_len = elems.saturating_mul(4).max(4);
                    let ptr = ctx.alloc(bytes_len)?;
                    allocs.push(ptr);
                    let zeros = vec![0u8; bytes_len];
                    ctx.copy_htod(ptr, &zeros)?;
                    if output_slot.is_none() {
                        output_slot = Some((ptr, elems));
                    }
                    raw_arg_storage.push(ptr.to_le_bytes());
                    raw_arg_lens.push(8);
                }
                DiffKernelArg::ScalarU32(v) => {
                    let mut buf = [0u8; 8];
                    buf[..4].copy_from_slice(&v.to_le_bytes());
                    raw_arg_storage.push(buf);
                    raw_arg_lens.push(4);
                }
                DiffKernelArg::ScalarF32(v) => {
                    let mut buf = [0u8; 8];
                    buf[..4].copy_from_slice(&v.to_le_bytes());
                    raw_arg_storage.push(buf);
                    raw_arg_lens.push(4);
                }
            }
        }

        let slices: Vec<&[u8]> = raw_arg_storage
            .iter()
            .zip(raw_arg_lens.iter())
            .map(|(buf, &len)| &buf[..len])
            .collect();

        ctx.launch_kernel(&func, grid, block, shared_mem, None, &slices)?;
        ctx.synchronize()?;

        let (out_ptr, out_elems) = output_slot.ok_or_else(|| {
            RuntimeError::InvalidArgument("execute_ptx_f32_on_backend requires OutputF32 arg".to_owned())
        })?;
        let mut raw = vec![0u8; out_elems * 4];
        if !raw.is_empty() {
            ctx.copy_dtoh(&mut raw, out_ptr)?;
        }
        let mut out = Vec::with_capacity(out_elems);
        for chunk in raw.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    })();

    for ptr in allocs {
        let _ = ctx.free(ptr);
    }
    let _ = ctx.unload_module(module);
    res
}

/// Runs identical PTX and inputs on two backends and returns a numerical [`DiffReport`].
#[allow(clippy::too_many_arguments)]
pub fn run_differential_ptx_f32(
    workload: &str,
    lhs_backend: Arc<dyn DeviceBackend>,
    rhs_backend: Arc<dyn DeviceBackend>,
    ptx: &str,
    kernel_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    args: &[DiffKernelArg<'_>],
    tol: DiffTolerance,
) -> Result<DiffReport, RuntimeError> {
    let lhs_name = format!(
        "{}:{}",
        lhs_backend.devices().first().map(|d| d.isa.as_str()).unwrap_or("unknown"),
        lhs_backend.devices().first().map(|d| d.backend_name).unwrap_or("unknown")
    );
    let rhs_name = format!(
        "{}:{}",
        rhs_backend.devices().first().map(|d| d.isa.as_str()).unwrap_or("unknown"),
        rhs_backend.devices().first().map(|d| d.backend_name).unwrap_or("unknown")
    );

    let lhs_out =
        execute_ptx_f32_on_backend(lhs_backend, ptx, kernel_name, grid, block, shared_mem, args)?;
    let rhs_out =
        execute_ptx_f32_on_backend(rhs_backend, ptx, kernel_name, grid, block, shared_mem, args)?;

    Ok(compare_f32_slices(
        workload, &lhs_name, &rhs_name, &lhs_out, &rhs_out, tol,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_compare_detects_pass_and_mismatch() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.00001f32, 2.0, 3.0, 4.0];
        let pass_rep = compare_f32_slices(
            "unit_pass",
            "gfx1100",
            "gfx942",
            &a,
            &b,
            DiffTolerance::default(),
        );
        assert!(pass_rep.passed);
        assert!(pass_rep.mismatches.is_empty());
        assert!(pass_rep.max_abs_err < 1e-4);

        let c = [1.0f32, 2.5, 3.0, 4.0];
        let fail_rep = compare_f32_slices(
            "unit_fail",
            "gfx1100",
            "gfx942",
            &a,
            &c,
            DiffTolerance::default(),
        );
        assert!(!fail_rep.passed);
        assert_eq!(fail_rep.mismatches.len(), 1);
        assert_eq!(fail_rep.mismatches[0].index, 1);
        assert!((fail_rep.mismatches[0].abs_err - 0.5).abs() < 1e-6);
    }
}
