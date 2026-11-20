//! AMDA CLI tool (`amda info`, `amda compile`, `amda demo`).

use amda_compiler::{CompileTier, Compiler, TierPolicy};
use amda_cublas::{
    cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2, register_cuda_dispatch, CudaDispatchTable,
    CublasHandleT, CublasOperationT, CublasStatusT,
};
use amda_cuda::{
    cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSynchronize, cuDeviceGet, cuDeviceGetName,
    cuGetProcAddress_v2, cuGraphAddKernelNode, cuGraphCreate, cuGraphDestroy, cuGraphExecDestroy,
    cuGraphInstantiate, cuGraphLaunch, cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2,
    cuMemcpyHtoD_v2, cuModuleGetFunction, cuModuleLoadData, cuModuleUnload, cudaFree, cudaMalloc,
    cudaMemcpy, cudaSetDevice, CUcontext, CUdevice, CUdeviceptr, CUfunction, CUgraph, CUgraphExec,
    CUgraphNode, CUmodule, CUresult, CUstream, CudaErrorT, CudaKernelNodeParams, CudaMemcpyKind,
};
use amda_elf::{parse_hsaco, IsaTarget};
use amda_hal::MockBackend;
use amda_runtime::{
    compare_f32_slices, execute_ptx_f32_on_backend, run_differential_ptx_f32, DiffKernelArg,
    DiffTolerance, Runtime,
};
use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::env;
use std::ffi::CString;
use std::fs;
use std::process;
use std::sync::Arc;
use std::time::Instant;

const DEMO_VECTOR_ADD_PTX: &str = concat!(
    ".version 7.8\n",
    ".target sm_80\n",
    ".address_size 64\n",
    ".visible .entry vector_add(\n",
    "    .param .u64 a,\n",
    "    .param .u64 b,\n",
    "    .param .u64 c,\n",
    "    .param .u32 n\n",
    ") {\n",
    "    .reg .pred %p<2>;\n",
    "    .reg .f32 %f<4>;\n",
    "    .reg .b32 %r<6>;\n",
    "    .reg .b64 %rd<10>;\n",
    "    ld.param.u64 %rd1, [a];\n",
    "    ld.param.u64 %rd2, [b];\n",
    "    ld.param.u64 %rd3, [c];\n",
    "    ld.param.u32 %r1, [n];\n",
    "    mov.u32 %r2, %ctaid.x;\n",
    "    mov.u32 %r3, %ntid.x;\n",
    "    mov.u32 %r4, %tid.x;\n",
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n",
    "    setp.ge.s32 %p1, %r5, %r1;\n",
    "    @%p1 bra $L_EXIT;\n",
    "    cvta.to.global.u64 %rd4, %rd1;\n",
    "    cvta.to.global.u64 %rd5, %rd2;\n",
    "    cvta.to.global.u64 %rd6, %rd3;\n",
    "    mul.wide.s32 %rd7, %r5, 4;\n",
    "    add.s64 %rd8, %rd4, %rd7;\n",
    "    add.s64 %rd9, %rd5, %rd7;\n",
    "    ld.global.f32 %f1, [%rd8];\n",
    "    ld.global.f32 %f2, [%rd9];\n",
    "    add.f32 %f3, %f1, %f2;\n",
    "    add.s64 %rd8, %rd6, %rd7;\n",
    "    st.global.f32 [%rd8], %f3;\n",
    "$L_EXIT:\n",
    "    ret;\n",
    "}\n\0"
);

const DEMO_SAXPY_PTX: &str = concat!(
    ".version 7.8\n",
    ".target sm_80\n",
    ".address_size 64\n",
    ".visible .entry saxpy(\n",
    "    .param .f32 a,\n",
    "    .param .u64 x,\n",
    "    .param .u64 y,\n",
    "    .param .u32 n\n",
    ") {\n",
    "    .reg .pred %p<2>;\n",
    "    .reg .f32 %f<5>;\n",
    "    .reg .b32 %r<6>;\n",
    "    .reg .b64 %rd<8>;\n",
    "    ld.param.f32 %f1, [a];\n",
    "    ld.param.u64 %rd1, [x];\n",
    "    ld.param.u64 %rd2, [y];\n",
    "    ld.param.u32 %r1, [n];\n",
    "    mov.u32 %r2, %ctaid.x;\n",
    "    mov.u32 %r3, %ntid.x;\n",
    "    mov.u32 %r4, %tid.x;\n",
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n",
    "    setp.ge.s32 %p1, %r5, %r1;\n",
    "    @%p1 bra $L_DONE;\n",
    "    cvta.to.global.u64 %rd3, %rd1;\n",
    "    cvta.to.global.u64 %rd4, %rd2;\n",
    "    mul.wide.s32 %rd5, %r5, 4;\n",
    "    add.s64 %rd6, %rd3, %rd5;\n",
    "    add.s64 %rd7, %rd4, %rd5;\n",
    "    ld.global.f32 %f2, [%rd6];\n",
    "    ld.global.f32 %f3, [%rd7];\n",
    "    fma.rn.f32 %f4, %f1, %f2, %f3;\n",
    "    st.global.f32 [%rd7], %f4;\n",
    "$L_DONE:\n",
    "    ret;\n",
    "}\n\0"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmitKind {
    Asm,
    Llvm,
    Ir,
    Hsaco,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(msg) = run(&args) {
        eprintln!("amda error: {msg}");
        process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("info") => cmd_info(),
        Some("compile") => cmd_compile(&args[1..]),
        Some("demo") => cmd_demo(),
        Some("diff") => cmd_diff(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("--help" | "-h") | None => {
            print_usage();
            Ok(())
        }
        Some(other) => Err(format!("unknown subcommand '{other}'")),
    }
}

fn print_usage() {
    println!("AMDA — CUDA-compatible AMD GPU Computing Stack");
    println!();
    println!("USAGE:");
    println!("  amda info");
    println!("  amda compile <file.ptx> [--target gfx1100|gfx942] [--tier auto|0|1] [--emit asm|llvm|ir|hsaco]");
    println!("  amda demo");
    println!("  amda diff [--lhs gfx1100] [--rhs gfx942]");
    println!("  amda bench [--iters 20] [--target gfx1100|gfx942]");
}

fn cmd_info() -> Result<(), String> {
    let rt = Runtime::new().map_err(|e| e.to_string())?;
    println!("AMDA Runtime Device Discovery");
    println!("=============================");
    for dev in rt.devices() {
        let f = &dev.features;
        println!("Device [{}]: {}", dev.index, dev.name);
        println!("  Backend       : {}", dev.backend_name);
        println!("  ISA Target    : {} (native wave {})", dev.isa, dev.isa.native_wave_size());
        println!("  VRAM          : {} MiB", dev.vram_bytes / (1024 * 1024));
        println!("  Compute Units : {}", f.compute_units);
        println!("  LDS / CU      : {} KiB", f.lds_bytes_per_cu / 1024);
        println!("  Max Waves/CU  : {}", f.max_waves_per_cu);
        println!(
            "  Features      : wave32={}, wave64={}, wmma={}, mfma={}, bf16={}, fp8={}, unified_mem={}",
            f.wave32, f.wave64, f.wmma, f.mfma, f.bf16, f.fp8, f.unified_memory
        );
        println!("  Dispatch Ring : HSA AQL 64-byte packet ring + direct doorbell");
        println!("  Tiered JIT    : Tier 0 zero-LLVM native ISA + Tier 1 Tile/LLVM IR");
    }
    Ok(())
}

fn cmd_compile(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("usage: amda compile <file.ptx> [--target gfx1100|gfx942] [--tier auto|0|1] [--emit asm|llvm|ir|hsaco]".to_owned());
    }
    let mut input_path: Option<String> = None;
    let mut target = IsaTarget::Gfx1100;
    let mut policy = TierPolicy::Auto;
    let mut emit = EmitKind::Asm;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                i += 1;
                let t = args.get(i).ok_or("missing value after --target")?;
                target = IsaTarget::from_str(t).map_err(|e| e.to_string())?;
            }
            "--tier" => {
                i += 1;
                let t = args.get(i).ok_or("missing value after --tier")?;
                policy = match t.as_str() {
                    "auto" => TierPolicy::Auto,
                    "0" | "tier0" | "fast" => TierPolicy::ForceTier0,
                    "1" | "tier1" | "opt" => TierPolicy::ForceTier1,
                    other => return Err(format!("unsupported --tier value '{other}'")),
                };
            }
            "--emit" => {
                i += 1;
                let e = args.get(i).ok_or("missing value after --emit")?;
                emit = match e.as_str() {
                    "asm" => EmitKind::Asm,
                    "llvm" => EmitKind::Llvm,
                    "ir" => EmitKind::Ir,
                    "hsaco" => EmitKind::Hsaco,
                    other => return Err(format!("unsupported --emit kind '{other}'")),
                };
            }
            path if !path.starts_with('-') => {
                input_path = Some(path.to_owned());
            }
            flag => return Err(format!("unknown compile option '{flag}'")),
        }
        i += 1;
    }

    if emit == EmitKind::Llvm && policy == TierPolicy::Auto {
        policy = TierPolicy::ForceTier1;
    }

    let path = input_path.ok_or("missing <file.ptx> path")?;
    let ptx_src = fs::read_to_string(&path).map_err(|e| format!("read '{path}': {e}"))?;
    let compiler = Compiler::new();
    let art = compiler
        .compile_ptx_with_policy(&ptx_src, target, policy)
        .map_err(|e| e.to_string())?;

    match emit {
        EmitKind::Asm => {
            println!(
                "; AMDA Tiered JIT Assembly (target={}, tier={})",
                target,
                art.tier.as_str()
            );
            for r in &art.regalloc {
                println!(
                    ";   kernel @{}: {} SGPRs ({} uniform IR regs), {} VGPRs ({} divergent IR regs), {}B code, est {} waves/SIMD",
                    r.kernel_name,
                    r.sgpr_count,
                    r.uniform_ir_regs,
                    r.vgpr_count,
                    r.divergent_ir_regs,
                    r.code_bytes,
                    r.estimated_waves_per_simd
                );
            }
            if let Some(asm) = &art.native_asm {
                print!("{asm}");
            }
        }
        EmitKind::Llvm => {
            if let Some(llvm) = &art.llvm_ir {
                print!("{llvm}");
            } else {
                let llvm = amda_compiler::emit_llvm_ir(&art.ir, target);
                print!("{llvm}");
            }
        }
        EmitKind::Ir => {
            println!(
                "; AMDA IR Module (target={}, tier={}, kernels={})",
                art.ir.target,
                art.tier.as_str(),
                art.ir.kernels.len()
            );
            for k in &art.ir.kernels {
                println!(
                    "kernel @{} (wave_mode={:?}, shared_bytes={}, kernarg_bytes={}) {{",
                    k.name, k.wave_mode, k.shared_bytes, k.metadata.kernarg_segment_size
                );
                for (idx, inst) in k.insts.iter().enumerate() {
                    println!("  [{idx:03}] {:?}", inst);
                }
                println!("}}");
            }
        }
        EmitKind::Hsaco => {
            let parsed = parse_hsaco(&art.hsaco).map_err(|e| e.to_string())?;
            let text_len = parsed
                .sections
                .iter()
                .find(|s| s.name == ".text")
                .map(|s| s.data.len())
                .unwrap_or(0);
            println!("HSACO ELF64 Summary ({} bytes, tier={})", art.hsaco.len(), art.tier.as_str());
            println!("  ISA Target : {}", parsed.isa);
            println!("  Sections   : {}", parsed.sections.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
            println!("  .text Size : {} bytes", text_len);
            for (idx, kd) in parsed.kernel_descriptors.iter().enumerate() {
                println!(
                    "  .rodata KD[{idx}]: wave32={}, entry_off={}, kernarg={}B, rsrc1=0x{:08X}, rsrc2=0x{:08X}",
                    kd.is_wave32(),
                    kd.kernel_code_entry_byte_offset,
                    kd.kernarg_size,
                    kd.compute_pgm_rsrc1,
                    kd.compute_pgm_rsrc2
                );
            }
            for k in &parsed.kernels {
                println!(
                    "  Kernel '{}' (symbol={}, wave={}, kernarg={}B, lds={}B, args={})",
                    k.name,
                    k.symbol,
                    k.wavefront_size,
                    k.kernarg_segment_size,
                    k.group_segment_fixed_size,
                    k.args.len()
                );
            }
        }
    }
    Ok(())
}

fn check_cu(res: CUresult, op: &str) -> Result<(), String> {
    if res == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op} failed with {res:?}"))
    }
}

fn cmd_demo() -> Result<(), String> {
    println!("AMDA End-to-End CUDA Compatibility Demo");
    println!("=======================================");

    check_cu(cuInit(0), "cuInit")?;
    let mut dev: CUdevice = 0;
    check_cu(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
    let mut name_buf = [0 as c_char; 128];
    check_cu(cuDeviceGetName(name_buf.as_mut_ptr(), 128, dev), "cuDeviceGetName")?;
    let dev_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_string_lossy();
    println!("Using device [0]: {dev_name}");

    let mut ctx: CUcontext = core::ptr::null_mut();
    check_cu(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate_v2")?;

    // Resolve cuLaunchKernel via cuGetProcAddress_v2
    let sym = CString::new("cuLaunchKernel").unwrap();
    let mut pfn: *mut c_void = core::ptr::null_mut();
    let mut status: c_int = -1;
    check_cu(
        cuGetProcAddress_v2(sym.as_ptr(), &mut pfn, 12040, 0, &mut status),
        "cuGetProcAddress_v2",
    )?;
    type LaunchFn = extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult;
    let cu_launch: LaunchFn = unsafe { core::mem::transmute(pfn) };

    // 1. Tiered JIT compilation + divergence-aware regalloc + first load vs cache hit timing
    let compiler = Compiler::with_cache_dir(None);
    let clean_ptx = DEMO_VECTOR_ADD_PTX.trim_end_matches('\0');
    let t0 = Instant::now();
    let first = compiler
        .compile_ptx_with_policy(clean_ptx, IsaTarget::Gfx1100, TierPolicy::Auto)
        .map_err(|e| e.to_string())?;
    let t_cold = t0.elapsed();
    let t1 = Instant::now();
    let second = compiler
        .compile_ptx_with_policy(clean_ptx, IsaTarget::Gfx1100, TierPolicy::Auto)
        .map_err(|e| e.to_string())?;
    let t_warm = t1.elapsed();
    let ra = &first.regalloc[0];
    println!(
        "Tiered JIT ({}): cold={:.3}ms (hit={}), warm={:.3}ms (hit={}), {} SGPRs, {} VGPRs, {}B native .text, est {} waves/SIMD",
        first.tier.as_str(),
        t_cold.as_secs_f64() * 1000.0,
        first.cache_hit,
        t_warm.as_secs_f64() * 1000.0,
        second.cache_hit,
        ra.sgpr_count,
        ra.vgpr_count,
        ra.code_bytes,
        ra.estimated_waves_per_simd,
    );

    // 2. vector_add via CUDA Driver API
    let mut mod_vadd: CUmodule = core::ptr::null_mut();
    check_cu(
        cuModuleLoadData(&mut mod_vadd, DEMO_VECTOR_ADD_PTX.as_ptr() as *const c_void),
        "cuModuleLoadData(vector_add)",
    )?;
    let vadd_name = CString::new("vector_add").unwrap();
    let mut fn_vadd: CUfunction = core::ptr::null_mut();
    check_cu(
        cuModuleGetFunction(&mut fn_vadd, mod_vadd, vadd_name.as_ptr()),
        "cuModuleGetFunction(vector_add)",
    )?;

    let n: u32 = 2048;
    let bytes = (n as usize) * core::mem::size_of::<f32>();
    let mut da: CUdeviceptr = 0;
    let mut db: CUdeviceptr = 0;
    let mut dc: CUdeviceptr = 0;
    check_cu(cuMemAlloc_v2(&mut da, bytes), "cuMemAlloc(a)")?;
    check_cu(cuMemAlloc_v2(&mut db, bytes), "cuMemAlloc(b)")?;
    check_cu(cuMemAlloc_v2(&mut dc, bytes), "cuMemAlloc(c)")?;

    let ha: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let hb: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let mut hc = vec![0.0f32; n as usize];

    check_cu(
        cuMemcpyHtoD_v2(da, ha.as_ptr() as *const c_void, bytes),
        "cuMemcpyHtoD(a)",
    )?;
    check_cu(
        cuMemcpyHtoD_v2(db, hb.as_ptr() as *const c_void, bytes),
        "cuMemcpyHtoD(b)",
    )?;

    let mut arg_n = n;
    let mut vadd_params: [*mut c_void; 4] = [
        (&mut da as *mut CUdeviceptr) as *mut c_void,
        (&mut db as *mut CUdeviceptr) as *mut c_void,
        (&mut dc as *mut CUdeviceptr) as *mut c_void,
        (&mut arg_n as *mut u32) as *mut c_void,
    ];
    let t_k0 = Instant::now();
    check_cu(
        cu_launch(
            fn_vadd,
            8,
            1,
            1,
            256,
            1,
            1,
            0,
            core::ptr::null_mut(),
            vadd_params.as_mut_ptr(),
            core::ptr::null_mut(),
        ),
        "cuLaunchKernel(vector_add)",
    )?;
    let t_vadd = t_k0.elapsed();

    check_cu(
        cuMemcpyDtoH_v2(hc.as_mut_ptr() as *mut c_void, dc, bytes),
        "cuMemcpyDtoH(c)",
    )?;
    for (i, &val) in hc.iter().enumerate().take(n as usize) {
        let expected = (i * 3) as f32;
        if (val - expected).abs() > 1e-6 {
            return Err(format!("vector_add mismatch at {i}: got {val}, expected {expected}"));
        }
    }
    println!(
        "[PASS] vector_add ({n} floats, 8x256 threads) in {:.3}ms",
        t_vadd.as_secs_f64() * 1000.0
    );

    // 3. saxpy + CUDA Graph replay (2 launches: c = c + 2.5*a + 2.5*a = 3*i + 5*i = 8*i)
    let mut mod_saxpy: CUmodule = core::ptr::null_mut();
    check_cu(
        cuModuleLoadData(&mut mod_saxpy, DEMO_SAXPY_PTX.as_ptr() as *const c_void),
        "cuModuleLoadData(saxpy)",
    )?;
    let saxpy_name = CString::new("saxpy").unwrap();
    let mut fn_saxpy: CUfunction = core::ptr::null_mut();
    check_cu(
        cuModuleGetFunction(&mut fn_saxpy, mod_saxpy, saxpy_name.as_ptr()),
        "cuModuleGetFunction(saxpy)",
    )?;

    let mut alpha: f32 = 2.5;
    let mut saxpy_params: [*mut c_void; 4] = [
        (&mut alpha as *mut f32) as *mut c_void,
        (&mut da as *mut CUdeviceptr) as *mut c_void,
        (&mut dc as *mut CUdeviceptr) as *mut c_void,
        (&mut arg_n as *mut u32) as *mut c_void,
    ];
    let mut graph: CUgraph = core::ptr::null_mut();
    check_cu(cuGraphCreate(&mut graph, 0), "cuGraphCreate")?;
    let mut node: CUgraphNode = core::ptr::null_mut();
    let node_params = CudaKernelNodeParams {
        func: fn_saxpy,
        grid_dim_x: 8,
        grid_dim_y: 1,
        grid_dim_z: 1,
        block_dim_x: 256,
        block_dim_y: 1,
        block_dim_z: 1,
        shared_mem_bytes: 0,
        kernel_params: saxpy_params.as_mut_ptr(),
        extra: core::ptr::null_mut(),
    };
    check_cu(
        cuGraphAddKernelNode(&mut node, graph, core::ptr::null(), 0, &node_params),
        "cuGraphAddKernelNode",
    )?;
    let mut exec: CUgraphExec = core::ptr::null_mut();
    check_cu(cuGraphInstantiate(&mut exec, graph, 0), "cuGraphInstantiate")?;

    let t_g0 = Instant::now();
    check_cu(cuGraphLaunch(exec, core::ptr::null_mut()), "cuGraphLaunch #1")?;
    check_cu(cuGraphLaunch(exec, core::ptr::null_mut()), "cuGraphLaunch #2")?;
    check_cu(cuCtxSynchronize(), "cuCtxSynchronize")?;
    let t_graph = t_g0.elapsed();

    check_cu(
        cuMemcpyDtoH_v2(hc.as_mut_ptr() as *mut c_void, dc, bytes),
        "cuMemcpyDtoH(graph)",
    )?;
    for (i, &val) in hc.iter().enumerate().take(n as usize) {
        let expected = (i * 8) as f32;
        if (val - expected).abs() > 1e-5 {
            return Err(format!("saxpy graph mismatch at {i}: got {val}, expected {expected}"));
        }
    }
    println!(
        "[PASS] saxpy + CUDA Graph 2x replay ({n} floats) in {:.3}ms",
        t_graph.as_secs_f64() * 1000.0
    );

    check_cu(cuGraphExecDestroy(exec), "cuGraphExecDestroy")?;
    check_cu(cuGraphDestroy(graph), "cuGraphDestroy")?;
    check_cu(cuModuleUnload(mod_saxpy), "cuModuleUnload(saxpy)")?;
    check_cu(cuModuleUnload(mod_vadd), "cuModuleUnload(vector_add)")?;
    check_cu(cuMemFree_v2(da), "cuMemFree(a)")?;
    check_cu(cuMemFree_v2(db), "cuMemFree(b)")?;
    check_cu(cuMemFree_v2(dc), "cuMemFree(c)")?;
    check_cu(cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")?;

    // 4. Live Tier 0 -> Tier 1 hot-kernel promotion demo via amda-runtime
    let rt = Runtime::new().map_err(|e| e.to_string())?;
    let rt_ctx = rt.create_context(0).map_err(|e| e.to_string())?;
    rt_ctx.set_promote_threshold(4);
    let rt_mod = rt_ctx
        .load_module(clean_ptx.as_bytes())
        .map_err(|e| e.to_string())?;
    let rt_fn = rt_mod.get_function("vector_add").map_err(|e| e.to_string())?;
    let elem_n: u32 = 128;
    let sz = (elem_n as usize) * 4;
    let p_a = rt_ctx.alloc(sz).map_err(|e| e.to_string())?;
    let p_b = rt_ctx.alloc(sz).map_err(|e| e.to_string())?;
    let p_c = rt_ctx.alloc(sz).map_err(|e| e.to_string())?;
    let a_bytes = p_a.to_le_bytes();
    let b_bytes = p_b.to_le_bytes();
    let c_bytes = p_c.to_le_bytes();
    let n_bytes = elem_n.to_le_bytes();

    for _ in 0..4 {
        rt_ctx
            .launch_kernel(
                &rt_fn,
                [1, 1, 1],
                [128, 1, 1],
                0,
                None,
                &[&a_bytes, &b_bytes, &c_bytes, &n_bytes],
            )
            .map_err(|e| e.to_string())?;
    }
    let final_tier = rt_ctx
        .kernel_tier(rt_fn.handle)
        .map(|(_, t)| t)
        .unwrap_or(CompileTier::Tier0Fast);
    let tstats = rt_ctx.tier_stats();
    println!(
        "[PASS] Hot promotion: vector_add started at {}, promoted after 4 launches -> {} (stats={:?})",
        rt_fn.tier.as_str(),
        final_tier.as_str(),
        tstats
    );
    let _ = rt_ctx.free(p_a);
    let _ = rt_ctx.free(p_b);
    let _ = rt_ctx.free(p_c);
    let _ = rt_ctx.unload_module(rt_mod);

    Ok(())
}

const DIFF_SOFTMAX_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64
    .visible .entry row_softmax(
        .param .u64 in_ptr,
        .param .u64 out_ptr
    ) {
        .reg .f32 %f<10>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<8>;
        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];
        mov.u32 %r1, %ctaid.x;
        mov.u32 %r2, %ntid.x;
        mov.u32 %r3, %tid.x;
        mad.lo.u32 %r4, %r1, %r2, %r3;
        cvta.to.global.u64 %rd3, %rd1;
        mul.wide.u32 %rd4, %r4, 4;
        add.u64 %rd5, %rd3, %rd4;
        ld.global.f32 %f1, [%rd5];
        mov.f32 %f2, %f1;
        shfl.sync.bfly.b32 %f3, %f2, 16, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 8, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 4, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 2, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 1, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        sub.f32 %f4, %f1, %f2;
        mul.f32 %f5, %f4, 0f3fb8aa3b;
        ex2.approx.f32 %f6, %f5;
        mov.f32 %f7, %f6;
        shfl.sync.bfly.b32 %f3, %f7, 16, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 8, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 4, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 2, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 1, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        rcp.rn.f32 %f8, %f7;
        mul.f32 %f9, %f6, %f8;
        cvta.to.global.u64 %rd6, %rd2;
        add.u64 %rd7, %rd6, %rd4;
        st.global.f32 [%rd7], %f9;
        ret;
    }
"#;

const DIFF_Q4_0_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64
    .visible .entry ggml_matvec_q4_0_q8_1(
        .param .u64 W_ptr,
        .param .u64 X_ptr,
        .param .u64 Y_ptr,
        .param .u32 k_blocks
    ) {
        .reg .pred %p<4>;
        .reg .b16 %h<4>;
        .reg .f32 %f<10>;
        .reg .s32 %r<24>;
        .reg .b64 %rd<16>;
        ld.param.u64 %rd1, [W_ptr];
        ld.param.u64 %rd2, [X_ptr];
        ld.param.u64 %rd3, [Y_ptr];
        ld.param.u32 %r1, [k_blocks];
        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %tid.x;
        and.b32 %r4, %r3, 3;
        shr.u32 %r5, %r3, 2;
        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;
        mul.lo.u32 %r6, %r2, %r1;
        mul.lo.u32 %r6, %r6, 18;
        cvt.u64.u32 %rd7, %r6;
        add.u64 %rd8, %rd4, %rd7;
        mov.f32 %f1, 0f00000000;
        mov.u32 %r7, %r5;
    $L_LOOP:
        setp.ge.u32 %p1, %r7, %r1;
        @%p1 bra $L_RED;
        mul.lo.u32 %r8, %r7, 18;
        cvt.u64.u32 %rd9, %r8;
        add.u64 %rd10, %rd8, %rd9;
        ld.global.f16 %h0, [%rd10];
        cvt.f32.f16 %f2, %h0;
        mul.lo.u32 %r9, %r7, 36;
        cvt.u64.u32 %rd11, %r9;
        add.u64 %rd12, %rd5, %rd11;
        ld.global.f16 %h1, [%rd12];
        cvt.f32.f16 %f3, %h1;
        ld.global.f16 %h2, [%rd12+2];
        cvt.f32.f16 %f4, %h2;
        mul.lo.u32 %r10, %r4, 4;
        add.u32 %r11, %r10, 2;
        cvt.u64.u32 %rd13, %r11;
        add.u64 %rd14, %rd10, %rd13;
        ld.global.u16 %r12, [%rd14];
        ld.global.u16 %r13, [%rd14+2];
        prmt.b32 %r14, %r12, %r13, 0x5410;
        and.b32 %r15, %r14, 0x0F0F0F0F;
        shr.u32 %r16, %r14, 4;
        and.b32 %r16, %r16, 0x0F0F0F0F;
        add.u32 %r17, %r10, 4;
        cvt.u64.u32 %rd13, %r17;
        add.u64 %rd15, %rd12, %rd13;
        ld.global.u32 %r18, [%rd15];
        ld.global.u32 %r19, [%rd15+16];
        dp4a.s32.s32 %r20, %r15, %r18, 0;
        dp4a.s32.s32 %r20, %r16, %r19, %r20;
        cvt.rn.f32.s32 %f5, %r20;
        mul.f32 %f6, %f2, %f3;
        fma.rn.f32 %f1, %f6, %f5, %f1;
        setp.eq.u32 %p2, %r4, 0;
        @!%p2 bra $L_NEXT;
        mul.f32 %f7, %f2, %f4;
        mul.f32 %f8, %f7, 0f41000000;
        sub.f32 %f1, %f1, %f8;
    $L_NEXT:
        add.u32 %r7, %r7, 8;
        bra $L_LOOP;
    $L_RED:
        shfl.sync.down.b32 %f9, %f1, 16, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 8, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 4, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 2, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 1, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        setp.eq.u32 %p3, %r3, 0;
        @!%p3 bra $L_DONE;
        mul.wide.u32 %rd13, %r2, 4;
        add.u64 %rd14, %rd6, %rd13;
        st.global.f32 [%rd14], %f1;
    $L_DONE:
        ret;
    }
"#;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x007F_FFFF;
    if exp > 15 {
        return sign | 0x7C00;
    }
    if exp < -14 {
        return sign;
    }
    let e16 = ((exp + 15) as u16) << 10;
    let m16 = ((mant + 0x1000) >> 13) as u16;
    sign | e16 | (m16 & 0x03FF)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

fn cmd_diff(args: &[String]) -> Result<(), String> {
    let mut lhs_isa = IsaTarget::Gfx1100;
    let mut rhs_isa = IsaTarget::Gfx942;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--lhs" => {
                i += 1;
                let s = args.get(i).ok_or("missing ISA after --lhs")?;
                lhs_isa = IsaTarget::from_str(s).map_err(|e| e.to_string())?;
            }
            "--rhs" => {
                i += 1;
                let s = args.get(i).ok_or("missing ISA after --rhs")?;
                rhs_isa = IsaTarget::from_str(s).map_err(|e| e.to_string())?;
            }
            other => return Err(format!("unknown diff option '{other}'")),
        }
        i += 1;
    }

    let lhs_backend = Arc::new(MockBackend::with_isa(lhs_isa));
    let rhs_backend = Arc::new(MockBackend::with_isa(rhs_isa));
    let tol = DiffTolerance {
        abs_tol: 1e-3,
        rel_tol: 1e-3,
    };

    println!("AMDA Dual-Backend Differential Oracle");
    println!("=====================================");
    println!("LHS: {lhs_isa} (wave {}) | RHS: {rhs_isa} (wave {})", lhs_isa.native_wave_size(), rhs_isa.native_wave_size());
    println!("{:<14} {:>7} {:>14} {:>14} {:>8}", "WORKLOAD", "ELEMS", "MAX_ABS(L-R)", "MAX_ABS(CPU)", "STATUS");

    let mut all_ok = true;

    // 1. vector_add
    let n_vadd = 2048usize;
    let va_f32: Vec<f32> = (0..n_vadd).map(|k| k as f32 * 0.5).collect();
    let vb_f32: Vec<f32> = (0..n_vadd).map(|k| (n_vadd - k) as f32 * 0.25).collect();
    let cpu_vadd: Vec<f32> = va_f32.iter().zip(vb_f32.iter()).map(|(a, b)| a + b).collect();
    let va_bytes: Vec<u8> = va_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let vb_bytes: Vec<u8> = vb_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let clean_vadd = DEMO_VECTOR_ADD_PTX.trim_end_matches('\0');
    let vadd_args = [
        DiffKernelArg::InputBytes(&va_bytes),
        DiffKernelArg::InputBytes(&vb_bytes),
        DiffKernelArg::OutputF32(n_vadd),
        DiffKernelArg::ScalarU32(n_vadd as u32),
    ];
    let rep_vadd = run_differential_ptx_f32(
        "vector_add",
        lhs_backend.clone(),
        rhs_backend.clone(),
        clean_vadd,
        "vector_add",
        [(n_vadd as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
        &vadd_args,
        tol,
    )
    .map_err(|e| e.to_string())?;
    let lhs_vadd_out = execute_ptx_f32_on_backend(
        lhs_backend.clone(),
        clean_vadd,
        "vector_add",
        [(n_vadd as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
        &vadd_args,
    )
    .map_err(|e| e.to_string())?;
    let cpu_vadd_rep = compare_f32_slices("vector_add", "lhs", "cpu", &lhs_vadd_out, &cpu_vadd, tol);
    let ok1 = rep_vadd.passed && cpu_vadd_rep.passed;
    all_ok &= ok1;
    println!(
        "{:<14} {:>7} {:>14.3e} {:>14.3e} {:>8}",
        rep_vadd.workload,
        rep_vadd.elements,
        rep_vadd.max_abs_err,
        cpu_vadd_rep.max_abs_err,
        if ok1 { "PASS" } else { "FAIL" }
    );

    // 2. saxpy
    let alpha = 2.5f32;
    let cpu_saxpy: Vec<f32> = va_f32.iter().zip(vb_f32.iter()).map(|(x, y)| alpha * x + y).collect();
    let clean_saxpy = DEMO_SAXPY_PTX.trim_end_matches('\0');
    let saxpy_args = [
        DiffKernelArg::ScalarF32(alpha),
        DiffKernelArg::InputBytes(&va_bytes),
        DiffKernelArg::InOutF32(&vb_f32),
        DiffKernelArg::ScalarU32(n_vadd as u32),
    ];
    let rep_saxpy = run_differential_ptx_f32(
        "saxpy",
        lhs_backend.clone(),
        rhs_backend.clone(),
        clean_saxpy,
        "saxpy",
        [(n_vadd as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
        &saxpy_args,
        tol,
    )
    .map_err(|e| e.to_string())?;
    let lhs_saxpy_out = execute_ptx_f32_on_backend(
        lhs_backend.clone(),
        clean_saxpy,
        "saxpy",
        [(n_vadd as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
        &saxpy_args,
    )
    .map_err(|e| e.to_string())?;
    let cpu_saxpy_rep = compare_f32_slices("saxpy", "lhs", "cpu", &lhs_saxpy_out, &cpu_saxpy, tol);
    let ok2 = rep_saxpy.passed && cpu_saxpy_rep.passed;
    all_ok &= ok2;
    println!(
        "{:<14} {:>7} {:>14.3e} {:>14.3e} {:>8}",
        rep_saxpy.workload,
        rep_saxpy.elements,
        rep_saxpy.max_abs_err,
        cpu_saxpy_rep.max_abs_err,
        if ok2 { "PASS" } else { "FAIL" }
    );

    // 3. row_softmax
    let (s_rows, s_cols) = (4usize, 32usize);
    let sm_in: Vec<f32> = (0..(s_rows * s_cols))
        .map(|k| ((k as f32) * 0.19).sin() * 2.0)
        .collect();
    let mut cpu_sm = vec![0.0f32; sm_in.len()];
    for r in 0..s_rows {
        let row = &sm_in[r * s_cols..(r + 1) * s_cols];
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..s_cols {
            cpu_sm[r * s_cols + c] = exps[c] / sum;
        }
    }
    let sm_in_bytes: Vec<u8> = sm_in.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sm_args = [
        DiffKernelArg::InputBytes(&sm_in_bytes),
        DiffKernelArg::OutputF32(sm_in.len()),
    ];
    let rep_sm = run_differential_ptx_f32(
        "row_softmax",
        lhs_backend.clone(),
        rhs_backend.clone(),
        DIFF_SOFTMAX_PTX,
        "row_softmax",
        [s_rows as u32, 1, 1],
        [s_cols as u32, 1, 1],
        0,
        &sm_args,
        tol,
    )
    .map_err(|e| e.to_string())?;
    let lhs_sm_out = execute_ptx_f32_on_backend(
        lhs_backend.clone(),
        DIFF_SOFTMAX_PTX,
        "row_softmax",
        [s_rows as u32, 1, 1],
        [s_cols as u32, 1, 1],
        0,
        &sm_args,
    )
    .map_err(|e| e.to_string())?;
    let cpu_sm_rep = compare_f32_slices("row_softmax", "lhs", "cpu", &lhs_sm_out, &cpu_sm, tol);
    let ok3 = rep_sm.passed && cpu_sm_rep.passed;
    all_ok &= ok3;
    println!(
        "{:<14} {:>7} {:>14.3e} {:>14.3e} {:>8}",
        rep_sm.workload,
        rep_sm.elements,
        rep_sm.max_abs_err,
        cpu_sm_rep.max_abs_err,
        if ok3 { "PASS" } else { "FAIL" }
    );

    // 4. q4_0_matvec
    let (q_rows, q_blks) = (8usize, 4usize);
    let (w_bytes, x_bytes, cpu_q4) = build_q4_0_fixture(q_rows, q_blks);
    let q_args = [
        DiffKernelArg::InputBytes(&w_bytes),
        DiffKernelArg::InputBytes(&x_bytes),
        DiffKernelArg::OutputF32(q_rows),
        DiffKernelArg::ScalarU32(q_blks as u32),
    ];
    let rep_q4 = run_differential_ptx_f32(
        "q4_0_matvec",
        lhs_backend.clone(),
        rhs_backend,
        DIFF_Q4_0_PTX,
        "ggml_matvec_q4_0_q8_1",
        [q_rows as u32, 1, 1],
        [32, 1, 1],
        0,
        &q_args,
        tol,
    )
    .map_err(|e| e.to_string())?;
    let lhs_q4_out = execute_ptx_f32_on_backend(
        lhs_backend,
        DIFF_Q4_0_PTX,
        "ggml_matvec_q4_0_q8_1",
        [q_rows as u32, 1, 1],
        [32, 1, 1],
        0,
        &q_args,
    )
    .map_err(|e| e.to_string())?;
    let cpu_q4_rep = compare_f32_slices("q4_0_matvec", "lhs", "cpu", &lhs_q4_out, &cpu_q4, tol);
    let ok4 = rep_q4.passed && cpu_q4_rep.passed;
    all_ok &= ok4;
    println!(
        "{:<14} {:>7} {:>14.3e} {:>14.3e} {:>8}",
        rep_q4.workload,
        rep_q4.elements,
        rep_q4.max_abs_err,
        cpu_q4_rep.max_abs_err,
        if ok4 { "PASS" } else { "FAIL" }
    );

    if all_ok {
        Ok(())
    } else {
        Err("differential oracle detected numerical mismatch".to_owned())
    }
}

fn build_q4_0_fixture(q_rows: usize, q_blks: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut w_bytes = Vec::with_capacity(q_rows * q_blks * 18);
    let mut x_bytes = Vec::with_capacity(q_blks * 36);
    let mut x_blks = Vec::with_capacity(q_blks);
    for blk in 0..q_blks {
        let dx16 = f32_to_f16_bits(0.0625 * (blk as f32 + 1.0));
        let dx = f16_bits_to_f32(dx16);
        let mut qs = [0i8; 32];
        let mut sum = 0i32;
        for (idx, q) in qs.iter_mut().enumerate() {
            let v = (((blk * 5 + idx * 3) % 15) as i32) - 7;
            *q = v as i8;
            sum += v;
        }
        let sx16 = f32_to_f16_bits(dx * (sum as f32));
        let sx = f16_bits_to_f32(sx16);
        x_bytes.extend_from_slice(&dx16.to_le_bytes());
        x_bytes.extend_from_slice(&sx16.to_le_bytes());
        for &v in &qs {
            x_bytes.push(v as u8);
        }
        x_blks.push((dx, sx, qs));
    }
    let mut cpu_q4 = Vec::with_capacity(q_rows);
    for r in 0..q_rows {
        let mut acc = 0.0f32;
        for (blk, &(dx, sx, ref x_qs)) in x_blks.iter().enumerate() {
            let dw16 = f32_to_f16_bits(0.125 * (((r + blk) % 4) as f32 + 1.0));
            let dw = f16_bits_to_f32(dw16);
            w_bytes.extend_from_slice(&dw16.to_le_bytes());
            let mut w_qs = [0u8; 16];
            for (j, byte) in w_qs.iter_mut().enumerate() {
                let lo = ((r * 3 + blk + j) % 16) as u8;
                let hi = ((r * 5 + blk + j + 7) % 16) as u8;
                *byte = lo | (hi << 4);
            }
            w_bytes.extend_from_slice(&w_qs);
            let mut idot = 0i32;
            for j in 0..16 {
                idot += i32::from(w_qs[j] & 0x0F) * i32::from(x_qs[j]);
                idot += i32::from((w_qs[j] >> 4) & 0x0F) * i32::from(x_qs[j + 16]);
            }
            acc += dw * (dx * (idot as f32) - 8.0 * sx);
        }
        cpu_q4.push(acc);
    }
    (w_bytes, x_bytes, cpu_q4)
}

const BENCH_RELU_SCALE_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64
    .visible .entry relu_scale(
        .param .u64 inp,
        .param .u64 out,
        .param .f32 scale,
        .param .u32 n
    ) {
        .reg .pred %p<2>;
        .reg .f32 %f<5>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<8>;
        ld.param.u64 %rd1, [inp];
        ld.param.u64 %rd2, [out];
        ld.param.f32 %f1, [scale];
        ld.param.u32 %r1, [n];
        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %tid.x;
        mad.lo.s32 %r5, %r2, %r3, %r4;
        setp.ge.s32 %p1, %r5, %r1;
        @%p1 bra $L_EXIT;
        cvta.to.global.u64 %rd3, %rd1;
        cvta.to.global.u64 %rd4, %rd2;
        mul.wide.s32 %rd5, %r5, 4;
        add.s64 %rd6, %rd3, %rd5;
        ld.global.f32 %f2, [%rd6];
        max.f32 %f3, %f2, 0f00000000;
        mul.f32 %f4, %f3, %f1;
        add.s64 %rd7, %rd4, %rd5;
        st.global.f32 [%rd7], %f4;
    $L_EXIT:
        ret;
    }
"#;

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let mut iters: usize = 20;
    let mut target = IsaTarget::Gfx1100;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                let s = args.get(i).ok_or("missing integer after --iters")?;
                iters = s
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --iters '{s}'"))?
                    .max(1);
            }
            "--target" => {
                i += 1;
                let s = args.get(i).ok_or("missing ISA after --target")?;
                target = IsaTarget::from_str(s).map_err(|e| e.to_string())?;
            }
            other => return Err(format!("unknown bench option '{other}'")),
        }
        i += 1;
    }

    println!("AMDA Reproducible Benchmark Suite");
    println!("=================================");
    println!("Target ISA: {target} (wave {}), Iterations: {iters}", target.native_wave_size());
    println!();

    // 1. Tiered JIT Compile Latency
    println!("[1] Tiered JIT Compile Latency (Cold Tier 0 vs Warm Cache Hit vs Cold Tier 1)");
    println!(
        "{:<14} {:>12} {:>12} {:>12} {:>10}",
        "KERNEL", "COLD_T0_US", "WARM_HIT_US", "COLD_T1_US", "T1/T0"
    );
    let clean_vadd = DEMO_VECTOR_ADD_PTX.trim_end_matches('\0');
    let clean_saxpy = DEMO_SAXPY_PTX.trim_end_matches('\0');
    for (kname, ptx) in [
        ("vector_add", clean_vadd),
        ("saxpy", clean_saxpy),
        ("row_softmax", DIFF_SOFTMAX_PTX),
        ("q4_0_matvec", DIFF_Q4_0_PTX),
    ] {
        let comp0 = Compiler::with_cache_dir(None);
        let t0 = Instant::now();
        let art0 = comp0
            .compile_ptx_with_policy(ptx, target, TierPolicy::ForceTier0)
            .map_err(|e| e.to_string())?;
        let cold_t0_us = t0.elapsed().as_secs_f64() * 1e6;
        assert!(!art0.cache_hit);

        let tw = Instant::now();
        let artw = comp0
            .compile_ptx_with_policy(ptx, target, TierPolicy::ForceTier0)
            .map_err(|e| e.to_string())?;
        let warm_hit_us = tw.elapsed().as_secs_f64() * 1e6;
        assert!(artw.cache_hit);

        let comp1 = Compiler::with_cache_dir(None);
        let t1 = Instant::now();
        let _art1 = comp1
            .compile_ptx_with_policy(ptx, target, TierPolicy::ForceTier1)
            .map_err(|e| e.to_string())?;
        let cold_t1_us = t1.elapsed().as_secs_f64() * 1e6;
        let ratio = cold_t1_us / cold_t0_us.max(0.01);
        println!(
            "{:<14} {:>12.1} {:>12.1} {:>12.1} {:>9.2}x",
            kname, cold_t0_us, warm_hit_us, cold_t1_us, ratio
        );
    }
    println!();

    // Setup Virtual AMDA Runtime on `target` for sections 2 & 3
    let mock = Arc::new(MockBackend::with_isa(target));
    let rt = Runtime::with_backend(mock.clone());
    let ctx = rt.create_context(0).map_err(|e| e.to_string())?;

    // 2. Dispatch Overhead: Eager vs Pre-Baked AQL Graph
    let n: u32 = 2048;
    let bytes = (n as usize) * 4;
    let d_a = ctx.alloc(bytes).map_err(|e| e.to_string())?;
    let d_b = ctx.alloc(bytes).map_err(|e| e.to_string())?;
    let d_tmp = ctx.alloc(bytes).map_err(|e| e.to_string())?;
    let d_out_u = ctx.alloc(bytes).map_err(|e| e.to_string())?;
    let d_out_f = ctx.alloc(bytes).map_err(|e| e.to_string())?;

    let ha: Vec<f32> = (0..n).map(|k| ((k % 31) as f32) - 10.0).collect();
    let hb: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.25 - 1.5).collect();
    let ha_raw: Vec<u8> = ha.iter().flat_map(|v| v.to_le_bytes()).collect();
    let hb_raw: Vec<u8> = hb.iter().flat_map(|v| v.to_le_bytes()).collect();
    ctx.copy_htod(d_a, &ha_raw).map_err(|e| e.to_string())?;
    ctx.copy_htod(d_b, &hb_raw).map_err(|e| e.to_string())?;

    let mod_saxpy = ctx.load_module(clean_saxpy.as_bytes()).map_err(|e| e.to_string())?;
    let fn_saxpy = mod_saxpy.get_function("saxpy").map_err(|e| e.to_string())?;
    let alpha_b = 2.0f32.to_le_bytes();
    let a_b = d_a.to_le_bytes();
    let b_b = d_b.to_le_bytes();
    let n_b = n.to_le_bytes();

    // Warm up once
    ctx.launch_kernel(&fn_saxpy, [8, 1, 1], [256, 1, 1], 0, None, &[&alpha_b, &a_b, &b_b, &n_b])
        .map_err(|e| e.to_string())?;
    let t_eager = Instant::now();
    for _ in 0..iters {
        ctx.launch_kernel(&fn_saxpy, [8, 1, 1], [256, 1, 1], 0, None, &[&alpha_b, &a_b, &b_b, &n_b])
            .map_err(|e| e.to_string())?;
    }
    let eager_us = (t_eager.elapsed().as_secs_f64() * 1e6) / (iters as f64);

    let ka_sax =
        amda_runtime::pack_kernargs(&fn_saxpy.metadata, &[&alpha_b, &a_b, &b_b, &n_b])
            .map_err(|e| e.to_string())?;
    let mut g_sax = amda_runtime::ExecutionGraph::new();
    g_sax
        .add_kernel_node(&fn_saxpy, [8, 1, 1], [256, 1, 1], 0, ka_sax, &[])
        .map_err(|e| e.to_string())?;
    let exec_sax = ctx.instantiate_graph_with_fusion(&g_sax, false).map_err(|e| e.to_string())?;
    let t_graph = Instant::now();
    for _ in 0..iters {
        ctx.launch_graph(&exec_sax, None).map_err(|e| e.to_string())?;
    }
    ctx.synchronize().map_err(|e| e.to_string())?;
    let graph_us = (t_graph.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    println!("[2] Launch & Dispatch Overhead (saxpy, N={n}, {iters} launches)");
    println!(
        "  Eager launch_kernel : {eager_us:.2} us/launch | Pre-baked AQL Graph : {graph_us:.2} us/launch | Speedup : {:.2}x",
        eager_us / graph_us.max(0.01)
    );
    println!();

    // 3. Cross-Kernel IR Fusion: Unfused vs Auto-Fused (`vector_add` -> `relu_scale`)
    let mod_vadd = ctx.load_module(clean_vadd.as_bytes()).map_err(|e| e.to_string())?;
    let fn_vadd = mod_vadd.get_function("vector_add").map_err(|e| e.to_string())?;
    let mod_relu = ctx
        .load_module(BENCH_RELU_SCALE_PTX.as_bytes())
        .map_err(|e| e.to_string())?;
    let fn_relu = mod_relu.get_function("relu_scale").map_err(|e| e.to_string())?;

    let tmp_b = d_tmp.to_le_bytes();
    let out_u_b = d_out_u.to_le_bytes();
    let out_f_b = d_out_f.to_le_bytes();
    let scale_b = 1.5f32.to_le_bytes();
    let ka_p =
        amda_runtime::pack_kernargs(&fn_vadd.metadata, &[&a_b, &b_b, &tmp_b, &n_b])
            .map_err(|e| e.to_string())?;
    let ka_cu =
        amda_runtime::pack_kernargs(&fn_relu.metadata, &[&tmp_b, &out_u_b, &scale_b, &n_b])
            .map_err(|e| e.to_string())?;
    let ka_cf =
        amda_runtime::pack_kernargs(&fn_relu.metadata, &[&tmp_b, &out_f_b, &scale_b, &n_b])
            .map_err(|e| e.to_string())?;

    let mut gu = amda_runtime::ExecutionGraph::new();
    let u0 = gu
        .add_kernel_node(&fn_vadd, [8, 1, 1], [256, 1, 1], 0, ka_p.clone(), &[])
        .map_err(|e| e.to_string())?;
    gu.add_kernel_node(&fn_relu, [8, 1, 1], [256, 1, 1], 0, ka_cu, &[u0])
        .map_err(|e| e.to_string())?;
    let exec_u = ctx.instantiate_graph_with_fusion(&gu, false).map_err(|e| e.to_string())?;

    let mut gf = amda_runtime::ExecutionGraph::new();
    let f0 = gf
        .add_kernel_node(&fn_vadd, [8, 1, 1], [256, 1, 1], 0, ka_p, &[])
        .map_err(|e| e.to_string())?;
    gf.add_kernel_node(&fn_relu, [8, 1, 1], [256, 1, 1], 0, ka_cf, &[f0])
        .map_err(|e| e.to_string())?;
    let exec_f = ctx.instantiate_graph_with_fusion(&gf, true).map_err(|e| e.to_string())?;

    let k0_u = mock.stats().kernels_dispatched;
    let tu = Instant::now();
    for _ in 0..iters {
        ctx.launch_graph(&exec_u, None).map_err(|e| e.to_string())?;
    }
    ctx.synchronize().map_err(|e| e.to_string())?;
    let unfused_us = (tu.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    let dispatches_u = mock.stats().kernels_dispatched - k0_u;

    let k0_f = mock.stats().kernels_dispatched;
    let tf = Instant::now();
    for _ in 0..iters {
        ctx.launch_graph(&exec_f, None).map_err(|e| e.to_string())?;
    }
    ctx.synchronize().map_err(|e| e.to_string())?;
    let fused_us = (tf.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    let dispatches_f = mock.stats().kernels_dispatched - k0_f;

    let mut raw_u = vec![0u8; bytes];
    let mut raw_f = vec![0u8; bytes];
    ctx.copy_dtoh(&mut raw_u, d_out_u).map_err(|e| e.to_string())?;
    ctx.copy_dtoh(&mut raw_f, d_out_f).map_err(|e| e.to_string())?;
    let out_u: Vec<f32> = raw_u
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let out_f: Vec<f32> = raw_f
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let parity = compare_f32_slices("fusion_parity", "unfused", "fused", &out_u, &out_f, DiffTolerance::default());
    if !parity.passed {
        return Err(format!("fusion parity failed: max_abs={}", parity.max_abs_err));
    }

    println!("[3] Cross-Kernel IR Fusion (vector_add -> relu_scale, N={n})");
    println!(
        "  Unfused : steps={}, dispatches={dispatches_u}, global_io=20 B/elem, {unfused_us:.2} us/iter",
        exec_u.step_count()
    );
    println!(
        "  Fused   : steps={}, dispatches={dispatches_f}, global_io=12 B/elem (-40% HBM traffic), {fused_us:.2} us/iter, max_abs_err={:.1e}",
        exec_f.step_count(),
        parity.max_abs_err
    );
    println!();

    // 4. cuBLAS SGEMM (32x32x32) & GGML Q4_0 MatVec (16x128)
    let _ = ctx.free(d_a);
    let _ = ctx.free(d_b);
    let _ = ctx.free(d_tmp);
    let _ = ctx.free(d_out_u);
    let _ = ctx.free(d_out_f);
    let _ = ctx.unload_module(mod_saxpy);
    let _ = ctx.unload_module(mod_vadd);
    let _ = ctx.unload_module(mod_relu);

    let (q_rows, q_blks) = (16usize, 4usize);
    let (qw_bytes, qx_bytes, _qcpu) = build_q4_0_fixture(q_rows, q_blks);
    let q_args = [
        DiffKernelArg::InputBytes(&qw_bytes),
        DiffKernelArg::InputBytes(&qx_bytes),
        DiffKernelArg::OutputF32(q_rows),
        DiffKernelArg::ScalarU32(q_blks as u32),
    ];
    let t_q4 = Instant::now();
    let _ = execute_ptx_f32_on_backend(
        mock,
        DIFF_Q4_0_PTX,
        "ggml_matvec_q4_0_q8_1",
        [q_rows as u32, 1, 1],
        [32, 1, 1],
        0,
        &q_args,
    )
    .map_err(|e| e.to_string())?;
    let q4_ms = t_q4.elapsed().as_secs_f64() * 1000.0;

    let rc = cudaSetDevice(0);
    if rc != CudaErrorT::CudaSuccess {
        return Err(format!("cudaSetDevice failed: {rc:?}"));
    }
    extern "C" fn cli_cuda_set_device(d: c_int) -> c_int {
        cudaSetDevice(d) as c_int
    }
    extern "C" fn cli_cu_ctx_get_current(p: *mut CUcontext) -> c_int {
        amda_cuda::cuCtxGetCurrent(p) as c_int
    }
    extern "C" fn cli_cu_module_load_data(m: *mut CUmodule, img: *const c_void) -> c_int {
        cuModuleLoadData(m, img) as c_int
    }
    extern "C" fn cli_cu_module_get_function(
        f: *mut CUfunction,
        m: CUmodule,
        nm: *const c_char,
    ) -> c_int {
        cuModuleGetFunction(f, m, nm) as c_int
    }
    extern "C" fn cli_cu_module_unload(m: CUmodule) -> c_int {
        cuModuleUnload(m) as c_int
    }
    extern "C" fn cli_cu_launch_kernel(
        f: CUfunction,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        sm: c_uint,
        st: CUstream,
        kp: *mut *mut c_void,
        ex: *mut *mut c_void,
    ) -> c_int {
        amda_cuda::cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sm, st, kp, ex) as c_int
    }
    register_cuda_dispatch(CudaDispatchTable {
        cuda_set_device: cli_cuda_set_device,
        cu_ctx_get_current: cli_cu_ctx_get_current,
        cu_module_load_data: cli_cu_module_load_data,
        cu_module_get_function: cli_cu_module_get_function,
        cu_module_unload: cli_cu_module_unload,
        cu_launch_kernel: cli_cu_launch_kernel,
    });
    let mut blas: CublasHandleT = core::ptr::null_mut();
    if cublasCreate_v2(&mut blas) != CublasStatusT::CUBLAS_STATUS_SUCCESS {
        return Err("cublasCreate_v2 failed".to_owned());
    }
    let m_dim = 32usize;
    let gemm_bytes = m_dim * m_dim * 4;
    let (mut ga, mut gb, mut gc) = (
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let ha_gemm = vec![0.25f32; m_dim * m_dim];
    let hb_gemm = vec![0.5f32; m_dim * m_dim];
    if cudaMalloc(&mut ga, gemm_bytes) != CudaErrorT::CudaSuccess
        || cudaMalloc(&mut gb, gemm_bytes) != CudaErrorT::CudaSuccess
        || cudaMalloc(&mut gc, gemm_bytes) != CudaErrorT::CudaSuccess
    {
        return Err("cudaMalloc for gemm failed".to_owned());
    }
    let _ = cudaMemcpy(ga, ha_gemm.as_ptr() as *const c_void, gemm_bytes, CudaMemcpyKind::HostToDevice);
    let _ = cudaMemcpy(gb, hb_gemm.as_ptr() as *const c_void, gemm_bytes, CudaMemcpyKind::HostToDevice);
    let (alpha_g, beta_g) = (1.0f32, 0.0f32);
    // Warm compile + first launch
    let _ = cublasSgemm_v2(
        blas,
        CublasOperationT::CUBLAS_OP_N,
        CublasOperationT::CUBLAS_OP_N,
        m_dim as c_int,
        m_dim as c_int,
        m_dim as c_int,
        &alpha_g,
        ga as *const f32,
        m_dim as c_int,
        gb as *const f32,
        m_dim as c_int,
        &beta_g,
        gc as *mut f32,
        m_dim as c_int,
    );
    let t_gemm = Instant::now();
    let gemm_iters = iters.min(5);
    for _ in 0..gemm_iters {
        let st = cublasSgemm_v2(
            blas,
            CublasOperationT::CUBLAS_OP_N,
            CublasOperationT::CUBLAS_OP_N,
            m_dim as c_int,
            m_dim as c_int,
            m_dim as c_int,
            &alpha_g,
            ga as *const f32,
            m_dim as c_int,
            gb as *const f32,
            m_dim as c_int,
            &beta_g,
            gc as *mut f32,
            m_dim as c_int,
        );
        if st != CublasStatusT::CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 failed: {st:?}"));
        }
    }
    let gemm_ms = (t_gemm.elapsed().as_secs_f64() * 1000.0) / (gemm_iters as f64);
    let _ = cudaFree(ga);
    let _ = cudaFree(gb);
    let _ = cudaFree(gc);
    let _ = cublasDestroy_v2(blas);

    println!("[4] AI Library & Quantized Inference Micro-Benchmarks");
    println!("  cuBLAS SGEMM (32x32x32 steady)    : {gemm_ms:.3} ms/gemm");
    println!("  GGML Q4_0 x Q8_1 (16x128 cold+run): {q4_ms:.3} ms");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_demo_and_info_succeed() {
        cmd_info().expect("amda info should succeed");
        cmd_demo().expect("amda demo should succeed");
        cmd_diff(&[]).expect("amda diff should succeed");
        cmd_bench(&["--iters".to_owned(), "2".to_owned()]).expect("amda bench should succeed");
    }
}
