//! AMDA core AMD-native GPU runtime.

pub mod context;
pub mod diff;
pub mod error;
pub mod graph;
pub mod memory;
pub mod module;
pub mod stream;
pub mod trace;

pub use context::{Context, Runtime, TierStats};
pub use diff::{
    compare_f32_slices, execute_ptx_f32_on_backend, run_differential_ptx_f32, DiffKernelArg,
    DiffMismatch, DiffReport, DiffTolerance,
};
pub use error::RuntimeError;
pub use graph::{ExecutableGraph, ExecutionGraph, GraphNode};
pub use memory::{MemoryPool, MemoryStats, VmmRegistry, VmmReservation};
pub use module::{pack_kernargs, KernelFunction, LoadedModule};
pub use stream::{Event, Stream};
pub use trace::{format_hex_preview, parse_trace_flag, trace_enabled, trace_log};

#[cfg(test)]
mod tests {
    use super::*;
    use amda_elf::IsaTarget;
    use amda_hal::MockBackend;
    use std::sync::Arc;

    const VEC_ADD_PTX: &str = r#"
        .version 7.8
        .target sm_80
        .address_size 64
        .visible .entry vector_add(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        ) {
            .reg .pred %p<2>;
            .reg .f32 %f<4>;
            .reg .b32 %r<6>;
            .reg .b64 %rd<10>;

            ld.param.u64 %rd1, [a];
            ld.param.u64 %rd2, [b];
            ld.param.u64 %rd3, [c];
            ld.param.u32 %r1, [n];

            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra $L_EXIT;

            cvta.to.global.u64 %rd4, %rd1;
            cvta.to.global.u64 %rd5, %rd2;
            cvta.to.global.u64 %rd6, %rd3;
            mul.wide.s32 %rd7, %r5, 4;
            add.s64 %rd8, %rd4, %rd7;
            add.s64 %rd9, %rd5, %rd7;
            ld.global.f32 %f1, [%rd8];
            ld.global.f32 %f2, [%rd9];
            add.f32 %f3, %f1, %f2;
            add.s64 %rd8, %rd6, %rd7;
            st.global.f32 [%rd8], %f3;

        $L_EXIT:
            ret;
        }
    "#;

    #[test]
    fn slab_pool_reuses_buffers_and_launches_kernel() {
        let rt = Runtime::with_backend(Arc::new(MockBackend::with_isa(IsaTarget::Gfx1100)));
        let ctx = rt.create_context(0).unwrap();

        let p1 = ctx.alloc(1024).unwrap();
        ctx.free(p1).unwrap();
        let p2 = ctx.alloc(1024).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(ctx.memory_stats().pool_hits, 1);
        ctx.free(p2).unwrap();

        let module = ctx.load_module(VEC_ADD_PTX.as_bytes()).unwrap();
        let func = module.get_function("vector_add").unwrap();

        let n = 300u32;
        let bytes = (n as usize) * 4;
        let da = ctx.alloc(bytes).unwrap();
        let db = ctx.alloc(bytes).unwrap();
        let dc = ctx.alloc(bytes).unwrap();

        let ha: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let hb: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let ha_bytes: Vec<u8> = ha.iter().flat_map(|f| f.to_le_bytes()).collect();
        let hb_bytes: Vec<u8> = hb.iter().flat_map(|f| f.to_le_bytes()).collect();

        ctx.copy_htod(da, &ha_bytes).unwrap();
        ctx.copy_htod(db, &hb_bytes).unwrap();

        let arg_a = da.to_le_bytes();
        let arg_b = db.to_le_bytes();
        let arg_c = dc.to_le_bytes();
        let arg_n = n.to_le_bytes();

        // 3 blocks x 128 threads = 384 threads (> 300, testing bounds check + multi-block HSA conversion)
        ctx.launch_kernel(
            &func,
            [3, 1, 1],
            [128, 1, 1],
            0,
            None,
            &[&arg_a, &arg_b, &arg_c, &arg_n],
        )
        .unwrap();

        let mut hc_bytes = vec![0u8; bytes];
        ctx.copy_dtoh(&mut hc_bytes, dc).unwrap();
        for (i, chunk) in hc_bytes.chunks_exact(4).enumerate() {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(val, (i * 3) as f32);
        }

        // Test graph capture & replay
        let kernarg_bytes =
            pack_kernargs(&func.metadata, &[&arg_a, &arg_b, &arg_c, &arg_n]).unwrap();
        let mut graph = ExecutionGraph::new();
        let n0 = graph
            .add_kernel_node(&func, [3, 1, 1], [128, 1, 1], 0, kernarg_bytes, &[])
            .unwrap();
        graph.add_barrier_node(&[n0]).unwrap();
        let exec = ctx.instantiate_graph(&graph).unwrap();
        ctx.launch_graph(&exec, None).unwrap();
        ctx.synchronize().unwrap();

        ctx.free(da).unwrap();
        ctx.free(db).unwrap();
        ctx.free(dc).unwrap();
        ctx.unload_module(module).unwrap();
    }

    #[test]
    fn hot_kernel_promotes_from_tier0_to_tier1() {
        let rt = Runtime::with_backend(Arc::new(MockBackend::with_isa(IsaTarget::Gfx1100)));
        let ctx = rt.create_context(0).unwrap();
        ctx.set_promote_threshold(3);

        let module = ctx.load_module(VEC_ADD_PTX.as_bytes()).unwrap();
        let func = module.get_function("vector_add").unwrap();
        assert_eq!(func.tier, amda_compiler::CompileTier::Tier0Fast);
        assert_eq!(
            ctx.kernel_tier(func.handle),
            Some((0, amda_compiler::CompileTier::Tier0Fast))
        );

        let n = 64u32;
        let bytes = (n as usize) * 4;
        let da = ctx.alloc(bytes).unwrap();
        let db = ctx.alloc(bytes).unwrap();
        let dc = ctx.alloc(bytes).unwrap();
        let arg_a = da.to_le_bytes();
        let arg_b = db.to_le_bytes();
        let arg_c = dc.to_le_bytes();
        let arg_n = n.to_le_bytes();

        for launch_idx in 1..=3 {
            ctx.launch_kernel(
                &func,
                [1, 1, 1],
                [64, 1, 1],
                0,
                None,
                &[&arg_a, &arg_b, &arg_c, &arg_n],
            )
            .unwrap();
            let expected_tier = if launch_idx < 3 {
                amda_compiler::CompileTier::Tier0Fast
            } else {
                amda_compiler::CompileTier::Tier1Optimizing
            };
            assert_eq!(ctx.kernel_tier(func.handle), Some((launch_idx, expected_tier)));
        }

        let stats = ctx.tier_stats();
        assert_eq!(stats.tier0_compiles, 1);
        assert_eq!(stats.tier1_compiles, 1);
        assert_eq!(stats.hot_promotions, 1);
        assert_eq!(stats.total_launches, 3);

        ctx.free(da).unwrap();
        ctx.free(db).unwrap();
        ctx.free(dc).unwrap();
        ctx.unload_module(module).unwrap();
    }

    #[test]
    fn trace_flag_parsing() {
        assert!(parse_trace_flag("1"));
        assert!(parse_trace_flag("true"));
        assert!(parse_trace_flag(" TRUE "));
        assert!(!parse_trace_flag("0"));
        assert!(!parse_trace_flag("false"));
        assert!(!parse_trace_flag(""));
    }

    const RELU_SCALE_PTX: &str = r#"
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

    #[test]
    fn graph_auto_fusion_merges_pipeline_bit_exact() {
        let mock = Arc::new(MockBackend::with_isa(IsaTarget::Gfx1100));
        let rt = Runtime::with_backend(Arc::clone(&mock) as Arc<dyn amda_hal::DeviceBackend>);
        let ctx = rt.create_context(0).unwrap();

        let mod_vadd = ctx.load_module(VEC_ADD_PTX.as_bytes()).unwrap();
        let fn_vadd = mod_vadd.get_function("vector_add").unwrap();
        let mod_relu = ctx.load_module(RELU_SCALE_PTX.as_bytes()).unwrap();
        let fn_relu = mod_relu.get_function("relu_scale").unwrap();

        let n = 300u32;
        let bytes = (n as usize) * 4;
        let d_a = ctx.alloc(bytes).unwrap();
        let d_b = ctx.alloc(bytes).unwrap();
        let d_tmp = ctx.alloc(bytes).unwrap();
        let d_out_unfused = ctx.alloc(bytes).unwrap();
        let d_out_fused = ctx.alloc(bytes).unwrap();

        let ha: Vec<f32> = (0..n).map(|i| (i as f32) - 100.0).collect();
        let hb: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.5 - 4.0).collect();
        ctx.copy_htod(
            d_a,
            &ha.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
        ctx.copy_htod(
            d_b,
            &hb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();

        let scale = 1.75f32;
        let arg_a = d_a.to_le_bytes();
        let arg_b = d_b.to_le_bytes();
        let arg_tmp = d_tmp.to_le_bytes();
        let arg_out_u = d_out_unfused.to_le_bytes();
        let arg_out_f = d_out_fused.to_le_bytes();
        let arg_scale = scale.to_le_bytes();
        let arg_n = n.to_le_bytes();

        let ka_prod =
            pack_kernargs(&fn_vadd.metadata, &[&arg_a, &arg_b, &arg_tmp, &arg_n]).unwrap();
        let ka_cons_u =
            pack_kernargs(&fn_relu.metadata, &[&arg_tmp, &arg_out_u, &arg_scale, &arg_n])
                .unwrap();
        let ka_cons_f =
            pack_kernargs(&fn_relu.metadata, &[&arg_tmp, &arg_out_f, &arg_scale, &arg_n])
                .unwrap();

        let mut graph_u = ExecutionGraph::new();
        let n0 = graph_u
            .add_kernel_node(&fn_vadd, [3, 1, 1], [128, 1, 1], 0, ka_prod.clone(), &[])
            .unwrap();
        graph_u
            .add_kernel_node(&fn_relu, [3, 1, 1], [128, 1, 1], 0, ka_cons_u, &[n0])
            .unwrap();
        let exec_u = ctx.instantiate_graph_with_fusion(&graph_u, false).unwrap();
        assert_eq!(exec_u.step_count(), 2);
        assert_eq!(exec_u.fused_pairs(), 0);

        let k_before_u = mock.stats().kernels_dispatched;
        ctx.launch_graph(&exec_u, None).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(mock.stats().kernels_dispatched - k_before_u, 2);

        let mut graph_f = ExecutionGraph::new();
        let m0 = graph_f
            .add_kernel_node(&fn_vadd, [3, 1, 1], [128, 1, 1], 0, ka_prod, &[])
            .unwrap();
        graph_f
            .add_kernel_node(&fn_relu, [3, 1, 1], [128, 1, 1], 0, ka_cons_f, &[m0])
            .unwrap();
        let exec_f = ctx.instantiate_graph_with_fusion(&graph_f, true).unwrap();
        assert_eq!(exec_f.step_count(), 1);
        assert_eq!(exec_f.fused_pairs(), 1);

        let k_before_f = mock.stats().kernels_dispatched;
        ctx.launch_graph(&exec_f, None).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(mock.stats().kernels_dispatched - k_before_f, 1);

        let mut raw_u = vec![0u8; bytes];
        let mut raw_f = vec![0u8; bytes];
        ctx.copy_dtoh(&mut raw_u, d_out_unfused).unwrap();
        ctx.copy_dtoh(&mut raw_f, d_out_fused).unwrap();
        assert_eq!(raw_u, raw_f);

        ctx.free(d_a).unwrap();
        ctx.free(d_b).unwrap();
        ctx.free(d_tmp).unwrap();
        ctx.free(d_out_unfused).unwrap();
        ctx.free(d_out_fused).unwrap();
        ctx.unload_module(mod_vadd).unwrap();
        ctx.unload_module(mod_relu).unwrap();
    }
}
