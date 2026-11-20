//! Execution Graphs: pre-baked AQL packet replay and cross-kernel fusion hooks.

use crate::error::RuntimeError;
use crate::memory::MemoryPool;
use crate::module::KernelFunction;
use crate::stream::Stream;
use amda_compiler::{
    analyze_kernel_fusion, find_fusible_buffer_params, fuse_elementwise_kernels,
    pack_fused_kernargs, CompileTier, Compiler, FusionCandidate, IrKernel, IrModule,
};
use amda_elf::{IsaTarget, KernelMetadata, ValueKind};
use amda_hal::{
    encode_aql_header, AqlPacketType, BarrierAndPacket, CodeObjectId, DeviceBackend, FenceScope,
    KernelDispatchPacket, KernelHandle,
};
use std::sync::Arc;

/// Node inside a staged [`ExecutionGraph`].
#[derive(Debug, Clone)]
pub enum GraphNode {
    KernelLaunch {
        kernel_handle: KernelHandle,
        metadata: KernelMetadata,
        ir_kernel: Option<Arc<IrKernel>>,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        kernarg_bytes: Vec<u8>,
    },
    MemcpyDtoD {
        dst: u64,
        src: u64,
        len: usize,
    },
    Barrier,
}

/// Directed acyclic GPU computation graph capturing kernel launches and copies.
#[derive(Debug, Default, Clone)]
pub struct ExecutionGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<(usize, usize)>,
}

impl ExecutionGraph {
    /// Creates an empty [`ExecutionGraph`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a kernel launch node and returns its node index.
    pub fn add_kernel_node(
        &mut self,
        func: &KernelFunction,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        kernarg_bytes: Vec<u8>,
        deps: &[usize],
    ) -> Result<usize, RuntimeError> {
        if grid_dim.contains(&0) || block_dim.contains(&0) {
            return Err(RuntimeError::InvalidArgument(
                "grid and block dimensions must be >= 1".to_owned(),
            ));
        }
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::KernelLaunch {
            kernel_handle: func.handle,
            metadata: func.metadata.clone(),
            ir_kernel: func.ir_kernel.clone(),
            grid_dim,
            block_dim,
            shared_mem,
            kernarg_bytes,
        });
        for &dep in deps {
            if dep >= idx {
                return Err(RuntimeError::InvalidArgument(
                    "graph dependency index out of bounds".to_owned(),
                ));
            }
            self.edges.push((dep, idx));
        }
        Ok(idx)
    }

    /// Appends a device-to-device copy node and returns its node index.
    pub fn add_memcpy_dtod_node(
        &mut self,
        dst: u64,
        src: u64,
        len: usize,
        deps: &[usize],
    ) -> Result<usize, RuntimeError> {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::MemcpyDtoD { dst, src, len });
        for &dep in deps {
            if dep >= idx {
                return Err(RuntimeError::InvalidArgument(
                    "graph dependency index out of bounds".to_owned(),
                ));
            }
            self.edges.push((dep, idx));
        }
        Ok(idx)
    }

    /// Appends an explicit barrier node.
    pub fn add_barrier_node(&mut self, deps: &[usize]) -> Result<usize, RuntimeError> {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Barrier);
        for &dep in deps {
            if dep >= idx {
                return Err(RuntimeError::InvalidArgument(
                    "graph dependency index out of bounds".to_owned(),
                ));
            }
            self.edges.push((dep, idx));
        }
        Ok(idx)
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Inspects consecutive kernel launches for cross-kernel JIT fusion opportunities.
    pub fn analyze_fusion_opportunities(&self) -> Vec<FusionCandidate> {
        let ir_refs: Vec<&IrKernel> = self
            .nodes
            .iter()
            .filter_map(|n| match n {
                GraphNode::KernelLaunch { ir_kernel, .. } => ir_kernel.as_deref(),
                _ => None,
            })
            .collect();
        analyze_kernel_fusion(&ir_refs)
    }

    /// Compiles this graph into an [`ExecutableGraph`] with pre-uploaded kernarg buffers
    /// and pre-baked 64-byte AQL dispatch packets (no automatic fusion without compiler).
    pub fn instantiate(
        &self,
        backend: Arc<dyn DeviceBackend>,
        memory: Arc<MemoryPool>,
        device: usize,
    ) -> Result<ExecutableGraph, RuntimeError> {
        self.instantiate_internal(backend, memory, None, IsaTarget::Gfx1100, device, false)
    }

    /// Compiles this graph into an [`ExecutableGraph`], optionally fusing consecutive compatible
    /// elementwise kernels into single AQL dispatch packets via `compiler`.
    pub fn instantiate_with_compiler(
        &self,
        backend: Arc<dyn DeviceBackend>,
        memory: Arc<MemoryPool>,
        compiler: &Compiler,
        isa: IsaTarget,
        device: usize,
        enable_fusion: bool,
    ) -> Result<ExecutableGraph, RuntimeError> {
        self.instantiate_internal(backend, memory, Some(compiler), isa, device, enable_fusion)
    }

    fn pointer_used_outside_pair(&self, ptr: u64, idx_a: usize, idx_b: usize) -> bool {
        if self.edges.iter().any(|&(src, dst)| src == idx_a && dst != idx_b) {
            return true;
        }
        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == idx_a || idx == idx_b {
                continue;
            }
            match node {
                GraphNode::MemcpyDtoD { dst, src, .. } => {
                    if *dst == ptr || *src == ptr {
                        return true;
                    }
                }
                GraphNode::KernelLaunch {
                    metadata,
                    kernarg_bytes,
                    ..
                } => {
                    for arg in &metadata.args {
                        if arg.value_kind == ValueKind::GlobalBuffer && arg.size == 8 {
                            let off = arg.offset as usize;
                            if let Some(slice) = kernarg_bytes.get(off..off + 8) {
                                let val = u64::from_le_bytes([
                                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5],
                                    slice[6], slice[7],
                                ]);
                                if val == ptr {
                                    return true;
                                }
                            }
                        }
                    }
                }
                GraphNode::Barrier => {}
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn try_fuse_pair(
        &self,
        idx: usize,
        backend: &dyn DeviceBackend,
        memory: &MemoryPool,
        compiler: &Compiler,
        isa: IsaTarget,
        device: usize,
        kernarg_allocs: &mut Vec<u64>,
        fused_code_objects: &mut Vec<CodeObjectId>,
    ) -> Result<Option<BakedStep>, RuntimeError> {
        let (
            GraphNode::KernelLaunch {
                ir_kernel: Some(prod_ir),
                grid_dim: p_grid,
                block_dim: p_block,
                shared_mem: p_smem,
                kernarg_bytes: prod_ka,
                ..
            },
            GraphNode::KernelLaunch {
                ir_kernel: Some(cons_ir),
                grid_dim: c_grid,
                block_dim: c_block,
                shared_mem: c_smem,
                kernarg_bytes: cons_ka,
                ..
            },
        ) = (&self.nodes[idx], &self.nodes[idx + 1])
        else {
            return Ok(None);
        };

        if p_grid != c_grid || p_block != c_block || *p_smem != 0 || *c_smem != 0 {
            return Ok(None);
        }
        let Some((prod_out, cons_in, shared_ptr)) =
            find_fusible_buffer_params(prod_ir, cons_ir, prod_ka, cons_ka)
        else {
            return Ok(None);
        };
        if self.pointer_used_outside_pair(shared_ptr, idx, idx + 1) {
            return Ok(None);
        }

        let Ok(plan) = fuse_elementwise_kernels(prod_ir, cons_ir, prod_out, cons_in) else {
            return Ok(None);
        };
        let Ok(fused_ka) = pack_fused_kernargs(&plan, prod_ir, cons_ir, prod_ka, cons_ka) else {
            return Ok(None);
        };

        let fused_mod = IrModule {
            target: isa,
            kernels: vec![plan.kernel],
        };
        let art = compiler.compile_ir_at_tier(&fused_mod, isa, CompileTier::Tier1Optimizing)?;
        let code_obj = backend.load_code_object(device, &art.hsaco)?;
        let (fused_handle, fused_meta) = match backend.get_kernel(code_obj, &fused_mod.kernels[0].name)
        {
            Ok(v) => v,
            Err(e) => {
                let _ = backend.unload_code_object(code_obj);
                return Err(e.into());
            }
        };
        fused_code_objects.push(code_obj);

        let step = bake_kernel_dispatch_step(
            backend,
            memory,
            device,
            fused_handle,
            &fused_meta,
            *p_grid,
            *p_block,
            0,
            &fused_ka,
            kernarg_allocs,
        )?;
        Ok(Some(step))
    }

    fn instantiate_internal(
        &self,
        backend: Arc<dyn DeviceBackend>,
        memory: Arc<MemoryPool>,
        compiler: Option<&Compiler>,
        isa: IsaTarget,
        device: usize,
        enable_fusion: bool,
    ) -> Result<ExecutableGraph, RuntimeError> {
        let mut steps = Vec::with_capacity(self.nodes.len());
        let mut kernarg_allocs = Vec::new();
        let mut fused_code_objects = Vec::new();
        let mut fused_pairs = 0usize;

        let build_res = (|| -> Result<(), RuntimeError> {
            let mut idx = 0usize;
            while idx < self.nodes.len() {
                if enable_fusion && idx + 1 < self.nodes.len() {
                    if let Some(comp) = compiler {
                        if let Some(fused_step) = self.try_fuse_pair(
                            idx,
                            &*backend,
                            &memory,
                            comp,
                            isa,
                            device,
                            &mut kernarg_allocs,
                            &mut fused_code_objects,
                        )? {
                            steps.push(fused_step);
                            fused_pairs += 1;
                            idx += 2;
                            continue;
                        }
                    }
                }

                match &self.nodes[idx] {
                    GraphNode::KernelLaunch {
                        kernel_handle,
                        metadata,
                        grid_dim,
                        block_dim,
                        shared_mem,
                        kernarg_bytes,
                        ..
                    } => {
                        let step = bake_kernel_dispatch_step(
                            &*backend,
                            &memory,
                            device,
                            *kernel_handle,
                            metadata,
                            *grid_dim,
                            *block_dim,
                            *shared_mem,
                            kernarg_bytes,
                            &mut kernarg_allocs,
                        )?;
                        steps.push(step);
                    }
                    GraphNode::MemcpyDtoD { dst, src, len } => {
                        steps.push(BakedStep::MemcpyDtoD {
                            dst: *dst,
                            src: *src,
                            len: *len,
                        });
                    }
                    GraphNode::Barrier => {
                        let pkt = BarrierAndPacket {
                            header: encode_aql_header(
                                AqlPacketType::BarrierAnd,
                                true,
                                FenceScope::System,
                                FenceScope::System,
                            ),
                            reserved0: 0,
                            reserved1: 0,
                            dep_signal: [0; 5],
                            reserved2: 0,
                            completion_signal: 0,
                        };
                        steps.push(BakedStep::AqlPacket(pkt.to_bytes()));
                    }
                }
                idx += 1;
            }
            Ok(())
        })();

        if let Err(err) = build_res {
            for ptr in kernarg_allocs {
                let _ = memory.free(&*backend, device, ptr);
            }
            for obj in fused_code_objects {
                let _ = backend.unload_code_object(obj);
            }
            return Err(err);
        }

        let fusion_candidates = self.analyze_fusion_opportunities();

        Ok(ExecutableGraph {
            backend,
            memory,
            device,
            steps,
            kernarg_allocs,
            fused_code_objects,
            fused_pairs,
            fusion_candidates,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn bake_kernel_dispatch_step(
    backend: &dyn DeviceBackend,
    memory: &MemoryPool,
    device: usize,
    kernel_handle: KernelHandle,
    metadata: &KernelMetadata,
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    shared_mem: u32,
    kernarg_bytes: &[u8],
    kernarg_allocs: &mut Vec<u64>,
) -> Result<BakedStep, RuntimeError> {
    let kernarg_ptr = if metadata.kernarg_segment_size > 0 {
        let ptr = memory.alloc(
            backend,
            device,
            metadata.kernarg_segment_size as usize,
            64,
        )?;
        if let Err(e) = backend.copy_htod(device, ptr, kernarg_bytes) {
            let _ = memory.free(backend, device, ptr);
            return Err(e.into());
        }
        kernarg_allocs.push(ptr);
        ptr
    } else {
        0
    };

    // Convert CUDA gridDim (blocks) into HSA grid_size (total work-items).
    let pkt = KernelDispatchPacket {
        header: encode_aql_header(
            AqlPacketType::KernelDispatch,
            true,
            FenceScope::System,
            FenceScope::System,
        ),
        setup: 3,
        workgroup_size_x: u16::try_from(block_dim[0])
            .map_err(|_| RuntimeError::InvalidArgument("block_dim.x exceeds u16".to_owned()))?,
        workgroup_size_y: u16::try_from(block_dim[1])
            .map_err(|_| RuntimeError::InvalidArgument("block_dim.y exceeds u16".to_owned()))?,
        workgroup_size_z: u16::try_from(block_dim[2])
            .map_err(|_| RuntimeError::InvalidArgument("block_dim.z exceeds u16".to_owned()))?,
        reserved0: 0,
        grid_size_x: grid_dim[0].saturating_mul(block_dim[0]),
        grid_size_y: grid_dim[1].saturating_mul(block_dim[1]),
        grid_size_z: grid_dim[2].saturating_mul(block_dim[2]),
        private_segment_size: metadata.private_segment_fixed_size,
        group_segment_size: shared_mem.max(metadata.group_segment_fixed_size),
        kernel_object: kernel_handle.0,
        kernarg_address: kernarg_ptr,
        reserved2: 0,
        completion_signal: 0,
    };
    Ok(BakedStep::AqlPacket(pkt.to_bytes()))
}

enum BakedStep {
    AqlPacket([u8; 64]),
    MemcpyDtoD { dst: u64, src: u64, len: usize },
}

/// Instantiated GPU execution graph holding pre-baked AQL packets ready for fast replay.
pub struct ExecutableGraph {
    backend: Arc<dyn DeviceBackend>,
    memory: Arc<MemoryPool>,
    device: usize,
    steps: Vec<BakedStep>,
    kernarg_allocs: Vec<u64>,
    fused_code_objects: Vec<CodeObjectId>,
    fused_pairs: usize,
    pub fusion_candidates: Vec<FusionCandidate>,
}

impl ExecutableGraph {
    /// Returns the number of pre-baked execution steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns how many producer-consumer kernel pairs were fused into single AQL packets.
    pub fn fused_pairs(&self) -> usize {
        self.fused_pairs
    }

    /// Replays all pre-baked AQL packets and device copies onto `stream`.
    pub fn launch(&self, stream: Stream) -> Result<(), RuntimeError> {
        for step in &self.steps {
            match step {
                BakedStep::AqlPacket(pkt) => {
                    self.backend.submit_aql_packet(stream.queue_id, pkt)?;
                }
                BakedStep::MemcpyDtoD { dst, src, len } => {
                    self.backend.copy_dtod(self.device, *dst, *src, *len)?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for ExecutableGraph {
    fn drop(&mut self) {
        for ptr in self.kernarg_allocs.drain(..) {
            let _ = self.memory.free(&*self.backend, self.device, ptr);
        }
        for obj in self.fused_code_objects.drain(..) {
            let _ = self.backend.unload_code_object(obj);
        }
    }
}
