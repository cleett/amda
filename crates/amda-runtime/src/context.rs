//! Core AMDA Runtime and Context managing devices, memory, modules, streams, and graphs.

use crate::error::RuntimeError;
use crate::graph::{ExecutableGraph, ExecutionGraph};
use crate::memory::{MemoryPool, MemoryStats, VmmRegistry};
use crate::module::{pack_kernargs, KernelFunction, LoadedModule};
use crate::stream::{Event, Stream};
use crate::trace::{format_hex_preview, trace_enabled, trace_log};
use amda_compiler::{promote_threshold_from_env, CompileTier, Compiler, IrModule};
use amda_hal::{
    encode_aql_header, try_open_kfd_backend, AqlPacketType, DeviceBackend, DeviceInfo,
    FenceScope, KernelDispatchPacket, KernelHandle, MockBackend, QueueId,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Cumulative tiered JIT compilation and hot-promotion counters for a [`Context`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierStats {
    pub tier0_compiles: u64,
    pub tier1_compiles: u64,
    pub hot_promotions: u64,
    pub total_launches: u64,
}

/// Top-level AMDA runtime owning the selected HAL backend and compiler driver.
pub struct Runtime {
    backend: Arc<dyn DeviceBackend>,
    compiler: Arc<Compiler>,
}

impl Runtime {
    /// Initializes the AMDA runtime according to `AMDA_BACKEND` (`auto`, `kfd`, or `mock`).
    pub fn new() -> Result<Self, RuntimeError> {
        let mode = std::env::var("AMDA_BACKEND")
            .unwrap_or_else(|_| "auto".to_owned())
            .to_ascii_lowercase();

        let backend: Arc<dyn DeviceBackend> = match mode.as_str() {
            "kfd" => Arc::from(try_open_kfd_backend()?),
            "mock" => Arc::new(MockBackend::from_env()),
            _ => match try_open_kfd_backend() {
                Ok(b) => Arc::from(b),
                Err(_) => Arc::new(MockBackend::from_env()),
            },
        };

        Ok(Self {
            backend,
            compiler: Arc::new(Compiler::new()),
        })
    }

    /// Creates a runtime explicitly backed by `backend`.
    pub fn with_backend(backend: Arc<dyn DeviceBackend>) -> Self {
        Self {
            backend,
            compiler: Arc::new(Compiler::new()),
        }
    }

    /// Returns all discovered GPU devices.
    pub fn devices(&self) -> &[DeviceInfo] {
        self.backend.devices()
    }

    /// Returns a reference to the underlying HAL backend.
    pub fn backend(&self) -> &Arc<dyn DeviceBackend> {
        &self.backend
    }

    /// Returns a reference to the compiler driver.
    pub fn compiler(&self) -> &Arc<Compiler> {
        &self.compiler
    }

    /// Creates an execution context bound to `device_index`.
    pub fn create_context(&self, device_index: usize) -> Result<Arc<Context>, RuntimeError> {
        let info = self
            .backend
            .devices()
            .get(device_index)
            .cloned()
            .ok_or(amda_hal::HalError::DeviceNotFound(device_index))?;
        let default_queue = self.backend.create_queue(device_index, 256)?;
        Ok(Arc::new(Context {
            device_index,
            device_info: info,
            backend: Arc::clone(&self.backend),
            compiler: Arc::clone(&self.compiler),
            memory: Arc::new(MemoryPool::new()),
            vmm: Mutex::new(VmmRegistry::new()),
            default_queue,
            kernel_tiers: Mutex::new(HashMap::new()),
            promote_threshold: AtomicU64::new(promote_threshold_from_env()),
            tier_stats: Mutex::new(TierStats::default()),
        }))
    }
}

/// Device-bound GPU execution context (analogous to `CUcontext`).
pub struct Context {
    device_index: usize,
    device_info: DeviceInfo,
    backend: Arc<dyn DeviceBackend>,
    compiler: Arc<Compiler>,
    memory: Arc<MemoryPool>,
    vmm: Mutex<VmmRegistry>,
    default_queue: QueueId,
    kernel_tiers: Mutex<HashMap<KernelHandle, (u64, CompileTier)>>,
    promote_threshold: AtomicU64,
    tier_stats: Mutex<TierStats>,
}

impl Context {
    /// Device index this context targets.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Device capabilities and ISA metadata.
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Underlying HAL backend.
    pub fn backend(&self) -> &Arc<dyn DeviceBackend> {
        &self.backend
    }

    /// Compiler instance associated with this context.
    pub fn compiler(&self) -> &Arc<Compiler> {
        &self.compiler
    }

    /// Default execution stream for this context.
    pub fn default_stream(&self) -> Stream {
        Stream {
            queue_id: self.default_queue,
            flags: 0,
        }
    }

    /// Allocates `size` bytes of GPU VRAM using the caching slab allocator.
    pub fn alloc(&self, size: usize) -> Result<u64, RuntimeError> {
        self.memory.alloc(&*self.backend, self.device_index, size, 256)
    }

    /// Frees a GPU VRAM buffer back into the slab pool.
    pub fn free(&self, ptr: u64) -> Result<(), RuntimeError> {
        self.memory.free(&*self.backend, self.device_index, ptr)
    }

    /// Returns slab memory pool statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.stats()
    }

    /// Checks whether `ptr` belongs to an active device allocation in this context.
    pub fn contains_device_ptr(&self, ptr: u64) -> Option<(u64, usize)> {
        self.memory.contains_ptr(ptr)
    }

    /// Returns cumulative Tier 0/1 compilation and hot-promotion counters.
    pub fn tier_stats(&self) -> TierStats {
        *self.tier_stats.lock().expect("tier stats lock")
    }

    /// Overrides the launch count threshold after which a Tier 0 kernel promotes to Tier 1.
    pub fn set_promote_threshold(&self, threshold: u64) {
        self.promote_threshold
            .store(threshold.max(1), Ordering::Relaxed);
    }

    /// Returns `(launch_count, active_tier)` for `handle`, if tracked.
    pub fn kernel_tier(&self, handle: KernelHandle) -> Option<(u64, CompileTier)> {
        self.kernel_tiers
            .lock()
            .expect("kernel tiers lock")
            .get(&handle)
            .copied()
    }

    /// Reserves a VMM virtual address range.
    pub fn vmm_reserve(&self, size: usize, align: usize) -> Result<u64, RuntimeError> {
        self.vmm
            .lock()
            .expect("vmm lock")
            .reserve_va(size, align)
    }

    /// Frees a VMM virtual address reservation.
    pub fn vmm_free(&self, va: u64) -> Result<(), RuntimeError> {
        self.vmm.lock().expect("vmm lock").free_va(va)
    }

    /// Copies bytes from host slice to GPU buffer.
    pub fn copy_htod(&self, dst: u64, src: &[u8]) -> Result<(), RuntimeError> {
        self.backend.copy_htod(self.device_index, dst, src)?;
        Ok(())
    }

    /// Copies bytes from GPU buffer into host slice.
    pub fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> Result<(), RuntimeError> {
        self.backend.copy_dtoh(self.device_index, dst, src)?;
        Ok(())
    }

    /// Copies `len` bytes between two GPU buffers.
    pub fn copy_dtod(&self, dst: u64, src: u64, len: usize) -> Result<(), RuntimeError> {
        self.backend.copy_dtod(self.device_index, dst, src, len)?;
        Ok(())
    }

    /// Compiles and loads a module from HSACO, fatbin, or PTX source bytes.
    pub fn load_module(&self, data: &[u8]) -> Result<LoadedModule, RuntimeError> {
        let t0 = trace_enabled().then(Instant::now);
        let module = LoadedModule::load(
            &*self.backend,
            &self.compiler,
            self.device_index,
            self.device_info.isa,
            data,
        )?;
        {
            let mut kt = self.kernel_tiers.lock().expect("kernel tiers lock");
            for kf in module.kernels.values() {
                kt.entry(kf.handle).or_insert((0, kf.tier));
            }
        }
        {
            let mut st = self.tier_stats.lock().expect("tier stats lock");
            match module.tier {
                CompileTier::Tier0Fast => st.tier0_compiles += 1,
                CompileTier::Tier1Optimizing => st.tier1_compiles += 1,
            }
        }
        if let Some(start) = t0 {
            let k_names: Vec<&str> = module.kernels.keys().map(String::as_str).collect();
            trace_log(
                "runtime",
                &format!(
                    "load_module dev={} isa={} tier={} input={}B kernels=[{}] dt_us={}",
                    self.device_index,
                    self.device_info.isa,
                    module.tier.as_str(),
                    data.len(),
                    k_names.join(","),
                    start.elapsed().as_micros()
                ),
            );
        }
        Ok(module)
    }

    /// Unloads a previously loaded module from the HAL backend.
    pub fn unload_module(&self, module: LoadedModule) -> Result<(), RuntimeError> {
        {
            let mut kt = self.kernel_tiers.lock().expect("kernel tiers lock");
            for kf in module.kernels.values() {
                kt.remove(&kf.handle);
            }
        }
        self.backend.unload_code_object(module.code_object)?;
        Ok(())
    }

    /// Creates a new stream backed by an independent AQL queue.
    pub fn create_stream(&self, flags: u32) -> Result<Stream, RuntimeError> {
        let queue_id = self.backend.create_queue(self.device_index, 256)?;
        Ok(Stream { queue_id, flags })
    }

    /// Destroys a stream.
    pub fn destroy_stream(&self, stream: Stream) -> Result<(), RuntimeError> {
        if stream.queue_id != self.default_queue {
            self.backend.destroy_queue(stream.queue_id)?;
        }
        Ok(())
    }

    /// Synchronizes all prior work on `stream`.
    pub fn synchronize_stream(&self, stream: Stream) -> Result<(), RuntimeError> {
        stream.synchronize(&*self.backend)
    }

    /// Synchronizes all work on the context's default stream.
    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.default_stream().synchronize(&*self.backend)
    }

    /// Creates an HSA signal-backed event.
    pub fn create_event(&self, flags: u32) -> Result<Event, RuntimeError> {
        let sig = self.backend.create_signal(0)?;
        Ok(Event::new(sig, flags))
    }

    /// Destroys an event.
    pub fn destroy_event(&self, event: Event) -> Result<(), RuntimeError> {
        self.backend.destroy_signal(event.signal_id)?;
        Ok(())
    }

    /// Records `event` on `stream`.
    pub fn record_event(&self, event: &Event, stream: Option<Stream>) -> Result<(), RuntimeError> {
        event.record(&*self.backend, stream.unwrap_or_else(|| self.default_stream()))
    }

    /// Waits for `event` completion.
    pub fn synchronize_event(&self, event: &Event) -> Result<(), RuntimeError> {
        event.synchronize(&*self.backend)
    }

    /// Non-blocking check whether `event` has completed.
    pub fn query_event(&self, event: &Event) -> Result<bool, RuntimeError> {
        event.query(&*self.backend)
    }

    /// Makes `stream` (or the context's default stream) wait for `event`.
    pub fn stream_wait_event(
        &self,
        stream: Option<Stream>,
        event: &Event,
    ) -> Result<(), RuntimeError> {
        let target = stream.unwrap_or_else(|| self.default_stream());
        target.wait_event(&*self.backend, event)
    }

    /// Launches a kernel onto `stream` (or the default stream), converting CUDA `grid_dim`
    /// (in blocks) to HSA `grid_size` (total work-items = `grid_dim * block_dim`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kernel(
        &self,
        func: &KernelFunction,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem: u32,
        stream: Option<Stream>,
        arg_slices: &[&[u8]],
    ) -> Result<(), RuntimeError> {
        if grid_dim.contains(&0) || block_dim.contains(&0) {
            return Err(RuntimeError::InvalidArgument(
                "grid and block dimensions must be >= 1".to_owned(),
            ));
        }

        let target_stream = stream.unwrap_or_else(|| self.default_stream());
        let kernarg_bytes = pack_kernargs(&func.metadata, arg_slices)?;
        let kernarg_ptr = if func.metadata.kernarg_segment_size > 0 {
            let ptr = self.memory.alloc(
                &*self.backend,
                self.device_index,
                func.metadata.kernarg_segment_size as usize,
                64,
            )?;
            if let Err(e) = self.backend.copy_htod(self.device_index, ptr, &kernarg_bytes) {
                let _ = self.memory.free(&*self.backend, self.device_index, ptr);
                return Err(e.into());
            }
            ptr
        } else {
            0
        };

        let sig = self.backend.create_signal(1)?;
        let setup_dims = if grid_dim[2] > 1 || block_dim[2] > 1 {
            3
        } else if grid_dim[1] > 1 || block_dim[1] > 1 {
            2
        } else {
            1
        };

        let pkt = KernelDispatchPacket {
            header: encode_aql_header(
                AqlPacketType::KernelDispatch,
                true,
                FenceScope::System,
                FenceScope::System,
            ),
            setup: setup_dims,
            workgroup_size_x: u16::try_from(block_dim[0]).map_err(|_| {
                RuntimeError::InvalidArgument("block_dim.x exceeds u16::MAX".to_owned())
            })?,
            workgroup_size_y: u16::try_from(block_dim[1]).map_err(|_| {
                RuntimeError::InvalidArgument("block_dim.y exceeds u16::MAX".to_owned())
            })?,
            workgroup_size_z: u16::try_from(block_dim[2]).map_err(|_| {
                RuntimeError::InvalidArgument("block_dim.z exceeds u16::MAX".to_owned())
            })?,
            reserved0: 0,
            grid_size_x: grid_dim[0]
                .checked_mul(block_dim[0])
                .ok_or_else(|| RuntimeError::InvalidArgument("grid_size_x overflow".to_owned()))?,
            grid_size_y: grid_dim[1]
                .checked_mul(block_dim[1])
                .ok_or_else(|| RuntimeError::InvalidArgument("grid_size_y overflow".to_owned()))?,
            grid_size_z: grid_dim[2]
                .checked_mul(block_dim[2])
                .ok_or_else(|| RuntimeError::InvalidArgument("grid_size_z overflow".to_owned()))?,
            private_segment_size: func.metadata.private_segment_fixed_size,
            group_segment_size: shared_mem.max(func.metadata.group_segment_fixed_size),
            kernel_object: func.handle.0,
            kernarg_address: kernarg_ptr,
            reserved2: 0,
            completion_signal: sig.0,
        };

        let t0 = trace_enabled().then(Instant::now);
        let submit_res = self
            .backend
            .submit_aql_packet(target_stream.queue_id, &pkt.to_bytes());
        let wait_res = if submit_res.is_ok() {
            self.backend.signal_wait(sig, 0)
        } else {
            Ok(())
        };

        let _ = self.backend.destroy_signal(sig);
        if kernarg_ptr != 0 {
            let _ = self.memory.free(&*self.backend, self.device_index, kernarg_ptr);
        }
        submit_res?;
        wait_res?;

        if let Some(start) = t0 {
            trace_log(
                "runtime",
                &format!(
                    "launch_kernel name={} grid={:?} block={:?} lds={}B kernarg=[{}] wait_us={}",
                    func.name,
                    grid_dim,
                    block_dim,
                    pkt.group_segment_size,
                    format_hex_preview(&kernarg_bytes, 24),
                    start.elapsed().as_micros()
                ),
            );
        }

        self.record_launch_and_maybe_promote(func)?;
        Ok(())
    }

    fn record_launch_and_maybe_promote(&self, func: &KernelFunction) -> Result<(), RuntimeError> {
        let threshold = self.promote_threshold.load(Ordering::Relaxed);
        let should_promote = {
            let mut kt = self.kernel_tiers.lock().expect("kernel tiers lock");
            let entry = kt.entry(func.handle).or_insert((0, func.tier));
            entry.0 += 1;
            entry.1 == CompileTier::Tier0Fast && entry.0 >= threshold
        };

        {
            let mut st = self.tier_stats.lock().expect("tier stats lock");
            st.total_launches += 1;
        }

        if should_promote {
            if let Some(ir_kernel) = &func.ir_kernel {
                let single_mod = IrModule {
                    target: self.device_info.isa,
                    kernels: vec![(**ir_kernel).clone()],
                };
                let _promoted = self.compiler.compile_ir_at_tier(
                    &single_mod,
                    self.device_info.isa,
                    CompileTier::Tier1Optimizing,
                )?;
                let mut kt = self.kernel_tiers.lock().expect("kernel tiers lock");
                if let Some(entry) = kt.get_mut(&func.handle) {
                    entry.1 = CompileTier::Tier1Optimizing;
                }
                let mut st = self.tier_stats.lock().expect("tier stats lock");
                st.hot_promotions += 1;
                st.tier1_compiles += 1;
            }
        }
        Ok(())
    }

    /// Instantiates an [`ExecutionGraph`] into an [`ExecutableGraph`] tied to this context,
    /// automatically applying cross-kernel IR fusion unless `AMDA_DISABLE_FUSION=1`.
    pub fn instantiate_graph(&self, graph: &ExecutionGraph) -> Result<ExecutableGraph, RuntimeError> {
        let enable_fusion = std::env::var("AMDA_DISABLE_FUSION").as_deref() != Ok("1");
        self.instantiate_graph_with_fusion(graph, enable_fusion)
    }

    /// Instantiates an [`ExecutionGraph`] with explicit control over cross-kernel fusion.
    pub fn instantiate_graph_with_fusion(
        &self,
        graph: &ExecutionGraph,
        enable_fusion: bool,
    ) -> Result<ExecutableGraph, RuntimeError> {
        let exec = graph.instantiate_with_compiler(
            Arc::clone(&self.backend),
            Arc::clone(&self.memory),
            &self.compiler,
            self.device_info.isa,
            self.device_index,
            enable_fusion,
        )?;
        if exec.fused_pairs() > 0 || !exec.fusion_candidates.is_empty() {
            let mut st = self.tier_stats.lock().expect("tier stats lock");
            st.hot_promotions += exec.fused_pairs().max(1) as u64;
            st.tier1_compiles += exec.fused_pairs().max(1) as u64;
        }
        Ok(exec)
    }

    /// Replays an [`ExecutableGraph`] on `stream` (or the context's default stream).
    pub fn launch_graph(
        &self,
        exec: &ExecutableGraph,
        stream: Option<Stream>,
    ) -> Result<(), RuntimeError> {
        let t0 = trace_enabled().then(Instant::now);
        let res = exec.launch(stream.unwrap_or_else(|| self.default_stream()));
        if let Some(start) = t0 {
            trace_log(
                "runtime",
                &format!(
                    "launch_graph steps={} status={} dt_us={}",
                    exec.step_count(),
                    if res.is_ok() { "ok" } else { "err" },
                    start.elapsed().as_micros()
                ),
            );
        }
        res
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _ = self.memory.trim(&*self.backend, self.device_index);
        let _ = self.backend.destroy_queue(self.default_queue);
    }
}
