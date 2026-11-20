//! In-process Virtual AMD GPU (`MockBackend`) with a bounds-checked Virtual MMU,
//! real AQL ring buffer processing, and a cooperative AMDA IR Command Processor.

use crate::aql::{
    decode_aql_header, AqlPacketType, AqlRingBuffer, BarrierAndPacket, KernelDispatchPacket,
};
use crate::backend::{CodeObjectId, DeviceBackend, KernelHandle, QueueId, SignalId};
use crate::device::{DeviceInfo, HardwareFeatures};
use crate::error::HalError;
use amda_compiler::{
    AddressSpace, AtomicOp, BinOp, CmpOp, IrInst, IrKernel, IrModule, IrOp, IrOperand, IrType,
    ShuffleMode, SpecialReg, UnaryMathOp, VoteMode,
};
use amda_elf::{parse_hsaco, IsaTarget, KernelMetadata};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

const VIRTUAL_VRAM_BASE: u64 = 0x0000_1000_0000_0000;
const VIRTUAL_GUARD_GAP: u64 = 4096;
const MAX_THREAD_STEPS: usize = 2_000_000;

/// Diagnostic execution counters recorded by [`MockBackend`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MockStats {
    pub packets_processed: u64,
    pub kernels_dispatched: u64,
    pub barriers_processed: u64,
    pub bytes_copied_htod: u64,
    pub bytes_copied_dtoh: u64,
    pub bytes_copied_dtod: u64,
}

#[derive(Default)]
struct VirtualMmu {
    next_va: u64,
    regions: BTreeMap<u64, Vec<u8>>,
}

impl VirtualMmu {
    fn new() -> Self {
        Self {
            next_va: VIRTUAL_VRAM_BASE,
            regions: BTreeMap::new(),
        }
    }

    fn alloc(&mut self, size: usize, align: usize) -> Result<u64, HalError> {
        let actual_size = size.max(1);
        let align = align.max(256).next_power_of_two() as u64;
        let va = (self.next_va + align - 1) & !(align - 1);
        let next = va
            .checked_add(actual_size as u64)
            .and_then(|v| v.checked_add(VIRTUAL_GUARD_GAP))
            .ok_or_else(|| HalError::OutOfMemory("virtual address space exhausted".to_owned()))?;
        self.regions.insert(va, vec![0u8; actual_size]);
        self.next_va = next;
        Ok(va)
    }

    fn free(&mut self, ptr: u64) -> Result<(), HalError> {
        if self.regions.remove(&ptr).is_none() {
            return Err(HalError::InvalidHandle("VRAM pointer"));
        }
        Ok(())
    }

    fn lookup_slice(&self, addr: u64, len: usize) -> Result<&[u8], HalError> {
        let (&base, buf) = self
            .regions
            .range(..=addr)
            .next_back()
            .ok_or_else(|| HalError::MemoryAccessFault {
                addr,
                size: len,
                reason: "unmapped virtual GPU address".to_owned(),
            })?;

        let offset = (addr - base) as usize;
        let end = offset.checked_add(len).ok_or_else(|| HalError::MemoryAccessFault {
            addr,
            size: len,
            reason: "address offset overflow".to_owned(),
        })?;
        if end > buf.len() {
            return Err(HalError::MemoryAccessFault {
                addr,
                size: len,
                reason: format!(
                    "read exceeds allocation [0x{base:016x}..0x{:016x})",
                    base + buf.len() as u64
                ),
            });
        }
        Ok(&buf[offset..end])
    }

    fn lookup_slice_mut(&mut self, addr: u64, len: usize) -> Result<&mut [u8], HalError> {
        let (&base, buf) = self
            .regions
            .range_mut(..=addr)
            .next_back()
            .ok_or_else(|| HalError::MemoryAccessFault {
                addr,
                size: len,
                reason: "unmapped virtual GPU address".to_owned(),
            })?;

        let offset = (addr - base) as usize;
        let end = offset.checked_add(len).ok_or_else(|| HalError::MemoryAccessFault {
            addr,
            size: len,
            reason: "address offset overflow".to_owned(),
        })?;
        let region_len = buf.len();
        if end > region_len {
            return Err(HalError::MemoryAccessFault {
                addr,
                size: len,
                reason: format!(
                    "write exceeds allocation [0x{base:016x}..0x{:016x})",
                    base + region_len as u64
                ),
            });
        }
        Ok(&mut buf[offset..end])
    }

    fn copy_htod(&mut self, dst: u64, src: &[u8]) -> Result<(), HalError> {
        let slice = self.lookup_slice_mut(dst, src.len())?;
        slice.copy_from_slice(src);
        Ok(())
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> Result<(), HalError> {
        let slice = self.lookup_slice(src, dst.len())?;
        dst.copy_from_slice(slice);
        Ok(())
    }

    fn copy_dtod(&mut self, dst: u64, src: u64, len: usize) -> Result<(), HalError> {
        let tmp = self.lookup_slice(src, len)?.to_vec();
        let dst_slice = self.lookup_slice_mut(dst, len)?;
        dst_slice.copy_from_slice(&tmp);
        Ok(())
    }
}

struct LoadedKernel {
    device: usize,
    ir: Arc<IrKernel>,
}

struct MockQueue {
    device: usize,
    ring: AqlRingBuffer,
}

/// In-process Virtual AMD GPU backend executing real AQL packets and lowered AMDA IR kernels.
pub struct MockBackend {
    devices: Vec<DeviceInfo>,
    mmu: RwLock<VirtualMmu>,
    signals: RwLock<HashMap<u64, Arc<AtomicI64>>>,
    queues: RwLock<HashMap<u64, Arc<MockQueue>>>,
    code_objects: RwLock<HashMap<u64, Vec<u64>>>,
    kernels: RwLock<HashMap<u64, LoadedKernel>>,
    next_id: AtomicU64,
    stats: Mutex<MockStats>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MockBackend {
    /// Instantiates a Virtual AMD GPU honoring `AMDA_MOCK_ARCH` (`gfx1100`, `gfx942`, etc.).
    pub fn from_env() -> Self {
        let isa = std::env::var("AMDA_MOCK_ARCH")
            .ok()
            .and_then(|s| IsaTarget::from_str(&s).ok())
            .unwrap_or(IsaTarget::Gfx1100);
        Self::with_isa(isa)
    }

    /// Instantiates a Virtual AMD GPU modeling a specific AMDGPU ISA generation.
    pub fn with_isa(isa: IsaTarget) -> Self {
        let (name, cus, vram_gb, unified) = match isa {
            IsaTarget::Gfx942 => ("AMD Instinct MI300X (Virtual AMDA GPU)", 304, 192, false),
            IsaTarget::Gfx90a => ("AMD Instinct MI250X (Virtual AMDA GPU)", 110, 64, false),
            IsaTarget::Gfx908 => ("AMD Instinct MI100 (Virtual AMDA GPU)", 120, 32, false),
            IsaTarget::Gfx1201 => ("AMD Radeon RX 9070 XT (Virtual AMDA GPU)", 64, 16, false),
            IsaTarget::Gfx1200 => ("AMD Radeon RDNA4 (Virtual AMDA GPU)", 56, 16, false),
            IsaTarget::Gfx1100 => ("AMD Radeon RX 7900 XTX (Virtual AMDA GPU)", 96, 24, false),
            IsaTarget::Gfx1101 => ("AMD Radeon RX 7800 XT (Virtual AMDA GPU)", 60, 16, false),
            IsaTarget::Gfx1102 => ("AMD Radeon RX 7600 (Virtual AMDA GPU)", 32, 8, false),
            IsaTarget::Gfx1103 => ("AMD Radeon 780M APU (Virtual AMDA GPU)", 12, 16, true),
            IsaTarget::Gfx1030 => ("AMD Radeon RX 6900 XT (Virtual AMDA GPU)", 80, 16, false),
        };

        let info = DeviceInfo {
            index: 0,
            name: name.to_owned(),
            isa,
            vram_bytes: vram_gb * 1024 * 1024 * 1024,
            features: HardwareFeatures::from_isa(isa, cus, unified),
            backend_name: "mock-virtual-gpu",
        };

        Self {
            devices: vec![info],
            mmu: RwLock::new(VirtualMmu::new()),
            signals: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            code_objects: RwLock::new(HashMap::new()),
            kernels: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            stats: Mutex::new(MockStats::default()),
        }
    }

    /// Returns diagnostic counters for virtual GPU activity.
    pub fn stats(&self) -> MockStats {
        *self.stats.lock().expect("mock stats lock")
    }

    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn check_device(&self, device: usize) -> Result<(), HalError> {
        if device < self.devices.len() {
            Ok(())
        } else {
            Err(HalError::DeviceNotFound(device))
        }
    }

    fn ring_doorbell(&self, queue: &MockQueue) -> Result<(), HalError> {
        while let Some(raw_pkt) = queue.ring.pop_packet() {
            self.execute_aql_packet(queue.device, &raw_pkt)?;
        }
        Ok(())
    }

    fn execute_aql_packet(&self, _device: usize, raw: &[u8; 64]) -> Result<(), HalError> {
        let (pkt_type, _, _, _) = decode_aql_header(u16::from_le_bytes([raw[0], raw[1]]));

        if let Ok(mut st) = self.stats.lock() {
            st.packets_processed += 1;
        }

        if pkt_type == AqlPacketType::KernelDispatch as u8 {
            let pkt = KernelDispatchPacket::from_bytes(raw);
            if let Ok(mut st) = self.stats.lock() {
                st.kernels_dispatched += 1;
            }
            self.dispatch_kernel_packet(&pkt)?;
            if pkt.completion_signal != 0 {
                self.complete_signal(pkt.completion_signal)?;
            }
            Ok(())
        } else if pkt_type == AqlPacketType::BarrierAnd as u8 {
            let pkt = BarrierAndPacket::from_bytes(raw);
            if let Ok(mut st) = self.stats.lock() {
                st.barriers_processed += 1;
            }
            for dep in pkt.dep_signal {
                if dep != 0 {
                    self.signal_wait(SignalId(dep), 0)?;
                }
            }
            if pkt.completion_signal != 0 {
                self.complete_signal(pkt.completion_signal)?;
            }
            Ok(())
        } else {
            Err(HalError::ExecutionError(format!(
                "unsupported AQL packet type {pkt_type}"
            )))
        }
    }

    fn complete_signal(&self, sig_id: u64) -> Result<(), HalError> {
        let sig = {
            let guard = self.signals.read().expect("signals lock");
            guard.get(&sig_id).cloned()
        };
        if let Some(atom) = sig {
            atom.store(0, Ordering::Release);
            Ok(())
        } else {
            Err(HalError::InvalidHandle("completion signal"))
        }
    }

    fn dispatch_kernel_packet(&self, pkt: &KernelDispatchPacket) -> Result<(), HalError> {
        let kernel = {
            let guard = self.kernels.read().expect("kernels lock");
            guard
                .get(&pkt.kernel_object)
                .map(|lk| (lk.device, Arc::clone(&lk.ir)))
                .ok_or(HalError::InvalidHandle("kernel object"))?
        };
        let (_dev, ir) = kernel;

        let wg_x = u32::from(pkt.workgroup_size_x);
        let wg_y = u32::from(pkt.workgroup_size_y);
        let wg_z = u32::from(pkt.workgroup_size_z);
        if wg_x == 0 || wg_y == 0 || wg_z == 0 {
            return Err(HalError::ExecutionError(
                "AQL KernelDispatchPacket workgroup dimensions must be >= 1".to_owned(),
            ));
        }

        let num_wg_x = pkt.grid_size_x.div_ceil(wg_x);
        let num_wg_y = pkt.grid_size_y.div_ceil(wg_y);
        let num_wg_z = pkt.grid_size_z.div_ceil(wg_z);

        let kernarg_bytes = {
            let mmu = self.mmu.read().expect("mmu read lock");
            if ir.metadata.kernarg_segment_size > 0 {
                mmu.lookup_slice(pkt.kernarg_address, ir.metadata.kernarg_segment_size as usize)?
                    .to_vec()
            } else {
                Vec::new()
            }
        };

        let mut label_pcs = HashMap::new();
        for (idx, inst) in ir.insts.iter().enumerate() {
            if let IrOp::Label(name) = &inst.op {
                label_pcs.insert(name.as_str(), idx);
            }
        }

        let lds_size = (pkt.group_segment_size.max(ir.shared_bytes)) as usize;
        let grid_dims = [num_wg_x, num_wg_y, num_wg_z];
        let block_dims = [wg_x, wg_y, wg_z];

        for gz in 0..num_wg_z {
            for gy in 0..num_wg_y {
                for gx in 0..num_wg_x {
                    self.execute_workgroup(
                        &ir,
                        &label_pcs,
                        &kernarg_bytes,
                        lds_size,
                        [gx, gy, gz],
                        block_dims,
                        grid_dims,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_workgroup(
        &self,
        kernel: &IrKernel,
        label_pcs: &HashMap<&str, usize>,
        kernarg_bytes: &[u8],
        lds_size: usize,
        cta_id: [u32; 3],
        block_dims: [u32; 3],
        grid_dims: [u32; 3],
    ) -> Result<(), HalError> {
        let mut lds = vec![0u8; lds_size];
        let total_threads = (block_dims[0] * block_dims[1] * block_dims[2]) as usize;
        let num_regs = kernel.reg_types.len();

        let mut threads = Vec::with_capacity(total_threads);
        for tz in 0..block_dims[2] {
            for ty in 0..block_dims[1] {
                for tx in 0..block_dims[0] {
                    let flat_tid = (tz * block_dims[1] + ty) * block_dims[0] + tx;
                    threads.push(ThreadState {
                        tid: [tx, ty, tz],
                        lane_id: flat_tid % 32,
                        pc: 0,
                        regs: vec![0u64; num_regs],
                        private_mem: vec![0u8; kernel.metadata.private_segment_fixed_size as usize],
                        waiting: WaitKind::Running,
                        steps: 0,
                    });
                }
            }
        }

        loop {
            let mut any_running = false;
            for thread in &mut threads {
                if thread.waiting != WaitKind::Running {
                    continue;
                }
                any_running = true;
                while thread.waiting == WaitKind::Running {
                    if thread.pc >= kernel.insts.len() {
                        thread.waiting = WaitKind::Finished;
                        break;
                    }
                    thread.steps += 1;
                    if thread.steps > MAX_THREAD_STEPS {
                        return Err(HalError::ExecutionError(format!(
                            "kernel '{}' exceeded max instruction step budget (infinite loop?)",
                            kernel.name
                        )));
                    }

                    let inst = &kernel.insts[thread.pc];
                    if !eval_pred(&inst.pred, &thread.regs) {
                        thread.pc += 1;
                        continue;
                    }

                    match &inst.op {
                        IrOp::Barrier => {
                            thread.waiting = WaitKind::Barrier;
                            break;
                        }
                        IrOp::WarpShuffle { .. } => {
                            thread.waiting = WaitKind::Shuffle;
                            break;
                        }
                        IrOp::WarpVote { .. } => {
                            thread.waiting = WaitKind::Vote;
                            break;
                        }
                        IrOp::Return => {
                            thread.waiting = WaitKind::Finished;
                            break;
                        }
                        _ => {
                            let next_pc = self.exec_thread_inst(
                                kernel,
                                inst,
                                thread,
                                label_pcs,
                                kernarg_bytes,
                                &mut lds,
                                cta_id,
                                block_dims,
                                grid_dims,
                            )?;
                            thread.pc = next_pc;
                        }
                    }
                }
            }

            if !any_running {
                // Either all finished, or waiting at Barrier / Shuffle / Vote
                let any_shuffle = threads.iter().any(|t| t.waiting == WaitKind::Shuffle);
                if any_shuffle {
                    // Snapshot pre-shuffle values across all threads currently at Shuffle
                    type ShuffleEntry = (u32, ShuffleMode, u64, u32, u32);
                    let mut shuf_vals: Vec<Option<ShuffleEntry>> = vec![None; threads.len()];
                    for (idx, t) in threads.iter().enumerate() {
                        if t.waiting == WaitKind::Shuffle {
                            if let IrOp::WarpShuffle {
                                dst,
                                mode,
                                src,
                                lane_delta,
                                width,
                            } = &kernel.insts[t.pc].op
                            {
                                let val = eval_operand_u64(
                                    src,
                                    t,
                                    cta_id,
                                    block_dims,
                                    grid_dims,
                                );
                                let delta = eval_operand_u64(
                                    lane_delta,
                                    t,
                                    cta_id,
                                    block_dims,
                                    grid_dims,
                                ) as u32;
                                shuf_vals[idx] = Some((*dst, *mode, val, delta, (*width).clamp(1, 32)));
                            }
                        }
                    }

                    for idx in 0..threads.len() {
                        if let Some((dst, mode, self_val, delta, width)) = shuf_vals[idx] {
                            let warp_base = (idx / 32) * 32;
                            let lane = idx % 32;
                            let d = delta as usize;
                            let w = width as usize;
                            let src_lane_opt = match mode {
                                ShuffleMode::Down => {
                                    let target = lane + d;
                                    if target < w { Some(target) } else { None }
                                }
                                ShuffleMode::Up => {
                                    if lane >= d { Some(lane - d) } else { None }
                                }
                                ShuffleMode::Bfly => {
                                    let target = lane ^ d;
                                    if target < w { Some(target) } else { None }
                                }
                                ShuffleMode::Idx => {
                                    let target = d & 0x1F;
                                    if target < w { Some(target) } else { None }
                                }
                            };
                            let picked = src_lane_opt
                                .and_then(|sl| {
                                    shuf_vals
                                        .get(warp_base + sl)
                                        .and_then(|opt| opt.map(|(_, _, v, _, _)| v))
                                })
                                .unwrap_or(self_val);
                            threads[idx].regs[dst as usize] = picked;
                            threads[idx].pc += 1;
                            threads[idx].waiting = WaitKind::Running;
                        }
                    }
                    continue;
                }

                let any_vote = threads.iter().any(|t| t.waiting == WaitKind::Vote);
                if any_vote {
                    let num_warps = threads.len().div_ceil(32);
                    let mut ballot_masks = vec![0u32; num_warps];
                    let mut active_masks = vec![0u32; num_warps];
                    let mut vote_info: Vec<Option<(VoteMode, u32)>> = vec![None; threads.len()];

                    for (idx, t) in threads.iter().enumerate() {
                        if t.waiting == WaitKind::Vote {
                            if let IrOp::WarpVote {
                                mode,
                                dst,
                                pred,
                                negated,
                            } = &kernel.insts[t.pc].op
                            {
                                let warp_id = idx / 32;
                                let lane = idx % 32;
                                active_masks[warp_id] |= 1u32 << lane;
                                let mut pval = (t.regs[*pred as usize] & 1) != 0;
                                if *negated {
                                    pval = !pval;
                                }
                                if pval {
                                    ballot_masks[warp_id] |= 1u32 << lane;
                                }
                                vote_info[idx] = Some((*mode, *dst));
                            }
                        }
                    }

                    for idx in 0..threads.len() {
                        if let Some((mode, dst)) = vote_info[idx] {
                            let warp_id = idx / 32;
                            let bmask = ballot_masks[warp_id];
                            let amask = active_masks[warp_id];
                            let res = match mode {
                                VoteMode::Ballot => u64::from(bmask),
                                VoteMode::Any => u64::from(bmask != 0),
                                VoteMode::All => u64::from(amask != 0 && (bmask & amask) == amask),
                            };
                            threads[idx].regs[dst as usize] = res;
                            threads[idx].pc += 1;
                            threads[idx].waiting = WaitKind::Running;
                        }
                    }
                    continue;
                }

                let any_barrier = threads.iter().any(|t| t.waiting == WaitKind::Barrier);
                if any_barrier {
                    for t in &mut threads {
                        if t.waiting == WaitKind::Barrier {
                            t.pc += 1;
                            t.waiting = WaitKind::Running;
                        }
                    }
                    continue;
                }

                break;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_thread_inst(
        &self,
        kernel: &IrKernel,
        inst: &IrInst,
        t: &mut ThreadState,
        label_pcs: &HashMap<&str, usize>,
        kernarg_bytes: &[u8],
        lds: &mut [u8],
        cta_id: [u32; 3],
        block_dims: [u32; 3],
        grid_dims: [u32; 3],
    ) -> Result<usize, HalError> {
        let mut next_pc = t.pc + 1;
        match &inst.op {
            IrOp::Label(_) => {}
            IrOp::LoadParam {
                dst,
                ty,
                param_idx,
                offset,
            } => {
                let p = kernel
                    .params
                    .get(*param_idx as usize)
                    .ok_or_else(|| HalError::ExecutionError("param index out of range".to_owned()))?;
                let start = (p.offset + *offset) as usize;
                let size = ty.byte_size() as usize;
                let slice = kernarg_bytes.get(start..start + size).ok_or_else(|| {
                    HalError::ExecutionError("kernarg read out of bounds".to_owned())
                })?;
                t.regs[*dst as usize] = decode_le_bits(slice);
            }
            IrOp::Load {
                dst,
                ty,
                space,
                base,
                offset,
            } => {
                let base_addr = eval_operand_u64(base, t, cta_id, block_dims, grid_dims);
                let addr = base_addr.wrapping_add_signed(*offset);
                let size = ty.byte_size() as usize;
                let val = if *space == AddressSpace::Shared {
                    let start = addr as usize;
                    let slice = lds.get(start..start + size).ok_or_else(|| {
                        HalError::MemoryAccessFault {
                            addr,
                            size,
                            reason: "LDS shared load out of bounds".to_owned(),
                        }
                    })?;
                    decode_le_bits(slice)
                } else if *space == AddressSpace::Private {
                    let start = addr as usize;
                    let slice = t.private_mem.get(start..start + size).ok_or_else(|| {
                        HalError::MemoryAccessFault {
                            addr,
                            size,
                            reason: "private scratch load out of bounds".to_owned(),
                        }
                    })?;
                    decode_le_bits(slice)
                } else {
                    let mmu = self.mmu.read().expect("mmu read lock");
                    let slice = mmu.lookup_slice(addr, size)?;
                    decode_le_bits(slice)
                };
                t.regs[*dst as usize] = val;
            }
            IrOp::Store {
                ty,
                space,
                base,
                offset,
                src,
            } => {
                let base_addr = eval_operand_u64(base, t, cta_id, block_dims, grid_dims);
                let addr = base_addr.wrapping_add_signed(*offset);
                let size = ty.byte_size() as usize;
                let bits = eval_operand_typed_bits(src, *ty, t, cta_id, block_dims, grid_dims);
                let raw = encode_le_bits(bits, size);
                if *space == AddressSpace::Shared {
                    let start = addr as usize;
                    let slice = lds.get_mut(start..start + size).ok_or_else(|| {
                        HalError::MemoryAccessFault {
                            addr,
                            size,
                            reason: "LDS shared store out of bounds".to_owned(),
                        }
                    })?;
                    slice.copy_from_slice(&raw[..size]);
                } else if *space == AddressSpace::Private {
                    let start = addr as usize;
                    let slice = t.private_mem.get_mut(start..start + size).ok_or_else(|| {
                        HalError::MemoryAccessFault {
                            addr,
                            size,
                            reason: "private scratch store out of bounds".to_owned(),
                        }
                    })?;
                    slice.copy_from_slice(&raw[..size]);
                } else {
                    let mut mmu = self.mmu.write().expect("mmu write lock");
                    let slice = mmu.lookup_slice_mut(addr, size)?;
                    slice.copy_from_slice(&raw[..size]);
                }
            }
            IrOp::CvtaToGlobal { dst, src } | IrOp::CvtaToShared { dst, src } => {
                t.regs[*dst as usize] = eval_operand_u64(src, t, cta_id, block_dims, grid_dims);
            }
            IrOp::Mov { dst, ty, src } => {
                t.regs[*dst as usize] =
                    eval_operand_typed_bits(src, *ty, t, cta_id, block_dims, grid_dims);
            }
            IrOp::Cvt {
                dst,
                dst_ty,
                src_ty,
                src,
            } => {
                let src_bits =
                    eval_operand_typed_bits(src, *src_ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = convert_value(*src_ty, *dst_ty, src_bits);
            }
            IrOp::Neg { dst, ty, src } => {
                let bits = eval_operand_typed_bits(src, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = match ty {
                    IrType::F32 => u64::from((-f32::from_bits(bits as u32)).to_bits()),
                    IrType::F64 => (-f64::from_bits(bits)).to_bits(),
                    IrType::I32 | IrType::U32 => u64::from((-(bits as i32)) as u32),
                    _ => (-(bits as i64)) as u64,
                };
            }
            IrOp::Abs { dst, ty, src } => {
                let bits = eval_operand_typed_bits(src, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = match ty {
                    IrType::F32 => u64::from(f32::from_bits(bits as u32).abs().to_bits()),
                    IrType::F64 => f64::from_bits(bits).abs().to_bits(),
                    IrType::I32 => u64::from((bits as i32).wrapping_abs() as u32),
                    IrType::I64 => (bits as i64).wrapping_abs() as u64,
                    _ => bits,
                };
            }
            IrOp::Bin {
                op,
                dst,
                ty,
                lhs,
                rhs,
            } => {
                let l = eval_operand_typed_bits(lhs, *ty, t, cta_id, block_dims, grid_dims);
                let r = eval_operand_typed_bits(rhs, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = eval_binop(*op, *ty, l, r)?;
            }
            IrOp::Mad { dst, ty, a, b, c } => {
                let va = eval_operand_typed_bits(a, *ty, t, cta_id, block_dims, grid_dims);
                let vb = eval_operand_typed_bits(b, *ty, t, cta_id, block_dims, grid_dims);
                let vc = eval_operand_typed_bits(c, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = eval_mad(*ty, va, vb, vc);
            }
            IrOp::Setp {
                cmp,
                dst,
                ty,
                lhs,
                rhs,
            } => {
                let l = eval_operand_typed_bits(lhs, *ty, t, cta_id, block_dims, grid_dims);
                let r = eval_operand_typed_bits(rhs, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = u64::from(eval_cmp(*cmp, *ty, l, r));
            }
            IrOp::Selp {
                dst,
                ty,
                a,
                b,
                pred,
            } => {
                let cond = (t.regs[*pred as usize] & 1) != 0;
                let chosen = if cond { a } else { b };
                t.regs[*dst as usize] =
                    eval_operand_typed_bits(chosen, *ty, t, cta_id, block_dims, grid_dims);
            }
            IrOp::UnaryMath { op, dst, ty, src } => {
                let bits = eval_operand_typed_bits(src, *ty, t, cta_id, block_dims, grid_dims);
                t.regs[*dst as usize] = eval_unary_math(*op, *ty, bits);
            }
            IrOp::Bfe {
                dst,
                signed,
                src,
                pos,
                len,
            } => {
                let s = eval_operand_u64(src, t, cta_id, block_dims, grid_dims) as u32;
                let p = (eval_operand_u64(pos, t, cta_id, block_dims, grid_dims) as u32) & 0xFF;
                let l = (eval_operand_u64(len, t, cta_id, block_dims, grid_dims) as u32) & 0xFF;
                let res = eval_bfe(s, p, l, *signed);
                t.regs[*dst as usize] = u64::from(res);
            }
            IrOp::Bfi {
                dst,
                insert,
                base,
                pos,
                len,
            } => {
                let f = eval_operand_u64(insert, t, cta_id, block_dims, grid_dims) as u32;
                let b = eval_operand_u64(base, t, cta_id, block_dims, grid_dims) as u32;
                let p = (eval_operand_u64(pos, t, cta_id, block_dims, grid_dims) as u32) & 0xFF;
                let l = (eval_operand_u64(len, t, cta_id, block_dims, grid_dims) as u32) & 0xFF;
                let res = eval_bfi(f, b, p, l);
                t.regs[*dst as usize] = u64::from(res);
            }
            IrOp::Prmt { dst, a, b, sel } => {
                let va = eval_operand_u64(a, t, cta_id, block_dims, grid_dims) as u32;
                let vb = eval_operand_u64(b, t, cta_id, block_dims, grid_dims) as u32;
                let vs = eval_operand_u64(sel, t, cta_id, block_dims, grid_dims) as u32;
                t.regs[*dst as usize] = u64::from(eval_prmt(va, vb, vs));
            }
            IrOp::Dp4a {
                dst,
                a_signed,
                b_signed,
                a,
                b,
                c,
            } => {
                let va = eval_operand_u64(a, t, cta_id, block_dims, grid_dims) as u32;
                let vb = eval_operand_u64(b, t, cta_id, block_dims, grid_dims) as u32;
                let vc = eval_operand_u64(c, t, cta_id, block_dims, grid_dims) as u32;
                t.regs[*dst as usize] = u64::from(eval_dp4a(va, vb, vc, *a_signed, *b_signed));
            }
            IrOp::Atomic {
                op,
                dst,
                ty,
                space,
                base,
                offset,
                val,
                cmp,
            } => {
                let base_addr = eval_operand_u64(base, t, cta_id, block_dims, grid_dims);
                let addr = base_addr.wrapping_add_signed(*offset);
                let size = ty.byte_size() as usize;
                let vbits = eval_operand_typed_bits(val, *ty, t, cta_id, block_dims, grid_dims);
                let cbits = cmp
                    .as_ref()
                    .map(|c| eval_operand_typed_bits(c, *ty, t, cta_id, block_dims, grid_dims))
                    .unwrap_or(0);
                let old_bits = if *space == AddressSpace::Shared {
                    let start = addr as usize;
                    let slice = lds.get_mut(start..start + size).ok_or_else(|| {
                        HalError::MemoryAccessFault {
                            addr,
                            size,
                            reason: "LDS shared atomic out of bounds".to_owned(),
                        }
                    })?;
                    let old = decode_le_bits(slice);
                    let new_v = eval_atomic(*op, *ty, old, vbits, cbits);
                    slice.copy_from_slice(&encode_le_bits(new_v, size)[..size]);
                    old
                } else {
                    let mut mmu = self.mmu.write().expect("mmu write lock");
                    let slice = mmu.lookup_slice_mut(addr, size)?;
                    let old = decode_le_bits(slice);
                    let new_v = eval_atomic(*op, *ty, old, vbits, cbits);
                    slice.copy_from_slice(&encode_le_bits(new_v, size)[..size]);
                    old
                };
                if let Some(d) = dst {
                    t.regs[*d as usize] = old_bits;
                }
            }
            IrOp::Branch(lbl) => {
                next_pc = *label_pcs.get(lbl.as_str()).ok_or_else(|| {
                    HalError::ExecutionError(format!("branch target '{lbl}' not found"))
                })?;
            }
            IrOp::Barrier
            | IrOp::MemBar
            | IrOp::WarpShuffle { .. }
            | IrOp::WarpVote { .. }
            | IrOp::Return => {}
        }
        Ok(next_pc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitKind {
    Running,
    Barrier,
    Shuffle,
    Vote,
    Finished,
}

struct ThreadState {
    tid: [u32; 3],
    lane_id: u32,
    pc: usize,
    regs: Vec<u64>,
    private_mem: Vec<u8>,
    waiting: WaitKind,
    steps: usize,
}

fn eval_pred(pred: &Option<amda_compiler::IrPredGuard>, regs: &[u64]) -> bool {
    match pred {
        None => true,
        Some(pg) => {
            let val = (regs[pg.reg as usize] & 1) != 0;
            if pg.negated {
                !val
            } else {
                val
            }
        }
    }
}

fn eval_special(
    sp: SpecialReg,
    t: &ThreadState,
    cta_id: [u32; 3],
    block_dims: [u32; 3],
    grid_dims: [u32; 3],
) -> u64 {
    match sp {
        SpecialReg::TidX => u64::from(t.tid[0]),
        SpecialReg::TidY => u64::from(t.tid[1]),
        SpecialReg::TidZ => u64::from(t.tid[2]),
        SpecialReg::CtaIdX => u64::from(cta_id[0]),
        SpecialReg::CtaIdY => u64::from(cta_id[1]),
        SpecialReg::CtaIdZ => u64::from(cta_id[2]),
        SpecialReg::NtidX => u64::from(block_dims[0]),
        SpecialReg::NtidY => u64::from(block_dims[1]),
        SpecialReg::NtidZ => u64::from(block_dims[2]),
        SpecialReg::NctaIdX => u64::from(grid_dims[0]),
        SpecialReg::NctaIdY => u64::from(grid_dims[1]),
        SpecialReg::NctaIdZ => u64::from(grid_dims[2]),
        SpecialReg::LaneId => u64::from(t.lane_id),
    }
}

fn eval_operand_u64(
    op: &IrOperand,
    t: &ThreadState,
    cta_id: [u32; 3],
    block_dims: [u32; 3],
    grid_dims: [u32; 3],
) -> u64 {
    match op {
        IrOperand::Reg(r) => t.regs[*r as usize],
        IrOperand::ImmU64(v) => *v,
        IrOperand::ImmI64(v) => *v as u64,
        IrOperand::ImmF64(f) => f.to_bits(),
        IrOperand::Special(sp) => eval_special(*sp, t, cta_id, block_dims, grid_dims),
        IrOperand::Param(idx) => u64::from(*idx),
        IrOperand::SharedOffset(off) | IrOperand::LocalOffset(off) => u64::from(*off),
    }
}

fn eval_operand_typed_bits(
    op: &IrOperand,
    ty: IrType,
    t: &ThreadState,
    cta_id: [u32; 3],
    block_dims: [u32; 3],
    grid_dims: [u32; 3],
) -> u64 {
    match op {
        IrOperand::ImmF64(f) => match ty {
            IrType::F16 => u64::from(f32_to_f16(*f as f32)),
            IrType::Bf16 => u64::from(f32_to_bf16(*f as f32)),
            IrType::F32 => u64::from((*f as f32).to_bits()),
            IrType::F64 => f.to_bits(),
            _ => *f as i64 as u64,
        },
        IrOperand::ImmI64(v) => match ty {
            IrType::F16 => u64::from(f32_to_f16(*v as f32)),
            IrType::Bf16 => u64::from(f32_to_bf16(*v as f32)),
            IrType::F32 => u64::from((*v as f32).to_bits()),
            IrType::F64 => (*v as f64).to_bits(),
            IrType::U8 | IrType::I8 => u64::from(*v as u8),
            IrType::U16 | IrType::I16 => u64::from(*v as u16),
            IrType::U32 | IrType::I32 => u64::from(*v as u32),
            _ => *v as u64,
        },
        IrOperand::ImmU64(v) => match ty {
            IrType::F16 => u64::from(f32_to_f16(*v as f32)),
            IrType::Bf16 => u64::from(f32_to_bf16(*v as f32)),
            IrType::F32 => u64::from((*v as f32).to_bits()),
            IrType::F64 => (*v as f64).to_bits(),
            IrType::U8 | IrType::I8 => u64::from(*v as u8),
            IrType::U16 | IrType::I16 => u64::from(*v as u16),
            IrType::U32 | IrType::I32 => u64::from(*v as u32),
            _ => *v,
        },
        _ => eval_operand_u64(op, t, cta_id, block_dims, grid_dims),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from((h >> 15) & 1) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = -14i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f_exp | (m << 13));
    }
    if exp == 31 {
        let nan_mant = if mant == 0 { 0 } else { (mant << 13) | 0x0040_0000 };
        return f32::from_bits(sign | (0xFF << 23) | nan_mant);
    }
    let f_exp = (exp + 112) << 23;
    f32::from_bits(sign | f_exp | (mant << 13))
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_m = (m >> shift) as u16;
        return sign | half_m;
    }
    let round_bit = (mant >> 12) & 1;
    let base_m = (mant >> 13) as u16;
    let combined = ((new_exp as u16) << 10) | base_m;
    sign | (combined + (round_bit as u16))
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let bias = 0x7FFF + lsb;
    (bits.wrapping_add(bias) >> 16) as u16
}

fn convert_value(src_ty: IrType, dst_ty: IrType, bits: u64) -> u64 {
    if src_ty == dst_ty {
        return bits;
    }
    enum Num {
        Signed(i64),
        Unsigned(u64),
        Float(f64),
    }
    let val = match src_ty {
        IrType::I8 => Num::Signed(i64::from(bits as u8 as i8)),
        IrType::U8 => Num::Unsigned(u64::from(bits as u8)),
        IrType::I16 => Num::Signed(i64::from(bits as u16 as i16)),
        IrType::U16 => Num::Unsigned(u64::from(bits as u16)),
        IrType::I32 => Num::Signed(i64::from(bits as u32 as i32)),
        IrType::I64 => Num::Signed(bits as i64),
        IrType::U32 => Num::Unsigned(u64::from(bits as u32)),
        IrType::U64 => Num::Unsigned(bits),
        IrType::F16 => Num::Float(f64::from(f16_to_f32(bits as u16))),
        IrType::Bf16 => Num::Float(f64::from(bf16_to_f32(bits as u16))),
        IrType::F32 => Num::Float(f64::from(f32::from_bits(bits as u32))),
        IrType::F64 => Num::Float(f64::from_bits(bits)),
        IrType::Pred => Num::Unsigned(bits & 1),
    };
    match dst_ty {
        IrType::I8 => {
            let v = match val {
                Num::Signed(s) => s as i8,
                Num::Unsigned(u) => u as i8,
                Num::Float(f) => f as i8,
            };
            u64::from(v as u8)
        }
        IrType::U8 => {
            let v = match val {
                Num::Signed(s) => s as u8,
                Num::Unsigned(u) => u as u8,
                Num::Float(f) => f as u8,
            };
            u64::from(v)
        }
        IrType::I16 => {
            let v = match val {
                Num::Signed(s) => s as i16,
                Num::Unsigned(u) => u as i16,
                Num::Float(f) => f as i16,
            };
            u64::from(v as u16)
        }
        IrType::U16 => {
            let v = match val {
                Num::Signed(s) => s as u16,
                Num::Unsigned(u) => u as u16,
                Num::Float(f) => f as u16,
            };
            u64::from(v)
        }
        IrType::I32 => {
            let v = match val {
                Num::Signed(s) => s as i32,
                Num::Unsigned(u) => u as i32,
                Num::Float(f) => f as i32,
            };
            u64::from(v as u32)
        }
        IrType::U32 => {
            let v = match val {
                Num::Signed(s) => s as u32,
                Num::Unsigned(u) => u as u32,
                Num::Float(f) => f as u32,
            };
            u64::from(v)
        }
        IrType::I64 => match val {
            Num::Signed(s) => s as u64,
            Num::Unsigned(u) => u,
            Num::Float(f) => (f as i64) as u64,
        },
        IrType::U64 => match val {
            Num::Signed(s) => s as u64,
            Num::Unsigned(u) => u,
            Num::Float(f) => f as u64,
        },
        IrType::F16 => {
            let f = match val {
                Num::Signed(s) => s as f32,
                Num::Unsigned(u) => u as f32,
                Num::Float(fl) => fl as f32,
            };
            u64::from(f32_to_f16(f))
        }
        IrType::Bf16 => {
            let f = match val {
                Num::Signed(s) => s as f32,
                Num::Unsigned(u) => u as f32,
                Num::Float(fl) => fl as f32,
            };
            u64::from(f32_to_bf16(f))
        }
        IrType::F32 => {
            let f = match val {
                Num::Signed(s) => s as f32,
                Num::Unsigned(u) => u as f32,
                Num::Float(fl) => fl as f32,
            };
            u64::from(f.to_bits())
        }
        IrType::F64 => {
            let f = match val {
                Num::Signed(s) => s as f64,
                Num::Unsigned(u) => u as f64,
                Num::Float(fl) => fl,
            };
            f.to_bits()
        }
        IrType::Pred => match val {
            Num::Signed(s) => u64::from(s != 0),
            Num::Unsigned(u) => u64::from(u != 0),
            Num::Float(f) => u64::from(f != 0.0),
        },
    }
}

fn eval_unary_math(op: UnaryMathOp, ty: IrType, bits: u64) -> u64 {
    if ty == IrType::F64 {
        let v = f64::from_bits(bits);
        let res = match op {
            UnaryMathOp::Rcp => 1.0 / v,
            UnaryMathOp::Sqrt => v.sqrt(),
            UnaryMathOp::Rsqrt => 1.0 / v.sqrt(),
            UnaryMathOp::Ex2 => v.exp2(),
            UnaryMathOp::Lg2 => v.log2(),
            UnaryMathOp::Sin => v.sin(),
            UnaryMathOp::Cos => v.cos(),
            UnaryMathOp::Tanh => v.tanh(),
        };
        res.to_bits()
    } else {
        let v = match ty {
            IrType::F16 => f16_to_f32(bits as u16),
            IrType::Bf16 => bf16_to_f32(bits as u16),
            _ => f32::from_bits(bits as u32),
        };
        let res = match op {
            UnaryMathOp::Rcp => 1.0f32 / v,
            UnaryMathOp::Sqrt => v.sqrt(),
            UnaryMathOp::Rsqrt => 1.0f32 / v.sqrt(),
            UnaryMathOp::Ex2 => v.exp2(),
            UnaryMathOp::Lg2 => v.log2(),
            UnaryMathOp::Sin => v.sin(),
            UnaryMathOp::Cos => v.cos(),
            UnaryMathOp::Tanh => v.tanh(),
        };
        match ty {
            IrType::F16 => u64::from(f32_to_f16(res)),
            IrType::Bf16 => u64::from(f32_to_bf16(res)),
            _ => u64::from(res.to_bits()),
        }
    }
}

fn eval_bfe(src: u32, pos: u32, len: u32, signed: bool) -> u32 {
    if len == 0 || pos >= 32 {
        return 0;
    }
    let actual_len = len.min(32 - pos);
    let mask = if actual_len == 32 {
        u32::MAX
    } else {
        (1u32 << actual_len) - 1
    };
    let field = (src >> pos) & mask;
    if signed && actual_len > 0 && actual_len < 32 {
        let sign_bit = 1u32 << (actual_len - 1);
        if (field & sign_bit) != 0 {
            field | !mask
        } else {
            field
        }
    } else {
        field
    }
}

fn eval_bfi(insert: u32, base: u32, pos: u32, len: u32) -> u32 {
    if len == 0 || pos >= 32 {
        return base;
    }
    let actual_len = len.min(32 - pos);
    let mask_lo = if actual_len == 32 {
        u32::MAX
    } else {
        (1u32 << actual_len) - 1
    };
    let mask = mask_lo << pos;
    (base & !mask) | ((insert & mask_lo) << pos)
}

fn eval_prmt(a: u32, b: u32, sel: u32) -> u32 {
    let ab: [u8; 8] = [
        a as u8,
        (a >> 8) as u8,
        (a >> 16) as u8,
        (a >> 24) as u8,
        b as u8,
        (b >> 8) as u8,
        (b >> 16) as u8,
        (b >> 24) as u8,
    ];
    let mut out = 0u32;
    for i in 0..4 {
        let nib = (sel >> (i * 4)) & 0xF;
        let idx = (nib & 7) as usize;
        let byte = ab[idx];
        let val = if (nib & 8) != 0 {
            if (byte & 0x80) != 0 { 0xFF } else { 0x00 }
        } else {
            byte
        };
        out |= u32::from(val) << (i * 8);
    }
    out
}

fn eval_dp4a(a: u32, b: u32, c: u32, a_signed: bool, b_signed: bool) -> u32 {
    let mut acc = c as i32;
    for i in 0..4 {
        let ba = (a >> (i * 8)) as u8;
        let bb = (b >> (i * 8)) as u8;
        let va = if a_signed { i32::from(ba as i8) } else { i32::from(ba) };
        let vb = if b_signed { i32::from(bb as i8) } else { i32::from(bb) };
        acc = acc.wrapping_add(va.wrapping_mul(vb));
    }
    acc as u32
}

fn eval_atomic(op: AtomicOp, ty: IrType, old: u64, val: u64, cmp: u64) -> u64 {
    if ty == IrType::F32 {
        let o = f32::from_bits(old as u32);
        let v = f32::from_bits(val as u32);
        let r = match op {
            AtomicOp::Add => o + v,
            AtomicOp::Min => o.min(v),
            AtomicOp::Max => o.max(v),
            AtomicOp::Exch => v,
            _ => v,
        };
        return u64::from(r.to_bits());
    }
    if matches!(ty, IrType::U64 | IrType::I64) {
        return match op {
            AtomicOp::Add => old.wrapping_add(val),
            AtomicOp::Min => {
                if ty == IrType::I64 {
                    (old as i64).min(val as i64) as u64
                } else {
                    old.min(val)
                }
            }
            AtomicOp::Max => {
                if ty == IrType::I64 {
                    (old as i64).max(val as i64) as u64
                } else {
                    old.max(val)
                }
            }
            AtomicOp::And => old & val,
            AtomicOp::Or => old | val,
            AtomicOp::Xor => old ^ val,
            AtomicOp::Exch => val,
            AtomicOp::Cas => {
                if old == cmp { val } else { old }
            }
        };
    }
    let o = old as u32;
    let v = val as u32;
    let c = cmp as u32;
    let r = match op {
        AtomicOp::Add => o.wrapping_add(v),
        AtomicOp::Min => {
            if ty == IrType::I32 {
                (o as i32).min(v as i32) as u32
            } else {
                o.min(v)
            }
        }
        AtomicOp::Max => {
            if ty == IrType::I32 {
                (o as i32).max(v as i32) as u32
            } else {
                o.max(v)
            }
        }
        AtomicOp::And => o & v,
        AtomicOp::Or => o | v,
        AtomicOp::Xor => o ^ v,
        AtomicOp::Exch => v,
        AtomicOp::Cas => {
            if o == c { v } else { o }
        }
    };
    u64::from(r)
}

fn eval_binop(op: BinOp, ty: IrType, l: u64, r: u64) -> Result<u64, HalError> {
    if matches!(op, BinOp::MulWide) {
        return Ok(match ty {
            IrType::I32 => {
                let a = i64::from(l as u32 as i32);
                let b = i64::from(r as u32 as i32);
                a.wrapping_mul(b) as u64
            }
            _ => {
                let a = u64::from(l as u32);
                let b = u64::from(r as u32);
                a.wrapping_mul(b)
            }
        });
    }

    match ty {
        IrType::F16 | IrType::Bf16 | IrType::F32 => {
            let to_f = |bits: u64| match ty {
                IrType::F16 => f16_to_f32(bits as u16),
                IrType::Bf16 => bf16_to_f32(bits as u16),
                _ => f32::from_bits(bits as u32),
            };
            let from_f = |f: f32| match ty {
                IrType::F16 => u64::from(f32_to_f16(f)),
                IrType::Bf16 => u64::from(f32_to_bf16(f)),
                _ => u64::from(f.to_bits()),
            };
            let a = to_f(l);
            let b = to_f(r);
            let res = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Rem => a % b,
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                _ => {
                    return Err(HalError::ExecutionError(format!(
                        "invalid float op {op:?}"
                    )))
                }
            };
            Ok(from_f(res))
        }
        IrType::F64 => {
            let a = f64::from_bits(l);
            let b = f64::from_bits(r);
            let res = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Rem => a % b,
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                _ => {
                    return Err(HalError::ExecutionError(format!(
                        "invalid float64 op {op:?}"
                    )))
                }
            };
            Ok(res.to_bits())
        }
        IrType::U8 | IrType::I8 | IrType::U16 | IrType::I16 | IrType::I32 | IrType::U32 => {
            let a = l as u32;
            let b = r as u32;
            let signed = matches!(ty, IrType::I8 | IrType::I16 | IrType::I32);
            let res = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Div => {
                    if b == 0 {
                        0
                    } else if signed {
                        ((a as i32).wrapping_div(b as i32)) as u32
                    } else {
                        a / b
                    }
                }
                BinOp::Rem => {
                    if b == 0 {
                        0
                    } else if signed {
                        ((a as i32).wrapping_rem(b as i32)) as u32
                    } else {
                        a % b
                    }
                }
                BinOp::Min => {
                    if signed { (a as i32).min(b as i32) as u32 } else { a.min(b) }
                }
                BinOp::Max => {
                    if signed { (a as i32).max(b as i32) as u32 } else { a.max(b) }
                }
                BinOp::Shl => a.wrapping_shl(b),
                BinOp::Shr => {
                    if signed {
                        ((a as i32).wrapping_shr(b)) as u32
                    } else {
                        a.wrapping_shr(b)
                    }
                }
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::MulWide => unreachable!(),
            };
            Ok(u64::from(res))
        }
        _ => {
            let signed = ty == IrType::I64;
            let res = match op {
                BinOp::Add => l.wrapping_add(r),
                BinOp::Sub => l.wrapping_sub(r),
                BinOp::Mul => l.wrapping_mul(r),
                BinOp::Div => {
                    if r == 0 {
                        0
                    } else if signed {
                        ((l as i64).wrapping_div(r as i64)) as u64
                    } else {
                        l / r
                    }
                }
                BinOp::Rem => {
                    if r == 0 {
                        0
                    } else if signed {
                        ((l as i64).wrapping_rem(r as i64)) as u64
                    } else {
                        l % r
                    }
                }
                BinOp::Min => {
                    if signed { (l as i64).min(r as i64) as u64 } else { l.min(r) }
                }
                BinOp::Max => {
                    if signed { (l as i64).max(r as i64) as u64 } else { l.max(r) }
                }
                BinOp::Shl => l.wrapping_shl(r as u32),
                BinOp::Shr => {
                    if signed {
                        ((l as i64).wrapping_shr(r as u32)) as u64
                    } else {
                        l.wrapping_shr(r as u32)
                    }
                }
                BinOp::And => l & r,
                BinOp::Or => l | r,
                BinOp::Xor => l ^ r,
                BinOp::MulWide => unreachable!(),
            };
            Ok(res)
        }
    }
}

fn eval_mad(ty: IrType, a: u64, b: u64, c: u64) -> u64 {
    match ty {
        IrType::F16 => {
            let fa = f16_to_f32(a as u16);
            let fb = f16_to_f32(b as u16);
            let fc = f16_to_f32(c as u16);
            u64::from(f32_to_f16(fa.mul_add(fb, fc)))
        }
        IrType::Bf16 => {
            let fa = bf16_to_f32(a as u16);
            let fb = bf16_to_f32(b as u16);
            let fc = bf16_to_f32(c as u16);
            u64::from(f32_to_bf16(fa.mul_add(fb, fc)))
        }
        IrType::F32 => {
            let fa = f32::from_bits(a as u32);
            let fb = f32::from_bits(b as u32);
            let fc = f32::from_bits(c as u32);
            u64::from(fa.mul_add(fb, fc).to_bits())
        }
        IrType::F64 => {
            let fa = f64::from_bits(a);
            let fb = f64::from_bits(b);
            let fc = f64::from_bits(c);
            fa.mul_add(fb, fc).to_bits()
        }
        IrType::U8 | IrType::I8 | IrType::U16 | IrType::I16 | IrType::I32 | IrType::U32 => {
            let x = a as u32;
            let y = b as u32;
            let z = c as u32;
            u64::from(x.wrapping_mul(y).wrapping_add(z))
        }
        _ => a.wrapping_mul(b).wrapping_add(c),
    }
}

fn eval_cmp_ord<T: PartialOrd + PartialEq>(cmp: CmpOp, a: T, b: T) -> bool {
    match cmp {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn eval_cmp(cmp: CmpOp, ty: IrType, l: u64, r: u64) -> bool {
    match ty {
        IrType::F16 => eval_cmp_ord(cmp, f16_to_f32(l as u16), f16_to_f32(r as u16)),
        IrType::Bf16 => eval_cmp_ord(cmp, bf16_to_f32(l as u16), bf16_to_f32(r as u16)),
        IrType::F32 => eval_cmp_ord(cmp, f32::from_bits(l as u32), f32::from_bits(r as u32)),
        IrType::F64 => eval_cmp_ord(cmp, f64::from_bits(l), f64::from_bits(r)),
        IrType::I8 => eval_cmp_ord(cmp, l as u8 as i8, r as u8 as i8),
        IrType::U8 => eval_cmp_ord(cmp, l as u8, r as u8),
        IrType::I16 => eval_cmp_ord(cmp, l as u16 as i16, r as u16 as i16),
        IrType::U16 => eval_cmp_ord(cmp, l as u16, r as u16),
        IrType::I32 => eval_cmp_ord(cmp, l as u32 as i32, r as u32 as i32),
        IrType::U32 | IrType::Pred => eval_cmp_ord(cmp, l as u32, r as u32),
        IrType::I64 => eval_cmp_ord(cmp, l as i64, r as i64),
        IrType::U64 => eval_cmp_ord(cmp, l, r),
    }
}

fn decode_le_bits(slice: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = slice.len().min(8);
    buf[..n].copy_from_slice(&slice[..n]);
    u64::from_le_bytes(buf)
}

fn encode_le_bits(val: u64, size: usize) -> [u8; 8] {
    let _ = size;
    val.to_le_bytes()
}

impl DeviceBackend for MockBackend {
    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    fn alloc_vram(&self, device: usize, size: usize, align: usize) -> Result<u64, HalError> {
        self.check_device(device)?;
        self.mmu.write().expect("mmu write lock").alloc(size, align)
    }

    fn free_vram(&self, device: usize, ptr: u64) -> Result<(), HalError> {
        self.check_device(device)?;
        self.mmu.write().expect("mmu write lock").free(ptr)
    }

    fn copy_htod(&self, device: usize, dst: u64, src: &[u8]) -> Result<(), HalError> {
        self.check_device(device)?;
        self.mmu
            .write()
            .expect("mmu write lock")
            .copy_htod(dst, src)?;
        if let Ok(mut st) = self.stats.lock() {
            st.bytes_copied_htod += src.len() as u64;
        }
        Ok(())
    }

    fn copy_dtoh(&self, device: usize, dst: &mut [u8], src: u64) -> Result<(), HalError> {
        self.check_device(device)?;
        self.mmu
            .read()
            .expect("mmu read lock")
            .copy_dtoh(dst, src)?;
        if let Ok(mut st) = self.stats.lock() {
            st.bytes_copied_dtoh += dst.len() as u64;
        }
        Ok(())
    }

    fn copy_dtod(&self, device: usize, dst: u64, src: u64, len: usize) -> Result<(), HalError> {
        self.check_device(device)?;
        self.mmu
            .write()
            .expect("mmu write lock")
            .copy_dtod(dst, src, len)?;
        if let Ok(mut st) = self.stats.lock() {
            st.bytes_copied_dtod += len as u64;
        }
        Ok(())
    }

    fn create_queue(&self, device: usize, ring_size: usize) -> Result<QueueId, HalError> {
        self.check_device(device)?;
        let id = self.alloc_id();
        let queue = Arc::new(MockQueue {
            device,
            ring: AqlRingBuffer::new(ring_size),
        });
        self.queues.write().expect("queues write lock").insert(id, queue);
        Ok(QueueId(id))
    }

    fn destroy_queue(&self, queue: QueueId) -> Result<(), HalError> {
        if self
            .queues
            .write()
            .expect("queues write lock")
            .remove(&queue.0)
            .is_none()
        {
            return Err(HalError::InvalidHandle("AQL queue"));
        }
        Ok(())
    }

    fn create_signal(&self, initial_value: i64) -> Result<SignalId, HalError> {
        let id = self.alloc_id();
        self.signals
            .write()
            .expect("signals write lock")
            .insert(id, Arc::new(AtomicI64::new(initial_value)));
        Ok(SignalId(id))
    }

    fn destroy_signal(&self, signal: SignalId) -> Result<(), HalError> {
        if self
            .signals
            .write()
            .expect("signals write lock")
            .remove(&signal.0)
            .is_none()
        {
            return Err(HalError::InvalidHandle("signal"));
        }
        Ok(())
    }

    fn signal_store(&self, signal: SignalId, value: i64) -> Result<(), HalError> {
        let guard = self.signals.read().expect("signals read lock");
        let atom = guard.get(&signal.0).ok_or(HalError::InvalidHandle("signal"))?;
        atom.store(value, Ordering::Release);
        Ok(())
    }

    fn signal_load(&self, signal: SignalId) -> Result<i64, HalError> {
        let guard = self.signals.read().expect("signals read lock");
        let atom = guard.get(&signal.0).ok_or(HalError::InvalidHandle("signal"))?;
        Ok(atom.load(Ordering::Acquire))
    }

    fn signal_wait(&self, signal: SignalId, expected: i64) -> Result<(), HalError> {
        let guard = self.signals.read().expect("signals read lock");
        let atom = guard.get(&signal.0).ok_or(HalError::InvalidHandle("signal"))?;
        let current = atom.load(Ordering::Acquire);
        if current == expected {
            Ok(())
        } else {
            Err(HalError::ExecutionError(format!(
                "signal {} expected {expected} but observed {current}",
                signal.0
            )))
        }
    }

    fn load_code_object(&self, device: usize, hsaco: &[u8]) -> Result<CodeObjectId, HalError> {
        self.check_device(device)?;
        let image = parse_hsaco(hsaco)
            .map_err(|e| HalError::InvalidCodeObject(format!("HSACO parse error: {e}")))?;
        let ir_bytes = image.amda_ir_payload.as_deref().ok_or_else(|| {
            HalError::InvalidCodeObject("HSACO lacks .amda.ir section for virtual execution".to_owned())
        })?;
        let ir_mod = IrModule::from_bytes(ir_bytes)
            .map_err(|e| HalError::InvalidCodeObject(format!("AMDA IR decode error: {e}")))?;

        let obj_id = self.alloc_id();
        let mut kernel_ids = Vec::with_capacity(ir_mod.kernels.len());
        {
            let mut kguard = self.kernels.write().expect("kernels write lock");
            for k in ir_mod.kernels {
                let kid = self.alloc_id();
                kguard.insert(
                    kid,
                    LoadedKernel {
                        device,
                        ir: Arc::new(k),
                    },
                );
                kernel_ids.push(kid);
            }
        }
        self.code_objects
            .write()
            .expect("code_objects write lock")
            .insert(obj_id, kernel_ids);
        Ok(CodeObjectId(obj_id))
    }

    fn unload_code_object(&self, obj: CodeObjectId) -> Result<(), HalError> {
        let kids = self
            .code_objects
            .write()
            .expect("code_objects write lock")
            .remove(&obj.0)
            .ok_or(HalError::InvalidHandle("code object"))?;
        let mut kguard = self.kernels.write().expect("kernels write lock");
        for kid in kids {
            kguard.remove(&kid);
        }
        Ok(())
    }

    fn get_kernel(
        &self,
        obj: CodeObjectId,
        name: &str,
    ) -> Result<(KernelHandle, KernelMetadata), HalError> {
        let kids = self
            .code_objects
            .read()
            .expect("code_objects read lock")
            .get(&obj.0)
            .cloned()
            .ok_or(HalError::InvalidHandle("code object"))?;

        let kguard = self.kernels.read().expect("kernels read lock");
        for kid in kids {
            if let Some(lk) = kguard.get(&kid) {
                if lk.ir.name == name {
                    return Ok((KernelHandle(kid), lk.ir.metadata.clone()));
                }
            }
        }
        Err(HalError::KernelNotFound(name.to_owned()))
    }

    fn submit_aql_packet(&self, queue: QueueId, packet: &[u8; 64]) -> Result<(), HalError> {
        let q = {
            let guard = self.queues.read().expect("queues read lock");
            guard
                .get(&queue.0)
                .cloned()
                .ok_or(HalError::InvalidHandle("AQL queue"))?
        };
        q.ring.push_packet(packet)?;
        self.ring_doorbell(&q)
    }
}
