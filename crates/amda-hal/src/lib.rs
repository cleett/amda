//! AMDA Hardware Abstraction Layer: 64-byte AQL packets & ring buffer,
//! Linux `/dev/kfd` ioctls & topology discovery, and Virtual AMD GPU (`MockBackend`).

pub mod aql;
pub mod backend;
pub mod device;
pub mod error;
pub mod kfd;
pub mod mock;

pub use aql::{
    decode_aql_header, encode_aql_header, AqlPacketType, AqlRingBuffer, BarrierAndPacket,
    FenceScope, KernelDispatchPacket,
};
pub use backend::{CodeObjectId, DeviceBackend, KernelHandle, QueueId, SignalId};
pub use device::{DeviceInfo, HardwareFeatures};
pub use error::HalError;
pub use kfd::{
    aql_ring_has_space, compute_kfd_queue_buffer_sizes, decode_gfx_target_version,
    device_info_from_kfd, parse_kfd_mem_bank_properties, parse_kfd_topology_properties,
    patch_kernel_descriptor_entry_offset, resolve_kernel_text_entry_offset,
    select_kfd_alloc_flags, try_open_kfd_backend, HsaSignalStruct, KfdGpuVaAllocator,
    KfdNodeProps, KfdQueueBufferSizes,
};
pub use mock::{MockBackend, MockStats};

#[cfg(test)]
mod tests {
    use super::*;
    use amda_compiler::Compiler;
    use amda_elf::IsaTarget;

    #[test]
    fn aql_packet_layout_and_ring_roundtrip() {
        assert_eq!(core::mem::size_of::<KernelDispatchPacket>(), 64);
        assert_eq!(core::mem::size_of::<BarrierAndPacket>(), 64);

        let hdr = encode_aql_header(
            AqlPacketType::KernelDispatch,
            true,
            FenceScope::System,
            FenceScope::Agent,
        );
        let (ty, barrier, acq, rel) = decode_aql_header(hdr);
        assert_eq!(ty, AqlPacketType::KernelDispatch as u8);
        assert!(barrier);
        assert_eq!(acq, FenceScope::System);
        assert_eq!(rel, FenceScope::Agent);

        let ring = AqlRingBuffer::new(16);
        let pkt = KernelDispatchPacket {
            header: hdr,
            setup: 1,
            workgroup_size_x: 256,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            reserved0: 0,
            grid_size_x: 1024,
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: 0,
            group_segment_size: 0,
            kernel_object: 42,
            kernarg_address: 0x1000,
            reserved2: 0,
            completion_signal: 7,
        };
        let slot = ring.push_packet(&pkt.to_bytes()).expect("push succeeds");
        assert_eq!(slot, 0);
        let popped = ring.pop_packet().expect("pop succeeds");
        assert_eq!(KernelDispatchPacket::from_bytes(&popped), pkt);
        assert!(ring.pop_packet().is_none());
    }

    #[test]
    fn kfd_sysfs_topology_parser_skips_cpu_and_parses_gpus() {
        let cpu_node = "\
cpu_cores_count 16
simd_count 0
gfx_target_version 0
";
        assert!(parse_kfd_topology_properties(cpu_node).is_none());

        let rx7900_node = "\
gpu_id 48392
simd_count 192
simd_per_cu 2
max_waves_per_simd 16
lds_size_in_kb 64
gfx_target_version 110000
local_mem_size 25769803776
drm_render_minor 128
";
        let props = parse_kfd_topology_properties(rx7900_node).expect("GPU node must parse");
        assert_eq!(props.gpu_id, 48392);
        let info = device_info_from_kfd(0, &props).expect("device info");
        assert_eq!(info.isa, IsaTarget::Gfx1100);
        assert_eq!(info.features.compute_units, 96);
        assert!(info.features.wave32);
        assert!(info.features.wmma);

        let mi300_node = "\
gpu_id 99123
simd_count 1216
simd_per_cu 4
max_waves_per_simd 8
lds_size_in_kb 64
gfx_target_version 90402
local_mem_size 206158430208
";
        let mi_props = parse_kfd_topology_properties(mi300_node).expect("MI300X node must parse");
        let mi_info = device_info_from_kfd(0, &mi_props).expect("MI300X device info");
        assert_eq!(mi_info.isa, IsaTarget::Gfx942);
        assert!(mi_info.features.mfma);
        assert!(mi_info.features.fp8);
        assert!(!mi_info.features.wave32);

        let r780m_node = "\
gpu_id 27701
simd_count 24
simd_per_cu 2
max_waves_per_simd 16
lds_size_in_kb 64
gfx_target_version 110003
local_mem_size 0
";
        let apu_props =
            parse_kfd_topology_properties(r780m_node).expect("gfx1103 APU node must parse");
        let apu_info = device_info_from_kfd(0, &apu_props).expect("gfx1103 device info");
        assert_eq!(apu_info.isa, IsaTarget::Gfx1103);
        assert_eq!(apu_info.features.compute_units, 12);
        assert!(apu_info.features.wave32);
        assert!(apu_info.features.wmma);
        assert!(apu_info.features.unified_memory);
    }

    #[test]
    fn mock_mmu_bounds_checking_and_aql_dispatch() {
        let backend = MockBackend::with_isa(IsaTarget::Gfx1100);
        let ptr = backend.alloc_vram(0, 16, 256).expect("alloc 16 bytes");
        backend
            .copy_htod(0, ptr, &[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("valid write");

        let mut bad_buf = [0u8; 32];
        assert!(matches!(
            backend.copy_dtoh(0, &mut bad_buf, ptr),
            Err(HalError::MemoryAccessFault { .. })
        ));

        // Direct AQL ring dispatch of a shared-memory barrier kernel
        let ptx = r#"
            .version 7.8
            .target sm_80
            .address_size 64
            .visible .entry lds_copy(.param .u64 out_ptr) {
                .shared .align 4 .f32 smem[4];
                .reg .b32 %r<3>;
                .reg .f32 %f<3>;
                .reg .b64 %rd<5>;
                ld.param.u64 %rd1, [lds_copy_param_0];
                mov.u32 %r0, %tid.x;
                cvt.rn.f32.u32 %f0, %r0;
                mul.wide.u32 %rd2, %r0, 4;
                mov.u64 %rd3, smem;
                add.u64 %rd3, %rd3, %rd2;
                st.shared.f32 [%rd3], %f0;
                bar.sync 0;
                xor.b32 %r1, %r0, 1;
                mul.wide.u32 %rd4, %r1, 4;
                mov.u64 %rd3, smem;
                add.u64 %rd3, %rd3, %rd4;
                ld.shared.f32 %f1, [%rd3];
                cvta.to.global.u64 %rd1, %rd1;
                add.u64 %rd1, %rd1, %rd2;
                st.global.f32 [%rd1], %f1;
                ret;
            }
        "#;
        // Adjust param name inside PTX so it matches `out_ptr`
        let ptx = ptx.replace("lds_copy_param_0", "out_ptr");

        let compiler = Compiler::with_cache_dir(None);
        let art = compiler.compile_ptx(&ptx, IsaTarget::Gfx1100).unwrap();
        let obj = backend.load_code_object(0, &art.hsaco).unwrap();
        let (khandle, kmeta) = backend.get_kernel(obj, "lds_copy").unwrap();

        let out_ptr = backend.alloc_vram(0, 16, 256).unwrap();
        let kernarg_ptr = backend
            .alloc_vram(0, kmeta.kernarg_segment_size as usize, 64)
            .unwrap();
        backend
            .copy_htod(0, kernarg_ptr, &out_ptr.to_le_bytes())
            .unwrap();

        let queue = backend.create_queue(0, 16).unwrap();
        let sig = backend.create_signal(1).unwrap();
        let pkt = KernelDispatchPacket {
            header: encode_aql_header(
                AqlPacketType::KernelDispatch,
                true,
                FenceScope::System,
                FenceScope::System,
            ),
            setup: 1,
            workgroup_size_x: 4,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            reserved0: 0,
            grid_size_x: 4, // total work-items = 4
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: 0,
            group_segment_size: kmeta.group_segment_fixed_size,
            kernel_object: khandle.0,
            kernarg_address: kernarg_ptr,
            reserved2: 0,
            completion_signal: sig.0,
        };
        backend.submit_aql_packet(queue, &pkt.to_bytes()).unwrap();
        backend.signal_wait(sig, 0).unwrap();

        let mut raw_out = [0u8; 16];
        backend.copy_dtoh(0, &mut raw_out, out_ptr).unwrap();
        let vals: [f32; 4] = [
            f32::from_le_bytes(raw_out[0..4].try_into().unwrap()),
            f32::from_le_bytes(raw_out[4..8].try_into().unwrap()),
            f32::from_le_bytes(raw_out[8..12].try_into().unwrap()),
            f32::from_le_bytes(raw_out[12..16].try_into().unwrap()),
        ];
        assert_eq!(vals, [1.0, 0.0, 3.0, 2.0]);
    }

    #[test]
    fn kfd_gpu_va_allocator_flags_and_descriptor_patch() {
        assert_eq!(core::mem::size_of::<HsaSignalStruct>(), 64);
        assert_eq!(core::mem::align_of::<HsaSignalStruct>(), 64);
        assert_ne!(kfd::AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, 0);
        assert_ne!(kfd::AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, 0);
        assert_ne!(kfd::AMDKFD_IOC_WAIT_EVENTS, 0);

        // Flag selection: APU vs dGPU
        let apu_flags = select_kfd_alloc_flags(true, true, false);
        assert_ne!(apu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0);
        assert_ne!(apu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0);
        assert_ne!(apu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, 0);

        let dgpu_flags = select_kfd_alloc_flags(false, false, false);
        assert_ne!(dgpu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0);
        assert_ne!(dgpu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0);
        assert_eq!(dgpu_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, 0);

        let dgpu_ctrl_flags = select_kfd_alloc_flags(false, false, true);
        assert_ne!(dgpu_ctrl_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0);
        assert_ne!(dgpu_ctrl_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0);
        assert_eq!(dgpu_ctrl_flags & kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0);

        // AQL ring wrap backpressure (`read_dispatch_id` vs `write_idx`)
        assert!(aql_ring_has_space(0, 0, 64));
        assert!(aql_ring_has_space(63, 0, 64));
        assert!(!aql_ring_has_space(64, 0, 64));
        assert!(aql_ring_has_space(64, 1, 64));
        assert!(aql_ring_has_space(5, u64::MAX - 10, 64));
        assert!(!aql_ring_has_space(u64::MAX.wrapping_add(65), u64::MAX, 64));

        // Page-aligned GPUVM VA allocator + neighbor coalescing
        let mut alloc = KfdGpuVaAllocator::new(0x10_0000, 0x20_0000);
        let a = alloc.alloc(8192, 4096).unwrap();
        let b = alloc.alloc(4096, 4096).unwrap();
        let c = alloc.alloc(4096, 8192).unwrap();
        assert_eq!(a % 4096, 0);
        assert_eq!(c % 8192, 0);
        alloc.free(b, 4096);
        alloc.free(a, 8192);
        // After coalescing a + b (12 KiB), allocating 12 KiB reuses `a`
        let reused = alloc.alloc(12 * 1024, 4096).unwrap();
        assert_eq!(reused, a);

        // Descriptor entry patching (`kernel_code_entry_byte_offset` at [16..24])
        let mut kd = [0u8; 64];
        patch_kernel_descriptor_entry_offset(&mut kd, 0x4000_0000, 0x4000_0280).unwrap();
        let patched = i64::from_le_bytes(kd[16..24].try_into().unwrap());
        assert_eq!(patched, 0x280);
    }

    #[test]
    fn mi300x_xcc_mem_banks_queue_sizes_and_entry_offset() {
        let mi300x_node = "\
gpu_id 51200
simd_count 1216
simd_per_cu 4
max_waves_per_simd 8
lds_size_in_kb 64
gfx_target_version 90402
num_xcc 8
local_mem_size 0
";
        let props = parse_kfd_topology_properties(mi300x_node).expect("MI300X props");
        assert_eq!(props.num_xcc, 8);
        assert_eq!(props.vram_bytes, 0);

        // Fallback mem_banks parsing: public HBM + ignores LDS bank -> 192 GiB
        let fb_bank = "\
heap_type 1
size_in_bytes 206158430208
flags 0
width 8192
";
        let lds_bank = "\
heap_type 4
size_in_bytes 65536
";
        let total_vram =
            parse_kfd_mem_bank_properties(fb_bank) + parse_kfd_mem_bank_properties(lds_bank);
        assert_eq!(total_vram, 192 * 1024 * 1024 * 1024);

        let q_sizes = compute_kfd_queue_buffer_sizes(304, 32, props.num_xcc);
        assert_eq!(q_sizes.eop_bytes, 8 * 4096);
        assert_eq!(q_sizes.ctl_stack_bytes % (8 * 4096), 0);
        assert_eq!(q_sizes.ctx_save_restore_bytes % (8 * 4096), 0);
        assert!(q_sizes.ctx_save_restore_bytes > q_sizes.ctl_stack_bytes + 304 * 1024 * 1024);

        // Raw ld.lld layout (.rodata @ 0x1E80, .text @ 0x3000 len 0x2000; f16 @ 0x3900 from kd[1] @ 0x1EC0)
        assert_eq!(
            resolve_kernel_text_entry_offset(0x1E80, 0x3000, 0x2000, 0, 0x1180),
            0x0
        );
        assert_eq!(
            resolve_kernel_text_entry_offset(0x1E80, 0x3000, 0x2000, 64, 0x1A40),
            0x0900
        );

        // Repacked AMDA HSACO (.text @ 0x1000 len 0x2000, .rodata @ 0x3000; already .text-relative)
        assert_eq!(
            resolve_kernel_text_entry_offset(0x3000, 0x1000, 0x2000, 0, 0),
            0x0
        );
        assert_eq!(
            resolve_kernel_text_entry_offset(0x3000, 0x1000, 0x2000, 64, 0x0900),
            0x0900
        );
    }
}
