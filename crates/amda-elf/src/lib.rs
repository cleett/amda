//! AMDGPU HSACO ELF64 parser/builder and `amdhsa` MessagePack metadata codec.

pub mod descriptor;
pub mod elf64;
pub mod error;
pub mod metadata;

pub use descriptor::{
    AmdhsaKernelDescriptor, AMDHSA_PROP_ENABLE_SGPR_DISPATCH_PTR,
    AMDHSA_PROP_ENABLE_SGPR_KERNARG_SEGMENT_PTR, AMDHSA_PROP_ENABLE_WAVEFRONT_SIZE32,
};
pub use elf64::{
    build_hsaco, build_hsaco_with_rodata, parse_hsaco, ElfSection, HsacoImage, IsaTarget,
    ELF_MAGIC, EM_AMDGPU, NT_AMDGPU_METADATA,
};
pub use error::ElfError;
pub use metadata::{
    decode_amdhsa_metadata, encode_amdhsa_metadata, KernelArgMetadata, KernelMetadata, ValueKind,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_kernels() -> Vec<KernelMetadata> {
        vec![KernelMetadata {
            name: "vector_add".to_owned(),
            symbol: "vector_add.kd".to_owned(),
            kernarg_segment_size: 28,
            group_segment_fixed_size: 256,
            private_segment_fixed_size: 0,
            wavefront_size: 32,
            max_flat_workgroup_size: 1024,
            args: vec![
                KernelArgMetadata {
                    name: "a".to_owned(),
                    offset: 0,
                    size: 8,
                    value_kind: ValueKind::GlobalBuffer,
                },
                KernelArgMetadata {
                    name: "b".to_owned(),
                    offset: 8,
                    size: 8,
                    value_kind: ValueKind::GlobalBuffer,
                },
                KernelArgMetadata {
                    name: "c".to_owned(),
                    offset: 16,
                    size: 8,
                    value_kind: ValueKind::GlobalBuffer,
                },
                KernelArgMetadata {
                    name: "n".to_owned(),
                    offset: 24,
                    size: 4,
                    value_kind: ValueKind::ByValue,
                },
            ],
        }]
    }

    #[test]
    fn round_trip_hsaco_and_metadata() {
        let kernels = sample_kernels();
        let code = [0x00, 0x00, 0x81, 0xBF, 0x01, 0x02, 0x03, 0x04];
        let ir_payload = b"AMDA_IR_TEST_PAYLOAD";

        let hsaco = build_hsaco(IsaTarget::Gfx1100, &kernels, &code, Some(ir_payload));
        assert_eq!(&hsaco[0..4], &ELF_MAGIC);

        let parsed = parse_hsaco(&hsaco).expect("HSACO should parse cleanly");
        assert_eq!(parsed.isa, IsaTarget::Gfx1100);
        assert_eq!(parsed.kernels, kernels);
        assert_eq!(parsed.amda_ir_payload.as_deref(), Some(ir_payload.as_slice()));

        let text_sec = parsed
            .sections
            .iter()
            .find(|s| s.name == ".text")
            .expect(".text section must exist");
        assert_eq!(text_sec.data, code);
    }

    #[test]
    fn round_trip_cdna_gfx942_flags() {
        let mut kernels = sample_kernels();
        kernels[0].wavefront_size = 64;
        let hsaco = build_hsaco(IsaTarget::Gfx942, &kernels, &[], None);
        let parsed = parse_hsaco(&hsaco).expect("gfx942 HSACO should parse");
        assert_eq!(parsed.isa, IsaTarget::Gfx942);
        assert_eq!(parsed.isa.native_wave_size(), 64);
        assert!(parsed.amda_ir_payload.is_none());
    }

    #[test]
    fn rejects_invalid_magic_and_machine() {
        assert_eq!(parse_hsaco(b"not an elf"), Err(ElfError::Truncated));

        let mut bad_magic = build_hsaco(IsaTarget::Gfx1100, &sample_kernels(), &[], None);
        bad_magic[0] = b'X';
        assert_eq!(parse_hsaco(&bad_magic), Err(ElfError::InvalidMagic));

        let mut bad_machine = build_hsaco(IsaTarget::Gfx1100, &sample_kernels(), &[], None);
        bad_machine[18] = 62; // EM_X86_64
        bad_machine[19] = 0;
        assert_eq!(
            parse_hsaco(&bad_machine),
            Err(ElfError::UnsupportedMachine(62))
        );
    }

    #[test]
    fn round_trip_amdhsa_kernel_descriptor_and_rodata() {
        let kernels = sample_kernels();
        let desc_rdna = AmdhsaKernelDescriptor::new(IsaTarget::Gfx1100, 12, 16, 256, 28, -4096);
        let raw_rdna = desc_rdna.to_bytes();
        let decoded_rdna = AmdhsaKernelDescriptor::from_bytes(&raw_rdna);
        assert_eq!(decoded_rdna, desc_rdna);
        assert!(decoded_rdna.is_wave32());
        assert_eq!(decoded_rdna.granulated_vgprs(), 1); // ceil(12/8)-1 = 1
        assert_eq!(decoded_rdna.kernel_code_entry_byte_offset, -4096);

        let desc_cdna1 = AmdhsaKernelDescriptor::new(IsaTarget::Gfx908, 12, 16, 256, 28, -4096);
        assert!(!desc_cdna1.is_wave32());
        assert_eq!(desc_cdna1.granulated_vgprs(), 2); // ceil(12/4)-1 = 2 on gfx908
        assert_eq!(desc_cdna1.granulated_sgprs(), 1); // ceil(16/8)-1 = 1
        assert_eq!(desc_cdna1.compute_pgm_rsrc3, 0);

        let desc_mi300x = AmdhsaKernelDescriptor::new(IsaTarget::Gfx942, 56, 30, 0, 76, 0);
        assert!(!desc_mi300x.is_wave32());
        assert_eq!(desc_mi300x.granulated_vgprs(), 6); // ceil(56/8)-1 = 6 on gfx942
        assert_eq!(desc_mi300x.granulated_sgprs(), 3); // ceil(30/8)-1 = 3
        assert_eq!(desc_mi300x.compute_pgm_rsrc3, 13); // ceil(56/4)-1 = 13
        assert_eq!(desc_mi300x.compute_pgm_rsrc1, 0x00AF_00C6);

        let code = 0xBFB0_0000u32.to_le_bytes();
        let hsaco = build_hsaco_with_rodata(
            IsaTarget::Gfx1100,
            &kernels,
            Some(&raw_rdna),
            &code,
            None,
        );
        let parsed = parse_hsaco(&hsaco).expect("HSACO with .rodata must parse");
        assert_eq!(parsed.kernel_descriptors, vec![desc_rdna]);
        assert!(parsed.sections.iter().any(|s| s.name == ".rodata"));
    }

    #[test]
    fn round_trip_rdna3_apu_gfx1103() {
        assert_eq!(IsaTarget::Gfx1103.to_elf_flags(), 0x044);
        assert_eq!(IsaTarget::from_str("gfx1103"), Ok(IsaTarget::Gfx1103));
        assert_eq!(IsaTarget::Gfx1103.native_wave_size(), 32);

        let desc = AmdhsaKernelDescriptor::new(IsaTarget::Gfx1103, 10, 14, 0, 16, 0);
        assert!(desc.is_wave32());

        let hsaco = build_hsaco_with_rodata(
            IsaTarget::Gfx1103,
            &sample_kernels(),
            Some(&desc.to_bytes()),
            &0xBFB0_0000u32.to_le_bytes(),
            None,
        );
        let parsed = parse_hsaco(&hsaco).expect("gfx1103 HSACO should parse");
        assert_eq!(parsed.isa, IsaTarget::Gfx1103);
        assert_eq!(parsed.kernel_descriptors, vec![desc]);
    }
}
