//! Module loader, kernel descriptor wrapper, and kernarg buffer packing.

use crate::error::RuntimeError;
use amda_compiler::{CompileTier, Compiler, IrKernel, IrModule, RegAllocSummary};
use amda_elf::{parse_hsaco, IsaTarget, KernelMetadata};
use amda_hal::{CodeObjectId, DeviceBackend, KernelHandle};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolved executable kernel inside a [`LoadedModule`].
#[derive(Debug, Clone)]
pub struct KernelFunction {
    pub name: String,
    pub handle: KernelHandle,
    pub metadata: KernelMetadata,
    pub ir_kernel: Option<Arc<IrKernel>>,
    pub tier: CompileTier,
    pub regalloc: Option<RegAllocSummary>,
}

/// Loaded module managing its HSACO code object and resolved kernel entry points.
pub struct LoadedModule {
    pub code_object: CodeObjectId,
    pub tier: CompileTier,
    pub kernels: HashMap<String, KernelFunction>,
}

impl LoadedModule {
    /// Compiles/validates `data` (HSACO, fatbin, or PTX), loads it onto `device`, and indexes kernels.
    pub fn load(
        backend: &dyn DeviceBackend,
        compiler: &Compiler,
        device: usize,
        isa: IsaTarget,
        data: &[u8],
    ) -> Result<Self, RuntimeError> {
        let artifact = compiler.compile_module_artifact(data, isa)?;
        let image = parse_hsaco(&artifact.hsaco)?;
        let ir_mod = image
            .amda_ir_payload
            .as_deref()
            .and_then(|bytes| IrModule::from_bytes(bytes).ok());

        let code_object = backend.load_code_object(device, &artifact.hsaco)?;
        let mut kernels = HashMap::with_capacity(image.kernels.len());

        for meta in image.kernels {
            let (handle, resolved_meta) = backend.get_kernel(code_object, &meta.name)?;
            let ir_kernel = ir_mod.as_ref().and_then(|m| {
                m.kernels
                    .iter()
                    .find(|k| k.name == resolved_meta.name)
                    .cloned()
                    .map(Arc::new)
            });
            let regalloc = artifact
                .regalloc
                .iter()
                .find(|r| r.kernel_name == resolved_meta.name)
                .cloned();
            kernels.insert(
                resolved_meta.name.clone(),
                KernelFunction {
                    name: resolved_meta.name.clone(),
                    handle,
                    metadata: resolved_meta,
                    ir_kernel,
                    tier: artifact.tier,
                    regalloc,
                },
            );
        }

        Ok(Self {
            code_object,
            tier: artifact.tier,
            kernels,
        })
    }

    /// Looks up a kernel entry point by name.
    pub fn get_function(&self, name: &str) -> Result<KernelFunction, RuntimeError> {
        self.kernels
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::KernelNotFound(name.to_owned()))
    }
}

/// Packs ordered argument byte slices into an AMDGPU `kernarg` segment buffer
/// matching `meta.args` offsets and `meta.kernarg_segment_size`.
pub fn pack_kernargs(meta: &KernelMetadata, arg_slices: &[&[u8]]) -> Result<Vec<u8>, RuntimeError> {
    if arg_slices.len() < meta.args.len() {
        return Err(RuntimeError::InvalidArgument(format!(
            "kernel '{}' expects {} arguments, got {}",
            meta.name,
            meta.args.len(),
            arg_slices.len()
        )));
    }

    let mut buf = vec![0u8; meta.kernarg_segment_size as usize];
    for (arg_meta, bytes) in meta.args.iter().zip(arg_slices.iter()) {
        let start = arg_meta.offset as usize;
        let size = arg_meta.size as usize;
        let end = start + size;
        if end > buf.len() {
            return Err(RuntimeError::InvalidArgument(format!(
                "kernel '{}' argument '{}' exceeds kernarg segment size",
                meta.name, arg_meta.name
            )));
        }
        let copy_len = bytes.len().min(size);
        buf[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
    }
    Ok(buf)
}
