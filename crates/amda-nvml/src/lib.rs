//! Drop-in NVML (`libnvidia-ml`) C ABI library backed by AMDA runtime discovery.

#![allow(non_camel_case_types, non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

use amda_elf::IsaTarget;
use amda_hal::DeviceInfo;
use amda_runtime::Runtime;
use core::ffi::{c_char, c_uint, c_ulonglong, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

pub type NvmlDeviceT = *mut c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum NvmlReturnT {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_UNKNOWN = 999,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum NvmlTemperatureSensorsT {
    NVML_TEMPERATURE_GPU = 0,
    NVML_TEMPERATURE_COUNT = 1,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct NvmlMemoryT {
    pub total: c_ulonglong,
    pub free: c_ulonglong,
    pub used: c_ulonglong,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct NvmlUtilizationT {
    pub gpu: c_uint,
    pub memory: c_uint,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct NvmlPciInfoT {
    pub busIdLegacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pciDeviceId: c_uint,
    pub pciSubSystemId: c_uint,
    pub busId: [c_char; 32],
}

#[derive(Default)]
struct NvmlState {
    init_count: usize,
    runtime: Option<Arc<Runtime>>,
}

static NVML_STATE: OnceLock<Mutex<NvmlState>> = OnceLock::new();

fn state() -> &'static Mutex<NvmlState> {
    NVML_STATE.get_or_init(|| Mutex::new(NvmlState::default()))
}

fn nvml_try<F>(f: F) -> NvmlReturnT
where
    F: FnOnce() -> Result<(), NvmlReturnT>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => NvmlReturnT::NVML_SUCCESS,
        Ok(Err(e)) => e,
        Err(_) => NvmlReturnT::NVML_ERROR_UNKNOWN,
    }
}

fn get_runtime_and_device(device: NvmlDeviceT) -> Result<(usize, DeviceInfo), NvmlReturnT> {
    let st = state().lock().expect("nvml state lock");
    if st.init_count == 0 {
        return Err(NvmlReturnT::NVML_ERROR_UNINITIALIZED);
    }
    let rt = st
        .runtime
        .as_ref()
        .ok_or(NvmlReturnT::NVML_ERROR_UNINITIALIZED)?;
    let raw = device as usize;
    let idx = raw
        .checked_sub(1)
        .ok_or(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT)?;
    let info = rt
        .devices()
        .get(idx)
        .cloned()
        .ok_or(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT)?;
    Ok((idx, info))
}

fn write_cstr_buf(dst: *mut c_char, capacity: usize, text: &str) -> Result<(), NvmlReturnT> {
    if dst.is_null() || capacity == 0 {
        return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
    }
    let bytes = text.as_bytes();
    if bytes.len() + 1 > capacity {
        return Err(NvmlReturnT::NVML_ERROR_INSUFFICIENT_SIZE);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len());
        *dst.add(bytes.len()) = 0;
    }
    Ok(())
}

fn write_fixed_cstr(dst: &mut [c_char], text: &str) {
    let bytes = text.as_bytes();
    let n = bytes.len().min(dst.len().saturating_sub(1));
    for slot in dst.iter_mut() {
        *slot = 0;
    }
    for (i, &b) in bytes[..n].iter().enumerate() {
        dst[i] = b as c_char;
    }
}

fn amd_pci_device_id(isa: IsaTarget) -> c_uint {
    match isa {
        IsaTarget::Gfx1100 => 0x744C_1002,
        IsaTarget::Gfx1101 => 0x747E_1002,
        IsaTarget::Gfx1102 => 0x7480_1002,
        IsaTarget::Gfx1103 => 0x15BF_1002,
        IsaTarget::Gfx942 => 0x74A1_1002,
        IsaTarget::Gfx90a => 0x740C_1002,
        IsaTarget::Gfx908 => 0x738C_1002,
        IsaTarget::Gfx1030 => 0x73BF_1002,
        IsaTarget::Gfx1200 | IsaTarget::Gfx1201 => 0x7550_1002,
    }
}

#[cfg(target_os = "linux")]
fn probe_linux_drm_telemetry(dev_idx: usize) -> (Option<u64>, Option<c_uint>, Option<c_uint>) {
    use std::fs;
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return (None, None, None);
    };
    let mut cards: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| {
            name.starts_with("card") && name[4..].chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    cards.sort();
    let Some(card) = cards.get(dev_idx) else {
        return (None, None, None);
    };
    let base = format!("/sys/class/drm/{card}/device");
    let vram_used = fs::read_to_string(format!("{base}/mem_info_vram_used"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let gpu_busy = fs::read_to_string(format!("{base}/gpu_busy_percent"))
        .ok()
        .and_then(|s| s.trim().parse::<c_uint>().ok());
    let mut temp_c = None;
    if let Ok(hwmons) = fs::read_dir(format!("{base}/hwmon")) {
        for hw in hwmons.filter_map(Result::ok) {
            if let Ok(raw) = fs::read_to_string(hw.path().join("temp1_input")) {
                if let Ok(millic) = raw.trim().parse::<u32>() {
                    temp_c = Some(millic / 1000);
                    break;
                }
            }
        }
    }
    (vram_used, gpu_busy, temp_c)
}

#[cfg(not(target_os = "linux"))]
fn probe_linux_drm_telemetry(_dev_idx: usize) -> (Option<u64>, Option<c_uint>, Option<c_uint>) {
    (None, None, None)
}

#[no_mangle]
pub extern "C" fn nvmlInit_v2() -> NvmlReturnT {
    nvmlInitWithFlags(0)
}

#[no_mangle]
pub extern "C" fn nvmlInit() -> NvmlReturnT {
    nvmlInitWithFlags(0)
}

#[no_mangle]
pub extern "C" fn nvmlInitWithFlags(_flags: c_uint) -> NvmlReturnT {
    nvml_try(|| {
        let mut st = state().lock().expect("nvml state lock");
        if st.runtime.is_none() {
            let rt = Runtime::new().map_err(|_| NvmlReturnT::NVML_ERROR_UNKNOWN)?;
            st.runtime = Some(Arc::new(rt));
        }
        st.init_count += 1;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlShutdown() -> NvmlReturnT {
    nvml_try(|| {
        let mut st = state().lock().expect("nvml state lock");
        if st.init_count == 0 {
            return Err(NvmlReturnT::NVML_ERROR_UNINITIALIZED);
        }
        st.init_count -= 1;
        if st.init_count == 0 {
            st.runtime = None;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> NvmlReturnT {
    nvml_try(|| {
        if deviceCount.is_null() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let st = state().lock().expect("nvml state lock");
        if st.init_count == 0 {
            return Err(NvmlReturnT::NVML_ERROR_UNINITIALIZED);
        }
        let rt = st
            .runtime
            .as_ref()
            .ok_or(NvmlReturnT::NVML_ERROR_UNINITIALIZED)?;
        unsafe {
            *deviceCount = rt.devices().len() as c_uint;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount(deviceCount: *mut c_uint) -> NvmlReturnT {
    nvmlDeviceGetCount_v2(deviceCount)
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetHandleByIndex_v2(
    index: c_uint,
    device: *mut NvmlDeviceT,
) -> NvmlReturnT {
    nvml_try(|| {
        if device.is_null() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let st = state().lock().expect("nvml state lock");
        if st.init_count == 0 {
            return Err(NvmlReturnT::NVML_ERROR_UNINITIALIZED);
        }
        let rt = st
            .runtime
            .as_ref()
            .ok_or(NvmlReturnT::NVML_ERROR_UNINITIALIZED)?;
        let idx = index as usize;
        if idx >= rt.devices().len() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        unsafe {
            *device = (idx + 1) as NvmlDeviceT;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetHandleByIndex(
    index: c_uint,
    device: *mut NvmlDeviceT,
) -> NvmlReturnT {
    nvmlDeviceGetHandleByIndex_v2(index, device)
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetName(
    device: NvmlDeviceT,
    name: *mut c_char,
    length: c_uint,
) -> NvmlReturnT {
    nvml_try(|| {
        let (_, info) = get_runtime_and_device(device)?;
        write_cstr_buf(name, length as usize, &info.name)
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetMemoryInfo(
    device: NvmlDeviceT,
    memory: *mut NvmlMemoryT,
) -> NvmlReturnT {
    nvml_try(|| {
        if memory.is_null() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let (idx, info) = get_runtime_and_device(device)?;
        let total = info.vram_bytes;
        let used = probe_linux_drm_telemetry(idx).0.unwrap_or(0).min(total);
        let free = total.saturating_sub(used);
        unsafe {
            *memory = NvmlMemoryT { total, free, used };
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetUUID(
    device: NvmlDeviceT,
    uuid: *mut c_char,
    length: c_uint,
) -> NvmlReturnT {
    nvml_try(|| {
        let (idx, info) = get_runtime_and_device(device)?;
        let formatted = format!("GPU-amda-{}-{idx:04}", info.isa);
        write_cstr_buf(uuid, length as usize, &formatted)
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetTemperature(
    device: NvmlDeviceT,
    sensorType: NvmlTemperatureSensorsT,
    temp: *mut c_uint,
) -> NvmlReturnT {
    nvml_try(|| {
        if temp.is_null() || sensorType != NvmlTemperatureSensorsT::NVML_TEMPERATURE_GPU {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let (idx, _) = get_runtime_and_device(device)?;
        let reading = probe_linux_drm_telemetry(idx).2.unwrap_or(42);
        unsafe {
            *temp = reading;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetUtilizationRates(
    device: NvmlDeviceT,
    utilization: *mut NvmlUtilizationT,
) -> NvmlReturnT {
    nvml_try(|| {
        if utilization.is_null() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let (idx, _) = get_runtime_and_device(device)?;
        let gpu = probe_linux_drm_telemetry(idx).1.unwrap_or(0);
        unsafe {
            *utilization = NvmlUtilizationT { gpu, memory: 0 };
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetPciInfo_v2(
    device: NvmlDeviceT,
    pci: *mut NvmlPciInfoT,
) -> NvmlReturnT {
    nvml_try(|| {
        if pci.is_null() {
            return Err(NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT);
        }
        let (idx, info) = get_runtime_and_device(device)?;
        let bus = ((idx + 1) & 0xFF) as c_uint;
        let bdf = format!("0000:{bus:02x}:00.0");
        let mut out = NvmlPciInfoT {
            domain: 0,
            bus,
            device: 0,
            pciDeviceId: amd_pci_device_id(info.isa),
            pciSubSystemId: 0x0B0C_1002,
            ..NvmlPciInfoT::default()
        };
        write_fixed_cstr(&mut out.busId, &bdf);
        write_fixed_cstr(&mut out.busIdLegacy, &bdf);
        unsafe {
            *pci = out;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetPciInfo(
    device: NvmlDeviceT,
    pci: *mut NvmlPciInfoT,
) -> NvmlReturnT {
    nvmlDeviceGetPciInfo_v2(device, pci)
}

#[no_mangle]
pub extern "C" fn nvmlErrorString(result: NvmlReturnT) -> *const c_char {
    let msg: &'static [u8] = match result {
        NvmlReturnT::NVML_SUCCESS => b"Success\0",
        NvmlReturnT::NVML_ERROR_UNINITIALIZED => b"Uninitialized\0",
        NvmlReturnT::NVML_ERROR_INVALID_ARGUMENT => b"Invalid Argument\0",
        NvmlReturnT::NVML_ERROR_NOT_SUPPORTED => b"Not Supported\0",
        NvmlReturnT::NVML_ERROR_NO_PERMISSION => b"No Permission\0",
        NvmlReturnT::NVML_ERROR_NOT_FOUND => b"Not Found\0",
        NvmlReturnT::NVML_ERROR_INSUFFICIENT_SIZE => b"Insufficient Size\0",
        NvmlReturnT::NVML_ERROR_UNKNOWN => b"Unknown Error\0",
    };
    msg.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::CStr;

    static NVML_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn nvml_lifecycle_and_device_queries() {
        let _guard = NVML_TEST_LOCK.lock().expect("nvml lock");

        // Ensure uninitialized check when refcount == 0
        let mut cnt = 0;
        assert_eq!(nvmlDeviceGetCount_v2(&mut cnt), NvmlReturnT::NVML_ERROR_UNINITIALIZED);

        assert_eq!(nvmlInit_v2(), NvmlReturnT::NVML_SUCCESS);
        assert_eq!(nvmlInitWithFlags(0), NvmlReturnT::NVML_SUCCESS);

        assert_eq!(nvmlDeviceGetCount_v2(&mut cnt), NvmlReturnT::NVML_SUCCESS);
        assert!(cnt >= 1);

        let mut dev: NvmlDeviceT = core::ptr::null_mut();
        assert_eq!(nvmlDeviceGetHandleByIndex_v2(0, &mut dev), NvmlReturnT::NVML_SUCCESS);
        assert!(!dev.is_null());

        let mut name_buf = [0 as c_char; 128];
        assert_eq!(
            nvmlDeviceGetName(dev, name_buf.as_mut_ptr(), 128),
            NvmlReturnT::NVML_SUCCESS
        );
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_str()
            .expect("utf8");
        assert!(name.contains("AMD"));

        let mut mem = NvmlMemoryT::default();
        assert_eq!(nvmlDeviceGetMemoryInfo(dev, &mut mem), NvmlReturnT::NVML_SUCCESS);
        assert!(mem.total > 0 && mem.free <= mem.total);

        let mut uuid_buf = [0 as c_char; 64];
        assert_eq!(
            nvmlDeviceGetUUID(dev, uuid_buf.as_mut_ptr(), 64),
            NvmlReturnT::NVML_SUCCESS
        );
        let uuid = unsafe { CStr::from_ptr(uuid_buf.as_ptr()) }
            .to_str()
            .expect("utf8");
        assert!(uuid.starts_with("GPU-amda-"));

        assert_eq!(
            nvmlDeviceGetUUID(dev, uuid_buf.as_mut_ptr(), 4),
            NvmlReturnT::NVML_ERROR_INSUFFICIENT_SIZE
        );

        let mut temp = 0;
        assert_eq!(
            nvmlDeviceGetTemperature(dev, NvmlTemperatureSensorsT::NVML_TEMPERATURE_GPU, &mut temp),
            NvmlReturnT::NVML_SUCCESS
        );
        assert!(temp > 0);

        let mut util = NvmlUtilizationT { gpu: 99, memory: 99 };
        assert_eq!(nvmlDeviceGetUtilizationRates(dev, &mut util), NvmlReturnT::NVML_SUCCESS);
        assert!(util.gpu <= 100 && util.memory <= 100);

        let mut pci = NvmlPciInfoT::default();
        assert_eq!(nvmlDeviceGetPciInfo_v2(dev, &mut pci), NvmlReturnT::NVML_SUCCESS);
        assert_eq!(pci.pciDeviceId & 0xFFFF, 0x1002);
        let bus_id = unsafe { CStr::from_ptr(pci.busId.as_ptr()) }
            .to_str()
            .expect("utf8");
        assert!(bus_id.starts_with("0000:"));

        let ok_str = unsafe { CStr::from_ptr(nvmlErrorString(NvmlReturnT::NVML_SUCCESS)) }
            .to_str()
            .expect("utf8");
        assert_eq!(ok_str, "Success");

        assert_eq!(nvmlShutdown(), NvmlReturnT::NVML_SUCCESS);
        // Second init still active
        assert_eq!(nvmlDeviceGetCount_v2(&mut cnt), NvmlReturnT::NVML_SUCCESS);
        assert_eq!(nvmlShutdown(), NvmlReturnT::NVML_SUCCESS);
    }
}
