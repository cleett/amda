//! Spec-accurate 64-byte HSA AQL packets and lock-free user-space ring buffer.

use crate::error::HalError;
use core::mem::{offset_of, size_of};
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};

/// HSA AQL packet type codes (stored in bits `7:0` of the 16-bit header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AqlPacketType {
    Invalid = 0,
    VendorSpecific = 1,
    KernelDispatch = 2,
    BarrierAnd = 3,
    AgentDispatch = 4,
    BarrierOr = 5,
}

/// HSA AQL memory fence scope (acquire / release bits in header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum FenceScope {
    None = 0,
    Agent = 1,
    System = 2,
}

/// Encodes a 16-bit HSA AQL packet header word.
pub const fn encode_aql_header(
    pkt_type: AqlPacketType,
    barrier: bool,
    acquire: FenceScope,
    release: FenceScope,
) -> u16 {
    (pkt_type as u16)
        | ((barrier as u16) << 8)
        | ((acquire as u16) << 9)
        | ((release as u16) << 11)
}

/// Decodes a 16-bit AQL header into `(raw_type, barrier, acquire, release)`.
pub const fn decode_aql_header(header: u16) -> (u8, bool, FenceScope, FenceScope) {
    let ty = (header & 0xff) as u8;
    let barrier = ((header >> 8) & 1) != 0;
    let acq = match (header >> 9) & 0x3 {
        1 => FenceScope::Agent,
        2 => FenceScope::System,
        _ => FenceScope::None,
    };
    let rel = match (header >> 11) & 0x3 {
        1 => FenceScope::Agent,
        2 => FenceScope::System,
        _ => FenceScope::None,
    };
    (ty, barrier, acq, rel)
}

/// HSA `hsa_kernel_dispatch_packet_t` (exactly 64 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct KernelDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub reserved2: u64,
    pub completion_signal: u64,
}

/// HSA `hsa_barrier_and_packet_t` (exactly 64 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BarrierAndPacket {
    pub header: u16,
    pub reserved0: u16,
    pub reserved1: u32,
    pub dep_signal: [u64; 5],
    pub reserved2: u64,
    pub completion_signal: u64,
}

const _: () = assert!(size_of::<KernelDispatchPacket>() == 64);
const _: () = assert!(size_of::<BarrierAndPacket>() == 64);
const _: () = assert!(offset_of!(KernelDispatchPacket, grid_size_x) == 12);
const _: () = assert!(offset_of!(KernelDispatchPacket, kernel_object) == 32);
const _: () = assert!(offset_of!(KernelDispatchPacket, kernarg_address) == 40);
const _: () = assert!(offset_of!(KernelDispatchPacket, completion_signal) == 56);
const _: () = assert!(offset_of!(BarrierAndPacket, dep_signal) == 8);
const _: () = assert!(offset_of!(BarrierAndPacket, completion_signal) == 56);

impl KernelDispatchPacket {
    /// Encodes this struct into a raw 64-byte little-endian AQL slot payload.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..2].copy_from_slice(&self.header.to_le_bytes());
        out[2..4].copy_from_slice(&self.setup.to_le_bytes());
        out[4..6].copy_from_slice(&self.workgroup_size_x.to_le_bytes());
        out[6..8].copy_from_slice(&self.workgroup_size_y.to_le_bytes());
        out[8..10].copy_from_slice(&self.workgroup_size_z.to_le_bytes());
        out[10..12].copy_from_slice(&self.reserved0.to_le_bytes());
        out[12..16].copy_from_slice(&self.grid_size_x.to_le_bytes());
        out[16..20].copy_from_slice(&self.grid_size_y.to_le_bytes());
        out[20..24].copy_from_slice(&self.grid_size_z.to_le_bytes());
        out[24..28].copy_from_slice(&self.private_segment_size.to_le_bytes());
        out[28..32].copy_from_slice(&self.group_segment_size.to_le_bytes());
        out[32..40].copy_from_slice(&self.kernel_object.to_le_bytes());
        out[40..48].copy_from_slice(&self.kernarg_address.to_le_bytes());
        out[48..56].copy_from_slice(&self.reserved2.to_le_bytes());
        out[56..64].copy_from_slice(&self.completion_signal.to_le_bytes());
        out
    }

    /// Decodes a raw 64-byte AQL packet into a [`KernelDispatchPacket`].
    pub fn from_bytes(b: &[u8; 64]) -> Self {
        Self {
            header: u16::from_le_bytes([b[0], b[1]]),
            setup: u16::from_le_bytes([b[2], b[3]]),
            workgroup_size_x: u16::from_le_bytes([b[4], b[5]]),
            workgroup_size_y: u16::from_le_bytes([b[6], b[7]]),
            workgroup_size_z: u16::from_le_bytes([b[8], b[9]]),
            reserved0: u16::from_le_bytes([b[10], b[11]]),
            grid_size_x: u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
            grid_size_y: u32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            grid_size_z: u32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            private_segment_size: u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
            group_segment_size: u32::from_le_bytes([b[28], b[29], b[30], b[31]]),
            kernel_object: u64::from_le_bytes([b[32], b[33], b[34], b[35], b[36], b[37], b[38], b[39]]),
            kernarg_address: u64::from_le_bytes([b[40], b[41], b[42], b[43], b[44], b[45], b[46], b[47]]),
            reserved2: u64::from_le_bytes([b[48], b[49], b[50], b[51], b[52], b[53], b[54], b[55]]),
            completion_signal: u64::from_le_bytes([b[56], b[57], b[58], b[59], b[60], b[61], b[62], b[63]]),
        }
    }
}

impl BarrierAndPacket {
    /// Encodes this barrier packet into a raw 64-byte little-endian AQL slot payload.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..2].copy_from_slice(&self.header.to_le_bytes());
        out[2..4].copy_from_slice(&self.reserved0.to_le_bytes());
        out[4..8].copy_from_slice(&self.reserved1.to_le_bytes());
        for i in 0..5 {
            out[8 + i * 8..16 + i * 8].copy_from_slice(&self.dep_signal[i].to_le_bytes());
        }
        out[48..56].copy_from_slice(&self.reserved2.to_le_bytes());
        out[56..64].copy_from_slice(&self.completion_signal.to_le_bytes());
        out
    }

    /// Decodes a raw 64-byte AQL packet into a [`BarrierAndPacket`].
    pub fn from_bytes(b: &[u8; 64]) -> Self {
        let mut dep_signal = [0u64; 5];
        for i in 0..5 {
            let s = &b[8 + i * 8..16 + i * 8];
            dep_signal[i] = u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
        }
        Self {
            header: u16::from_le_bytes([b[0], b[1]]),
            reserved0: u16::from_le_bytes([b[2], b[3]]),
            reserved1: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            dep_signal,
            reserved2: u64::from_le_bytes([b[48], b[49], b[50], b[51], b[52], b[53], b[54], b[55]]),
            completion_signal: u64::from_le_bytes([b[56], b[57], b[58], b[59], b[60], b[61], b[62], b[63]]),
        }
    }
}

/// Single 64-byte atomic AQL slot enforcing HSA payload-before-header publication.
struct AqlSlot {
    header: AtomicU16,
    head_words: [AtomicU16; 3],
    tail_qwords: [AtomicU64; 7],
}

impl AqlSlot {
    fn new() -> Self {
        Self {
            header: AtomicU16::new(0),
            head_words: [AtomicU16::new(0), AtomicU16::new(0), AtomicU16::new(0)],
            tail_qwords: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    fn write_publish(&self, packet: &[u8; 64]) {
        for (i, w) in self.head_words.iter().enumerate() {
            let off = 2 + i * 2;
            w.store(u16::from_le_bytes([packet[off], packet[off + 1]]), Ordering::Relaxed);
        }
        for (i, qw) in self.tail_qwords.iter().enumerate() {
            let off = 8 + i * 8;
            let b = &packet[off..off + 8];
            qw.store(
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
                Ordering::Relaxed,
            );
        }
        let header = u16::from_le_bytes([packet[0], packet[1]]);
        // Release ensures all payload bytes [2..64] are visible to the CP before header becomes valid.
        self.header.store(header, Ordering::Release);
    }

    fn read_consume(&self) -> Option<[u8; 64]> {
        let header = self.header.load(Ordering::Acquire);
        if (header & 0xff) == AqlPacketType::Invalid as u16 {
            return None;
        }
        let mut out = [0u8; 64];
        out[0..2].copy_from_slice(&header.to_le_bytes());
        for (i, w) in self.head_words.iter().enumerate() {
            let off = 2 + i * 2;
            out[off..off + 2].copy_from_slice(&w.load(Ordering::Relaxed).to_le_bytes());
        }
        for (i, qw) in self.tail_qwords.iter().enumerate() {
            let off = 8 + i * 8;
            out[off..off + 8].copy_from_slice(&qw.load(Ordering::Relaxed).to_le_bytes());
        }
        self.header.store(AqlPacketType::Invalid as u16, Ordering::Release);
        Some(out)
    }
}

/// Power-of-two AQL ring buffer matching the HSA user-mode queue architecture.
pub struct AqlRingBuffer {
    slots: Vec<AqlSlot>,
    mask: usize,
    write_index: AtomicU64,
    read_index: AtomicU64,
}

impl AqlRingBuffer {
    /// Creates an AQL ring buffer rounded up to the next power of two (minimum 16 slots).
    pub fn new(requested_slots: usize) -> Self {
        let cap = requested_slots.max(16).next_power_of_two();
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(AqlSlot::new());
        }
        Self {
            slots,
            mask: cap - 1,
            write_index: AtomicU64::new(0),
            read_index: AtomicU64::new(0),
        }
    }

    /// Capacity in 64-byte AQL packets.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Current producer write index.
    pub fn write_index(&self) -> u64 {
        self.write_index.load(Ordering::Acquire)
    }

    /// Current consumer read index.
    pub fn read_index(&self) -> u64 {
        self.read_index.load(Ordering::Acquire)
    }

    /// Writes a 64-byte packet into the ring using HSA Release publication and returns the slot index.
    pub fn push_packet(&self, packet: &[u8; 64]) -> Result<u64, HalError> {
        let write = self.write_index.load(Ordering::Relaxed);
        let read = self.read_index.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.slots.len() as u64 {
            return Err(HalError::QueueFull);
        }
        let slot_idx = (write as usize) & self.mask;
        self.slots[slot_idx].write_publish(packet);
        self.write_index.store(write + 1, Ordering::Release);
        Ok(write)
    }

    /// Pops the next published AQL packet for the command processor.
    pub fn pop_packet(&self) -> Option<[u8; 64]> {
        let read = self.read_index.load(Ordering::Relaxed);
        let write = self.write_index.load(Ordering::Acquire);
        if read >= write {
            return None;
        }
        let slot_idx = (read as usize) & self.mask;
        let pkt = self.slots[slot_idx].read_consume()?;
        self.read_index.store(read + 1, Ordering::Release);
        Some(pkt)
    }
}
