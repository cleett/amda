//! Streams (wrapping HAL AQL queues) and Events (wrapping HSA signals & timestamps).

use crate::error::RuntimeError;
use amda_hal::{
    encode_aql_header, AqlPacketType, BarrierAndPacket, DeviceBackend, FenceScope, QueueId,
    SignalId,
};
use std::sync::Mutex;
use std::time::Instant;

/// Execution stream backed by a dedicated hardware AQL queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    pub queue_id: QueueId,
    pub flags: u32,
}

impl Stream {
    /// Enqueues a barrier packet with a fresh signal and waits for completion.
    pub fn synchronize(&self, backend: &dyn DeviceBackend) -> Result<(), RuntimeError> {
        let sig = backend.create_signal(1)?;
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
            completion_signal: sig.0,
        };
        let submit_res = backend.submit_aql_packet(self.queue_id, &pkt.to_bytes());
        let wait_res = if submit_res.is_ok() {
            backend.signal_wait(sig, 0)
        } else {
            Ok(())
        };
        let _ = backend.destroy_signal(sig);
        submit_res?;
        wait_res?;
        Ok(())
    }

    /// Enqueues a hardware barrier packet waiting for `event`'s signal before proceeding.
    pub fn wait_event(&self, backend: &dyn DeviceBackend, event: &Event) -> Result<(), RuntimeError> {
        let pkt = BarrierAndPacket {
            header: encode_aql_header(
                AqlPacketType::BarrierAnd,
                true,
                FenceScope::System,
                FenceScope::System,
            ),
            reserved0: 0,
            reserved1: 0,
            dep_signal: [event.signal_id.0, 0, 0, 0, 0],
            reserved2: 0,
            completion_signal: 0,
        };
        backend.submit_aql_packet(self.queue_id, &pkt.to_bytes())?;
        Ok(())
    }
}

/// Synchronization and timing event backed by an HSA hardware signal.
pub struct Event {
    pub signal_id: SignalId,
    pub flags: u32,
    recorded_at: Mutex<Option<Instant>>,
}

impl Event {
    /// Creates a new event wrapping `signal_id`.
    pub fn new(signal_id: SignalId, flags: u32) -> Self {
        Self {
            signal_id,
            flags,
            recorded_at: Mutex::new(None),
        }
    }

    /// Records the event on `stream` by submitting an AQL barrier referencing its signal.
    pub fn record(
        &self,
        backend: &dyn DeviceBackend,
        stream: Stream,
    ) -> Result<(), RuntimeError> {
        backend.signal_store(self.signal_id, 1)?;
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
            completion_signal: self.signal_id.0,
        };
        backend.submit_aql_packet(stream.queue_id, &pkt.to_bytes())?;
        *self.recorded_at.lock().expect("event timestamp lock") = Some(Instant::now());
        Ok(())
    }

    /// Blocks until the event has completed.
    pub fn synchronize(&self, backend: &dyn DeviceBackend) -> Result<(), RuntimeError> {
        backend.signal_wait(self.signal_id, 0)?;
        Ok(())
    }

    /// Queries non-blockingly whether the event's completion signal has reached `0`.
    pub fn query(&self, backend: &dyn DeviceBackend) -> Result<bool, RuntimeError> {
        Ok(backend.signal_load(self.signal_id)? == 0)
    }

    /// Computes elapsed milliseconds between `start` and `end`.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32, RuntimeError> {
        let t0 = *start.recorded_at.lock().expect("event timestamp lock");
        let t1 = *end.recorded_at.lock().expect("event timestamp lock");
        match (t0, t1) {
            (Some(s), Some(e)) => {
                let dur = e.saturating_duration_since(s);
                Ok((dur.as_secs_f64() * 1000.0) as f32)
            }
            _ => Err(RuntimeError::NotReady),
        }
    }
}
