use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const SLOTS: usize = 4;

struct Slot{
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    work: f32,
}

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    slots: Vec<Slot>,
    period_ns: f64,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Ray tracing timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Timestamp readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(IDLE)),
                work: 0.0,
            })
            .collect();

        Self {
            query_set,
            resolve_buffer,
            slots,
            period_ns: queue.get_timestamp_period() as f64,
        }
    }

    pub fn acquire_slot(&mut self, work: f32) -> Option<usize> {
        let idx = self.slots.iter().position(|s| s.state.load(Ordering::Acquire) == IDLE)?;
        self.slots[idx].work = work;
        Some(idx)
    }

    pub fn pass_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder, slot: usize) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.slots[slot].buffer, 0, 16);
    }

    pub fn after_submit(&mut self, slot: usize) {
        let state = self.slots[slot].state.clone();
        state.store(PENDING, Ordering::Release);
        self.slots[slot]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
    }

    pub fn collect(&mut self) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for slot in &self.slots {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            if let Ok(data) = slot.buffer.slice(..).get_mapped_range() {
                let start = u64::from_le_bytes(data[0..8].try_into().unwrap());
                let end = u64::from_le_bytes(data[8..16].try_into().unwrap());
                let ms = end.saturating_sub(start) as f64 * self.period_ns / 1.0e6;
                if ms > 0.0 {
                    out.push((slot.work, ms as f32));
                }
            }
            slot.buffer.unmap();
            slot.state.store(IDLE, Ordering::Release);
        }
        out
    }
}
