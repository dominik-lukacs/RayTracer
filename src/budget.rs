pub const MAX_STEPS_PER_PIXEL: f32 = 1024.0;
const MIN_STEPS_PER_PIXEL: f32 = 1.0 / 4096.0;

pub struct Budget {
    pub target_hz: f32,
    pub gpu_fraction: f32,

    steps_per_pixel: f32,
    ms_per_step: f32,
    frame_time_ema: f32,
}

impl Budget {
    pub fn new(target_hz: f32) -> Self {
        Self {
            target_hz,
            gpu_fraction: 0.6,
            steps_per_pixel: 0.25,
            ms_per_step: 0.0,
            frame_time_ema: 0.0,
        }
    }

    pub fn steps_per_pixel(&self) -> f32 {
        self.steps_per_pixel
    }

    pub fn target_ms(&self) -> f32 {
        1000.0 / self.target_hz.max(1.0) * self.gpu_fraction
    }

    pub fn invalidate(&mut self) {
        self.ms_per_step = 0.0;
        self.steps_per_pixel = self.steps_per_pixel.min(0.25);
    }

    pub fn on_gpu_sample(&mut self, work: f32, gpu_ms: f32) {
        if work <= 0.0 || gpu_ms <= 0.0 {
            return;
        }
        let cost = gpu_ms / work;
        if self.ms_per_step == 0.0 {
            self.ms_per_step = cost;
        } else if cost > self.ms_per_step * 1.3 {
            self.ms_per_step = cost;
        } else {
            self.ms_per_step += 0.25 * (cost - self.ms_per_step);
        }

        let desired = self.target_ms() / self.ms_per_step;
        self.steps_per_pixel = if desired < self.steps_per_pixel {
            desired
        } else {
            desired.min(self.steps_per_pixel * 1.25)
        }
        .clamp(MIN_STEPS_PER_PIXEL, MAX_STEPS_PER_PIXEL);
    }

    pub fn on_frame_time(&mut self, dt: f32) {
        self.frame_time_ema = if self.frame_time_ema == 0.0 {
            dt
        } else {
            self.frame_time_ema + 0.1 * (dt - self.frame_time_ema)
        };
        let target = 1.0 / self.target_hz.max(1.0);
        if self.frame_time_ema > target * 1.04 {
            self.steps_per_pixel *= 0.92;
        } else if self.frame_time_ema < target * 1.015 {
            self.steps_per_pixel *= 1.01;
        }
        self.steps_per_pixel = self.steps_per_pixel.clamp(MIN_STEPS_PER_PIXEL, MAX_STEPS_PER_PIXEL);
    }
}
