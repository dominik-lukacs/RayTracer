pub struct Settings {
    pub max_bounces: u32,
    pub target_hz: f32,
    pub gpu_fraction: f32,
}

pub struct Stats {
    pub fps: f32,
    pub frame_time: f32,
    pub gpu_ms: Option<f32>,
    pub steps_per_pixel: f32,
    pub total_steps: f64,
}

pub struct GuiApp {}

impl GuiApp {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ctx: &egui::Context, stats: &Stats, settings: &mut Settings) -> bool {
        let mut reset = false;
        
        egui::Window::new("FPS")
        .anchor(egui::Align2::LEFT_TOP, egui::Vec2::new(10.0, 10.0))
        .title_bar(false)
        .resizable(false)
        .interactable(false)
        .collapsible(false)
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            ui.label(format!("FPS: {:.2}", stats.fps));
            ui.label(format!("Frame Time: {:.2} ms", stats.frame_time * 1000.0));
            match stats.gpu_ms {
                Some(ms) => ui.label(format!("Ray tracing GPU: {:.2} ms", ms)),
                None => ui.label("Ray tracing GPU: n/a (no timestamp queries)"),
            };
            ui.label(format!("Budget: {:.2} bounce-steps/px/frame", stats.steps_per_pixel));
            ui.label(format!("Accumulated: {:.0} bounce-steps/px", stats.total_steps));
        });

        egui::Window::new("Render")
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-10.0, 10.0))
        .resizable(false)
        .show(ctx, |ui| {
            if ui.add(egui::Slider::new(&mut settings.max_bounces, 1..=1000).logarithmic(true).text("Max bounces")).changed()
            {
                reset = true;
            }
            ui.add(egui::Slider::new(&mut settings.target_hz, 30.0..=360.0).text("Target Hz"));
            ui.add(egui::Slider::new(&mut settings.gpu_fraction, 0.1..=0.95).text("GPU budget"));
            if ui.button("Restart accumulation").clicked() {
                reset = true;
            }
        });

        reset
    }
}
