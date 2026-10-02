pub struct GuiApp {}

impl GuiApp {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ctx: &egui::Context, fps: f32, frame_time: f32) {
        egui::Window::new("FPS")
        .anchor(egui::Align2::LEFT_TOP, egui::Vec2::new(10.0, 10.0))
        .title_bar(false)
        .resizable(false)
        .interactable(false)
        .collapsible(false)
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            ui.label(format!("FPS: {:.2}", fps));
            ui.label(format!("Frame Time: {:.2} ms", frame_time * 1000.0));
        });
    }
}
