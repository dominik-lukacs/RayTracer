use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent::{
        CloseRequested, RedrawRequested, Resized, ScaleFactorChanged
    },
    window::{Window, WindowAttributes, WindowId}
};
use crate::renderer::Renderer;
use crate::scene::Scene;

pub struct App {
    pub renderer: Option<Renderer>,
    pub scene: Arc<Scene>,
    pub window: Option<Arc<Window>>,
    pub window_id: Option<WindowId>,
    pub start_time: std::time::Instant,
    pub last_time: std::time::Instant,
}


impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_inner_size(LogicalSize::new(800, 600))
            .with_title("GPU Ray Tracer");
        let window = event_loop.create_window(window_attributes).unwrap();
        self.window_id = Some(window.id());
        let window = Arc::new(window);
        self.window = Some(window.clone());

        let renderer = pollster::block_on(Renderer::new(event_loop.owned_display_handle(), window.clone(), self.scene.clone()));
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    )
    {
        // Return if it's not our window's event
        if self.window_id != Some(window_id) { return; }
        let Some(renderer) = self.renderer.as_mut() else { return };

        let consumed = renderer.handle_egui_event(&event);

        match event {
            CloseRequested => event_loop.exit(),
            Resized(physical_size) => {
                renderer.resize(physical_size);
            },
            ScaleFactorChanged { scale_factor: _, .. } => {
                // TODO we don't get the inner size anymore, winit sends another Resized event afterwards
                // If our renderer ever needs the scale_factor, this is where we get it though
            },
            RedrawRequested => {
                let delta_time = self.last_time.elapsed().as_secs_f32();
                self.last_time = std::time::Instant::now();
                renderer.update(delta_time);
                renderer.render();
            },
            _ => {},
        }

        _ = consumed;
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

