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

#[derive(Default)]
pub struct App {
    pub renderer: Option<Renderer>,
    pub scene: Option<Scene>,
}


impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_inner_size(LogicalSize::new(800, 600))
            .with_title("GPU Ray Tracer");
        let window = event_loop.create_window(window_attributes).unwrap();
        let window_id = Some(window.id());
        let window = Arc::new(window);

        let renderer = pollster::block_on(Renderer::new(event_loop.owned_display_handle(), window.clone(), self.scene.unwrap()));
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

        match event {
            CloseRequested => event_loop.exit(),
            Resized(_physical_size) => {
                // TODO
            },
            ScaleFactorChanged { scale_factor: _, .. } => {
                // TODO we don't get the inner size anymore, winit sends another Resized event afterwards
                // If our renderer ever needs the scale_factor, this is where we get it though
            },
            RedrawRequested => {
                // TODO implement once renderer works
            },
            _ => {},
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        self.window.request_redraw();
    }
}

