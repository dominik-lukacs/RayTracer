use winit::{application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent::{CloseRequested, RedrawRequested, Resized, ScaleFactorChanged}, event_loop::{ControlFlow, EventLoop}, window::{Window, WindowAttributes, WindowId}};

struct App {
    window: Option<Window>,
    window_id: Option<WindowId>,

    start_time: std::time::Instant,
    last_time: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        let time = std::time::Instant::now();
        Self { window: Default::default(), window_id: Default::default(), start_time: time, last_time: time.clone() }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_inner_size(LogicalSize::new(800, 600))
            .with_title("GPU Ray Tracer");
        let window = event_loop.create_window(window_attributes).unwrap();
        self.window_id = Some(window.id());
        self.window = Some(window);
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
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}

