use glam::UVec2;
use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame};
use raylib::prelude::*;
use raylib::{ffi::SetTraceLogLevel, prelude::TraceLogLevel};
use std::fs;
use std::path::{Path, PathBuf};

mod bvh;
mod cube;
mod model;
mod sketch;

const TIMESTEP: f32 = 1.0 / sketch::FRAMES_PER_SECOND as f32;
const WINDOW_TITLE: &str = "software-renderer-rs";

struct CaptureConfig {
    frame_dir: PathBuf,
    gif_path: PathBuf,
    still_path: PathBuf,
    total_frames: u32,
    next_frame: u32,
}

impl CaptureConfig {
    fn from_cli() -> Option<Self> {
        let wants_capture = std::env::args().skip(1).any(|arg| arg == "--capture-demo");
        if !wants_capture {
            return None;
        }

        let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        Some(Self {
            frame_dir: project_root.join("target/capture_frames"),
            gif_path: project_root.join("assets/preview.gif"),
            still_path: project_root.join("assets/screenshot.png"),
            total_frames: sketch::FRAMES_PER_SECOND,
            next_frame: 0,
        })
    }

    fn setup(&self) -> Result<(), String> {
        if self.frame_dir.exists() {
            fs::remove_dir_all(&self.frame_dir).map_err(|e| {
                format!(
                    "failed removing old frame dir {}: {e}",
                    self.frame_dir.display()
                )
            })?;
        }

        fs::create_dir_all(&self.frame_dir).map_err(|e| {
            format!(
                "failed creating frame dir {}: {e}",
                self.frame_dir.display()
            )
        })?;

        if let Some(parent) = self.gif_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed creating gif dir {}: {e}", parent.display()))?;
        }
        if let Some(parent) = self.still_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed creating still dir {}: {e}", parent.display()))?;
        }

        Ok(())
    }

    fn record_frame(&mut self, rl: &mut RaylibHandle, rlt: &RaylibThread) {
        let filename = self
            .frame_dir
            .join(format!("frame_{:04}.png", self.next_frame));
        rl.take_screenshot(rlt, filename.to_string_lossy().as_ref());
        self.next_frame += 1;
    }

    fn done(&self) -> bool {
        self.next_frame >= self.total_frames
    }

    fn finalize(&self) -> Result<(), String> {
        let mut frame_paths = Vec::with_capacity(self.total_frames as usize);
        for i in 0..self.total_frames {
            let path = self.frame_dir.join(format!("frame_{:04}.png", i));
            if !path.exists() {
                return Err(format!("missing captured frame {}", path.display()));
            }
            frame_paths.push(path);
        }

        let still_idx = frame_paths.len() / 2;
        fs::copy(&frame_paths[still_idx], &self.still_path).map_err(|e| {
            format!(
                "failed copying still {} -> {}: {e}",
                frame_paths[still_idx].display(),
                self.still_path.display()
            )
        })?;

        let gif_file = fs::File::create(&self.gif_path)
            .map_err(|e| format!("failed creating gif {}: {e}", self.gif_path.display()))?;
        let mut encoder = GifEncoder::new(gif_file);
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| format!("failed setting gif repeat: {e}"))?;

        for frame_path in &frame_paths {
            let rgba = image::open(frame_path)
                .map_err(|e| format!("failed reading frame {}: {e}", frame_path.display()))?
                .to_rgba8();
            let frame = Frame::from_parts(
                rgba,
                0,
                0,
                Delay::from_numer_denom_ms(1000, sketch::FRAMES_PER_SECOND),
            );
            encoder
                .encode_frame(frame)
                .map_err(|e| format!("failed encoding gif frame {}: {e}", frame_path.display()))?;
        }

        if self.frame_dir.exists() {
            fs::remove_dir_all(&self.frame_dir).map_err(|e| {
                format!(
                    "failed cleaning frame dir {}: {e}",
                    self.frame_dir.display()
                )
            })?;
        }

        Ok(())
    }
}

fn main() {
    let mut capture = CaptureConfig::from_cli();
    if let Some(cfg) = capture.as_ref() {
        if let Err(err) = cfg.setup() {
            eprintln!("Capture setup error: {err}");
            std::process::exit(1);
        }
    }

    let (mut rl, mut rlt) = raylib::init().title(WINDOW_TITLE).build();
    unsafe {
        SetTraceLogLevel(TraceLogLevel::LOG_WARNING as i32);
    }

    let capture_mode = capture.is_some();
    let dims = UVec2::new(480, 320);
    let window_dims = if capture_mode {
        dims
    } else {
        UVec2::new(1280, 720)
    };
    // let window_dims = UVec2::new(512, 512);
    // let dims = window_dims;
    // let dims = UVec2::new(128, 128);
    // let dims = UVec2::new(120, 80);
    // let dims = UVec2::new(240, 160);
    let fullscreen = false;
    let mut state = sketch::State::new(dims);
    rl.set_window_size(window_dims.x as i32, window_dims.y as i32);
    if fullscreen {
        rl.toggle_fullscreen();
        rl.set_window_size(rl.get_screen_width(), rl.get_screen_height());
    }

    center_window(&mut rl, window_dims);
    if capture_mode {
        rl.set_target_fps(sketch::FRAMES_PER_SECOND as u32);
    }
    let mouse_scale = dims.as_vec2() / window_dims.as_vec2();
    rl.set_mouse_scale(mouse_scale.x as f32, mouse_scale.y as f32);

    let mut render_texture = rl
        .load_render_texture(&rlt, dims.x, dims.y)
        .unwrap_or_else(|e| {
            println!("Error creating render texture: {}", e);
            std::process::exit(1);
        });

    while state.running && !rl.window_should_close() {
        sketch::process_events_and_input(&mut rl, &mut state);

        if capture_mode {
            sketch::step(&mut rl, &mut rlt, &mut state);
        } else {
            let dt = rl.get_frame_time();
            state.time_since_last_update += dt;
            while state.time_since_last_update > TIMESTEP {
                state.time_since_last_update -= TIMESTEP;
                sketch::step(&mut rl, &mut rlt, &mut state);
            }
        }

        {
            let mut draw_handle = rl.begin_drawing(&rlt);
            {
                let low_res_draw_handle =
                    &mut draw_handle.begin_texture_mode(&rlt, &mut render_texture);
                low_res_draw_handle.clear_background(Color::BLACK);
                sketch::draw(&mut state, low_res_draw_handle);
            }
            scale_and_blit_render_texture_to_window(
                &mut draw_handle,
                &mut render_texture,
                fullscreen,
                window_dims,
            );
        }

        if let Some(cfg) = capture.as_mut() {
            cfg.record_frame(&mut rl, &rlt);
            if cfg.done() {
                state.running = false;
            }
        }
    }

    if let Some(cfg) = capture {
        if let Err(err) = cfg.finalize() {
            eprintln!("Capture finalize error: {err}");
            std::process::exit(1);
        }
        println!("Saved still screenshot to {}", cfg.still_path.display());
        println!("Saved preview GIF to {}", cfg.gif_path.display());
    }
}

pub fn center_window(rl: &mut raylib::RaylibHandle, window_dims: UVec2) {
    // Center on the current monitor (multi-monitor friendly, including negative coordinates).
    let monitor = raylib::core::window::get_current_monitor();
    let monitor_pos = unsafe { raylib::ffi::GetMonitorPosition(monitor) };
    let monitor_w = raylib::core::window::get_monitor_width(monitor).max(1);
    let monitor_h = raylib::core::window::get_monitor_height(monitor).max(1);

    let ww = window_dims.x as i32;
    let wh = window_dims.y as i32;

    let x = monitor_pos.x as i32 + ((monitor_w - ww).max(0) / 2);
    let y = monitor_pos.y as i32 + ((monitor_h - wh).max(0) / 2);
    rl.set_window_position(x, y);
    rl.set_target_fps(144);
}

pub fn scale_and_blit_render_texture_to_window(
    draw_handle: &mut RaylibDrawHandle,
    render_texture: &mut RenderTexture2D,
    fullscreen: bool,
    window_dims: UVec2,
) {
    let source_rec = Rectangle::new(
        0.0,
        0.0,
        render_texture.texture.width as f32,
        -render_texture.texture.height as f32,
    );
    // dest rec should be the fullscreen resolution if graphics.fullscreen, otherwise window_dims
    let dest_rec = if fullscreen {
        // get the fullscreen resolution
        let screen_width = draw_handle.get_screen_width();
        let screen_height = draw_handle.get_screen_height();
        Rectangle::new(0.0, 0.0, screen_width as f32, screen_height as f32)
    } else {
        Rectangle::new(0.0, 0.0, window_dims.x as f32, window_dims.y as f32)
    };

    let origin = Vector2::new(0.0, 0.0);

    draw_handle.draw_texture_pro(
        render_texture,
        source_rec,
        dest_rec,
        origin,
        0.0,
        Color::WHITE,
    );
}
