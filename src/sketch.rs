use glam::{Mat4, Vec2, Vec3, Vec4};
use image::{DynamicImage, GenericImageView};
use raylib::prelude::*;

use crate::model::{load_cube, load_from_obj, TriIndices};

pub const FRAMES_PER_SECOND: u32 = 60;

const DOUBLE_SIDED: bool = false;
const FRONT_FACE_CCW: bool = true;

pub struct Cam {
    pub pos: Vec3,
    pub dir: Vec3,
}

pub struct State {
    pub running: bool,
    pub time_since_last_update: f32,
    pub sim_time: f32,

    pub cam: Cam,
    pub texture: image::DynamicImage,

    pub render_dims: glam::UVec2,
    pub z_buffer: Vec<f32>,

    pub cube: crate::model::Model,
    pub hill: crate::model::Model,
    pub water: crate::model::Model,
    pub water_base_verts: crate::model::Vertices,
    pub palm: crate::model::Model,
    pub tree: crate::model::Model,
}

impl State {
    pub fn new(render_dims: glam::UVec2) -> Self {
        let texture = image::open("./box.png").unwrap_or_else(|e| {
            println!("Error opening image: {}", e);
            std::process::exit(1);
        });

        let z_len = (render_dims.x as usize) * (render_dims.y as usize);
        let water = crate::model::gen_water_plane(24, 24, 30.0, 30.0, -0.9);
        let water_base_verts = water.verts.clone();
        Self {
            running: true,
            time_since_last_update: 0.0,
            sim_time: 0.0,

            cam: Cam {
                pos: Vec3::new(0.0, 0.0, 2.0),
                dir: Vec3::new(0.0, 0.0, -1.0),
            },
            texture,
            render_dims,
            z_buffer: vec![f32::MAX; z_len],
            cube: load_cube(),
            hill: crate::model::gen_sandy_hill(10, 16, 6.0, 0.5),
            water,
            water_base_verts,
            palm: crate::model::gen_palm_tree(),
            tree: load_from_obj("./treepot.obj"),
        }
    }
}

pub fn calc_normals(verts: &[Vec3], tri_indices: &TriIndices) -> Vec<Vec3> {
    let mut normals = Vec::new();
    for tri in tri_indices {
        // Get the vertices of the triangle
        let v0 = verts[tri[0]];
        let v1 = verts[tri[1]];
        let v2 = verts[tri[2]];

        // Calculate two edges of the triangle
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let normal = edge1.cross(edge2);
        let normal = normal.normalize();

        normals.push(normal);
    }
    normals
}

pub fn build_mvp(pos: Vec3, rot: f32, scale: Vec3, cam: &Cam, aspect_ratio: f32) -> Mat4 {
    let mut model = Mat4::IDENTITY;
    model = model * Mat4::from_translation(pos);
    model = model * Mat4::from_scale(scale);
    model = model * Mat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), rot);

    let view = Mat4::look_at_rh(cam.pos, cam.pos + cam.dir * 10.0, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective_rh(std::f32::consts::PI / 2.0, aspect_ratio, 0.001, 100.0);
    let mvp = projection * view * model;

    mvp
}

pub fn transform(vert: Vec3, mvp: Mat4) -> Vec3 {
    let mut transformed_vert = mvp * vert.extend(1.0);
    transformed_vert /= transformed_vert.w; // Perspective division
    Vec3::new(transformed_vert.x, transformed_vert.y, transformed_vert.z)
}

pub fn transform_some(verts: &[Vec3], mvp: Mat4) -> Vec<Vec3> {
    let mut transformed_verts = Vec::new();
    for vert in verts {
        let mut transformed_vert = mvp * vert.extend(1.0);
        transformed_vert /= transformed_vert.w; // Perspective division
        transformed_verts.push(Vec3::new(
            transformed_vert.x,
            transformed_vert.y,
            transformed_vert.z,
        ));
    }

    transformed_verts
}

/// converts a point in NDC to screen coordinates
/// NDC is in the range [-1, 1],
/// screen coordinates are in the range [0, render_resolution]
/// and represent the pixel coordinates of the screen, where y is the bottom of the screen
pub fn ndc_to_sc(v: &Vec3, render_resolution: Vec2) -> Vec3 {
    // Match the C++ renderer mapping:
    // sx = (ndc.x*0.5 + 0.5) * (w-1)
    // sy = (1 - (ndc.y*0.5 + 0.5)) * (h-1)   (y down)
    let w = (render_resolution.x - 1.0).max(1.0);
    let h = (render_resolution.y - 1.0).max(1.0);
    Vec3::new(
        (v.x * 0.5 + 0.5) * w,
        (1.0 - (v.y * 0.5 + 0.5)) * h,
        v.z,
    )
}

pub fn sample_texture(texture: &DynamicImage, uv: Vec2) -> Color {
    // Match the C++ reference `sample_repeat(u,v)`:
    // wrap into [0,1), then x=int(u*(w-1)), y=int(v*(h-1)).
    let u = uv.x - uv.x.floor();
    let v = uv.y - uv.y.floor();

    let w = texture.width().max(1) as i32;
    let h = texture.height().max(1) as i32;

    let tex_x = (u * (w - 1).max(1) as f32) as i32;
    let tex_y = (v * (h - 1).max(1) as f32) as i32;

    // clamp tex_y and tex_x
    let tex_y = tex_y.min(h - 1).max(0);
    let tex_x = tex_x.min(w - 1).max(0);

    let c = texture.get_pixel(tex_x as u32, tex_y as u32);
    Color::new(c[0], c[1], c[2], c[3])
}

#[derive(Clone, Copy, Debug)]
pub struct ScreenVert {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub inv_w: f32,
    pub u_over_w: f32,
    pub v_over_w: f32,
}

#[inline]
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

#[derive(Clone, Copy, Debug)]
pub struct ClipVert {
    pub clip: Vec4,
    pub uv: Vec2,
}

fn clip_poly_against_plane<F>(poly: &[ClipVert], dist_fn: F) -> Vec<ClipVert>
where
    F: Fn(Vec4) -> f32,
{
    if poly.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(poly.len() + 2);

    let mut prev = *poly.last().unwrap();
    let mut prev_d = dist_fn(prev.clip);
    let mut prev_in = prev_d >= 0.0;

    for &cur in poly {
        let cur_d = dist_fn(cur.clip);
        let cur_in = cur_d >= 0.0;

        if prev_in && cur_in {
            out.push(cur);
        } else if prev_in && !cur_in {
            let denom = prev_d - cur_d;
            if denom != 0.0 {
                let t = prev_d / denom;
                let clip = prev.clip + (cur.clip - prev.clip) * t;
                let uv = prev.uv + (cur.uv - prev.uv) * t;
                out.push(ClipVert { clip, uv });
            }
        } else if !prev_in && cur_in {
            let denom = prev_d - cur_d;
            if denom != 0.0 {
                let t = prev_d / denom;
                let clip = prev.clip + (cur.clip - prev.clip) * t;
                let uv = prev.uv + (cur.uv - prev.uv) * t;
                out.push(ClipVert { clip, uv });
            }
            out.push(cur);
        }

        prev = cur;
        prev_d = cur_d;
        prev_in = cur_in;
    }

    out
}

fn clip_triangle(v0: ClipVert, v1: ClipVert, v2: ClipVert) -> Vec<ClipVert> {
    // Clip volume (OpenGL): -w<=x<=w, -w<=y<=w, -w<=z<=w
    let left = |c: Vec4| c.x + c.w;
    let right = |c: Vec4| -c.x + c.w;
    let bottom = |c: Vec4| c.y + c.w;
    let top = |c: Vec4| -c.y + c.w;
    let nearp = |c: Vec4| c.z + c.w;
    let farp = |c: Vec4| -c.z + c.w;

    let mut poly = vec![v0, v1, v2];
    poly = clip_poly_against_plane(&poly, left);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly = clip_poly_against_plane(&poly, right);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly = clip_poly_against_plane(&poly, bottom);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly = clip_poly_against_plane(&poly, top);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly = clip_poly_against_plane(&poly, nearp);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly = clip_poly_against_plane(&poly, farp);
    if poly.len() < 3 {
        return Vec::new();
    }
    poly
}

pub fn draw_texture_tri(
    d: &mut impl RaylibDraw,
    texture: &DynamicImage,
    a: ScreenVert,
    b: ScreenVert,
    c: ScreenVert,
    z_buffer: &mut [f32],
    z_buffer_w: usize,
    viewport_w: i32,
    viewport_h: i32,
) {
    // Bounding box.
    let minx_f = a.x.min(b.x).min(c.x);
    let maxx_f = a.x.max(b.x).max(c.x);
    let miny_f = a.y.min(b.y).min(c.y);
    let maxy_f = a.y.max(b.y).max(c.y);

    let minx = (minx_f.floor() as i32).max(0);
    let maxx = (maxx_f.ceil() as i32).min(viewport_w - 1);
    let miny = (miny_f.floor() as i32).max(0);
    let maxy = (maxy_f.ceil() as i32).min(viewport_h - 1);
    if minx > maxx || miny > maxy {
        return;
    }

    let area = edge_fn(a.x, a.y, b.x, b.y, c.x, c.y);
    if area == 0.0 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in miny..=maxy {
        let py = y as f32 + 0.5;
        for x in minx..=maxx {
            let px = x as f32 + 0.5;
            let w0 = edge_fn(b.x, b.y, c.x, c.y, px, py);
            let w1 = edge_fn(c.x, c.y, a.x, a.y, px, py);
            let w2 = edge_fn(a.x, a.y, b.x, b.y, px, py);

            // Inside test supports both windings (area sign).
            if area > 0.0 {
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
            } else if w0 > 0.0 || w1 > 0.0 || w2 > 0.0 {
                continue;
            }

            let alpha = w0 * inv_area;
            let beta = w1 * inv_area;
            let gamma = w2 * inv_area;

            let z = alpha * a.z + beta * b.z + gamma * c.z;
            let idx = (y as usize) * z_buffer_w + (x as usize);
            // NDC z: near=-1 is closer than far=+1.
            if z >= z_buffer[idx] {
                continue;
            }

            let invw = alpha * a.inv_w + beta * b.inv_w + gamma * c.inv_w;
            if invw == 0.0 {
                continue;
            }
            let uow = alpha * a.u_over_w + beta * b.u_over_w + gamma * c.u_over_w;
            let vow = alpha * a.v_over_w + beta * b.v_over_w + gamma * c.v_over_w;
            let u = uow / invw;
            let v = vow / invw;

            let color = sample_texture(texture, Vec2::new(u, v));
            if color.a == 0 {
                continue;
            }

            // Match C++ behavior: depth-test as usual, then alpha blend, and still write depth.
            z_buffer[idx] = z;
            d.draw_pixel(x, y, color);
        }
    }
}

pub fn draw_solid_tri(
    d: &mut impl RaylibDraw,
    color: Color,
    a: ScreenVert,
    b: ScreenVert,
    c: ScreenVert,
    z_buffer: &mut [f32],
    z_buffer_w: usize,
    viewport_w: i32,
    viewport_h: i32,
) {
    // Bounding box.
    let minx_f = a.x.min(b.x).min(c.x);
    let maxx_f = a.x.max(b.x).max(c.x);
    let miny_f = a.y.min(b.y).min(c.y);
    let maxy_f = a.y.max(b.y).max(c.y);

    let minx = (minx_f.floor() as i32).max(0);
    let maxx = (maxx_f.ceil() as i32).min(viewport_w - 1);
    let miny = (miny_f.floor() as i32).max(0);
    let maxy = (maxy_f.ceil() as i32).min(viewport_h - 1);
    if minx > maxx || miny > maxy {
        return;
    }

    let area = edge_fn(a.x, a.y, b.x, b.y, c.x, c.y);
    if area == 0.0 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in miny..=maxy {
        let py = y as f32 + 0.5;
        for x in minx..=maxx {
            let px = x as f32 + 0.5;
            let w0 = edge_fn(b.x, b.y, c.x, c.y, px, py);
            let w1 = edge_fn(c.x, c.y, a.x, a.y, px, py);
            let w2 = edge_fn(a.x, a.y, b.x, b.y, px, py);

            // Inside test supports both windings (area sign).
            if area > 0.0 {
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
            } else if w0 > 0.0 || w1 > 0.0 || w2 > 0.0 {
                continue;
            }

            let alpha = w0 * inv_area;
            let beta = w1 * inv_area;
            let gamma = w2 * inv_area;

            let z = alpha * a.z + beta * b.z + gamma * c.z;
            let idx = (y as usize) * z_buffer_w + (x as usize);
            if z >= z_buffer[idx] {
                continue;
            }

            if color.a == 255 {
                z_buffer[idx] = z;
            }
            d.draw_pixel(x, y, color);
        }
    }
}

pub fn draw_tri(d: &mut impl RaylibDraw, verts: &[Vec3]) {
    // draw lines connecting the vertices
    for i in 0..verts.len() {
        let v0 = verts[i];
        let v1 = verts[(i + 1) % verts.len()];
        d.draw_line(
            v0.x as i32,
            v0.y as i32,
            v1.x as i32,
            v1.y as i32,
            Color::RED,
        );
    }
}

pub fn process_events_and_input(rl: &mut RaylibHandle, state: &mut State) {
    if rl.is_key_pressed(raylib::consts::KeyboardKey::KEY_ESCAPE) {
        state.running = false;
    }

    // move cam with wasd
    let speed = 0.025;
    let up = Vec3::new(0.0, 1.0, 0.0);

    let right = state.cam.dir.cross(up);
    let left = -right;

    let down = -up;
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_SPACE) {
        state.cam.pos += up * speed;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_LEFT_SHIFT) {
        state.cam.pos += down * speed;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_A) {
        state.cam.pos += left * speed;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_D) {
        state.cam.pos += right * speed;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_W) {
        state.cam.pos += state.cam.dir * speed;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_S) {
        state.cam.pos -= state.cam.dir * speed;
    }
    // q and e for rotating camera
    let rot_speed = 0.01;
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_Q) {
        let rotmat = Mat4::from_axis_angle(up, rot_speed);
        let dir4 = Vec4::new(state.cam.dir.x, state.cam.dir.y, state.cam.dir.z, 1.0);
        let new_dir = rotmat * dir4;
        let dir3 = Vec3::new(new_dir.x, new_dir.y, new_dir.z);
        state.cam.dir = dir3;
    }
    if rl.is_key_down(raylib::consts::KeyboardKey::KEY_E) {
        let rotmat = Mat4::from_axis_angle(up, -rot_speed);
        let dir4 = state.cam.dir.extend(1.0);
        let new_dir = rotmat * dir4;
        let dir3 = Vec3::new(new_dir.x, new_dir.y, new_dir.z);
        state.cam.dir = dir3;
    }
}

pub fn step(rl: &mut RaylibHandle, rlt: &mut RaylibThread, state: &mut State) {
    // camera should always look at origin, and rotate around origin
    // let t = (rl.get_time() * 1.0) as f32;
    // const D: f32 = 5.0;
    // let pos = Vec3::new(t.sin() * D, 2.0, t.cos() * D);
    // let dir = (Vec3::ZERO - state.cam.pos).normalize();
    // state.cam = Cam { pos, dir };

    // Fixed-step simulation time used for procedural animation (e.g. water waves).
    state.sim_time += 1.0 / FRAMES_PER_SECOND as f32;
}

pub fn draw(state: &mut State, d: &mut RaylibTextureMode<RaylibDrawHandle>) {
    state.z_buffer.fill(f32::INFINITY);

    let cube = &state.cube;
    let hill = &state.hill;
    let water = &mut state.water;
    let palm = &state.palm;
    let tree = &state.tree;

    // needed for transforming verts
    let render_resolution = state.render_dims.as_vec2();
    let viewport_w = state.render_dims.x as i32;
    let viewport_h = state.render_dims.y as i32;
    let aspect_ratio = render_resolution.x / render_resolution.y;

    let size = 1.0;
    let scale = Vec3::new(size, size, size);

    // Explicit alpha blending (RGBA) for textures.
    let mut d = d.begin_blend_mode(BlendMode::BLEND_ALPHA);

    // Animate water vertices (simple traveling wave).
    {
        let t = state.sim_time;
        let amp = 0.45;
        let kx = 0.28;
        let kz = 0.22;
        let speed = 1.0;
        for (v, base) in water.verts.iter_mut().zip(state.water_base_verts.iter()) {
            let x = base.x;
            let z = base.z;
            let wave = (x * kx + t * speed).sin() * (z * kz + t * (speed * 0.8)).cos();
            v.y = base.y + amp * wave;
        }
    }

    // Draw hill first (opaque).
    {
        let hill_pos = Vec3::new(0.0, -1.6, -10.0);
        let hill_scale = Vec3::splat(1.0);
        let mvp = build_mvp(hill_pos, 0.0, hill_scale, &state.cam, aspect_ratio);

        let mut clip_verts = Vec::with_capacity(hill.verts.len());
        for v in &hill.verts {
            clip_verts.push(mvp * v.extend(1.0));
        }

        for i in 0..hill.tri_indices.len() {
            let i0 = hill.tri_indices[i][0] as usize;
            let i1 = hill.tri_indices[i][1] as usize;
            let i2 = hill.tri_indices[i][2] as usize;

            let v0 = ClipVert {
                clip: clip_verts[i0],
                uv: Vec2::ZERO,
            };
            let v1 = ClipVert {
                clip: clip_verts[i1],
                uv: Vec2::ZERO,
            };
            let v2 = ClipVert {
                clip: clip_verts[i2],
                uv: Vec2::ZERO,
            };

            let poly = clip_triangle(v0, v1, v2);
            if poly.len() < 3 {
                continue;
            }

            let [r, g, b, a] = hill.tri_colors[i];
            let face_color = Color::new(r, g, b, a);

            let base = poly[0];
            for k in 1..(poly.len() - 1) {
                let a = base;
                let b = poly[k];
                let c = poly[k + 1];

                if a.clip.w == 0.0 || b.clip.w == 0.0 || c.clip.w == 0.0 {
                    continue;
                }

                let invwa = 1.0 / a.clip.w;
                let invwb = 1.0 / b.clip.w;
                let invwc = 1.0 / c.clip.w;
                let ndc_a = Vec3::new(a.clip.x * invwa, a.clip.y * invwa, a.clip.z * invwa);
                let ndc_b = Vec3::new(b.clip.x * invwb, b.clip.y * invwb, b.clip.z * invwb);
                let ndc_c = Vec3::new(c.clip.x * invwc, c.clip.y * invwc, c.clip.z * invwc);

                let area_ndc = (ndc_b.x - ndc_a.x) * (ndc_c.y - ndc_a.y)
                    - (ndc_b.y - ndc_a.y) * (ndc_c.x - ndc_a.x);
                if !DOUBLE_SIDED {
                    if FRONT_FACE_CCW {
                        if area_ndc <= 0.0 {
                            continue;
                        }
                    } else if area_ndc >= 0.0 {
                        continue;
                    }
                }

                let sc_a = ndc_to_sc(&ndc_a, render_resolution);
                let sc_b = ndc_to_sc(&ndc_b, render_resolution);
                let sc_c = ndc_to_sc(&ndc_c, render_resolution);

                let sa = ScreenVert {
                    x: sc_a.x,
                    y: sc_a.y,
                    z: ndc_a.z,
                    inv_w: invwa,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sb = ScreenVert {
                    x: sc_b.x,
                    y: sc_b.y,
                    z: ndc_b.z,
                    inv_w: invwb,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sc = ScreenVert {
                    x: sc_c.x,
                    y: sc_c.y,
                    z: ndc_c.z,
                    inv_w: invwc,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };

                draw_solid_tri(
                    &mut d,
                    face_color,
                    sa,
                    sb,
                    sc,
                    &mut state.z_buffer,
                    state.render_dims.x as usize,
                    viewport_w,
                    viewport_h,
                );
            }
        }
    }

    // Draw water (semi-transparent), after hill so it alpha-blends over sand where visible.
    {
        let water_pos = Vec3::new(0.0, 0.0, -12.0);
        let water_scale = Vec3::splat(1.0);
        let mvp = build_mvp(water_pos, 0.0, water_scale, &state.cam, aspect_ratio);

        let mut clip_verts = Vec::with_capacity(water.verts.len());
        for v in &water.verts {
            clip_verts.push(mvp * v.extend(1.0));
        }

        for i in 0..water.tri_indices.len() {
            let i0 = water.tri_indices[i][0] as usize;
            let i1 = water.tri_indices[i][1] as usize;
            let i2 = water.tri_indices[i][2] as usize;

            let v0 = ClipVert {
                clip: clip_verts[i0],
                uv: Vec2::ZERO,
            };
            let v1 = ClipVert {
                clip: clip_verts[i1],
                uv: Vec2::ZERO,
            };
            let v2 = ClipVert {
                clip: clip_verts[i2],
                uv: Vec2::ZERO,
            };

            let poly = clip_triangle(v0, v1, v2);
            if poly.len() < 3 {
                continue;
            }

            let [r, g, b, a] = water.tri_colors[i];
            let face_color = Color::new(r, g, b, a);

            let base = poly[0];
            for k in 1..(poly.len() - 1) {
                let a = base;
                let b = poly[k];
                let c = poly[k + 1];

                if a.clip.w == 0.0 || b.clip.w == 0.0 || c.clip.w == 0.0 {
                    continue;
                }

                let invwa = 1.0 / a.clip.w;
                let invwb = 1.0 / b.clip.w;
                let invwc = 1.0 / c.clip.w;
                let ndc_a = Vec3::new(a.clip.x * invwa, a.clip.y * invwa, a.clip.z * invwa);
                let ndc_b = Vec3::new(b.clip.x * invwb, b.clip.y * invwb, b.clip.z * invwb);
                let ndc_c = Vec3::new(c.clip.x * invwc, c.clip.y * invwc, c.clip.z * invwc);

                let area_ndc = (ndc_b.x - ndc_a.x) * (ndc_c.y - ndc_a.y)
                    - (ndc_b.y - ndc_a.y) * (ndc_c.x - ndc_a.x);
                if !DOUBLE_SIDED {
                    if FRONT_FACE_CCW {
                        if area_ndc <= 0.0 {
                            continue;
                        }
                    } else if area_ndc >= 0.0 {
                        continue;
                    }
                }

                let sc_a = ndc_to_sc(&ndc_a, render_resolution);
                let sc_b = ndc_to_sc(&ndc_b, render_resolution);
                let sc_c = ndc_to_sc(&ndc_c, render_resolution);

                let sa = ScreenVert {
                    x: sc_a.x,
                    y: sc_a.y,
                    z: ndc_a.z,
                    inv_w: invwa,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sb = ScreenVert {
                    x: sc_b.x,
                    y: sc_b.y,
                    z: ndc_b.z,
                    inv_w: invwb,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sc = ScreenVert {
                    x: sc_c.x,
                    y: sc_c.y,
                    z: ndc_c.z,
                    inv_w: invwc,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };

                draw_solid_tri(
                    &mut d,
                    face_color,
                    sa,
                    sb,
                    sc,
                    &mut state.z_buffer,
                    state.render_dims.x as usize,
                    viewport_w,
                    viewport_h,
                );
            }
        }
    }

    // Draw palm (solid-color) sitting on the hill.
    {
        let palm_pos = Vec3::new(1.5, -1.6 + 1.35, -10.0); // roughly on hill surface
        let palm_scale = Vec3::splat(1.0);
        let mvp = build_mvp(palm_pos, 0.0, palm_scale, &state.cam, aspect_ratio);

        let mut clip_verts = Vec::with_capacity(palm.verts.len());
        for v in &palm.verts {
            clip_verts.push(mvp * v.extend(1.0));
        }

        // Palm is solid-color: per-tri colors must exist.
        for i in 0..palm.tri_indices.len() {
            let i0 = palm.tri_indices[i][0] as usize;
            let i1 = palm.tri_indices[i][1] as usize;
            let i2 = palm.tri_indices[i][2] as usize;

            let v0 = ClipVert {
                clip: clip_verts[i0],
                uv: Vec2::ZERO,
            };
            let v1 = ClipVert {
                clip: clip_verts[i1],
                uv: Vec2::ZERO,
            };
            let v2 = ClipVert {
                clip: clip_verts[i2],
                uv: Vec2::ZERO,
            };

            let poly = clip_triangle(v0, v1, v2);
            if poly.len() < 3 {
                continue;
            }

            let [r, g, b, a] = palm.tri_colors[i];
            let face_color = Color::new(r, g, b, a);

            let base = poly[0];
            for k in 1..(poly.len() - 1) {
                let a = base;
                let b = poly[k];
                let c = poly[k + 1];

                if a.clip.w == 0.0 || b.clip.w == 0.0 || c.clip.w == 0.0 {
                    continue;
                }

                let invwa = 1.0 / a.clip.w;
                let invwb = 1.0 / b.clip.w;
                let invwc = 1.0 / c.clip.w;
                let ndc_a = Vec3::new(a.clip.x * invwa, a.clip.y * invwa, a.clip.z * invwa);
                let ndc_b = Vec3::new(b.clip.x * invwb, b.clip.y * invwb, b.clip.z * invwb);
                let ndc_c = Vec3::new(c.clip.x * invwc, c.clip.y * invwc, c.clip.z * invwc);

                let area_ndc = (ndc_b.x - ndc_a.x) * (ndc_c.y - ndc_a.y)
                    - (ndc_b.y - ndc_a.y) * (ndc_c.x - ndc_a.x);
                if !DOUBLE_SIDED {
                    if FRONT_FACE_CCW {
                        if area_ndc <= 0.0 {
                            continue;
                        }
                    } else if area_ndc >= 0.0 {
                        continue;
                    }
                }

                let sc_a = ndc_to_sc(&ndc_a, render_resolution);
                let sc_b = ndc_to_sc(&ndc_b, render_resolution);
                let sc_c = ndc_to_sc(&ndc_c, render_resolution);

                let sa = ScreenVert {
                    x: sc_a.x,
                    y: sc_a.y,
                    z: ndc_a.z,
                    inv_w: invwa,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sb = ScreenVert {
                    x: sc_b.x,
                    y: sc_b.y,
                    z: ndc_b.z,
                    inv_w: invwb,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };
                let sc = ScreenVert {
                    x: sc_c.x,
                    y: sc_c.y,
                    z: ndc_c.z,
                    inv_w: invwc,
                    u_over_w: 0.0,
                    v_over_w: 0.0,
                };

                draw_solid_tri(
                    &mut d,
                    face_color,
                    sa,
                    sb,
                    sc,
                    &mut state.z_buffer,
                    state.render_dims.x as usize,
                    viewport_w,
                    viewport_h,
                );
            }
        }
    }

    let frame_time = d.get_frame_time();
    let frame_time_ms = frame_time * 1000.0;
    let mut cursor_y = 0.0;
    const FONT_SIZE: i32 = 12;
    d.draw_text(
        &format!("Frame Time: {:.2}ms", frame_time_ms),
        0,
        cursor_y as i32,
        FONT_SIZE,
        Color::WHITE,
    );
    cursor_y += FONT_SIZE as f32;

    // print the cam pos
    d.draw_text(
        &format!("Cam Pos: {:?}", state.cam.pos),
        0,
        cursor_y as i32,
        FONT_SIZE,
        Color::WHITE,
    );
    cursor_y += FONT_SIZE as f32;
    // print the cam dir
    d.draw_text(
        &format!("Cam Dir: {:?}", state.cam.dir),
        0,
        cursor_y as i32,
        FONT_SIZE,
        Color::WHITE,
    );
    let mouse_pos = d.get_mouse_position();
    d.draw_circle(mouse_pos.x as i32, mouse_pos.y as i32, 6.0, Color::GREEN);
}
