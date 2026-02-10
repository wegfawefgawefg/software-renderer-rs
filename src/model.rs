use glam::{Vec2, Vec3};

use crate::cube;

pub type Vertices = Vec<Vec3>;
pub type Normals = Vec<Vec3>;
pub type TriVertexNormals = Vec<[Vec3; 3]>;
pub type Indices = Vec<usize>;
pub type TriIndices = Vec<[usize; 3]>;
pub type TexCoords = Vec<Vec2>;
pub type TriTexCoords = Vec<[Vec2; 3]>;
pub type TriColors = Vec<[u8; 4]>;

pub struct Model {
    pub verts: Vertices,
    pub tri_indices: TriIndices,
    pub tri_tex_coords: TriTexCoords,
    // Optional per-triangle RGBA color (used for solid-color meshes).
    // If empty, the renderer should fall back to texturing.
    pub tri_colors: TriColors,
}

pub fn load_cube() -> Model {
    Model {
        verts: cube::get_cube_verts(),
        tri_indices: cube::gen_cube_tri_indices(),
        tri_tex_coords: cube::gen_cube_tex_coords(),
        tri_colors: Vec::new(),
    }
}

pub fn gen_sandy_hill(stacks: usize, slices: usize, radius: f32, squash_y: f32) -> Model {
    // Low-poly hemisphere (y >= 0), squashed vertically; per-face "sandy" color.
    // Parameterization:
    // theta: polar angle from +Y down toward equator, [0, pi/2]
    // phi: azimuth around Y axis, [0, 2*pi)
    let stacks = stacks.max(2);
    let slices = slices.max(3);
    let squash_y = squash_y.max(0.001);

    let mut verts: Vertices = Vec::new();
    let mut tri_indices: TriIndices = Vec::new();
    let mut tri_colors: TriColors = Vec::new();

    // Build vertex grid.
    // (stacks+1) rings including pole and equator; (slices+1) to wrap seam.
    let mut ring_index = vec![0usize; (stacks + 1) * (slices + 1)];
    let idx_at = |si: usize, pi: usize, slices: usize| si * (slices + 1) + pi;

    for si in 0..=stacks {
        let t = si as f32 / stacks as f32;
        let theta = (std::f32::consts::FRAC_PI_2) * t;
        let y = theta.cos() * radius * squash_y;
        let r = theta.sin() * radius;

        for pi in 0..=slices {
            let u = pi as f32 / slices as f32;
            let phi = std::f32::consts::TAU * u;
            let x = phi.cos() * r;
            let z = phi.sin() * r;

            let vi = verts.len();
            verts.push(Vec3::new(x, y, z));
            ring_index[idx_at(si, pi, slices)] = vi;
        }
    }

    // Triangulate quads between rings.
    for si in 0..stacks {
        for pi in 0..slices {
            let i00 = ring_index[idx_at(si, pi, slices)];
            let i01 = ring_index[idx_at(si, pi + 1, slices)];
            let i10 = ring_index[idx_at(si + 1, pi, slices)];
            let i11 = ring_index[idx_at(si + 1, pi + 1, slices)];

            // Winding: CCW when viewed from outside (top/side of the hemisphere).
            // If you can see it from underneath but not above, the winding is flipped.
            tri_indices.push([i00, i11, i10]);
            tri_indices.push([i00, i01, i11]);
        }
    }

    // Sandy base color with slight per-face variation (deterministic).
    let base = [226u8, 205u8, 140u8, 255u8];
    for ti in 0..tri_indices.len() {
        // Tiny brightness jitter [-10..+10] to make low-poly facets pop.
        let h = (ti as u32).wrapping_mul(2654435761);
        let jitter = ((h >> 28) as i32) - 8; // [-8..+7]
        let add = (jitter * 2) as i32;
        let mut c = base;
        for ch in 0..3 {
            let v = c[ch] as i32 + add;
            c[ch] = v.clamp(0, 255) as u8;
        }
        tri_colors.push(c);
    }

    Model {
        verts,
        tri_indices,
        tri_tex_coords: Vec::new(),
        tri_colors,
    }
}

pub fn gen_water_plane(nx: usize, nz: usize, width: f32, depth: f32, y: f32) -> Model {
    // Grid in XZ plane at height y. Intended for deformation (waves) by mutating vertex Y.
    let nx = nx.max(1);
    let nz = nz.max(1);

    let mut verts: Vertices = Vec::with_capacity((nx + 1) * (nz + 1));
    for iz in 0..=nz {
        let tz = iz as f32 / nz as f32;
        let z = (tz - 0.5) * depth;
        for ix in 0..=nx {
            let tx = ix as f32 / nx as f32;
            let x = (tx - 0.5) * width;
            verts.push(Vec3::new(x, y, z));
        }
    }

    let idx = |ix: usize, iz: usize| iz * (nx + 1) + ix;
    let mut tri_indices: TriIndices = Vec::with_capacity(nx * nz * 2);
    for iz in 0..nz {
        for ix in 0..nx {
            let i00 = idx(ix, iz);
            let i10 = idx(ix + 1, iz);
            let i01 = idx(ix, iz + 1);
            let i11 = idx(ix + 1, iz + 1);

            // CCW when viewed from +Y (normal points up).
            tri_indices.push([i00, i11, i10]);
            tri_indices.push([i00, i01, i11]);
        }
    }

    // Soft blue with slight per-face jitter for "faceted" low-poly water.
    let base = [40u8, 140u8, 190u8, 210u8];
    let mut tri_colors: TriColors = Vec::with_capacity(tri_indices.len());
    for ti in 0..tri_indices.len() {
        let h = (ti as u32).wrapping_mul(2246822519);
        let jitter = ((h >> 29) as i32) - 4; // [-4..+3]
        let add = jitter * 3;
        let mut c = base;
        for ch in 0..3 {
            c[ch] = ((c[ch] as i32) + add).clamp(0, 255) as u8;
        }
        tri_colors.push(c);
    }

    Model {
        verts,
        tri_indices,
        tri_tex_coords: Vec::new(),
        tri_colors,
    }
}

fn append_mesh(
    dst_verts: &mut Vertices,
    dst_tris: &mut TriIndices,
    dst_tri_colors: &mut TriColors,
    src_verts: &[Vec3],
    src_tris: &[[usize; 3]],
    src_color: [u8; 4],
) {
    let base = dst_verts.len();
    dst_verts.extend_from_slice(src_verts);
    dst_tris.extend(src_tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    dst_tri_colors.extend(std::iter::repeat(src_color).take(src_tris.len()));
}

pub fn gen_lowpoly_sphere(stacks: usize, slices: usize, radius: f32) -> (Vertices, TriIndices) {
    let stacks = stacks.max(2);
    let slices = slices.max(3);

    let mut verts: Vertices = Vec::new();
    let mut tris: TriIndices = Vec::new();

    let mut ring_index = vec![0usize; (stacks + 1) * (slices + 1)];
    let idx_at = |si: usize, pi: usize, slices: usize| si * (slices + 1) + pi;

    for si in 0..=stacks {
        let t = si as f32 / stacks as f32;
        let theta = std::f32::consts::PI * t; // [0..pi]
        let y = theta.cos() * radius;
        let r = theta.sin() * radius;
        for pi in 0..=slices {
            let u = pi as f32 / slices as f32;
            let phi = std::f32::consts::TAU * u;
            let x = phi.cos() * r;
            let z = phi.sin() * r;

            let vi = verts.len();
            verts.push(Vec3::new(x, y, z));
            ring_index[idx_at(si, pi, slices)] = vi;
        }
    }

    for si in 0..stacks {
        for pi in 0..slices {
            let i00 = ring_index[idx_at(si, pi, slices)];
            let i01 = ring_index[idx_at(si, pi + 1, slices)];
            let i10 = ring_index[idx_at(si + 1, pi, slices)];
            let i11 = ring_index[idx_at(si + 1, pi + 1, slices)];

            // CCW when viewed from outside.
            tris.push([i00, i10, i11]);
            tris.push([i00, i11, i01]);
        }
    }

    (verts, tris)
}

pub fn gen_palm_tree() -> Model {
    // Simple procedural palm: curved trunk + fronds + coconuts. All solid-color.
    let mut verts: Vertices = Vec::new();
    let mut tri_indices: TriIndices = Vec::new();
    let mut tri_colors: TriColors = Vec::new();

    // Trunk.
    {
        let height = 5.0;
        let stacks = 18usize;
        let slices = 10usize;
        let r_bottom = 0.35;
        let r_top = 0.18;
        let bulb = 0.20;
        let bend = 0.8;
        let trunk_color = [120u8, 92u8, 60u8, 255u8];

        let mut t_verts: Vertices = Vec::with_capacity((stacks + 1) * (slices + 1));
        let mut t_tris: TriIndices = Vec::with_capacity(stacks * slices * 2);
        let idx = |si: usize, pi: usize| si * (slices + 1) + pi;

        for si in 0..=stacks {
            let t = si as f32 / stacks as f32;
            let y = t * height;
            let center = Vec3::new(bend * t * t, y, 0.0);
            let taper = r_bottom + (r_top - r_bottom) * t;
            let bulge = bulb * (1.0 - t) * (1.0 - t);
            let r = (taper + bulge).max(0.02);

            for pi in 0..=slices {
                let u = pi as f32 / slices as f32;
                let phi = std::f32::consts::TAU * u;
                let x = phi.cos() * r;
                let z = phi.sin() * r;
                t_verts.push(center + Vec3::new(x, 0.0, z));
            }
        }

        for si in 0..stacks {
            for pi in 0..slices {
                let i00 = idx(si, pi);
                let i01 = idx(si, pi + 1);
                let i10 = idx(si + 1, pi);
                let i11 = idx(si + 1, pi + 1);

                // Outward winding.
                t_tris.push([i00, i10, i11]);
                t_tris.push([i00, i11, i01]);
            }
        }

        append_mesh(
            &mut verts,
            &mut tri_indices,
            &mut tri_colors,
            &t_verts,
            &t_tris,
            trunk_color,
        );
    }

    // Fronds (ribbon strips).
    {
        let frond_count = 8;
        let frond_len = 3.2;
        let frond_w = 0.35;
        let segs = 8usize;
        let leaf_color = [40u8, 150u8, 70u8, 255u8];

        for fi in 0..frond_count {
            let yaw = (fi as f32 / frond_count as f32) * std::f32::consts::TAU;
            let pitch = -0.55; // droop
            let rot =
                glam::Quat::from_rotation_y(yaw) * glam::Quat::from_rotation_z(pitch);

            let base = Vec3::new(0.8, 5.0, 0.0); // roughly matches trunk top/bend

            let mut f_verts: Vertices = Vec::with_capacity((segs + 1) * 2);
            let mut f_tris: TriIndices = Vec::with_capacity(segs * 2);

            for si in 0..=segs {
                let t = si as f32 / segs as f32;
                let x = t * frond_len;
                let y = -0.25 * (t * t) * frond_len; // curve down
                let center = Vec3::new(x, y, 0.0);
                let w = frond_w * (1.0 - 0.6 * t);
                let left = center + Vec3::new(0.0, 0.0, -w);
                let right = center + Vec3::new(0.0, 0.0, w);
                f_verts.push(base + rot * left);
                f_verts.push(base + rot * right);
            }

            let idx = |si: usize, lr: usize| si * 2 + lr;
            for si in 0..segs {
                let i00 = idx(si, 0);
                let i01 = idx(si, 1);
                let i10 = idx(si + 1, 0);
                let i11 = idx(si + 1, 1);

                // Double-sided leaves: emit both windings so backface culling won't hide them.
                let t0 = [i00, i10, i11];
                let t1 = [i00, i11, i01];
                f_tris.push(t0);
                f_tris.push(t1);
                f_tris.push([t0[0], t0[2], t0[1]]);
                f_tris.push([t1[0], t1[2], t1[1]]);
            }

            append_mesh(
                &mut verts,
                &mut tri_indices,
                &mut tri_colors,
                &f_verts,
                &f_tris,
                leaf_color,
            );
        }
    }

    // Coconuts.
    {
        let coco_color = [120u8, 90u8, 55u8, 255u8];
        let (s_verts, s_tris) = gen_lowpoly_sphere(6, 10, 0.22);
        let centers = [
            Vec3::new(0.75, 4.8, 0.15),
            Vec3::new(0.95, 4.7, -0.05),
            Vec3::new(0.85, 4.65, 0.35),
        ];
        for c in centers {
            let mut v2 = s_verts.clone();
            for v in &mut v2 {
                *v += c;
            }
            append_mesh(
                &mut verts,
                &mut tri_indices,
                &mut tri_colors,
                &v2,
                &s_tris,
                coco_color,
            );
        }
    }

    Model {
        verts,
        tri_indices,
        tri_tex_coords: Vec::new(),
        tri_colors,
    }
}

// Keep the name used by `src/sketch.rs`.
pub fn load_from_obj(path: &str) -> Model {
    load_model_from_obj(path)
}

pub fn load_model_from_obj(path: &str) -> Model {
    let (models, _materials) =
        tobj::load_obj(path, &tobj::LoadOptions::default()).expect("Failed to OBJ load file");
    // let materials = materials.expect("Failed to load MTL file");

    println!("Number of models          = {}", models.len());
    // println!("Number of materials       = {}", materials.len());

    let m = models.first().unwrap();
    let mesh = &m.mesh;
    println!("");
    println!("name             = \'{}\'", m.name);
    println!("mesh.material_id = {:?}", mesh.material_id);

    println!("face_count       = {}", mesh.face_arities.len());
    let mut next_face = 0;
    for face in 0..mesh.face_arities.len() {
        let end = next_face + mesh.face_arities[face] as usize;

        let face_indices = &mesh.indices[next_face..end];
        println!(" face[{}].indices          = {:?}", face, face_indices);

        if !mesh.texcoord_indices.is_empty() {
            let texcoord_face_indices = &mesh.texcoord_indices[next_face..end];
            println!(
                " face[{}].texcoord_indices = {:?}",
                face, texcoord_face_indices
            );
        }
        if !mesh.normal_indices.is_empty() {
            let normal_face_indices = &mesh.normal_indices[next_face..end];
            println!(
                " face[{}].normal_indices   = {:?}",
                face, normal_face_indices
            );
        }

        next_face = end;
    }

    // Normals and texture coordinates are also loaded, but not printed in
    // this example.
    println!("positions        = {}", mesh.positions.len() / 3);
    assert!(mesh.positions.len() % 3 == 0);

    for vtx in 0..mesh.positions.len() / 3 {
        println!(
            "              position[{}] = ({}, {}, {})",
            vtx,
            mesh.positions[3 * vtx],
            mesh.positions[3 * vtx + 1],
            mesh.positions[3 * vtx + 2]
        );
    }

    Model {
        verts: cube::get_cube_verts(),
        tri_indices: cube::gen_cube_tri_indices(),
        tri_tex_coords: cube::gen_cube_tex_coords(),
        tri_colors: Vec::new(),
    }
}
