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
