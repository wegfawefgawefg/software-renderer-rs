use glam::{Vec2, Vec3};

use crate::cube;

pub type Vertices = Vec<Vec3>;
pub type Normals = Vec<Vec3>;
pub type TriVertexNormals = Vec<[Vec3; 3]>;
pub type Indices = Vec<usize>;
pub type TriIndices = Vec<[usize; 3]>;
pub type TexCoords = Vec<Vec2>;
pub type TriTexCoords = Vec<[Vec2; 3]>;

pub struct Model {
    pub verts: Vertices,
    pub tri_indices: TriIndices,
    pub tri_tex_coords: TriTexCoords,
}

pub fn load_cube() -> Model {
    Model {
        verts: cube::get_cube_verts(),
        tri_indices: cube::gen_cube_tri_indices(),
        tri_tex_coords: cube::gen_cube_tex_coords(),
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
    }
}
