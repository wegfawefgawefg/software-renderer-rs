use glam::{Vec2, Vec3};

pub fn get_cube_verts() -> Vec<Vec3> {
    let s = 0.5;
    // 24-vertex cube (4 verts per face) so each face can have its own UV orientation/seams.
    // Face vertex order is (top-left, bottom-left, bottom-right, top-right) when viewed from
    // outside the cube.
    vec![
        // +Z
        Vec3::new(-s, s, s),
        Vec3::new(-s, -s, s),
        Vec3::new(s, -s, s),
        Vec3::new(s, s, s),
        // -Z
        Vec3::new(s, s, -s),
        Vec3::new(s, -s, -s),
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, s, -s),
        // +X
        Vec3::new(s, s, s),
        Vec3::new(s, -s, s),
        Vec3::new(s, -s, -s),
        Vec3::new(s, s, -s),
        // -X
        Vec3::new(-s, s, -s),
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, -s, s),
        Vec3::new(-s, s, s),
        // +Y
        Vec3::new(-s, s, -s),
        Vec3::new(-s, s, s),
        Vec3::new(s, s, s),
        Vec3::new(s, s, -s),
        // -Y
        Vec3::new(-s, -s, s),
        Vec3::new(-s, -s, -s),
        Vec3::new(s, -s, -s),
        Vec3::new(s, -s, s),
    ]
}

type CubeTriIndices = Vec<[usize; 3]>;
pub fn gen_cube_tri_indices() -> CubeTriIndices {
    // CCW winding when viewed from outside the cube (right-handed coordinates).
    // Each face is 4 vertices; triangles are (0,1,2) and (0,2,3) in that local block.
    let mut tris = Vec::with_capacity(12);
    for face in 0..6 {
        let b = face * 4;
        tris.push([b + 0, b + 1, b + 2]);
        tris.push([b + 0, b + 2, b + 3]);
    }
    tris
}

type CubeTexCoords = Vec<[Vec2; 3]>;
pub fn gen_cube_tex_coords() -> CubeTexCoords {
    // Match the C++ reference: repeat wrap, v=0 at top (y down).
    //
    // Local face UVs for the 4-vert block (top-left, bottom-left, bottom-right, top-right).
    let tl = Vec2::new(0.0, 0.0);
    let bl = Vec2::new(0.0, 1.0);
    let br = Vec2::new(1.0, 1.0);
    let tr = Vec2::new(1.0, 0.0);

    let mut uvs = Vec::with_capacity(12);
    for _face in 0..6 {
        uvs.push([tl, bl, br]);
        uvs.push([tl, br, tr]);
    }
    uvs
}
