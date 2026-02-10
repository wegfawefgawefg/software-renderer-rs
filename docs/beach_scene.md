# Beach Scene Roadmap

This renderer currently supports:
- Solid-color triangle meshes (per-triangle RGBA)
- Textured triangle meshes (UV + perspective-correct interpolation)
- Z-buffer
- Backface culling and OpenGL-style frustum clipping

The goal: a simple procedural "beach" scene with a sandy hill/island, animated water, a palm tree,
and a few extra props (clouds, floating boxes).

## Procedural Meshes

### Sandy Hill (done)
- Low-poly hemisphere (bottom half removed)
- Vertically squashed
- One color per face with slight per-face jitter for faceted look

### Water (done)
- XZ grid plane (solid-color)
- Vertex Y animated each frame via a simple wave function
- Semi-transparent color (alpha blend)

### Palm Tree (next)

Build it from 3 parts:

1. Trunk (procedural cylinder)
- Rings along height and slices around
- Radius function `r(t)`:
  - taper: lerp(bottom, top)
  - bulb: extra radius near base (`(1-t)^2` or gaussian bump)
- Bend:
  - offset the ring center along a curve (`center(t) = (bend * t^2, y, 0)`)
  - optional: orient cross-section using the curve tangent for nicer bends

2. Fronds (procedural ribbon strips)
- Each frond is a subdivided quad strip (two verts per segment)
- Curve downward by rotating segments or using a simple parametric curve
- Instance 6-10 fronds around the crown with different yaw/pitch

3. Coconuts (tiny low-poly spheres)
- 2-4 small spheres clustered near the crown

## Raycasts / Placement

For placing props "on the sand", we want raycasts against the hill mesh.

Suggested approach:
- Build a static triangle list for the hill in world space.
- Build a BVH (AABB tree) over triangles (fast traversal, no tuning parameters).
- `raycast(ray_origin, ray_dir) -> Option<Hit { t, pos, normal }>`
- Use the hit position/normal to place and orient objects (palm trunk base, floating items).

Water placement should use the analytic wave function directly (ray against `y = wave(x,z,t)`),
since the water mesh deforms every frame.

