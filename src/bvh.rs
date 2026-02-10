use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3, // should be normalized
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub pos: Vec3,
    pub normal: Vec3,
    pub tri_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn expand_aabb(&mut self, o: &Aabb) {
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn hit(&self, ray: &Ray, mut tmin: f32, mut tmax: f32) -> bool {
        // Slab test.
        for axis in 0..3 {
            let o = ray.origin[axis];
            let d = ray.dir[axis];
            let inv_d = 1.0 / d;
            let mut t0 = (self.min[axis] - o) * inv_d;
            let mut t1 = (self.max[axis] - o) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmax < tmin {
                return false;
            }
        }
        true
    }
}

fn tri_bounds(t: &Triangle) -> Aabb {
    let mut b = Aabb::empty();
    b.expand_point(t.a);
    b.expand_point(t.b);
    b.expand_point(t.c);
    b
}

fn tri_centroid(t: &Triangle) -> Vec3 {
    (t.a + t.b + t.c) / 3.0
}

fn ray_tri(ray: &Ray, tri: &Triangle, t_min: f32, t_max: f32) -> Option<(f32, Vec3)> {
    // Moller-Trumbore.
    let eps = 1e-7;
    let e1 = tri.b - tri.a;
    let e2 = tri.c - tri.a;
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < eps {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - tri.a;
    let u = tvec.dot(p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = tvec.cross(e1);
    let v = ray.dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    if t < t_min || t > t_max {
        return None;
    }
    let n = e1.cross(e2).normalize_or_zero();
    Some((t, n))
}

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    left: i32,
    right: i32,
    start: u32,
    count: u16,
}

impl Node {
    fn leaf(bounds: Aabb, start: u32, count: u16) -> Self {
        Self {
            bounds,
            left: -1,
            right: -1,
            start,
            count,
        }
    }

    fn inner(bounds: Aabb, left: i32, right: i32) -> Self {
        Self {
            bounds,
            left,
            right,
            start: 0,
            count: 0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left < 0
    }
}

pub struct Bvh {
    tris: Vec<Triangle>,
    tri_ids: Vec<usize>,
    nodes: Vec<Node>,
}

impl Bvh {
    pub fn build(tris: Vec<Triangle>) -> Self {
        let mut tri_ids: Vec<usize> = (0..tris.len()).collect();
        let mut nodes = Vec::new();
        let mut bvh = Self {
            tris,
            tri_ids,
            nodes,
        };
        let _root = bvh.build_node(0, bvh.tri_ids.len());
        bvh
    }

    fn build_node(&mut self, start: usize, end: usize) -> i32 {
        let count = end - start;

        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        for &id in &self.tri_ids[start..end] {
            let tb = tri_bounds(&self.tris[id]);
            bounds.expand_aabb(&tb);
            centroid_bounds.expand_point(tri_centroid(&self.tris[id]));
        }

        // Leaf threshold.
        if count <= 8 {
            let node_id = self.nodes.len() as i32;
            self.nodes
                .push(Node::leaf(bounds, start as u32, count as u16));
            return node_id;
        }

        // Split on largest centroid axis.
        let ext = centroid_bounds.max - centroid_bounds.min;
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };

        let mid = (start + end) / 2;
        self.tri_ids[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            tri_centroid(&self.tris[a])[axis]
                .total_cmp(&tri_centroid(&self.tris[b])[axis])
        });

        let node_id = self.nodes.len() as i32;
        // placeholder
        self.nodes.push(Node::leaf(bounds, 0, 0));

        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[node_id as usize] = Node::inner(bounds, left, right);
        node_id
    }

    pub fn raycast(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_t = t_max;
        let mut best_hit: Option<Hit> = None;

        let mut stack = Vec::with_capacity(64);
        stack.push(0i32);

        while let Some(nid) = stack.pop() {
            let n = &self.nodes[nid as usize];
            if !n.bounds.hit(ray, t_min, best_t) {
                continue;
            }

            if n.is_leaf() {
                let s = n.start as usize;
                let e = s + n.count as usize;
                for &tri_index in &self.tri_ids[s..e] {
                    if let Some((t, normal)) = ray_tri(ray, &self.tris[tri_index], t_min, best_t) {
                        best_t = t;
                        let pos = ray.origin + ray.dir * t;
                        best_hit = Some(Hit {
                            t,
                            pos,
                            normal,
                            tri_index,
                        });
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }

        best_hit
    }
}

