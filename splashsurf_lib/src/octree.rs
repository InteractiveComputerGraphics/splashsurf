use nalgebra::Vector3;
use smallvec::SmallVec;

use crate::mesh::HexMesh3d;
use crate::uniform_grid::{Direction, PointIndex};
use crate::{Index, Real, UniformGrid};

/// Octree representation of a set of particles
#[derive(Clone, Debug)]
pub struct Octree<I: Index> {
    root: OctreeNode<I>,
}

/// A single node in an Octree, may be a leaf (containing particles) or a node with further child nodes
#[derive(Clone, Debug)]
pub struct OctreeNode<I: Index> {
    min_corner: PointIndex<I>,
    max_corner: PointIndex<I>,
    body: NodeBody<I>,
}

type OctreeNodeChildrenStorage<I> = SmallVec<[Box<OctreeNode<I>>; 8]>;
type OctreeNodeParticleStorage = SmallVec<[usize; 8]>;

#[derive(Clone, Debug)]
enum NodeBody<I: Index> {
    Children {
        children: OctreeNodeChildrenStorage<I>,
    },
    Leaf {
        particles: OctreeNodeParticleStorage,
    },
}

impl<I: Index> Octree<I> {
    pub fn new<R: Real>(
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) -> Self {
        profile!("build octree");

        let mut root = OctreeNode::new_root(grid, particle_positions.len());
        root.subdivide_recursively(grid, particle_positions, particles_per_cell);
        Self { root }
    }

    /// Returns a reference to the root node of the octree
    pub fn root(&self) -> &OctreeNode<I> {
        &self.root
    }

    /// Returns an iterator that yields all nodes of the octree in depth-first order
    pub fn depth_first_iter(&self) -> impl Iterator<Item = &OctreeNode<I>> {
        let mut queue = Vec::new();
        queue.push(&self.root);

        let iter = move || -> Option<&OctreeNode<I>> {
            if let Some(next_node) = queue.pop() {
                // Check if the node has children
                if let Some(children) = next_node.children() {
                    // Enqueue all children
                    queue.extend(children.iter().rev().map(std::ops::Deref::deref));
                }

                Some(next_node)
            } else {
                None
            }
        };

        std::iter::from_fn(iter)
    }

    /// Constructs a hex mesh visualizing the cells of the octree, may contain hanging and duplicate vertices as cells are not connected
    pub fn into_hexmesh<R: Real>(&self, grid: &UniformGrid<I, R>) -> HexMesh3d<R> {
        profile!("convert octree into hexmesh");

        let mut mesh = HexMesh3d {
            vertices: Vec::new(),
            cells: Vec::new(),
        };

        for node in self.depth_first_iter() {
            if node.is_leaf() {
                let lower_coords = grid.point_coordinates(&node.min_corner);
                let upper_coords = grid.point_coordinates(&node.max_corner);

                let vertices = vec![
                    lower_coords,
                    Vector3::new(upper_coords[0], lower_coords[1], lower_coords[2]),
                    Vector3::new(upper_coords[0], upper_coords[1], lower_coords[2]),
                    Vector3::new(lower_coords[0], upper_coords[1], lower_coords[2]),
                    Vector3::new(lower_coords[0], lower_coords[1], upper_coords[2]),
                    Vector3::new(upper_coords[0], lower_coords[1], upper_coords[2]),
                    upper_coords,
                    Vector3::new(lower_coords[0], upper_coords[1], upper_coords[2]),
                ];

                let offset = mesh.vertices.len();
                let cell = [
                    offset + 0,
                    offset + 1,
                    offset + 2,
                    offset + 3,
                    offset + 4,
                    offset + 5,
                    offset + 6,
                    offset + 7,
                ];

                mesh.vertices.extend(vertices);
                mesh.cells.push(cell);
            }
        }

        mesh
    }
}

impl<I: Index> NodeBody<I> {
    pub fn new_leaf<IndexVec: Into<OctreeNodeParticleStorage>>(particles: IndexVec) -> Self {
        NodeBody::Leaf {
            particles: particles.into(),
        }
    }

    pub fn new_with_children<OctreeNodeVec: Into<OctreeNodeChildrenStorage<I>>>(
        children: OctreeNodeVec,
    ) -> Self {
        let children = children.into();
        assert_eq!(children.len(), 8);
        NodeBody::Children { children }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            NodeBody::Leaf { .. } => true,
            _ => false,
        }
    }

    pub fn particles(&self) -> Option<&[usize]> {
        match self {
            NodeBody::Leaf { particles } => Some(particles.as_slice()),
            _ => None,
        }
    }

    pub fn children(&self) -> Option<&[Box<OctreeNode<I>>]> {
        match self {
            NodeBody::Children { children } => Some(children.as_slice()),
            _ => None,
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut [Box<OctreeNode<I>>]> {
        match self {
            NodeBody::Children { children } => Some(children.as_mut_slice()),
            _ => None,
        }
    }
}

impl<I: Index> OctreeNode<I> {
    fn new_root<R: Real>(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        let n_points = grid.points_per_dim();
        let min_point = [I::zero(), I::zero(), I::zero()];
        let max_point = [
            n_points[0] - I::one(),
            n_points[1] - I::one(),
            n_points[2] - I::one(),
        ];

        Self {
            min_corner: grid
                .get_point(&min_point)
                .expect("Cannot get lower corner of grid"),
            max_corner: grid
                .get_point(&max_point)
                .expect("Cannot get upper corner of grid"),
            body: NodeBody::new_leaf((0..n_particles).collect::<SmallVec<_>>()),
        }
    }

    fn new_leaf(
        min_corner: PointIndex<I>,
        max_corner: PointIndex<I>,
        particles: OctreeNodeParticleStorage,
    ) -> Self {
        Self {
            min_corner,
            max_corner,
            body: NodeBody::new_leaf(particles),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.body.is_leaf()
    }

    pub fn particles(&self) -> Option<&[usize]> {
        self.body.particles()
    }

    pub fn children(&self) -> Option<&[Box<OctreeNode<I>>]> {
        self.body.children()
    }

    fn subdivide_recursively<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        if let Some(particles) = self.body.particles() {
            if particles.len() < particles_per_cell {
                return;
            }
        } else {
            return;
        }

        // TODO: Replace recursion using tree visitor?
        self.subdivide(grid, particle_positions);
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                child_node.subdivide_recursively(grid, particle_positions, particles_per_cell);
            }
        }
    }

    fn subdivide<R: Real>(&mut self, grid: &UniformGrid<I, R>, particle_positions: &[Vector3<R>]) {
        if !can_split(&self.min_corner, &self.max_corner) {
            return;
        }

        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles } = &self.body {
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let particles = particles.clone();
            let mut octants = vec![OctantFlags::default(); particles.len()];
            let mut counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            assert_eq!(particles.len(), octants.len());
            for (particle, octant) in particles.iter().copied().zip(octants.iter_mut()) {
                let relative_pos = particle_positions[particle] - split_coordinates;
                *octant = OctantFlags::classify(&relative_pos);
                counters[Octant::from_flags(*octant) as usize] += 1;
            }

            let mut children = SmallVec::with_capacity(8);
            for (octant_flags, octant_particle_count) in OctantFlags::all()
                .iter()
                .copied()
                .zip(counters.iter().copied())
            {
                let min_corner = octant_flags
                    .combine_point_index(grid, &self.min_corner, &split_point)
                    .expect("Failed to get corner point of octree subcell");
                let max_corner = octant_flags
                    .combine_point_index(grid, &split_point, &self.max_corner)
                    .expect("Failed to get corner point of octree subcell");

                let mut octant_particles = SmallVec::with_capacity(octant_particle_count);
                for (i, octant_i) in octants.iter().copied().enumerate() {
                    if octant_i == octant_flags {
                        octant_particles.push(i);
                    }
                }
                assert_eq!(octant_particles.len(), octant_particle_count);

                let child = Box::new(OctreeNode::new_leaf(
                    min_corner,
                    max_corner,
                    octant_particles,
                ));

                children.push(child);
            }

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct OctantFlags {
    x_axis: Direction,
    y_axis: Direction,
    z_axis: Direction,
}

impl OctantFlags {
    #[inline(always)]
    pub const fn all() -> &'static [OctantFlags; 8] {
        &ALL_OCTANT_FLAGS
    }

    #[inline(always)]
    pub const fn from_bool(x_positive: bool, y_positive: bool, z_positive: bool) -> Self {
        Self {
            x_axis: Direction::from_bool(x_positive),
            y_axis: Direction::from_bool(y_positive),
            z_axis: Direction::from_bool(z_positive),
        }
    }

    pub const fn from_octant(octant: Octant) -> Self {
        match octant {
            Octant::NegNegNeg => Self::from_bool(false, false, false),
            Octant::PosNegNeg => Self::from_bool(true, false, false),
            Octant::NegPosNeg => Self::from_bool(false, true, false),
            Octant::PosPosNeg => Self::from_bool(true, true, false),
            Octant::NegNegPos => Self::from_bool(false, false, true),
            Octant::PosNegPos => Self::from_bool(true, false, true),
            Octant::NegPosPos => Self::from_bool(false, true, true),
            Octant::PosPosPos => Self::from_bool(true, true, true),
        }
    }

    /// Classifies a point relative to zero into the corresponding octant
    #[inline(always)]
    pub fn classify<R: Real>(point: &Vector3<R>) -> Self {
        Self::from_bool(
            point[0].is_positive(),
            point[1].is_positive(),
            point[2].is_positive(),
        )
    }

    /// Combines two vectors by choosing between their components depending on the octant
    pub fn combine_point_index<I: Index, R: Real>(
        &self,
        grid: &UniformGrid<I, R>,
        lower: &PointIndex<I>,
        upper: &PointIndex<I>,
    ) -> Option<PointIndex<I>> {
        let lower = lower.index();
        let upper = upper.index();

        let combined_index = [
            if self.x_axis.is_positive() {
                upper[0]
            } else {
                lower[0]
            },
            if self.y_axis.is_positive() {
                upper[1]
            } else {
                lower[1]
            },
            if self.z_axis.is_positive() {
                upper[2]
            } else {
                lower[2]
            },
        ];

        grid.get_point(&combined_index)
    }
}

impl From<Octant> for OctantFlags {
    fn from(octant: Octant) -> Self {
        Self::from_octant(octant)
    }
}

impl Default for OctantFlags {
    fn default() -> Self {
        OctantFlags {
            x_axis: Direction::Negative,
            y_axis: Direction::Negative,
            z_axis: Direction::Negative,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Octant {
    NegNegNeg = 0,
    PosNegNeg = 1,
    NegPosNeg = 2,
    PosPosNeg = 3,
    NegNegPos = 4,
    PosNegPos = 5,
    NegPosPos = 6,
    PosPosPos = 7,
}

impl Octant {
    #[inline(always)]
    pub const fn all() -> &'static [Octant; 8] {
        &ALL_OCTANTS
    }

    #[inline(always)]
    pub const fn from_flags(flags: OctantFlags) -> Self {
        use Direction::*;
        let OctantFlags {
            x_axis,
            y_axis,
            z_axis,
        } = flags;
        match (x_axis, y_axis, z_axis) {
            (Negative, Negative, Negative) => Octant::NegNegNeg,
            (Positive, Negative, Negative) => Octant::PosNegNeg,
            (Negative, Positive, Negative) => Octant::NegPosNeg,
            (Positive, Positive, Negative) => Octant::PosPosNeg,
            (Negative, Negative, Positive) => Octant::NegNegPos,
            (Positive, Negative, Positive) => Octant::PosNegPos,
            (Negative, Positive, Positive) => Octant::NegPosPos,
            (Positive, Positive, Positive) => Octant::PosPosPos,
        }
    }
}

impl From<OctantFlags> for Octant {
    fn from(flags: OctantFlags) -> Self {
        Self::from_flags(flags)
    }
}

const ALL_OCTANTS: [Octant; 8] = [
    Octant::NegNegNeg,
    Octant::PosNegNeg,
    Octant::NegPosNeg,
    Octant::PosPosNeg,
    Octant::NegNegPos,
    Octant::PosNegPos,
    Octant::NegPosPos,
    Octant::PosPosPos,
];

const ALL_OCTANT_FLAGS: [OctantFlags; 8] = [
    OctantFlags::from_octant(Octant::NegNegNeg),
    OctantFlags::from_octant(Octant::PosNegNeg),
    OctantFlags::from_octant(Octant::NegPosNeg),
    OctantFlags::from_octant(Octant::PosPosNeg),
    OctantFlags::from_octant(Octant::NegNegPos),
    OctantFlags::from_octant(Octant::PosNegPos),
    OctantFlags::from_octant(Octant::NegPosPos),
    OctantFlags::from_octant(Octant::PosPosPos),
];

#[cfg(test)]
mod test_octant {
    use super::*;

    #[test]
    fn test_octant_iter_all_consistency() {
        for (i, octant) in Octant::all().iter().copied().enumerate() {
            assert_eq!(octant as usize, i);
            assert_eq!(octant, unsafe {
                std::mem::transmute::<u8, Octant>(i as u8)
            });
        }
    }

    #[test]
    fn test_octant_flags_iter_all_consistency() {
        assert_eq!(Octant::all().len(), OctantFlags::all().len());
        for (octant, octant_flags) in Octant::all()
            .iter()
            .copied()
            .zip(OctantFlags::all().iter().copied())
        {
            assert_eq!(octant, Octant::from(octant_flags));
            assert_eq!(octant_flags, OctantFlags::from(octant));
        }
    }
}

fn can_split<I: Index>(lower: &PointIndex<I>, upper: &PointIndex<I>) -> bool {
    let lower = lower.index();
    let upper = upper.index();

    upper[0] - lower[0] > I::one()
        && upper[1] - lower[1] > I::one()
        && upper[2] - lower[2] > I::one()
}

fn get_split_point<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    lower: &PointIndex<I>,
    upper: &PointIndex<I>,
) -> Option<PointIndex<I>> {
    let two = I::one() + I::one();

    let lower = lower.index();
    let upper = upper.index();

    let mid_indices = [
        (lower[0] + upper[0]) / two,
        (lower[1] + upper[1]) / two,
        (lower[2] + upper[2]) / two,
    ];

    grid.get_point(&mid_indices)
}
