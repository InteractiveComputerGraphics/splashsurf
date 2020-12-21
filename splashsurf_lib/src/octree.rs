use nalgebra::Vector3;
use rayon::prelude::*;
use rayon::ScopeFifo;
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
    /// Creates a new [Octree] with a single leaf node containing all vertices
    pub fn new<R: Real>(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        Self {
            root: OctreeNode::new_root(grid, n_particles),
        }
    }

    /// Subdivide the octree recursively using the given splitting criterion
    pub fn subdivide_recursively<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        profile!("Octree::subdivide_recursively");
        self.root
            .subdivide_recursively(grid, particle_positions, particles_per_cell);
    }

    /// Subdivide the octree recursively in parallel using the given splitting criterion
    pub fn subdivide_recursively_par<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        profile!("Octree::subdivide_recursively_par");
        rayon::scope_fifo(|s| {
            self.root
                .subdivide_recursively_par(s, grid, particle_positions, particles_per_cell);
        });
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
    pub fn into_hexmesh<R: Real>(
        &self,
        grid: &UniformGrid<I, R>,
        only_non_empty: bool,
    ) -> HexMesh3d<R> {
        profile!("convert octree into hexmesh");

        let mut mesh = HexMesh3d {
            vertices: Vec::new(),
            cells: Vec::new(),
        };

        for node in self.depth_first_iter() {
            if node.is_leaf() {
                if only_non_empty && node.particles().map(|p| p.is_empty()).unwrap_or(true) {
                    continue;
                }

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

impl<I: Index> OctreeNode<I> {
    pub fn min_corner(&self) -> &PointIndex<I> {
        &self.min_corner
    }

    pub fn max_corner(&self) -> &PointIndex<I> {
        &self.max_corner
    }

    /// Returns whether this is a leaf node, i.e. it may contain particles and has no child nodes
    pub fn is_leaf(&self) -> bool {
        self.body.is_leaf()
    }

    /// Returns a slice of the particles of this node if it is a leaf node
    pub fn particles(&self) -> Option<&[usize]> {
        self.body.particles()
    }

    /// Retruns a slice of the child nodes if this is not a leaf node
    pub fn children(&self) -> Option<&[Box<OctreeNode<I>>]> {
        self.body.children()
    }

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

    fn subdivide_recursively<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        if let Some(particles) = self.body.particles() {
            // Check if this leaf is already below the limit of particles per cell
            if particles.len() < particles_per_cell {
                return;
            }
        } else {
            // Early out if called on a non-leaf node
            return;
        }

        // Perform one octree split on the leaf
        self.subdivide(grid, particle_positions);

        // TODO: Replace recursion with iteration
        // TODO: Parallelize using tasks

        // Continue subdivision recursively in the new child nodes
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                child_node.subdivide_recursively(grid, particle_positions, particles_per_cell);
            }
        }
    }

    fn subdivide_recursively_par<'scope, R: Real>(
        &'scope mut self,
        s: &ScopeFifo<'scope>,
        grid: &'scope UniformGrid<I, R>,
        particle_positions: &'scope [Vector3<R>],
        particles_per_cell: usize,
    ) {
        if let Some(particles) = self.body.particles() {
            // Check if this leaf is already below the limit of particles per cell
            if particles.len() < particles_per_cell {
                return;
            }
        } else {
            // Early out if called on a non-leaf node
            return;
        }

        // Perform one octree split on the leaf
        self.subdivide_par(grid, particle_positions);

        // TODO: Replace recursion with iteration
        // TODO: Parallelize using tasks

        // Continue subdivision recursively in the new child nodes
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                s.spawn_fifo(move |s| {
                    child_node.subdivide_recursively_par(
                        s,
                        grid,
                        particle_positions,
                        particles_per_cell,
                    )
                });
            }
        }
    }

    fn subdivide<R: Real>(&mut self, grid: &UniformGrid<I, R>, particle_positions: &[Vector3<R>]) {
        if !can_split(&self.min_corner, &self.max_corner) {
            return;
        }

        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut octants = vec![Octant::NegNegNeg; particles.len()];
            let mut counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            // Classify all particles of this leaf into its octants
            assert_eq!(particles.len(), octants.len());
            for (particle, octant) in particles.iter().copied().zip(octants.iter_mut()) {
                let relative_pos = particle_positions[particle] - split_coordinates;
                *octant = OctantAxisDirections::classify(&relative_pos).into();
                counters[*octant as usize] += 1;
            }

            // Construct the node for each octant
            let mut children = SmallVec::with_capacity(8);
            for (current_octant, octant_particle_count) in
                Octant::all().iter().copied().zip(counters.iter().copied())
            {
                let current_octant_dir = OctantAxisDirections::from(current_octant);

                let min_corner = current_octant_dir
                    .combine_point_index(grid, &self.min_corner, &split_point)
                    .expect("Failed to get corner point of octree subcell");
                let max_corner = current_octant_dir
                    .combine_point_index(grid, &split_point, &self.max_corner)
                    .expect("Failed to get corner point of octree subcell");

                let mut octant_particles = SmallVec::with_capacity(octant_particle_count);
                octant_particles.extend(
                    particles
                        .iter()
                        .copied()
                        .zip(octants.iter())
                        // Skip particles from other octants
                        .filter(|(_, &particle_i_octant)| particle_i_octant == current_octant)
                        .map(|(particle_i, _)| particle_i),
                );
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

    fn subdivide_par<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
    ) {
        if !can_split(&self.min_corner, &self.max_corner) {
            return;
        }

        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut octants = vec![Octant::NegNegNeg; particles.len()];
            let counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            // Classify all particles of this leaf into its octants
            assert_eq!(particles.len(), octants.len());
            let counters: [usize; 8] = particles
                .as_slice()
                .par_iter()
                .copied()
                .zip(octants.par_iter_mut())
                .fold_with(counters, |mut counters, (particle, octant)| {
                    let relative_pos = particle_positions[particle] - split_coordinates;
                    *octant = OctantAxisDirections::classify(&relative_pos).into();
                    counters[*octant as usize] += 1;
                    counters
                })
                .reduce(
                    || counters.clone(),
                    |mut counters, c| {
                        for (c_a, c_b) in counters.iter_mut().zip(c.iter().copied()) {
                            *c_a += c_b;
                        }
                        counters
                    },
                );

            // Construct the node for each octant
            let children = Octant::all()
                .par_iter()
                .copied()
                .zip(counters.par_iter().copied())
                .map(|(current_octant, octant_particle_count)| {
                    let current_octant_dir = OctantAxisDirections::from(current_octant);

                    let min_corner = current_octant_dir
                        .combine_point_index(grid, &self.min_corner, &split_point)
                        .expect("Failed to get corner point of octree subcell");
                    let max_corner = current_octant_dir
                        .combine_point_index(grid, &split_point, &self.max_corner)
                        .expect("Failed to get corner point of octree subcell");

                    let mut octant_particles = SmallVec::with_capacity(octant_particle_count);
                    octant_particles.extend(
                        particles
                            .iter()
                            .copied()
                            .zip(octants.iter())
                            // Skip particles from other octants
                            .filter(|(_, &particle_i_octant)| particle_i_octant == current_octant)
                            .map(|(particle_i, _)| particle_i),
                    );
                    assert_eq!(octant_particles.len(), octant_particle_count);

                    let child = Box::new(OctreeNode::new_leaf(
                        min_corner,
                        max_corner,
                        octant_particles,
                    ));

                    child
                })
                .collect::<Vec<_>>();

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
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

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct OctantAxisDirections {
    x_axis: Direction,
    y_axis: Direction,
    z_axis: Direction,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
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

impl OctantAxisDirections {
    #[inline(always)]
    pub const fn all() -> &'static [OctantAxisDirections; 8] {
        &ALL_OCTANT_DIRECTIONS
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

impl From<Octant> for OctantAxisDirections {
    fn from(octant: Octant) -> Self {
        Self::from_octant(octant)
    }
}

impl Octant {
    #[inline(always)]
    pub const fn all() -> &'static [Octant; 8] {
        &ALL_OCTANTS
    }

    #[inline(always)]
    pub const fn from_directions(directions: OctantAxisDirections) -> Self {
        use Direction::*;
        let OctantAxisDirections {
            x_axis,
            y_axis,
            z_axis,
        } = directions;
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

impl From<OctantAxisDirections> for Octant {
    fn from(directions: OctantAxisDirections) -> Self {
        Self::from_directions(directions)
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

const ALL_OCTANT_DIRECTIONS: [OctantAxisDirections; 8] = [
    OctantAxisDirections::from_octant(Octant::NegNegNeg),
    OctantAxisDirections::from_octant(Octant::PosNegNeg),
    OctantAxisDirections::from_octant(Octant::NegPosNeg),
    OctantAxisDirections::from_octant(Octant::PosPosNeg),
    OctantAxisDirections::from_octant(Octant::NegNegPos),
    OctantAxisDirections::from_octant(Octant::PosNegPos),
    OctantAxisDirections::from_octant(Octant::NegPosPos),
    OctantAxisDirections::from_octant(Octant::PosPosPos),
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
    fn test_octant_directions_iter_all_consistency() {
        assert_eq!(Octant::all().len(), OctantAxisDirections::all().len());
        for (octant, octant_directions) in Octant::all()
            .iter()
            .copied()
            .zip(OctantAxisDirections::all().iter().copied())
        {
            assert_eq!(octant, Octant::from(octant_directions));
            assert_eq!(octant_directions, OctantAxisDirections::from(octant));
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