use nalgebra::Vector3;
use rayon::prelude::*;
use rayon::ScopeFifo;
use smallvec::SmallVec;

use crate::mesh::HexMesh3d;
use crate::uniform_grid::{PointIndex, UniformGrid};
use crate::{
    AxisAlignedBoundingBox, AxisAlignedBoundingBox3d, GridConstructionError, Index, Real,
    ThreadSafe,
};

use octant_helper::{Octant, OctantAxisDirections, OctantDirectionFlags};

// TODO: Unify splitting with/without margin using some generic approach

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
        ghost_particle_count: usize,
    },
}

/// Trait that is used by an octree to decide whether an octree node should be further split or subdivided
trait LeafSplitCriterion<I: Index> {
    /// Returns whether the specified node should be split
    fn split_leaf(&self, node: &OctreeNode<I>) -> bool;
}

/// Split criterion that decides based on whether the number of non-ghost particles in a node is above a limit
struct MaxNonGhostParticleLeafSplitCriterion {
    max_particles: usize,
}

impl MaxNonGhostParticleLeafSplitCriterion {
    fn new(max_particles: usize) -> Self {
        Self { max_particles }
    }
}

impl<I: Index> LeafSplitCriterion<I> for MaxNonGhostParticleLeafSplitCriterion {
    /// Returns true if the number of non-ghost particles in a node is above a limit
    fn split_leaf(&self, node: &OctreeNode<I>) -> bool {
        if let NodeBody::Leaf {
            particles,
            ghost_particle_count,
        } = &node.body
        {
            // Check if this leaf is already below the limit of particles per cell
            return particles.len() - *ghost_particle_count > self.max_particles;
        } else {
            // Early out if called on a non-leaf node
            return false;
        }
    }
}

/// Split criterion that decides based on whether the node's extents are larger than 1 cell in all dimensions
struct MinimumExtentSplitCriterion<I> {
    minimum_extent: I,
}

impl<I: Index> MinimumExtentSplitCriterion<I> {
    fn new(minimum_extent: I) -> Self {
        Self { minimum_extent }
    }
}

impl<I: Index> LeafSplitCriterion<I> for MinimumExtentSplitCriterion<I> {
    /// Returns true only if all extents of the octree node are larger than 1 cell
    fn split_leaf(&self, node: &OctreeNode<I>) -> bool {
        let lower = node.min_corner.index();
        let upper = node.max_corner.index();

        upper[0] - lower[0] > self.minimum_extent
            && upper[1] - lower[1] > self.minimum_extent
            && upper[2] - lower[2] > self.minimum_extent
    }
}

impl<I: Index, A, B> LeafSplitCriterion<I> for (A, B)
where
    A: LeafSplitCriterion<I>,
    B: LeafSplitCriterion<I>,
{
    fn split_leaf(&self, node: &OctreeNode<I>) -> bool {
        self.0.split_leaf(node) && self.1.split_leaf(node)
    }
}

fn default_split_criterion<I: Index>(
    particles_per_cell: usize,
) -> (
    MaxNonGhostParticleLeafSplitCriterion,
    MinimumExtentSplitCriterion<I>,
) {
    (
        MaxNonGhostParticleLeafSplitCriterion::new(particles_per_cell),
        MinimumExtentSplitCriterion::new(I::one()),
    )
}

/*
trait OctantClassifier<R: Real> {
    fn classify_particles(
        &mut self,
        split_coordinates: &Vector3<R>,
        particle_positions: &[Vector3<R>],
        particles: &[usize],
    );

    fn classify_particles_par(
        &mut self,
        split_coordinates: &Vector3<R>,
        particle_positions: &[Vector3<R>],
        particles: &[usize],
    );

    fn octant_particles(&self, octant: Octant, particles: &[usize]) -> OctreeNodeParticleStorage;
    fn octant_ghost_particles_count(&self, octant: Octant) -> usize;
}

struct MarginClassifier<R: Real> {
    margin: R,

    particle_octant_flags: Vec<OctantDirectionFlags>,
    counters: [usize; 8],
    non_ghost_counters: [usize; 8],
}

impl<R: Real> MarginClassifier<R> {
    fn new(margin: R) -> Self {
        Self {
            margin,
            particle_octant_flags: Vec::new(),
            counters: [0, 0, 0, 0, 0, 0, 0, 0],
            non_ghost_counters: [0, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl<R: Real> OctantClassifier<R> for MarginClassifier<R> {
    fn classify_particles(
        &mut self,
        split_coordinates: &Vector3<R>,
        particle_positions: &[Vector3<R>],
        particles: &[usize],
    ) {
        self.particle_octant_flags.clear();
        self.particle_octant_flags
            .resize(particles.len(), OctantDirectionFlags::empty());

        // Initial values for the fold
        let zeros = || ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);

        let margin = self.margin;
        let (counters, non_ghost_counters) = particles
            .iter()
            .copied()
            .zip(self.particle_octant_flags.iter_mut())
            .fold(
                zeros(),
                |(mut counters, mut non_ghost_counters), (particle_idx, particle_octant_flags)| {
                    let relative_pos = particle_positions[particle_idx] - split_coordinates;

                    // Check what the main octant of the particle is (to count ghost particles)
                    {
                        let main_octant: Octant =
                            OctantAxisDirections::classify(&relative_pos).into();
                        non_ghost_counters[main_octant as usize] += 1;
                    }

                    // Classify into all octants with margin
                    {
                        *particle_octant_flags =
                            OctantDirectionFlags::classify_with_margin(&relative_pos, margin);

                        // Increase the counter of each octant that contains the current particle
                        OctantDirectionFlags::all_unique_octants()
                            .iter()
                            .zip(counters.iter_mut())
                            .filter(|(octant, _)| particle_octant_flags.contains(**octant))
                            .for_each(|(_, counter)| {
                                *counter += 1;
                            });
                    }

                    (counters, non_ghost_counters)
                },
            );

        self.counters = counters;
        self.non_ghost_counters = non_ghost_counters;
    }

    fn classify_particles_par(
        &mut self,
        split_coordinates: &Vector3<R>,
        particle_positions: &[Vector3<R>],
        particles: &[usize],
    ) {
        self.particle_octant_flags.clear();
        self.particle_octant_flags
            .resize(particles.len(), OctantDirectionFlags::empty());

        // Initial values for the fold
        let zeros = || ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);

        let margin = self.margin;
        let (counters, non_ghost_counters) = particles
            .par_iter()
            .copied()
            .zip(self.particle_octant_flags.par_iter_mut())
            .fold(
                zeros,
                |(mut counters, mut non_ghost_counters), (particle_idx, particle_octant_flags)| {
                    let relative_pos = particle_positions[particle_idx] - split_coordinates;

                    // Check what the main octant of the particle is (to count ghost particles)
                    {
                        let main_octant: Octant =
                            OctantAxisDirections::classify(&relative_pos).into();
                        non_ghost_counters[main_octant as usize] += 1;
                    }

                    // Classify particle into all octants with margin
                    {
                        *particle_octant_flags =
                            OctantDirectionFlags::classify_with_margin(&relative_pos, margin);

                        // Increase the counter of each octant that contains the current particle
                        OctantDirectionFlags::all_unique_octants()
                            .iter()
                            .zip(counters.iter_mut())
                            .filter(|(octant, _)| particle_octant_flags.contains(**octant))
                            .for_each(|(_, counter)| {
                                *counter += 1;
                            });
                    }

                    (counters, non_ghost_counters)
                },
            )
            // Sum up all counter arrays
            .reduce(
                zeros,
                |(mut counters_acc, mut non_ghost_counters_acc), (counters, non_ghost_counters)| {
                    for i in 0..8 {
                        counters_acc[i] += counters[i];
                        non_ghost_counters_acc[i] += non_ghost_counters[i];
                    }
                    (counters_acc, non_ghost_counters_acc)
                },
            );

        self.counters = counters;
        self.non_ghost_counters = non_ghost_counters;
    }

    fn octant_particles(&self, octant: Octant, particles: &[usize]) -> OctreeNodeParticleStorage {
        let mut octant_particles = SmallVec::with_capacity(self.counters[octant as usize]);
        let current_octant_flags = OctantDirectionFlags::from(octant);

        octant_particles.extend(
            particles
                .iter()
                .copied()
                .zip(self.particle_octant_flags.iter())
                // Skip particles from other octants
                .filter(|(_, &particle_i_octant)| particle_i_octant.contains(current_octant_flags))
                .map(|(particle_i, _)| particle_i),
        );
        octant_particles
    }

    fn octant_ghost_particles_count(&self, octant: Octant) -> usize {
        self.counters[octant as usize] - self.non_ghost_counters[octant as usize]
    }
}
*/

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
        profile!("octree subdivide_recursively");

        let split_criterion = default_split_criterion(particles_per_cell);
        self.root
            .subdivide_recursively(grid, particle_positions, &split_criterion);
    }

    /// Subdivide the octree recursively in parallel using the given splitting criterion
    pub fn subdivide_recursively_par<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        profile!("octree subdivide_recursively_par");

        let split_criterion = default_split_criterion(particles_per_cell);
        rayon::scope_fifo(|s| {
            self.root
                .subdivide_recursively_par(s, grid, particle_positions, &split_criterion);
        });
    }

    /// Subdivide the octree recursively using the given splitting criterion
    pub fn subdivide_recursively_margin<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
        margin: R,
    ) {
        profile!("octree subdivide_recursively_margin");

        /*
        let split_criterion = default_split_criterion(particles_per_cell);
        self.root
            .subdivide_recursively_margin(grid, particle_positions, &split_criterion, margin);
         */

        let split_criterion = default_split_criterion(particles_per_cell);
        rayon::scope_fifo(|s| {
            self.root.subdivide_recursively_margin_par(
                s,
                grid,
                particle_positions,
                &split_criterion,
                margin,
            );
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

    /// Returns an iterator that yields all leafs of the octree in depth-first order
    pub fn leaf_iter(&self) -> impl Iterator<Item = &OctreeNode<I>> {
        self.depth_first_iter().filter(|n| n.is_leaf())
    }

    /// Constructs a hex mesh visualizing the cells of the octree, may contain hanging and duplicate vertices as cells are not connected
    pub fn hexmesh<R: Real>(&self, grid: &UniformGrid<I, R>, only_non_empty: bool) -> HexMesh3d<R> {
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
    /// Returns the [PointIndex] of the lower corner of the octree node
    pub fn min_corner(&self) -> &PointIndex<I> {
        &self.min_corner
    }

    /// Returns the [PointIndex] of the upper corner of the octree node
    pub fn max_corner(&self) -> &PointIndex<I> {
        &self.max_corner
    }

    /// Returns the AABB represented by this octree node
    pub fn aabb<R: Real>(&self, grid: &UniformGrid<I, R>) -> AxisAlignedBoundingBox3d<R> {
        AxisAlignedBoundingBox::new(
            grid.point_coordinates(&self.min_corner),
            grid.point_coordinates(&self.max_corner),
        )
    }

    /// Constructs a [crate::UniformGrid] that represents the domain of this octree node
    pub fn grid<R: Real>(
        &self,
        min: &Vector3<R>,
        cell_size: R,
    ) -> Result<UniformGrid<I, R>, GridConstructionError<I, R>> {
        let min_corner = self.min_corner.index();
        let max_corner = self.max_corner.index();

        let n_cells_per_dim = [
            max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2],
        ];

        UniformGrid::new(min, &n_cells_per_dim, cell_size)
    }

    /// Returns whether this is a leaf node, i.e. it may contain particles and has no child nodes
    pub fn is_leaf(&self) -> bool {
        self.body.is_leaf()
    }

    /// Returns a slice of the particles of this node if it is a leaf node
    pub fn particles(&self) -> Option<&[usize]> {
        self.body.particles()
    }

    /// Returns a slice of the child nodes if this is not a leaf node
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
            body: NodeBody::new_leaf((0..n_particles).collect::<SmallVec<_>>(), 0),
        }
    }

    fn new_leaf(
        min_corner: PointIndex<I>,
        max_corner: PointIndex<I>,
        particles: OctreeNodeParticleStorage,
        ghost_particle_count: usize,
    ) -> Self {
        Self {
            min_corner,
            max_corner,
            body: NodeBody::new_leaf(particles, ghost_particle_count),
        }
    }

    /// Subdivides this [OctantNode] and all new children until all leaves have at most the specified number of particles
    fn subdivide_recursively<R: Real, S: LeafSplitCriterion<I>>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        split_criterion: &S,
    ) {
        // Stop recursion if split criterion is not fulfilled
        if !split_criterion.split_leaf(self) {
            return;
        }

        // Perform one octree split on the leaf
        self.subdivide(grid, particle_positions);

        // TODO: Replace recursion with iteration

        // Continue subdivision recursively in the new child nodes
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                child_node.subdivide_recursively(grid, particle_positions, split_criterion);
            }
        }
    }

    /// Subdivides this [OctantNode] and all new children in parallel until all leaves have at most the specified number of particles
    fn subdivide_recursively_par<'scope, R: Real, S: LeafSplitCriterion<I> + ThreadSafe>(
        &'scope mut self,
        s: &ScopeFifo<'scope>,
        grid: &'scope UniformGrid<I, R>,
        particle_positions: &'scope [Vector3<R>],
        split_criterion: &'scope S,
    ) {
        // Stop recursion if split criterion is not fulfilled
        if !split_criterion.split_leaf(self) {
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
                        split_criterion,
                    )
                });
            }
        }
    }

    /// Subdivides this [OctreeNode] and all new children until all leaves have at most the specified number of particles
    fn subdivide_recursively_margin<R: Real, S: LeafSplitCriterion<I>>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        split_criterion: &S,
        margin: R,
    ) {
        // Stop recursion if split criterion is not fulfilled
        if !split_criterion.split_leaf(self) {
            return;
        }

        // Perform one octree split on the leaf
        self.subdivide_with_margin(grid, particle_positions, margin);
        //self.subdivide_generic_par(grid, particle_positions, MarginClassifier::new(margin));

        // TODO: Replace recursion with iteration

        // Continue subdivision recursively in the new child nodes
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                child_node.subdivide_recursively_margin(
                    grid,
                    particle_positions,
                    split_criterion,
                    margin,
                );
            }
        }
    }

    fn subdivide_recursively_margin_par<'scope, R: Real, S: LeafSplitCriterion<I> + ThreadSafe>(
        &'scope mut self,
        s: &ScopeFifo<'scope>,
        grid: &'scope UniformGrid<I, R>,
        particle_positions: &'scope [Vector3<R>],
        split_criterion: &'scope S,
        margin: R,
    ) {
        // Stop recursion if split criterion is not fulfilled
        if !split_criterion.split_leaf(self) {
            return;
        }

        // Perform one octree split on the leaf
        //self.subdivide_with_margin(grid, particle_positions, margin);
        self.subdivide_with_margin_par(grid, particle_positions, margin);
        //self.subdivide_generic(grid, particle_positions, MarginClassifier::new(margin));
        //self.subdivide_generic_par(grid, particle_positions, MarginClassifier::new(margin));

        // TODO: Replace recursion with iteration

        // Continue subdivision recursively in the new child nodes
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                s.spawn_fifo(move |s| {
                    child_node.subdivide_recursively_margin_par(
                        s,
                        grid,
                        particle_positions,
                        split_criterion,
                        margin,
                    );
                });
            }
        }
    }

    /// Performs a subdivision of this [OctreeNode] by converting its body form a leaf to a body containing its child octants
    fn subdivide<R: Real>(&mut self, grid: &UniformGrid<I, R>, particle_positions: &[Vector3<R>]) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
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
                    0,
                ));

                children.push(child);
            }

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }

    /// Performs a subdivision of this [OctreeNode] in parallel by converting its body form a leaf to a body containing its child octants
    fn subdivide_par<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
    ) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
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
            let mut children = Vec::with_capacity(8);
            Octant::all()
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
                        0,
                    ));

                    child
                })
                .collect_into_vec(&mut children);

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }

    /// Performs a subdivision of this [OctreeNode] while considering a margin with "ghost particles" around each octant
    pub fn subdivide_with_margin<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        margin: R,
    ) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut octant_flags = vec![OctantDirectionFlags::empty(); particles.len()];
            let mut counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
            let mut non_ghost_counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            // Classify all particles of this leaf into its octants
            assert_eq!(particles.len(), octant_flags.len());
            for (particle_idx, particle_octant_flags) in
                particles.iter().copied().zip(octant_flags.iter_mut())
            {
                let relative_pos = particle_positions[particle_idx] - split_coordinates;

                // Check what the main octant of the particle is (to count ghost particles)
                {
                    let main_octant: Octant = OctantAxisDirections::classify(&relative_pos).into();
                    non_ghost_counters[main_octant as usize] += 1;
                }

                // Classify into all octants with margin
                {
                    *particle_octant_flags =
                        OctantDirectionFlags::classify_with_margin(&relative_pos, margin);

                    // Increase the counter of each octant that contains the current particle
                    OctantDirectionFlags::all_unique_octants()
                        .iter()
                        .zip(counters.iter_mut())
                        .filter(|(octant, _)| particle_octant_flags.contains(**octant))
                        .for_each(|(_, counter)| {
                            *counter += 1;
                        });
                }
            }

            // Construct the node for each octant
            let mut children = SmallVec::with_capacity(8);
            for (&current_octant, (&octant_particle_count, &octant_non_ghost_count)) in
                Octant::all()
                    .iter()
                    .zip(counters.iter().zip(non_ghost_counters.iter()))
            {
                let current_octant_dir = OctantAxisDirections::from(current_octant);
                let current_octant_flags = OctantDirectionFlags::from(current_octant);

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
                        .zip(octant_flags.iter())
                        // Skip particles from other octants
                        .filter(|(_, &particle_i_octant)| {
                            particle_i_octant.contains(current_octant_flags)
                        })
                        .map(|(particle_i, _)| particle_i),
                );
                assert_eq!(octant_particles.len(), octant_particle_count);

                let child = Box::new(OctreeNode::new_leaf(
                    min_corner,
                    max_corner,
                    octant_particles,
                    octant_particle_count - octant_non_ghost_count,
                ));

                children.push(child);
            }

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }

    pub fn subdivide_with_margin_par<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        margin: R,
    ) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut octant_flags = vec![OctantDirectionFlags::empty(); particles.len()];

            // Initial values for the fold
            let zeros = || ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);

            // Classify all particles of this leaf into its octants
            assert_eq!(particles.len(), octant_flags.len());
            let (counters, non_ghost_counters) = particles
                .par_iter()
                .copied()
                .zip(octant_flags.par_iter_mut())
                .fold(
                    zeros,
                    |(mut counters, mut non_ghost_counters),
                     (particle_idx, particle_octant_flags)| {
                        let relative_pos = particle_positions[particle_idx] - split_coordinates;

                        // Check what the main octant of the particle is (to count ghost particles)
                        {
                            let main_octant: Octant =
                                OctantAxisDirections::classify(&relative_pos).into();
                            non_ghost_counters[main_octant as usize] += 1;
                        }

                        // Classify into all octants with margin
                        {
                            *particle_octant_flags =
                                OctantDirectionFlags::classify_with_margin(&relative_pos, margin);

                            // Increase the counter of each octant that contains the current particle
                            OctantDirectionFlags::all_unique_octants()
                                .iter()
                                .zip(counters.iter_mut())
                                .filter(|(octant, _)| particle_octant_flags.contains(**octant))
                                .for_each(|(_, counter)| {
                                    *counter += 1;
                                });
                        }

                        (counters, non_ghost_counters)
                    },
                ) // Sum up all counter arrays
                .reduce(
                    zeros,
                    |(mut counters_acc, mut non_ghost_counters_acc),
                     (counters, non_ghost_counters)| {
                        for i in 0..8 {
                            counters_acc[i] += counters[i];
                            non_ghost_counters_acc[i] += non_ghost_counters[i];
                        }
                        (counters_acc, non_ghost_counters_acc)
                    },
                );

            // Construct the octree node for each octant
            let mut children = Vec::with_capacity(8);
            Octant::all()
                .par_iter()
                .zip(counters.par_iter().zip(non_ghost_counters.par_iter()))
                .map(
                    |(&current_octant, (&octant_particle_count, &octant_non_ghost_count))| {
                        let current_octant_dir = OctantAxisDirections::from(current_octant);
                        let current_octant_flags = OctantDirectionFlags::from(current_octant);

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
                                .zip(octant_flags.iter())
                                // Skip particles from other octants
                                .filter(|(_, &particle_i_octant)| {
                                    particle_i_octant.contains(current_octant_flags)
                                })
                                .map(|(particle_i, _)| particle_i),
                        );
                        assert_eq!(octant_particles.len(), octant_particle_count);

                        let child = Box::new(OctreeNode::new_leaf(
                            min_corner,
                            max_corner,
                            octant_particles,
                            octant_particle_count - octant_non_ghost_count,
                        ));

                        child
                    },
                )
                .collect_into_vec(&mut children);

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }

    /*
    fn subdivide_generic<R: Real, Oc: OctantClassifier<R>>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        mut classifier: Oc,
    ) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinate = grid.point_coordinates(&split_point);

            // Classify all particles of this leaf into its octants
            classifier.classify_particles(&split_coordinate, particle_positions, particles);

            // Construct the octree node for each octant
            let mut children = SmallVec::with_capacity(8);
            Octant::all()
                .iter()
                .map(|&current_octant| {
                    let current_octant_dir = OctantAxisDirections::from(current_octant);

                    let min_corner = current_octant_dir
                        .combine_point_index(grid, &self.min_corner, &split_point)
                        .expect("Failed to get corner point of octree subcell");
                    let max_corner = current_octant_dir
                        .combine_point_index(grid, &split_point, &self.max_corner)
                        .expect("Failed to get corner point of octree subcell");

                    let octant_particles = classifier.octant_particles(current_octant, particles);
                    let ghost_particle_count =
                        classifier.octant_ghost_particles_count(current_octant);

                    let child = Box::new(OctreeNode::new_leaf(
                        min_corner,
                        max_corner,
                        octant_particles,
                        ghost_particle_count,
                    ));

                    child
                })
                .for_each(|child| children.push(child));

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }

    fn subdivide_generic_par<R: Real, Oc: OctantClassifier<R> + ThreadSafe>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        mut classifier: Oc,
    ) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles, .. } = &self.body {
            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinate = grid.point_coordinates(&split_point);

            // Classify all particles of this leaf into its octants
            classifier.classify_particles_par(&split_coordinate, particle_positions, particles);

            // Construct the octree node node for each octant
            let mut children = Vec::with_capacity(8);
            Octant::all()
                .par_iter()
                .map(|&current_octant| {
                    let current_octant_dir = OctantAxisDirections::from(current_octant);

                    let min_corner = current_octant_dir
                        .combine_point_index(grid, &self.min_corner, &split_point)
                        .expect("Failed to get corner point of octree subcell");
                    let max_corner = current_octant_dir
                        .combine_point_index(grid, &split_point, &self.max_corner)
                        .expect("Failed to get corner point of octree subcell");

                    let octant_particles = classifier.octant_particles(current_octant, particles);
                    let ghost_particle_count =
                        classifier.octant_ghost_particles_count(current_octant);

                    let child = Box::new(OctreeNode::new_leaf(
                        min_corner,
                        max_corner,
                        octant_particles,
                        ghost_particle_count,
                    ));

                    child
                })
                .collect_into_vec(&mut children);

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }
     */
}

impl<I: Index> NodeBody<I> {
    pub fn new_leaf<IndexVec: Into<OctreeNodeParticleStorage>>(
        particles: IndexVec,
        ghost_particle_count: usize,
    ) -> Self {
        NodeBody::Leaf {
            particles: particles.into(),
            ghost_particle_count,
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
            NodeBody::Leaf { particles, .. } => Some(particles.as_slice()),
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

/// Returns the [PointIndex] of the octree subdivision point for an [OctreeNode] with the given lower and upper points
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

mod octant_helper {
    use bitflags::bitflags;
    use nalgebra::Vector3;

    use crate::uniform_grid::{Direction, PointIndex, UniformGrid};
    use crate::{Index, Real};

    bitflags! {
        pub struct OctantDirectionFlags: u8 {
            const X_NEG = 0b00000001;
            const X_POS = 0b00000010;
            const Y_NEG = 0b00000100;
            const Y_POS = 0b00001000;
            const Z_NEG = 0b00010000;
            const Z_POS = 0b00100000;

            const NEG_NEG_NEG = Self::X_NEG.bits | Self::Y_NEG.bits | Self::Z_NEG.bits;
            const POS_NEG_NEG = Self::X_POS.bits | Self::Y_NEG.bits | Self::Z_NEG.bits;
            const NEG_POS_NEG = Self::X_NEG.bits | Self::Y_POS.bits | Self::Z_NEG.bits;
            const POS_POS_NEG = Self::X_POS.bits | Self::Y_POS.bits | Self::Z_NEG.bits;
            const NEG_NEG_POS = Self::X_NEG.bits | Self::Y_NEG.bits | Self::Z_POS.bits;
            const POS_NEG_POS = Self::X_POS.bits | Self::Y_NEG.bits | Self::Z_POS.bits;
            const NEG_POS_POS = Self::X_NEG.bits | Self::Y_POS.bits | Self::Z_POS.bits;
            const POS_POS_POS = Self::X_POS.bits | Self::Y_POS.bits | Self::Z_POS.bits;
        }
    }

    const ALL_UNIQUE_OCTANT_DIRECTION_FLAGS: [OctantDirectionFlags; 8] = [
        OctantDirectionFlags::NEG_NEG_NEG,
        OctantDirectionFlags::POS_NEG_NEG,
        OctantDirectionFlags::NEG_POS_NEG,
        OctantDirectionFlags::POS_POS_NEG,
        OctantDirectionFlags::NEG_NEG_POS,
        OctantDirectionFlags::POS_NEG_POS,
        OctantDirectionFlags::NEG_POS_POS,
        OctantDirectionFlags::POS_POS_POS,
    ];

    impl OctantDirectionFlags {
        #[inline(always)]
        pub const fn all_unique_octants() -> &'static [OctantDirectionFlags] {
            &ALL_UNIQUE_OCTANT_DIRECTION_FLAGS
        }

        /// Classifies a point relative to zero into all octants it belongs by considering a margin around the octants
        #[inline(always)]
        pub fn classify_with_margin<R: Real>(point: &Vector3<R>, margin: R) -> Self {
            let mut flags = OctantDirectionFlags::empty();
            flags.set(OctantDirectionFlags::X_NEG, point.x < margin);
            flags.set(OctantDirectionFlags::X_POS, point.x > -margin);
            flags.set(OctantDirectionFlags::Y_NEG, point.y < margin);
            flags.set(OctantDirectionFlags::Y_POS, point.y > -margin);
            flags.set(OctantDirectionFlags::Z_NEG, point.z < margin);
            flags.set(OctantDirectionFlags::Z_POS, point.z > -margin);
            flags
        }

        #[inline(always)]
        pub fn from_octant(octant: Octant) -> Self {
            match octant {
                Octant::NegNegNeg => Self::NEG_NEG_NEG,
                Octant::PosNegNeg => Self::POS_NEG_NEG,
                Octant::NegPosNeg => Self::NEG_POS_NEG,
                Octant::PosPosNeg => Self::POS_POS_NEG,
                Octant::NegNegPos => Self::NEG_NEG_POS,
                Octant::PosNegPos => Self::POS_NEG_POS,
                Octant::NegPosPos => Self::NEG_POS_POS,
                Octant::PosPosPos => Self::POS_POS_POS,
            }
        }

        #[inline(always)]
        pub fn from_directions(directions: OctantAxisDirections) -> Self {
            let mut flags = OctantDirectionFlags::empty();
            flags.set(OctantDirectionFlags::X_NEG, directions.x_axis.is_negative());
            flags.set(OctantDirectionFlags::X_POS, directions.x_axis.is_positive());
            flags.set(OctantDirectionFlags::Y_NEG, directions.y_axis.is_negative());
            flags.set(OctantDirectionFlags::Y_POS, directions.y_axis.is_positive());
            flags.set(OctantDirectionFlags::Z_NEG, directions.z_axis.is_negative());
            flags.set(OctantDirectionFlags::Z_POS, directions.z_axis.is_positive());
            flags
        }
    }

    impl From<Octant> for OctantDirectionFlags {
        fn from(octant: Octant) -> Self {
            Self::from_octant(octant)
        }
    }

    impl From<OctantAxisDirections> for OctantDirectionFlags {
        fn from(directions: OctantAxisDirections) -> Self {
            Self::from_directions(directions)
        }
    }

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub struct OctantAxisDirections {
        x_axis: Direction,
        y_axis: Direction,
        z_axis: Direction,
    }

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    #[repr(u8)]
    pub enum Octant {
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
        #[allow(dead_code)]
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
}
