//! Octree for spatially partitioning particle sets

use crate::generic_tree::*;
use crate::marching_cubes::SurfacePatch;
use crate::mesh::{HexMesh3d, MeshAttribute, MeshWithData, TriMesh3d};
use crate::topology::{Axis, Direction};
use crate::uniform_grid::{PointIndex, UniformGrid};
use crate::utils::{ChunkSize, ParallelPolicy};
use crate::{
    marching_cubes, new_map, profile, AxisAlignedBoundingBox3d, GridConstructionError, Index,
    MapType, Real,
};
use arrayvec::ArrayVec;
use log::info;
use nalgebra::Vector3;
use octant_helper::{HalfspaceFlags, Octant, OctantAxisDirections};
use rayon::prelude::*;
use smallvec::SmallVec;
use split_criterion::{default_split_criterion, LeafSplitCriterion};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use thread_local::ThreadLocal;

// TODO: Make margin an Option

/// Criterion used for the subdivision of the spatial decomposition of the particle collection
#[derive(Clone, Debug)]
pub enum SubdivisionCriterion {
    /// Perform octree subdivision until an upper limit of particles is reached per chunk, automatically chosen based on number of threads
    MaxParticleCountAuto,
    /// Perform octree subdivision until an upper limit of particles is reached per chunk, based on the given fixed number of particles
    MaxParticleCount(usize),
}

/// Data structure for octree based spatial subdivision of particles sets, for tree iteration/visitation use the [`root`](Self::root) [`OctreeNode`]
#[derive(Clone, Debug)]
pub struct Octree<I: Index, R: Real> {
    /// Root node of the tree
    root: OctreeNode<I, R>,
    /// Counter for assigning ids to subdivided nodes
    next_id: usize,
}

/// Represents a node in the octree hierarchy and stores child nodes, implements tree iteration/visitation from the [`generic_tree`](crate::generic_tree) module
#[derive(Clone, Debug)]
pub struct OctreeNode<I: Index, R: Real> {
    /// Id of the node used to identify it for debugging
    id: usize,
    /// All child nodes of this octree node
    children: ArrayVec<[Box<Self>; 8]>,
    /// Lower corner point of the octree node on the background grid
    min_corner: PointIndex<I>,
    /// Upper corner point of the octree node on the background grid
    max_corner: PointIndex<I>,
    /// AABB of the octree node
    aabb: AxisAlignedBoundingBox3d<R>,
    /// Additional data associated to this octree node
    data: NodeData<I, R>,
}

impl<I: Index, R: Real> TreeNode for OctreeNode<I, R> {
    /// Returns a slice of all child nodes
    fn children(&self) -> &[Box<Self>] {
        self.children.as_slice()
    }
}

impl<I: Index, R: Real> TreeNodeMut for OctreeNode<I, R> {
    /// Returns a mutable slice of all child nodes
    fn children_mut(&mut self) -> &mut [Box<Self>] {
        self.children.as_mut_slice()
    }
}

/// Optional data that may be stored in [`OctreeNode`]s
#[derive(Clone, Debug)]
pub enum NodeData<I: Index, R: Real> {
    /// Empty variant
    None,
    /// Storage for a set of SPH particles
    ParticleSet(ParticleSet),
    /// A patch that was already meshed
    SurfacePatch(SurfacePatchWrapper<I, R>),
}

impl<I: Index, R: Real> Default for NodeData<I, R> {
    /// Returns an empty data instance
    fn default() -> Self {
        Self::None
    }
}

/// Stores the particle ids and the number of ghost particles inside an octree leaf
#[derive(Clone, Debug)]
pub struct ParticleSet {
    // The particles belonging to this set
    pub particles: OctreeNodeParticleStorage,
    // Number of ghost particles in this particle set
    pub ghost_particle_count: usize,
}

/// Wrapper for an internal `SurfacePatch` to avoid leaking too much implementation details
#[derive(Clone, Debug)]
pub struct SurfacePatchWrapper<I: Index, R: Real> {
    pub(crate) patch: SurfacePatch<I, R>,
}

impl<I: Index, R: Real> From<SurfacePatch<I, R>> for SurfacePatchWrapper<I, R> {
    fn from(patch: SurfacePatch<I, R>) -> Self {
        Self { patch }
    }
}

impl<I: Index, R: Real> SurfacePatchWrapper<I, R> {
    pub fn mesh(&self) -> &TriMesh3d<R> {
        &self.patch.mesh
    }
}

type OctreeNodeParticleStorage = SmallVec<[usize; 6]>;

impl<I: Index, R: Real> Octree<I, R> {
    /// Creates a new octree with a single leaf node containing all vertices
    pub fn new(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        Self {
            root: OctreeNode::new_root(grid, n_particles),
            next_id: 0,
        }
    }

    /// Create a new octree and perform subdivision with the specified margin
    ///
    /// The margin is used to assign ghost particles to octree nodes. Each octant resulting
    /// from the subdivision gets assigned all particles that are directly inside it plus all
    /// particles from its parent that are within the given margin around the octant.
    pub fn new_subdivided(
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        subdivision_criterion: SubdivisionCriterion,
        margin: R,
        enable_multi_threading: bool,
        enable_stitching: bool,
    ) -> Self {
        let mut tree = Octree::new(&grid, particle_positions.len());

        if enable_multi_threading {
            tree.par_subdivide_recursively_margin(
                grid,
                particle_positions,
                subdivision_criterion,
                margin,
                enable_stitching,
            );
        } else {
            tree.subdivide_recursively_margin(
                grid,
                particle_positions,
                subdivision_criterion,
                margin,
                enable_stitching,
            );
        }

        tree
    }

    /// Returns a reference to the root node of the octree
    pub fn root(&self) -> &OctreeNode<I, R> {
        &self.root
    }

    /// Returns a mutable reference to the root node of the octree
    pub fn root_mut(&mut self) -> &mut OctreeNode<I, R> {
        &mut self.root
    }

    /// Subdivide the octree recursively using the given splitting criterion and a margin to add ghost particles
    pub fn subdivide_recursively_margin(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        subdivision_criterion: SubdivisionCriterion,
        margin: R,
        enable_stitching: bool,
    ) {
        profile!("octree subdivide_recursively_margin");

        let split_criterion = default_split_criterion(
            subdivision_criterion,
            particle_positions.len(),
            enable_stitching,
        );

        let next_id = AtomicUsize::new(0);
        self.root.visit_mut_bfs(|node| {
            // Stop recursion if split criterion is not fulfilled
            if !split_criterion.split_leaf(node) {
                return;
            }

            // Perform one octree split on the node
            node.subdivide_with_margin(grid, particle_positions, margin, &next_id);
        });
        self.next_id = next_id.into_inner();
    }

    /// Subdivide the octree recursively and in parallel using the given splitting criterion and a margin to add ghost particles
    pub fn par_subdivide_recursively_margin(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        subdivision_criterion: SubdivisionCriterion,
        margin: R,
        enable_stitching: bool,
    ) {
        profile!("octree subdivide_recursively_margin_par");

        let split_criterion = default_split_criterion(
            subdivision_criterion,
            particle_positions.len(),
            enable_stitching,
        );
        let parallel_policy = ParallelPolicy::default();

        let next_id = AtomicUsize::new(0);
        let visitor = {
            let next_id = &next_id;
            move |node: &mut OctreeNode<I, R>| {
                // Stop recursion if split criterion is not fulfilled
                if !split_criterion.split_leaf(node) {
                    return;
                }

                // Perform one octree split on the leaf
                if node
                    .data
                    .particle_set()
                    .expect("Node is not a leaf")
                    .particles
                    .len()
                    < parallel_policy.min_task_size
                {
                    node.subdivide_with_margin(grid, particle_positions, margin, &next_id);
                } else {
                    node.par_subdivide_with_margin(
                        grid,
                        particle_positions,
                        margin,
                        &parallel_policy,
                        &next_id,
                    );
                }
            }
        };

        self.root.par_visit_mut_bfs(visitor);
        self.next_id = next_id.into_inner();
    }

    /// Constructs a hex mesh visualizing the cells of the octree, may contain hanging and duplicate vertices as cells are not connected
    pub fn hexmesh(
        &self,
        grid: &UniformGrid<I, R>,
        only_non_empty: bool,
    ) -> MeshWithData<R, HexMesh3d<R>> {
        profile!("convert octree into hexmesh");

        let mut mesh = HexMesh3d {
            vertices: Vec::new(),
            cells: Vec::new(),
        };

        let mut ids = Vec::new();
        self.root.dfs_iter().for_each(|node| {
            if node.children().is_empty() {
                if only_non_empty
                    && node
                        .data()
                        .particle_set()
                        .map(|ps| ps.particles.is_empty())
                        .unwrap_or(true)
                {
                    return;
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
                ids.push(node.id as u64);
            }
        });

        assert_eq!(mesh.cells.len(), ids.len());
        MeshWithData::new(mesh).with_cell_data(MeshAttribute::new("node_id", ids))
    }
}

impl<I: Index, R: Real> OctreeNode<I, R> {
    pub fn new(
        id: usize,
        min_corner: PointIndex<I>,
        max_corner: PointIndex<I>,
        aabb: AxisAlignedBoundingBox3d<R>,
    ) -> Self {
        Self::with_data(id, min_corner, max_corner, aabb, NodeData::None)
    }

    fn new_root(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        let n_points = grid.points_per_dim();
        let min_point = [I::zero(), I::zero(), I::zero()];
        let max_point = [
            n_points[0] - I::one(),
            n_points[1] - I::one(),
            n_points[2] - I::one(),
        ];

        Self::with_data(
            0,
            grid.get_point(min_point)
                .expect("Cannot get lower corner of grid"),
            grid.get_point(max_point)
                .expect("Cannot get upper corner of grid"),
            grid.aabb().clone(),
            NodeData::new_particle_set((0..n_particles).collect::<SmallVec<_>>(), 0),
        )
    }

    fn with_data(
        id: usize,
        min_corner: PointIndex<I>,
        max_corner: PointIndex<I>,
        aabb: AxisAlignedBoundingBox3d<R>,
        data: NodeData<I, R>,
    ) -> Self {
        Self {
            id,
            children: Default::default(),
            min_corner,
            max_corner,
            aabb,
            data,
        }
    }

    /// Returns the id of the node
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns a reference to the data stored in the node
    pub fn data(&self) -> &NodeData<I, R> {
        &self.data
    }

    /// Returns a mutable reference to the data stored in the node
    pub(crate) fn data_mut(&mut self) -> &mut NodeData<I, R> {
        &mut self.data
    }

    /// Returns the [`PointIndex`] of the lower corner of the octree node
    pub fn min_corner(&self) -> &PointIndex<I> {
        &self.min_corner
    }

    /// Returns the [`PointIndex`] of the upper corner of the octree node
    pub fn max_corner(&self) -> &PointIndex<I> {
        &self.max_corner
    }

    /// Returns the AABB represented by this octree node
    pub fn aabb(&self) -> &AxisAlignedBoundingBox3d<R> {
        &self.aabb
    }

    /// Constructs a [`UniformGrid`](crate::UniformGrid) that represents the domain of this octree node
    pub fn grid(
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

    /// Performs a subdivision of this node while considering a margin for "ghost particles" around each octant
    pub fn subdivide_with_margin(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        margin: R,
        next_id: &AtomicUsize,
    ) {
        // Convert node body from Leaf to Children
        if let NodeData::ParticleSet(particle_set) = self.data.take() {
            let particles = particle_set.particles;

            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut halfspace_flags = vec![HalfspaceFlags::empty(); particles.len()];
            let mut counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
            let mut non_ghost_counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            // Classify all particles of this leaf into the halfspaces relative to the split point
            assert_eq!(particles.len(), halfspace_flags.len());
            for (particle_idx, particle_halfspace_flags) in
                particles.iter().copied().zip(halfspace_flags.iter_mut())
            {
                let pos = particle_positions[particle_idx];
                let relative_pos = pos - split_coordinates;

                let is_ghost_particle = !self.aabb.contains_point(&pos);

                // Check what the main octant (without margin) of the particle is to count ghost particles later
                if !is_ghost_particle {
                    let main_octant: Octant = OctantAxisDirections::classify(&relative_pos).into();
                    non_ghost_counters[main_octant as usize] += 1;
                }

                // Classify into all halfspaces with margin
                {
                    *particle_halfspace_flags =
                        HalfspaceFlags::classify_with_margin(&relative_pos, margin);

                    // Increase the counter of each octant that contains the current particle
                    HalfspaceFlags::all_unique_octants()
                        .iter()
                        .zip(counters.iter_mut())
                        .filter(|(octant, _)| particle_halfspace_flags.contains(**octant))
                        .for_each(|(_, counter)| {
                            *counter += 1;
                        });
                }
            }

            // Construct the node for each octant
            let mut children = ArrayVec::new();
            for (&current_octant, (&octant_particle_count, &octant_non_ghost_count)) in
                Octant::all()
                    .iter()
                    .zip(counters.iter().zip(non_ghost_counters.iter()))
            {
                let current_octant_dir = OctantAxisDirections::from(current_octant);
                let current_octant_flags = HalfspaceFlags::from(current_octant);

                let min_corner = current_octant_dir
                    .combine_point_index(grid, &self.min_corner, &split_point)
                    .expect("Failed to get corner point of octree subcell");
                let max_corner = current_octant_dir
                    .combine_point_index(grid, &split_point, &self.max_corner)
                    .expect("Failed to get corner point of octree subcell");

                let child_aabb = AxisAlignedBoundingBox3d::new(
                    grid.point_coordinates(&min_corner),
                    grid.point_coordinates(&max_corner),
                );

                let mut octant_particles = SmallVec::with_capacity(octant_particle_count);
                octant_particles.extend(
                    particles
                        .iter()
                        .copied()
                        .zip(halfspace_flags.iter())
                        // Skip particles from other octants
                        .filter(|(_, &particle_i_halfspaces)| {
                            particle_i_halfspaces.contains(current_octant_flags)
                        })
                        .map(|(particle_i, _)| particle_i),
                );
                assert_eq!(octant_particles.len(), octant_particle_count);

                let child = Box::new(OctreeNode::with_data(
                    next_id.fetch_add(1, Ordering::SeqCst),
                    min_corner,
                    max_corner,
                    child_aabb,
                    NodeData::new_particle_set(
                        octant_particles,
                        octant_particle_count - octant_non_ghost_count,
                    ),
                ));

                children.push(child);
            }

            // Assign new children to the current node
            self.children = children;
        } else {
            panic!("Only nodes with ParticleSet data can be subdivided");
        };
    }

    /// Parallel subdivision of this node while considering a margin for "ghost particles" around each octant
    pub fn par_subdivide_with_margin(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        margin: R,
        parallel_policy: &ParallelPolicy,
        next_id: &AtomicUsize,
    ) {
        // Convert node body from Leaf to Children
        if let NodeData::ParticleSet(particle_set) = self.data.take() {
            let particles = particle_set.particles;

            // Obtain the point used as the octree split/pivot point
            let split_point = get_split_point(grid, &self.min_corner, &self.max_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            let mut octant_flags = vec![HalfspaceFlags::empty(); particles.len()];

            // Initial values for the thread local counters
            let zeros = || ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);
            let zeros_cell = || RefCell::new(zeros());

            let tl_counters = ThreadLocal::new();
            let chunk_size = ChunkSize::new(parallel_policy, particles.len()).chunk_size;

            // Classify all particles of this leaf into its octants
            assert_eq!(particles.len(), octant_flags.len());
            particles
                .par_chunks(chunk_size)
                .zip(octant_flags.par_chunks_mut(chunk_size))
                .for_each(|(idx_chunk, flags_chunk)| {
                    // Obtain references to the thread-local counters
                    let mut counters_borrow_mut = tl_counters.get_or(zeros_cell).borrow_mut();
                    let counters_ref_mut = &mut *counters_borrow_mut;
                    let (counters, non_ghost_counters) =
                        (&mut counters_ref_mut.0, &mut counters_ref_mut.1);

                    idx_chunk
                        .iter()
                        .copied()
                        .zip(flags_chunk.iter_mut())
                        .for_each(|(particle_idx, particle_octant_flags)| {
                            let pos = particle_positions[particle_idx];
                            let relative_pos = particle_positions[particle_idx] - split_coordinates;

                            let is_ghost_particle = !self.aabb.contains_point(&pos);

                            // Check what the main octant of the particle is (to count ghost particles)
                            if !is_ghost_particle {
                                let main_octant: Octant =
                                    OctantAxisDirections::classify(&relative_pos).into();
                                non_ghost_counters[main_octant as usize] += 1;
                            }

                            // Classify into all octants with margin
                            {
                                *particle_octant_flags =
                                    HalfspaceFlags::classify_with_margin(&relative_pos, margin);

                                // Increase the counter of each octant that contains the current particle
                                HalfspaceFlags::all_unique_octants()
                                    .iter()
                                    .zip(counters.iter_mut())
                                    .filter(|(octant, _)| particle_octant_flags.contains(**octant))
                                    .for_each(|(_, counter)| {
                                        *counter += 1;
                                    });
                            }
                        })
                });

            // Sum up all thread local counter arrays
            let (counters, non_ghost_counters) = tl_counters.into_iter().fold(
                zeros(),
                |(mut counters_acc, mut non_ghost_counters_acc), counter_cell| {
                    let (counters, non_ghost_counters) = counter_cell.into_inner();
                    for i in 0..8 {
                        counters_acc[i] += counters[i];
                        non_ghost_counters_acc[i] += non_ghost_counters[i];
                    }
                    (counters_acc, non_ghost_counters_acc)
                },
            );

            // TODO: Would be nice to collect directly into a ArrayVec but that doesn't seem to be possible
            //  (at least without some unsafe magic with uninit)
            let mut children = Vec::with_capacity(8);
            // Construct the octree node for each octant
            Octant::all()
                .par_iter()
                .zip(counters.par_iter().zip(non_ghost_counters.par_iter()))
                .map(
                    |(&current_octant, (&octant_particle_count, &octant_non_ghost_count))| {
                        let current_octant_dir = OctantAxisDirections::from(current_octant);
                        let current_octant_flags = HalfspaceFlags::from(current_octant);

                        let min_corner = current_octant_dir
                            .combine_point_index(grid, &self.min_corner, &split_point)
                            .expect("Failed to get corner point of octree subcell");
                        let max_corner = current_octant_dir
                            .combine_point_index(grid, &split_point, &self.max_corner)
                            .expect("Failed to get corner point of octree subcell");

                        let child_aabb = AxisAlignedBoundingBox3d::new(
                            grid.point_coordinates(&min_corner),
                            grid.point_coordinates(&max_corner),
                        );

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

                        let child = Box::new(OctreeNode::with_data(
                            next_id.fetch_add(1, Ordering::SeqCst),
                            min_corner,
                            max_corner,
                            child_aabb,
                            NodeData::new_particle_set(
                                octant_particles,
                                octant_particle_count - octant_non_ghost_count,
                            ),
                        ));

                        child
                    },
                )
                .collect_into_vec(&mut children);

            // Assign children to this node
            self.children = children.into_iter().collect::<ArrayVec<_>>();
        } else {
            panic!("Only nodes with ParticleSet data can be subdivided");
        };
    }

    fn stitch_children_orthogonal_to(
        &mut self,
        children_map: &mut MapType<OctantAxisDirections, SurfacePatch<I, R>>,
        stitching_axis: Axis,
        iso_surface_threshold: R,
    ) {
        profile!("stitch_children_orthogonal_to");

        for mut octant in OctantAxisDirections::all().iter().copied() {
            // Iterate over every octant pair only once
            if octant.direction(stitching_axis).is_positive() {
                continue;
            }

            // First try to get negative side, it might not exist because children were already merged before to another octant in the map
            let negative_side = if let Some(negative_patch) = children_map.remove(&octant) {
                negative_patch
            } else {
                continue;
            };

            // If the negative side on the stitching axis exists, the positive side must also exist
            octant.set_direction(stitching_axis, Direction::Positive);
            let positive_side = children_map.remove(&octant).expect("Child node missing!");

            let stitched_patch = marching_cubes::stitch_surface_patches(
                iso_surface_threshold,
                stitching_axis,
                negative_side,
                positive_side,
            );

            // Add stitched surface back to map, setting the direction of the octant of the stitched patch to positive
            children_map.insert(octant, stitched_patch);
        }
    }

    /// Stitches together the [`SurfacePatch`]es stored in the children of this node if this is the direct parent of only leaf nodes
    pub(crate) fn stitch_surface_patches(&mut self, iso_surface_threshold: R) {
        profile!("stitch_surface_patches");

        // If this node has no children there is nothing to stitch
        if self.children.is_empty() {
            panic!("A node can only be stitched if it has children!");
        }

        // Don't try to stitch if there are children that still have children
        for child in self.children.iter() {
            if !child.children.is_empty() {
                panic!("A node can only be stitched if all children are leaf nodes!");
            }
        }

        let mut children_map: MapType<_, SurfacePatch<I, R>> = {
            let old_children = std::mem::take(&mut self.children);

            let mut children_map = new_map();
            for (child, octant) in old_children.into_iter().zip(Octant::all().iter().copied()) {
                let octant_directions = OctantAxisDirections::from(octant);
                children_map.insert(
                    octant_directions,
                    child
                        .data
                        .into_surface_patch()
                        .expect("Surface patch missing!")
                        .patch,
                );
            }

            children_map
        };

        self.stitch_children_orthogonal_to(&mut children_map, Axis::X, iso_surface_threshold);
        self.stitch_children_orthogonal_to(&mut children_map, Axis::Y, iso_surface_threshold);
        self.stitch_children_orthogonal_to(&mut children_map, Axis::Z, iso_surface_threshold);

        assert_eq!(
            children_map.len(),
            1,
            "After stitching, there should be only one child left."
        );

        for (_, mut stitched_patch) in children_map.into_iter() {
            stitched_patch.stitching_level += 1;
            self.data = NodeData::SurfacePatch(stitched_patch.into());
            break;
        }

        assert!(
            self.children.is_empty(),
            "After stitching, the node should not have any children."
        );
    }
}

impl<I: Index, R: Real> NodeData<I, R> {
    fn new_particle_set<P: Into<OctreeNodeParticleStorage>>(
        particles: P,
        ghost_particle_count: usize,
    ) -> Self {
        let particles = particles.into();
        NodeData::ParticleSet(ParticleSet {
            particles,
            ghost_particle_count,
        })
    }

    /// Returns a reference to the contained particle set if it contains one
    pub fn particle_set(&self) -> Option<&ParticleSet> {
        if let Self::ParticleSet(particle_set) = self {
            Some(particle_set)
        } else {
            None
        }
    }

    /// Returns a reference to the contained surface patch if it contains one
    pub fn surface_patch(&self) -> Option<&SurfacePatchWrapper<I, R>> {
        if let Self::SurfacePatch(surface_patch) = self {
            Some(surface_patch)
        } else {
            None
        }
    }

    /// Consumes self and returns the ParticleSet if it contained one
    pub fn into_particle_set(self) -> Option<ParticleSet> {
        if let Self::ParticleSet(particle_set) = self {
            Some(particle_set)
        } else {
            None
        }
    }

    /// Consumes self and returns the SurfacePatch if it contained one
    pub fn into_surface_patch(self) -> Option<SurfacePatchWrapper<I, R>> {
        if let Self::SurfacePatch(surface_patch) = self {
            Some(surface_patch)
        } else {
            None
        }
    }

    /// Returns the stored data and leaves `None` in its place
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Returns the stored data and leaves `None` in its place
    pub fn replace(&mut self, new_data: Self) {
        *self = new_data;
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

    grid.get_point(mid_indices)
}

mod split_criterion {
    use super::*;

    /// Trait that is used by an octree to decide whether an octree node should be further split or subdivided
    pub(super) trait LeafSplitCriterion<I: Index, R: Real> {
        /// Returns whether the specified node should be split
        fn split_leaf(&self, node: &OctreeNode<I, R>) -> bool;
    }

    /// Split criterion that decides based on whether the number of non-ghost particles in a node is above a limit
    pub(super) struct MaxNonGhostParticleLeafSplitCriterion {
        max_particles: usize,
    }

    impl MaxNonGhostParticleLeafSplitCriterion {
        fn new(max_particles: usize) -> Self {
            Self { max_particles }
        }
    }

    impl<I: Index, R: Real> LeafSplitCriterion<I, R> for MaxNonGhostParticleLeafSplitCriterion {
        /// Returns true if the number of non-ghost particles in a node is above a limit
        fn split_leaf(&self, node: &OctreeNode<I, R>) -> bool {
            match &node.data {
                NodeData::ParticleSet(particle_set) => {
                    // Check if this leaf is already below the limit of particles per cell
                    return particle_set.particles.len() - particle_set.ghost_particle_count
                        > self.max_particles;
                }
                // Early out if called on a non-leaf node
                _ => return false,
            };
        }
    }

    /// Split criterion that decides based on whether the node's extents are larger than 1 cell in all dimensions
    pub(super) struct MinimumExtentSplitCriterion<I> {
        minimum_extent: I,
    }

    impl<I: Index> MinimumExtentSplitCriterion<I> {
        fn new(minimum_extent: I) -> Self {
            Self { minimum_extent }
        }
    }

    impl<I: Index, R: Real> LeafSplitCriterion<I, R> for MinimumExtentSplitCriterion<I> {
        /// Only returns true if a splitting of the node does not result in a node that is smaller than the allowed minimum extent
        fn split_leaf(&self, node: &OctreeNode<I, R>) -> bool {
            let lower = node.min_corner.index();
            let upper = node.max_corner.index();

            upper[0] - lower[0] >= self.minimum_extent + self.minimum_extent
                && upper[1] - lower[1] >= self.minimum_extent + self.minimum_extent
                && upper[2] - lower[2] >= self.minimum_extent + self.minimum_extent
        }
    }

    impl<I: Index, R: Real, A, B> LeafSplitCriterion<I, R> for (A, B)
    where
        A: LeafSplitCriterion<I, R>,
        B: LeafSplitCriterion<I, R>,
    {
        fn split_leaf(&self, node: &OctreeNode<I, R>) -> bool {
            self.0.split_leaf(node) && self.1.split_leaf(node)
        }
    }

    pub(super) fn default_split_criterion<I: Index>(
        subdivision_criterion: SubdivisionCriterion,
        num_particles: usize,
        enable_stitching: bool,
    ) -> (
        MaxNonGhostParticleLeafSplitCriterion,
        MinimumExtentSplitCriterion<I>,
    ) {
        let particles_per_cell = match subdivision_criterion {
            SubdivisionCriterion::MaxParticleCount(count) => count,
            SubdivisionCriterion::MaxParticleCountAuto => {
                ChunkSize::new(&ParallelPolicy::default(), num_particles)
                    .with_log("particles", "octree generation")
                    .chunk_size
            }
        };

        info!(
            "Building octree with at most {} particles per leaf",
            particles_per_cell
        );

        (
            MaxNonGhostParticleLeafSplitCriterion::new(particles_per_cell),
            MinimumExtentSplitCriterion::new(if enable_stitching {
                I::one() + I::one() + I::one()
            } else {
                I::one()
            }),
        )
    }
}

mod octant_helper {
    use bitflags::bitflags;
    use nalgebra::Vector3;

    use crate::topology::{Axis, Direction};
    use crate::uniform_grid::{PointIndex, UniformGrid};
    use crate::{Index, Real};

    /// All octants of a 3D cartesian coordinate system
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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

    /// Representation of a cartesian coordinate system octant using a direction along each coordinate axis
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    pub struct OctantAxisDirections {
        pub x_axis: Direction,
        pub y_axis: Direction,
        pub z_axis: Direction,
    }

    bitflags! {
        pub struct HalfspaceFlags: u8 {
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

    impl HalfspaceFlags {
        #[inline(always)]
        pub const fn all_unique_octants() -> &'static [HalfspaceFlags] {
            &ALL_UNIQUE_OCTANT_DIRECTION_FLAGS
        }

        /// Classifies a point relative to zero into all halfspaces it belongs including a margin around the halfspace boundary
        #[inline(always)]
        pub fn classify_with_margin<R: Real>(point: &Vector3<R>, margin: R) -> Self {
            let mut flags = HalfspaceFlags::empty();
            flags.set(HalfspaceFlags::X_NEG, point.x < margin);
            flags.set(HalfspaceFlags::X_POS, point.x > -margin);
            flags.set(HalfspaceFlags::Y_NEG, point.y < margin);
            flags.set(HalfspaceFlags::Y_POS, point.y > -margin);
            flags.set(HalfspaceFlags::Z_NEG, point.z < margin);
            flags.set(HalfspaceFlags::Z_POS, point.z > -margin);
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
            let mut flags = HalfspaceFlags::empty();
            flags.set(HalfspaceFlags::X_NEG, directions.x_axis.is_negative());
            flags.set(HalfspaceFlags::X_POS, directions.x_axis.is_positive());
            flags.set(HalfspaceFlags::Y_NEG, directions.y_axis.is_negative());
            flags.set(HalfspaceFlags::Y_POS, directions.y_axis.is_positive());
            flags.set(HalfspaceFlags::Z_NEG, directions.z_axis.is_negative());
            flags.set(HalfspaceFlags::Z_POS, directions.z_axis.is_positive());
            flags
        }
    }

    impl From<Octant> for HalfspaceFlags {
        fn from(octant: Octant) -> Self {
            Self::from_octant(octant)
        }
    }

    impl From<OctantAxisDirections> for HalfspaceFlags {
        fn from(directions: OctantAxisDirections) -> Self {
            Self::from_directions(directions)
        }
    }

    const ALL_UNIQUE_OCTANT_DIRECTION_FLAGS: [HalfspaceFlags; 8] = [
        HalfspaceFlags::NEG_NEG_NEG,
        HalfspaceFlags::POS_NEG_NEG,
        HalfspaceFlags::NEG_POS_NEG,
        HalfspaceFlags::POS_POS_NEG,
        HalfspaceFlags::NEG_NEG_POS,
        HalfspaceFlags::POS_NEG_POS,
        HalfspaceFlags::NEG_POS_POS,
        HalfspaceFlags::POS_POS_POS,
    ];

    impl OctantAxisDirections {
        #[allow(dead_code)]
        #[inline(always)]
        pub const fn all() -> &'static [OctantAxisDirections; 8] {
            &ALL_OCTANT_DIRECTIONS
        }

        #[inline(always)]
        pub const fn from_bool(x_positive: bool, y_positive: bool, z_positive: bool) -> Self {
            Self {
                x_axis: Direction::new_positive(x_positive),
                y_axis: Direction::new_positive(y_positive),
                z_axis: Direction::new_positive(z_positive),
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

        pub fn direction(&self, axis: Axis) -> Direction {
            match axis {
                Axis::X => self.x_axis,
                Axis::Y => self.y_axis,
                Axis::Z => self.z_axis,
            }
        }

        pub fn set_direction(&mut self, axis: Axis, direction: Direction) {
            match axis {
                Axis::X => self.x_axis = direction,
                Axis::Y => self.y_axis = direction,
                Axis::Z => self.z_axis = direction,
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

            grid.get_point(combined_index)
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
