//!
//! Library for surface reconstruction using marching cubes for SPH particle data. Entry point is the [reconstruct_surface] function.
//!

/// Re-export the version of coarse_prof used by this crate, if profiling is enabled
#[cfg(feature = "profiling")]
pub use coarse_prof;
/// Re-export the version of nalgebra used by this crate
pub use nalgebra;
/// Re-export the version of vtkio used by this crate, if vtk support is enabled
#[cfg(feature = "vtk_extras")]
pub use vtkio;

#[cfg(feature = "profiling")]
/// Invokes coarse_prof::profile! with the given expression
macro_rules! profile {
    ($body:expr) => {
        coarse_prof::profile!($body);
    };
}

#[cfg(not(feature = "profiling"))]
/// No-op macro if profiling is disabled
macro_rules! profile {
    ($body:expr) => {
        $body
    };
}

mod aabb;
/// Computation of sparse density maps (evaluation of particle densities and mapping onto sparse grids)
pub mod density_map;
/// Generic octree implementation for basic algorithms
pub mod generic_tree;
/// SPH kernel function implementations
pub mod kernel;
/// Triangulation of density maps using marching cubes
pub mod marching_cubes;
mod marching_cubes_lut;
/// Basic mesh types used by the library and implementation of VTK export
pub mod mesh;
/// Simple neighborhood search based on spatial hashing
pub mod neighborhood_search;
mod numeric_types;
/// Octree implementation to spatially partition particle sets
pub mod octree;
/// Helper types for cartesian coordinate system topology
pub mod topology;
/// Types related to the virtual background grid used for marching cubes
pub mod uniform_grid;
mod utils;
/// Workspace for reusing allocated memory between multiple reconstructions
pub(crate) mod workspace;

use log::info;
use nalgebra::Vector3;
use thiserror::Error as ThisError;

use mesh::TriMesh3d;
use octree::Octree;
use uniform_grid::PointIndex;
use workspace::{LocalReconstructionWorkspace, ReconstructionWorkspace};

use crate::generic_tree::{ParVisitableTree, VisitableTree};
use crate::marching_cubes::SurfacePatch;
use crate::octree::{NodeData, OctreeNode};
pub use aabb::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d};
pub use density_map::DensityMap;
pub use numeric_types::{Index, Real, ThreadSafe};
pub use octree::SubdivisionCriterion;
pub use uniform_grid::{GridConstructionError, UniformGrid};

// TODO: Add documentation of feature flags
// TODO: Feature flag for multi threading

// TODO: Remove anyhow/thiserror from lib?
// TODO: Write more unit tests (e.g. AABB, UniformGrid, neighborhood search)
// TODO: Write some integration tests
// TODO: Test kernels with property based testing?
// TODO: Investigate why reconstruction crashes with an AABB that is too small
// TODO: Add free particles back again after triangulation as sphere meshes if they were removed
// TODO: Check why, when particle density is erroneously initialized with zero, the cell interpolation crashes
// TODO: Detect free particles by just comparing with the SPH density of a free particle? (no need for extra neighborhood search?)
// TODO: Ensure that if an AABB is adapted for an operation (e.g. a margin is added), that it shrinks towards the original center of the AABB
// TODO: More and better error messages with distinct types
// TODO: Make flat indices strongly typed
// TODO: Windowed approach that supports multi threading and dense operations without hashmap
// TODO: Make deterministic ordering a feature flag / runtime option
// TODO: Function that detects smallest usable index type

pub(crate) type HashState = fxhash::FxBuildHasher;
pub(crate) type MapType<K, V> = std::collections::HashMap<K, V, HashState>;
pub(crate) fn new_map<K, V>() -> MapType<K, V> {
    MapType::with_hasher(HashState::default())
}

/*
// Switch to BTreeMap in debug mode for easier debugging due to deterministic iteration order
#[cfg(debug_assertions)]
pub(crate) type MapType<K, V> = std::collections::BTreeMap<K, V>;
#[cfg(not(debug_assertions))]
pub(crate) type MapType<K, V> = std::collections::HashMap<K, V, HashState>;

// Function for consistent construction of the used map type (depending on debug/release build)
#[cfg(debug_assertions)]
pub(crate) fn new_map<K: std::cmp::Ord, V>() -> MapType<K, V> {
    MapType::new()
}
#[cfg(not(debug_assertions))]
pub(crate) fn new_map<K, V>() -> MapType<K, V> {
    MapType::with_hasher(HashState::default())
}
*/

pub(crate) type ParallelMapType<K, V> = dashmap::DashMap<K, V, HashState>;

/// Macro version of Option::map that allows using e.g. using the ?-operator in the map expression
macro_rules! map_option {
    ($some_optional:expr, $value_identifier:ident => $value_transformation:expr) => {
        match $some_optional {
            Some($value_identifier) => Some($value_transformation),
            None => None,
        }
    };
}

/// Parameters for the spatial decomposition
#[derive(Clone, Debug)]
pub struct SpatialDecompositionParameters<R: Real> {
    /// Criterion used for subdivision of the octree cells
    pub subdivision_criterion: SubdivisionCriterion,
    /// Safety factor applied to the kernel radius when it's used as a margin to collect ghost particles in the leaf nodes
    pub ghost_particle_safety_factor: Option<R>,
}

impl<R: Real> SpatialDecompositionParameters<R> {
    /// Tries to convert the parameters from one [Real] type to another [Real] type, returns None if conversion fails
    pub fn try_convert<T: Real>(&self) -> Option<SpatialDecompositionParameters<T>> {
        Some(SpatialDecompositionParameters {
            subdivision_criterion: self.subdivision_criterion.clone(),
            ghost_particle_safety_factor: map_option!(
                &self.ghost_particle_safety_factor,
                r => r.try_convert()?
            ),
        })
    }
}

/// Parameters for the surface reconstruction
#[derive(Clone, Debug)]
pub struct Parameters<R: Real> {
    /// Radius per particle (used to calculate the particle volume)
    pub particle_radius: R,
    /// Rest density of the fluid
    pub rest_density: R,
    /// Compact support radius of the kernel, i.e. distance from the particle where kernel reaches zero (in distance units, not relative to particle radius)
    pub kernel_radius: R,
    /// Particles without neighbors within the splash detection radius are considered "splash" or "free particles".
    /// They are filtered out and processed separately. Currently they are only skipped during the surface reconstruction.
    pub splash_detection_radius: Option<R>,
    /// Edge length of the marching cubes implicit background grid (in distance units, not relative to particle radius)
    pub cube_size: R,
    /// Density threshold value to distinguish between the inside (above threshold) and outside (below threshold) of the fluid
    pub iso_surface_threshold: R,
    /// Manually restrict the domain to the surface reconstruction.
    /// If not provided, the smallest AABB enclosing all particles is computed instead.
    pub domain_aabb: Option<AxisAlignedBoundingBox3d<R>>,
    /// Whether to allow multi threading within the surface reconstruction procedure
    pub enable_multi_threading: bool,
    /// Parameters for the spatial decomposition (octree subdivision) of the particles.
    /// If not provided, no octree is generated and a global approach is used instead.
    pub spatial_decomposition: Option<SpatialDecompositionParameters<R>>,
}

impl<R: Real> Parameters<R> {
    /// Tries to convert the parameters from one [Real] type to another [Real] type, returns None if conversion fails
    pub fn try_convert<T: Real>(&self) -> Option<Parameters<T>> {
        Some(Parameters {
            particle_radius: self.particle_radius.try_convert()?,
            rest_density: self.rest_density.try_convert()?,
            kernel_radius: self.kernel_radius.try_convert()?,
            splash_detection_radius: map_option!(
                &self.splash_detection_radius,
                r => r.try_convert()?
            ),
            cube_size: self.cube_size.try_convert()?,
            iso_surface_threshold: self.iso_surface_threshold.try_convert()?,
            domain_aabb: map_option!(&self.domain_aabb, aabb => aabb.try_convert()?),
            enable_multi_threading: self.enable_multi_threading,
            spatial_decomposition: map_option!(&self.spatial_decomposition, sd => sd.try_convert()?),
        })
    }
}

/// Result data returned when the surface reconstruction was successful
#[derive(Clone, Debug)]
pub struct SurfaceReconstruction<I: Index, R: Real> {
    /// Background grid that was used as a basis for generating the density map for marching cubes
    grid: UniformGrid<I, R>,
    /// Octree built for domain decomposition
    octree: Option<Octree<I, R>>,
    /// Point-based density map generated from the particles that was used as input to marching cubes
    density_map: Option<DensityMap<I, R>>,
    /// Surface mesh that is the result of the surface reconstruction
    mesh: TriMesh3d<R>,
    /// Workspace with allocated memory for subsequent surface reconstructions
    workspace: ReconstructionWorkspace<I, R>,
}

impl<I: Index, R: Real> Default for SurfaceReconstruction<I, R> {
    /// Returns an empty [SurfaceReconstruction] to pass into the inplace surface reconstruction
    fn default() -> Self {
        Self {
            grid: UniformGrid::new_zero(),
            octree: None,
            density_map: None,
            mesh: TriMesh3d::default(),
            workspace: ReconstructionWorkspace::default(),
        }
    }
}

impl<I: Index, R: Real> SurfaceReconstruction<I, R> {
    /// Returns a reference to the actual triangulated surface mesh that is the result of the reconstruction
    pub fn mesh(&self) -> &TriMesh3d<R> {
        &self.mesh
    }

    /// Returns a reference to the octree generated for spatial decomposition of the input particles
    pub fn octree(&self) -> Option<&Octree<I, R>> {
        self.octree.as_ref()
    }

    /// Returns a reference to the sparse density map (discretized on the vertices of the background grid) that is used as input for marching cubes
    pub fn density_map(&self) -> Option<&DensityMap<I, R>> {
        self.density_map.as_ref()
    }

    /// Returns a reference to the virtual background grid that was used as a basis for discretization of the density map for marching cubes, can be used to convert the density map to a hex mesh (using [sparse_density_map_to_hex_mesh](density_map::sparse_density_map_to_hex_mesh))
    pub fn grid(&self) -> &UniformGrid<I, R> {
        &self.grid
    }
}

impl<I: Index, R: Real> From<SurfaceReconstruction<I, R>> for TriMesh3d<R> {
    /// Extracts the reconstructed mesh
    fn from(result: SurfaceReconstruction<I, R>) -> Self {
        result.mesh
    }
}

/// Error type returned when the surface reconstruction fails
#[non_exhaustive]
#[derive(Debug, ThisError)]
pub enum ReconstructionError<I: Index, R: Real> {
    /// Errors that occur during the implicit construction of the virtual background grid used for the density map and marching cubes
    #[error("grid construction: {0}")]
    GridConstructionError(GridConstructionError<I, R>),
    /// Any error that is not represented by some other explicit variant
    #[error("unknown error")]
    Unknown(anyhow::Error),
}

impl<I: Index, R: Real> From<GridConstructionError<I, R>> for ReconstructionError<I, R> {
    /// Allows automatic conversion of a [GridConstructionError] to a [ReconstructionError]
    fn from(error: GridConstructionError<I, R>) -> Self {
        ReconstructionError::GridConstructionError(error)
    }
}

impl<I: Index, R: Real> From<anyhow::Error> for ReconstructionError<I, R> {
    /// Allows automatic conversion of an anyhow::Error to a [ReconstructionError]
    fn from(error: anyhow::Error) -> Self {
        ReconstructionError::Unknown(error)
    }
}

/// Initializes the global thread pool used by this library with the given parameters.
///
/// Initialization of the global thread pool happens exactly once.
/// Therefore, if you call `initialize_thread_pool` a second time, it will return an error.
/// An `Ok` result indicates that this is the first initialization of the thread pool.
pub fn initialize_thread_pool(num_threads: usize) -> Result<(), anyhow::Error> {
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
    Ok(())
}

/// Performs a marching cubes surface construction of the fluid represented by the given particle positions
#[inline(never)]
pub fn reconstruct_surface<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
) -> Result<SurfaceReconstruction<I, R>, ReconstructionError<I, R>> {
    profile!("reconstruct_surface");
    let mut surface = SurfaceReconstruction::default();
    reconstruct_surface_inplace(particle_positions, parameters, &mut surface)?;
    Ok(surface)
}

/// Performs a marching cubes surface construction of the fluid represented by the given particle positions, inplace
pub fn reconstruct_surface_inplace<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    // Clear the existing mesh
    output_surface.mesh.clear();

    // Initialize grid for the reconstruction
    output_surface.grid = grid_for_reconstruction(
        particle_positions,
        parameters.particle_radius,
        parameters.kernel_radius,
        parameters.cube_size,
        parameters.domain_aabb.as_ref(),
        parameters.enable_multi_threading,
    )?;
    let grid = &output_surface.grid;
    log_grid_info(grid);

    if parameters.spatial_decomposition.is_some() {
        reconstruct_surface_inplace_octree(particle_positions, parameters, output_surface)?
    } else {
        profile!("reconstruct_surface_inplace");

        let mut workspace = output_surface
            .workspace
            .get_local_with_capacity(particle_positions.len())
            .borrow_mut();

        // Clear the current mesh, as reconstruction will be appended to output
        output_surface.mesh.clear();
        // Perform global reconstruction without octree
        reconstruct_single_surface_append(
            &mut *workspace,
            grid,
            None,
            None,
            particle_positions,
            parameters,
            &mut output_surface.mesh,
        );

        /*
        let particle_indices = splash_detection_radius.map(|splash_detection_radius| {
            let neighborhood_list = neighborhood_search::search::<I, R>(
                &grid.aabb(),
                particle_positions,
                splash_detection_radius,
                enable_multi_threading,
            );

            let mut active_particles = Vec::new();
            for (particle_i, neighbors) in neighborhood_list.iter().enumerate() {
                if !neighbors.is_empty() {
                    active_particles.push(particle_i);
                }
            }

            active_particles
        });
        */

        // TODO: Set this correctly
        output_surface.density_map = None;
    }

    Ok(())
}

/// Called from general surface reconstruction function if spatial decomposition has to be performed
fn reconstruct_surface_inplace_octree<'a, I: Index, R: Real>(
    global_particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_inplace_octree");

    // The grid was already generated by the calling public function
    let grid = &output_surface.grid;

    output_surface.octree =
        if let Some(decomposition_parameters) = &parameters.spatial_decomposition {
            let margin_factor = decomposition_parameters
                .ghost_particle_safety_factor
                .unwrap_or(R::one());

            Some(Octree::new_subdivided(
                &grid,
                global_particle_positions,
                decomposition_parameters.subdivision_criterion.clone(),
                parameters.kernel_radius * margin_factor,
                parameters.enable_multi_threading,
            ))
        } else {
            panic!("Called octree-based surface reconstruction without decomposition criterion");
        };

    let octree = output_surface.octree.as_ref().expect("No octree generated");

    // Disable all multi-threading in sub-tasks for now (sub-tasks are processed in parallel instead)
    let parameters = &{
        let mut p = parameters.clone();
        p.enable_multi_threading = false;
        p
    };

    // Clear all local meshes
    {
        let tl_workspaces = &mut output_surface.workspace;
        // Clear all thread local meshes
        tl_workspaces
            .local_workspaces_mut()
            .iter_mut()
            .for_each(|local_workspace| {
                local_workspace.borrow_mut().mesh.clear();
            });
    }

    // Perform individual surface reconstructions on all non-empty leaves of the octree
    {
        let tl_workspaces = &output_surface.workspace;

        profile!("parallel domain decomposed surface reconstruction");
        octree
            .root()
            .par_visit_bfs(|octree_leaf: &OctreeNode<I, R>| {
                let particles = if let Some(particle_set) = octree_leaf.data().particle_set() {
                    &particle_set.particles
                } else {
                    // Skip non-leaf nodes
                    return;
                };

                let mut tl_workspace = tl_workspaces
                    .get_local_with_capacity(particles.len())
                    .borrow_mut();

                info!("Processing octree leaf with {} particles", particles.len());

                // Generate grid for the subdomain of this octree leaf
                let leaf_aabb = octree_leaf.aabb(grid);
                let subdomain_grid = &octree_leaf
                    .grid(leaf_aabb.min(), grid.cell_size())
                    .expect("Unable to construct Octree node grid");
                let subdomain_offset = octree_leaf.min_corner();
                log_grid_info(subdomain_grid);

                // Take particle position storage from workspace and fill it with positions of the leaf
                let particle_positions = {
                    let mut leaf_particle_positions =
                        std::mem::take(&mut tl_workspace.particle_positions);
                    leaf_particle_positions.clear();
                    utils::reserve_total(&mut leaf_particle_positions, particles.len());

                    // Extract the particle positions of the leaf
                    leaf_particle_positions.extend(
                        particles
                            .iter()
                            .copied()
                            .map(|idx| global_particle_positions[idx]),
                    );

                    leaf_particle_positions
                };

                // Take the thread local mesh (don't clear just append)
                let mut tl_mesh = std::mem::take(&mut tl_workspace.mesh);

                reconstruct_single_surface_append(
                    &mut *tl_workspace,
                    grid,
                    Some(subdomain_offset),
                    Some(subdomain_grid),
                    particle_positions.as_slice(),
                    parameters,
                    &mut tl_mesh,
                );

                // Put back the particle position and mesh storage
                tl_workspace.particle_positions = particle_positions;
                tl_workspace.mesh = tl_mesh;
            });
    };

    // Append local meshes to global mesh
    {
        let tl_workspaces = &mut output_surface.workspace;
        // Clear the current global mesh
        output_surface.mesh.clear();
        // Append all thread local meshes to the global mesh
        tl_workspaces.local_workspaces_mut().iter_mut().fold(
            &mut output_surface.mesh,
            |global_mesh, local_workspace| {
                global_mesh.append(&mut local_workspace.borrow_mut().mesh);
                global_mesh
            },
        );

        info!(
            "Global mesh has {} triangles and {} vertices.",
            output_surface.mesh.triangles.len(),
            output_surface.mesh.vertices.len()
        );
    }

    output_surface.density_map = None;

    Ok(())
}

fn reconstruct_surface_octree_recursive<'a, I: Index, R: Real>(
    global_particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_inplace_octree");

    // The grid was already generated by the calling public function
    let grid = &output_surface.grid;

    output_surface.octree =
        if let Some(decomposition_parameters) = &parameters.spatial_decomposition {
            let margin_factor = decomposition_parameters
                .ghost_particle_safety_factor
                .unwrap_or(R::one());

            Some(Octree::new_subdivided(
                &grid,
                global_particle_positions,
                decomposition_parameters.subdivision_criterion.clone(),
                parameters.kernel_radius * margin_factor,
                parameters.enable_multi_threading,
            ))
        } else {
            panic!("Called octree-based surface reconstruction without decomposition criterion");
        };

    // Make a clone of the octree, as we want to preserve the particle lists for the user while this local octree gets merged later
    let mut octree = output_surface.octree.clone().expect("No octree generated");

    // Disable all multi-threading in sub-tasks for now (sub-tasks are processed in parallel instead)
    let parameters = &{
        let mut p = parameters.clone();
        p.enable_multi_threading = false;
        p
    };

    // Perform individual surface reconstructions on all non-empty leaves of the octree
    {
        let tl_workspaces = &output_surface.workspace;

        profile!("parallel domain decomposed surface reconstruction");
        octree
            .root_mut()
            .par_visit_mut_bfs(|octree_node: &mut OctreeNode<I, R>| {
                let particles = if let Some(particle_set) = octree_node.data().particle_set() {
                    &particle_set.particles
                } else {
                    // TODO: Instead of skipping use DFS and start stitching when children are done.
                    // Skip non-leaf nodes
                    return;
                };

                let mut tl_workspace = tl_workspaces
                    .get_local_with_capacity(particles.len())
                    .borrow_mut();

                info!("Processing octree leaf with {} particles", particles.len());

                // Generate grid for the subdomain of this octree leaf
                let leaf_aabb = octree_node.aabb(grid);
                let subdomain_grid = &octree_node
                    .grid(leaf_aabb.min(), grid.cell_size())
                    .expect("Unable to construct Octree node grid");
                let subdomain_offset = octree_node.min_corner();
                log_grid_info(subdomain_grid);

                // Take particle position storage from workspace and fill it with positions of the leaf
                let particle_positions = {
                    let mut leaf_particle_positions =
                        std::mem::take(&mut tl_workspace.particle_positions);
                    leaf_particle_positions.clear();
                    utils::reserve_total(&mut leaf_particle_positions, particles.len());

                    // Extract the particle positions of the leaf
                    leaf_particle_positions.extend(
                        particles
                            .iter()
                            .copied()
                            .map(|idx| global_particle_positions[idx]),
                    );

                    leaf_particle_positions
                };

                let surface_patch = reconstruct_surface_patch(
                    &mut *tl_workspace,
                    grid,
                    subdomain_offset,
                    subdomain_grid,
                    particle_positions.as_slice(),
                    parameters,
                );

                // Store triangulation in the leaf
                octree_node
                    .data_mut()
                    .replace(NodeData::SurfacePatch(surface_patch));

                // Put back the particle position storage
                tl_workspace.particle_positions = particle_positions;
            });
    };

    // Merge meshes
    {
        // Clear the current global mesh
        output_surface.mesh.clear();

        octree
            .root_mut()
            .visit_mut_bfs(|octree_node: &mut OctreeNode<I, R>| {
                if let NodeData::SurfacePatch(surface_patch) = octree_node.data_mut() {
                    output_surface.mesh.append(&mut surface_patch.mesh);
                }
            });

        info!(
            "Global mesh has {} triangles and {} vertices.",
            output_surface.mesh.triangles.len(),
            output_surface.mesh.vertices.len()
        );
    }

    output_surface.density_map = None;

    Ok(())
}

/// Reconstruct a surface, appends triangulation to the given mesh
fn reconstruct_single_surface_append<'a, I: Index, R: Real>(
    workspace: &mut LocalReconstructionWorkspace<I, R>,
    grid: &UniformGrid<I, R>,
    subdomain_offset: Option<&PointIndex<I>>,
    subdomain_grid: Option<&UniformGrid<I, R>>,
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_mesh: &'a mut TriMesh3d<R>,
) {
    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    info!("Starting neighborhood search...");
    neighborhood_search::search_inplace::<I, R>(
        &grid.aabb(),
        particle_positions,
        parameters.kernel_radius,
        parameters.enable_multi_threading,
        &mut workspace.particle_neighbor_lists,
    );

    info!("Computing particle densities...");
    density_map::compute_particle_densities_inplace::<I, R>(
        particle_positions,
        workspace.particle_neighbor_lists.as_slice(),
        parameters.kernel_radius,
        particle_rest_mass,
        parameters.enable_multi_threading,
        &mut workspace.particle_densities,
    );

    // Create a new density map, reusing memory with the workspace is bad for cache efficiency
    // Alternatively one could reuse memory with a custom caching allocator
    let mut density_map = new_map().into();
    density_map::generate_sparse_density_map(
        grid,
        subdomain_offset,
        subdomain_grid,
        particle_positions,
        workspace.particle_densities.as_slice(),
        None,
        particle_rest_mass,
        parameters.kernel_radius,
        parameters.cube_size,
        parameters.enable_multi_threading,
        &mut density_map,
    );

    marching_cubes::triangulate_density_map_append::<I, R>(
        grid,
        subdomain_offset,
        subdomain_grid,
        &density_map,
        parameters.iso_surface_threshold,
        output_mesh,
    );
}

/// Reconstruct a surface, appends triangulation to the given mesh
fn reconstruct_surface_patch<I: Index, R: Real>(
    workspace: &mut LocalReconstructionWorkspace<I, R>,
    grid: &UniformGrid<I, R>,
    subdomain_offset: &PointIndex<I>,
    subdomain_grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
) -> SurfacePatch<I, R> {
    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    info!("Starting neighborhood search...");
    neighborhood_search::search_inplace::<I, R>(
        &grid.aabb(),
        particle_positions,
        parameters.kernel_radius,
        parameters.enable_multi_threading,
        &mut workspace.particle_neighbor_lists,
    );

    info!("Computing particle densities...");
    density_map::compute_particle_densities_inplace::<I, R>(
        particle_positions,
        workspace.particle_neighbor_lists.as_slice(),
        parameters.kernel_radius,
        particle_rest_mass,
        parameters.enable_multi_threading,
        &mut workspace.particle_densities,
    );

    // Create a new density map, reusing memory with the workspace is bad for cache efficiency
    // Alternatively one could reuse memory with a custom caching allocator
    let mut density_map = new_map().into();
    density_map::generate_sparse_density_map(
        grid,
        Some(subdomain_offset),
        Some(subdomain_grid),
        particle_positions,
        workspace.particle_densities.as_slice(),
        None,
        particle_rest_mass,
        parameters.kernel_radius,
        parameters.cube_size,
        parameters.enable_multi_threading,
        &mut density_map,
    );

    marching_cubes::triangulate_density_map_with_stitching_data::<I, R>(
        grid,
        subdomain_offset,
        subdomain_grid,
        &density_map,
        parameters.iso_surface_threshold,
    )
}

/// Logs the information about the given grid
fn log_grid_info<I: Index, R: Real>(grid: &UniformGrid<I, R>) {
    info!(
        "Using a grid with {:?}x{:?}x{:?} points and {:?}x{:?}x{:?} cells of edge length {}.",
        grid.points_per_dim()[0],
        grid.points_per_dim()[1],
        grid.points_per_dim()[2],
        grid.cells_per_dim()[0],
        grid.cells_per_dim()[1],
        grid.cells_per_dim()[2],
        grid.cell_size()
    );
    info!("The resulting domain size is: {:?}", grid.aabb());
}

/// Constructs the background grid for marching cubes based on the parameters supplied to the surface reconstruction
pub fn grid_for_reconstruction<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_radius: R,
    kernel_radius: R,
    cube_size: R,
    domain_aabb: Option<&AxisAlignedBoundingBox3d<R>>,
    enable_multi_threading: bool,
) -> Result<UniformGrid<I, R>, ReconstructionError<I, R>> {
    let domain_aabb = if let Some(domain_aabb) = domain_aabb {
        domain_aabb.clone()
    } else {
        profile!("compute minimum enclosing aabb");

        let mut domain_aabb = {
            let mut aabb = if enable_multi_threading {
                AxisAlignedBoundingBox3d::from_points_par(particle_positions)
            } else {
                AxisAlignedBoundingBox3d::from_points(particle_positions)
            };
            aabb.grow_uniformly(particle_radius);
            aabb
        };

        info!(
            "Minimal enclosing bounding box of particles was computed as: {:?}",
            domain_aabb
        );

        // Ensure that we have enough margin around the particles such that the every particle's kernel support is completely in the domain
        let kernel_margin =
            density_map::compute_kernel_evaluation_radius::<I, R>(kernel_radius, cube_size)
                .kernel_evaluation_radius;
        domain_aabb.grow_uniformly(kernel_margin);

        domain_aabb
    };

    Ok(UniformGrid::from_aabb(&domain_aabb, cube_size)?)
}
