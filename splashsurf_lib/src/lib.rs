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
/// Types related to the virtual background grid used for marching cubes
pub mod uniform_grid;
mod utils;

pub use aabb::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d};
pub use density_map::DensityMap;
pub use numeric_types::{Index, Real, ThreadSafe};
pub use uniform_grid::{GridConstructionError, UniformGrid};

use log::info;
use nalgebra::Vector3;
use rayon::prelude::*;
use thiserror::Error as ThisError;

use crate::uniform_grid::PointIndex;
use mesh::TriMesh3d;
use octree::Octree;

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

/// Criterion used for spatial decomposition of the particle collection
#[derive(Clone, Debug)]
pub enum SpatialDecompositionCriterion {
    /// Perform octree subdivision until a maximum number of particles is reached per chunk
    MaxParticleCount,
}

/// Parameters for the surface reconstruction
#[derive(Clone, Debug)]
pub struct Parameters<R: Real> {
    /// Radius per particle (used to calculate the particle volume)
    pub particle_radius: R,
    /// Rest density of the fluid
    pub rest_density: R,
    /// Compact support radius of the kernel, i.e. distance from the particle where kernel reaches zero
    pub kernel_radius: R,
    /// Particles without neighbors within the splash detection radius are considered "splash" or "free particles".
    /// They are filtered out and processed separately. Currently they are only skipped during the surface reconstruction.
    pub splash_detection_radius: Option<R>,
    /// Edge length of the marching cubes implicit background grid
    pub cube_size: R,
    /// Density threshold value to distinguish between the inside (above threshold) and outside (below threshold) of the fluid
    pub iso_surface_threshold: R,
    /// Manually restrict the domain to the surface reconstruction.
    /// If not provided, the smallest AABB enclosing all particles is computed instead.
    pub domain_aabb: Option<AxisAlignedBoundingBox3d<R>>,
    /// Whether to allow multi threading within the surface reconstruction procedure
    pub enable_multi_threading: bool,
    /// Strategy used for spatial decomposition (octree subdivision) of the particles.
    /// If not provided, no octree is generated and a global approach is used instead.
    pub spatial_decomposition: Option<SpatialDecompositionCriterion>,
}

/// Macro version of Option::map that allows using e.g. using the ?-operator in the map expression
macro_rules! map_option {
    ($some_optional:expr, $value_identifier:ident => $value_transformation:expr) => {
        match $some_optional {
            Some($value_identifier) => Some($value_transformation),
            None => None,
        }
    };
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
            spatial_decomposition: self.spatial_decomposition.clone(),
        })
    }
}

/// Result data returned when the surface reconstruction was successful
#[derive(Clone, Debug)]
pub struct SurfaceReconstruction<I: Index, R: Real> {
    /// The background grid that was used as a basis for generating the density map for marching cubes
    grid: UniformGrid<I, R>,
    /// Octree built for domain decomposition
    octree: Option<Octree<I>>,
    /// The point-based density map generated from the particles that was used as input to marching cubes
    density_map: Option<DensityMap<I, R>>,
    /// The actual mesh that is the result of the surface reconstruction
    mesh: TriMesh3d<R>,
}

impl<I: Index, R: Real> Default for SurfaceReconstruction<I, R> {
    /// Returns an empty [SurfaceReconstruction] to pass into the inplace surface reconstruction
    fn default() -> Self {
        Self {
            grid: UniformGrid::new_zero(),
            octree: None,
            density_map: None,
            mesh: TriMesh3d::default(),
        }
    }
}

impl<I: Index, R: Real> SurfaceReconstruction<I, R> {
    /// Returns a reference to the actual triangulated surface mesh that is the result of the reconstruction
    pub fn mesh(&self) -> &TriMesh3d<R> {
        &self.mesh
    }

    /// Returns a reference to the octree generated for spatial decomposition of the input particles
    pub fn octree(&self) -> Option<&Octree<I>> {
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
    output_surface.grid = grid_for_reconstruction(
        particle_positions,
        parameters.particle_radius,
        parameters.cube_size,
        parameters.domain_aabb.as_ref(),
    )?;
    let grid = &output_surface.grid;

    log_grid_info(grid);

    if parameters.spatial_decomposition.is_some() {
        reconstruct_surface_inplace_octree(particle_positions, parameters, output_surface)?
    } else {
        profile!("reconstruct_surface_inplace");

        reconstruct_single_surface(
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

fn reconstruct_surface_inplace_octree<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_inplace_octree");

    // Disable multi-threading in sub-tasks for now
    let parameters = &{
        let mut p = parameters.clone();
        p.enable_multi_threading = false;
        p
    };

    // The grid was already generated by the calling public function
    let grid = &output_surface.grid;

    output_surface.octree = if parameters.spatial_decomposition.is_some() {
        let particles_per_cell = utils::ChunkSize::new(particle_positions.len())
            .with_log("particles")
            .chunk_size;

        info!(
            "Building octree with at most {} particles per leaf",
            particles_per_cell
        );
        let mut tree = Octree::new(&grid, particle_positions.len());
        tree.subdivide_recursively_margin(
            grid,
            particle_positions,
            particles_per_cell,
            parameters.kernel_radius,
        );
        Some(tree)
    } else {
        panic!("Called octree-based surface reconstruction without decomposition criterion");
    };

    let octree = output_surface.octree.as_ref().expect("No octree generated");

    // Collect the particle lists of all octree leaf nodes
    let octree_leaves: Vec<_> = octree
        .leaf_iter()
        .filter(|octree_leaf| {
            !octree_leaf
                .particles()
                .expect("Octree node has to be a leaf")
                .is_empty()
        })
        .collect();

    info!("Octree has {} leaf nodes", octree_leaves.len());

    // Perform individual surface reconstructions on all non-empty leaves of the octree
    let global_mesh = {
        profile!("parallel domain decomposed surface reconstruction");
        octree_leaves
            .par_iter()
            .copied()
            .fold(TriMesh3d::default, |mut global_mesh, octree_leaf| {
                let particles = octree_leaf
                    .particles()
                    .expect("Octree node has to be a leaf");

                info!("Processing octree leaf with {} particles", particles.len());

                // Generate grid for the subdomain of this octree leaf
                let leaf_aabb = octree_leaf.aabb(grid);
                let subdomain_grid = &octree_leaf
                    .grid(leaf_aabb.min(), grid.cell_size())
                    .expect("Unable to construct Octree node grid");
                let subdomain_offset = octree_leaf.min_corner();
                log_grid_info(subdomain_grid);

                // TODO: Use thread_local workspace for this
                let particle_positions = particles
                    .iter()
                    .copied()
                    .map(|idx| particle_positions[idx])
                    .collect::<Vec<_>>();

                let mut subdomain_mesh = TriMesh3d::default();
                reconstruct_single_surface(
                    grid,
                    Some(subdomain_offset),
                    Some(subdomain_grid),
                    particle_positions.as_slice(),
                    parameters,
                    &mut subdomain_mesh,
                );

                // Append the subdomain mesh to the global mesh
                global_mesh.append(subdomain_mesh);
                global_mesh
            })
            .reduce(TriMesh3d::default, |mut global_mesh, local_mesh| {
                global_mesh.append(local_mesh);
                global_mesh
            })
    };

    info!(
        "Global mesh has {} triangles and {} vertices.",
        global_mesh.triangles.len(),
        global_mesh.vertices.len()
    );

    output_surface.density_map = None;
    output_surface.mesh = global_mesh;

    Ok(())
}

fn reconstruct_single_surface<'a, I: Index, R: Real>(
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

    let particle_densities = {
        info!("Starting neighborhood search...");

        let particle_neighbor_lists = neighborhood_search::search::<I, R>(
            &grid.aabb(),
            particle_positions,
            parameters.kernel_radius,
            parameters.enable_multi_threading,
        );

        info!("Computing particle densities...");

        density_map::compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists.as_slice(),
            parameters.kernel_radius,
            particle_rest_mass,
            parameters.enable_multi_threading,
        )
    };

    // TODO: Use thread_local workspace for this
    let density_map = density_map::generate_sparse_density_map(
        grid,
        subdomain_offset,
        subdomain_grid,
        particle_positions,
        particle_densities.as_slice(),
        None,
        particle_rest_mass,
        parameters.kernel_radius,
        parameters.cube_size,
        parameters.enable_multi_threading,
    );

    marching_cubes::triangulate_density_map::<I, R>(
        subdomain_grid.unwrap_or(grid),
        &density_map,
        parameters.iso_surface_threshold,
        output_mesh,
    );
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
    cube_size: R,
    domain_aabb: Option<&AxisAlignedBoundingBox3d<R>>,
) -> Result<UniformGrid<I, R>, ReconstructionError<I, R>> {
    let domain_aabb = if let Some(domain_aabb) = domain_aabb {
        domain_aabb.clone()
    } else {
        profile!("compute minimum enclosing aabb");

        let mut domain_aabb = {
            let mut aabb = AxisAlignedBoundingBox3d::from_points(particle_positions);
            aabb.grow_uniformly(particle_radius);
            aabb
        };

        info!(
            "Minimal enclosing bounding box of particles was computed as: {:?}",
            domain_aabb
        );

        domain_aabb.scale_uniformly(R::one().times(2));
        domain_aabb
    };

    Ok(UniformGrid::from_aabb(&domain_aabb, cube_size)?)
}
