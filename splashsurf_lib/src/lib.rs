#![cfg_attr(doc_cfg, feature(doc_cfg))]

//!
//! Library for surface reconstruction of SPH particle data using marching cubes.
//!
//! Entry points are the [`reconstruct_surface`] or [`reconstruct_surface_inplace`] functions.
//!
//! ## Feature flags
//! The following features are all non-default features to reduce the amount of additional dependencies.
//!
//! - **`vtk_extras`**: Enables helper functions and trait implementations to export meshes using [`vtkio`](https://github.com/elrnv/vtkio).
//!  In particular it adds `From` impls for the [mesh](crate::mesh) types used by this crate to convert them to
//!  [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) and [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
//!  types. If the feature is enabled, The crate exposes its `vtkio` dependency as `splashsurflib::vtkio`.
//! - **`io`**: Enables the [`io`] module, containing functions to load and store particle and mesh files
//!  from various file formats, e.g. `VTK`, `OBJ`, `BGEO` etc. This feature implies the `vtk_extras` feature.
//!  It is disabled by default because a pure "online" surface reconstruction might not need any file IO.
//!  The feature adds several dependencies to support the file formats.
//! - **`profiling`**: Enables profiling of internal functions. The resulting data can be displayed using the functions
//!  from the [`profiling`] module. Furthermore, it exposes the [`profile`] macro that can be used e.g.
//!  by binary crates calling into this library to add their own profiling scopes to the measurements.
//!  If this features is not enabled, the macro will just expend to a no-op and remove the (small)
//!  performance overhead of the profiling.
//!

use log::info;
/// Re-export the version of `nalgebra` used by this crate
pub use nalgebra;
use nalgebra::Vector3;
use thiserror::Error as ThisError;
/// Re-export the version of `vtkio` used by this crate, if vtk support is enabled
#[cfg(feature = "vtk_extras")]
pub use vtkio;

pub use crate::aabb::{AxisAlignedBoundingBox, Aabb2d, Aabb3d};
pub use crate::density_map::DensityMap;
pub use crate::octree::SubdivisionCriterion;
pub use crate::traits::{Index, Real, ThreadSafe};
pub use crate::uniform_grid::UniformGrid;

use crate::density_map::DensityMapError;
use crate::marching_cubes::MarchingCubesError;
use crate::mesh::TriMesh3d;
use crate::octree::Octree;
use crate::uniform_grid::GridConstructionError;
use crate::workspace::ReconstructionWorkspace;

#[cfg(feature = "profiling")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "profiling")))]
pub mod profiling;
#[doc(hidden)]
pub mod profiling_macro;

mod aabb;
pub mod density_map;
pub mod generic_tree;
#[cfg(feature = "io")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "io")))]
pub mod io;
pub mod kernel;
pub mod marching_cubes;
pub mod mesh;
pub mod neighborhood_search;
pub mod octree;
mod reconstruction;
pub mod sph_interpolation;
pub mod topology;
mod traits;
pub mod uniform_grid;
#[macro_use]
mod utils;
pub(crate) mod workspace;

// TODO: Add documentation of feature flags
// TODO: Feature flag for multi threading
// TODO: Feature flag to disable (debug level) logging?

// TODO: Remove anyhow/thiserror from lib?
// TODO: Write more unit tests (e.g. AABB, UniformGrid, neighborhood search)
// TODO: Test kernels with property based testing?
// TODO: More and better error messages with distinct types
// TODO: Make flat indices strongly typed
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

/// Parameters for the spatial decomposition
#[derive(Clone, Debug)]
pub struct SpatialDecompositionParameters<R: Real> {
    /// Criterion used for subdivision of the octree cells
    pub subdivision_criterion: SubdivisionCriterion,
    /// Safety factor applied to the kernel radius when it's used as a margin to collect ghost particles in the leaf nodes
    pub ghost_particle_safety_factor: Option<R>,
    /// Whether to enable stitching of all disjoint subdomain meshes to a global manifold mesh
    pub enable_stitching: bool,
    /// Which method to use for computing the densities of the particles
    pub particle_density_computation: ParticleDensityComputationStrategy,
}

/// Available strategies for the computation of the particle densities
#[derive(Copy, Clone, Debug)]
pub enum ParticleDensityComputationStrategy {
    /// Compute the particle densities globally before performing domain decomposition.
    ///
    /// With this approach the particle densities are computed globally on all particles before any
    /// domain decomposition is performed.
    ///
    /// This approach is guaranteed to lead to consistent results and does not depend on the following
    /// decomposition. However, it is also by far the *slowest method* as global operations (especially
    /// the global neighborhood search) are much slower.
    Global,
    /// Compute particle densities for all particles locally followed by a synchronization step.
    ///
    /// **This is the recommended approach.**
    /// The particle densities will be evaluated for all particles per subdomain, possibly in parallel.
    /// Afterwards, the values for all non-ghost particles are written to a global array.
    /// This happens in a separate step before performing any reconstructions
    /// For the following reconstruction procedure, each subdomain will update the densities of its ghost particles
    /// from this global array. This ensures that all ghost-particles receive correct density values
    /// without requiring to double the width of the ghost-particle margin just to ensure correct values
    /// for the actual inner ghost-particles (i.e. in contrast to the completely local approach).
    ///
    /// The actual synchronization overhead is relatively low and this approach is often the fastest method.
    ///
    /// This approach should always lead consistent results. Only in very rare cases when a particle is not
    /// uniquely assigned during domain decomposition this might lead to problems. If you encounter such
    /// problems with this approach please report it as a bug.
    SynchronizeSubdomains,
    /// Compute densities locally per subdomain without global synchronization.
    ///
    /// The particle densities will be evaluated per subdomain on-the-fly just before the reconstruction
    /// of the subdomain happens. In order to compute correct densities for the ghost particles of each
    /// subdomain it is required that the ghost-particle margin is at least two times the kernel compact
    /// support radius. This may add a lot of additional ghost-particles to each subdomain.
    ///
    /// If the ghost-particle margin is not set wide enough, this may lead to density differences on subdomain
    /// boundaries. Otherwise this approach robust with respect to the classification of particles into the
    /// subdomains.
    IndependentSubdomains,
}

impl<R: Real> SpatialDecompositionParameters<R> {
    /// Tries to convert the parameters from one [`Real`] type to another [`Real`] type, returns `None` if conversion fails
    pub fn try_convert<T: Real>(&self) -> Option<SpatialDecompositionParameters<T>> {
        Some(SpatialDecompositionParameters {
            subdivision_criterion: self.subdivision_criterion.clone(),
            ghost_particle_safety_factor: map_option!(
                &self.ghost_particle_safety_factor,
                r => r.try_convert()?
            ),
            enable_stitching: self.enable_stitching,
            particle_density_computation: self.particle_density_computation,
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
    pub compact_support_radius: R,
    /// Edge length of the marching cubes implicit background grid (in distance units, not relative to particle radius)
    pub cube_size: R,
    /// Density threshold value to distinguish between the inside (above threshold) and outside (below threshold) of the fluid
    pub iso_surface_threshold: R,
    /// Manually restrict the domain to the surface reconstruction.
    /// If not provided, the smallest AABB enclosing all particles is computed instead.
    pub domain_aabb: Option<Aabb3d<R>>,
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
            compact_support_radius: self.compact_support_radius.try_convert()?,
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
    /// Octree constructed for domain decomposition
    octree: Option<Octree<I, R>>,
    /// Point-based density map generated from the particles that was used as input to marching cubes
    density_map: Option<DensityMap<I, R>>,
    /// Per particle densities
    particle_densities: Option<Vec<R>>,
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
            particle_densities: None,
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

    /// Returns a reference to the octree generated for spatial decomposition of the input particles (mostly useful for debugging visualization)
    pub fn octree(&self) -> Option<&Octree<I, R>> {
        self.octree.as_ref()
    }

    /// Returns a reference to the sparse density map (discretized on the vertices of the background grid) that is used as input for marching cubes (always `None` when using domain decomposition)
    pub fn density_map(&self) -> Option<&DensityMap<I, R>> {
        self.density_map.as_ref()
    }

    /// Returns a reference to the global particle density vector if it was computed during the reconstruction (always `None` when using independent subdomains with domain decomposition)
    pub fn particle_densities(&self) -> Option<&Vec<R>> {
        self.particle_densities.as_ref()
    }

    /// Returns a reference to the virtual background grid that was used as a basis for discretization of the density map for marching cubes, can be used to convert the density map to a hex mesh (using [`density_map::sparse_density_map_to_hex_mesh`])
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
    /// Error that occurred during the initialization of the implicit background grid used for all subsequent stages
    #[error("grid construction")]
    GridConstructionError(
        #[source]
        #[from]
        GridConstructionError<I, R>,
    ),
    /// Error that occurred during the construction of the density map
    #[error("density map generation")]
    DensityMapGenerationError(
        #[source]
        #[from]
        DensityMapError<R>,
    ),
    /// Error that occurred during the marching cubes stage of the reconstruction
    #[error("marching cubes")]
    MarchingCubesError(
        #[source]
        #[from]
        MarchingCubesError,
    ),
    /// Any error that is not represented by some other explicit variant
    #[error(transparent)]
    Unknown(#[from] anyhow::Error),
}

/// Initializes the global thread pool used by this library with the given parameters.
///
/// Initialization of the global thread pool happens exactly once.
/// Therefore, if you call `initialize_thread_pool` a second time, it will return an error.
/// An `Ok` result indicates that this is the first initialization of the thread pool.
pub fn initialize_thread_pool(num_threads: usize) -> Result<(), anyhow::Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;
    Ok(())
}

/// Performs a marching cubes surface construction of the fluid represented by the given particle positions
#[inline(never)]
pub fn reconstruct_surface<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
) -> Result<SurfaceReconstruction<I, R>, ReconstructionError<I, R>> {
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
        parameters.compact_support_radius,
        parameters.cube_size,
        parameters.domain_aabb.as_ref(),
        parameters.enable_multi_threading,
    )?;

    output_surface.grid.log_grid_info();

    if parameters.spatial_decomposition.is_some() {
        reconstruction::reconstruct_surface_domain_decomposition(
            particle_positions,
            parameters,
            output_surface,
        )?;
    } else {
        reconstruction::reconstruct_surface_global(particle_positions, parameters, output_surface)?;
    }

    Ok(())
}

/// Constructs the background grid for marching cubes based on the parameters supplied to the surface reconstruction
pub fn grid_for_reconstruction<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_radius: R,
    compact_support_radius: R,
    cube_size: R,
    domain_aabb: Option<&Aabb3d<R>>,
    enable_multi_threading: bool,
) -> Result<UniformGrid<I, R>, ReconstructionError<I, R>> {
    let domain_aabb = if let Some(domain_aabb) = domain_aabb {
        domain_aabb.clone()
    } else {
        profile!("compute minimum enclosing aabb");

        let mut domain_aabb = {
            let mut aabb = if enable_multi_threading {
                Aabb3d::par_from_points(particle_positions)
            } else {
                Aabb3d::from_points(particle_positions)
            };
            aabb.grow_uniformly(particle_radius);
            aabb
        };

        info!(
            "Minimal enclosing bounding box of particles was computed as: {:?}",
            domain_aabb
        );

        // Ensure that we have enough margin around the particles such that the every particle's kernel support is completely in the domain
        let kernel_margin = density_map::compute_kernel_evaluation_radius::<I, R>(
            compact_support_radius,
            cube_size,
        )
        .kernel_evaluation_radius;
        domain_aabb.grow_uniformly(kernel_margin);

        domain_aabb
    };

    Ok(UniformGrid::from_aabb(&domain_aabb, cube_size)?)
}
