#![cfg_attr(docsrs, feature(doc_cfg))]

//!
//! Library for surface reconstruction of SPH particle data using marching cubes.
//!
//! Entry points are the [`reconstruct_surface`] or [`reconstruct_surface_inplace`] functions.
//!
//! ## Feature flags
//! The following features are all non-default features to reduce the amount of additional dependencies.
//!
//! - **`vtk_extras`**: Enables helper functions and trait implementations to export meshes using [`vtkio`](https://github.com/elrnv/vtkio).
//!   In particular it adds `From` impls for the [mesh] types used by this crate to convert them to
//!   [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) and [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
//!   types. If the feature is enabled, The crate exposes its `vtkio` dependency as `splashsurflib::vtkio`.
//! - **`io`**: Enables the [`io`] module, containing functions to load and store particle and mesh files
//!   from various file formats, e.g. `VTK`, `OBJ`, `BGEO` etc. This feature implies the `vtk_extras` feature.
//!   It is disabled by default because a pure "online" surface reconstruction might not need any file IO.
//!   The feature adds several dependencies to support the file formats.
//! - **`profiling`**: Enables profiling of internal functions. The resulting data can be displayed using the functions
//!   from the [`profiling`] module of this crate. Furthermore, it exposes the [`profile`] macro that can be used e.g.
//!   by binary crates calling into this library to add their own profiling scopes to the measurements.
//!   If this features is not enabled, the macro will just expend to a no-op and remove the (small)
//!   performance overhead of the profiling.
//!

use log::info;
/// Re-export the version of `nalgebra` used by this crate
pub use nalgebra;
use nalgebra::Vector3;
use std::borrow::Cow;
use std::hash::Hash;
use thiserror::Error as ThisError;
/// Re-export the version of `vtkio` used by this crate, if vtk support is enabled
#[cfg(feature = "vtk_extras")]
pub use vtkio;

pub use crate::aabb::{Aabb2d, Aabb3d, AxisAlignedBoundingBox};
pub use crate::density_map::DensityMap;
pub use crate::traits::{Index, Real, RealConvert, ThreadSafe};
pub use crate::uniform_grid::UniformGrid;

use crate::density_map::DensityMapError;
use crate::marching_cubes::MarchingCubesError;
use crate::mesh::TriMesh3d;
use crate::uniform_grid::GridConstructionError;
use crate::workspace::ReconstructionWorkspace;

#[cfg(feature = "profiling")]
#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
pub mod profiling;
#[doc(hidden)]
pub mod profiling_macro;

mod aabb;
pub(crate) mod dense_subdomains;
pub mod density_map;
pub mod generic_tree;
pub mod halfedge_mesh;
#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
pub mod io;
pub mod kernel;
pub mod marching_cubes;
pub mod mesh;
pub mod neighborhood_search;
pub mod postprocessing;
pub(crate) mod reconstruction;
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
pub(crate) type SetType<K> = std::collections::HashSet<K, HashState>;
pub(crate) fn new_map<K, V>() -> MapType<K, V> {
    // TODO: Remove this function
    Default::default()
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
fn new_parallel_map<K: Eq + Hash, V>() -> ParallelMapType<K, V> {
    ParallelMapType::with_hasher(HashState::default())
}

/// Approach used for spatial decomposition of the surface reconstruction and its parameters
#[derive(Clone, Debug)]
pub enum SpatialDecomposition {
    /// Use a uniform grid of subdomains with contiguous (dense) marching cubes grids per subdomain
    ///
    /// Only subdomains containing at least one particle will be processed.
    /// The small contiguous grid per subdomain make this approach very cache efficient.
    UniformGrid(GridDecompositionParameters),
}

/// Default parameters for the spatial decomposition use the uniform grid based decomposition approach
impl Default for SpatialDecomposition {
    fn default() -> Self {
        Self::UniformGrid(GridDecompositionParameters::default())
    }
}

/// Parameters for the uniform grid-based spatial decomposition
#[derive(Clone, Debug)]
pub struct GridDecompositionParameters {
    /// Each uniform subdomain will be a cube consisting of this number of MC cube cells along each coordinate axis
    pub subdomain_num_cubes_per_dim: u32,
}

impl Default for GridDecompositionParameters {
    fn default() -> Self {
        Self {
            subdomain_num_cubes_per_dim: 64,
        }
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
    /// Bounding box of particles to reconstruct
    ///
    /// All particles outside of this domain will be filtered out before the reconstruction.
    /// The surface reconstruction always results in a closed mesh around the particles.
    /// The final mesh can extend beyond this AABB due to the smoothing of the kernel.
    /// If not provided, the smallest AABB enclosing all particles is computed instead.
    pub particle_aabb: Option<Aabb3d<R>>,
    /// Whether to allow multi threading within the surface reconstruction procedure
    pub enable_multi_threading: bool,
    /// Parameters for the spatial decomposition of the surface reconstruction
    /// If not provided, no spatial decomposition is performed and a global approach is used instead.
    pub spatial_decomposition: Option<SpatialDecomposition>,
    /// Whether to return the global particle neighborhood list from the reconstruction.
    /// Depending on the settings of the reconstruction, neighborhood lists are only computed locally
    /// in subdomains. Enabling this flag joins this data over all particles which can add a small overhead.
    pub global_neighborhood_list: bool,
}

impl<R: Real> Parameters<R> {
    /// Tries to convert the parameters from one [Real] type to another [Real] type, returns `None` if conversion fails
    pub fn try_convert<T: Real>(&self) -> Option<Parameters<T>> {
        Some(Parameters {
            particle_radius: self.particle_radius.try_convert()?,
            rest_density: self.rest_density.try_convert()?,
            compact_support_radius: self.compact_support_radius.try_convert()?,
            cube_size: self.cube_size.try_convert()?,
            iso_surface_threshold: self.iso_surface_threshold.try_convert()?,
            particle_aabb: map_option!(&self.particle_aabb, aabb => aabb.try_convert()?),
            enable_multi_threading: self.enable_multi_threading,
            spatial_decomposition: self.spatial_decomposition.clone(),
            global_neighborhood_list: self.global_neighborhood_list,
        })
    }
}

/// Result data returned when the surface reconstruction was successful
#[derive(Clone, Debug)]
pub struct SurfaceReconstruction<I: Index, R: Real> {
    /// Background grid that was used as a basis for generating the density map for marching cubes
    grid: UniformGrid<I, R>,
    /// Per particle densities (contains only data of particles inside the domain)
    particle_densities: Option<Vec<R>>,
    /// If an AABB was specified to restrict the reconstruction, this stores per input particle whether they were inside
    particle_inside_aabb: Option<Vec<bool>>,
    /// Per particles neighbor lists
    particle_neighbors: Option<Vec<Vec<usize>>>,
    /// Surface mesh that is the result of the surface reconstruction
    mesh: TriMesh3d<R>,
    /// Workspace with allocated memory for subsequent surface reconstructions
    workspace: ReconstructionWorkspace<R>,
}

impl<I: Index, R: Real> Default for SurfaceReconstruction<I, R> {
    /// Returns an empty [SurfaceReconstruction] to pass into the in-place surface reconstruction
    fn default() -> Self {
        Self {
            grid: UniformGrid::new_zero(),
            particle_densities: None,
            particle_neighbors: None,
            particle_inside_aabb: None,
            mesh: TriMesh3d::default(),
            workspace: ReconstructionWorkspace::default(),
        }
    }
}

impl<I: Index, R: Real> SurfaceReconstruction<I, R> {
    /// Returns a reference to the surface mesh that is the result of the reconstruction
    pub fn mesh(&self) -> &TriMesh3d<R> {
        &self.mesh
    }

    /// Returns a reference to the global particle density vector if computed during the reconstruction (currently, all reconstruction approaches return this)
    pub fn particle_densities(&self) -> Option<&Vec<R>> {
        self.particle_densities.as_ref()
    }

    /// Returns a reference to the global list of per-particle neighborhood lists if computed during the reconstruction (`None` if not specified in the parameters)
    pub fn particle_neighbors(&self) -> Option<&Vec<Vec<usize>>> {
        self.particle_neighbors.as_ref()
    }

    /// Returns a reference to the virtual background grid that was used for marching cubes
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

/// Performs a marching cubes surface construction of the fluid represented by the given particle positions, in-place
pub fn reconstruct_surface_inplace<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    // Clear the existing mesh
    output_surface.mesh.clear();

    // Filter out particles
    let filtered_particle_positions = if let Some(particle_aabb) = &parameters.particle_aabb {
        profile!("filtering particles");

        use rayon::prelude::*;
        let mut particle_inside = output_surface
            .particle_inside_aabb
            .take()
            .unwrap_or_default();
        utils::reserve_total(&mut particle_inside, particle_positions.len());
        particle_positions
            .par_iter()
            .map(|p| particle_aabb.contains_point(p))
            .collect_into_vec(&mut particle_inside);
        let particle_inside_count = particle_inside.par_iter().copied().filter(|i| *i).count();

        // Take temporary storage for filtered particles from workspace
        let mut filtered_particles =
            std::mem::take(output_surface.workspace.filtered_particles_mut());
        filtered_particles.clear();
        utils::reserve_total(&mut filtered_particles, particle_inside_count);

        // Collect filtered particles
        filtered_particles.extend(
            particle_positions
                .iter()
                .zip(particle_inside.iter().copied())
                .filter(|(_, is_inside)| *is_inside)
                .map(|(p, _)| p)
                .cloned(),
        );

        output_surface.particle_inside_aabb = Some(particle_inside);
        Cow::Owned(filtered_particles)
    } else {
        Cow::Borrowed(particle_positions)
    };
    let particle_positions = filtered_particle_positions.as_ref();

    // Initialize grid for the reconstruction
    output_surface.grid = grid_for_reconstruction(
        particle_positions,
        parameters.particle_radius,
        parameters.compact_support_radius,
        parameters.cube_size,
        parameters.particle_aabb.as_ref(),
        parameters.enable_multi_threading,
    )?;

    output_surface.grid.log_grid_info();

    match &parameters.spatial_decomposition {
        Some(SpatialDecomposition::UniformGrid(_)) => {
            reconstruction::reconstruct_surface_subdomain_grid::<I, R>(
                particle_positions,
                parameters,
                output_surface,
            )?
        }
        None => reconstruction::reconstruct_surface_global(
            particle_positions,
            parameters,
            output_surface,
        )?,
    }

    // Put back temporary storage for filtered particles for next reconstruction
    if let Cow::Owned(mut filtered_particles) = filtered_particle_positions {
        filtered_particles.clear();
        *output_surface.workspace.filtered_particles_mut() = filtered_particles;
    }

    Ok(())
}

/// Constructs the background grid for marching cubes based on the parameters supplied to the surface reconstruction
pub fn grid_for_reconstruction<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_radius: R,
    compact_support_radius: R,
    cube_size: R,
    particle_aabb: Option<&Aabb3d<R>>,
    enable_multi_threading: bool,
) -> Result<UniformGrid<I, R>, ReconstructionError<I, R>> {
    let mut particle_aabb = if let Some(particle_aabb) = particle_aabb {
        particle_aabb.clone()
    } else {
        profile!("compute minimum enclosing aabb");

        let particle_aabb = {
            let mut aabb = if enable_multi_threading {
                Aabb3d::par_from_points(particle_positions)
            } else {
                Aabb3d::from_points(particle_positions)
            };
            // TODO: Is this really necessary? This seems unnecessary purely for the density map...
            aabb.grow_uniformly(particle_radius);
            aabb
        };

        info!(
            "Bounding box of particles with margin for levelset evaluation: {:?} to {:?}",
            particle_aabb.min().as_slice(),
            particle_aabb.max().as_slice()
        );

        particle_aabb
    };

    // Ensure that we have enough margin around the particles such that the every particle's kernel support is completely in the domain
    let kernel_margin =
        density_map::compute_kernel_evaluation_radius::<I, R>(compact_support_radius, cube_size)
            .kernel_evaluation_radius;
    particle_aabb.grow_uniformly(kernel_margin);

    Ok(UniformGrid::from_aabb(&particle_aabb, cube_size)?)
}
