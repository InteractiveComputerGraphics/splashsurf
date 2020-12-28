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
/// Types related to the virtual background grid used for marching cubes
mod uniform_grid;
mod utils;

pub use aabb::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d};
pub use density_map::DensityMap;
pub use numeric_types::{Index, Real, ThreadSafe};
pub use uniform_grid::{GridConstructionError, UniformGrid};

use log::info;
use mesh::TriMesh3d;
use nalgebra::Vector3;
use thiserror::Error as ThisError;

// TODO: Add documentation of feature flags
// TODO: Add documentation of the parameter struct

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

/// Parameters for the surface reconstruction
#[derive(Clone, Debug)]
pub struct Parameters<R: Real> {
    pub particle_radius: R,
    pub rest_density: R,
    pub kernel_radius: R,
    pub splash_detection_radius: Option<R>,
    pub cube_size: R,
    pub iso_surface_threshold: R,
    pub domain_aabb: Option<AxisAlignedBoundingBox3d<R>>,
    pub enable_multi_threading: bool,
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
        })
    }
}

/// Result data returned when the surface reconstruction was successful
#[derive(Clone, Debug)]
pub struct SurfaceReconstruction<I: Index, R: Real> {
    /// The background grid that was used as a basis for generating the density map for marching cubes
    grid: UniformGrid<I, R>,
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
    fn from(error: GridConstructionError<I, R>) -> Self {
        ReconstructionError::GridConstructionError(error)
    }
}

impl<I: Index, R: Real> From<anyhow::Error> for ReconstructionError<I, R> {
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

pub fn reconstruct_surface_inplace<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_inplace");

    let Parameters {
        particle_radius,
        rest_density,
        kernel_radius,
        splash_detection_radius,
        cube_size,
        iso_surface_threshold,
        domain_aabb,
        enable_multi_threading,
    } = parameters.clone();

    surface.grid = grid_for_reconstruction(
        particle_positions,
        particle_radius,
        cube_size,
        domain_aabb.as_ref(),
    )?;
    let grid = &surface.grid;

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

    let particle_rest_density = rest_density;
    let particle_rest_volume =
        R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap() * particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    let particle_densities = {
        info!("Starting neighborhood search...");

        let particle_neighbor_lists = neighborhood_search::search::<I, R>(
            &grid.aabb(),
            particle_positions,
            kernel_radius,
            enable_multi_threading,
        );

        info!("Computing particle densities...");

        density_map::compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists.as_slice(),
            kernel_radius,
            particle_rest_mass,
            enable_multi_threading,
        )
    };

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

    let density_map = density_map::generate_sparse_density_map::<I, R>(
        &grid,
        particle_positions,
        particle_densities.as_slice(),
        particle_indices.as_ref().map(|is| is.as_slice()),
        particle_rest_mass,
        kernel_radius,
        cube_size,
        enable_multi_threading,
    );

    marching_cubes::triangulate_density_map::<I, R>(
        &grid,
        &density_map,
        iso_surface_threshold,
        &mut surface.mesh,
    );

    surface.density_map = Some(density_map);

    Ok(())
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
