extern crate nalgebra as na;

mod aabb;
pub mod density_map;
pub mod kernel;
pub mod marching_cubes;
mod marching_cubes_lut;
pub mod mesh;
pub mod neighborhood_search;
mod numeric_types;
mod uniform_grid;
mod utils;

pub use aabb::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d};
pub use numeric_types::{Index, Real};
pub use uniform_grid::{GridConstructionError, UniformGrid};

use coarse_prof::profile;
use log::info;
use mesh::TriMesh3d;
use na::Vector3;
use thiserror::Error as ThisError;

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

// Switch to BTreeMap in debug mode for easier debugging due to deterministic iteration order
#[cfg(debug_assertions)]
pub(crate) type MapType<K, V> = std::collections::BTreeMap<K, V>;
#[cfg(not(debug_assertions))]
pub(crate) type MapType<K, V> = std::collections::HashMap<K, V>;

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
    pub allow_multi_threading: bool,
}

macro_rules! map_option {
    ($some_optional:expr, $value_identifier:ident => $value_transformation:expr) => {
        match $some_optional {
            Some($value_identifier) => Some($value_transformation),
            None => None,
        }
    };
}

impl<R: Real> Parameters<R> {
    /// Tries to convert the parameters from one real type to another real type, returns None if conversion fails
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
            allow_multi_threading: self.allow_multi_threading,
        })
    }
}

/// Data returned by when surface reconstruction was successful
#[derive(Clone, Debug)]
pub struct SurfaceReconstruction<I: Index, R: Real> {
    /// The background grid that was used as a basis for generating the density map for marching cubes
    grid: UniformGrid<I, R>,
    /// The point-based density map generated from the particles that was used as input to marching cubes
    point_data: MapType<I, R>,
    /// The actual mesh that is the result of the surface reconstruction
    mesh: TriMesh3d<R>,
}

impl<I: Index, R: Real> SurfaceReconstruction<I, R> {
    /// Returns a reference to the background grid that was used as a basis for generating the density map for marching cubes
    pub fn grid(&self) -> &UniformGrid<I, R> {
        &self.grid
    }

    /// Returns a reference to the point-based density map generated from the particles that was used as input to marching cubes
    pub fn point_data(&self) -> &MapType<I, R> {
        &self.point_data
    }

    /// Returns a reference to the actual mesh that is the result of the surface reconstruction
    pub fn mesh(&self) -> &TriMesh3d<R> {
        &self.mesh
    }
}

impl<I: Index, R: Real> From<SurfaceReconstruction<I, R>> for TriMesh3d<R> {
    fn from(result: SurfaceReconstruction<I, R>) -> Self {
        result.mesh
    }
}

/// Error type returned when surface reconstruction fails
#[non_exhaustive]
#[derive(Debug, ThisError)]
pub enum ReconstructionError<I: Index, R: Real> {
    #[error("grid construction: {0}")]
    GridConstructionError(GridConstructionError<I, R>),
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

/// Runs a marching cubes surface construction over the given particle positions
#[inline(never)]
pub fn reconstruct_surface<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
) -> Result<SurfaceReconstruction<I, R>, ReconstructionError<I, R>> {
    profile!("reconstruct_surface");

    let Parameters {
        particle_radius,
        rest_density,
        kernel_radius,
        splash_detection_radius,
        cube_size,
        iso_surface_threshold,
        domain_aabb,
        allow_multi_threading,
    } = parameters.clone();

    let grid = grid_for_reconstruction(
        particle_positions,
        particle_radius,
        cube_size,
        domain_aabb.as_ref(),
    )?;

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
            allow_multi_threading,
        );

        info!("Computing particle densities...");

        density_map::compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists.as_slice(),
            kernel_radius,
            particle_rest_mass,
            allow_multi_threading,
        )
    };

    let particle_indices = splash_detection_radius.map(|splash_detection_radius| {
        let neighborhood_list = neighborhood_search::search::<I, R>(
            &grid.aabb(),
            particle_positions,
            splash_detection_radius,
            allow_multi_threading,
        );

        let mut active_particles = Vec::new();
        for (particle_i, neighbors) in neighborhood_list.iter().enumerate() {
            if !neighbors.is_empty() {
                active_particles.push(particle_i);
            }
        }

        active_particles
    });

    let point_data = density_map::generate_sparse_density_map::<I, R>(
        &grid,
        particle_positions,
        particle_densities.as_slice(),
        particle_indices.as_ref().map(|is| is.as_slice()),
        particle_rest_mass,
        kernel_radius,
        cube_size,
    );

    let mesh =
        marching_cubes::triangulate_point_data::<I, R>(&grid, &point_data, iso_surface_threshold);

    Ok(SurfaceReconstruction {
        grid,
        point_data,
        mesh,
    })
}

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
