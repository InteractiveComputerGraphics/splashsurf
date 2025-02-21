//! Computation of sparse density maps (evaluation of particle densities and mapping onto sparse grids)
//!
//! This module provides functions for the computation of per-particle densities and the discretization
//! of the resulting fluid density field by mapping onto a discrete background grid.
//!
//! Currently, only sparse density maps are implemented.
//!
//! ## Sparse density maps
//! The [`DensityMap`] stores fluid density values for each point of an implicit background grid
//! where the density is not trivially zero. This is the case for all points that are inside or at
//! least within some tolerance to the compact support radius of a particle.
//! In case of a sparse density map, the values are stored in a hashmap. The keys are so called
//! "flat point indices". These are computed from the background grid point coordinates `(i,j,k)`
//! analogous to multidimensional array index flattening. That means for a grid with dimensions
//! `[n_x, n_y, n_z]`, the flat point index is given by the expression `i*n_x + j*n_y + k*n_z`.
//! For these point index operations, the [`UniformGrid`] is used.
//!
//! Note that all density mapping functions always use the global background grid for flat point
//! indices, even if the density map is only generated for a smaller subdomain.

use crate::aabb::Aabb3d;
use crate::kernel::DiscreteSquaredDistanceCubicKernel;
use crate::mesh::{HexMesh3d, MeshAttribute, MeshWithData};
use crate::neighborhood_search::NeighborhoodList;
use crate::uniform_grid::UniformGrid;
use crate::utils::{ChunkSize, ParallelPolicy};
use crate::{HashState, Index, MapType, ParallelMapType, Real, new_map, profile};
use dashmap::ReadOnlyView as ReadDashMap;
use log::{info, trace, warn};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::cell::RefCell;
use thiserror::Error as ThisError;
use thread_local::ThreadLocal;

// TODO: Document formulas for the computation of the values
// TODO: Document that we actually evaluate the SPH interpolation of the constant function f(x) = 1

/// Errors that can occur during generation of the density map
#[derive(Debug, ThisError)]
pub enum DensityMapError<R: Real> {
    /// Indicates that domain for the density map is inconsistent or degenerate
    ///
    /// For the density map computation the user specified domain is shrunk ensuring that all
    /// remaining particles only influence grid points on the interior of this domain. If the initial
    /// user specified domain is too small, this can result in an inconsistent or degenerate domain.
    #[error("the adapted subdomain for the density map is inconsistent/degenerate")]
    InvalidDomain {
        /// The margin by which the user specified domain is shrunk
        margin: R,
        /// The final (invalid) domain after the margin is applied to the user specified domain
        domain: Aabb3d<R>,
    },
}

/// Computes the individual densities of particles using a standard SPH sum
#[inline(never)]
pub fn compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    compact_support_radius: R,
    particle_rest_mass: R,
    enable_multi_threading: bool,
) -> Vec<R> {
    let mut densities = Vec::new();
    if enable_multi_threading {
        parallel_compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists,
            compact_support_radius,
            particle_rest_mass,
            &mut densities,
        )
    } else {
        sequential_compute_particle_densities::<I, R, _>(
            particle_positions,
            particle_neighbor_lists,
            compact_support_radius,
            particle_rest_mass,
            &mut densities,
        )
    }
    densities
}

/// Computes the individual densities of particles inplace using a standard SPH sum
#[inline(never)]
pub fn compute_particle_densities_inplace<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    compact_support_radius: R,
    particle_rest_mass: R,
    enable_multi_threading: bool,
    densities: &mut Vec<R>,
) {
    if enable_multi_threading {
        parallel_compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists,
            compact_support_radius,
            particle_rest_mass,
            densities,
        )
    } else {
        sequential_compute_particle_densities::<I, R, _>(
            particle_positions,
            particle_neighbor_lists,
            compact_support_radius,
            particle_rest_mass,
            densities,
        )
    }
}

fn init_density_storage<R: Real>(densities: &mut Vec<R>, new_len: usize) {
    // Ensure that length is correct
    densities.resize(new_len, R::zero());
    // Existing values don't have to be set to zero, as they are overwritten later anyway
}

/// Computes the individual densities of particles using a standard SPH sum, sequential implementation
#[inline(never)]
pub fn sequential_compute_particle_densities<I: Index, R: Real, Nl: NeighborhoodList + ?Sized>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &Nl,
    compact_support_radius: R,
    particle_rest_mass: R,
    particle_densities: &mut Vec<R>,
) {
    profile!("sequential_compute_particle_densities");

    sequential_compute_particle_densities_filtered::<I, R, Nl>(
        particle_positions,
        particle_neighbor_lists,
        compact_support_radius,
        particle_rest_mass,
        particle_densities,
        |_| true,
    )
}

#[inline(never)]
pub fn sequential_compute_particle_densities_filtered<
    I: Index,
    R: Real,
    Nl: NeighborhoodList + ?Sized,
>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &Nl,
    compact_support_radius: R,
    particle_rest_mass: R,
    particle_densities: &mut Vec<R>,
    filter: impl Fn(usize) -> bool,
) {
    profile!("sequential_compute_particle_densities_filtered");

    init_density_storage(particle_densities, particle_positions.len());

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, compact_support_radius);

    for (i, particle_i_position) in particle_positions
        .iter()
        .enumerate()
        .filter(|(i, _)| filter(*i))
    {
        let mut particle_i_density = kernel.evaluate(R::zero());
        for particle_j_position in particle_neighbor_lists
            .neighbors(i)
            .iter()
            .map(|&j| &particle_positions[j])
        {
            let r_squared = (particle_j_position - particle_i_position).norm_squared();
            particle_i_density += kernel.evaluate(r_squared);
        }
        particle_i_density *= particle_rest_mass;
        particle_densities[i] = particle_i_density;
    }
}

/// Computes the individual densities of particles using a standard SPH sum, multi-threaded implementation
#[inline(never)]
pub fn parallel_compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    compact_support_radius: R,
    particle_rest_mass: R,
    particle_densities: &mut Vec<R>,
) {
    profile!("parallel_compute_particle_densities");

    init_density_storage(particle_densities, particle_positions.len());

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, compact_support_radius);

    particle_positions
        .par_iter()
        .with_min_len(8)
        .zip_eq(particle_neighbor_lists.par_iter())
        .zip_eq(particle_densities.par_iter_mut())
        .for_each(
            |((particle_i_position, particle_i_neighbors), particle_i_density)| {
                let mut density = kernel.evaluate(R::zero());
                for particle_j_position in
                    particle_i_neighbors.iter().map(|&j| &particle_positions[j])
                {
                    let r_squared = (particle_j_position - particle_i_position).norm_squared();
                    density += kernel.evaluate(r_squared);
                }
                density *= particle_rest_mass;
                *particle_i_density = density;
            },
        );
}

/// A sparse density map
///
/// The density map contains values for all points of the background grid where the density is not
/// trivially zero (which is the case when a point is outside of the compact support of any particles).
#[derive(Clone, Debug)]
pub enum DensityMap<I: Index, R: Real> {
    Standard(MapType<I, R>),
    DashMap(ReadDashMap<I, R, HashState>),
}

impl<I: Index, R: Real> Default for DensityMap<I, R> {
    fn default() -> Self {
        DensityMap::Standard(MapType::default())
    }
}

impl<I: Index, R: Real> From<MapType<I, R>> for DensityMap<I, R> {
    fn from(map: MapType<I, R>) -> Self {
        Self::Standard(map)
    }
}

impl<I: Index, R: Real> From<ParallelMapType<I, R>> for DensityMap<I, R> {
    fn from(map: ParallelMapType<I, R>) -> Self {
        Self::DashMap(map.into_read_only())
    }
}

impl<I: Index, R: Real> DensityMap<I, R> {
    /// Converts the contained map into a vector of tuples of (flat_point_index, density)
    pub fn to_vec(&self) -> Vec<(I, R)> {
        match self {
            DensityMap::Standard(map) => map.iter().map(|(&i, &r)| (i, r)).collect(),
            DensityMap::DashMap(map) => map.iter().map(|(&i, &r)| (i, r)).collect(),
        }
    }

    /// Returns the number of density entries
    pub fn len(&self) -> usize {
        match self {
            DensityMap::Standard(map) => map.len(),
            DensityMap::DashMap(map) => map.len(),
        }
    }

    /// Returns the density value at the specified flat point index
    pub fn get(&self, flat_point_index: I) -> Option<R> {
        match self {
            DensityMap::Standard(map) => map.get(&flat_point_index).copied(),
            DensityMap::DashMap(map) => map.get(&flat_point_index).copied(),
        }
    }

    /// Calls a closure for each `(flat_point_index, density_value)` tuple in the map
    pub fn for_each<F: FnMut(I, R)>(&self, f: F) {
        let mut f = f;
        match self {
            DensityMap::Standard(map) => map.iter().for_each(|(&i, &r)| f(i, r)),
            DensityMap::DashMap(map) => map.iter().for_each(|(&i, &r)| f(i, r)),
        }
    }
}

/// Computes a sparse density map for the fluid based on the specified background grid
#[inline(never)]
pub fn generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    compact_support_radius: R,
    cube_size: R,
    allow_threading: bool,
    density_map: &mut DensityMap<I, R>,
) -> Result<(), DensityMapError<R>> {
    trace!(
        "Starting construction of sparse density map... (Input: {} particles)",
        if let Some(active_particles) = active_particles {
            active_particles.len()
        } else {
            particle_positions.len()
        }
    );

    if allow_threading {
        *density_map = parallel_generate_sparse_density_map(
            grid,
            particle_positions,
            particle_densities,
            active_particles,
            particle_rest_mass,
            compact_support_radius,
            cube_size,
        )?
    } else {
        *density_map = sequential_generate_sparse_density_map(
            grid,
            particle_positions,
            particle_densities,
            active_particles,
            particle_rest_mass,
            compact_support_radius,
            cube_size,
        )?
    }

    trace!(
        "Sparse density map was constructed. (Output: density map with {} grid point data entries)",
        density_map.len()
    );

    Ok(())
}

/// Computes a sparse density map for the fluid based on the specified background grid, sequential implementation
#[inline(never)]
pub fn sequential_generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    compact_support_radius: R,
    cube_size: R,
) -> Result<DensityMap<I, R>, DensityMapError<R>> {
    profile!("sequential_generate_sparse_density_map");

    let mut sparse_densities = new_map();

    let density_map_generator = SparseDensityMapGenerator::try_new(
        grid,
        compact_support_radius,
        cube_size,
        particle_rest_mass,
    )?;

    let process_particle = |particle_data: (&Vector3<R>, R)| {
        let (particle, particle_density) = particle_data;
        density_map_generator.compute_particle_density_contribution(
            grid,
            &mut sparse_densities,
            particle,
            particle_density,
        );
    };

    match active_particles {
        None => particle_positions
            .iter()
            .zip(particle_densities.iter().copied())
            .for_each(process_particle),
        Some(indices) => indices
            .iter()
            .map(|&i| &particle_positions[i])
            .zip(indices.iter().map(|&i| particle_densities[i]))
            .for_each(process_particle),
    }

    Ok(sparse_densities.into())
}

/// Computes a sparse density map for the fluid based on the specified background grid, multi-threaded implementation
#[inline(never)]
pub fn parallel_generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    compact_support_radius: R,
    cube_size: R,
) -> Result<DensityMap<I, R>, DensityMapError<R>> {
    profile!("parallel_generate_sparse_density_map");

    // Each thread will write to its own local density map
    let sparse_densities: ThreadLocal<RefCell<MapType<I, R>>> = ThreadLocal::new();

    // Generate thread local density maps
    {
        let density_map_generator = SparseDensityMapGenerator::try_new(
            grid,
            compact_support_radius,
            cube_size,
            particle_rest_mass,
        )?;

        profile!("generate thread local maps");

        match active_particles {
            // Process particles, when no list of active particles was provided
            None => {
                let chunk_size =
                    ChunkSize::new(&ParallelPolicy::default(), particle_positions.len())
                        .with_log("particles", "density map generation")
                        .chunk_size;

                particle_positions
                    .par_chunks(chunk_size)
                    .zip(particle_densities.par_chunks(chunk_size))
                    .for_each(|(position_chunk, density_chunk)| {
                        // Obtain mutable reference to thread local density map
                        let map = sparse_densities
                            .get_or(|| RefCell::new(MapType::with_hasher(HashState::default())));
                        let mut mut_map = map.borrow_mut();

                        let process_particle_map = |particle_data: (&Vector3<R>, R)| {
                            let (particle, particle_density) = particle_data;
                            density_map_generator.compute_particle_density_contribution(
                                grid,
                                &mut mut_map,
                                particle,
                                particle_density,
                            );
                        };

                        assert_eq!(position_chunk.len(), density_chunk.len());
                        position_chunk
                            .iter()
                            .zip(density_chunk.iter().copied())
                            .for_each(process_particle_map);
                    })
            }
            // Process particles, when only a subset is active
            Some(indices) => {
                let chunk_size = ChunkSize::new(&ParallelPolicy::default(), indices.len())
                    .with_log("active particles", "density map generation")
                    .chunk_size;

                indices.par_chunks(chunk_size).for_each(|index_chunk| {
                    // Obtain mutable reference to thread local density map
                    let map = sparse_densities
                        .get_or(|| RefCell::new(MapType::with_hasher(HashState::default())));
                    let mut mut_map = map.borrow_mut();

                    let process_particle_map = |particle_data: (&Vector3<R>, R)| {
                        let (particle, particle_density) = particle_data;
                        density_map_generator.compute_particle_density_contribution(
                            grid,
                            &mut mut_map,
                            particle,
                            particle_density,
                        );
                    };

                    index_chunk
                        .iter()
                        .map(|&i| (&particle_positions[i], particle_densities[i]))
                        .for_each(process_particle_map);
                });
            }
        }
    }

    // Merge the thread local density maps
    {
        profile!("merge thread local maps to global map");

        // Collect all thread local maps into a single vec
        let mut local_density_maps = sparse_densities
            .into_iter()
            .map(|m| m.into_inner())
            .collect::<Vec<_>>();

        info!(
            "Merging {} thread local density maps to a single global map...",
            local_density_maps.len()
        );

        // Merge local density maps in parallel by summing the density contributions
        let global_density_map = ParallelMapType::with_hasher(HashState::default());
        local_density_maps.par_iter_mut().for_each(|local_map| {
            for (idx, density) in local_map.drain() {
                *global_density_map.entry(idx).or_insert(R::zero()) += density;
            }
        });

        Ok(global_density_map.into())
    }
}

/// Internal helper type used to evaluate the density contribution for a particle
struct SparseDensityMapGenerator<I: Index, R: Real> {
    particle_rest_mass: R,
    half_supported_cells: I,
    supported_points: I,
    kernel_evaluation_radius_sq: R,
    kernel: DiscreteSquaredDistanceCubicKernel<R>,
    allowed_domain: Aabb3d<R>,
}

pub(crate) struct GridKernelExtents<I: Index, R: Real> {
    // The number of cells in each direction from a particle's cell that can be affected by its compact support
    pub half_supported_cells: I,
    // The total number of points per dimension that can be affected by a particle's compact support
    pub supported_points: I,
    // The resulting maximum kernel evaluation radius (more than the kernel compact support)
    pub kernel_evaluation_radius: R,
}

pub(crate) fn compute_kernel_evaluation_radius<I: Index, R: Real>(
    compact_support_radius: R,
    cube_size: R,
) -> GridKernelExtents<I, R> {
    // The number of cells in each direction from a particle that can be affected by its compact support
    let half_supported_cells_real = (compact_support_radius / cube_size).ceil();
    // Convert to index type for cell and point indexing
    let half_supported_cells: I = half_supported_cells_real.to_index_unchecked();

    // The total number of cells per dimension that can be affected by a particle's compact support
    let supported_cells: I = half_supported_cells.times(2) + I::one();
    // The number of points corresponding to the number of supported cells
    let supported_points: I = I::one() + supported_cells;

    // Evaluate kernel in a smaller domain, points outside of this radius have to be assumed to be outside the iso-surface
    let kernel_evaluation_radius =
        cube_size * half_supported_cells_real * (R::one() + R::default_epsilon().sqrt());

    GridKernelExtents {
        half_supported_cells,
        supported_points,
        kernel_evaluation_radius,
    }
}

// TODO: Maybe remove allowed domain check? And require this is done before, using the active_particles array?
impl<I: Index, R: Real> SparseDensityMapGenerator<I, R> {
    fn try_new(
        grid: &UniformGrid<I, R>,
        compact_support_radius: R,
        cube_size: R,
        particle_rest_mass: R,
    ) -> Result<Self, DensityMapError<R>> {
        let GridKernelExtents {
            half_supported_cells,
            supported_points,
            kernel_evaluation_radius,
        } = compute_kernel_evaluation_radius(compact_support_radius, cube_size);

        // Pre-compute the kernel which can be queried using squared distances
        let kernel_evaluation_radius_sq = kernel_evaluation_radius * kernel_evaluation_radius;
        let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, compact_support_radius);

        // Shrink the allowed domain for particles by the kernel evaluation radius. This ensures that all cells/points
        // that are affected by a particle are actually part of the domain/grid, so it does not have to be checked in the loops below.
        // However, any particles inside of this margin, close to the border of the originally requested domain will be ignored.
        //
        // This also implies that this density map should always represent a closed surfaces.
        // If particles were closer to the AABB boundary than this margin, there could be holes in the resulting level-set.
        let allowed_domain = {
            let mut aabb = grid.aabb().clone();
            aabb.grow_uniformly(kernel_evaluation_radius.neg());
            aabb
        };

        if allowed_domain.is_degenerate() || !allowed_domain.is_consistent() {
            warn!(
                "The allowed domain of particles for a subdomain is inconsistent/degenerate: {:?}",
                allowed_domain
            );
            warn!(
                "No particles can be found in this domain. Increase the domain of the surface reconstruction to avoid this."
            );
            Err(DensityMapError::InvalidDomain {
                margin: kernel_evaluation_radius,
                domain: allowed_domain,
            })
        } else {
            Ok(Self {
                half_supported_cells,
                supported_points,
                kernel_evaluation_radius_sq,
                kernel,
                allowed_domain,
                particle_rest_mass,
            })
        }
    }

    /// Computes all density contributions of a particle to the background grid into the given map
    fn compute_particle_density_contribution(
        &self,
        grid: &UniformGrid<I, R>,
        sparse_densities: &mut MapType<I, R>,
        particle: &Vector3<R>,
        particle_density: R,
    ) {
        // Skip particles outside allowed domain
        if !self.allowed_domain.contains_point(particle) {
            return;
        }

        // Compute grid points affected by the particle
        let min_supported_point_ijk = {
            let cell_ijk = grid.enclosing_cell(particle);
            [
                cell_ijk[0] - self.half_supported_cells,
                cell_ijk[1] - self.half_supported_cells,
                cell_ijk[2] - self.half_supported_cells,
            ]
        };

        let max_supported_point_ijk = [
            min_supported_point_ijk[0] + self.supported_points,
            min_supported_point_ijk[1] + self.supported_points,
            min_supported_point_ijk[2] + self.supported_points,
        ];

        self.particle_support_loop(
            sparse_densities,
            grid,
            &min_supported_point_ijk,
            &max_supported_point_ijk,
            particle,
            particle_density,
        );
    }

    /// Loops over a cube of background grid points that are potentially in the support radius of the particle and evaluates density contributions
    #[inline(always)]
    fn particle_support_loop(
        &self,
        sparse_densities: &mut MapType<I, R>,
        grid: &UniformGrid<I, R>,
        min_supported_point_ijk: &[I; 3],
        max_supported_point_ijk: &[I; 3],
        particle: &Vector3<R>,
        particle_density: R,
    ) {
        // Compute the volume of this particle
        let particle_volume = self.particle_rest_mass / particle_density;

        // TODO: Check performance with just using multiplication
        let min_supported_point = grid.point_coordinates_array(min_supported_point_ijk);

        // dx, dy, dz are the deltas of the supported points as seen from the current particle position
        let mut dx = min_supported_point[0] - particle[0]
            // Subtract cell size because it will be added in the beginning of each loop iteration
            // this is done to avoid multiplications
            - grid.cell_size();

        // A range loop cannot be used here because the Step trait is unstable
        // but it is required for the Iter impl on Range
        // therefore a manual while loop has to be used

        // Loop over all points that might receive a density contribution from this particle
        let mut i = min_supported_point_ijk[0];
        while i != max_supported_point_ijk[0] {
            dx += grid.cell_size();
            let dxdx = dx * dx;

            let mut dy = min_supported_point[1] - particle[1] - grid.cell_size();
            let mut j = min_supported_point_ijk[1];
            while j != max_supported_point_ijk[1] {
                dy += grid.cell_size();
                let dydy = dy * dy;

                let mut dz = min_supported_point[2] - particle[2] - grid.cell_size();
                let mut k = min_supported_point_ijk[2];
                while k != max_supported_point_ijk[2] {
                    dz += grid.cell_size();
                    let dzdz = dz * dz;

                    let r_squared = dxdx + dydy + dzdz;
                    if r_squared < self.kernel_evaluation_radius_sq {
                        let density_contribution =
                            particle_volume * self.kernel.evaluate(r_squared);

                        let flat_point_index = grid.flatten_point_indices(i, j, k);
                        *sparse_densities
                            .entry(flat_point_index)
                            .or_insert(R::zero()) += density_contribution;
                    }
                    k += I::one();
                }
                j += I::one();
            }
            i += I::one();
        }
    }
}

/// Converts a sparse density map (based on the implicit background grid) to a sparse hexahedral mesh with explicit coordinates for the cells' vertices.
#[inline(never)]
pub fn sparse_density_map_to_hex_mesh<I: Index, R: Real>(
    density_map: &DensityMap<I, R>,
    grid: &UniformGrid<I, R>,
    default_value: R,
) -> MeshWithData<R, HexMesh3d<R>> {
    profile!("sparse_density_map_to_hex_mesh");

    let mut mesh = HexMesh3d {
        vertices: Vec::new(),
        cells: Vec::new(),
    };
    let mut values = Vec::new();
    let mut cells = new_map();

    // Create vertices and cells for points with values
    density_map.for_each(|flat_point_index, point_value| {
        let point = grid.try_unflatten_point_index(flat_point_index).unwrap();
        let point_coords = grid.point_coordinates(&point);

        // Create vertex
        let vertex_index = mesh.vertices.len();
        mesh.vertices.push(point_coords);
        values.push(point_value);

        // Iterate over all cells that are adjacent to the point and store vertex index
        let neighborhood = grid.get_point_neighborhood(&point);
        for cell in grid.cells_adjacent_to_point(&neighborhood).iter().flatten() {
            let flat_cell_index = grid.flatten_cell_index(cell);

            let cell_connectivity_entry = cells
                .entry(flat_cell_index)
                .or_insert_with(|| [None, None, None, None, None, None, None, None]);

            let local_point_index = cell.local_point_index_of(point.index()).unwrap();
            cell_connectivity_entry[local_point_index] = Some(vertex_index);
        }
    });

    // Add missing vertices of cells using default values
    let mut additional_vertices = new_map();
    for (flat_cell_index, cell_vertices) in cells.iter_mut() {
        let cell = grid.try_unflatten_cell_index(*flat_cell_index).unwrap();

        for (local_point_index, vertex) in cell_vertices.iter_mut().enumerate() {
            if vertex.is_none() {
                // Map local point index to global index in grid
                let point = cell.global_point_index_of(local_point_index).unwrap();
                let flat_point_index = grid.flatten_point_index(&point);

                // Try to lookup the vertex associated with the point or create a new one with default value
                let vertex_entry =
                    additional_vertices
                        .entry(flat_point_index)
                        .or_insert_with(|| {
                            let point_coords = grid.point_coordinates(&point);
                            let vertex_index = mesh.vertices.len();
                            mesh.vertices.push(point_coords);
                            values.push(default_value);

                            vertex_index
                        });

                *vertex = Some(*vertex_entry);
            }
        }
    }

    // Add all cells to the mesh
    mesh.cells.reserve(cells.len());
    for (_, cell_vertices) in cells.iter() {
        mesh.cells.push([
            cell_vertices[0].unwrap(),
            cell_vertices[1].unwrap(),
            cell_vertices[2].unwrap(),
            cell_vertices[3].unwrap(),
            cell_vertices[4].unwrap(),
            cell_vertices[5].unwrap(),
            cell_vertices[6].unwrap(),
            cell_vertices[7].unwrap(),
        ]);
    }

    MeshWithData::new(mesh).with_point_data(MeshAttribute::new_real_scalar(
        "density".to_string(),
        values,
    ))
}
