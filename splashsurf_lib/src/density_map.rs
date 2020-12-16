use std::cell::RefCell;

use dashmap::ReadOnlyView as ReadDashMap;
use log::{info, warn};
use nalgebra::Vector3;
use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::kernel::DiscreteSquaredDistanceCubicKernel;
use crate::mesh::{HexMesh3d, MeshWithPointData};
use crate::uniform_grid::UniformGrid;
use crate::{new_map, HashState, Index, MapType, ParallelMapType, Real};

/// Computes the individual densities of particles using a standard SPH sum
#[inline(never)]
pub fn compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    kernel_radius: R,
    particle_rest_mass: R,
    enable_multi_threading: bool,
) -> Vec<R> {
    if enable_multi_threading {
        parallel_compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists,
            kernel_radius,
            particle_rest_mass,
        )
    } else {
        sequential_compute_particle_densities::<I, R>(
            particle_positions,
            particle_neighbor_lists,
            kernel_radius,
            particle_rest_mass,
        )
    }
}

/// Computes the individual densities of particles using a standard SPH sum, sequential implementation
#[inline(never)]
pub fn sequential_compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    kernel_radius: R,
    particle_rest_mass: R,
) -> Vec<R> {
    profile!("sequential_compute_particle_densities");

    let mut particle_densities = vec![R::zero(); particle_positions.len()];

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new(1000, kernel_radius);

    for (i, (particle_i_position, particle_i_neighbors)) in particle_positions
        .iter()
        .zip(particle_neighbor_lists.iter())
        .enumerate()
    {
        let mut particle_i_density = kernel.evaluate(R::zero());
        for particle_j_position in particle_i_neighbors.iter().map(|&j| &particle_positions[j]) {
            let r_squared = (particle_j_position - particle_i_position).norm_squared();
            particle_i_density += kernel.evaluate(r_squared);
        }
        particle_i_density *= particle_rest_mass;
        particle_densities[i] = particle_i_density;
    }

    particle_densities
}

/// Computes the individual densities of particles using a standard SPH sum, multi-threaded implementation
#[inline(never)]
pub fn parallel_compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    kernel_radius: R,
    particle_rest_mass: R,
) -> Vec<R> {
    profile!("parallel_compute_particle_densities");

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new(1000, kernel_radius);

    let particle_densities = particle_positions
        .par_iter()
        .zip_eq(particle_neighbor_lists.par_iter())
        .map(|(particle_i_position, particle_i_neighbors)| {
            let mut particle_i_density = kernel.evaluate(R::zero());
            for particle_j_position in particle_i_neighbors.iter().map(|&j| &particle_positions[j])
            {
                let r_squared = (particle_j_position - particle_i_position).norm_squared();
                particle_i_density += kernel.evaluate(r_squared);
            }
            particle_i_density *= particle_rest_mass;
            particle_i_density
        })
        .collect();

    particle_densities
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
    pub fn to_vec(&self) -> Vec<(I, R)> {
        match self {
            DensityMap::Standard(map) => map.iter().map(|(&i, &r)| (i, r)).collect(),
            DensityMap::DashMap(map) => map.iter().map(|(&i, &r)| (i, r)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DensityMap::Standard(map) => map.len(),
            DensityMap::DashMap(map) => map.len(),
        }
    }

    pub fn get(&self, flat_point_index: I) -> Option<R> {
        match self {
            DensityMap::Standard(map) => map.get(&flat_point_index).copied(),
            DensityMap::DashMap(map) => map.get(&flat_point_index).copied(),
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (I, R)> + 'a {
        let iter_standard = std::iter::once(if let DensityMap::Standard(map) = self {
            Some(map)
        } else {
            None
        })
        .flatten()
        .flat_map(|map| map.iter())
        .map(|(&i, &r)| (i, r));

        let iter_dash = std::iter::once(if let DensityMap::DashMap(map) = self {
            Some(map)
        } else {
            None
        })
        .flatten()
        .flat_map(|map| map.iter())
        .map(|(&i, &r)| (i, r));

        iter_standard.chain(iter_dash)
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
    kernel_radius: R,
    cube_size: R,
    allow_threading: bool,
) -> DensityMap<I, R> {
    // TODO: Reduce code duplication between these two functions
    if allow_threading {
        parallel_generate_sparse_density_map(
            grid,
            particle_positions,
            particle_densities,
            active_particles,
            particle_rest_mass,
            kernel_radius,
            cube_size,
        )
    } else {
        sequential_generate_sparse_density_map(
            grid,
            particle_positions,
            particle_densities,
            active_particles,
            particle_rest_mass,
            kernel_radius,
            cube_size,
        )
    }
}

/// Computes a sparse density map for the fluid based on the specified background grid, sequential implementation
#[inline(never)]
pub fn sequential_generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    kernel_radius: R,
    cube_size: R,
) -> DensityMap<I, R> {
    profile!("sequential_generate_sparse_density_map");

    info!(
        "Starting construction of sparse density map for {} particles...",
        if let Some(active_particles) = active_particles {
            active_particles.len()
        } else {
            particle_positions.len()
        }
    );

    let mut sparse_densities = new_map();

    // The number of cells in each direction from a particle that can be affected by a its compact support
    let half_supported_cells_real = (kernel_radius / cube_size).ceil();
    // Convert to index type for cell and point indexing
    let half_supported_cells: I = half_supported_cells_real.to_index_unchecked();
    // The total number of points per dimension that can be affected by a particle's compact support
    //  + account for an additional layer of points that cover the positive outside of the supported cells
    let supported_points = I::one() + half_supported_cells.times(2);

    let kernel_evaluation_radius = cube_size.times_f64(1.0 + 1e-3) * half_supported_cells_real;
    let kernel_evaluation_radius_sq = kernel_evaluation_radius * kernel_evaluation_radius;

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new(1000, kernel_radius);

    // Shrink the allowed domain for particles by the kernel evaluation radius. This ensures that all cells/points
    // that are affected by a particle are actually part of the domain/grid, so it does not have to be checked in the loops below.
    // However, any particles inside of this margin, close to the border of the originally requested domain will be ignored.
    let allowed_domain = {
        let mut aabb = grid.aabb().clone();
        aabb.grow_uniformly(kernel_evaluation_radius.neg().times(1));
        aabb
    };

    if allowed_domain.is_degenerate() || !allowed_domain.is_consistent() {
        warn!(
            "After taking the kernel evaluation radius into account, the allowed domain of particles is inconsistent/degenerate: {:?}",
            allowed_domain
        );
        warn!("No particles can be found in this domain. Increase the domain of surface reconstruction to avoid this.");
    } else {
        info!(
            "To take into account the kernel evaluation radius, the allowed domain of particles was restricted to: {:?}",
            allowed_domain
        );

        let process_particle = |particle_data: (&Vector3<R>, R)| {
            let (particle, particle_density) = particle_data;

            // Skip particles outside of allowed domain
            if !allowed_domain.contains_point(particle) {
                return;
            }

            // Compute the volume of this particle
            let particle_volume = particle_rest_mass / particle_density;

            // Compute grid points affected by the particle
            let min_supported_point_ijk = {
                let cell_ijk = grid.enclosing_cell(particle);

                [
                    cell_ijk[0] - half_supported_cells,
                    cell_ijk[1] - half_supported_cells,
                    cell_ijk[2] - half_supported_cells,
                ]
            };

            let max_supported_point_ijk = [
                min_supported_point_ijk[0] + supported_points,
                min_supported_point_ijk[1] + supported_points,
                min_supported_point_ijk[2] + supported_points,
            ];

            let min_supported_point = grid.point_coordinates_array(&min_supported_point_ijk);

            let mut dx = particle[0] - min_supported_point[0]
                // Subtract cell size because it will be added in the beginning of each loop iteration
                // this is done to avoid multiplications
                + grid.cell_size();

            // A range loop cannot be used here because the Step trait is unstable
            // but it is required for the Iter impl on Range
            // therefore a manual while loop has to be used
            let mut i = min_supported_point_ijk[0];
            while i != max_supported_point_ijk[0] {
                dx -= grid.cell_size();
                let dxdx = dx * dx;

                let mut dy = particle[1] - min_supported_point[1] + grid.cell_size();
                let mut j = min_supported_point_ijk[1];
                while j != max_supported_point_ijk[1] {
                    dy -= grid.cell_size();
                    let dydy = dy * dy;

                    let mut dz = particle[2] - min_supported_point[2] + grid.cell_size();
                    let mut k = min_supported_point_ijk[2];
                    while k != max_supported_point_ijk[2] {
                        dz -= grid.cell_size();
                        let dzdz = dz * dz;

                        let r_squared = dxdx + dydy + dzdz;
                        if r_squared < kernel_evaluation_radius_sq {
                            let density_contribution = particle_volume * kernel.evaluate(r_squared);

                            let flat_point_index = grid.flatten_point_indices(i, j, k);
                            *sparse_densities
                                .entry(flat_point_index)
                                .or_insert(R::zero()) += density_contribution;
                        }
                        k = k + I::one();
                    }
                    j = j + I::one();
                }
                i = i + I::one();
            }
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
    }

    info!(
        "Sparse density map with {} point data values was constructed.",
        sparse_densities.len()
    );
    info!("Construction of sparse density map done.");

    sparse_densities.into()
}

/// Computes a sparse density map for the fluid based on the specified background grid, multi-threaded implementation
#[inline(never)]
pub fn parallel_generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    kernel_radius: R,
    cube_size: R,
) -> DensityMap<I, R> {
    profile!("parallel_generate_sparse_density_map");

    info!(
        "Starting construction of sparse density map for {} particles...",
        if let Some(active_particles) = active_particles {
            active_particles.len()
        } else {
            particle_positions.len()
        }
    );

    // Each thread will write to its own local density map
    let sparse_densities: ThreadLocal<RefCell<MapType<I, R>>> = ThreadLocal::new();

    // The number of cells in each direction from a particle that can be affected by a its compact support
    let half_supported_cells_real = (kernel_radius / cube_size).ceil();
    // Convert to index type for cell and point indexing
    let half_supported_cells: I = half_supported_cells_real.to_index_unchecked();
    // The total number of points per dimension that can be affected by a particle's compact support
    //  + account for an additional layer of points that cover the positive outside of the supported cells
    let supported_points = I::one() + half_supported_cells.times(2);

    let kernel_evaluation_radius = cube_size.times_f64(1.0 + 1e-3) * half_supported_cells_real;
    let kernel_evaluation_radius_sq = kernel_evaluation_radius * kernel_evaluation_radius;

    // Pre-compute the kernel which can be queried using squared distances
    let kernel = DiscreteSquaredDistanceCubicKernel::new(1000, kernel_radius);

    // Shrink the allowed domain for particles by the kernel evaluation radius. This ensures that all cells/points
    // that are affected by a particle are actually part of the domain/grid, so it does not have to be checked in the loops below.
    // However, any particles inside of this margin, close to the border of the originally requested domain will be ignored.
    let allowed_domain = {
        let mut aabb = grid.aabb().clone();
        aabb.grow_uniformly(kernel_evaluation_radius.neg().times(1));
        aabb
    };

    if allowed_domain.is_degenerate() || !allowed_domain.is_consistent() {
        warn!(
            "After taking the kernel evaluation radius into account, the allowed domain of particles is inconsistent/degenerate: {:?}",
            allowed_domain
        );
        warn!("No particles can be found in this domain. Increase the domain of surface reconstruction to avoid this.");
    } else {
        profile!("generate thread local maps");

        info!(
            "To take into account the kernel evaluation radius, the allowed domain of particles was restricted to: {:?}",
            allowed_domain
        );

        let process_particle = |sparse_densities: &mut MapType<I, R>,
                                particle_data: (&Vector3<R>, R)| {
            let (particle, particle_density) = particle_data;

            // Skip particles outside of allowed domain
            if !allowed_domain.contains_point(particle) {
                return;
            }

            // Compute the volume of this particle
            let particle_volume = particle_rest_mass / particle_density;

            // Compute grid points affected by the particle
            let min_supported_point_ijk = {
                let cell_ijk = grid.enclosing_cell(particle);

                [
                    cell_ijk[0] - half_supported_cells,
                    cell_ijk[1] - half_supported_cells,
                    cell_ijk[2] - half_supported_cells,
                ]
            };

            let max_supported_point_ijk = [
                min_supported_point_ijk[0] + supported_points,
                min_supported_point_ijk[1] + supported_points,
                min_supported_point_ijk[2] + supported_points,
            ];

            let min_supported_point = grid.point_coordinates_array(&min_supported_point_ijk);

            let mut dx = particle[0] - min_supported_point[0]
                // Subtract cell size because it will be added in the beginning of each loop iteration
                // this is done to avoid multiplications
                + grid.cell_size();

            // A range loop cannot be used here because the Step trait is unstable
            // but it is required for the Iter impl on Range
            // therefore a manual while loop has to be used
            let mut i = min_supported_point_ijk[0];
            while i != max_supported_point_ijk[0] {
                dx -= grid.cell_size();
                let dxdx = dx * dx;

                let mut dy = particle[1] - min_supported_point[1] + grid.cell_size();
                let mut j = min_supported_point_ijk[1];
                while j != max_supported_point_ijk[1] {
                    dy -= grid.cell_size();
                    let dydy = dy * dy;

                    let mut dz = particle[2] - min_supported_point[2] + grid.cell_size();
                    let mut k = min_supported_point_ijk[2];
                    while k != max_supported_point_ijk[2] {
                        dz -= grid.cell_size();
                        let dzdz = dz * dz;

                        let r_squared = dxdx + dydy + dzdz;
                        if r_squared < kernel_evaluation_radius_sq {
                            let density_contribution = particle_volume * kernel.evaluate(r_squared);

                            let flat_point_index = grid.flatten_point_indices(i, j, k);
                            *sparse_densities
                                .entry(flat_point_index)
                                .or_insert(R::zero()) += density_contribution;
                        }
                        k = k + I::one();
                    }
                    j = j + I::one();
                }
                i = i + I::one();
            }
        };

        let compute_chunk_size = |num_particles: usize| -> usize {
            let min_chunk_size = 100.max(num_particles);
            let chunks_per_cpu = 10;

            let num_cpus = num_cpus::get();
            let num_chunks = chunks_per_cpu * num_cpus;
            let chunk_size = (num_particles / num_chunks).min(min_chunk_size);

            info!(
                "Splitting particles into {} chunks (with {} particles each) for density map generation",
                num_chunks, chunk_size
            );
            chunk_size
        };

        match active_particles {
            // Process particles, when no list of active particles was provided
            None => {
                let chunk_size = compute_chunk_size(particle_positions.len());
                particle_positions
                    .par_chunks(chunk_size)
                    .zip(particle_densities.par_chunks(chunk_size))
                    .for_each(|(position_chunk, density_chunk)| {
                        // Obtain mutable reference to thread local density map
                        let map = sparse_densities
                            .get_or(|| RefCell::new(MapType::with_hasher(HashState::default())));
                        let mut mut_map = map.borrow_mut();

                        let process_particle_map = |particle_data: (&Vector3<R>, R)| {
                            process_particle(&mut mut_map, particle_data);
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
                let chunk_size = compute_chunk_size(indices.len());
                indices.par_chunks(chunk_size).for_each(|index_chunk| {
                    // Obtain mutable reference to thread local density map
                    let map = sparse_densities
                        .get_or(|| RefCell::new(MapType::with_hasher(HashState::default())));
                    let mut mut_map = map.borrow_mut();

                    let process_particle_map = |particle_data: (&Vector3<R>, R)| {
                        process_particle(&mut mut_map, particle_data);
                    };

                    index_chunk
                        .iter()
                        .map(|&i| (&particle_positions[i], particle_densities[i]))
                        .for_each(process_particle_map);
                });
            }
        }
    }

    {
        profile!("merge thread local maps to global map");

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

        info!(
            "Global sparse density map with {} grid point data values was constructed.",
            global_density_map.len()
        );
        info!("Construction of sparse density map done.");

        global_density_map.into()
    }
}

/// Converts a sparse density map (based on the implicit background grid) to a sparse hexahedral mesh with explicit coordinates for the cells' vertices.
#[inline(never)]
pub fn sparse_density_map_to_hex_mesh<I: Index, R: Real>(
    density_map: &DensityMap<I, R>,
    grid: &UniformGrid<I, R>,
    default_value: R,
) -> MeshWithPointData<HexMesh3d<R>, R> {
    profile!("sparse_density_map_to_hex_mesh");

    let mut mesh = HexMesh3d {
        vertices: Vec::new(),
        cells: Vec::new(),
    };
    let mut values = Vec::new();
    let mut cells = new_map();

    // Create vertices and cells for points with values
    for (flat_point_index, point_value) in density_map.iter() {
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
                .or_insert([None, None, None, None, None, None, None, None]);

            let local_point_index = cell.local_point_index_of(point.index()).unwrap();
            cell_connectivity_entry[local_point_index] = Some(vertex_index);
        }
    }

    // Add missing vertices of cells using default values
    let mut additional_vertices = new_map();
    for (flat_cell_index, cell_vertices) in cells.iter_mut() {
        let cell = grid.try_unflatten_cell_index(*flat_cell_index).unwrap();

        for (local_point_index, vertex) in cell_vertices.iter_mut().enumerate() {
            if vertex.is_none() {
                // Map local point index to global index in grid
                let point = cell.global_index_of(local_point_index).unwrap();
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

    MeshWithPointData { mesh, data: values }
}
