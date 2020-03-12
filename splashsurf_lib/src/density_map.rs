use coarse_prof::profile;
use log::{info, warn};
use na::Vector3;
use rayon::prelude::*;

use crate::kernel::DiscreteSquaredDistanceCubicKernel;
use crate::mesh::{HexMesh3d, MeshWithPointData};
use crate::uniform_grid::UniformGrid;
use crate::{Index, MapType, Real};

#[inline(never)]
pub fn compute_particle_densities<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    kernel_radius: R,
    particle_rest_mass: R,
    allow_multi_threading: bool,
) -> Vec<R> {
    if allow_multi_threading {
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

#[inline(never)]
pub fn generate_sparse_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    active_particles: Option<&[usize]>,
    particle_rest_mass: R,
    kernel_radius: R,
    cube_size: R,
) -> MapType<I, R> {
    profile!("generate_sparse_density_map");

    match active_particles {
        None => _generate_sparse_density(
            grid,
            particle_positions.iter(),
            particle_densities.iter().copied(),
            particle_rest_mass,
            kernel_radius,
            cube_size,
        ),
        Some(indices) => _generate_sparse_density(
            grid,
            indices.iter().map(|&i| &particle_positions[i]),
            particle_densities.iter().copied(),
            particle_rest_mass,
            kernel_radius,
            cube_size,
        ),
    }
}

fn _generate_sparse_density<
    'a,
    I: Index,
    R: Real,
    ParticlePosIter: ExactSizeIterator<Item = &'a Vector3<R>>,
    ParticleDensityIter: Iterator<Item = R>,
>(
    grid: &UniformGrid<I, R>,
    particle_positions_iter: ParticlePosIter,
    particle_density_iter: ParticleDensityIter,
    particle_rest_mass: R,
    kernel_radius: R,
    cube_size: R,
) -> MapType<I, R> {
    info!(
        "Starting construction of sparse density map for {} particles...",
        particle_positions_iter.len()
    );

    let mut sparse_densities = MapType::new();

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

        for (particle, particle_density) in particle_positions_iter.zip(particle_density_iter) {
            // Skip particles outside of allowed domain
            if !allowed_domain.contains_point(particle) {
                continue;
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
        }
    }

    info!(
        "Sparse density map with {} point data values was constructed.",
        sparse_densities.len()
    );
    info!("Construction of sparse density map done.");

    sparse_densities
}

#[inline(never)]
pub fn sparse_density_map_to_hex_mesh<I: Index, R: Real>(
    point_data: &MapType<I, R>,
    grid: &UniformGrid<I, R>,
    default_value: R,
) -> MeshWithPointData<HexMesh3d<R>, R> {
    profile!("point_data_to_hex_mesh");

    let mut mesh = HexMesh3d {
        vertices: Vec::new(),
        cells: Vec::new(),
    };
    let mut values = Vec::new();
    let mut cells = MapType::new();

    // Create vertices and cells for points with values
    for (&flat_point_index, &point_value) in point_data {
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
    let mut additional_vertices = MapType::new();
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
