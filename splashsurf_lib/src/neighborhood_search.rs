//! Simple neighborhood search based on spatial hashing
//!
//! This module provides basic sequential and parallel neighborhood search implementations using
//! spatial hashing. The algorithms return per-particle neighborhood list with indices of all particles
//! that are within the given radius of the particle.

use crate::uniform_grid::UniformGrid;
use crate::utils::SendSyncWrapper;
use crate::{
    new_map, profile, AxisAlignedBoundingBox3d, HashState, Index, MapType, ParallelMapType, Real,
};
use nalgebra::Vector3;
use rayon::prelude::*;

// TODO: Replace some unwrap() calls with errors, e.g. if domain excludes some particles that are neighbors
// TODO: Check if input parameters are valid (valid domain, valid search radius)
// TODO: Write tests with sample data for neighborhood search

/// Performs a neighborhood search, returning the indices of all neighboring particles in the given search radius per particle
#[inline(never)]
pub fn search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    enable_multi_threading: bool,
) -> Vec<Vec<usize>> {
    let mut particle_neighbor_lists = Vec::new();
    if enable_multi_threading {
        parallel_search::<I, R>(
            domain,
            particle_positions,
            search_radius,
            &mut particle_neighbor_lists,
        )
    } else {
        sequential_search::<I, R>(
            domain,
            particle_positions,
            search_radius,
            &mut particle_neighbor_lists,
        )
    }
    particle_neighbor_lists
}

/// Performs a neighborhood search inplace, stores the indices of all neighboring particles in the given search radius per particle in the given vector
#[inline(never)]
pub fn search_inplace<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    enable_multi_threading: bool,
    particle_neighbor_lists: &mut Vec<Vec<usize>>,
) {
    if enable_multi_threading {
        parallel_search::<I, R>(
            domain,
            particle_positions,
            search_radius,
            particle_neighbor_lists,
        )
    } else {
        sequential_search::<I, R>(
            domain,
            particle_positions,
            search_radius,
            particle_neighbor_lists,
        )
    }
}

/// Allocates enough storage for the given number of particles and clears all existing neighborhood lists
fn init_neighborhood_list(neighborhood_list: &mut Vec<Vec<usize>>, new_len: usize) {
    let old_len = neighborhood_list.len();
    // Reset all neighbor lists that won't be truncated
    for particle_list in neighborhood_list.iter_mut().take(old_len.min(new_len)) {
        particle_list.clear();
    }

    // Ensure that length is correct
    neighborhood_list.resize_with(new_len, || Vec::with_capacity(15));
}

/// Allocates enough storage for the given number of particles and clears all existing neighborhood lists in parallel
fn par_init_neighborhood_list(neighborhood_list: &mut Vec<Vec<usize>>, new_len: usize) {
    let old_len = neighborhood_list.len();
    // Reset all neighbor lists that won't be truncated
    neighborhood_list
        .par_iter_mut()
        .with_min_len(8)
        .take(old_len.min(new_len))
        .for_each(|particle_list| {
            particle_list.clear();
        });

    // Ensure that length is correct
    neighborhood_list.resize_with(new_len, || Vec::with_capacity(15));
}

/// Performs a neighborhood search, returning the indices of all neighboring particles in the given search radius per particle, sequential implementation
#[inline(never)]
pub fn sequential_search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
) {
    // TODO: Use ArrayStorage from femproto instead of Vec of Vecs?
    // FIXME: Replace unwraps?
    profile!("neighborhood_search");

    assert!(
        search_radius > R::zero(),
        "Search radius for neighborhood search has to be positive!"
    );
    assert!(
        domain.is_consistent(),
        "Domain for neighborhood search has to be consistent!"
    );
    assert!(
        !domain.is_degenerate(),
        "Domain for neighborhood search cannot be degenerate!"
    );

    let search_radius_squared = search_radius * search_radius;

    // Create a new grid for neighborhood search
    let grid = UniformGrid::from_aabb(&domain, search_radius)
        .expect("Failed to construct grid for neighborhood search!");
    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell =
        sequential_generate_cell_to_particle_map::<I, R>(&grid, particle_positions);

    // Build neighborhood lists cell by cell
    init_neighborhood_list(neighborhood_list, particle_positions.len());
    {
        profile!("calculate_particle_neighbors_seq");
        let mut potential_neighbor_particle_vecs = Vec::new();
        for (&flat_cell_index, particles) in &particles_per_cell {
            let current_cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();

            // Collect references to the particle lists of all existing adjacent cells and the cell itself
            potential_neighbor_particle_vecs.clear();
            potential_neighbor_particle_vecs.extend(
                grid.cells_adjacent_to_cell(&current_cell)
                    .chain(std::iter::once(current_cell))
                    .filter_map(|c| {
                        let flat_cell_index = grid.flatten_cell_index(&c);
                        particles_per_cell.get(&flat_cell_index)
                    }),
            );

            // Returns an iterator over all particles of all adjacent cells and the cell itself
            let potential_neighbor_particle_iter = || {
                potential_neighbor_particle_vecs
                    .iter()
                    .flat_map(|v| v.iter())
            };

            // Iterate over all particles of the current cell
            for &particle_i in particles {
                let pos_i = &particle_positions[particle_i];
                let particle_i_neighbors = &mut neighborhood_list[particle_i];

                // Check for neighborhood with all neighboring cells
                for &particle_j in potential_neighbor_particle_iter() {
                    if particle_i == particle_j {
                        continue;
                    }

                    let pos_j = &particle_positions[particle_j];
                    if (pos_j - pos_i).norm_squared() < search_radius_squared {
                        // A neighbor was found
                        particle_i_neighbors.push(particle_j);
                    }
                }
            }
        }
    }
}

/// Performs a neighborhood search, returning the indices of all neighboring particles in the given search radius per particle, multi-thread implementation
#[inline(never)]
pub fn parallel_search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
) {
    profile!("par_neighborhood_search");

    assert!(
        search_radius > R::zero(),
        "Search radius for neighborhood search has to be positive!"
    );
    assert!(
        domain.is_consistent(),
        "Domain for neighborhood search has to be consistent!"
    );
    assert!(
        !domain.is_degenerate(),
        "Domain for neighborhood search cannot be degenerate!"
    );

    let search_radius_squared = search_radius * search_radius;

    // Create a new grid for neighborhood search
    let grid = UniformGrid::from_aabb(&domain, search_radius)
        .expect("Failed to construct grid for neighborhood search!");

    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell_map =
        parallel_generate_cell_to_particle_map::<I, R>(&grid, particle_positions).into_read_only();
    let particles_per_cell_vec: Vec<(I, Vec<usize>)> = particles_per_cell_map
        .iter()
        .map(|(&i, v)| (i, v.clone()))
        .collect::<Vec<_>>();

    // Extract, per cell, the particle lists of all adjacent cells
    let adjacent_cell_particle_vecs = {
        profile!("get_cell_neighborhoods_par");
        particles_per_cell_vec
            .par_iter()
            .map(|(flat_cell_index, _)| {
                let current_cell = grid.try_unflatten_cell_index(*flat_cell_index).unwrap();

                // Collect references to the particle lists of all existing adjacent cells
                let potential_neighbor_particle_vecs: Vec<&Vec<usize>> = grid
                    .cells_adjacent_to_cell(&current_cell)
                    .filter_map(|c| {
                        let flat_cell_index = grid.flatten_cell_index(&c);
                        particles_per_cell_map.get(&flat_cell_index)
                    })
                    .collect();
                potential_neighbor_particle_vecs
            })
            .collect::<Vec<_>>()
    };

    // TODO: Compute the default capacity of neighborhood lists from rest volume of particles
    par_init_neighborhood_list(neighborhood_list, particle_positions.len());
    // We are ok with making the ptr Send+Sync because we are only going to write into disjoint fields of the Vec
    let neighborhood_list_mut_ptr = unsafe { SendSyncWrapper::new(neighborhood_list.as_mut_ptr()) };

    {
        profile!("calculate_particle_neighbors_par");
        particles_per_cell_vec.par_iter().enumerate().for_each(
            |(cell_k, (_, cell_k_particles))| {
                // The particle lists of all cells adjacent to the current cell
                let cell_k_adjacent_particle_vecs = &adjacent_cell_particle_vecs[cell_k];

                // Iterate over all particles of the current cell
                for (i, &particle_i) in cell_k_particles.iter().enumerate() {
                    let pos_i = &particle_positions[particle_i];
                    // We know that the we will only access disjoint locations, because a particle index can only appear in one cell
                    let particle_i_neighbors =
                        unsafe { &mut *neighborhood_list_mut_ptr.get().add(particle_i) };

                    // Check for neighborhood with particles of all adjacent cells
                    for &adjacent_cell_particles in cell_k_adjacent_particle_vecs.iter() {
                        for &particle_j in adjacent_cell_particles.iter() {
                            let pos_j = &particle_positions[particle_j];
                            if (pos_j - pos_i).norm_squared() < search_radius_squared {
                                // A neighbor was found
                                particle_i_neighbors.push(particle_j);
                            }
                        }
                    }

                    // Check for neighborhood with all remaining particles of the same cell
                    for &particle_j in cell_k_particles.iter().skip(i + 1) {
                        let pos_j = &particle_positions[particle_j];
                        if (pos_j - pos_i).norm_squared() < search_radius_squared {
                            // A neighbor was found
                            particle_i_neighbors.push(particle_j);

                            // Add neighborhood transitively
                            let particle_j_neighbors =
                                unsafe { &mut *neighborhood_list_mut_ptr.get().add(particle_j) };
                            particle_j_neighbors.push(particle_i);
                        }
                    }
                }
            },
        );
    }
}

/// Stats of a neighborhood list
#[derive(Clone, Debug)]
pub struct NeighborhoodStats {
    /// A histogram over the count of particle neighbors per particle (e.g. `histogram[0]` -> count of particles without neighbors, `histogram[1]` -> count of particles with one neighbor, etc.)
    pub histogram: Vec<usize>,
    /// Number of particles that have neighbors
    pub particles_with_neighbors: usize,
    /// The size of the largest neighborhood
    pub max_neighbors: usize,
    /// Average number of neighbors per particle (excluding particles without neighbors)
    pub avg_neighbors: f64,
}

/// Computes stats (avg. neighbors, histogram) of the given neighborhood list
pub fn compute_neigborhood_stats(neighborhood_list: &Vec<Vec<usize>>) -> NeighborhoodStats {
    let mut max_neighbors = 0;
    let mut total_neighbors = 0;
    let mut particles_with_neighbors = 0;
    let mut neighbor_histogram: Vec<usize> = vec![0; 1];

    for neighborhood in neighborhood_list.iter() {
        if !neighborhood.is_empty() {
            if neighbor_histogram.len() < neighborhood.len() + 1 {
                neighbor_histogram.resize(neighborhood.len() + 1, 0);
            }
            neighbor_histogram[neighborhood.len()] += 1;

            max_neighbors = max_neighbors.max(neighborhood.len());
            total_neighbors += neighborhood.len();
            particles_with_neighbors += 1;
        } else {
            neighbor_histogram[0] += 1;
        }
    }

    let avg_neighbors = total_neighbors as f64 / particles_with_neighbors as f64;

    /*
    println!(
        "Max neighbors: {}, Avg neighbors: {:.3}, particles with neighbors: {:.3}%",
        max_neighbors,
        avg_neighbors,
        (nonzero_neighborhoods as f64 / particle_positions.len() as f64) * 100.0
    );
    println!("Histogram:");
    for (i, &count) in neighbor_histogram.iter().enumerate() {
        println!("{:2} neighbors: {:10}", i, count);
    }
     */

    NeighborhoodStats {
        histogram: neighbor_histogram,
        particles_with_neighbors,
        max_neighbors,
        avg_neighbors,
    }
}

// Generates a map for spatially hashed indices of all particles (map from cell -> enclosed particles)
#[inline(never)]
fn sequential_generate_cell_to_particle_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
) -> MapType<I, Vec<usize>> {
    profile!("sequential_generate_cell_to_particle_map");
    let mut particles_per_cell = new_map();

    // Compute average particle density for initial cell capacity
    let cell_dims = grid.cells_per_dim();
    let n_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];
    let avg_density = particle_positions.len() / n_cells.to_usize().unwrap_or(1);

    // Assign all particles to enclosing cells
    for (particle_i, particle) in particle_positions.iter().enumerate() {
        let cell_ijk = grid.enclosing_cell(particle);
        let cell = grid.get_cell(cell_ijk).unwrap();
        let flat_cell_index = grid.flatten_cell_index(&cell);

        particles_per_cell
            .entry(flat_cell_index)
            .or_insert_with(|| Vec::with_capacity(avg_density))
            .push(particle_i);
    }

    particles_per_cell
}

#[inline(never)]
fn parallel_generate_cell_to_particle_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
) -> ParallelMapType<I, Vec<usize>> {
    profile!("parallel_generate_cell_to_particle_map");
    let particles_per_cell = ParallelMapType::with_hasher(HashState::default());

    // Assign all particles to enclosing cells
    particle_positions
        .par_iter()
        .enumerate()
        .for_each(|(particle_i, particle)| {
            let cell_ijk = grid.enclosing_cell(particle);
            let cell = grid.get_cell(cell_ijk).unwrap();
            let flat_cell_index = grid.flatten_cell_index(&cell);

            particles_per_cell
                .entry(flat_cell_index)
                .or_insert_with(Vec::new)
                .push(particle_i);
        });

    particles_per_cell
}
