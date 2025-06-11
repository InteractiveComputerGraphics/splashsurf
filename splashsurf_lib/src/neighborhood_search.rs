//! Simple neighborhood search based on spatial hashing
//!
//! This module provides basic sequential and parallel neighborhood search implementations using
//! spatial hashing. The algorithms return per-particle neighborhood list with indices of all particles
//! that are within the given radius of the particle.

use crate::uniform_grid::UniformGrid;
use crate::utils::UnsafeSlice;
use crate::{Aabb3d, HashState, Index, MapType, ParallelMapType, Real, new_map, profile};
use nalgebra::Vector3;
use rayon::prelude::*;

// TODO: Replace some unwrap() calls with errors, e.g. if domain excludes some particles that are neighbors
// TODO: Check if input parameters are valid (valid domain, valid search radius)
// TODO: Write tests with sample data for neighborhood search

/// Performs a neighborhood search, returning the indices of all neighboring particles in the given search radius per particle
#[inline(never)]
pub fn search<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    enable_multi_threading: bool,
) -> Vec<Vec<usize>> {
    let mut particle_neighbor_lists = Vec::new();
    if enable_multi_threading {
        neighborhood_search_spatial_hashing_parallel::<I, R>(
            domain,
            particle_positions,
            search_radius,
            &mut particle_neighbor_lists,
        )
    } else {
        neighborhood_search_spatial_hashing::<I, R>(
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
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    enable_multi_threading: bool,
    particle_neighbor_lists: &mut Vec<Vec<usize>>,
) {
    if enable_multi_threading {
        neighborhood_search_spatial_hashing_parallel::<I, R>(
            domain,
            particle_positions,
            search_radius,
            particle_neighbor_lists,
        )
    } else {
        neighborhood_search_spatial_hashing::<I, R>(
            domain,
            particle_positions,
            search_radius,
            particle_neighbor_lists,
        )
    }
}

/// Performs a naive neighborhood search with `O(N^2)` complexity, only recommended for testing
#[inline(never)]
pub fn neighborhood_search_naive<R: Real>(
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
) {
    profile!("neighborhood_search_naive");

    init_neighborhood_list(neighborhood_list, particle_positions.len());
    let search_radius_squared = search_radius * search_radius;

    for (idx_i, pos_i) in particle_positions.iter().enumerate() {
        let neighbors_i = neighborhood_list.get_mut(idx_i).unwrap();
        for (idx_j, pos_j) in particle_positions.iter().enumerate() {
            let is_neighbor = (pos_j - pos_i).norm_squared() <= search_radius_squared;
            if is_neighbor && idx_j != idx_i {
                neighbors_i.push(idx_j);
            }
        }
    }
}

/// Allocates enough storage for the given number of particles and clears all existing neighborhood lists
fn init_neighborhood_list(neighborhood_list: &mut Vec<Vec<usize>>, new_len: usize) {
    profile!("init_neighborhood_list");

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

/// Performs a neighborhood search (sequential implementation)
///
/// Returns the indices of all neighboring particles in the given search radius per particle as a `Vec<Vec<usize>>`.
#[inline(never)]
pub fn neighborhood_search_spatial_hashing<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
) {
    neighborhood_search_spatial_hashing_filtered::<I, R>(
        domain,
        particle_positions,
        search_radius,
        neighborhood_list,
        |_| true,
    )
}

/// Performs a neighborhood search (sequential implementation, with filter)
///
/// Returns the indices of all neighboring particles in the given search radius per particle as a `Vec<Vec<usize>>`.
/// The filter specifies which particles the neighbor lists should be computed for (`true`: compute neighbors).
/// Note that the particles that were filtered out will still appear in the neighbor lists of the particles that were not filtered out.
#[inline(never)]
pub fn neighborhood_search_spatial_hashing_filtered<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
    filter: impl Fn(usize) -> bool,
) {
    // FIXME: Replace unwraps?
    profile!("neighborhood_search_spatial_hashing_filtered");

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
    let grid = UniformGrid::from_aabb(domain, search_radius)
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
                if !filter(particle_i) {
                    continue;
                }

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

/// Stores particle neighborhood lists contiguously in memory using a second offset array
pub struct FlatNeighborhoodList {
    /// Offsets to the start of the neighborhood list of the given particle (very last entry contains total number of neighbor particles)
    pub neighbor_ptr: Vec<usize>,
    /// Flat particle neighborhood list storage
    pub neighbors: Vec<usize>,
}

impl Default for FlatNeighborhoodList {
    fn default() -> Self {
        Self {
            neighbor_ptr: vec![0],
            neighbors: vec![],
        }
    }
}

impl FlatNeighborhoodList {
    /// Returns the total number of particles this list has neighborhood information about
    pub fn len(&self) -> usize {
        self.neighbor_ptr.len() - 1
    }

    pub fn iter(&self) -> impl Iterator<Item = &[usize]> + '_ {
        (0..self.neighbor_ptr.len()).flat_map(|i| self.get_neighbors(i))
    }

    /// Returns a slice containing the neighborhood list of the given particle
    pub fn get_neighbors(&self, particle_i: usize) -> Option<&[usize]> {
        let range = self
            .neighbor_ptr
            .get(particle_i)
            .copied()
            .zip(self.neighbor_ptr.get(particle_i + 1).copied());
        range.and_then(|(start, end)| self.neighbors.get(start..end))
    }

    /// Returns a mutable slice containing the neighborhood list of the given particle
    pub fn get_neighbors_mut(&mut self, particle_i: usize) -> Option<&mut [usize]> {
        let range = self
            .neighbor_ptr
            .get(particle_i)
            .copied()
            .zip(self.neighbor_ptr.get(particle_i + 1).copied());
        range.and_then(|(start, end)| self.neighbors.get_mut(start..end))
    }

    /// Converts this flat neighbor list to its nested `Vec` representation
    pub fn to_vec_vec(self) -> Vec<Vec<usize>> {
        let num_particles = self.len();
        let mut neighbor_list = vec![Vec::new(); num_particles];
        neighbor_list
            .iter_mut()
            .enumerate()
            .for_each(|(i, nl)| nl.extend(self.get_neighbors(i).unwrap().iter().copied()));
        neighbor_list
    }
}

/// Trait unifying different particle neighborhood list storage formats
pub trait NeighborhoodList {
    /// Returns the total number of particles this list has neighborhood information about
    fn len(&self) -> usize;
    /// Returns the neighborhood list of the given particle
    fn neighbors(&self, particle_i: usize) -> &[usize];
}

impl NeighborhoodList for FlatNeighborhoodList {
    fn len(&self) -> usize {
        self.len()
    }

    fn neighbors(&self, particle_i: usize) -> &[usize] {
        self.get_neighbors(particle_i).unwrap()
    }
}

/// Implementation for `Vec<Vec<usize>>` and `[Vec<usize>]`
impl<T: AsRef<[Vec<usize>]> + ?Sized> NeighborhoodList for T {
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn neighbors(&self, particle_i: usize) -> &[usize] {
        &self.as_ref()[particle_i]
    }
}

/// Performs a neighborhood search (sequential implementation, returning a [`FlatNeighborhoodList`])
///
/// Returns the indices of all neighboring particles in the given search radius per particle as a [`FlatNeighborhoodList`].
pub fn neighborhood_search_spatial_hashing_flat<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut FlatNeighborhoodList,
) {
    profile!("neighborhood_search_spatial_hashing_flat");

    neighborhood_search_spatial_hashing_flat_filtered::<I, R>(
        domain,
        particle_positions,
        search_radius,
        neighborhood_list,
        |_| true,
    )
}

/// Performs a neighborhood search (sequential implementation, with filter, returning a [`FlatNeighborhoodList`])
///
/// Returns the indices of all neighboring particles in the given search radius per particle as a [`FlatNeighborhoodList`].
/// The filter specifies which particles the neighbor lists should be computed for (`true`: compute neighbors).
/// Note that the particles that were filtered out will still appear in the neighbor lists of the particles that were not filtered out.
pub fn neighborhood_search_spatial_hashing_flat_filtered<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut FlatNeighborhoodList,
    filter: impl Fn(usize) -> bool,
) {
    profile!("neighborhood_search_spatial_hashing_flat_filtered");

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
    let grid = UniformGrid::from_aabb(domain, search_radius)
        .expect("Failed to construct grid for neighborhood search!");
    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell =
        sequential_generate_cell_to_particle_map_with_positions::<I, R>(&grid, particle_positions);

    {
        neighborhood_list.neighbor_ptr.clear();
        neighborhood_list
            .neighbor_ptr
            .resize(particle_positions.len() + 1, 0);
        neighborhood_list.neighbors.clear();
    }

    {
        profile!("collect particle neighbors");
        let mut cell_buffer = Vec::with_capacity(27);
        let mut candidate_buffer = Vec::with_capacity(1000);
        for (particle_i, pos_i) in particle_positions.iter().enumerate() {
            // Store start index of the current particle neighbor list
            neighborhood_list.neighbor_ptr[particle_i] = neighborhood_list.neighbors.len();

            if !filter(particle_i) {
                continue;
            }

            // Cell of the current particle
            let current_cell = grid.get_cell(grid.enclosing_cell(pos_i)).unwrap();

            // Collect indices of cells to check
            cell_buffer.clear();
            cell_buffer.extend(
                grid.cells_adjacent_to_cell(&current_cell)
                    .chain(std::iter::once(current_cell))
                    .map(|c| grid.flatten_cell_index(&c)),
            );

            // Compute the total number of particles that are neighbor candidates
            let num_candidates: usize = cell_buffer
                .iter()
                .filter_map(|c| particles_per_cell.get(&c))
                .map(|(indices, _)| indices.len())
                .sum();
            candidate_buffer.resize(num_candidates, (0, R::zero()));

            // Compute distances to all neighbor candidates
            let mut counter = 0;
            cell_buffer
                .iter()
                .filter_map(|c| particles_per_cell.get(&c))
                .for_each(|(indices, positions)| {
                    for (&particle_j, pos_j) in indices.iter().zip(positions.iter()) {
                        candidate_buffer[counter] = (particle_j, (pos_j - pos_i).norm_squared());
                        counter += 1;
                    }
                });

            // Filter for particles that are actually neighbors
            neighborhood_list.neighbors.extend(
                candidate_buffer
                    .iter()
                    .filter(|(idx, dist)| *idx != particle_i && *dist < search_radius_squared)
                    .map(|(idx, _)| idx),
            );
        }
        // Store end of the last neighbor list
        *neighborhood_list.neighbor_ptr.last_mut().unwrap() = neighborhood_list.neighbors.len();
    }
}

/// Performs a neighborhood search (multithreaded implementation)
///
/// Returns the indices of all neighboring particles in the given search radius per particle as a `Vec<Vec<usize>>`.
#[inline(never)]
pub fn neighborhood_search_spatial_hashing_parallel<I: Index, R: Real>(
    domain: &Aabb3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    neighborhood_list: &mut Vec<Vec<usize>>,
) {
    profile!("neighborhood_search_spatial_hashing_parallel");

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
    let grid = UniformGrid::from_aabb(domain, search_radius)
        .expect("Failed to construct grid for neighborhood search!");

    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell_map =
        parallel_generate_cell_to_particle_map::<I, R>(&grid, particle_positions).into_read_only();
    let particles_per_cell_vec: Vec<(I, Vec<usize>)> = {
        profile!("cell_map_to_vec");
        particles_per_cell_map
            .iter()
            .map(|(&i, v)| (i, v.clone()))
            .collect::<Vec<_>>()
    };

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
    {
        profile!("init_neighbor_list_storage");
        par_init_neighborhood_list(neighborhood_list, particle_positions.len());
    }

    // We have to share the pointer to the neighborhood list storage between threads to avoid unnecessary copies and expensive merging.
    // SAFETY: In principle this can be done soundly because
    //  1. It is only UB when the pointer is actually dereferenced.
    //  2. When the pointer is incremented to disjoint locations before being dereferenced,
    //     there can only be one mutable reference to each entry in the storage which is not UB.
    // These conditions have to be guaranteed by the code that uses the pointer below.
    let neighborhood_list_mut_ptr = UnsafeSlice::new(neighborhood_list.as_mut_slice());

    {
        profile!("calculate_particle_neighbors_par");
        particles_per_cell_vec.par_iter().enumerate().for_each(
            |(cell_k, (_, cell_k_particles))| {
                // The particle lists of all cells adjacent to the current cell
                let cell_k_adjacent_particle_vecs = &adjacent_cell_particle_vecs[cell_k];

                let mut local_buffer = Vec::with_capacity(50);

                let neighborhood_test =
                    |pos_i: Vector3<R>, particle_j: usize, local_buffer: &mut Vec<usize>| {
                        let pos_j = unsafe { *particle_positions.get_unchecked(particle_j) };

                        // TODO: We might not be able to guarantee that this is symmetric.
                        //  Therefore, it might be possible that only one side of some neighborhood relationships gets detected.
                        if (pos_j - pos_i).norm_squared() < search_radius_squared {
                            // A neighbor was found
                            //particle_i_neighbors.push(particle_j);
                            local_buffer.push(particle_j);
                        }
                    };

                // Iterate over all particles of the current cell
                for (i, particle_i) in cell_k_particles.iter().copied().enumerate() {
                    let pos_i = unsafe { *particle_positions.get_unchecked(particle_i) };

                    // Check for neighborhood with particles of all adjacent cells
                    // Transitive neighborhood relationship is not handled explicitly.
                    // Instead, it will be handled when the cell of `particle_j` is processed.
                    cell_k_adjacent_particle_vecs
                        .iter()
                        .copied()
                        .flatten()
                        .copied()
                        .for_each(|particle_j| {
                            neighborhood_test(pos_i, particle_j, &mut local_buffer)
                        });

                    // Check particles of own cell until particle itself
                    cell_k_particles[..i]
                        .iter()
                        .copied()
                        .for_each(|particle_j| {
                            neighborhood_test(pos_i, particle_j, &mut local_buffer)
                        });

                    // Check particles of own cell after particle itself
                    cell_k_particles[i + 1..]
                        .iter()
                        .copied()
                        .for_each(|particle_j| {
                            neighborhood_test(pos_i, particle_j, &mut local_buffer)
                        });

                    if !local_buffer.is_empty() {
                        // Get mutable reference to the neighborhood list of `particle_i`
                        // SAFETY: This is sound because
                        //  1. Here, we only write to neighborhood lists of particles in the current cell `cell_k`.
                        //  2. The particles of the current cell `cell_k` are only handled by this closure invocation in sequence.
                        //  3. The spatial hashing guarantees that a particle is stored only once and in a single cell.
                        // => We only dereference and write to strictly disjoint regions in memory
                        let particle_i_neighbors =
                            unsafe { neighborhood_list_mut_ptr.get_mut_unchecked(particle_i) };

                        particle_i_neighbors.extend(local_buffer.iter());
                        local_buffer.clear();
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

/// Generates a map for spatially hashed indices of all particles (map from cell -> enclosed particles)
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

/// Generates a map for spatially hashed indices and positions of all particles (map from cell -> enclosed particles)
#[inline(never)]
fn sequential_generate_cell_to_particle_map_with_positions<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
) -> MapType<I, (Vec<usize>, Vec<Vector3<R>>)> {
    profile!("sequential_generate_cell_to_particle_map_with_positions");
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

        let (cell_indices, cell_positions) = particles_per_cell
            .entry(flat_cell_index)
            .or_insert_with(|| {
                (
                    Vec::with_capacity(avg_density),
                    Vec::with_capacity(avg_density),
                )
            });
        cell_indices.push(particle_i);
        cell_positions.push(particle.clone());
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
