use coarse_prof::profile;
use dashmap::DashMap;
use na::Vector3;
use rayon::prelude::*;

use crate::uniform_grid::UniformGrid;
use crate::utils::SendSyncWrapper;
use crate::{AxisAlignedBoundingBox3d, Index, MapType, Real};

// TODO: Replace some unwrap() calls with errors

// Generates a map for spatially hashed indices of all particles (map from cell -> enclosed particles)
#[inline(never)]
fn sequential_generate_cell_to_particle_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
) -> MapType<I, Vec<usize>> {
    profile!("sequential_generate_cell_to_particle_map");
    let mut particles_per_cell = MapType::new();

    // Assign all particles to enclosing cells
    for (particle_i, particle) in particle_positions.iter().enumerate() {
        let cell_ijk = grid.enclosing_cell(particle);
        let cell = grid.get_cell(&cell_ijk).unwrap();
        let flat_cell_index = grid.flatten_cell_index(&cell);

        particles_per_cell
            .entry(flat_cell_index)
            .or_insert_with(Vec::new)
            .push(particle_i);
    }

    particles_per_cell
}

#[inline(never)]
fn parallel_generate_cell_to_particle_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
) -> (DashMap<I, usize>, Vec<(I, Vec<usize>)>) {
    profile!("parallel_generate_cell_to_particle_map");
    let particles_per_cell: DashMap<I, usize> = DashMap::new();

    // This is obviously super slow. Maybe it's not worth to parallelize this function
    use std::sync::{Arc, RwLock};
    let particles_per_cell_storage = Arc::new(RwLock::new(Vec::new()));

    // Assign all particles to enclosing cells
    particle_positions
        .par_iter()
        .enumerate()
        .for_each(|(particle_i, particle)| {
            let cell_ijk = grid.enclosing_cell(particle);
            let cell = grid.get_cell(&cell_ijk).unwrap();
            let flat_cell_index = grid.flatten_cell_index(&cell);

            let index = *particles_per_cell
                .entry(flat_cell_index)
                .or_insert_with(|| {
                    let mut mutable_storage = particles_per_cell_storage.write().unwrap();
                    let new_index = mutable_storage.len();
                    mutable_storage.push((flat_cell_index, Vec::with_capacity(15)));
                    new_index
                })
                .value();

            particles_per_cell_storage
                .write()
                .unwrap()
                .get_mut(index)
                .unwrap()
                .1
                .push(particle_i);
        });

    let particles_per_cell_storage = Arc::try_unwrap(particles_per_cell_storage)
        .unwrap()
        .into_inner()
        .unwrap();

    (particles_per_cell, particles_per_cell_storage)
}

#[inline(never)]
pub fn search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
    allow_multi_threading: bool,
) -> Vec<Vec<usize>> {
    if allow_multi_threading {
        parallel_search::<I, R>(domain, particle_positions, search_radius)
    } else {
        sequential_search::<I, R>(domain, particle_positions, search_radius)
    }
}

#[inline(never)]
pub fn sequential_search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
) -> Vec<Vec<usize>> {
    // TODO: Use ArrayStorage from femproto instead of Vec of Vecs
    // FIXME: Replace unwraps?
    profile!("neighborhood_search");

    let search_radius_squared = search_radius * search_radius;

    // Create a new grid for neighborhood search
    let grid = UniformGrid::from_aabb(&domain, search_radius).unwrap();
    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell =
        sequential_generate_cell_to_particle_map::<I, R>(&grid, particle_positions);

    // Build neighborhood lists cell by cell
    let mut neighborhood_list = vec![Vec::new(); particle_positions.len()];
    {
        profile!("calculate_particle_neighbors_seq");
        for (&flat_cell_index, particles) in &particles_per_cell {
            let current_cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();

            // Collect references to the particle lists of all existing adjacent cells and the cell itself
            let potential_neighbor_particle_vecs: Vec<&Vec<usize>> = grid
                .cells_adjacent_to_cell(&current_cell)
                .chain(std::iter::once(current_cell))
                .filter_map(|c| {
                    let flat_cell_index = grid.flatten_cell_index(&c);
                    particles_per_cell.get(&flat_cell_index)
                })
                .collect();

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

    neighborhood_list
}

#[inline(never)]
pub fn parallel_search<I: Index, R: Real>(
    domain: &AxisAlignedBoundingBox3d<R>,
    particle_positions: &[Vector3<R>],
    search_radius: R,
) -> Vec<Vec<usize>> {
    profile!("neighborhood_search");

    let search_radius_squared = search_radius * search_radius;

    // Create a new grid for neighborhood search
    let grid = UniformGrid::from_aabb(&domain, search_radius).unwrap();

    // Map for spatially hashed storage of all particles (map from cell -> enclosed particles)
    let particles_per_cell_map =
        sequential_generate_cell_to_particle_map::<I, R>(&grid, particle_positions);
    let particles_per_cell_vec: Vec<(I, Vec<usize>)> = particles_per_cell_map
        .iter()
        .map(|(&i, v)| (i, v.clone()))
        .collect::<Vec<_>>();

    // In order to use Dashmap here, the whole process below has to be changed:
    //     In `adjacent_cell_particle_vecs`, we cannot store references into the map anymore.
    //     Therefore, these references have to point into a vector (e.g. `particles_per_cell_vec`) instead.
    //     For this, we need a map that maps the flat cell index to the position in the cell vec.
    //     This could be created here, or alternatively, already during the construction,
    //     inside of `parallel_generate_cell_to_particle_map`
    /*
    let (map, storage) = parallel_generate_cell_to_particle_map::<I, R>(&grid, particle_positions);
    let particles_per_cell_map = map;
    let particles_per_cell_vec = storage;

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
                        particles_per_cell_map
                            .get(&flat_cell_index)
                            .map(|entry| *entry.value())
                            .map(|vec_index| &particles_per_cell_vec.get(vec_index).unwrap().1)
                    })
                    .collect();
                potential_neighbor_particle_vecs
            })
            .collect::<Vec<_>>()
    };
    */

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
    let mut neighborhood_list = vec![Vec::with_capacity(15); particle_positions.len()];
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

    /*
    let mut max_neighbors = 0;
    let mut total_neighbors = 0;
    let mut nonzero_neighborhoods = 0;
    let mut neighbor_histogram: Vec<usize> = vec![0; 1];

    for neighborhood in neighborhood_list.iter() {
        if !neighborhood.is_empty() {
            if neighbor_histogram.len() < neighborhood.len() + 1 {
                neighbor_histogram.resize(neighborhood.len() + 1, 0);
            }
            neighbor_histogram[neighborhood.len()] += 1;

            max_neighbors = max_neighbors.max(neighborhood.len());
            total_neighbors += neighborhood.len();
            nonzero_neighborhoods += 1;
        } else {
            neighbor_histogram[0] += 1;
        }
    }

    let avg_neighbors = total_neighbors as f64 / nonzero_neighborhoods as f64;
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

    neighborhood_list
}
