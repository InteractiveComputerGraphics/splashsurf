//! Internal functions related to the subdomain-based parallel surface reconstruction
//!
//! Note that we do not guarantee API stability of these functions, they are only exposed for
//! testing and benchmarking purposes.

use anyhow::{Context, anyhow};
use itertools::Itertools;
use log::{info, trace};
use nalgebra::Vector3;
use num_integer::Integer;
use num_traits::{FromPrimitive, NumCast};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use thread_local::ThreadLocal;

use crate::density_map::sequential_compute_particle_densities_filtered;
use crate::kernel::{
    CubicSplineKernel, KernelType, Poly6Kernel, SpikyKernel, SymmetricKernel3d,
    WendlandQuinticC2Kernel,
};
use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::mesh::{HexMesh3d, TriMesh3d};
use crate::neighborhood_search::{
    FlatNeighborhoodList, neighborhood_search_spatial_hashing_flat_filtered,
    neighborhood_search_spatial_hashing_parallel,
};
use crate::topology::Direction;
use crate::uniform_grid::{EdgeIndex, GridConstructionError, UniformCartesianCubeGrid3d};
use crate::{
    Aabb3d, MapType, Parameters, RealConvert, SpatialDecomposition, SurfaceReconstruction, kernel,
    new_map, new_parallel_map, profile,
};
use crate::{Index, Real};

// TODO: Implement single-threaded processing

type GlobalIndex = i64;

pub(crate) struct ParametersSubdomainGrid<I: Index, R: Real> {
    /// SPH particle radius (in simulation units)
    #[allow(unused)]
    pub particle_radius: R,
    /// Rest mass of each particle
    pub particle_rest_mass: R,
    /// SPH kernel compact support radius (in simulation units)
    pub compact_support_radius: R,
    /// Density value for the iso-surface
    pub surface_threshold: R,
    /// MC cube size (in simulation units)
    pub cube_size: R,
    /// Size of a subdomain in multiples of MC cubes
    pub subdomain_cubes: I,
    /// Margin for ghost particles around each subdomain
    pub ghost_particle_margin: R,
    /// Implicit global MC background grid (required to compute consistent float coordinates at domain boundaries)
    pub global_marching_cubes_grid: UniformCartesianCubeGrid3d<GlobalIndex, R>,
    /// Implicit subdomain grid
    pub subdomain_grid: UniformCartesianCubeGrid3d<I, R>,
    /// Chunk size for chunked parallel processing
    pub chunk_size: usize,
    /// Whether to return the global particle neighborhood list instead of only using per-domain lists internally
    pub global_neighborhood_list: bool,
    /// The SPH kernel that should be used
    pub kernel_type: KernelType,
}

impl<I: Index, R: Real> ParametersSubdomainGrid<I, R> {
    pub(crate) fn global_marching_cubes_grid(
        &self,
    ) -> Result<UniformCartesianCubeGrid3d<I, R>, GridConstructionError<I, R>> {
        let n_cells = self.global_marching_cubes_grid.cells_per_dim();
        UniformCartesianCubeGrid3d::new(
            self.global_marching_cubes_grid.aabb().min(),
            &[
                I::from(n_cells[0]).ok_or(GridConstructionError::IndexTypeTooSmallCellsPerDim)?,
                I::from(n_cells[1]).ok_or(GridConstructionError::IndexTypeTooSmallCellsPerDim)?,
                I::from(n_cells[2]).ok_or(GridConstructionError::IndexTypeTooSmallCellsPerDim)?,
            ],
            self.global_marching_cubes_grid.cell_size(),
        )
    }
}

/// Result of the subdomain decomposition procedure
pub(crate) struct Subdomains<I: Index> {
    // Flat subdomain coordinate indices (same order as the particle list)
    flat_subdomain_indices: Vec<I>,
    // Particles of each subdomain (including ghost particles)
    per_subdomain_particles: Vec<Vec<usize>>,
}

pub(crate) fn initialize_parameters<I: Index, R: Real>(
    parameters: &Parameters<R>,
    _particles: &[Vector3<R>],
    output_surface: &SurfaceReconstruction<I, R>,
) -> Result<ParametersSubdomainGrid<I, R>, anyhow::Error> {
    let chunk_size = 500;

    let SpatialDecomposition::UniformGrid(grid_parameters) = &parameters.spatial_decomposition
    else {
        return Err(anyhow!(
            "spatial decomposition parameters for uniform grid are missing"
        ));
    };

    // A subdomain will be a cube consisting of this number of MC cubes along each coordinate axis
    let subdomain_cubes_in = grid_parameters.subdomain_num_cubes_per_dim;
    let subdomain_cubes = I::from_u32(subdomain_cubes_in)
        .expect("number of subdomain cubes has to fit in index type");
    let subdomain_cubes_global = GlobalIndex::from_u32(subdomain_cubes_in)
        .expect("number of subdomain cubes has to fit in global index type");

    // Physical particle properties
    let particle_radius = parameters.particle_radius;
    let particle_rest_density = parameters.rest_density;
    let compact_support_radius = parameters.compact_support_radius;
    let cube_size = parameters.cube_size;
    let surface_threshold = parameters.iso_surface_threshold;

    let particle_rest_volume = kernel::Volume::cube_particle(particle_radius);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    let ghost_particle_margin =
        (compact_support_radius / cube_size).ceil() * cube_size * 1.01.convert();

    // Compute information of ghost margin volume for debugging
    {
        let ghost_margin_cubes = I::from((ghost_particle_margin / cube_size).ceil())
            .expect("ghost margin cube count has to fit in index type");

        let vol_subdomain = subdomain_cubes
            .checked_cubed()
            .expect("number of cubes per subdomain has to be representable in index type");
        let vol_margin = (ghost_margin_cubes * I::two() + subdomain_cubes)
            .checked_cubed()
            .expect(
                "number of cubes per subdomain with margin has to be representable in index type",
            )
            - vol_subdomain;

        info!(
            "The ghost margin volume per subdomain is {:.2}% of the subdomain volume",
            (vol_margin.to_real().unwrap_or(R::one())
                / vol_subdomain.to_real().unwrap_or(R::one()))
                * 100.0.convert()
        );
        info!(
            "The ghost margin per subdomain is {:.2} MC cells or {:.2} subdomains wide",
            ghost_particle_margin / cube_size,
            ghost_particle_margin / (cube_size * subdomain_cubes.to_real_unchecked())
        );
    }

    // AABB of the particles
    let aabb = output_surface.grid.aabb();

    let global_mc_grid = UniformCartesianCubeGrid3d::<GlobalIndex, R>::new(
        aabb.min(),
        &output_surface
            .grid
            .cells_per_dim()
            .map(|c| <GlobalIndex as NumCast>::from(c).unwrap()),
        cube_size,
    )
    .context("construct initial global marching cubes cell grid")?;
    trace!("Initial global MC Grid: {:?}", global_mc_grid);

    // MC cubes along each coordinate axis of the entire global MC background grid
    let cells_per_dim = global_mc_grid.cells_per_dim();
    // Compute the number of subdomains along each coordinate axis
    let num_subdomains = [
        int_ceil_div(cells_per_dim[0], subdomain_cubes_global),
        int_ceil_div(cells_per_dim[1], subdomain_cubes_global),
        int_ceil_div(cells_per_dim[2], subdomain_cubes_global),
    ];

    let num_global_mc_cells = (|| -> Option<_> {
        Some([
            num_subdomains[0].checked_mul(subdomain_cubes_global)?,
            num_subdomains[1].checked_mul(subdomain_cubes_global)?,
            num_subdomains[2].checked_mul(subdomain_cubes_global)?,
        ])
    })()
    .context("compute global number of marching cubes cells per dimension")?;

    let global_mc_grid = UniformCartesianCubeGrid3d::<GlobalIndex, R>::new(
        global_mc_grid.aabb().min(),
        &num_global_mc_cells,
        cube_size,
    )
    .context("construct final global marching cubes cell grid")?;
    trace!("Global MC Grid: {:?}", global_mc_grid);

    // Convert number of subdomains back to local index type
    let num_subdomains = (|| -> Option<_> {
        Some([
            I::from(num_subdomains[0])?,
            I::from(num_subdomains[1])?,
            I::from(num_subdomains[2])?,
        ])
    })()
    .context("convert number of subdomains per dimension to local index type")?;

    // Compute total number of subdomains
    let subdomain_count: I = (|| -> Option<_> {
        num_subdomains[0].checked_mul(&num_subdomains[1].checked_mul(&num_subdomains[2])?)
    })()
    .context("compute total number of subdomains")?;
    // Edge length of a subdomain in absolute units
    let subdomain_size = cube_size * subdomain_cubes.to_real_unchecked();
    // Background grid of the subdomains
    let subdomain_grid = UniformCartesianCubeGrid3d::<I, R>::new(
        global_mc_grid.aabb().min(),
        &num_subdomains,
        subdomain_size,
    )?;

    {
        let nc = subdomain_cubes;
        let [nx, ny, nz] = subdomain_grid.cells_per_dim();
        let [mc_x, mc_y, mc_z] = global_mc_grid.cells_per_dim();
        info!("Number of subdomains: {subdomain_count} ({nx}x{ny}x{nz})");
        info!(
            "Number of MC cells per subdomain: {} ({nc}x{nc}x{nc})",
            nc.cubed()
        );
        info!(
            "Number of MC cells globally: {} ({mc_x}x{mc_y}x{mc_z})",
            mc_x * mc_y * mc_z
        );
        trace!("Subdomain grid: {:?}", subdomain_grid);
    }

    Ok(ParametersSubdomainGrid {
        particle_radius,
        particle_rest_mass,
        compact_support_radius,
        surface_threshold,
        cube_size,
        subdomain_cubes,
        ghost_particle_margin,
        global_marching_cubes_grid: global_mc_grid,
        subdomain_grid,
        chunk_size,
        global_neighborhood_list: parameters.global_neighborhood_list,
        kernel_type: parameters.kernel_type.clone(),
    })
}

#[allow(unused)]
pub(crate) fn extract_narrow_band<I: Index, R: Real>(
    parameters: &ParametersSubdomainGrid<I, R>,
    particles: &[Vector3<R>],
) -> Vec<Vector3<R>> {
    profile!("Filter narrow band");

    let compact_support_radius = parameters.compact_support_radius;
    let ghost_particle_margin = (compact_support_radius / parameters.cube_size).ceil()
        * parameters.cube_size
        * 1.01.convert();

    // AABB of the particles
    let aabb = {
        let mut aabb = Aabb3d::<R>::par_from_points(particles);
        // Add some safety margin, this should be large enough such that all mesh vertices are guaranteed to be inside of it
        aabb.grow_uniformly(ghost_particle_margin);
        info!("Enlarged AABB: {:?}", aabb);
        aabb
    };

    let neighbor_lists = {
        profile!("Global neighborhood search");
        let mut neighbors = Vec::new();
        neighborhood_search_spatial_hashing_parallel::<GlobalIndex, _>(
            &aabb,
            particles,
            compact_support_radius,
            &mut neighbors,
        );
        neighbors
    };

    let narrow_band = {
        profile!("Identify narrow band");
        let surface_particle_neighbors = 30;
        let surface_particles = neighbor_lists
            .iter()
            .enumerate()
            .filter_map(|(i, nl)| (nl.len() < surface_particle_neighbors).then_some(i))
            .collect::<Vec<_>>();

        info!("");
        info!(
            "Number of pure \"surface particles\": {}",
            surface_particles.len()
        );

        let mut first_ring = surface_particles
            .iter()
            .copied()
            .flat_map(|i| neighbor_lists[i].iter().copied())
            .collect::<Vec<_>>();
        info!("First ring before dedup: {}", first_ring.len());
        {
            profile!("Dedup first ring");
            first_ring.sort_unstable();
            first_ring.dedup();
        }
        info!("First ring after dedup: {}", first_ring.len());

        let mut second_ring = first_ring
            .iter()
            .copied()
            .flat_map(|i| neighbor_lists[i].iter().copied())
            .collect::<Vec<_>>();
        info!("Second ring before dedup: {}", second_ring.len());
        {
            profile!("Dedup second ring");
            second_ring.sort_unstable();
            second_ring.dedup();
        }
        info!("Second ring after dedup: {}", second_ring.len());

        let mut narrow_band = surface_particles;
        narrow_band.append(&mut first_ring);
        narrow_band.append(&mut second_ring);
        info!("All before dedup: {}", narrow_band.len());
        {
            profile!("Dedup entire narrow band");
            narrow_band.sort_unstable();
            narrow_band.dedup();
        }
        info!(
            "All after dedup: {} ({:.3}%), interior particles: {}",
            narrow_band.len(),
            (narrow_band.len() as f64 / particles.len() as f64) * 100.0,
            particles.len() - narrow_band.len()
        );
        narrow_band
    };

    {
        profile!("Collect narrow band positions");

        narrow_band
            .into_iter()
            .map(|i| particles[i])
            .collect::<Vec<_>>()
    }
}

/// Performs classification and decomposition of particles into a regular grid of subdomains
pub(crate) fn decomposition<
    I: Index,
    R: Real,
    C: subdomain_classification::ParticleToSubdomainClassifier<I, R>,
>(
    parameters: &ParametersSubdomainGrid<I, R>,
    particles: &[Vector3<R>],
) -> Result<Subdomains<I>, anyhow::Error> {
    profile!("decomposition");
    info!("Starting classification of particles into subdomains.");

    // Count the number of particles and ghost particles per subdomain (with thread local counters)
    let per_subdomain_counter_tls = ThreadLocal::<RefCell<MapType<I, usize>>>::new();
    {
        profile!("classifying particles");

        particles
            .par_chunks(parameters.chunk_size)
            .for_each(|particle_chunk| {
                let mut per_subdomain_counter = per_subdomain_counter_tls
                    .get_or(|| RefCell::new(new_map()))
                    .borrow_mut();
                let mut classifier = C::new();

                for particle in particle_chunk.iter() {
                    classifier.classify_particle(
                        particle,
                        &parameters.subdomain_grid,
                        parameters.ghost_particle_margin,
                    );
                    for i in 0..classifier.len() {
                        let flat_subdomain_idx = classifier.get(i);
                        *per_subdomain_counter.entry(flat_subdomain_idx).or_insert(0) += 1;
                    }
                }
            });
    }

    // Merge all thread local subdomain particle counters
    let global_per_subdomain_counter = new_parallel_map();
    {
        profile!("merging TL per cell particle counters");

        let per_subdomain_counter_tls = per_subdomain_counter_tls
            .into_iter()
            .map(RefCell::into_inner)
            .collect::<Vec<_>>();

        per_subdomain_counter_tls
            .into_par_iter()
            .for_each(|per_cell_counter| {
                for (flat_cell_index, count) in per_cell_counter {
                    *global_per_subdomain_counter
                        .entry(flat_cell_index)
                        .or_insert(0) += count;
                }
            });
    }

    // Mapping from flat subdomain coordinate index to offset into contiguous subdomain storage
    let mut subdomain_compressed_indices = new_map();
    // Inverse mapping (offset to flat subdomain coordinate)
    let mut flat_subdomain_indices = vec![I::zero(); global_per_subdomain_counter.len()];

    let per_subdomain_particle_count: Vec<usize> = {
        profile!("initializing flat subdomain data and index mapping");

        global_per_subdomain_counter
            .into_iter()
            .enumerate()
            .map(|(i, (flat_cell_index, particle_count))| {
                subdomain_compressed_indices.insert(flat_cell_index, i);
                flat_subdomain_indices[i] = flat_cell_index;
                particle_count
            })
            .collect()
    };

    let mut per_subdomain_particles = Vec::with_capacity(per_subdomain_particle_count.len());
    per_subdomain_particles.resize_with(per_subdomain_particle_count.len(), || {
        Mutex::new(Vec::new())
    });

    {
        profile!("copying particles to subdomains");

        particles
            .par_chunks(parameters.chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, particle_chunk)| {
                let chunk_offset = chunk_idx * parameters.chunk_size;
                let mut classifier = C::new();
                for (particle_idx, particle) in particle_chunk.iter().enumerate() {
                    let particle_idx = chunk_offset + particle_idx;
                    classifier.classify_particle(
                        particle,
                        &parameters.subdomain_grid,
                        parameters.ghost_particle_margin,
                    );
                    for i in 0..classifier.len() {
                        let flat_subdomain_idx = classifier.get(i);

                        let compressed_subdomain_idx =
                            subdomain_compressed_indices[&flat_subdomain_idx];

                        // Lock the subdomain for writing
                        let mut subdomain_particles =
                            per_subdomain_particles[compressed_subdomain_idx].lock();
                        // Reserve full size of subdomain if it's still empty
                        if subdomain_particles.is_empty() {
                            let particle_count =
                                per_subdomain_particle_count[compressed_subdomain_idx];
                            subdomain_particles.reserve(particle_count);
                        }
                        // Add the particle to the subdomain
                        subdomain_particles.push(particle_idx);
                    }
                }
            });
    }

    // Remove mutexes
    let mut per_subdomain_particles = per_subdomain_particles
        .into_iter()
        .map(Mutex::into_inner)
        .collect::<Vec<_>>();

    // Sort subdomain particle sets by index so that overlapping particle regions of subdomains
    // will be in the same order
    {
        profile!("sort subdomain particles");
        per_subdomain_particles
            .par_iter_mut()
            .for_each(|particles| {
                //use rand::prelude::SliceRandom;
                //let mut rng = rand::thread_rng();
                //particles.shuffle(&mut rng)
                particles.sort_unstable();
            });
    }

    Ok(Subdomains {
        flat_subdomain_indices,
        per_subdomain_particles,
    })
}

pub(crate) fn compute_global_densities_and_neighbors<I: Index, R: Real, K: SymmetricKernel3d<R>>(
    parameters: &ParametersSubdomainGrid<I, R>,
    global_particles: &[Vector3<R>],
    subdomains: &Subdomains<I>,
) -> (Vec<R>, Vec<Vec<usize>>) {
    profile!(parent, "compute_global_density_vector");
    info!("Starting computation of global density vector.");

    let global_particle_densities = Mutex::new(vec![R::zero(); global_particles.len()]);
    let global_neighbors = Mutex::new(vec![Vec::new(); global_particles.len()]);

    #[derive(Default)]
    struct SubdomainWorkspace<R: Real> {
        // Particle positions of this subdomain
        subdomain_particles: Vec<Vector3<R>>,
        // Per particle neighborhood lists
        neighborhood_lists: FlatNeighborhoodList,
        // Per particle density values of this subdomain
        particle_densities: Vec<R>,
        // Per particle flag whether the particle is in the interior of this subdomain (non-ghost particle)
        is_inside: Vec<bool>,
    }

    let workspace_tls = ThreadLocal::<RefCell<SubdomainWorkspace<R>>>::new();

    subdomains
        .flat_subdomain_indices
        .par_iter()
        .copied()
        .zip(subdomains.per_subdomain_particles.par_iter())
        .for_each(|(flat_subdomain_idx, subdomain_particle_indices)| {
            profile!("subdomain density computation", parent = parent);

            // Obtain thread local workspace and clear it
            let mut workspace = workspace_tls.get_or_default().borrow_mut();

            let SubdomainWorkspace {
                subdomain_particles,
                neighborhood_lists,
                particle_densities,
                is_inside,
            } = &mut *workspace;

            let flat_subdomain_idx: I = flat_subdomain_idx;
            let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();

            // Collect all particle positions of this subdomain
            {
                profile!("collect subdomain data");
                gather_subdomain_data(
                    global_particles,
                    subdomain_particle_indices,
                    subdomain_particles,
                );
            }

            // Get the cell index and AABB of the subdomain
            let subdomain_idx = parameters
                .subdomain_grid
                .try_unflatten_cell_index(flat_subdomain_idx)
                .expect("Subdomain cell does not exist");
            let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);

            // AABB used for neighborhood search, must encompass all particles, including ghost particles
            let margin_aabb = {
                let mut margin_aabb = subdomain_aabb.clone();
                // TODO: Verify if we can omit this extra margin?
                margin_aabb.grow_uniformly(parameters.ghost_particle_margin * 1.5.convert());
                margin_aabb
            };

            {
                profile!("initialize particle filter");
                is_inside.clear();
                reserve_total(is_inside, subdomain_particle_indices.len());
                is_inside.extend(
                    subdomain_particles
                        .iter()
                        .map(|p| subdomain_aabb.contains_point(p)),
                );
            }

            neighborhood_search_spatial_hashing_flat_filtered::<I, R>(
                &margin_aabb,
                subdomain_particles,
                parameters.compact_support_radius,
                neighborhood_lists,
                |i| is_inside[i],
            );

            sequential_compute_particle_densities_filtered::<I, R, _, K>(
                subdomain_particles,
                neighborhood_lists,
                parameters.compact_support_radius,
                parameters.particle_rest_mass,
                particle_densities,
                |i| is_inside[i],
            );

            // Write particle densities into global storage
            {
                profile!("update global density values");
                // Lock global vector while this subdomain writes into it
                let mut global_particle_densities = global_particle_densities.lock();
                is_inside
                    .iter()
                    .copied()
                    .zip(
                        subdomain_particle_indices
                            .iter()
                            .copied()
                            .zip(particle_densities.iter().copied()),
                    )
                    // Update density values only for particles inside the subdomain (ghost particles have wrong values)
                    .filter(|(is_inside, _)| *is_inside)
                    .for_each(|(_, (particle_idx, density))| {
                        global_particle_densities[particle_idx] = density;
                    });
            }

            // Write particle neighbor lists into global storage
            if parameters.global_neighborhood_list {
                profile!("update global neighbor list");
                // Lock global vector while this subdomain writes into it
                let mut global_neighbors = global_neighbors.lock();
                is_inside
                    .iter()
                    .copied()
                    .zip(
                        subdomain_particle_indices
                            .iter()
                            .copied()
                            .zip(neighborhood_lists.iter()),
                    )
                    // Update density values only for particles inside of the subdomain (ghost particles have wrong values)
                    .filter(|(is_inside, _)| *is_inside)
                    .for_each(|(_, (particle_idx, neighbors))| {
                        global_neighbors[particle_idx] = neighbors
                            .iter()
                            .copied()
                            .map(|local| subdomain_particle_indices[local])
                            .collect();
                    });
            }
        });

    let global_particle_densities = global_particle_densities.into_inner();
    let global_neighbors = global_neighbors.into_inner();

    (global_particle_densities, global_neighbors)
}

pub(crate) struct SurfacePatch<I: Index, R: Real> {
    pub vertices: Vec<Vector3<R>>,
    pub triangles: Vec<[usize; 3]>,
    pub vertex_inside_count: usize,
    pub triangle_inside_count: usize,
    pub vertex_inside_flags: Vec<bool>,
    pub triangle_inside_flags: Vec<bool>,
    pub exterior_vertex_edge_indices: Vec<(I, EdgeIndex<I>)>,
}

/// Returns the lower and upper subdomain grid vertex indices possibly influenced by the given particle (upper index is exclusive)
#[inline(always)]
fn particle_influence_aabb<I: Index, R: Real>(
    particle: &Vector3<R>,
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<I, R>,
    cube_radius: I,
) -> ([I; 3], [I; 3]) {
    let extents = subdomain_mc_grid.points_per_dim();

    // Note: This assumes that enclosing_cell can return negative indices for ghost particles
    //  (they are outside the subdomain grid but part of the particle list).

    // Get grid cell containing particle
    let particle_cell = subdomain_mc_grid.enclosing_cell(particle);

    // Compute lower and upper bounds of the grid points possibly affected by the particle
    // We want to loop over the vertices of the enclosing cells plus all points in `cube_radius` distance from the cell

    let lower = [0, 1, 2].map(|i| {
        particle_cell[i]
            .saturating_sub(&cube_radius)
            .max(I::zero())
            .min(extents[i])
    });

    let upper = [0, 1, 2].map(|i| {
        // We add 2 because
        //  - we want to loop over all grid points of the cell (+1 for upper points) + the radius
        //  - the upper range limit is exclusive (+1)
        (particle_cell[i] + cube_radius + I::two())
            .min(extents[i])
            .max(I::zero())
    });

    (lower, upper)
}

/// Converts a local (subdomain) grid point index to a global grid point index
fn local_to_global_point_ijk<I: Index>(
    local_point_ijk: [I; 3],
    subdomain_ijk: [I; 3],
    cells_per_subdomain: [I; 3],
) -> [GlobalIndex; 3] {
    let local_point_ijk = local_point_ijk.map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
    let subdomain_ijk = subdomain_ijk.map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
    let cells_per_subdomain =
        cells_per_subdomain.map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
    let [i, j, k] = local_point_ijk;

    [
        subdomain_ijk[0] * cells_per_subdomain[0] + i,
        subdomain_ijk[1] * cells_per_subdomain[1] + j,
        subdomain_ijk[2] * cells_per_subdomain[2] + k,
    ]
}

/// Auto-dispatching density grid loop for f32/i64: chooses AVX2+FMA on x86, NEON on aarch64, otherwise scalar fallback
pub fn density_grid_loop_auto(
    levelset_grid: &mut [f32],
    subdomain_particles: &[Vector3<f32>],
    subdomain_particle_densities: &[f32],
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<i64, f32>,
    subdomain_ijk: &[i64; 3],
    global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
    cube_radius: i64,
    squared_support_with_margin: f32,
    particle_rest_mass: f32,
    kernel_support: f32,
    kernel_type: &KernelType,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return density_grid_loop_avx(
                    levelset_grid,
                    subdomain_particles,
                    subdomain_particle_densities,
                    subdomain_mc_grid,
                    subdomain_ijk,
                    global_mc_grid,
                    cube_radius,
                    squared_support_with_margin,
                    particle_rest_mass,
                    kernel_support,
                    kernel_type,
                );
            }
        }
    }

    // TODO: Not supported on arm (32 bit) until https://github.com/rust-lang/rust/issues/111800 is resolved
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return density_grid_loop_neon(
                    levelset_grid,
                    subdomain_particles,
                    subdomain_particle_densities,
                    subdomain_mc_grid,
                    subdomain_ijk,
                    global_mc_grid,
                    cube_radius,
                    squared_support_with_margin,
                    particle_rest_mass,
                    kernel_support,
                    kernel_type,
                );
            }
        }
    }

    // Fallback: scalar generic implementation
    match kernel_type {
        KernelType::CubicSpline => {
            density_grid_loop_scalar::<i64, f32, CubicSplineKernel<f32>>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                squared_support_with_margin,
                particle_rest_mass,
                &CubicSplineKernel::new(kernel_support),
            );
        }
        KernelType::Poly6 => {
            density_grid_loop_scalar::<i64, f32, Poly6Kernel<f32>>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                squared_support_with_margin,
                particle_rest_mass,
                &Poly6Kernel::new(kernel_support),
            );
        }
        KernelType::Spiky => {
            density_grid_loop_scalar::<i64, f32, SpikyKernel<f32>>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                squared_support_with_margin,
                particle_rest_mass,
                &SpikyKernel::new(kernel_support),
            );
        }
        KernelType::WendlandQuinticC2 => {
            density_grid_loop_scalar::<i64, f32, WendlandQuinticC2Kernel<f32>>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                squared_support_with_margin,
                particle_rest_mass,
                &WendlandQuinticC2Kernel::new(kernel_support),
            );
        }
    }
}

pub fn density_grid_loop_scalar<I: Index, R: Real, K: SymmetricKernel3d<R>>(
    levelset_grid: &mut [R],
    subdomain_particles: &[Vector3<R>],
    subdomain_particle_densities: &[R],
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<I, R>,
    subdomain_ijk: &[I; 3],
    global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, R>,
    cube_radius: I,
    squared_support_with_margin: R,
    particle_rest_mass: R,
    kernel: &K,
) {
    profile!("density grid loop (scalar)");
    let mc_grid = subdomain_mc_grid;

    for (p_i, rho_i) in subdomain_particles
        .iter()
        .copied()
        .zip(subdomain_particle_densities.iter().copied())
    {
        let (lower, upper) = particle_influence_aabb(&p_i, &mc_grid, cube_radius);

        // Loop over all grid points around the enclosing cell
        for i in I::range(lower[0], upper[0]).iter() {
            for j in I::range(lower[1], upper[1]).iter() {
                for k in I::range(lower[2], upper[2]).iter() {
                    let point_ijk = [i, j, k];
                    let local_point = mc_grid
                        .get_point(point_ijk)
                        .expect("point has to be part of the subdomain grid");

                    let mc_cells_per_subdomain = mc_grid.cells_per_dim();

                    // Use global coordinate calculation for consistency with neighboring domains
                    let global_point_ijk = local_to_global_point_ijk(
                        point_ijk,
                        *subdomain_ijk,
                        *mc_cells_per_subdomain,
                    );
                    let global_point = global_mc_grid
                        .get_point(global_point_ijk)
                        .expect("point has to be part of the global mc grid");
                    let point_coordinates = global_mc_grid.point_coordinates(&global_point);

                    let dx = p_i - point_coordinates;
                    let dx_norm_sq = dx.norm_squared();

                    if dx_norm_sq < squared_support_with_margin {
                        let v_i = particle_rest_mass / rho_i;
                        let r = dx_norm_sq.sqrt();
                        let w_ij = kernel.evaluate(r);
                        //let w_ij = kernel.evaluate(dx_norm_sq);

                        let interpolated_value = v_i * w_ij;

                        let flat_point_idx = mc_grid.flatten_point_index(&local_point);
                        let flat_point_idx = flat_point_idx.to_usize().unwrap();
                        levelset_grid[flat_point_idx] += interpolated_value;
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub fn density_grid_loop_neon(
    levelset_grid: &mut [f32],
    subdomain_particles: &[Vector3<f32>],
    subdomain_particle_densities: &[f32],
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<i64, f32>,
    subdomain_ijk: &[i64; 3],
    global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
    cube_radius: i64,
    _squared_support_with_margin: f32,
    particle_rest_mass: f32,
    kernel_support: f32,
    kernel_type: &KernelType,
) {
    use crate::kernel::NeonKernel;
    use core::arch::aarch64::*;
    const LANES: usize = 4;

    // Ensure that we can safely compute global MC vertex positions via u32 indices per dimension
    assert!(
        global_mc_grid
            .points_per_dim()
            .iter()
            .copied()
            .all(|p| p < u32::MAX as i64),
        "global marching cubes grid has too many vertices per dimension ({:?}) to fit into u32 (max: {}) for density_grid_loop_neon",
        global_mc_grid.points_per_dim(),
        u32::MAX
    );

    #[target_feature(enable = "neon")]
    fn density_grid_loop<K: NeonKernel>(
        levelset_grid: &mut [f32],
        subdomain_particles: &[Vector3<f32>],
        subdomain_particle_densities: &[f32],
        subdomain_mc_grid: &UniformCartesianCubeGrid3d<i64, f32>,
        subdomain_ijk: &[i64; 3],
        global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
        cube_radius: i64,
        _squared_support_with_margin: f32,
        particle_rest_mass: f32,
        kernel_support: f32,
    ) {
        let kernel_neon = K::new(kernel_support);

        profile!("density grid loop (neon)");
        let mc_grid = subdomain_mc_grid;
        let mc_points = mc_grid.points_per_dim();
        let dim_y = mc_points[1] as usize;
        let dim_z = mc_points[2] as usize;

        // Preload constants used inside the hot loops
        const K_OFFSETS_ARR: [u32; 4] = [0, 1, 2, 3];
        let k_offsets = unsafe { vld1q_u32(K_OFFSETS_ARR.as_ptr()) };
        let global_min = global_mc_grid.aabb().min();
        let min_z_v = vdupq_n_f32(global_min[2]);
        let cube_size = global_mc_grid.cell_size();
        let support = kernel_support;
        let support_sq_v = vdupq_n_f32(support * support);

        for (p_i, rho_i) in subdomain_particles
            .iter()
            .copied()
            .zip(subdomain_particle_densities.iter().copied())
        {
            let v_i = particle_rest_mass / rho_i;

            let (lower, upper) = particle_influence_aabb(&p_i, &mc_grid, cube_radius);
            let lower = lower.map(|i| i as usize);
            let upper = upper.map(|i| i as usize);

            let remainder = (upper[2] - lower[2]) % LANES;
            let upper_k_aligned = upper[2] - remainder;

            let subdomain_ijk = subdomain_ijk.map(|i| i as u32);
            let mc_cells = mc_grid.cells_per_dim().map(|i| i as u32);

            // Broadcast particle coordinates once per particle
            let particle_xs = vdupq_n_f32(p_i[0]);
            let particle_ys = vdupq_n_f32(p_i[1]);
            let particle_zs = vdupq_n_f32(p_i[2]);

            // Function to evaluate kernel contribution of current particle to 8 grid points at once
            let evaluate_contribution =
                |k: usize, grid_xs: float32x4_t, grid_ys: float32x4_t| -> float32x4_t {
                    // Compute global k for 4 consecutive points
                    let global_k_base = vdupq_n_u32(subdomain_ijk[2] * mc_cells[2] + k as u32);
                    let global_k = vaddq_u32(global_k_base, k_offsets);
                    let global_k_f32 = vcvtq_f32_u32(global_k);
                    let grid_zs = vmlaq_n_f32(min_z_v, global_k_f32, cube_size);
                    // Deltas
                    let dxs = vsubq_f32(particle_xs, grid_xs);
                    let dys = vsubq_f32(particle_ys, grid_ys);
                    let dzs = vsubq_f32(particle_zs, grid_zs);
                    // Distance squared
                    let dist_sq = vmlaq_f32(vmulq_f32(dxs, dxs), dys, dys);
                    let dist_sq = vmlaq_f32(dist_sq, dzs, dzs);
                    // Cheap mask to skip work if all lanes are outside support
                    let inside_mask = vcltq_f32(dist_sq, support_sq_v);
                    let any_inside = vmaxvq_u32(inside_mask);
                    if any_inside == 0 {
                        return vdupq_n_f32(0.0);
                    }
                    // Compute weights, then mask out lanes outside support
                    let dist = vsqrtq_f32(dist_sq);
                    let w = unsafe { kernel_neon.evaluate(dist) };
                    vbslq_f32(inside_mask, w, vdupq_n_f32(0.0))
                };

            // Loop over grid points in support region of the particle
            for i in lower[0]..upper[0] {
                for j in lower[1]..upper[1] {
                    let flat_index_base = (i * dim_y + j) * dim_z;

                    let global_i = subdomain_ijk[0] * mc_cells[0] + i as u32;
                    let global_j = subdomain_ijk[1] * mc_cells[1] + j as u32;

                    let grid_x = global_i as f32 * cube_size + global_min[0];
                    let grid_y = global_j as f32 * cube_size + global_min[1];
                    let grid_xs = vdupq_n_f32(grid_x);
                    let grid_ys = vdupq_n_f32(grid_y);

                    for k in (lower[2]..upper_k_aligned).step_by(LANES) {
                        let flat_point_idx = flat_index_base + k;
                        let w_ij = evaluate_contribution(k, grid_xs, grid_ys);

                        let prev_val = unsafe {
                            vld1q_f32(levelset_grid.as_ptr().offset(flat_point_idx as isize))
                        };
                        let val = vmlaq_n_f32(prev_val, w_ij, v_i);
                        unsafe {
                            vst1q_f32(
                                levelset_grid.as_mut_ptr().offset(flat_point_idx as isize),
                                val,
                            )
                        };
                    }

                    // Handle remainder
                    if remainder > 0 {
                        let flat_point_idx = flat_index_base + upper_k_aligned;
                        let w_ij = evaluate_contribution(upper_k_aligned, grid_xs, grid_ys);
                        let w_ij_v = vmulq_n_f32(w_ij, v_i);
                        let w_ij_v =
                            unsafe { std::mem::transmute::<float32x4_t, [f32; LANES]>(w_ij_v) };
                        for rr in 0..remainder {
                            levelset_grid[flat_point_idx + rr] += w_ij_v[rr];
                        }
                    }
                }
            }
        }
    }

    match kernel_type {
        KernelType::CubicSpline => {
            density_grid_loop::<kernel::CubicSplineKernelNeonF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::Poly6 => {
            density_grid_loop::<kernel::Poly6KernelNeonF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::Spiky => {
            density_grid_loop::<kernel::SpikyKernelNeonF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::WendlandQuinticC2 => {
            density_grid_loop::<kernel::WendlandQuinticC2KernelNeonF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn density_grid_loop_avx(
    levelset_grid: &mut [f32],
    subdomain_particles: &[Vector3<f32>],
    subdomain_particle_densities: &[f32],
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<i64, f32>,
    subdomain_ijk: &[i64; 3],
    global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
    cube_radius: i64,
    _squared_support_with_margin: f32,
    particle_rest_mass: f32,
    kernel_support: f32,
    kernel_type: &KernelType,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use crate::kernel::AvxKernel;

    const LANES: usize = 8;

    // Ensure that we can safely compute global MC vertex positions via i32 indices per dimension
    assert!(
        global_mc_grid
            .points_per_dim()
            .iter()
            .copied()
            .all(|p| p < i32::MAX as i64),
        "global marching cubes grid has too many vertices per dimension ({:?}) to fit into i32 (max: {}) for density_grid_loop_avx",
        global_mc_grid.points_per_dim(),
        i32::MAX
    );

    #[target_feature(enable = "avx2,fma")]
    fn density_grid_loop<K: AvxKernel>(
        levelset_grid: &mut [f32],
        subdomain_particles: &[Vector3<f32>],
        subdomain_particle_densities: &[f32],
        subdomain_mc_grid: &UniformCartesianCubeGrid3d<i64, f32>,
        subdomain_ijk: &[i64; 3],
        global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
        cube_radius: i64,
        _squared_support_with_margin: f32,
        particle_rest_mass: f32,
        kernel_support: f32,
    ) {
        let kernel_avx = K::new(kernel_support);

        profile!("density grid loop (avx)");
        let mc_grid = subdomain_mc_grid;
        let mc_points = mc_grid.points_per_dim();
        let dim_y = mc_points[1] as usize;
        let dim_z = mc_points[2] as usize;

        // Preload constants used inside the hot loops
        let k_offsets_i32 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let global_min = global_mc_grid.aabb().min();
        let min_z_v = _mm256_set1_ps(global_min[2]);
        let cube_size = global_mc_grid.cell_size();
        let cube_size_ps = _mm256_set1_ps(cube_size);
        let support = kernel_support;
        let support_sq_v = _mm256_set1_ps(support * support);

        for (p_i, rho_i) in subdomain_particles
            .iter()
            .copied()
            .zip(subdomain_particle_densities.iter().copied())
        {
            let v_i = particle_rest_mass / rho_i;

            let (lower, upper) = particle_influence_aabb(&p_i, &mc_grid, cube_radius);
            let lower = lower.map(|i| i as usize);
            let upper = upper.map(|i| i as usize);

            let remainder = (upper[2] - lower[2]) % LANES;
            let upper_k_aligned = upper[2] - remainder;

            let subdomain_ijk_i32 = subdomain_ijk.map(|i| i as i32);
            let mc_cells = mc_grid.cells_per_dim().map(|i| i as i32);

            // Broadcast particle coordinates once per particle
            let particle_xs = _mm256_set1_ps(p_i[0]);
            let particle_ys = _mm256_set1_ps(p_i[1]);
            let particle_zs = _mm256_set1_ps(p_i[2]);
            let v_i_ps = _mm256_set1_ps(v_i);

            // Function to evaluate kernel contribution of current particle to 8 grid points at once
            let evaluate_contribution = |k: usize, grid_xs: __m256, grid_ys: __m256| -> __m256 {
                // Compute global k for 8 consecutive points using i32 lanes and convert to f32
                let base_k = subdomain_ijk_i32[2] * mc_cells[2] + k as i32;
                let global_k_i32 = _mm256_add_epi32(_mm256_set1_epi32(base_k), k_offsets_i32);
                let global_k_f32 = _mm256_cvtepi32_ps(global_k_i32);
                let grid_zs = _mm256_fmadd_ps(global_k_f32, cube_size_ps, min_z_v);

                // Deltas
                let dxs = _mm256_sub_ps(particle_xs, grid_xs);
                let dys = _mm256_sub_ps(particle_ys, grid_ys);
                let dzs = _mm256_sub_ps(particle_zs, grid_zs);

                // Distance squared
                let dist_sq = {
                    let t = _mm256_fmadd_ps(dxs, dxs, _mm256_mul_ps(dys, dys));
                    _mm256_fmadd_ps(dzs, dzs, t)
                };

                // Cheap mask to skip work if all lanes are outside support
                let inside_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(dist_sq, support_sq_v);
                if _mm256_movemask_ps(inside_mask) == 0 {
                    return _mm256_set1_ps(0.0);
                }

                // Compute weights, then mask out lanes outside support
                let dist = _mm256_sqrt_ps(dist_sq);
                let w = unsafe { kernel_avx.evaluate(dist) };
                _mm256_and_ps(w, inside_mask)
            };

            // Loop over grid points in support region of the particle
            for i in lower[0]..upper[0] {
                for j in lower[1]..upper[1] {
                    let flat_index_base = (i * dim_y + j) * dim_z;

                    let global_i = subdomain_ijk_i32[0] * mc_cells[0] + i as i32;
                    let global_j = subdomain_ijk_i32[1] * mc_cells[1] + j as i32;

                    let grid_x = global_i as f32 * cube_size + global_min[0];
                    let grid_y = global_j as f32 * cube_size + global_min[1];
                    let grid_xs = _mm256_set1_ps(grid_x);
                    let grid_ys = _mm256_set1_ps(grid_y);

                    for k in (lower[2]..upper_k_aligned).step_by(LANES) {
                        let flat_point_idx = flat_index_base + k;
                        let w_ij = evaluate_contribution(k, grid_xs, grid_ys);

                        unsafe {
                            let prev_val =
                                _mm256_loadu_ps(levelset_grid.as_ptr().add(flat_point_idx));
                            let val = _mm256_fmadd_ps(w_ij, v_i_ps, prev_val);
                            _mm256_storeu_ps(levelset_grid.as_mut_ptr().add(flat_point_idx), val);
                        }
                    }

                    // Handle remainder
                    if remainder > 0 {
                        let flat_point_idx = flat_index_base + upper_k_aligned;
                        let w_ij = evaluate_contribution(upper_k_aligned, grid_xs, grid_ys);
                        unsafe {
                            let val = _mm256_mul_ps(w_ij, v_i_ps);
                            let mut tmp: [f32; LANES] = [0.0; LANES];
                            _mm256_storeu_ps(tmp.as_mut_ptr(), val);
                            for rr in 0..remainder {
                                levelset_grid[flat_point_idx + rr] += tmp[rr];
                            }
                        }
                    }
                }
            }
        }
    }

    match kernel_type {
        KernelType::CubicSpline => {
            density_grid_loop::<kernel::CubicSplineKernelAvxF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::Poly6 => {
            density_grid_loop::<kernel::Poly6KernelAvxF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::Spiky => {
            density_grid_loop::<kernel::SpikyKernelAvxF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
        KernelType::WendlandQuinticC2 => {
            density_grid_loop::<kernel::WendlandQuinticC2KernelAvxF32>(
                levelset_grid,
                subdomain_particles,
                subdomain_particle_densities,
                subdomain_mc_grid,
                subdomain_ijk,
                global_mc_grid,
                cube_radius,
                _squared_support_with_margin,
                particle_rest_mass,
                kernel_support,
            );
        }
    }
}

pub fn density_grid_loop_sparse<I: Index, R: Real, K: SymmetricKernel3d<R>>(
    levelset_grid: &mut [R],
    index_cache: &mut Vec<I>,
    subdomain_particles: &[Vector3<R>],
    subdomain_particle_densities: &[R],
    subdomain_mc_grid: &UniformCartesianCubeGrid3d<I, R>,
    subdomain_ijk: &[I; 3],
    global_mc_grid: &UniformCartesianCubeGrid3d<GlobalIndex, R>,
    cube_radius: I,
    squared_support_with_margin: R,
    particle_rest_mass: R,
    surface_threshold: R,
    kernel: &K,
) {
    profile!("density grid loop (sparse)");
    let mc_grid = subdomain_mc_grid;

    for (p_i, rho_i) in subdomain_particles
        .iter()
        .copied()
        .zip(subdomain_particle_densities.iter().copied())
    {
        let (lower, upper) = particle_influence_aabb(&p_i, &mc_grid, cube_radius);

        // Loop over all grid points around the enclosing cell
        for i in I::range(lower[0], upper[0]).iter() {
            for j in I::range(lower[1], upper[1]).iter() {
                for k in I::range(lower[2], upper[2]).iter() {
                    let point_ijk = [i, j, k];
                    let local_point = mc_grid
                        .get_point(point_ijk)
                        .expect("point has to be part of the subdomain grid");

                    let mc_cells_per_subdomain = mc_grid.cells_per_dim();

                    // Use global coordinate calculation for consistency with neighboring domains
                    let global_point_ijk = local_to_global_point_ijk(
                        point_ijk,
                        *subdomain_ijk,
                        *mc_cells_per_subdomain,
                    );
                    let global_point = global_mc_grid
                        .get_point(global_point_ijk)
                        .expect("point has to be part of the global mc grid");
                    let point_coordinates = global_mc_grid.point_coordinates(&global_point);

                    let dx = p_i - point_coordinates;
                    let dx_norm_sq = dx.norm_squared();

                    if dx_norm_sq < squared_support_with_margin {
                        let v_i = particle_rest_mass / rho_i;
                        let r = dx_norm_sq.sqrt();
                        let w_ij = kernel.evaluate(r);
                        //let w_ij = kernel.evaluate(dx_norm_sq);

                        let interpolated_value = v_i * w_ij;

                        let flat_point_idx = mc_grid.flatten_point_index(&local_point);
                        let flat_point_idx = flat_point_idx.to_usize().unwrap();
                        levelset_grid[flat_point_idx] += interpolated_value;

                        if levelset_grid[flat_point_idx] > surface_threshold {
                            for c in mc_grid
                                .cells_adjacent_to_point(
                                    &mc_grid.get_point_neighborhood(&local_point),
                                )
                                .iter()
                                .flatten()
                            {
                                let flat_cell_index = mc_grid.flatten_cell_index(c);
                                index_cache.push(flat_cell_index);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn reconstruct_subdomains<I: Index, R: Real, K: SymmetricKernel3d<R> + Sync>(
    parameters: &ParametersSubdomainGrid<I, R>,
    global_particles: &[Vector3<R>],
    global_particle_densities: &[R],
    subdomains: &Subdomains<I>,
    enable_simd: bool,
) -> Vec<SurfacePatch<I, R>> {
    profile!(parent, "reconstruction");

    let squared_support = parameters.compact_support_radius * parameters.compact_support_radius;
    // Add 1% so that we don't exclude grid points that are just on the kernel boundary
    let squared_support_with_margin = squared_support * 1.01.convert();
    // Compute radial distance in terms of grid points we have to evaluate for each particle
    let cube_radius = I::from((parameters.compact_support_radius / parameters.cube_size).ceil())
        .expect("kernel radius in cubes has to fit in index type");
    // Kernel
    let kernel = K::new(parameters.compact_support_radius);
    //let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, parameters.compact_support_radius);

    let mc_total_cells = parameters.subdomain_cubes.cubed();
    let mc_total_points = (parameters.subdomain_cubes + I::one()).cubed();

    assert!(
        mc_total_points.to_usize().is_some(),
        "number of mc cubes per subdomain must be fit into usize"
    );

    let max_particles = subdomains
        .per_subdomain_particles
        .iter()
        .map(|p| p.len())
        .max()
        .unwrap_or(0);
    info!("Largest subdomain has {} particles.", max_particles);

    // Maximum number of particles such that a subdomain will be considered "sparse" (5% of max)
    let sparse_limit = (max_particles / (100 / 5)).max(100);
    info!(
        "Subdomains with {} or less particles will be considered sparse.",
        sparse_limit
    );

    info!("Starting reconstruction (level-set evaluation and local triangulation).");

    // Returns a unique identifier for any edge index of a subdomain that can be later used for stitching
    let globalize_local_edge = |mc_grid: &UniformCartesianCubeGrid3d<I, R>,
                                subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
                                subdomain_index: I,
                                local_edge: &EdgeIndex<I>|
     -> (I, EdgeIndex<I>) {
        // We globalize the boundary edge index by translating the local edge index to the subdomain
        // where it lies on the lower boundary of that domain.

        let max_mc_point_index = mc_grid.points_per_dim().map(|i| i - I::one());
        let max_subdomain_index = subdomain_grid
            .cells_per_dim()
            .map(|i| i.saturating_sub(&I::one()).max(I::zero()));

        // Check along which axes this edge is on the max boundary
        let is_max = local_edge.axis().orthogonal_axes().map(|orth_axis| {
            if local_edge.origin().index()[orth_axis.dim()] == max_mc_point_index[orth_axis.dim()] {
                // We are on the max side of this domain along the axis
                true
            } else {
                // We are either
                //  - On the min side of this domain along the axis
                //  - Somewhere in the middle (in this case this axis is irrelevant)
                false
            }
        });

        if !is_max[0] && !is_max[1] {
            // Edge is already in the correct subdomain
            (subdomain_index, *local_edge)
        } else {
            // We have to translate to the neighboring subdomain (+1 in all directions where is_max == true)
            let subdomain_cell = subdomain_grid
                .try_unflatten_cell_index(subdomain_index)
                .expect("invalid subdomain index");

            let mut target_subdomain_ijk = *subdomain_cell.index();
            let mut target_local_origin_ijk = *local_edge.origin().index();

            // Obtain index of new subdomain and new origin point
            for (&orth_axis, &is_max) in local_edge
                .axis()
                .orthogonal_axes()
                .iter()
                .zip(is_max.iter())
            {
                if is_max {
                    // Clamp the step to the subdomain grid because we are not interested in subdomains outside the grid
                    // (globalization is not needed on the outermost boundary of the entire problem domain)
                    target_subdomain_ijk[orth_axis.dim()] = (target_subdomain_ijk[orth_axis.dim()]
                        + I::one())
                    .min(max_subdomain_index[orth_axis.dim()]);
                    // Move origin point from max boundary to min boundary
                    target_local_origin_ijk[orth_axis.dim()] = I::zero();
                }
            }

            let target_subdomain = subdomain_grid
                .get_cell(target_subdomain_ijk)
                .expect("target subdomain has to exist");
            let flat_target_subdomain = subdomain_grid.flatten_cell_index(&target_subdomain);

            // We re-use the same marching cubes domain here because the domain is anyway rectangular,
            // therefore this shift gives the same result
            let new_local_edge = mc_grid
                .get_edge(target_local_origin_ijk, local_edge.axis())
                .expect("failed to translate edge");

            (flat_target_subdomain, new_local_edge)
        }
    };

    #[derive(Default)]
    struct SubdomainWorkspace<I: Index, R: Real> {
        // Particle positions of this subdomain
        subdomain_particles: Vec<Vector3<R>>,
        // Per particle density values of this subdomain
        subdomain_particle_densities: Vec<R>,
        // Cache for the level-set values
        levelset_grid: Vec<R>,
        // Cache for indices
        index_cache: Vec<I>,
    }

    let workspace_tls = ThreadLocal::<RefCell<SubdomainWorkspace<I, R>>>::new();

    let reconstruct_patch = |flat_subdomain_idx: I,
                             subdomain_particle_indices: &Vec<usize>,
                             sparse: bool| {
        // Obtain thread local workspace and clear it
        let mut workspace = workspace_tls.get_or_default().borrow_mut();

        let SubdomainWorkspace {
            subdomain_particles,
            subdomain_particle_densities,
            levelset_grid,
            index_cache,
        } = &mut *workspace;

        let flat_subdomain_idx: I = flat_subdomain_idx;
        let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();

        // Collect all particle positions and densities of this subdomain
        {
            //profile!("collect subdomain data");
            gather_subdomain_data(
                global_particles,
                subdomain_particle_indices,
                subdomain_particles,
            );
            gather_subdomain_data(
                global_particle_densities,
                subdomain_particle_indices,
                subdomain_particle_densities,
            );
        }

        // Get the cell index and AABB of the subdomain
        let subdomain_idx = parameters
            .subdomain_grid
            .try_unflatten_cell_index(flat_subdomain_idx)
            .expect("Subdomain cell does not exist");
        let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);

        let mc_grid = UniformCartesianCubeGrid3d::new(
            subdomain_aabb.min(),
            &[parameters.subdomain_cubes; 3],
            parameters.cube_size,
        )
        .unwrap();

        levelset_grid.fill(R::zero());
        levelset_grid.resize(mc_total_points.to_usize().unwrap(), R::zero());

        index_cache.clear();

        if sparse {
            density_grid_loop_sparse(
                levelset_grid.as_mut_slice(),
                index_cache,
                subdomain_particles.as_slice(),
                subdomain_particle_densities.as_slice(),
                &mc_grid,
                &subdomain_idx.index(),
                &parameters.global_marching_cubes_grid,
                cube_radius,
                squared_support_with_margin,
                parameters.particle_rest_mass,
                parameters.surface_threshold,
                &kernel,
            );
        } else {
            use std::any::TypeId;
            use std::mem::transmute;
            if enable_simd
                && TypeId::of::<I>() == TypeId::of::<i64>()
                && TypeId::of::<R>() == TypeId::of::<f32>()
            {
                density_grid_loop_auto(
                    unsafe { transmute::<&mut [R], &mut [f32]>(levelset_grid.as_mut_slice()) },
                    unsafe {
                        transmute::<&[Vector3<R>], &[Vector3<f32>]>(subdomain_particles.as_slice())
                    },
                    unsafe { transmute::<&[R], &[f32]>(subdomain_particle_densities.as_slice()) },
                    unsafe {
                        transmute::<
                            &UniformCartesianCubeGrid3d<I, R>,
                            &UniformCartesianCubeGrid3d<i64, f32>,
                        >(&mc_grid)
                    },
                    &subdomain_idx.index().map(|i| i.to_i64().unwrap()),
                    unsafe {
                        transmute::<
                            &UniformCartesianCubeGrid3d<GlobalIndex, R>,
                            &UniformCartesianCubeGrid3d<GlobalIndex, f32>,
                        >(&parameters.global_marching_cubes_grid)
                    },
                    cube_radius.to_i64().unwrap(),
                    squared_support_with_margin.to_f32().unwrap(),
                    parameters.particle_rest_mass.to_f32().unwrap(),
                    parameters.compact_support_radius.to_f32().unwrap(),
                    &parameters.kernel_type,
                );
            } else {
                density_grid_loop_scalar(
                    levelset_grid.as_mut_slice(),
                    subdomain_particles.as_slice(),
                    subdomain_particle_densities.as_slice(),
                    &mc_grid,
                    &subdomain_idx.index(),
                    &parameters.global_marching_cubes_grid,
                    cube_radius,
                    squared_support_with_margin,
                    parameters.particle_rest_mass,
                    &kernel,
                );
            }
        }

        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        let mut vertex_inside_count = 0;
        let mut triangle_inside_count = 0;

        let mut vertex_inside_flags = Vec::new();
        let mut triangle_inside_flags = Vec::new();

        let mut exterior_vertex_edge_indices = Vec::new();

        let mut edge_to_vertex = new_map();

        let mut triangulate_cell = |flat_cell_idx: I| {
            let cell = mc_grid.try_unflatten_cell_index(flat_cell_idx).unwrap();

            let mut vertices_inside = [true; 8];
            let mut any_inside = false;
            for local_point_index in 0..8 {
                let point = cell.global_point_index_of(local_point_index).unwrap();
                let flat_point_idx = mc_grid.flatten_point_index(&point);
                let flat_point_idx = flat_point_idx.to_usize().unwrap();
                // Get value of density map
                let density_value = levelset_grid[flat_point_idx];
                // Update inside/outside surface flag
                let vertex_inside = density_value > parameters.surface_threshold;
                any_inside |= vertex_inside;
                vertices_inside[local_point_index] = vertex_inside;
            }

            if !any_inside {
                // No vertices inside surface, skip triangulation
                return;
            }

            for triangle in marching_cubes_triangulation_iter(&vertices_inside) {
                let mut global_triangle = [0; 3];
                for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
                    let edge = cell
                        .global_edge_index_of(local_edge_index as usize)
                        .unwrap();
                    let vertex_index = *edge_to_vertex.entry(edge).or_insert_with(|| {
                        // TODO: Nonlinear interpolation

                        let origin_coords = mc_grid.point_coordinates(edge.origin());
                        let target_coords = mc_grid.point_coordinates(&edge.target());

                        let flat_origin_idx = mc_grid
                            .flatten_point_index(edge.origin())
                            .to_usize()
                            .unwrap();
                        let flat_target_idx = mc_grid
                            .flatten_point_index(&edge.target())
                            .to_usize()
                            .unwrap();

                        let origin_value = levelset_grid[flat_origin_idx];
                        let target_value = levelset_grid[flat_target_idx];

                        let alpha = (parameters.surface_threshold - origin_value)
                            / (target_value - origin_value);
                        let interpolated_coords =
                            origin_coords * (R::one() - alpha) + target_coords * alpha;
                        let vertex_coords = interpolated_coords;

                        vertices.push(vertex_coords);
                        let vertex_index = vertices.len() - 1;

                        let is_interior_vertex = !mc_grid.is_boundary_edge(&edge);
                        vertex_inside_count += is_interior_vertex as usize;
                        vertex_inside_flags.push(is_interior_vertex);

                        if !is_interior_vertex {
                            exterior_vertex_edge_indices.push(globalize_local_edge(
                                &mc_grid,
                                &parameters.subdomain_grid,
                                flat_subdomain_idx,
                                &edge,
                            ));
                        }

                        vertex_index
                    });

                    global_triangle[v_idx] = vertex_index;
                }

                let all_tri_vertices_inside = global_triangle
                    .iter()
                    .copied()
                    .all(|v_idx| vertex_inside_flags[v_idx]);

                triangles.push(global_triangle);
                triangle_inside_count += all_tri_vertices_inside as usize;
                triangle_inside_flags.push(all_tri_vertices_inside);
            }
        };

        {
            profile!("mc triangulation loop");

            if sparse {
                index_cache.sort_unstable();
                for flat_cell_idx in index_cache.iter().copied().dedup() {
                    triangulate_cell(flat_cell_idx);
                }
            } else {
                for flat_cell_idx in I::range(I::zero(), mc_total_cells).iter() {
                    triangulate_cell(flat_cell_idx);
                }
            }
        }

        SurfacePatch {
            vertices,
            triangles,
            vertex_inside_count,
            triangle_inside_count,
            vertex_inside_flags,
            triangle_inside_flags,
            exterior_vertex_edge_indices,
        }
    };

    let mut surface_patches = Vec::with_capacity(subdomains.flat_subdomain_indices.len());
    subdomains
        .flat_subdomain_indices
        .par_iter()
        .copied()
        .zip(subdomains.per_subdomain_particles.par_iter())
        .map(|(flat_subdomain_idx, subdomain_particle_indices)| {
            // TODO: Decision should consider size of compact support radius in relation to subdomain
            //  If support radius is huge, the dense method might be better even for few particles
            if subdomain_particle_indices.len() <= sparse_limit {
                profile!("subdomain reconstruction (sparse)", parent = parent);
                reconstruct_patch(flat_subdomain_idx, subdomain_particle_indices, true)
            } else {
                profile!("subdomain reconstruction (dense)", parent = parent);
                reconstruct_patch(flat_subdomain_idx, subdomain_particle_indices, false)
            }
        })
        .collect_into_vec(&mut surface_patches);

    surface_patches
}

pub(crate) fn stitching<I: Index, R: Real>(
    surface_patches: Vec<SurfacePatch<I, R>>,
) -> TriMesh3d<R> {
    profile!("stitching");
    info!("Starting stitching of subdomain meshes to global mesh.");

    // Calculate offsets of interior vertices and triangles
    let vert_and_tri_offsets = {
        profile!("surface patch offset scan");

        std::iter::once((0, 0))
            .chain(surface_patches.iter().scan((0, 0), |offsets, patch| {
                let (vert_offset, tri_offset) = offsets;
                *vert_offset += patch.vertex_inside_count;
                *tri_offset += patch.triangle_inside_count;
                Some(*offsets)
            }))
            .collect::<Vec<_>>()
    };

    let (total_interior_vert_count, total_interior_tri_count) = vert_and_tri_offsets
        .last()
        .copied()
        .expect("there has to be at least one entry in the offset list");

    let mut interior_vertices = vec![Vector3::<R>::zeros(); total_interior_vert_count];
    let mut interior_triangles = vec![[0, 0, 0]; total_interior_tri_count];

    let mut exterior_vertices = Vec::new();
    let mut exterior_triangles = Vec::new();
    let mut exterior_vertex_mapping = new_map();

    {
        profile!("copy interior verts/tris and deduplicate exterior verts");

        if vert_and_tri_offsets.len() > 1 {
            vert_and_tri_offsets
                .windows(2)
                .zip(surface_patches.iter())
                .for_each(|(offsets, patch)| {
                    if let [start_offsets, end_offsets] = offsets {
                        let (start_verts, start_tris) = *start_offsets;
                        let (end_verts, end_tris) = *end_offsets;

                        let global_vertex_offset = start_verts;

                        let mut local_to_global_vertex_mapping = vec![0; patch.vertices.len()];

                        // Copy interior vertices
                        {
                            let out_verts = &mut interior_vertices[start_verts..end_verts];

                            // Copy all interior vertices into global storage
                            patch
                                .vertices
                                .iter()
                                .zip(patch.vertex_inside_flags.iter())
                                .enumerate()
                                // Skip all exterior vertices
                                .filter_map(|(i, (v, is_interior))| is_interior.then_some((i, v)))
                                .enumerate()
                                .for_each(|(new_local_idx, (old_local_idx, vert))| {
                                    out_verts[new_local_idx] = *vert;
                                    local_to_global_vertex_mapping[old_local_idx] =
                                        global_vertex_offset + new_local_idx;
                                });
                        }

                        // Copy interior triangles
                        {
                            let out_tris = &mut interior_triangles[start_tris..end_tris];

                            // Copy all interior triangle into global storage
                            patch
                                .triangles
                                .iter()
                                .zip(patch.triangle_inside_flags.iter())
                                // Skip all exterior triangles
                                .filter_map(|(tri, is_interior)| is_interior.then_some(tri))
                                // Count only the interior triangles
                                .enumerate()
                                .for_each(|(tri_idx, tri)| {
                                    out_tris[tri_idx] = [
                                        local_to_global_vertex_mapping[tri[0]],
                                        local_to_global_vertex_mapping[tri[1]],
                                        local_to_global_vertex_mapping[tri[2]],
                                    ];
                                });
                        }

                        // Insert & deduplicate exterior vertices
                        {
                            patch
                                .vertices
                                .iter()
                                .zip(patch.vertex_inside_flags.iter())
                                .enumerate()
                                // Skip all interior vertices
                                .filter_map(|(i, (v, is_interior))| {
                                    (!is_interior).then_some((i, v))
                                })
                                // For each exterior vertex there is a corresponding globalized edge index
                                .zip(patch.exterior_vertex_edge_indices.iter())
                                .for_each(|((old_local_idx, vert), edge_index)| {
                                    let global_index = *exterior_vertex_mapping
                                        .entry(*edge_index)
                                        .or_insert_with(|| {
                                            // Exterior vertices will come after all interior vertices in the mesh
                                            let global_index =
                                                total_interior_vert_count + exterior_vertices.len();
                                            exterior_vertices.push(*vert);
                                            global_index
                                        });
                                    local_to_global_vertex_mapping[old_local_idx] = global_index;
                                });
                        }

                        // Insert exterior triangles
                        {
                            patch
                                .triangles
                                .iter()
                                .zip(patch.triangle_inside_flags.iter())
                                // Skip all exterior triangles
                                .filter_map(|(tri, is_interior)| (!is_interior).then_some(tri))
                                .for_each(|tri| {
                                    exterior_triangles.push(tri.map(|local_vert| {
                                        local_to_global_vertex_mapping[local_vert]
                                    }))
                                });
                        }
                    }
                });
        }
    }

    let mut vertices = interior_vertices;
    vertices.append(&mut exterior_vertices);

    let mut triangles = interior_triangles;
    triangles.append(&mut exterior_triangles);

    TriMesh3d {
        vertices,
        triangles,
    }
}

pub(crate) mod subdomain_classification {
    use super::*;

    /// Trait for assignment of particles to their subdomains
    pub trait ParticleToSubdomainClassifier<I: Index, R: Real> {
        /// Constructs a new classifier of this type
        fn new() -> Self;
        /// Classifies a particle into all subdomains it belongs to
        fn classify_particle(
            &mut self,
            particle: &Vector3<R>,
            subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
            ghost_particle_margin: R,
        );
        /// Returns the number of subdomains that were assigned to a particle in the last call to `classify_particle`
        fn len(&self) -> usize;
        /// Returns the `i`-th subdomain assigned to a particle in the last call to `classify_particle`
        fn get(&self, i: usize) -> I;
    }

    /// Classifier that assign a particle to its "owning" subdomain and all subdomains where it's a ghost particle
    pub struct GhostMarginClassifier<I: Index> {
        subdomains: Vec<I>,
    }

    impl<I: Index, R: Real> ParticleToSubdomainClassifier<I, R> for GhostMarginClassifier<I> {
        fn new() -> Self {
            GhostMarginClassifier {
                subdomains: Vec::with_capacity(27),
            }
        }

        fn classify_particle(
            &mut self,
            particle: &Vector3<R>,
            subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
            ghost_particle_margin: R,
        ) {
            self.subdomains.clear();
            ghost_particle_classification(
                particle,
                subdomain_grid,
                ghost_particle_margin,
                &mut self.subdomains,
            )
        }

        #[inline(always)]
        fn len(&self) -> usize {
            self.subdomains.len()
        }

        #[inline(always)]
        fn get(&self, i: usize) -> I {
            self.subdomains[i]
        }
    }

    /// Assigns a particle into the subdomain that contains it and all subdomains where it's a ghost particle.
    fn ghost_particle_classification<I: Index, R: Real>(
        particle: &Vector3<R>,
        subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
        ghost_particle_margin: R,
        subdomains: &mut Vec<I>,
    ) {
        // Find the owning subdomain of the particle
        let subdomain_ijk = subdomain_grid.enclosing_cell(particle);
        // Make sure particle is part of computational domain
        if subdomain_grid.get_cell(subdomain_ijk).is_none() {
            // TODO: When/why does this occur?
            return;
        }

        let dx = subdomain_grid.cell_size();

        // The number of subdomains that have to be checked in each direction
        let subdomain_radius = I::from((ghost_particle_margin / dx).ceil())
            .map(|i| i.to_i32())
            .flatten()
            .expect(
                "ghost particle margin relative to subdomains has to fit in index type and i32",
            );

        // Get corner points spanning the owning subdomain
        let subdomain_aabb = subdomain_grid.cell_aabb(
            &subdomain_grid
                .get_cell(subdomain_ijk)
                .expect("Subdomain has to be part of grid"),
        );
        let min_corner = subdomain_aabb.min();
        let max_corner = subdomain_aabb.max();

        // Checks whether the current particle is within the ghost particle margin of the half plane reached by the given step and along an axis
        let is_in_ghost_margin_single_dim = |step: i32, axis: usize| -> bool {
            let step_offset_real = R::from(step.abs() - 1).unwrap();
            if step > 0 {
                ((max_corner[axis] + step_offset_real * dx) - particle[axis])
                    < ghost_particle_margin
            } else if step < 0 {
                (particle[axis] - (min_corner[axis] - step_offset_real * dx))
                    < ghost_particle_margin
            } else {
                // step == 0
                true
            }
        };

        // Checks whether the current particle is within the ghost particle margin of the neighbor subdomain reached by the given steps
        let is_in_ghost_margin = |x_step: i32, y_step: i32, z_step: i32| -> bool {
            is_in_ghost_margin_single_dim(x_step, 0)
                && is_in_ghost_margin_single_dim(y_step, 1)
                && is_in_ghost_margin_single_dim(z_step, 2)
        };

        let checked_apply_step = |index: I, step: i32| -> Option<I> {
            let direction = if step > 0 {
                Some(Direction::Positive)
            } else if step < 0 {
                Some(Direction::Negative)
            } else {
                None
            };
            direction
                .map(|d| d.checked_apply_step(index, I::from(step.abs()).unwrap()))
                .unwrap_or(Some(index))
        };

        let checked_apply_step_ijk =
            |ijk: [I; 3], x_step: i32, y_step: i32, z_step: i32| -> Option<[I; 3]> {
                Some([
                    checked_apply_step(ijk[0], x_step)?,
                    checked_apply_step(ijk[1], y_step)?,
                    checked_apply_step(ijk[2], z_step)?,
                ])
            };

        let r = subdomain_radius;
        for i in -r..=r {
            for j in -r..=r {
                for k in -r..=r {
                    // Check if the particle is in the ghost particle margin of the subdomain i,j,k
                    let in_ghost_margin = is_in_ghost_margin(i, j, k);

                    if in_ghost_margin {
                        if let Some(neighbor_subdomain_ijk) =
                            checked_apply_step_ijk(subdomain_ijk, i, j, k)
                            && let Some(cell) = subdomain_grid.get_cell(neighbor_subdomain_ijk)
                        {
                            subdomains.push(subdomain_grid.flatten_cell_index(&cell));
                        }
                    }
                }
            }
        }
    }
}

pub(crate) mod debug {
    use super::*;

    /// Prints statistics of the given list of subdomains
    #[allow(unused)]
    pub(crate) fn subdomain_stats<I: Index, R: Real>(
        parameters: &ParametersSubdomainGrid<I, R>,
        particles: &[Vector3<R>],
        subdomains: &Subdomains<I>,
    ) {
        profile!("subdomain stats");
        info!("Statistics");

        let per_subdomain_particle_count = subdomains
            .per_subdomain_particles
            .iter()
            .map(|p| p.len())
            .collect::<Vec<_>>();

        info!(
            "Occupied subdomains: {}",
            per_subdomain_particle_count.len()
        );

        /*
        info!("Printing subdomain particle counts:");
        for subdomain in subdomains.per_subdomain_particles {
            info!("{}", subdomain.lock().len());
        }
        */

        info!("Smallest Subdomains:");
        for i in 0..11 {
            let c = per_subdomain_particle_count
                .iter()
                .copied()
                .filter(|c| *c == i)
                .count();

            info!(
                "Number of subdomains with {} particles: {} ({:.2}% of number of subdomains)",
                i,
                c,
                (c as f64 / per_subdomain_particle_count.len() as f64) * 100.0
            );
        }

        info!("Other stats:");
        for n in [10, 50, 100, 500, 1000, 2000, 10000] {
            let c = per_subdomain_particle_count
                .iter()
                .copied()
                .filter(|c| *c <= n)
                .count();

            info!(
                "Number of subdomains with {} or fewer particles: {} ({:.2}% of number of subdomains)",
                n,
                c,
                (c as f64 / per_subdomain_particle_count.len() as f64) * 100.0
            );
        }

        {
            let mut per_subdomain_particle_count = per_subdomain_particle_count.clone();
            per_subdomain_particle_count.sort();

            {
                let largest_subdomain_particle_count =
                    per_subdomain_particle_count[per_subdomain_particle_count.len() - 1];
                info!("Largest subdomain has {largest_subdomain_particle_count} particles");

                for f in [0.95, 0.9, 0.8, 0.75, 0.5, 0.1] {
                    let c = per_subdomain_particle_count
                        .iter()
                        .copied()
                        .filter(|c| *c >= (f * largest_subdomain_particle_count as f64) as usize)
                        .count();

                    let n = per_subdomain_particle_count
                        .iter()
                        .copied()
                        .filter(|c| *c >= (f * largest_subdomain_particle_count as f64) as usize)
                        .sum::<usize>();

                    info!(
                        "Number of subdomains with {} or more particles ({}% of largest subdomain): {} ({:.2}% of number of subdomains), in sum {} particles ({:.2}% of all particles)",
                        (f * largest_subdomain_particle_count as f64) as usize,
                        f * 100.0,
                        c,
                        (c as f64 / per_subdomain_particle_count.len() as f64) * 100.0,
                        n,
                        100.0 * (n as f64 / particles.len() as f64)
                    );
                }
            }

            info!("Largest subdomains:");
            for i in 0..10 {
                if let Some(&count) =
                    per_subdomain_particle_count.get(per_subdomain_particle_count.len() - 1 - i)
                {
                    info!(
                        "{} particles ({:.2}% of all particles)",
                        count,
                        100.0 * (count as f64 / particles.len() as f64)
                    );
                }
            }
        }
    }

    #[allow(unused)]
    pub(crate) fn subdomains_to_hexmesh<I: Index, R: Real>(
        parameters: &ParametersSubdomainGrid<I, R>,
        subdomains: &Subdomains<I>,
    ) -> HexMesh3d<R> {
        let mut hexmesh = HexMesh3d::default();
        let subdomain_grid = &parameters.subdomain_grid;

        // Loop over all non-empty subdomains
        for &flat_subdomain_idx in &subdomains.flat_subdomain_indices {
            let subdomain_ijk = subdomain_grid
                .try_unflatten_cell_index(flat_subdomain_idx as I)
                .unwrap();
            let [i, j, k] = *subdomain_ijk.index();

            let vertex_offset = hexmesh.vertices.len();

            {
                let mut push_vertex = |abc: [i8; 3]| {
                    let [a, b, c] = abc;
                    hexmesh.vertices.push(
                        subdomain_grid.point_coordinates(
                            &subdomain_grid
                                .get_point([
                                    i + I::from(a).unwrap(),
                                    j + I::from(b).unwrap(),
                                    k + I::from(c).unwrap(),
                                ])
                                .unwrap(),
                        ),
                    );
                };

                push_vertex([0, 0, 0]);
                push_vertex([1, 0, 0]);
                push_vertex([1, 1, 0]);
                push_vertex([0, 1, 0]);
                push_vertex([0, 0, 1]);
                push_vertex([1, 0, 1]);
                push_vertex([1, 1, 1]);
                push_vertex([0, 1, 1]);
            }

            hexmesh.cells.push([
                vertex_offset,
                vertex_offset + 1,
                vertex_offset + 2,
                vertex_offset + 3,
                vertex_offset + 4,
                vertex_offset + 5,
                vertex_offset + 6,
                vertex_offset + 7,
            ]);
        }

        hexmesh
    }

    /// Counts subdomains with only ghost particles and no particles in interior
    #[allow(unused)]
    pub(crate) fn count_no_owned_particles_subdomains<I: Index, R: Real>(
        parameters: &ParametersSubdomainGrid<I, R>,
        particles: &[Vector3<R>],
        subdomains: &Subdomains<I>,
    ) -> usize {
        profile!(parent, "count_no_owned_particles_subdomains");

        let no_owned_particles_counter = AtomicUsize::new(0);

        subdomains
            .flat_subdomain_indices
            .par_iter()
            .copied()
            .zip(subdomains.per_subdomain_particles.par_iter())
            .for_each(|(flat_subdomain_idx, subdomain_particle_indices)| {
                profile!("inner subdomain_tasks", parent = parent);

                let flat_subdomain_idx: I = flat_subdomain_idx;
                let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();

                // Collect all particle positions of this subdomain
                let subdomain_particles = subdomain_particle_indices
                    .iter()
                    .copied()
                    .map(|idx| particles[idx])
                    .collect::<Vec<_>>();

                // Get the cell index and AABB of the subdomain
                let subdomain_idx = parameters
                    .subdomain_grid
                    .try_unflatten_cell_index(flat_subdomain_idx)
                    .expect("Subdomain cell does not exist");
                let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);

                // Count the number of owned (non-ghost particles) of this domain
                let non_ghost_particle_count = subdomain_particles
                    .iter()
                    .filter(|p| subdomain_aabb.contains_point(*p))
                    .count();
                if non_ghost_particle_count == 0 {
                    no_owned_particles_counter.fetch_add(1, Ordering::AcqRel);
                }
            });

        no_owned_particles_counter.into_inner()
    }
}

/// Performs integer division and rounds the result up if there is a remainder
fn int_ceil_div<T: Integer + Copy>(numerator: T, denominator: T) -> T {
    numerator / denominator + (numerator % denominator).min(T::one())
}

/// Ensures that at least the specified total capacity is reserved for the given vector
fn reserve_total<T>(vec: &mut Vec<T>, total_capacity: usize) {
    if total_capacity > vec.capacity() {
        vec.reserve(total_capacity - vec.capacity());
    }
}

/// Gathers particle related data from global storage to subdomain storage
fn gather_subdomain_data<T: Copy>(
    global_data: &[T],
    subdomain_particle_indices: &[usize],
    subdomain_data: &mut Vec<T>,
) {
    subdomain_data.clear();
    reserve_total(subdomain_data, subdomain_particle_indices.len());
    subdomain_data.extend(
        subdomain_particle_indices
            .iter()
            .copied()
            .map(|idx| global_data[idx]),
    );
}
