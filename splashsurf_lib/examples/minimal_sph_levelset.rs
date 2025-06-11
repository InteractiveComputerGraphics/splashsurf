//! Example with a simple multithreaded marching cubes implementation for reconstruction of particle data from SPH simulations.
//!
//! Note that this example only reconstructs a triangle soup for a simpler and faster implementation.

use anyhow::anyhow;
use clap::Parser;
use log::info;
use rayon::prelude::*;
#[cfg(feature = "io")]
use splashsurf_lib::io;
use splashsurf_lib::kernel::SymmetricKernel3d;
use splashsurf_lib::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use splashsurf_lib::mesh::TriMesh3d;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::uniform_grid::UniformCartesianCubeGrid3d;
use splashsurf_lib::{Aabb3d, UniformGrid, kernel};
use splashsurf_lib::{Real, profile};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};
use thread_local::ThreadLocal;

// The real type used for the reconstruction, can be changed to f64 if higher precision is needed
type R = f32;
// The index type used for the grid
type I = usize;

#[derive(Clone, Debug, clap::Parser)]
struct CommandlineArgs {
    /// The input file containing the particle data
    #[arg(long, short = 'i')]
    pub input_file: String,
    /// The output file to write the reconstructed mesh to
    #[arg(long, short = 'o')]
    pub output_file: String,
    /// Whether to output the level set function values on the marching cubes grid instead of the triangulated mesh
    #[arg(long)]
    pub output_levelset: bool,
    /// The particle radius of the input data
    #[arg(short = 'r', long)]
    pub particle_radius: R,
    /// The rest density of the fluid
    #[arg(long, default_value = "1000.0")]
    pub rest_density: R,
    /// The smoothing length radius used for the SPH kernel, the kernel compact support radius will be twice the smoothing length (in multiples of the particle radius)
    #[arg(short = 'l', long)]
    pub smoothing_length: R,
    /// The cube edge length used for marching cubes in multiples of the particle radius, corresponds to the cell size of the implicit background grid
    #[arg(short = 'c', long)]
    pub cube_size: R,
    /// The iso-surface threshold for the density, i.e., the normalized value of the reconstructed density level that indicates the fluid surface (in multiples of the rest density)
    #[arg(short = 't', long, default_value = "0.6")]
    pub surface_threshold: R,
}

pub enum LevelSetSign {
    Inside,
    Outside,
}

/// Interface that has to be provided by a level set to be reconstructed using marching cubes
pub trait MarchingCubesLevelSet<R: Real> {
    /// Returns the sign of the level set function at the given coordinate
    fn evaluate_sign(&self, coordinate: &Vector3<R>) -> LevelSetSign;
    /// Returns the value of the level set function at the given coordinate
    fn evaluate(&self, coordinate: &Vector3<R>) -> R;
}

/// Reconstructs a triangle soup from a level set function
pub fn marching_cubes<R: Real, L: MarchingCubesLevelSet<R> + Send + Sync>(
    level_set: L,
    grid: &UniformCartesianCubeGrid3d<I, R>,
) -> Result<TriMesh3d<R>, anyhow::Error> {
    profile!("marching_cubes");

    let mut mesh = TriMesh3d::default();
    let level_set = &level_set;

    let vertices = ThreadLocal::<RefCell<Vec<Vector3<R>>>>::new();
    let triangles = ThreadLocal::<RefCell<Vec<[usize; 3]>>>::new();

    let n_cells = grid.cells_per_dim().iter().product();
    let cell_indices = (0..n_cells).into_iter().collect::<Vec<_>>();

    info!("Starting to process {} cells", cell_indices.len());
    print!("Start: ");
    cell_indices.par_chunks(5000).for_each(|chunk| {
        let vertices = vertices.get_or(|| RefCell::new(Vec::new()));
        let mut mut_vertices = vertices.borrow_mut();
        let triangles = triangles.get_or(|| RefCell::new(Vec::new()));
        let mut mut_triangles = triangles.borrow_mut();

        for i in chunk.iter().copied() {
            if let Some(cell) = grid.try_unflatten_cell_index(i) {
                let mut vertices_inside = [true; 8];
                for local_point_index in 0..8 {
                    let point = cell.global_point_index_of(local_point_index).unwrap();
                    let coords = grid.point_coordinates(&point);

                    match level_set.evaluate_sign(&coords) {
                        LevelSetSign::Inside => vertices_inside[local_point_index] = true,
                        LevelSetSign::Outside => vertices_inside[local_point_index] = false,
                    }
                }

                for triangle in marching_cubes_triangulation_iter(&vertices_inside) {
                    let mut global_triangle = [0; 3];
                    for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
                        let edge = cell
                            .global_edge_index_of(local_edge_index as usize)
                            .unwrap();
                        let vertex_index = {
                            let vertex_index = mut_vertices.len();

                            let origin_coords = grid.point_coordinates(edge.origin());
                            let target_coords = grid.point_coordinates(&edge.target());

                            let origin_value = level_set.evaluate(&origin_coords);
                            let target_value = level_set.evaluate(&target_coords);

                            let diff = target_value - origin_value;
                            let c = (origin_value / diff).abs();
                            let p = origin_coords + (target_coords - origin_coords) * c;

                            let vertex_coords = p;
                            mut_vertices.push(vertex_coords);
                            vertex_index
                        };
                        global_triangle[v_idx] = vertex_index;
                    }
                    mut_triangles.push(global_triangle);
                }
            }
        }

        print!("+");
    });
    println!();

    info!("Merging thread local results...");
    for (vertices, triangles) in vertices.into_iter().zip(triangles.into_iter()) {
        let vertex_offset = mesh.vertices.len();

        let vertices = vertices.into_inner();
        let triangles = triangles.into_inner();

        mesh.vertices.extend(vertices);
        for triangle in triangles {
            let mut global_triangle = [0; 3];
            for (v_idx, vertex_index) in triangle.iter().enumerate() {
                global_triangle[v_idx] = vertex_offset + vertex_index;
            }
            mesh.triangles.push(global_triangle);
        }
    }

    Ok(mesh)
}

pub struct MarchingCubesGrid {
    pub grid: UniformCartesianCubeGrid3d<I, R>,
    pub values: Vec<R>,
    pub threshold: R,
}

impl MarchingCubesLevelSet<f32> for MarchingCubesGrid {
    fn evaluate_sign(&self, coordinate: &Vector3<f32>) -> LevelSetSign {
        let rel_coordinate = (coordinate - self.grid.aabb().min()) / self.grid.cell_size();
        let ijk = [
            rel_coordinate.x.round() as I,
            rel_coordinate.y.round() as I,
            rel_coordinate.z.round() as I,
        ];

        if let Some(index) = self.grid.get_point(ijk) {
            let index = self.grid.flatten_point_index(&index);
            if self.values[index] < self.threshold {
                LevelSetSign::Outside
            } else {
                LevelSetSign::Inside
            }
        } else {
            LevelSetSign::Outside
        }
    }

    fn evaluate(&self, coordinate: &Vector3<f32>) -> f32 {
        let rel_coordinate = (coordinate - self.grid.aabb().min()) / self.grid.cell_size();
        let ijk = [
            rel_coordinate.x.round() as I,
            rel_coordinate.y.round() as I,
            rel_coordinate.z.round() as I,
        ];

        if let Some(index) = self
            .grid
            .get_point(ijk)
            .map(|p| self.grid.flatten_point_index(&p))
        {
            self.values[index] - self.threshold
        } else {
            f32::MIN // or some other value indicating outside
        }
    }
}

fn reconstruct() -> Result<(), anyhow::Error> {
    profile!("reconstruct");

    let args = CommandlineArgs::parse();
    info!("Command line args: {args:?}");

    info!("Loading particles from: \"{}\"", args.input_file);
    let particles = io::particles_from_file::<f32, _>(args.input_file)?;
    info!("Loaded {} particles", particles.len());

    let cube_size = args.cube_size * args.particle_radius;
    let compact_support_radius = 2.0 * args.smoothing_length * args.particle_radius;

    let mut particle_aabb = Aabb3d::par_from_points(&particles);
    info!(
        "Minimal enclosing bounding box of particles was computed as: {:?}",
        particle_aabb
    );

    // Ensure that we have enough margin around the particles such that the every particle's kernel support is completely in the domain

    // The number of cells in each direction from a particle that can be affected by its compact support
    let half_supported_cells_real = (compact_support_radius / cube_size).ceil() as R;
    // Convert to index type for cell and point indexing
    let half_supported_cells: I = half_supported_cells_real.to_index_unchecked();

    // The total number of cells per dimension that can be affected by a particle's compact support
    let supported_cells: I = half_supported_cells * 2 + 1;
    // The number of points corresponding to the number of supported cells
    let supported_points: I = 1 + supported_cells;

    // Evaluate kernel in a smaller domain, points outside of this radius have to be assumed to be outside the iso-surface
    let kernel_evaluation_radius =
        cube_size * half_supported_cells_real * (1.0 + R::EPSILON.sqrt());

    particle_aabb.grow_uniformly(kernel_evaluation_radius);

    let grid = UniformGrid::<I, R>::from_aabb(&particle_aabb, cube_size)?;
    let kernel = kernel::CubicSplineKernel::new(compact_support_radius);

    let particle_rest_volume = (2.0 * args.particle_radius).powi(3);
    let particle_rest_mass = particle_rest_volume * args.rest_density;

    info!(
        "Evaluating level set function on grid with a total of {} grid vertices...",
        grid.points_per_dim().iter().product::<I>()
    );

    let particle_densities = {
        profile!("compute_particle_densities");
        let mut densities = vec![0.0; particles.len()];

        let mut nl = Vec::with_capacity(particles.len());
        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<I, R>(
            grid.aabb(),
            &particles,
            compact_support_radius,
            &mut nl,
        );

        {
            profile!("density_loop");
            densities.par_iter_mut().enumerate().for_each(|(i, den)| {
                *den += particle_rest_mass * kernel.evaluate(0.0);
                for j in nl[i].iter() {
                    let dx = particles[i] - particles[*j];
                    let r = dx.norm();
                    *den += particle_rest_mass * kernel.evaluate(r);
                }
            });
        }

        densities
    };

    let mut function_values = vec![0.0; grid.points_per_dim().iter().product()];
    let mut function_values_vol_frac = vec![0.0; grid.points_per_dim().iter().product()];
    {
        profile!("evaluate_levelset_function");

        {
            // There is no atomic f32, so we have to rely on u32 and convert the bit-representations
            let function_values = unsafe {
                std::mem::transmute::<&mut [f32], &[AtomicU32]>(function_values.as_mut_slice())
            };

            let function_values_vol_frac = unsafe {
                std::mem::transmute::<&mut [f32], &[AtomicU32]>(
                    function_values_vol_frac.as_mut_slice(),
                )
            };

            particles
                .par_iter()
                .enumerate()
                .for_each(|(particle_idx, particle)| {
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

                    let mut i = min_supported_point_ijk[0];
                    while i != max_supported_point_ijk[0] {
                        let mut j = min_supported_point_ijk[1];
                        while j != max_supported_point_ijk[1] {
                            let mut k = min_supported_point_ijk[2];
                            while k != max_supported_point_ijk[2] {
                                if let Some(point_index) = grid.get_point([i, j, k]) {
                                    let coords = grid.point_coordinates(&point_index);
                                    let dx = particle - coords;
                                    let r = dx.norm();

                                    if r <= kernel_evaluation_radius {
                                        let index = grid.flatten_point_index(&point_index);

                                        //let vol = particle_rest_volume;
                                        let vol =
                                            particle_rest_mass / particle_densities[particle_idx];
                                        let kernel_r = kernel.evaluate(r);

                                        let _ = function_values[index].fetch_update(
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                            |val| {
                                                let val_f32 = f32::from_bits(val);
                                                Some((val_f32 + vol * kernel_r).to_bits())
                                            },
                                        );

                                        let _ = function_values_vol_frac[index].fetch_update(
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                            |val| {
                                                let val_f32 = f32::from_bits(val);
                                                Some(
                                                    (val_f32 + particle_rest_volume * kernel_r)
                                                        .to_bits(),
                                                )
                                            },
                                        );
                                    }
                                }
                                k += 1;
                            }
                            j += 1;
                        }
                        i += 1;
                    }
                });
        }
    };

    if args.output_levelset {
        profile!("output_levelset");

        let [ni, nj, nk] = grid.points_per_dim().clone();

        let mut points_flat = Vec::with_capacity(3 * ni * nj * nk);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let point_index = grid.get_point([i, j, k]).unwrap();
                    let coords = grid.point_coordinates(&point_index);
                    points_flat.push(coords.x);
                    points_flat.push(coords.y);
                    points_flat.push(coords.z);
                }
            }
        }

        let mut structured = vtkio::model::StructuredGridPiece {
            extent: vtkio::model::Extent::Dims([ni as u32, nj as u32, nk as u32]),
            points: points_flat.into(),
            data: Default::default(),
        };
        structured
            .data
            .point
            .push(vtkio::model::Attribute::scalars("levelset", 1).with_data(function_values));
        structured.data.point.push(
            vtkio::model::Attribute::scalars("vol_frac", 1).with_data(function_values_vol_frac),
        );
        let grid = vtkio::model::DataSet::inline(structured);

        #[cfg(feature = "io")]
        info!("Writing mesh to: \"{}\"", args.output_file);
        io::vtk_format::write_vtk(grid, args.output_file, "mesh")?;
    } else {
        info!("Running marching cubes triangulation...");
        let mc_grid = MarchingCubesGrid {
            grid: grid.clone(),
            values: function_values,
            threshold: args.surface_threshold,
        };
        let mesh = marching_cubes(mc_grid, &grid)?;

        info!(
            "Vertices: {} triangles: {}",
            mesh.vertices.len(),
            mesh.triangles.len()
        );

        #[cfg(feature = "io")]
        info!("Writing mesh to: \"{}\"", args.output_file);
        io::vtk_format::write_vtk(&mesh, args.output_file, "mesh")?;
    }

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    let _ = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}] {}",
                chrono::Local::now().format("%T%.3f"),
                record.level(),
                message
            ))
        })
        .chain(std::io::stdout())
        .apply()
        .map_err(|e| anyhow!("Unable to apply logger configuration ({:?})", e))?;

    reconstruct()?;

    info!("Timings:");
    splashsurf_lib::profiling::write_to_string()
        .unwrap()
        .split("\n")
        .filter(|l| l.len() > 0)
        .for_each(|l| info!("  {}", l));
    Ok(())
}
