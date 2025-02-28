use std::borrow::Cow;

use numpy::{Element, PyReadonlyArray2};
use pyo3::{prelude::*, Bound, IntoPyObjectExt, Python, PyAny};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use splashsurf_lib::{
    mesh::{AttributeData, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d},
    nalgebra::Vector3,
    sph_interpolation::SphInterpolator,
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition, SurfaceReconstruction,
};

use crate::structs::{PySurfaceReconstructionF32, PySurfaceReconstructionF64, PyTriMesh3dF32, PyTriMesh3dF64};

#[pyclass]
struct ReconstructionRunnerPostprocessingArgs {
    mesh_cleanup: bool,
    decimate_barnacles: bool,
    keep_vertices: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    output_mesh_smoothing_weights: bool,
}

fn post_processing_generic<I: Index, R: Real>(
    particle_positions: Vec<Vector3<R>>,
    reconstruction: &SurfaceReconstruction<I, R>,
    params: &splashsurf_lib::Parameters<R>,
    postprocessing: &ReconstructionRunnerPostprocessingArgs,
) -> Result<TriMesh3d<R>, anyhow::Error> {
    //profile!("surface reconstruction");

    let grid = reconstruction.grid();
    let mut mesh_with_data = MeshWithData::new(Cow::Borrowed(reconstruction.mesh()));

    // Perform post-processing
    {
        //profile!("postprocessing");
        let mut vertex_connectivity = None;

        if postprocessing.mesh_cleanup {
            //info!("Post-processing: Performing mesh cleanup");
            //let tris_before = mesh_with_data.mesh.triangles.len();
            //let verts_before = mesh_with_data.mesh.vertices.len();
            vertex_connectivity = Some(splashsurf_lib::postprocessing::marching_cubes_cleanup(
                mesh_with_data.mesh.to_mut(),
                grid,
                5,
                postprocessing.keep_vertices,
            ));
            //let tris_after = mesh_with_data.mesh.triangles.len();
            //let verts_after = mesh_with_data.mesh.vertices.len();
            //info!("Post-processing: Cleanup reduced number of vertices to {:.2}% and number of triangles to {:.2}% of original mesh.", (verts_after as f64 / verts_before as f64) * 100.0, (tris_after as f64 / tris_before as f64) * 100.0)
        }

        // Decimate mesh if requested
        if postprocessing.decimate_barnacles {
            //info!("Post-processing: Performing decimation");
            vertex_connectivity = Some(splashsurf_lib::postprocessing::decimation(
                mesh_with_data.mesh.to_mut(),
                postprocessing.keep_vertices,
            ));
        }

        // Initialize SPH interpolator if required later
        let interpolator_required =
            postprocessing.mesh_smoothing_weights || postprocessing.sph_normals;
        //    || !attributes.is_empty();
        let interpolator = if interpolator_required {
            //profile!("initialize interpolator");
            //info!("Post-processing: Initializing interpolator...");

            //info!(
            //    "Constructing global acceleration structure for SPH interpolation to {} vertices...",
            //    mesh_with_data.vertices().len()
            //);

            let particle_rest_density = params.rest_density;
            let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
                * params.particle_radius.powi(3);
            let particle_rest_mass = particle_rest_volume * particle_rest_density;

            let particle_densities = reconstruction
                .particle_densities()
                .ok_or_else(|| anyhow::anyhow!("Particle densities were not returned by surface reconstruction but are required for SPH normal computation"))?
                .as_slice();
            assert_eq!(
                particle_positions.len(),
                particle_densities.len(),
                "There has to be one density value per particle"
            );

            Some(SphInterpolator::new(
                &particle_positions,
                particle_densities,
                particle_rest_mass,
                params.compact_support_radius,
            ))
        } else {
            None
        };

        // Compute mesh vertex-vertex connectivity map if required later
        let vertex_connectivity_required = postprocessing.normals_smoothing_iters.is_some()
            || postprocessing.mesh_smoothing_iters.is_some();
        if vertex_connectivity.is_none() && vertex_connectivity_required {
            vertex_connectivity = Some(mesh_with_data.mesh.vertex_vertex_connectivity());
        }

        // Compute smoothing weights if requested
        let smoothing_weights = if postprocessing.mesh_smoothing_weights {
            //profile!("compute smoothing weights");
            //info!("Post-processing: Computing smoothing weights...");

            // TODO: Switch between parallel/single threaded
            // TODO: Re-use data from reconstruction?

            // Global neighborhood search
            let nl = reconstruction
                .particle_neighbors()
                .map(Cow::Borrowed)
                .unwrap_or_else(||
                    {
                        let search_radius = params.compact_support_radius;

                        let mut domain = Aabb3d::from_points(particle_positions.as_slice());
                        domain.grow_uniformly(search_radius);

                        let mut nl = Vec::new();
                        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<I, R>(
                            &domain,
                            particle_positions.as_slice(),
                            search_radius,
                            &mut nl,
                        );
                        assert_eq!(nl.len(), particle_positions.len());
                        Cow::Owned(nl)
                    }
                );

            // Compute weighted neighbor count
            let squared_r = params.compact_support_radius * params.compact_support_radius;
            let weighted_ncounts = nl
                .par_iter()
                .enumerate()
                .map(|(i, nl)| {
                    nl.iter()
                        .copied()
                        .map(|j| {
                            let dist =
                                (particle_positions[i] - particle_positions[j]).norm_squared();

                            R::one() - (dist / squared_r).clamp(R::zero(), R::one())
                        })
                        .fold(R::zero(), R::add)
                })
                .collect::<Vec<_>>();

            let vertex_weighted_num_neighbors = {
                //profile!("interpolate weighted neighbor counts");
                interpolator
                    .as_ref()
                    .expect("interpolator is required")
                    .interpolate_scalar_quantity(
                        weighted_ncounts.as_slice(),
                        mesh_with_data.vertices(),
                        true,
                    )
            };

            let smoothing_weights = {
                let offset = R::zero();
                let normalization =
                    R::from_f64(postprocessing.mesh_smoothing_weights_normalization).expect(
                        "smoothing weight normalization value cannot be represented as Real type",
                    ) - offset;

                // Normalize number of neighbors
                let smoothing_weights = vertex_weighted_num_neighbors
                    .par_iter()
                    .copied()
                    .map(|n| (n - offset).max(R::zero()))
                    .map(|n| (n / normalization).min(R::one()))
                    // Smooth-Step function
                    .map(|x| x.powi(5).times(6) - x.powi(4).times(15) + x.powi(3).times(10))
                    .collect::<Vec<_>>();

                if postprocessing.output_mesh_smoothing_weights {
                    // Raw distance-weighted number of neighbors value per vertex (can be used to determine normalization value)
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "wnn".to_string(),
                        AttributeData::ScalarReal(vertex_weighted_num_neighbors),
                    ));
                    // Final smoothing weights per vertex
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "sw".to_string(),
                        AttributeData::ScalarReal(smoothing_weights.clone()),
                    ));
                }

                smoothing_weights
            };

            Some(smoothing_weights)
        } else {
            None
        };

        // Perform smoothing if requested
        if let Some(mesh_smoothing_iters) = postprocessing.mesh_smoothing_iters {
            //profile!("mesh smoothing");
            //info!("Post-processing: Smoothing mesh...");

            // TODO: Switch between parallel/single threaded

            let smoothing_weights = smoothing_weights
                .unwrap_or_else(|| vec![R::one(); mesh_with_data.vertices().len()]);

            splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
                mesh_with_data.mesh.to_mut(),
                vertex_connectivity
                    .as_ref()
                    .expect("vertex connectivity is required"),
                mesh_smoothing_iters,
                R::one(),
                &smoothing_weights,
            );
        }
    }

    Ok(mesh_with_data.mesh.into_owned())
}


pub fn post_processing_py_interface<'py, R: Real + Element>(
    particles: PyReadonlyArray2<R>,
    reconstruction: &SurfaceReconstruction<i64, R>,
    particle_radius: R,
    rest_density: R,
    smoothing_length: R,
    cube_size: R,
    iso_surface_threshold: R,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[R; 3]>,
    aabb_max: Option<[R; 3]>,
) -> TriMesh3d<R> {
    let particle_positions: Vec<Vector3<R>> = particles
        .as_array()
        .outer_iter()
        .map(|row| Vector3::new(row[0], row[1], row[2]))
        .collect();

    let aabb;
    if aabb_min == None || aabb_max == None {
        aabb = None;
    } else {
        aabb = Some(Aabb3d::new(
            Vector3::from(aabb_min.unwrap()),
            Vector3::from(aabb_max.unwrap()),
        ));
    }

    let spatial_decomposition;
    if use_custom_grid_decomposition {
        let mut grid_params = GridDecompositionParameters::default();
        grid_params.subdomain_num_cubes_per_dim = subdomain_num_cubes_per_dim;
        spatial_decomposition = Some(SpatialDecomposition::UniformGrid(grid_params));
    } else {
        spatial_decomposition = None;
    }

    let params = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        compact_support_radius: (smoothing_length * particle_radius).times_f64(2.0),
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb: aabb,
        enable_multi_threading,
        spatial_decomposition,
        global_neighborhood_list,
    };

    let postprocessing_args = ReconstructionRunnerPostprocessingArgs {
        mesh_cleanup: true,
        decimate_barnacles: true,
        keep_vertices: false,
        sph_normals: true,
        normals_smoothing_iters: Some(5),
        mesh_smoothing_iters: Some(5),
        mesh_smoothing_weights: true,
        mesh_smoothing_weights_normalization: 100.0,
        output_mesh_smoothing_weights: true,
    };

    let mesh = post_processing_generic::<i64, R>(
        particle_positions,
        &reconstruction,
        &params,
        &postprocessing_args,
    )
    .unwrap();

    mesh
}

#[pyfunction]
#[pyo3(name = "post_processing_f32")]
#[pyo3(signature = (particles, reconstruction, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading,
    global_neighborhood_list, use_custom_grid_decomposition, subdomain_num_cubes_per_dim,
    aabb_min, aabb_max
))]
pub fn post_processing_py_f32<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<f32>,
    reconstruction: &PySurfaceReconstructionF32,
    particle_radius: f32,
    rest_density: f32,
    smoothing_length: f32,
    cube_size: f32,
    iso_surface_threshold: f32,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[f32; 3]>,
    aabb_max: Option<[f32; 3]>,
) -> Bound<'py, PyAny> {
    let mesh = post_processing_py_interface::<f32>(particles, &reconstruction.inner, particle_radius, rest_density, smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading, global_neighborhood_list, use_custom_grid_decomposition, subdomain_num_cubes_per_dim, aabb_min, aabb_max);
    PyTriMesh3dF32::new(mesh).into_bound_py_any(py).unwrap()
}

#[pyfunction]
#[pyo3(name = "post_processing_f64")]
#[pyo3(signature = (particles, reconstruction, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading,
    global_neighborhood_list, use_custom_grid_decomposition, subdomain_num_cubes_per_dim,
    aabb_min, aabb_max
))]
pub fn post_processing_py_f64<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<f64>,
    reconstruction: &PySurfaceReconstructionF64,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
) -> Bound<'py, PyAny> {
    let mesh = post_processing_py_interface::<f64>(particles, &reconstruction.inner, particle_radius, rest_density, smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading, global_neighborhood_list, use_custom_grid_decomposition, subdomain_num_cubes_per_dim, aabb_min, aabb_max);
    PyTriMesh3dF64::new(mesh).into_bound_py_any(py).unwrap()
}