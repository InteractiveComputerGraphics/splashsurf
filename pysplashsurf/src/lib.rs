use anyhow;
use ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::types::PyFloat;
use pyo3::{prelude::*, IntoPyObjectExt};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use splashsurf_lib::mesh::{AttributeData, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d};
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::sph_interpolation::SphInterpolator;
use splashsurf_lib::uniform_grid::UniformCartesianCubeGrid3d;
use splashsurf_lib::{
    postprocessing::marching_cubes_cleanup,
    reconstruct_surface,
};
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition, SurfaceReconstruction,
};
use std::borrow::Cow;

/// Reconstruct the surface from only particle positions
fn reconstruct_surface_py<I: Index, R: Real>(
    particles: Vec<Vector3<R>>,
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
) -> (
    Array2<usize>,
    Array2<R>,
    ([R; 3], [R; 3], R, [I; 3], [I; 3]),
) {
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

    let surface = reconstruct_surface(&particles, &params).unwrap();

    let grid = surface.grid();
    let aabb = grid.aabb();

    let grid_info = {
        let min = [aabb.min()[0], aabb.min()[1], aabb.min()[2]];
        let max = [aabb.max()[0], aabb.max()[1], aabb.max()[2]];
        (
            min,
            max,
            grid.cell_size().clone(),
            grid.points_per_dim().clone(),
            grid.cells_per_dim().clone(),
        )
    };

    let mesh = surface.mesh();

    let points: Vec<R> = mesh.vertices.iter().flatten().copied().collect();
    let tris: Vec<usize> = mesh.triangles.iter().flatten().copied().collect();
    let triangles = ndarray::Array2::from_shape_vec((mesh.triangles.len(), 3), tris).unwrap();
    let vertices = ndarray::Array2::from_shape_vec((mesh.vertices.len(), 3), points).unwrap();

    (triangles, vertices, grid_info)
}

fn reconstruct_surface_py_interface<'py, R: Real + Element>(
    py: Python<'py>,
    particles: &Bound<'py, PyArray2<R>>,
    particle_radius: Py<PyFloat>,
    rest_density: Py<PyFloat>,
    smoothing_length: Py<PyFloat>,
    cube_size: Py<PyFloat>,
    iso_surface_threshold: Py<PyFloat>,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[Py<PyFloat>; 3]>,
    aabb_max: Option<[Py<PyFloat>; 3]>,
) -> (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    ([R; 3], [R; 3], R, [i64; 3], [i64; 3]),
) {
    let particles: PyReadonlyArray2<R> = particles.extract().unwrap();
    let particle_positions: Vec<Vector3<R>> = particles
        .as_array()
        .outer_iter()
        .map(|row| Vector3::new(row[0], row[1], row[2]))
        .collect();

    let aabb_min: Option<[R; 3]> = aabb_min.map(|x| {
        let mut res = [R::zero(); 3];
        for i in 0..3 {
            res[i] = R::from_f64(x[i].extract::<f64>(py).unwrap()).unwrap();
        }
        res
    });

    let aabb_max: Option<[R; 3]> = aabb_max.map(|x| {
        let mut res = [R::zero(); 3];
        for i in 0..3 {
            res[i] = R::from_f64(x[i].extract::<f64>(py).unwrap()).unwrap();
        }
        res
    });

    let (triangles, vertices, grid_info) = reconstruct_surface_py::<i64, R>(
        particle_positions,
        R::from_f64(particle_radius.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(rest_density.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(smoothing_length.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(cube_size.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(iso_surface_threshold.extract::<f64>(py).unwrap()).unwrap(),
        enable_multi_threading,
        global_neighborhood_list,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        aabb_min,
        aabb_max,
    );

    (
        triangles.into_pyarray(py).into_any(),
        vertices.into_pyarray(py).into_any(),
        grid_info,
    )
}

#[pyfunction]
#[pyo3(name = "reconstruct_surface")]
#[pyo3(signature = (particles, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading=false,
    global_neighborhood_list=false, use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64,
    aabb_min = None, aabb_max = None
))]
fn reconstruct_surface_py_dynamic<'py>(
    py: Python<'py>,
    particles: &Bound<'py, PyAny>,
    particle_radius: Py<PyFloat>,
    rest_density: Py<PyFloat>,
    smoothing_length: Py<PyFloat>,
    cube_size: Py<PyFloat>,
    iso_surface_threshold: Py<PyFloat>,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[Py<PyFloat>; 3]>,
    aabb_max: Option<[Py<PyFloat>; 3]>,
) -> (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    (
        Py<PyAny>,
        Py<PyAny>,
        Bound<'py, PyFloat>,
        [i64; 3],
        [i64; 3],
    ),
) {
    if let Ok(particles) = particles.downcast::<PyArray2<f32>>() {
        let (triangles, vertices, grid_info) = reconstruct_surface_py_interface::<f32>(
            py,
            particles,
            particle_radius,
            rest_density,
            smoothing_length,
            cube_size,
            iso_surface_threshold,
            enable_multi_threading,
            global_neighborhood_list,
            use_custom_grid_decomposition,
            subdomain_num_cubes_per_dim,
            aabb_min,
            aabb_max,
        );

        let grid_info = (
            grid_info.0.into_py_any(py).unwrap(),
            grid_info.1.into_py_any(py).unwrap(),
            grid_info.2.into_pyobject(py).unwrap(),
            grid_info.3,
            grid_info.4,
        );

        (triangles, vertices, grid_info)
    } else if let Ok(particles) = particles.downcast::<PyArray2<f64>>() {
        let (triangles, vertices, grid_info) = reconstruct_surface_py_interface::<f64>(
            py,
            particles,
            particle_radius,
            rest_density,
            smoothing_length,
            cube_size,
            iso_surface_threshold,
            enable_multi_threading,
            global_neighborhood_list,
            use_custom_grid_decomposition,
            subdomain_num_cubes_per_dim,
            aabb_min,
            aabb_max,
        );

        let grid_info = (
            grid_info.0.into_py_any(py).unwrap(),
            grid_info.1.into_py_any(py).unwrap(),
            grid_info.2.into_pyobject(py).unwrap(),
            grid_info.3,
            grid_info.4,
        );

        (triangles, vertices, grid_info)
    } else {
        panic!("Couldn't convert particles to f32 or f64 array!")
    }
}

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

#[pyfunction]
#[pyo3(name = "post_processing_f32")]
#[pyo3(signature = (particles, *, particle_radius=0.025, rest_density=1000.0,
    smoothing_length=2.0, cube_size=0.5, iso_surface_threshold=0.6, enable_multi_threading=false,
    global_neighborhood_list=false, use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64,
    aabb_min = None, aabb_max = None
))]
fn post_processing_py_f32<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<f32>,
    //reconstruction: &SurfaceReconstruction<i64, f32>,
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
) -> (Bound<'py, PyArray2<usize>>, Bound<'py, PyArray2<f32>>) {
    let particle_positions: Vec<Vector3<f32>> = particles
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

    let reconstruction = reconstruct_surface::<i64, f32>(&particle_positions, &params).unwrap();

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

    let mesh = post_processing_generic::<i64, f32>(
        particle_positions,
        &reconstruction,
        &params,
        &postprocessing_args,
    )
    .unwrap();

    let points: Vec<f32> = mesh.vertices.iter().flatten().copied().collect();
    let tris: Vec<usize> = mesh.triangles.iter().flatten().copied().collect();
    let triangles = ndarray::Array2::from_shape_vec((mesh.triangles.len(), 3), tris).unwrap();
    let vertices = ndarray::Array2::from_shape_vec((mesh.vertices.len(), 3), points).unwrap();

    (triangles.into_pyarray(py), vertices.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup")]
#[pyo3(signature = (triangles, vertices, grid_info, *, max_iter, keep_vertices))]
fn marching_cubes_cleanup_py<'py>(
    triangles: PyReadonlyArray2<usize>,
    vertices: PyReadonlyArray2<f64>,
    grid_info: ([f64; 3], [f64; 3], f64, [i64; 3], [i64; 3]),
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    let mut mesh = {
        let mut res = TriMesh3d::<f64>::default();

        let verts = {
            let binding = vertices.as_array();
            let verts: ndarray::ArrayBase<ndarray::CowRepr<'_, f64>, ndarray::Dim<[usize; 1]>> =
                binding.flatten();
            let mut res = Vec::new();
            let mut i = 0;
            while i < verts.len() {
                res.push(Vector3::new(verts[i], verts[i + 1], verts[i + 2]));
                i += 3;
            }
            res
        };

        let tris = {
            let binding = triangles.as_array();
            let tris = binding.flatten();
            let mut res = Vec::new();
            let mut i = 0;
            while i < tris.len() {
                res.push([tris[i], tris[i + 1], tris[i + 2]]);
                i += 3;
            }
            res
        };

        res.vertices = verts;
        res.triangles = tris;

        res
    };

    let grid = UniformCartesianCubeGrid3d::<i64, f64>::new(
        &Vector3::new(grid_info.0[0], grid_info.0[1], grid_info.0[2]),
        &grid_info.4,
        grid_info.2,
    )
    .expect("returned grid");

    let verts_map = marching_cubes_cleanup(&mut mesh, &grid, max_iter, keep_vertices);

    verts_map
}

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface_py_dynamic, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes_cleanup_py, m)?);
    Ok(())
}
