use anyhow::anyhow;
use log::info;
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use splashsurf_lib::{
    mesh::{AttributeData, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d},
    nalgebra::{Unit, Vector3},
    profile,
    sph_interpolation::SphInterpolator,
    Aabb3d, Index, Real, SurfaceReconstruction,
};
use std::borrow::Cow;

use crate::{
    mesh::{TriMeshWithDataF32, TriMeshWithDataF64},
    reconstruction::{reconstruct_surface_py, SurfaceReconstructionF32, SurfaceReconstructionF64},
};

fn reconstruction_pipeline_generic<I: Index, R: Real>(
    particles: &[Vector3<R>],
    attributes: Vec<MeshAttribute<R>>,
    particle_radius: R,
    rest_density: R,
    smoothing_length: R,
    cube_size: R,
    iso_surface_threshold: R,
    aabb_min: Option<[R; 3]>,
    aabb_max: Option<[R; 3]>,
    enable_multi_threading: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    global_neighborhood_list: bool,
    // check_mesh_closed: bool,
    // check_mesh_manifold: bool,
    // check_mesh_orientation: bool,
    // check_mesh_debug: bool,
    mesh_cleanup: bool,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    // generate_quads: bool,
    // quad_max_edge_diag_ratio: f64,
    // quad_max_normal_angle: f64,
    // quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    mesh_aabb_min: Option<[R; 3]>,
    mesh_aabb_max: Option<[R; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> Result<(MeshWithData<R, TriMesh3d<R>>, SurfaceReconstruction<I, R>), anyhow::Error> {
    profile!("surface reconstruction");

    let compact_support_radius = R::from_f64(2.0).unwrap() * smoothing_length * particle_radius;

    // Perform the surface reconstruction
    let reconstruction = reconstruct_surface_py::<I, R>(
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

    // let grid = reconstruction.grid();
    let mut mesh_with_data: MeshWithData<R, TriMesh3d<R>> =
        MeshWithData::new(reconstruction.mesh().clone());

    // Perform post-processing
    {
        profile!("postprocessing");
        let mut vertex_connectivity = None;

        if mesh_cleanup {
            info!("Post-processing: Performing mesh cleanup");
            let tris_before = mesh_with_data.mesh.triangles.len();
            let verts_before = mesh_with_data.mesh.vertices.len();
            vertex_connectivity = Some(splashsurf_lib::postprocessing::marching_cubes_cleanup(
                &mut mesh_with_data.mesh,
                reconstruction.grid(),
                5,
                keep_vertices,
            ));
            let tris_after = mesh_with_data.mesh.triangles.len();
            let verts_after = mesh_with_data.mesh.vertices.len();
            info!("Post-processing: Cleanup reduced number of vertices to {:.2}% and number of triangles to {:.2}% of original mesh.", (verts_after as f64 / verts_before as f64) * 100.0, (tris_after as f64 / tris_before as f64) * 100.0)
        }

        // Decimate mesh if requested
        if decimate_barnacles {
            info!("Post-processing: Performing decimation");
            vertex_connectivity = Some(splashsurf_lib::postprocessing::decimation(
                &mut mesh_with_data.mesh,
                keep_vertices,
            ));
        }

        // Initialize SPH interpolator if required later
        let interpolator_required = mesh_smoothing_weights || sph_normals;

        let interpolator = if interpolator_required {
            profile!("initialize interpolator");
            info!("Post-processing: Initializing interpolator...");

            info!(
                "Constructing global acceleration structure for SPH interpolation to {} vertices...",
                mesh_with_data.vertices().len()
            );

            let particle_rest_density = rest_density;
            let particle_rest_volume =
                R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap() * particle_radius.powi(3);
            let particle_rest_mass = particle_rest_volume * particle_rest_density;

            let particle_densities = reconstruction
                .particle_densities()
                .ok_or_else(|| anyhow::anyhow!("Particle densities were not returned by surface reconstruction but are required for SPH normal computation"))?
                .as_slice();
            assert_eq!(
                particles.len(),
                particle_densities.len(),
                "There has to be one density value per particle"
            );

            Some(SphInterpolator::new(
                &particles,
                particle_densities,
                particle_rest_mass,
                compact_support_radius,
            ))
        } else {
            None
        };

        // Compute mesh vertex-vertex connectivity map if required later
        let vertex_connectivity_required =
            normals_smoothing_iters.is_some() || mesh_smoothing_iters.is_some();
        if vertex_connectivity.is_none() && vertex_connectivity_required {
            vertex_connectivity = Some(mesh_with_data.mesh.vertex_vertex_connectivity());
        }

        // Compute smoothing weights if requested
        let smoothing_weights = if mesh_smoothing_weights {
            profile!("compute smoothing weights");
            info!("Post-processing: Computing smoothing weights...");

            // TODO: Switch between parallel/single threaded
            // TODO: Re-use data from reconstruction?

            // Global neighborhood search
            let nl = reconstruction
                .particle_neighbors()
                .map(Cow::Borrowed)
                .unwrap_or_else(||
                    {
                        let search_radius = compact_support_radius;

                        let mut domain = Aabb3d::from_points(particles);
                        domain.grow_uniformly(search_radius);

                        let mut nl = Vec::new();
                        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<I, R>(
                            &domain,
                            particles,
                            search_radius,
                            &mut nl,
                        );
                        assert_eq!(nl.len(), particles.len());
                        Cow::Owned(nl)
                    }
                );

            // Compute weighted neighbor count
            let squared_r = compact_support_radius * compact_support_radius;
            let weighted_ncounts = nl
                .par_iter()
                .enumerate()
                .map(|(i, nl)| {
                    nl.iter()
                        .copied()
                        .map(|j| {
                            let dist = (particles[i] - particles[j]).norm_squared();

                            R::one() - (dist / squared_r).clamp(R::zero(), R::one())
                        })
                        .fold(R::zero(), R::add)
                })
                .collect::<Vec<_>>();

            let vertex_weighted_num_neighbors = {
                profile!("interpolate weighted neighbor counts");
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
                let normalization = R::from_f64(mesh_smoothing_weights_normalization).expect(
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

                if output_mesh_smoothing_weights {
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
        if let Some(mesh_smoothing_iters) = mesh_smoothing_iters {
            profile!("mesh smoothing");
            info!("Post-processing: Smoothing mesh...");

            // TODO: Switch between parallel/single threaded

            let smoothing_weights = smoothing_weights
                .unwrap_or_else(|| vec![R::one(); mesh_with_data.vertices().len()]);

            splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
                &mut mesh_with_data.mesh,
                vertex_connectivity
                    .as_ref()
                    .expect("vertex connectivity is required"),
                mesh_smoothing_iters,
                R::one(),
                &smoothing_weights,
            );
        }

        // Add normals to mesh if requested
        if compute_normals {
            profile!("compute normals");
            info!("Post-processing: Computing surface normals...");

            // Compute normals
            let normals = if sph_normals {
                info!("Using SPH interpolation to compute surface normals");

                let sph_normals = interpolator
                    .as_ref()
                    .expect("interpolator is required")
                    .interpolate_normals(mesh_with_data.vertices());
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(sph_normals)
            } else {
                info!("Using area weighted triangle normals for surface normals");
                profile!("mesh.par_vertex_normals");
                let tri_normals = mesh_with_data.mesh.par_vertex_normals();

                // Convert unit vectors to plain vectors
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(tri_normals)
            };

            // Smooth normals
            if let Some(smoothing_iters) = normals_smoothing_iters {
                info!("Post-processing: Smoothing normals...");

                let mut smoothed_normals = normals.clone();
                splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
                    &mut smoothed_normals,
                    vertex_connectivity
                        .as_ref()
                        .expect("vertex connectivity is required"),
                    smoothing_iters,
                );

                mesh_with_data.point_attributes.push(MeshAttribute::new(
                    "normals".to_string(),
                    AttributeData::Vector3Real(smoothed_normals),
                ));
                if output_raw_normals {
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "raw_normals".to_string(),
                        AttributeData::Vector3Real(normals),
                    ));
                }
            } else {
                mesh_with_data.point_attributes.push(MeshAttribute::new(
                    "normals".to_string(),
                    AttributeData::Vector3Real(normals),
                ));
            }
        }

        // Interpolate attributes if requested
        if !attributes.is_empty() {
            profile!("interpolate attributes");
            info!("Post-processing: Interpolating attributes...");
            let interpolator = interpolator.as_ref().expect("interpolator is required");

            for attribute in attributes.into_iter() {
                info!("Interpolating attribute \"{}\"...", attribute.name);

                match attribute.data {
                    AttributeData::ScalarReal(values) => {
                        let interpolated_values = interpolator.interpolate_scalar_quantity(
                            values.as_slice(),
                            mesh_with_data.vertices(),
                            true,
                        );
                        mesh_with_data.point_attributes.push(MeshAttribute::new(
                            attribute.name,
                            AttributeData::ScalarReal(interpolated_values),
                        ));
                    }
                    AttributeData::Vector3Real(values) => {
                        let interpolated_values = interpolator.interpolate_vector_quantity(
                            values.as_slice(),
                            mesh_with_data.vertices(),
                            true,
                        );
                        mesh_with_data.point_attributes.push(MeshAttribute::new(
                            attribute.name,
                            AttributeData::Vector3Real(interpolated_values),
                        ));
                    }
                    _ => unimplemented!("Interpolation of this attribute type not implemented"),
                }
            }
        }
    }

    // Remove and clamp cells outside of AABB
    let mesh_aabb = if aabb_min != None && aabb_max != None {
        Some(Aabb3d::new(
            Vector3::from(mesh_aabb_min.unwrap()),
            Vector3::from(mesh_aabb_max.unwrap()),
        ))
    } else {
        None
    };

    let mesh_with_data = if let Some(mesh_aabb) = &mesh_aabb {
        profile!("clamp mesh to aabb");
        info!("Post-processing: Clamping mesh to AABB...");

        mesh_with_data.par_clamp_with_aabb(
            &mesh_aabb
                .try_convert()
                .ok_or_else(|| anyhow!("Failed to convert mesh AABB"))?,
            mesh_aabb_clamp_vertices,
            keep_vertices,
        )
    } else {
        mesh_with_data
    };

    // Convert triangles to quads
    // let (tri_mesh, tri_quad_mesh) = if generate_quads {
    //     info!("Post-processing: Convert triangles to quads...");
    //     let non_squareness_limit = R::from_f64(quad_max_edge_diag_ratio).unwrap();
    //     let normal_angle_limit_rad =
    //         R::from_f64(quad_max_normal_angle.to_radians()).unwrap();
    //     let max_interior_angle =
    //         R::from_f64(quad_max_interior_angle.to_radians()).unwrap();

    //     let tri_quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
    //         &mesh_with_data.mesh,
    //         non_squareness_limit,
    //         normal_angle_limit_rad,
    //         max_interior_angle,
    //     );

    //     (None, Some(mesh_with_data.with_mesh(tri_quad_mesh)))
    // } else {
    //     (Some(mesh_with_data), None)
    // };
    Ok((mesh_with_data, reconstruction))
}

fn attrs_conversion<'py, R: Real + Element>(
    attributes_to_interpolate: Bound<'py, PyDict>,
) -> Vec<MeshAttribute<R>> {
    let mut attrs: Vec<MeshAttribute<R>> = Vec::new();
    for (key, value) in attributes_to_interpolate.iter() {
        let key_str: String = key
            .downcast::<PyString>()
            .expect("Key wasn't a string")
            .extract()
            .unwrap();

        if let Ok(value) = value.downcast::<PyArray1<u64>>() {
            let value: Vec<u64> = value
                .extract::<PyReadonlyArray1<u64>>()
                .unwrap()
                .as_slice()
                .unwrap()
                .to_vec();
            let mesh_attr = MeshAttribute::new(key_str, AttributeData::ScalarU64(value));
            attrs.push(mesh_attr);
        } else if let Ok(value) = value.downcast::<PyArray1<R>>() {
            let value: Vec<R> = value
                .extract::<PyReadonlyArray1<R>>()
                .unwrap()
                .as_slice()
                .unwrap()
                .to_vec();
            let mesh_attr = MeshAttribute::new(key_str, AttributeData::ScalarReal(value));
            attrs.push(mesh_attr);
        } else if let Ok(value) = value.downcast::<PyArray2<R>>() {
            let value: PyReadonlyArray2<R> = value.extract().unwrap();

            let value_slice = value.as_slice().unwrap();
            let value_slice: &[Vector3<R>] = bytemuck::cast_slice(value_slice);

            let mesh_attr =
                MeshAttribute::new(key_str, AttributeData::Vector3Real(value_slice.to_vec()));
            attrs.push(mesh_attr);
        } else {
            println!("Couldnt downcast attribute {} to valid type", &key_str);
        }
    }
    attrs
}

#[pyfunction]
#[pyo3(name = "reconstruction_pipeline_f32")]
#[pyo3(signature = (particles, *, attributes_to_interpolate, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold,
    aabb_min = None, aabb_max = None, enable_multi_threading=false,
    use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64, global_neighborhood_list=false,
    mesh_cleanup, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
    normals_smoothing_iters, mesh_smoothing_iters, mesh_smoothing_weights,
    mesh_smoothing_weights_normalization, output_mesh_smoothing_weights,
    output_raw_normals, mesh_aabb_min, mesh_aabb_max, mesh_aabb_clamp_vertices
))]
pub fn reconstruction_pipeline_py_f32<'py>(
    particles: &Bound<'py, PyArray2<f32>>,
    attributes_to_interpolate: Bound<'py, PyDict>,
    particle_radius: f32,
    rest_density: f32,
    smoothing_length: f32,
    cube_size: f32,
    iso_surface_threshold: f32,
    aabb_min: Option<[f32; 3]>,
    aabb_max: Option<[f32; 3]>,
    enable_multi_threading: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    global_neighborhood_list: bool,
    mesh_cleanup: bool,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    mesh_aabb_min: Option<[f32; 3]>,
    mesh_aabb_max: Option<[f32; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> (TriMeshWithDataF32, SurfaceReconstructionF32) {
    let particles: PyReadonlyArray2<f32> = particles.extract().unwrap();

    let particle_positions = particles.as_slice().unwrap();
    let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particle_positions);

    let attrs = attrs_conversion(attributes_to_interpolate);

    let (mesh, reconstruction) = reconstruction_pipeline_generic::<i64, f32>(
        particle_positions,
        attrs,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        aabb_min,
        aabb_max,
        enable_multi_threading,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        global_neighborhood_list,
        mesh_cleanup,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        output_mesh_smoothing_weights,
        output_raw_normals,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_aabb_clamp_vertices,
    )
    .unwrap();

    (
        TriMeshWithDataF32::new(mesh),
        SurfaceReconstructionF32::new(reconstruction),
    )
}

#[pyfunction]
#[pyo3(name = "reconstruction_pipeline_f64")]
#[pyo3(signature = (particles, *, attributes_to_interpolate, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold,
    aabb_min = None, aabb_max = None, enable_multi_threading=false,
    use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64, global_neighborhood_list=false,
    mesh_cleanup, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
    normals_smoothing_iters, mesh_smoothing_iters, mesh_smoothing_weights,
    mesh_smoothing_weights_normalization, output_mesh_smoothing_weights,
    output_raw_normals, mesh_aabb_min, mesh_aabb_max, mesh_aabb_clamp_vertices
))]
pub fn reconstruction_pipeline_py_f64<'py>(
    particles: &Bound<'py, PyArray2<f64>>,
    attributes_to_interpolate: Bound<'py, PyDict>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
    enable_multi_threading: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    global_neighborhood_list: bool,
    mesh_cleanup: bool,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> (TriMeshWithDataF64, SurfaceReconstructionF64) {
    let particles: PyReadonlyArray2<f64> = particles.extract().unwrap();

    let particle_positions = particles.as_slice().unwrap();
    let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particle_positions);

    let attrs = attrs_conversion(attributes_to_interpolate);

    let (mesh, reconstruction) = reconstruction_pipeline_generic::<i64, f64>(
        particle_positions,
        attrs,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        aabb_min,
        aabb_max,
        enable_multi_threading,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        global_neighborhood_list,
        mesh_cleanup,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        output_mesh_smoothing_weights,
        output_raw_normals,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_aabb_clamp_vertices,
    )
    .unwrap();

    (
        TriMeshWithDataF64::new(mesh),
        SurfaceReconstructionF64::new(reconstruction),
    )
}
