use crate::mesh::PyMeshWithData;
use crate::{
    mesh::{
        MixedTriQuadMeshWithDataF32, MixedTriQuadMeshWithDataF64, TriMeshWithDataF32,
        TriMeshWithDataF64,
    },
    reconstruction::{SurfaceReconstructionF32, SurfaceReconstructionF64},
};
use anyhow::anyhow;
use numpy as np;
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition,
    mesh::{AttributeData, MeshAttribute},
    nalgebra::Vector3,
};
use std::borrow::Cow;

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "reconstruction_pipeline_multi")]
#[pyo3(signature = (particles, *, attributes_to_interpolate = None,
    particle_radius, rest_density = 1000.0, smoothing_length, cube_size, iso_surface_threshold = 0.6,
    aabb_min = None, aabb_max = None, multi_threading = true,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup = false, mesh_cleanup_snap_dist = None, decimate_barnacles = false, keep_vertices = false, compute_normals = false, sph_normals = false,
    normals_smoothing_iters = None, mesh_smoothing_iters = None, mesh_smoothing_weights = true, mesh_smoothing_weights_normalization = 13.0,
    generate_quads = false, quad_max_edge_diag_ratio = 1.75, quad_max_normal_angle = 10.0, quad_max_interior_angle = 135.0,
    output_mesh_smoothing_weights = false, output_raw_normals = false, output_raw_mesh = false,
    mesh_aabb_min = None, mesh_aabb_max = None, mesh_aabb_clamp_vertices = true, dtype = None
))]
pub fn reconstruction_pipeline_multi<'py>(
    particles: &Bound<'py, PyUntypedArray>,
    attributes_to_interpolate: Option<Bound<'py, PyDict>>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
    multi_threading: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    mesh_cleanup_snap_dist: Option<f64>,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    generate_quads: bool,
    quad_max_edge_diag_ratio: f64,
    quad_max_normal_angle: f64,
    quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    output_raw_mesh: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
    dtype: Option<Bound<'py, PyArrayDescr>>,
) -> PyResult<Option<PyMeshWithData>> {
    let py = particles.py();
    let element_type = dtype.unwrap_or_else(|| particles.dtype());

    let particle_aabb = aabb_min
        .zip(aabb_max)
        .map(|(min, max)| Aabb3d::new(Vector3::from(min), Vector3::from(max)));

    let mesh_aabb = mesh_aabb_min
        .zip(mesh_aabb_max)
        .map(|(min, max)| Aabb3d::new(Vector3::from(min), Vector3::from(max)));

    let spatial_decomposition = if subdomain_grid {
        SpatialDecomposition::UniformGrid(GridDecompositionParameters {
            subdomain_num_cubes_per_dim,
            auto_disable: subdomain_grid_auto_disable,
        })
    } else {
        SpatialDecomposition::None
    };

    let parameters = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        compact_support_radius: 2.0 * smoothing_length * particle_radius,
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb,
        enable_multi_threading: multi_threading,
        spatial_decomposition,
        global_neighborhood_list: mesh_smoothing_weights,
    };

    let postprocessing_args = splashsurf::reconstruct::ReconstructionPostprocessingParameters {
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        mesh_cleanup_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        interpolate_attributes: None,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        generate_quads,
        quad_max_edge_diag_ratio,
        quad_max_normal_angle,
        quad_max_interior_angle,
        output_mesh_smoothing_weights,
        output_raw_normals,
        output_raw_mesh,
        mesh_aabb,
        mesh_aabb_clamp_vertices,
    };

    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        println!("Detected f32 particle array");
        let particles = particles.downcast::<PyArray2<f32>>()?;
        let reconstruction = reconstruction_pipeline_generic_impl::<u64, _>(
            particles,
            attributes_to_interpolate,
            &parameters
                .try_convert()
                .expect("failed to convert reconstruction parameters to f32"),
            &postprocessing_args,
        )?;
        let mesh = reconstruction
            .tri_mesh
            .map(|mesh_with_data| PyMeshWithData::try_from(mesh_with_data));
        mesh.transpose().map_err(|err| anyhow!(err).into())
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        println!("Detected f64 particle array");
        let particles = particles.downcast::<PyArray2<f64>>()?;
        let reconstruction = reconstruction_pipeline_generic_impl::<u64, _>(
            particles,
            attributes_to_interpolate,
            &parameters,
            &postprocessing_args,
        )?;
        let mesh = reconstruction
            .tri_mesh
            .map(|mesh_with_data| PyMeshWithData::try_from(mesh_with_data));
        mesh.transpose().map_err(|err| anyhow!(err).into())
    } else {
        Err(PyTypeError::new_err(format!(
            "Unsupported element type: {}",
            element_type
        )))
    }
}

fn reconstruction_pipeline_generic_impl<'py, I: Index, R: Real + Element>(
    particles: &Bound<'py, PyArray2<R>>,
    attributes_to_interpolate: Option<Bound<'py, PyDict>>,
    parameters: &splashsurf_lib::Parameters<R>,
    postprocessing_args: &splashsurf::reconstruct::ReconstructionPostprocessingParameters,
) -> Result<splashsurf::reconstruct::ReconstructionResult<I, R>, anyhow::Error> {
    let particles: PyReadonlyArray2<R> = particles.readonly();
    let particle_positions: &[Vector3<R>] = bytemuck::cast_slice(particles.as_slice()?);

    enum AttributePyView<'a, R: Real + Element> {
        U64(PyReadonlyArray1<'a, u64>),
        Float(PyReadonlyArray1<'a, R>),
        FloatVec3(PyReadonlyArray2<'a, R>),
    }

    let mut attr_names = Vec::new();
    let mut attr_views = Vec::new();

    // Collect readonly views of all attribute arrays
    for (key, value) in attributes_to_interpolate.iter().flatten() {
        let key_str: String = key
            .downcast::<PyString>()
            .expect("attribute key has to be a string")
            .extract()?;

        if let Ok(value) = value.downcast::<PyArray1<u64>>() {
            attr_views.push(AttributePyView::U64(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray1<R>>() {
            attr_views.push(AttributePyView::Float(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray2<R>>() {
            attr_views.push(AttributePyView::FloatVec3(value.readonly()));
            attr_names.push(key_str);
        } else {
            println!("Failed to downcast attribute {} to valid type", &key_str);
        }
    }

    // Get slices from attribute views and construct borrowed MeshAttributes
    let attributes = attr_names
        .into_iter()
        .zip(attr_views.iter())
        .map(|(name, view)| -> Result<MeshAttribute<R>, anyhow::Error> {
            let data = match view {
                AttributePyView::U64(view) => {
                    AttributeData::ScalarU64(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::Float(view) => {
                    AttributeData::ScalarReal(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::FloatVec3(view) => {
                    let vec3_slice: &[Vector3<R>] = bytemuck::cast_slice(view.as_slice()?);
                    AttributeData::Vector3Real(Cow::Borrowed(vec3_slice.into()))
                }
            };
            Ok(MeshAttribute::new(name, data))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut postprocessing_args = postprocessing_args.clone();
    postprocessing_args.interpolate_attributes =
        (!attributes.is_empty()).then(|| attributes.iter().map(|a| a.name.clone()).collect());

    splashsurf::reconstruct::reconstruction_pipeline(
        particle_positions,
        &attributes,
        &parameters,
        &postprocessing_args,
    )
}

fn reconstruction_pipeline_generic<'py, I: Index, R: Real + Element>(
    particles: &Bound<'py, PyArray2<R>>,
    attributes_to_interpolate: Bound<'py, PyDict>,
    particle_radius: R,
    rest_density: R,
    smoothing_length: R,
    cube_size: R,
    iso_surface_threshold: R,
    aabb_min: Option<[R; 3]>,
    aabb_max: Option<[R; 3]>,
    multi_threading: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    mesh_cleanup_snap_dist: Option<f64>,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    generate_quads: bool,
    quad_max_edge_diag_ratio: f64,
    quad_max_normal_angle: f64,
    quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    output_raw_mesh: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> Result<splashsurf::reconstruct::ReconstructionResult<I, R>, anyhow::Error> {
    let particles: PyReadonlyArray2<R> = particles.readonly();
    let particle_positions: &[Vector3<R>] = bytemuck::cast_slice(particles.as_slice()?);

    enum AttributePyView<'a, R: Real + Element> {
        U64(PyReadonlyArray1<'a, u64>),
        Float(PyReadonlyArray1<'a, R>),
        FloatVec3(PyReadonlyArray2<'a, R>),
    }

    let mut attr_names = Vec::new();
    let mut attr_views = Vec::new();

    // Collect readonly views of all attribute arrays
    for (key, value) in attributes_to_interpolate.iter() {
        let key_str: String = key
            .downcast::<PyString>()
            .expect("Key wasn't a string")
            .extract()?;

        if let Ok(value) = value.downcast::<PyArray1<u64>>() {
            attr_views.push(AttributePyView::U64(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray1<R>>() {
            attr_views.push(AttributePyView::Float(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray2<R>>() {
            attr_views.push(AttributePyView::FloatVec3(value.readonly()));
            attr_names.push(key_str);
        } else {
            println!("Couldn't downcast attribute {} to valid type", &key_str);
        }
    }

    // Get slices from attribute views and construct borrowed MeshAttributes
    let attributes = attr_names
        .into_iter()
        .zip(attr_views.iter())
        .map(|(name, view)| -> Result<MeshAttribute<R>, anyhow::Error> {
            let data = match view {
                AttributePyView::U64(view) => {
                    AttributeData::ScalarU64(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::Float(view) => {
                    AttributeData::ScalarReal(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::FloatVec3(view) => {
                    let vec3_slice: &[Vector3<R>] = bytemuck::cast_slice(view.as_slice()?);
                    AttributeData::Vector3Real(Cow::Borrowed(vec3_slice.into()))
                }
            };
            Ok(MeshAttribute::new(name, data))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let aabb = if let (Some(aabb_min), Some(aabb_max)) = (aabb_min, aabb_max) {
        // Convert the min and max arrays to Vector3
        Some(Aabb3d::new(
            Vector3::from(aabb_min),
            Vector3::from(aabb_max),
        ))
    } else {
        None
    };

    let spatial_decomposition = if subdomain_grid {
        SpatialDecomposition::UniformGrid(GridDecompositionParameters {
            subdomain_num_cubes_per_dim,
            auto_disable: subdomain_grid_auto_disable,
        })
    } else {
        SpatialDecomposition::None
    };

    let params = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        compact_support_radius: R::from_float(2.0) * smoothing_length * particle_radius,
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb: aabb,
        enable_multi_threading: multi_threading,
        spatial_decomposition,
        global_neighborhood_list: mesh_smoothing_weights,
    };

    let mesh_aabb =
        if let (Some(mesh_aabb_min), Some(mesh_aabb_max)) = (mesh_aabb_min, mesh_aabb_max) {
            // Convert the min and max arrays to Vector3
            Some(Aabb3d::new(
                Vector3::from(mesh_aabb_min),
                Vector3::from(mesh_aabb_max),
            ))
        } else {
            None
        };

    let postprocessing_args = splashsurf::reconstruct::ReconstructionPostprocessingParameters {
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        mesh_cleanup_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        interpolate_attributes: Some(attributes.iter().map(|a| a.name.clone()).collect()),
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        generate_quads,
        quad_max_edge_diag_ratio,
        quad_max_normal_angle,
        quad_max_interior_angle,
        output_mesh_smoothing_weights,
        output_raw_normals,
        output_raw_mesh,
        mesh_aabb,
        mesh_aabb_clamp_vertices,
    };

    splashsurf::reconstruct::reconstruction_pipeline(
        particle_positions,
        &attributes,
        &params,
        &postprocessing_args,
    )
}

#[pyfunction]
#[pyo3(name = "reconstruction_pipeline_f32")]
#[pyo3(signature = (particles, *, attributes_to_interpolate, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold,
    aabb_min = None, aabb_max = None, multi_threading = true,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup, mesh_cleanup_snap_dist = None, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
    normals_smoothing_iters, mesh_smoothing_iters, mesh_smoothing_weights, mesh_smoothing_weights_normalization,
    generate_quads = false, quad_max_edge_diag_ratio = 1.75, quad_max_normal_angle = 10.0, quad_max_interior_angle = 135.0,
    output_mesh_smoothing_weights, output_raw_normals, output_raw_mesh=false,
    mesh_aabb_min, mesh_aabb_max, mesh_aabb_clamp_vertices
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
    multi_threading: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    mesh_cleanup_snap_dist: Option<f64>,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    generate_quads: bool,
    quad_max_edge_diag_ratio: f64,
    quad_max_normal_angle: f64,
    quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    output_raw_mesh: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> PyResult<(
    Option<TriMeshWithDataF32>,
    Option<MixedTriQuadMeshWithDataF32>,
    Option<SurfaceReconstructionF32>,
)> {
    let splashsurf::reconstruct::ReconstructionResult {
        tri_mesh,
        tri_quad_mesh,
        raw_reconstruction: reconstruction,
    } = reconstruction_pipeline_generic::<i64, f32>(
        particles,
        attributes_to_interpolate,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        aabb_min,
        aabb_max,
        multi_threading,
        subdomain_grid,
        subdomain_grid_auto_disable,
        subdomain_num_cubes_per_dim,
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        mesh_cleanup_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        generate_quads,
        quad_max_edge_diag_ratio,
        quad_max_normal_angle,
        quad_max_interior_angle,
        output_mesh_smoothing_weights,
        output_raw_normals,
        output_raw_mesh,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_aabb_clamp_vertices,
    )?;

    Ok((
        tri_mesh.map(TriMeshWithDataF32::new),
        tri_quad_mesh.map(MixedTriQuadMeshWithDataF32::new),
        reconstruction.map(SurfaceReconstructionF32::new),
    ))
}

#[pyfunction]
#[pyo3(name = "reconstruction_pipeline_f64")]
#[pyo3(signature = (particles, *, attributes_to_interpolate, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold,
    aabb_min = None, aabb_max = None, multi_threading = true,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup, mesh_cleanup_snap_dist = None, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
    normals_smoothing_iters, mesh_smoothing_iters, mesh_smoothing_weights, mesh_smoothing_weights_normalization,
    generate_quads = false, quad_max_edge_diag_ratio = 1.75, quad_max_normal_angle = 10.0, quad_max_interior_angle = 135.0,
    output_mesh_smoothing_weights, output_raw_normals, output_raw_mesh=false,
    mesh_aabb_min, mesh_aabb_max, mesh_aabb_clamp_vertices
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
    multi_threading: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    mesh_cleanup_snap_dist: Option<f64>,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    generate_quads: bool,
    quad_max_edge_diag_ratio: f64,
    quad_max_normal_angle: f64,
    quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    output_raw_mesh: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
) -> PyResult<(
    Option<TriMeshWithDataF64>,
    Option<MixedTriQuadMeshWithDataF64>,
    Option<SurfaceReconstructionF64>,
)> {
    let splashsurf::reconstruct::ReconstructionResult {
        tri_mesh,
        tri_quad_mesh,
        raw_reconstruction: reconstruction,
    } = reconstruction_pipeline_generic::<i64, f64>(
        particles,
        attributes_to_interpolate,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        aabb_min,
        aabb_max,
        multi_threading,
        subdomain_grid,
        subdomain_grid_auto_disable,
        subdomain_num_cubes_per_dim,
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        mesh_cleanup_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        generate_quads,
        quad_max_edge_diag_ratio,
        quad_max_normal_angle,
        quad_max_interior_angle,
        output_mesh_smoothing_weights,
        output_raw_normals,
        output_raw_mesh,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_aabb_clamp_vertices,
    )?;

    Ok((
        tri_mesh.map(TriMeshWithDataF64::new),
        tri_quad_mesh.map(MixedTriQuadMeshWithDataF64::new),
        reconstruction.map(SurfaceReconstructionF64::new),
    ))
}
