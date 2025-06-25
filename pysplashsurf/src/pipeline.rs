use crate::{
    mesh::{
        MixedTriQuadMeshWithDataF32, MixedTriQuadMeshWithDataF64, TriMeshWithDataF32,
        TriMeshWithDataF64,
    },
    reconstruction::{SurfaceReconstructionF32, SurfaceReconstructionF64},
};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition,
    mesh::{AttributeData, MeshAttribute},
    nalgebra::Vector3,
};

fn reconstruction_pipeline_generic<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
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
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    max_rel_snap_dist: Option<f64>,
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
    let aabb = if let (Some(aabb_min), Some(aabb_max)) = (aabb_min, aabb_max) {
        // Convert the min and max arrays to Vector3
        Some(Aabb3d::new(
            Vector3::from(aabb_min),
            Vector3::from(aabb_max),
        ))
    } else {
        None
    };

    let spatial_decomposition = use_custom_grid_decomposition.then(|| {
        let mut grid_params = GridDecompositionParameters::default();
        grid_params.subdomain_num_cubes_per_dim = subdomain_num_cubes_per_dim;
        SpatialDecomposition::UniformGrid(grid_params)
    });

    let params = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        compact_support_radius: R::from_f64(2.0).unwrap() * smoothing_length * particle_radius,
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb: aabb,
        enable_multi_threading,
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
        mesh_cleanup_snap_dist: max_rel_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        interpolate_attributes: Vec::new(),
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
        attributes,
        &params,
        &postprocessing_args,
    )
}

fn attrs_conversion<R: Real + Element>(
    attributes_to_interpolate: Bound<PyDict>,
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
    aabb_min = None, aabb_max = None, enable_multi_threading = false,
    use_custom_grid_decomposition = false, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup, max_rel_snap_dist = None, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
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
    enable_multi_threading: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    max_rel_snap_dist: Option<f64>,
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
    let particles: PyReadonlyArray2<f32> = particles.extract()?;

    let particle_positions = particles.as_slice()?;
    let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particle_positions);

    let attrs = attrs_conversion(attributes_to_interpolate);

    let splashsurf::reconstruct::ReconstructionResult {
        tri_mesh,
        tri_quad_mesh,
        raw_reconstruction: reconstruction,
    } = reconstruction_pipeline_generic::<i64, f32>(
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
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        max_rel_snap_dist,
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
    )
    .unwrap();

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
    aabb_min = None, aabb_max = None, enable_multi_threading = false,
    use_custom_grid_decomposition = false, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup, max_rel_snap_dist = None, decimate_barnacles, keep_vertices, compute_normals, sph_normals,
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
    enable_multi_threading: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    max_rel_snap_dist: Option<f64>,
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
    let particles: PyReadonlyArray2<f64> = particles.extract()?;

    let particle_positions = particles.as_slice()?;
    let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particle_positions);

    let attrs = attrs_conversion(attributes_to_interpolate);

    let splashsurf::reconstruct::ReconstructionResult {
        tri_mesh,
        tri_quad_mesh,
        raw_reconstruction: reconstruction,
    } = reconstruction_pipeline_generic::<i64, f64>(
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
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        max_rel_snap_dist,
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
    )
    .unwrap();

    Ok((
        tri_mesh.map(TriMeshWithDataF64::new),
        tri_quad_mesh.map(MixedTriQuadMeshWithDataF64::new),
        reconstruction.map(SurfaceReconstructionF64::new),
    ))
}
