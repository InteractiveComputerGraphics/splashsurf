use pyo3::prelude::*;

mod uniform_grid;
mod mesh;
mod aabb;
mod sph_interpolation;

mod marching_cubes;
mod post_processing;
mod reconstruction;
mod neighborhood_search;
mod pipeline;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<mesh::TriMesh3dF32>()?;
    let _ = m.add_class::<mesh::TriMesh3dF64>()?;
    let _ = m.add_class::<mesh::MixedTriQuadMesh3dF32>()?;
    let _ = m.add_class::<mesh::MixedTriQuadMesh3dF64>()?;

    let _ = m.add_class::<mesh::TriMeshWithDataF32>()?;
    let _ = m.add_class::<mesh::TriMeshWithDataF64>()?;
    let _ = m.add_class::<mesh::MixedTriQuadMeshWithDataF32>()?;
    let _ = m.add_class::<mesh::MixedTriQuadMeshWithDataF64>()?;

    let _ = m.add_class::<uniform_grid::UniformGridF32>()?;
    let _ = m.add_class::<uniform_grid::UniformGridF64>()?;

    let _ = m.add_class::<reconstruction::SurfaceReconstructionF32>()?;
    let _ = m.add_class::<reconstruction::SurfaceReconstructionF64>()?;

    let _ = m.add_class::<sph_interpolation::SphInterpolatorF32>()?;
    let _ = m.add_class::<sph_interpolation::SphInterpolatorF64>()?;

    let _ = m.add_class::<aabb::Aabb3dF32>()?;
    let _ = m.add_class::<aabb::Aabb3dF64>()?;

    let _ = m.add_function(wrap_pyfunction!(reconstruction::reconstruct_surface_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(reconstruction::reconstruct_surface_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::convert_tris_to_quads_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::convert_tris_to_quads_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::marching_cubes_cleanup_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::marching_cubes_cleanup_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(marching_cubes::check_mesh_consistency_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes::check_mesh_consistency_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::decimation_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::decimation_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::par_laplacian_smoothing_inplace_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::par_laplacian_smoothing_inplace_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::par_laplacian_smoothing_normals_inplace_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::par_laplacian_smoothing_normals_inplace_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(pipeline::reconstruction_pipeline_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(pipeline::reconstruction_pipeline_py_f64, m)?);

    Ok(())
}
