use pyo3::prelude::*;

mod structs;

mod marching_cubes;
mod decimation;
mod laplacian_smoothing;
mod post_processing;
mod reconstruct_surface;
mod neighborhood_search;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<structs::PyTriMesh3dF32>()?;
    let _ = m.add_class::<structs::PyTriMesh3dF64>()?;

    let _ = m.add_class::<structs::PyMixedTriQuadMesh3dF32>()?;
    let _ = m.add_class::<structs::PyMixedTriQuadMesh3dF64>()?;

    let _ = m.add_class::<structs::PyUniformGridF32>()?;
    let _ = m.add_class::<structs::PyUniformGridF64>()?;

    let _ = m.add_class::<structs::PySurfaceReconstructionF32>()?;
    let _ = m.add_class::<structs::PySurfaceReconstructionF64>()?;

    let _ = m.add_class::<structs::PyMeshWithDataF32>()?;
    let _ = m.add_class::<structs::PyMeshWithDataF64>()?;

    let _ = m.add_class::<structs::PySphInterpolatorF32>()?;
    let _ = m.add_class::<structs::PySphInterpolatorF64>()?;

    let _ = m.add_class::<structs::PyAabb3dF32>()?;
    let _ = m.add_class::<structs::PyAabb3dF64>()?;

    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface::reconstruct_surface_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface::reconstruct_surface_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::post_processing_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::post_processing_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(post_processing::convert_tris_to_quads_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::convert_tris_to_quads_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(marching_cubes::marching_cubes_cleanup_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes::marching_cubes_cleanup_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(marching_cubes::check_mesh_consistency_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes::check_mesh_consistency_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(decimation::decimation_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(decimation::decimation_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::par_laplacian_smoothing_inplace_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::par_laplacian_smoothing_inplace_py_f64, m)?);
    let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::par_laplacian_smoothing_normals_inplace_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::par_laplacian_smoothing_normals_inplace_py_f64, m)?);

    let _ = m.add_function(wrap_pyfunction!(neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f64, m)?);

    // let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::calculate_smoothed_normals_py_f32, m)?);
    // let _ = m.add_function(wrap_pyfunction!(laplacian_smoothing::calculate_smoothed_normals_py_f64, m)?);

    Ok(())
}
