use pyo3::prelude::*;

mod uniform_grid;
mod mesh;
mod aabb;
mod sph_interpolation;

mod marching_cubes;
mod post_processing;
mod reconstruction;
mod neighborhood_search;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<mesh::PyTriMesh3dF32>()?;
    let _ = m.add_class::<mesh::PyTriMesh3dF64>()?;

    let _ = m.add_class::<mesh::PyMixedTriQuadMesh3dF32>()?;
    let _ = m.add_class::<mesh::PyMixedTriQuadMesh3dF64>()?;

    let _ = m.add_class::<mesh::PyMeshWithDataF32>()?;
    let _ = m.add_class::<mesh::PyMeshWithDataF64>()?;

    let _ = m.add_class::<uniform_grid::PyUniformGridF32>()?;
    let _ = m.add_class::<uniform_grid::PyUniformGridF64>()?;

    let _ = m.add_class::<reconstruction::PySurfaceReconstructionF32>()?;
    let _ = m.add_class::<reconstruction::PySurfaceReconstructionF64>()?;

    let _ = m.add_class::<sph_interpolation::PySphInterpolatorF32>()?;
    let _ = m.add_class::<sph_interpolation::PySphInterpolatorF64>()?;

    let _ = m.add_class::<aabb::PyAabb3dF32>()?;
    let _ = m.add_class::<aabb::PyAabb3dF64>()?;

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

    Ok(())
}
