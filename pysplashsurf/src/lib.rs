use pyo3::prelude::*;

mod reconstruct_surface;
mod post_processing;
mod marching_cubes_cleanup;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface::reconstruct_surface_py_dynamic, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::post_processing_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes_cleanup::marching_cubes_cleanup_py, m)?);
    Ok(())
}
