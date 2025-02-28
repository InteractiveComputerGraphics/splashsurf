use pyo3::prelude::*;

mod structs;

mod marching_cubes_cleanup;
mod post_processing;
mod reconstruct_surface;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<structs::PyTriMesh3dF32>()?;
    let _ = m.add_class::<structs::PyTriMesh3dF64>()?;

    let _ = m.add_class::<structs::PyUniformGridF32>()?;
    let _ = m.add_class::<structs::PyUniformGridF64>()?;

    let _ = m.add_class::<structs::PySurfaceReconstructionF32>()?;
    let _ = m.add_class::<structs::PySurfaceReconstructionF64>()?;

    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface::reconstruct_surface_py_dynamic, m)?);
    let _ = m.add_function(wrap_pyfunction!(post_processing::post_processing_py_f32, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes_cleanup::marching_cubes_cleanup_py, m)?);

    Ok(())
}
