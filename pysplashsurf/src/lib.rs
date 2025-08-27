use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3_stub_gen::define_stub_info_gatherer;

use splashsurf::cli;

#[cfg(target_pointer_width = "32")]
pub(crate) use u32 as NumpyUsize;
#[cfg(target_pointer_width = "64")]
pub(crate) use u64 as NumpyUsize;
#[cfg(not(any(target_pointer_width = "64", target_pointer_width = "32")))]
compile_error!("Unsupported target pointer width, only 32 and 64 bit are supported.");

mod aabb;
mod mesh;
mod sph_interpolation;
mod uniform_grid;

mod marching_cubes;
mod neighborhood_search;
mod pipeline;
mod post_processing;
mod reconstruction;

pub(crate) mod utils;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mesh::TriMesh3dF32>()?;
    m.add_class::<mesh::TriMesh3dF64>()?;
    m.add_class::<mesh::MixedTriQuadMesh3dF32>()?;
    m.add_class::<mesh::MixedTriQuadMesh3dF64>()?;

    m.add_class::<mesh::TriMeshWithDataF32>()?;
    m.add_class::<mesh::TriMeshWithDataF64>()?;
    m.add_class::<mesh::MixedTriQuadMeshWithDataF32>()?;
    m.add_class::<mesh::MixedTriQuadMeshWithDataF64>()?;

    m.add_class::<mesh::PyTriMesh3d>()?;
    m.add_class::<mesh::PyMixedTriQuadMesh3d>()?;
    m.add_class::<mesh::PyMeshWithData>()?;

    m.add_class::<uniform_grid::PyUniformGrid>()?;
    m.add_class::<reconstruction::PySurfaceReconstruction>()?;

    m.add_class::<sph_interpolation::SphInterpolatorF32>()?;
    m.add_class::<sph_interpolation::SphInterpolatorF64>()?;

    m.add_class::<aabb::Aabb3dF32>()?;
    m.add_class::<aabb::Aabb3dF64>()?;

    m.add_function(wrap_pyfunction!(reconstruction::reconstruct_surface, m)?)?;

    m.add_function(wrap_pyfunction!(
        post_processing::convert_tris_to_quads_py_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        post_processing::convert_tris_to_quads_py_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        post_processing::marching_cubes_cleanup,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(marching_cubes::check_mesh_consistency, m)?)?;

    m.add_function(wrap_pyfunction!(post_processing::decimation_py_f32, m)?)?;
    m.add_function(wrap_pyfunction!(post_processing::decimation_py_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        post_processing::par_laplacian_smoothing_inplace_py_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        post_processing::par_laplacian_smoothing_inplace_py_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        post_processing::par_laplacian_smoothing_normals_inplace_py_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        post_processing::par_laplacian_smoothing_normals_inplace_py_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        neighborhood_search::neighborhood_search_spatial_hashing_parallel_py_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(pipeline::reconstruction_pipeline, m)?)?;

    m.add_function(wrap_pyfunction!(run_splashsurf_py, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "run_splashsurf")]
fn run_splashsurf_py<'py>(args: Bound<'py, PyList>) -> PyResult<()> {
    cli::run_splashsurf(args.iter().map(|arg| {
        arg.downcast::<PyString>()
            .expect("Argument wasn't a string")
            .extract::<String>()
            .unwrap()
    }))?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
