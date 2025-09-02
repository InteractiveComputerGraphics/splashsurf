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
mod postprocessing;
mod reconstruction;

pub(crate) mod utils;

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mesh::PyTriMesh3d>()?;
    m.add_class::<mesh::PyMixedTriQuadMesh3d>()?;
    m.add_class::<mesh::PyMeshWithData>()?;
    m.add_class::<mesh::PyVertexVertexConnectivity>()?;
    m.add_class::<mesh::PyMeshAttribute>()?;
    m.add_class::<mesh::MeshType>()?;

    m.add_class::<aabb::PyAabb3d>()?;
    m.add_class::<neighborhood_search::PyNeighborhoodLists>()?;
    m.add_class::<uniform_grid::PyUniformGrid>()?;
    m.add_class::<reconstruction::PySurfaceReconstruction>()?;
    m.add_class::<sph_interpolation::PySphInterpolator>()?;

    use wrap_pyfunction as wrap;

    m.add_function(wrap!(reconstruction::reconstruct_surface, m)?)?;
    m.add_function(wrap!(marching_cubes::check_mesh_consistency, m)?)?;
    m.add_function(wrap!(marching_cubes::marching_cubes, m)?)?;
    m.add_function(wrap!(postprocessing::marching_cubes_cleanup, m)?)?;
    m.add_function(wrap!(postprocessing::convert_tris_to_quads, m)?)?;
    m.add_function(wrap!(postprocessing::barnacle_decimation, m)?)?;
    m.add_function(wrap!(postprocessing::laplacian_smoothing_parallel, m)?)?;
    m.add_function(wrap!(
        postprocessing::laplacian_smoothing_normals_parallel,
        m
    )?)?;

    m.add_function(wrap!(
        neighborhood_search::neighborhood_search_spatial_hashing_parallel,
        m
    )?)?;

    m.add_function(wrap!(pipeline::reconstruction_pipeline, m)?)?;

    m.add_function(wrap!(run_splashsurf_py, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "run_splashsurf")]
fn run_splashsurf_py<'py>(args: Bound<'py, PyList>) -> PyResult<()> {
    cli::run_splashsurf(args.iter().map(|arg| {
        arg.downcast::<PyString>()
            .expect("argument wasn't a string")
            .extract::<String>()
            .unwrap()
    }))?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
