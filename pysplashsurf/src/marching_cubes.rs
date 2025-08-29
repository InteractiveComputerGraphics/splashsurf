use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::mesh::{PyTriMesh3d, get_triangle_mesh_generic};
use crate::uniform_grid::PyUniformGrid;
use crate::utils::*;

/// Checks the consistency of a reconstructed surface mesh (watertightness, manifoldness), optionally returns a string with details if problems are found
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "check_mesh_consistency")]
#[pyo3(signature = (mesh, grid, *, check_closed = true, check_manifold = true, debug = false))]
pub fn check_mesh_consistency<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: &Bound<'py, PyAny>,
    grid: &PyUniformGrid,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<Option<String>> {
    let py = mesh.py();

    // Try to extract the triangle mesh;
    let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(pyerr_only_triangle_mesh)?;
    let mesh = mesh.borrow(py);

    if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_f32()) {
        Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
            grid,
            mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .err())
    } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_f64()) {
        Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
            grid,
            mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .err())
    } else {
        Err(pyerr_mesh_grid_scalar_mismatch())
    }
}
