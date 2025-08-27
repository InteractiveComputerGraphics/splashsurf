use crate::mesh::{MeshType, PyMeshWithData, PyTriMesh3d};
use crate::uniform_grid::PyUniformGrid;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

/// Checks the consistency of a reconstructed surface mesh (watertightness, manifoldness), optionally returns a string with details if problems are found
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "check_mesh_consistency")]
#[pyo3(signature = (mesh, grid, *, check_closed = true, check_manifold = true, debug = false))]
pub fn check_mesh_consistency<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: Bound<'py, PyAny>,
    grid: &PyUniformGrid,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<Option<String>> {
    if let Ok(mesh) = mesh.downcast::<PyTriMesh3d>() {
        let mesh = mesh.borrow();
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
            Err(PyTypeError::new_err(
                "invalid combination of grid and mesh scalar data types",
            ))
        }
    } else if let Ok(mesh) = mesh.downcast::<PyMeshWithData>()
        && let mesh = mesh.borrow()
        && mesh.mesh_cell_type() == MeshType::Tri3d
    {
        if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_tri_f32()) {
            Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
                grid,
                &mesh.mesh,
                check_closed,
                check_manifold,
                debug,
            )
            .err())
        } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_tri_f64()) {
            Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
                grid,
                &mesh.mesh,
                check_closed,
                check_manifold,
                debug,
            )
            .err())
        } else {
            Err(PyTypeError::new_err(
                "invalid combination of grid and mesh scalar data types",
            ))
        }
    } else {
        Err(PyTypeError::new_err(
            "unsupported mesh type for consistency check, only triangle meshes are supported",
        ))
    }
}
