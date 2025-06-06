use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    mesh::{TriMesh3dF32, TriMesh3dF64, TriMeshWithDataF32, TriMeshWithDataF64},
    uniform_grid::{UniformGridF32, UniformGridF64},
};

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f32")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f32<'py>(
    py: Python,
    grid: &UniformGridF32,
    mesh: PyObject,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    if let Ok(mesh) = mesh.downcast_bound::<TriMesh3dF32>(py) {
        splashsurf_lib::marching_cubes::check_mesh_consistency(
            &grid.inner,
            &mesh.borrow().inner,
            check_closed,
            check_manifold,
            debug,
        )
        .map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else if let Ok(mesh) = mesh.downcast_bound::<TriMeshWithDataF32>(py) {
        splashsurf_lib::marching_cubes::check_mesh_consistency(
            &grid.inner,
            &mesh.borrow().inner.mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f64")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f64<'py>(
    py: Python,
    grid: &UniformGridF64,
    mesh: PyObject,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    if let Ok(mesh) = mesh.downcast_bound::<TriMesh3dF64>(py) {
        splashsurf_lib::marching_cubes::check_mesh_consistency(
            &grid.inner,
            &mesh.borrow().inner,
            check_closed,
            check_manifold,
            debug,
        )
        .map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else if let Ok(mesh) = mesh.downcast_bound::<TriMeshWithDataF64>(py) {
        splashsurf_lib::marching_cubes::check_mesh_consistency(
            &grid.inner,
            &mesh.borrow().inner.mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}
