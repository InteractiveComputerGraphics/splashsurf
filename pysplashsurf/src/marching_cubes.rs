use pyo3::{exceptions::{PyRuntimeError, PyValueError}, prelude::*};

use crate::{mesh::{PyTriMesh3dF32, PyTriMesh3dF64, PyTriMeshWithDataF32, PyTriMeshWithDataF64}, uniform_grid::{PyUniformGridF32, PyUniformGridF64}};

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f32")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f32<'py>(
    py: Python,
    grid: &PyUniformGridF32,
    mesh: PyObject,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    if mesh.downcast_bound::<PyTriMesh3dF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<PyTriMesh3dF32>(py).unwrap();
        splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.borrow().inner, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else if mesh.downcast_bound::<PyTriMeshWithDataF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<PyTriMeshWithDataF32>(py).unwrap();
        splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.borrow().inner.mesh, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f64")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f64<'py>(
    py: Python,
    grid: &PyUniformGridF64,
    mesh: PyObject,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    if mesh.downcast_bound::<PyTriMesh3dF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<PyTriMesh3dF64>(py).unwrap();
        splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.borrow().inner, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else if mesh.downcast_bound::<PyTriMeshWithDataF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<PyTriMeshWithDataF64>(py).unwrap();
        splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.borrow().inner.mesh, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}