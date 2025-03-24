use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{mesh::{PyTriMesh3dF32, PyTriMesh3dF64}, uniform_grid::{PyUniformGridF32, PyUniformGridF64}};

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f32")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f32<'py>(
    grid: &PyUniformGridF32,
    mesh: &PyTriMesh3dF32,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.inner, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
}

#[pyfunction]
#[pyo3(name = "check_mesh_consistency_f64")]
#[pyo3(signature = (grid, mesh, *, check_closed, check_manifold, debug))]
pub fn check_mesh_consistency_py_f64<'py>(
    grid: &PyUniformGridF64,
    mesh: &PyTriMesh3dF64,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<()> {
    splashsurf_lib::marching_cubes::check_mesh_consistency(&grid.inner, &mesh.inner, check_closed, check_manifold, debug).map_err(|x| PyErr::new::<PyRuntimeError, _>(x))
}