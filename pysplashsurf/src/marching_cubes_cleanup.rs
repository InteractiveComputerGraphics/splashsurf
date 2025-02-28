use pyo3::prelude::*;
use splashsurf_lib::postprocessing::marching_cubes_cleanup;

use crate::structs::{PyTriMesh3dF64, PyUniformGridF64, PyTriMesh3dF32, PyUniformGridF32};

#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup_f64")]
#[pyo3(signature = (mesh, grid, *, max_iter, keep_vertices))]
pub fn marching_cubes_cleanup_py_f64<'py>(
    mesh: &mut PyTriMesh3dF64,
    grid: &PyUniformGridF64,
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    
    let verts_map = marching_cubes_cleanup(&mut mesh.inner, &grid.inner, max_iter, keep_vertices);

    verts_map
}

#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup_f32")]
#[pyo3(signature = (mesh, grid, *, max_iter, keep_vertices))]
pub fn marching_cubes_cleanup_py_f32<'py>(
    mesh: &mut PyTriMesh3dF32,
    grid: &PyUniformGridF32,
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    
    let verts_map = marching_cubes_cleanup(&mut mesh.inner, &grid.inner, max_iter, keep_vertices);

    verts_map
}