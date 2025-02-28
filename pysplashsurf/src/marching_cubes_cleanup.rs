use pyo3::prelude::*;
use splashsurf_lib::postprocessing::marching_cubes_cleanup;

use crate::structs::{PyTriMesh3dF64, PyUniformGridF64};

#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup")]
#[pyo3(signature = (mesh, grid, *, max_iter, keep_vertices))]
pub fn marching_cubes_cleanup_py<'py>(
    mesh: &mut PyTriMesh3dF64,
    grid: &PyUniformGridF64,
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {

    let verts_map = marching_cubes_cleanup(&mut mesh.inner, &grid.inner, max_iter, keep_vertices);

    verts_map
}
