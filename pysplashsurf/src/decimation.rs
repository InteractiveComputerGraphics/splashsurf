use pyo3::prelude::*;
use splashsurf_lib::postprocessing::decimation;

use crate::structs::{PyTriMesh3dF64, PyTriMesh3dF32};

#[pyfunction]
#[pyo3(name = "decimation_f64")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f64<'py>(
    mesh: &mut PyTriMesh3dF64,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    decimation(&mut mesh.inner, keep_vertices)
}

#[pyfunction]
#[pyo3(name = "decimation_f32")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f32<'py>(
    mesh: &mut PyTriMesh3dF32,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    decimation(&mut mesh.inner, keep_vertices)
}