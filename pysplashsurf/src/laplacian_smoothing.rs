use pyo3::prelude::*;
use splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace;

use crate::structs::{PyTriMesh3dF64, PyTriMesh3dF32};

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f64")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f64<'py>(
    mesh: &mut PyTriMesh3dF64,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f64,
    weights: Vec<f64> // ToDo: Same here
) {
    par_laplacian_smoothing_inplace(&mut mesh.inner, &vertex_connectivity, iterations, beta, &weights);
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f32")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f32<'py>(
    mesh: &mut PyTriMesh3dF32,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f32,
    weights: Vec<f32> // ToDo: Same here
) {
    par_laplacian_smoothing_inplace(&mut mesh.inner, &vertex_connectivity, iterations, beta, &weights);
}