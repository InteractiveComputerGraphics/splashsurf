use ndarray::ArrayViewMut2;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use splashsurf_lib::{nalgebra::Vector3, postprocessing::par_laplacian_smoothing_inplace};

use crate::structs::{PyMeshWithDataF32, PyMeshWithDataF64};

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f64")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f64<'py>(
    mesh: &mut PyMeshWithDataF64,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f64,
    weights: Vec<f64> // ToDo: Same here
) {
    par_laplacian_smoothing_inplace(&mut mesh.inner.mesh, &vertex_connectivity, iterations, beta, &weights);
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f32")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f32<'py>(
    mesh: &mut PyMeshWithDataF32,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f32,
    weights: Vec<f32> // ToDo: Same here
) {
    par_laplacian_smoothing_inplace(&mut mesh.inner.mesh, &vertex_connectivity, iterations, beta, &weights);
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_normals_inplace_f32")]
#[pyo3(signature = (normals, vertex_connectivity, iterations))]
pub fn par_laplacian_smoothing_normals_inplace_py_f32<'py>(
    normals: &Bound<'py, PyArray2<f32>>,
    vertex_connectivity: Vec<Vec<usize>>,
    iterations: usize,
) {
    let mut normals: ArrayViewMut2<f32> = unsafe { normals.as_array_mut() };
    let mut normals_vec: Vec<Vector3<f32>> = bytemuck::cast_vec(normals.as_slice().unwrap().to_vec()); // Copies data temporarily into a vec
    splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(&mut normals_vec, &vertex_connectivity, iterations);
    normals.as_slice_mut().unwrap().copy_from_slice(&bytemuck::cast_slice(normals_vec.as_slice())); // Copy back to numpy array
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_normals_inplace_f64")]
#[pyo3(signature = (normals, vertex_connectivity, iterations))]
pub fn par_laplacian_smoothing_normals_inplace_py_f64<'py>(
    normals: &Bound<'py, PyArray2<f64>>,
    vertex_connectivity: Vec<Vec<usize>>,
    iterations: usize,
) {
    let mut normals: ArrayViewMut2<f64> = unsafe { normals.as_array_mut() };
    let mut normals_vec: Vec<Vector3<f64>> = bytemuck::cast_vec(normals.as_slice().unwrap().to_vec()); // Copies data temporarily into a vec
    splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(&mut normals_vec, &vertex_connectivity, iterations);
    normals.as_slice_mut().unwrap().copy_from_slice(&bytemuck::cast_slice(normals_vec.as_slice())); // Copy back to numpy array
}