use ndarray::ArrayViewMut2;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use splashsurf_lib::nalgebra::Vector3;

use crate::{mesh::{PyMeshWithDataF32, PyMeshWithDataF64, PyMixedTriQuadMesh3dF32, PyMixedTriQuadMesh3dF64, PyTriMesh3dF32, PyTriMesh3dF64}, uniform_grid::{PyUniformGridF32, PyUniformGridF64}};

#[pyfunction]
#[pyo3(name = "convert_tris_to_quads_f64")]
#[pyo3(signature = (mesh, *, non_squareness_limit, normal_angle_limit_rad, max_interior_angle))]
pub fn convert_tris_to_quads_py_f64<'py>(
    mesh: &PyTriMesh3dF64,
    non_squareness_limit: f64,
    normal_angle_limit_rad: f64,
    max_interior_angle: f64,
) -> PyMixedTriQuadMesh3dF64 {
    PyMixedTriQuadMesh3dF64::new(splashsurf_lib::postprocessing::convert_tris_to_quads(&mesh.inner, non_squareness_limit, normal_angle_limit_rad, max_interior_angle))
}

#[pyfunction]
#[pyo3(name = "convert_tris_to_quads_f32")]
#[pyo3(signature = (mesh, *, non_squareness_limit, normal_angle_limit_rad, max_interior_angle))]
pub fn convert_tris_to_quads_py_f32<'py>(
    mesh: &PyTriMesh3dF32,
    non_squareness_limit: f32,
    normal_angle_limit_rad: f32,
    max_interior_angle: f32,
) -> PyMixedTriQuadMesh3dF32 {
    PyMixedTriQuadMesh3dF32::new(splashsurf_lib::postprocessing::convert_tris_to_quads(&mesh.inner, non_squareness_limit, normal_angle_limit_rad, max_interior_angle))
}

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
    splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(&mut mesh.inner.mesh, &vertex_connectivity, iterations, beta, &weights);
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
    splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(&mut mesh.inner.mesh, &vertex_connectivity, iterations, beta, &weights);
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

#[pyfunction]
#[pyo3(name = "decimation_f64")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f64<'py>(
    mesh: &mut PyTriMesh3dF64,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    splashsurf_lib::postprocessing::decimation(&mut mesh.inner, keep_vertices)
}

#[pyfunction]
#[pyo3(name = "decimation_f32")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f32<'py>(
    mesh: &mut PyTriMesh3dF32,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    splashsurf_lib::postprocessing::decimation(&mut mesh.inner, keep_vertices)
}

#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup_f64")]
#[pyo3(signature = (mesh, grid, *, max_iter, keep_vertices))]
pub fn marching_cubes_cleanup_py_f64<'py>(
    mesh: &mut PyTriMesh3dF64,
    grid: &PyUniformGridF64,
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    splashsurf_lib::postprocessing::marching_cubes_cleanup(&mut mesh.inner, &grid.inner, max_iter, keep_vertices)
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
    splashsurf_lib::postprocessing::marching_cubes_cleanup(&mut mesh.inner, &grid.inner, max_iter, keep_vertices)
}