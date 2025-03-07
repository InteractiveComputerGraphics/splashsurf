use pyo3::prelude::*;
use splashsurf_lib::{mesh::{AttributeData, MeshAttribute}, nalgebra::{Unit, Vector3}, postprocessing::par_laplacian_smoothing_inplace};

use crate::structs::{PyMeshWithDataF32, PyTriMesh3dF32, PyTriMesh3dF64};

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

#[pyfunction]
#[pyo3(name = "calculate_smoothed_normals_f32")]
#[pyo3(signature = (mesh_with_data, vertex_connectivity, *, smoothing_iters))]
pub fn calculate_smoothed_normals_py_f32<'py>(
    mesh_with_data: &mut PyMeshWithDataF32,
    vertex_connectivity: Vec<Vec<usize>>,
    smoothing_iters: usize
) {
    let tri_normals = mesh_with_data.inner.mesh.par_vertex_normals();
    let mut normals = bytemuck::allocation::cast_vec::<Unit<Vector3<f32>>, Vector3<f32>>(tri_normals);

    splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
        &mut normals,
        &vertex_connectivity,
        smoothing_iters,
    );

    mesh_with_data.inner.point_attributes.push(MeshAttribute::new(
        "normals".to_string(),
        AttributeData::Vector3Real(normals),
    ));
}