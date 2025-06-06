use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use splashsurf_lib::{nalgebra::Vector3, neighborhood_search::*};

use crate::aabb::{Aabb3dF32, Aabb3dF64};

#[pyfunction]
#[pyo3(name = "neighborhood_search_spatial_hashing_parallel_f64")]
#[pyo3(signature = (domain, particle_positions, search_radius))]
pub fn neighborhood_search_spatial_hashing_parallel_py_f64<'py>(
    domain: &Aabb3dF64,
    particle_positions: &Bound<'py, PyArray2<f64>>,
    search_radius: f64,
) -> PyResult<Vec<Vec<usize>>> {
    let mut nl: Vec<Vec<usize>> = Vec::new();

    let particle_positions: PyReadonlyArray2<f64> = particle_positions.extract()?;
    let particle_positions = particle_positions.as_slice()?;
    let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particle_positions);

    neighborhood_search_spatial_hashing_parallel::<i64, f64>(
        &domain.inner,
        particle_positions,
        search_radius,
        &mut nl,
    );

    Ok(nl)
}

#[pyfunction]
#[pyo3(name = "neighborhood_search_spatial_hashing_parallel_f32")]
#[pyo3(signature = (domain, particle_positions, search_radius))]
pub fn neighborhood_search_spatial_hashing_parallel_py_f32<'py>(
    domain: &Aabb3dF32,
    particle_positions: &Bound<'py, PyArray2<f32>>,
    search_radius: f32,
) -> PyResult<Vec<Vec<usize>>> {
    let mut nl: Vec<Vec<usize>> = Vec::new();

    let particle_positions: PyReadonlyArray2<f32> = particle_positions.extract()?;
    let particle_positions = particle_positions.as_slice()?;
    let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particle_positions);

    neighborhood_search_spatial_hashing_parallel::<i64, f32>(
        &domain.inner,
        particle_positions,
        search_radius,
        &mut nl,
    );

    Ok(nl)
}
