use numpy as np;
use numpy::prelude::*;
use numpy::{PyArray2, PyUntypedArray};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::nalgebra::Vector3;

use crate::aabb::PyAabb3d;
use crate::utils::*;

// TODO: Bindings for flat neighborhood search
// TODO: Bindings for computing particle densities

/// Per particle neighborhood lists
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "NeighborhoodLists")]
pub struct PyNeighborhoodLists {
    inner: Vec<Vec<usize>>,
}

impl From<Vec<Vec<usize>>> for PyNeighborhoodLists {
    fn from(nl: Vec<Vec<usize>>) -> Self {
        Self { inner: nl }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNeighborhoodLists {
    /// Returns the number of particles for which neighborhood lists are stored
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns the neighborhood list for the particle at the given index
    pub fn __getitem__(&self, idx: isize) -> PyResult<Vec<usize>> {
        let len = self.inner.len() as isize;
        let idx = if idx < 0 { len + idx } else { idx };
        if idx < 0 || idx >= len {
            Err(PyIndexError::new_err("index out of bounds"))
        } else {
            Ok(self.inner[idx as usize].clone())
        }
    }

    /// Returns all stored neighborhood lists as a list of lists
    pub fn get_neighborhood_lists(&self) -> Vec<Vec<usize>> {
        self.inner.clone()
    }
}

/// Performs a neighborhood search using spatial hashing (multithreaded implementation)
///
/// Parameters
/// ----------
/// particles : numpy.ndarray
///     A two-dimensional numpy array of shape (N, 3) containing the positions of the particles.
/// domain
///     An axis-aligned bounding box (AABB) of the particles used for spatial hashing.
///     The neighborhood search fails if particles are outside the domain.
/// search_radius
///     The radius per particle where other particles are considered neighbors.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "neighborhood_search_spatial_hashing_parallel")]
#[pyo3(signature = (particle_positions, domain, search_radius))]
pub fn neighborhood_search_spatial_hashing_parallel<'py>(
    particle_positions: &Bound<'py, PyUntypedArray>,
    domain: &Bound<'py, PyAabb3d>,
    search_radius: f64,
) -> PyResult<PyNeighborhoodLists> {
    let mut nl: Vec<Vec<usize>> = Vec::new();

    let py = particle_positions.py();
    let element_type = particle_positions.dtype();
    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        let particle_positions = particle_positions
            .downcast::<PyArray2<f32>>()?
            .try_readonly()?;
        let particles: &[Vector3<f32>] = bytemuck::cast_slice(particle_positions.as_slice()?);

        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<i64, f32>(
            &domain.borrow().inner(),
            particles,
            search_radius as f32,
            &mut nl,
        );
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        let particle_positions = particle_positions
            .downcast::<PyArray2<f64>>()?
            .try_readonly()?;
        let particles: &[Vector3<f64>] = bytemuck::cast_slice(particle_positions.as_slice()?);

        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<i64, f64>(
            &domain.borrow().inner(),
            particles,
            search_radius,
            &mut nl,
        );
    } else {
        return Err(pyerr_unsupported_scalar());
    }

    Ok(nl.into())
}
