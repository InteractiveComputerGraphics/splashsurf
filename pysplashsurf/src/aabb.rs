use numpy as np;
use numpy::prelude::*;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyUntypedArray};
use pyo3::{PyResult, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{Aabb3d, Real, nalgebra::Vector3};

use crate::utils::*;

/// Three-dimensional axis-aligned bounding box defined by its minimum and maximum corners
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "Aabb3d")]
pub struct PyAabb3d {
    min: Vector3<f64>,
    max: Vector3<f64>,
}

impl<R: Real> From<Aabb3d<R>> for PyAabb3d {
    fn from(aabb: Aabb3d<R>) -> Self {
        Self {
            min: aabb.min().map(|x| x.to_f64().unwrap()),
            max: aabb.max().map(|x| x.to_f64().unwrap()),
        }
    }
}

impl PyAabb3d {
    /// Convert to an [`splashsurf_lib::Aabb3d`] with the given scalar type
    pub(crate) fn inner<R: Real>(&self) -> Aabb3d<R> {
        Aabb3d::new(
            self.min.map(|x| R::from_f64(x).unwrap()),
            self.max.map(|x| R::from_f64(x).unwrap()),
        )
    }

    fn from_points_generic<'py, R: Real + Element>(
        points: &Bound<'py, PyArray2<R>>,
    ) -> PyResult<Self> {
        let points = points.try_readonly()?;
        let points_vec: &[Vector3<R>] = bytemuck::cast_slice(points.as_slice()?);
        Ok(Self::from(Aabb3d::par_from_points(points_vec)))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAabb3d {
    /// Constructs an AABB with the given min and max coordinates
    #[staticmethod]
    pub fn from_min_max<'py>(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            min: Vector3::from(min),
            max: Vector3::from(max),
        }
    }

    /// Constructs the smallest AABB fitting around all the given points
    #[staticmethod]
    pub fn from_points<'py>(points: &Bound<'py, PyUntypedArray>) -> PyResult<Self> {
        let py = points.py();
        let element_type = points.dtype();
        if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
            Self::from_points_generic(points.downcast::<PyArray2<f32>>()?)
        } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
            Self::from_points_generic(points.downcast::<PyArray2<f64>>()?)
        } else {
            Err(pyerr_unsupported_scalar())
        }
    }

    /// The min coordinate of the AABB
    #[getter]
    pub fn min<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray::from_slice(py, self.min.as_slice())
    }

    /// The max coordinate of the AABB
    #[getter]
    pub fn max<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray::from_slice(py, self.max.as_slice())
    }

    /// Checks if the given point is inside the AABB, the AABB is considered to be half-open to its max coordinate
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        let point = &Vector3::from(point);
        point >= &self.min && point < &self.max
    }
}
