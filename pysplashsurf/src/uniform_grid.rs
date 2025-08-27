use crate::utils::impl_from_mesh;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::UniformGrid;

enum PyUniformGridData {
    F32(UniformGrid<u64, f32>),
    F64(UniformGrid<u64, f64>),
}

/// Struct containing the parameters of the uniform grid used for the surface reconstruction
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "UniformGrid")]
pub struct PyUniformGrid {
    inner: PyUniformGridData,
}

impl_from_mesh!(PyUniformGrid, UniformGrid<u64, f32> => PyUniformGridData::F32);
impl_from_mesh!(PyUniformGrid, UniformGrid<u64, f64> => PyUniformGridData::F64);

impl PyUniformGrid {
    pub(crate) fn as_f32(&self) -> Option<&UniformGrid<u64, f32>> {
        match &self.inner {
            PyUniformGridData::F32(grid) => Some(grid),
            _ => None,
        }
    }

    pub(crate) fn as_f64(&self) -> Option<&UniformGrid<u64, f64>> {
        match &self.inner {
            PyUniformGridData::F64(grid) => Some(grid),
            _ => None,
        }
    }
}
