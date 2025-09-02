use crate::aabb::PyAabb3d;
use crate::utils;
use crate::utils::{IndexT, enum_wrapper_impl_from};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{Real, UniformGrid};

enum PyUniformGridData {
    F32(UniformGrid<IndexT, f32>),
    F64(UniformGrid<IndexT, f64>),
}

/// Struct containing the parameters of the uniform grid used for the surface reconstruction
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "UniformGrid")]
pub struct PyUniformGrid {
    inner: PyUniformGridData,
}

enum_wrapper_impl_from!(PyUniformGrid, UniformGrid<IndexT, f32> => PyUniformGridData::F32);
enum_wrapper_impl_from!(PyUniformGrid, UniformGrid<IndexT, f64> => PyUniformGridData::F64);

impl PyUniformGrid {
    pub(crate) fn try_from_generic<R: Real>(mut grid: UniformGrid<IndexT, R>) -> PyResult<Self> {
        utils::transmute_replace_into::<_, UniformGrid<IndexT, f32>, _>(&mut grid, UniformGrid::new_zero())
            .or_else(|| {
                utils::transmute_replace_into::<_, UniformGrid<IndexT, f64>, _>(&mut grid, UniformGrid::new_zero())
            })
            .ok_or_else(|| PyTypeError::new_err("unsupported type of grid, only i64 for Index and f32 and f64 for Real type are supported"))
    }

    pub(crate) fn as_f32(&self) -> Option<&UniformGrid<IndexT, f32>> {
        match &self.inner {
            PyUniformGridData::F32(grid) => Some(grid),
            _ => None,
        }
    }

    pub(crate) fn as_f64(&self) -> Option<&UniformGrid<IndexT, f64>> {
        match &self.inner {
            PyUniformGridData::F64(grid) => Some(grid),
            _ => None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUniformGrid {
    /// The AABB of the grid containing all marching cubes vertices influenced by the particle kernels
    #[getter]
    pub fn aabb(&self) -> PyAabb3d {
        match &self.inner {
            PyUniformGridData::F32(grid) => PyAabb3d::from(grid.aabb().clone()),
            PyUniformGridData::F64(grid) => PyAabb3d::from(grid.aabb().clone()),
        }
    }

    /// Returns the cell size of the uniform grid (the marching cubes voxel size)
    #[getter]
    pub fn cell_size(&self) -> f64 {
        match &self.inner {
            PyUniformGridData::F32(grid) => grid.cell_size() as f64,
            PyUniformGridData::F64(grid) => grid.cell_size(),
        }
    }

    /// Returns the number of points (marching cubes vertices) per dimension in the uniform grid
    #[getter]
    pub fn npoints_per_dim(&self) -> [IndexT; 3] {
        match &self.inner {
            PyUniformGridData::F32(grid) => grid.points_per_dim().clone(),
            PyUniformGridData::F64(grid) => grid.points_per_dim().clone(),
        }
    }

    /// Returns the number of cells (marching cubes voxels) per dimension in the uniform grid
    #[getter]
    pub fn ncells_per_dim(&self) -> [IndexT; 3] {
        match &self.inner {
            PyUniformGridData::F32(grid) => grid.cells_per_dim().clone(),
            PyUniformGridData::F64(grid) => grid.cells_per_dim().clone(),
        }
    }
}
