use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use splashsurf_lib::{mesh::TriMesh3d, Aabb3d, UniformGrid};

macro_rules! create_mesh_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: TriMesh3d<$type>,
        }

        impl $name {
            pub fn new(data: TriMesh3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: Vec<$type> = self.inner.vertices.iter().flatten().copied().collect();
                let vertices =
                    ndarray::Array2::from_shape_vec((self.inner.vertices.len(), 3), points)
                        .unwrap();
                vertices.into_pyarray(py)
            }

            #[getter]
            fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
                let tris: Vec<usize> = self.inner.triangles.iter().flatten().copied().collect();
                let triangles =
                    ndarray::Array2::from_shape_vec((self.inner.triangles.len(), 3), tris).unwrap();
                triangles.into_pyarray(py)
            }
        }
    };
}

macro_rules! create_grid_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: UniformGrid<i64, $type>,
        }

        impl $name {
            pub fn new(data: UniformGrid<i64, $type>) -> Self {
                Self { inner: data }
            }
        }
    };
}

create_mesh_interface!(PyTriMesh3dF64, f64);
create_mesh_interface!(PyTriMesh3dF32, f32);

create_grid_interface!(PyUniformGridF64, f64);
create_grid_interface!(PyUniformGridF32, f32);
