use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use splashsurf_lib::{mesh::TriMesh3d, UniformGrid, SurfaceReconstruction};

macro_rules! create_mesh_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: TriMesh3d<$type>,
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
            pub inner: UniformGrid<i64, $type>,
        }

        impl $name {
            pub fn new(data: UniformGrid<i64, $type>) -> Self {
                Self { inner: data }
            }
        }
    };
}

macro_rules! create_reconstruction_interface {
    ($name: ident, $type: ident, $mesh_class: ident, $grid_class: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: SurfaceReconstruction<i64, $type>,
        }

        impl $name {
            pub fn new(data: SurfaceReconstruction<i64, $type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh().clone())
            }

            #[getter]
            fn grid(&self) -> $grid_class {
                $grid_class::new(self.inner.grid().clone())
            }
        }
    };
}

create_mesh_interface!(PyTriMesh3dF64, f64);
create_mesh_interface!(PyTriMesh3dF32, f32);

create_grid_interface!(PyUniformGridF64, f64);
create_grid_interface!(PyUniformGridF32, f32);

create_reconstruction_interface!(PySurfaceReconstructionF64, f64, PyTriMesh3dF64, PyUniformGridF64);
create_reconstruction_interface!(PySurfaceReconstructionF32, f32, PyTriMesh3dF32, PyUniformGridF32);