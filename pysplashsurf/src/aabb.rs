use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, PyResult};
use splashsurf_lib::{nalgebra::Vector3, Aabb3d};

macro_rules! create_aabb3d_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: Aabb3d<$type>
        }

        impl $name {
            pub fn new(data: Aabb3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(min: [$type; 3], max: [$type; 3]) -> PyResult<Self> {
                Ok($name::new(Aabb3d::<$type>::new(Vector3::from_column_slice(&min), Vector3::from_column_slice(&max))))
            }

            #[staticmethod]
            fn from_points<'py>(points: &Bound<'py, PyArray2<$type>>) -> $name {
                let points: PyReadonlyArray2<$type> = points.extract().unwrap();
                let points = points.as_slice().unwrap();
                let points: &[Vector3<$type>] = bytemuck::cast_slice(points);

                $name::new(Aabb3d::from_points(points))
            }

            fn grow_uniformly(&mut self, margin: $type) {
                self.inner.grow_uniformly(margin);
            }
        }
    };
}

create_aabb3d_interface!(PyAabb3dF64, f64);
create_aabb3d_interface!(PyAabb3dF32, f32);