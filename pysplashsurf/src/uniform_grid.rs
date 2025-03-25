use pyo3::prelude::*;
use splashsurf_lib::UniformGrid;

macro_rules! create_grid_interface {
    ($name: ident, $type: ident) => {
        /// UniformGrid wrapper
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



create_grid_interface!(PyUniformGridF64, f64);
create_grid_interface!(PyUniformGridF32, f32);