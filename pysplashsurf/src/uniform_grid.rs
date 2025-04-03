use pyo3::prelude::*;
use splashsurf_lib::UniformGrid;
use pyo3_stub_gen::derive::*;

macro_rules! create_grid_interface {
    ($name: ident, $type: ident) => {
        /// UniformGrid wrapper
        #[gen_stub_pyclass]
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



create_grid_interface!(UniformGridF64, f64);
create_grid_interface!(UniformGridF32, f32);