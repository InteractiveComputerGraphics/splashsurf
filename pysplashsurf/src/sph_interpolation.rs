use ndarray::{ArrayView, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{PyResult, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{
    nalgebra::{Unit, Vector3},
    sph_interpolation::SphInterpolator,
};

macro_rules! create_sph_interpolator_interface {
    ($name: ident, $type: ident) => {
        /// SphInterpolator wrapper
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: SphInterpolator<$type>,
        }

        impl $name {
            pub fn new(data: SphInterpolator<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(
                particle_positions: &Bound<'py, PyArray2<$type>>,
                particle_densities: Vec<$type>,
                particle_rest_mass: $type,
                compact_support_radius: $type,
            ) -> PyResult<Self> {
                let particle_positions: PyReadonlyArray2<$type> =
                    particle_positions.extract().unwrap();
                let particle_positions = particle_positions.as_slice().unwrap();
                let particle_positions: &[Vector3<$type>] =
                    bytemuck::cast_slice(particle_positions);

                Ok($name::new(SphInterpolator::new(
                    particle_positions,
                    particle_densities.as_slice(),
                    particle_rest_mass,
                    compact_support_radius,
                )))
            }

            /// Interpolates a scalar per particle quantity to the given points, panics if the there are less per-particles values than particles
            fn interpolate_scalar_quantity<'py>(
                &self,
                particle_quantity: Vec<$type>,
                interpolation_points: &Bound<'py, PyArray2<$type>>,
                first_order_correction: bool,
            ) -> PyResult<Vec<$type>> {
                let interpolation_points: PyReadonlyArray2<$type> =
                    interpolation_points.extract()?;
                let interpolation_points = interpolation_points.as_slice()?;
                let interpolation_points: &[Vector3<$type>] =
                    bytemuck::cast_slice(interpolation_points);

                Ok(self.inner.interpolate_scalar_quantity(
                    particle_quantity.as_slice(),
                    interpolation_points,
                    first_order_correction,
                ))
            }

            /// Interpolates surface normals (i.e. normalized SPH gradient of the indicator function) of the fluid to the given points using SPH interpolation
            fn interpolate_normals<'py>(
                &self,
                py: Python<'py>,
                interpolation_points: &Bound<'py, PyArray2<$type>>,
            ) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let interpolation_points: PyReadonlyArray2<$type> =
                    interpolation_points.extract()?;
                let interpolation_points = interpolation_points.as_slice()?;
                let interpolation_points: &[Vector3<$type>] =
                    bytemuck::cast_slice(interpolation_points);

                let normals_vec = self.inner.interpolate_normals(interpolation_points);
                let normals_vec =
                    bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals).unwrap();

                Ok(normals.to_pyarray(py))
            }

            /// Interpolates a vectorial per particle quantity to the given points, panics if the there are less per-particles values than particles
            fn interpolate_vector_quantity<'py>(
                &self,
                py: Python<'py>,
                particle_quantity: &Bound<'py, PyArray2<$type>>,
                interpolation_points: &Bound<'py, PyArray2<$type>>,
                first_order_correction: bool,
            ) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let interpolation_points: PyReadonlyArray2<$type> =
                    interpolation_points.extract()?;
                let interpolation_points = interpolation_points.as_slice()?;
                let interpolation_points: &[Vector3<$type>] =
                    bytemuck::cast_slice(interpolation_points);

                let particle_quantity: PyReadonlyArray2<$type> = particle_quantity.extract()?;
                let particle_quantity = particle_quantity.as_slice()?;
                let particle_quantity: &[Vector3<$type>] = bytemuck::cast_slice(particle_quantity);

                let res_vec = self.inner.interpolate_vector_quantity(
                    particle_quantity,
                    interpolation_points,
                    first_order_correction,
                );
                let res_vec = bytemuck::allocation::cast_vec::<Vector3<$type>, $type>(res_vec);

                let res: &[$type] = res_vec.as_slice();
                let res: ArrayView2<$type> =
                    ArrayView::from_shape((res.len() / 3, 3), res).unwrap();

                Ok(res.to_pyarray(py))
            }
        }
    };
}

create_sph_interpolator_interface!(SphInterpolatorF64, f64);
create_sph_interpolator_interface!(SphInterpolatorF32, f32);
