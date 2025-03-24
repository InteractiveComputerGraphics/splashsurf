use numpy::{PyArray2, ToPyArray, PyReadonlyArray2};
use ndarray::{ArrayView, ArrayView2};
use pyo3::{prelude::*, PyResult};
use splashsurf_lib::{nalgebra::{Unit, Vector3}, sph_interpolation::SphInterpolator};

macro_rules! create_sph_interpolator_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: SphInterpolator<$type>,
        }

        impl $name {
            pub fn new(data: SphInterpolator<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(
                particle_positions: &Bound<'py, PyArray2<$type>>,
                particle_densities: Vec<$type>, 
                particle_rest_mass: $type, 
                compact_support_radius: $type
            ) -> PyResult<Self> {
                let particle_positions: PyReadonlyArray2<$type> = particle_positions.extract().unwrap();
                let particle_positions = particle_positions.as_slice().unwrap();
                let particle_positions: &[Vector3<$type>] = bytemuck::cast_slice(particle_positions);

                Ok($name::new(SphInterpolator::new(particle_positions, particle_densities.as_slice(), particle_rest_mass, compact_support_radius)))
            }

            fn interpolate_scalar_quantity<'py>(
                &self, 
                particle_quantity: Vec<$type>, 
                interpolation_points: &Bound<'py, PyArray2<$type>>, 
                first_order_correction: bool
            ) -> Vec<$type> {
                let interpolation_points: PyReadonlyArray2<$type> = interpolation_points.extract().unwrap();
                let interpolation_points = interpolation_points.as_slice().unwrap();
                let interpolation_points: &[Vector3<$type>] = bytemuck::cast_slice(interpolation_points);

                self.inner.interpolate_scalar_quantity(particle_quantity.as_slice(), interpolation_points, first_order_correction)
            }

            fn interpolate_normals<'py>(
                &self, 
                py: Python<'py>,
                interpolation_points: &Bound<'py, PyArray2<$type>>,
            ) -> Bound<'py, PyArray2<$type>> {
                let interpolation_points: PyReadonlyArray2<$type> = interpolation_points.extract().unwrap();
                let interpolation_points = interpolation_points.as_slice().unwrap();
                let interpolation_points: &[Vector3<$type>] = bytemuck::cast_slice(interpolation_points);

                let normals_vec = self.inner.interpolate_normals(interpolation_points);
                let normals_vec = bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);
                //let normals_vec: Vec<$type> = bytemuck::cas

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals)
                        .unwrap();
                
                normals.to_pyarray(py)
            }
        }
    };
}

create_sph_interpolator_interface!(PySphInterpolatorF64, f64);
create_sph_interpolator_interface!(PySphInterpolatorF32, f32);