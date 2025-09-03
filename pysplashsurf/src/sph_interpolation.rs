use numpy as np;
use numpy::prelude::*;
use numpy::{Element, PyArray1, PyArray2, PyUntypedArray};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::nalgebra::SVector;
use splashsurf_lib::{
    Real,
    nalgebra::{Unit, Vector3},
    sph_interpolation::SphInterpolator,
};

use crate::utils::*;

enum PySphInterpolatorWrapper {
    F32(SphInterpolator<f32>),
    F64(SphInterpolator<f64>),
}

/// Interpolator of per-particle quantities to arbitrary points using SPH interpolation (with cubic kernel)
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "SphInterpolator")]
pub struct PySphInterpolator {
    inner: PySphInterpolatorWrapper,
}

enum_wrapper_impl_from!(PySphInterpolator, SphInterpolator<f32> => PySphInterpolatorWrapper::F32);
enum_wrapper_impl_from!(PySphInterpolator, SphInterpolator<f64> => PySphInterpolatorWrapper::F64);

impl PySphInterpolator {
    fn new_generic<'py, R: Real + Element>(
        particle_positions: &Bound<'py, PyUntypedArray>,
        particle_densities: &Bound<'py, PyUntypedArray>,
        particle_rest_mass: f64,
        compact_support_radius: f64,
    ) -> PyResult<PySphInterpolator>
    where
        PySphInterpolator: From<SphInterpolator<R>>,
    {
        if let (Ok(particles), Ok(densities)) = (
            particle_positions.downcast::<PyArray2<R>>(),
            particle_densities.downcast::<PyArray1<R>>(),
        ) {
            let particles = particles.try_readonly()?;
            let particles: &[Vector3<R>] = bytemuck::cast_slice(particles.as_slice()?);

            let densities = densities.try_readonly()?;
            let densities = densities.as_slice()?;

            Ok(PySphInterpolator::from(SphInterpolator::new(
                particles,
                densities,
                R::from_float(particle_rest_mass),
                R::from_float(compact_support_radius),
            )))
        } else {
            Err(pyerr_scalar_type_mismatch())
        }
    }

    fn interpolate_normals_generic<'py, R: Real + Element>(
        interpolator: &SphInterpolator<R>,
        interpolation_points: &Bound<'py, PyUntypedArray>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let py = interpolation_points.py();
        if let Ok(points) = interpolation_points.downcast::<PyArray2<R>>() {
            let points = points.try_readonly()?;
            let points: &[Vector3<R>] = bytemuck::cast_slice(points.as_slice()?);

            let normals_vec = interpolator.interpolate_normals(points);
            Ok(bytemuck::cast_vec::<Unit<Vector3<R>>, R>(normals_vec)
                .into_pyarray(py)
                .reshape((points.len(), 3))?
                .into_any()
                .downcast_into::<PyUntypedArray>()
                .expect("downcast should not fail"))
        } else {
            Err(pyerr_unsupported_scalar())
        }
    }

    fn interpolate_vector_generic<'py, R: Real + Element>(
        interpolator: &SphInterpolator<R>,
        particle_quantity: &Bound<'py, PyUntypedArray>,
        interpolation_points: &Bound<'py, PyUntypedArray>,
        first_order_correction: bool,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let shape_in = particle_quantity.shape();
        if ![1, 2].contains(&shape_in.len()) || shape_in[0] != interpolator.size() {
            return Err(PyValueError::new_err(
                "unsupported shape of per particle quantity",
            ));
        }
        let n_components = shape_in.get(1).copied().unwrap_or(1);
        let shape_out = {
            let mut s = shape_in.to_vec();
            s[0] = interpolation_points.shape()[0];
            s
        };

        // Get the per-particle quantity as a read-only contiguous slice
        let quantity = if let Ok(q) = particle_quantity.downcast::<PyArray1<R>>() {
            q.to_dyn().try_readonly()
        } else if let Ok(q) = particle_quantity.downcast::<PyArray2<R>>() {
            q.to_dyn().try_readonly()
        } else {
            return Err(pyerr_scalar_type_mismatch());
        }?;
        let quantity = quantity.as_slice()?;

        let points = interpolation_points
            .downcast::<PyArray2<R>>()
            .map_err(|_| pyerr_scalar_type_mismatch())?
            .try_readonly()?;
        let points: &[Vector3<R>] = bytemuck::cast_slice(points.as_slice()?);

        fn interpolate_ndim<'py, const D: usize, R: Real + Element>(
            py: Python<'py>,
            interpolator: &SphInterpolator<R>,
            points: &[Vector3<R>],
            quantity: &[R],
            first_order_correction: bool,
            shape: &[usize],
        ) -> PyResult<Bound<'py, PyUntypedArray>> {
            let quantity: &[SVector<R, D>] = bytemuck::cast_slice(quantity);
            let interpolated =
                interpolator.interpolate_vector_quantity(quantity, points, first_order_correction);
            Ok(bytemuck::cast_vec::<_, R>(interpolated)
                .into_pyarray(py)
                .reshape(shape)?
                .into_any()
                .downcast_into::<PyUntypedArray>()
                .expect("downcast should not fail"))
        }

        let py = particle_quantity.py();
        let i = interpolator;
        let shape = &shape_out;
        match n_components {
            1 => interpolate_ndim::<1, R>(py, i, points, quantity, first_order_correction, shape),
            2 => interpolate_ndim::<2, R>(py, i, points, quantity, first_order_correction, shape),
            3 => interpolate_ndim::<3, R>(py, i, points, quantity, first_order_correction, shape),
            4 => interpolate_ndim::<4, R>(py, i, points, quantity, first_order_correction, shape),
            5 => interpolate_ndim::<5, R>(py, i, points, quantity, first_order_correction, shape),
            6 => interpolate_ndim::<6, R>(py, i, points, quantity, first_order_correction, shape),
            7 => interpolate_ndim::<7, R>(py, i, points, quantity, first_order_correction, shape),
            8 => interpolate_ndim::<8, R>(py, i, points, quantity, first_order_correction, shape),
            9 => interpolate_ndim::<9, R>(py, i, points, quantity, first_order_correction, shape),
            _ => Err(PyValueError::new_err(
                "only vector quantities with up to 9 dimensions are supported",
            )),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySphInterpolator {
    /// Constructs an SPH interpolator (with cubic kernels) for the given particles
    ///
    /// Parameters
    /// ----------
    /// particle_positions : numpy.ndarray
    ///     A two-dimensional numpy array of shape (N, 3) containing the positions of the particles that are used for interpolation.
    /// particle_densities : numpy.ndarray
    ///     A one-dimensional numpy array of shape (N,) containing the densities of the particles.
    /// particle_rest_mass
    ///     The rest mass of each particle (assumed to be the same for all particles).
    /// compact_support_radius
    ///     The compact support radius of the cubic spline kernel used for interpolation.
    #[new]
    fn py_new<'py>(
        particle_positions: &Bound<'py, PyUntypedArray>,
        particle_densities: &Bound<'py, PyUntypedArray>,
        particle_rest_mass: f64,
        compact_support_radius: f64,
    ) -> PyResult<Self> {
        let py = particle_positions.py();
        let element_type = particle_positions.dtype();

        if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
            Self::new_generic::<f32>(
                particle_positions,
                particle_densities,
                particle_rest_mass,
                compact_support_radius,
            )
        } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
            Self::new_generic::<f64>(
                particle_positions,
                particle_densities,
                particle_rest_mass,
                compact_support_radius,
            )
        } else {
            Err(pyerr_unsupported_scalar())
        }
    }

    /// Interpolates a scalar or vectorial per particle quantity to the given points
    #[pyo3(signature = (particle_quantity, interpolation_points, *, first_order_correction = false))]
    fn interpolate_quantity<'py>(
        &self,
        particle_quantity: &Bound<'py, PyUntypedArray>,
        interpolation_points: &Bound<'py, PyUntypedArray>,
        first_order_correction: bool,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &self.inner {
            PySphInterpolatorWrapper::F32(interp) => Self::interpolate_vector_generic::<f32>(
                interp,
                particle_quantity,
                interpolation_points,
                first_order_correction,
            ),
            PySphInterpolatorWrapper::F64(interp) => Self::interpolate_vector_generic::<f64>(
                interp,
                particle_quantity,
                interpolation_points,
                first_order_correction,
            ),
        }
    }

    /// Interpolates surface normals (i.e. normalized SPH gradient of the indicator function) of the fluid to the given points using SPH interpolation
    fn interpolate_normals<'py>(
        &self,
        interpolation_points: &Bound<'py, PyUntypedArray>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &self.inner {
            PySphInterpolatorWrapper::F32(interp) => {
                Self::interpolate_normals_generic::<f32>(interp, interpolation_points)
            }
            PySphInterpolatorWrapper::F64(interp) => {
                Self::interpolate_normals_generic::<f64>(interp, interpolation_points)
            }
        }
    }
}
