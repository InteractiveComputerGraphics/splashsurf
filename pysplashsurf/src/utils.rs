use ndarray::{ArrayView, IxDyn};
use numpy::{Element, PyArray, PyUntypedArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::{Bound, PyAny, PyErr, PyResult};
use splashsurf_lib::Real;
use splashsurf_lib::nalgebra::SVector;

/// The index type used for all grids and reconstructions in this crate
pub(crate) type IndexT = i64;

pub(crate) fn pyerr_unsupported_scalar() -> PyErr {
    PyTypeError::new_err("unsupported mesh scalar data type, only f32 and f64 are supported")
}

pub(crate) fn pyerr_scalar_type_mismatch() -> PyErr {
    PyTypeError::new_err(
        "unsupported combination of scalar data types, all parameters must have the same type (f32 or f64)",
    )
}

pub(crate) fn pyerr_only_triangle_mesh() -> PyErr {
    PyTypeError::new_err("unsupported mesh type, only triangle meshes are supported")
}

pub(crate) fn pyerr_only_tri_and_tri_quad_mesh() -> PyErr {
    PyTypeError::new_err(
        "unsupported mesh type, only triangle and mixed triangle-quad meshes are supported",
    )
}

macro_rules! enum_wrapper_impl_from {
    ($pyclass:ident, $mesh:ty => $target_enum:path) => {
        impl From<$mesh> for $pyclass {
            fn from(mesh: $mesh) -> Self {
                Self {
                    inner: $target_enum(mesh),
                }
            }
        }
    };
}

macro_rules! enum_impl_from {
    ($enum_t:ident, $from_t:ty => $to_variant:path) => {
        impl From<$from_t> for $enum_t {
            fn from(value: $from_t) -> Self {
                $to_variant(value)
            }
        }
    };
}

pub(crate) use enum_impl_from;
pub(crate) use enum_wrapper_impl_from;

pub enum PyFloatVecWrapper {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

enum_impl_from!(PyFloatVecWrapper, Vec<f32> => PyFloatVecWrapper::F32);
enum_impl_from!(PyFloatVecWrapper, Vec<f64> => PyFloatVecWrapper::F64);

impl PyFloatVecWrapper {
    pub fn try_from_generic<R: Real + 'static>(mut vec: Vec<R>) -> PyResult<Self> {
        transmute_same_take::<Vec<R>, Vec<f32>>(&mut vec)
            .map(PyFloatVecWrapper::F32)
            .or_else(|| {
                transmute_same_take::<Vec<R>, Vec<f64>>(&mut vec).map(PyFloatVecWrapper::F64)
            })
            .ok_or_else(pyerr_unsupported_scalar)
    }

    pub fn view<'py>(&self, container: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
        match self {
            PyFloatVecWrapper::F32(v) => view_scalar_generic(v, container),
            PyFloatVecWrapper::F64(v) => view_scalar_generic(v, container),
        }
    }
}

/// Transmutes a mutable reference from a generic type to a concrete type if they are identical, otherwise returns None
pub(crate) fn transmute_same_mut<GenericSrc: 'static, ConcreteSrc: 'static>(
    value: &mut GenericSrc,
) -> Option<&mut ConcreteSrc> {
    if std::any::TypeId::of::<GenericSrc>() == std::any::TypeId::of::<ConcreteSrc>() {
        Some(unsafe { std::mem::transmute::<&mut GenericSrc, &mut ConcreteSrc>(value) })
    } else {
        None
    }
}

/// Transmutes between types if they are identical and takes the value out of the source
pub(crate) fn transmute_same_take<GenericSrc: 'static, ConcreteSrc: Default + 'static>(
    value: &mut GenericSrc,
) -> Option<ConcreteSrc> {
    transmute_same_mut::<GenericSrc, ConcreteSrc>(value).map(|value_ref| std::mem::take(value_ref))
}

/// Transmutes from a generic type to a concrete type if they are identical, takes the value and converts it into the target type
pub(crate) fn transmute_take_into<
    GenericSrc: 'static,
    ConcreteSrc: Default + Into<Target> + 'static,
    Target,
>(
    value: &mut GenericSrc,
) -> Option<Target> {
    transmute_same_mut::<GenericSrc, ConcreteSrc>(value)
        .map(|value_ref| std::mem::take(value_ref).into())
}

/// Transmutes from a generic type to a concrete type if they are identical, replaces the value and converts it into the target type
pub(crate) fn transmute_replace_into<
    GenericSrc: 'static,
    ConcreteSrc: Into<Target> + 'static,
    Target,
>(
    value: &mut GenericSrc,
    replacement: ConcreteSrc,
) -> Option<Target> {
    transmute_same_mut::<GenericSrc, ConcreteSrc>(value)
        .map(|value_ref| std::mem::replace(value_ref, replacement).into())
}

pub(crate) fn view_generic<'py, R: Element>(
    values: &[R],
    shape: &[usize],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    assert_eq!(
        shape.iter().product::<usize>(),
        values.len(),
        "shape does not match values length"
    );
    let array: ArrayView<R, IxDyn> =
        ArrayView::from_shape(shape, values).map_err(anyhow::Error::new)?;
    let pyarray = unsafe { PyArray::borrow_from_array(&array, container) };
    Ok(pyarray
        .into_any()
        .downcast_into::<PyUntypedArray>()
        .expect("downcast should not fail"))
}

pub(crate) fn view_scalar_generic<'py, R: Element>(
    values: &[R],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    view_generic(values, &[values.len()], container)
}

pub(crate) fn view_vec_generic<'py, R: Real + Element, const D: usize>(
    values: &[SVector<R, D>],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let coordinates: &[R] = bytemuck::cast_slice(values);
    view_generic(coordinates, &[values.len(), D], container)
}
