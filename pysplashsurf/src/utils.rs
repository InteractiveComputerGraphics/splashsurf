use pyo3::PyErr;
use pyo3::exceptions::PyTypeError;

pub(crate) type IndexT = i64;

pub(crate) fn pyerr_unsupported_scalar() -> PyErr {
    PyTypeError::new_err("unsupported mesh scalar data type, only f32 and f64 are supported")
}

pub(crate) fn pyerr_mesh_grid_scalar_mismatch() -> PyErr {
    PyTypeError::new_err(
        "unsupported mesh and grid scalar data type combination, both have to be either f32 or f64",
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

macro_rules! impl_from_mesh {
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
pub(crate) use impl_from_mesh;

/// Transmutes from a generic type to a concrete type if they are identical, takes the value and converts it into the target type
pub fn transmute_take_into<
    GenericSrc: 'static,
    ConcreteSrc: Default + Into<Target> + 'static,
    Target,
>(
    value: &mut GenericSrc,
) -> Option<Target> {
    if std::any::TypeId::of::<GenericSrc>() == std::any::TypeId::of::<ConcreteSrc>() {
        let value_ref = unsafe { std::mem::transmute::<&mut GenericSrc, &mut ConcreteSrc>(value) };
        Some(std::mem::take(value_ref).into())
    } else {
        None
    }
}
