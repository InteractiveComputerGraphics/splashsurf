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

pub fn transmute_replace_into<
    GenericSrc: 'static,
    ConcreteSrc: Into<Target> + 'static,
    Target,
>(
    value: &mut GenericSrc,
    replace: ConcreteSrc,
) -> Option<Target> {
    if std::any::TypeId::of::<GenericSrc>() == std::any::TypeId::of::<ConcreteSrc>() {
        let value_ref = unsafe { std::mem::transmute::<&mut GenericSrc, &mut ConcreteSrc>(value) };
        Some(std::mem::replace(value_ref, replace).into())
    } else {
        None
    }
}
