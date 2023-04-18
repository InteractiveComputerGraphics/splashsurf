use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{AddAssign, MulAssign, SubAssign};

use bytemuck::Pod;
use nalgebra::{RealField, SVector};
use num_integer::Integer;
use num_traits::{Bounded, CheckedAdd, CheckedMul, CheckedSub, FromPrimitive, ToPrimitive};

/// Convenience trait that combines `Send` and `Sync`
pub trait ThreadSafe: Sync + Send {}
impl<T> ThreadSafe for T where T: Sync + Send {}

/// Trait that has to be implemented for types to be used as background grid cell indices in the context of the library
pub trait Index:
    Copy
    + Hash
    + Integer
    + Bounded
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + AddAssign
    + SubAssign
    + MulAssign
    + FromPrimitive
    + ToPrimitive
    + Default
    + Debug
    + Display
    + Pod
    + ThreadSafe
    + 'static
{
    /// Converts this value to the specified [`Real`] type `T` by converting first to `f64` followed by `T::from_f64`. If the value cannot be represented by the target type, `None` is returned.
    fn to_real<R: Real>(self) -> Option<R> {
        R::from_f64(self.to_f64()?)
    }

    /// Converts this value to the specified [`Real`] type, panics if the value cannot be represented by the target type.
    fn to_real_unchecked<R: Real>(self) -> R {
        R::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies this value by the specified `i32` coefficient. Panics if the coefficient cannot be converted into the target type.
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }
}

/// Trait that has to be implemented for types to be used as floating points values in the context of the library (e.g. for coordinates, density values)
pub trait Real:
    RealField
        // Required by RStar and not part of RealFied anymore
        + Bounded
        // Not part of RealField anymore
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Pod
        + ThreadSafe
{
    /// Tries to convert this value to another [`Real`] type `T` by converting first to `f64` followed by `T::from_f64`. If the value cannot be represented by the target type, `None` is returned.
    fn try_convert<T: Real>(self) -> Option<T> {
        Some(T::from_f64(self.to_f64()?)?)
    }

    /// Tries to convert the values of a statically sized `nalgebra::SVector` to another type, same behavior as [`Real::try_convert`]
    fn try_convert_vec_from<R, const D: usize>(vec: &SVector<R, D>) -> Option<SVector<Self, D>>
    where
        R: Real,
    {
        let mut converted = SVector::<Self, D>::zeros();
        for i in 0..D {
            converted[i] = vec[i].try_convert()?
        }
        Some(converted)
    }

    /// Converts this value to the specified [`Index`] type. If the value cannot be represented by the target type, `None` is returned.
    fn to_index<I: Index>(self) -> Option<I> {
        I::from_f64(self.to_f64()?)
    }

    /// Converts this value to the specified [`Index`] type, panics if the value cannot be represented by the target type.
    fn to_index_unchecked<I: Index>(self) -> I {
        I::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies this value by the specified `i32` coefficient. Panics if the coefficient cannot be converted into the target type.
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }

    /// Multiplies this value by the specified `f64` coefficient. Panics if the coefficient cannot be converted into the target type.
    fn times_f64(self, x: f64) -> Self {
        self.mul(Self::from_f64(x).unwrap())
    }
}

impl<T> Index for T where
    T: Copy
        + Hash
        + Integer
        + Bounded
        + CheckedAdd
        + CheckedSub
        + CheckedMul
        + AddAssign
        + SubAssign
        + MulAssign
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Display
        + Pod
        + ThreadSafe
        + 'static
{
}

impl<
        T: RealField
            + Bounded
            + Copy
            + FromPrimitive
            + ToPrimitive
            + Debug
            + Default
            + Pod
            + ThreadSafe
            + 'static,
    > Real for T
{
}
