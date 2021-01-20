use std::fmt::{Debug, Display};
use std::hash::Hash;

use bitflags::_core::ops::{AddAssign, MulAssign, SubAssign};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, RealField, VectorN};
use num::{Bounded, CheckedAdd, CheckedMul, CheckedSub, FromPrimitive, Integer, ToPrimitive};

/// Trait that has to be implemented by [Index] and [Real] types to use them in parallelized algorithms
pub trait ThreadSafe: Sync + Send {}
impl<T> ThreadSafe for T where T: Sync + Send {}

/// Trait that has to be implemented for types to be used as grid cell indices in the context of the library
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
    + ThreadSafe
    + 'static
{
    /// Converts the value to the specified [Real] type. If the value cannot be represented by the target type, `None` is returned.
    fn to_real<R: Real>(self) -> Option<R> {
        R::from_f64(self.to_f64()?)
    }

    /// Converts the value to the specified [Real] type, panics if the value cannot be represented by the target type.
    fn to_real_unchecked<R: Real>(self) -> R {
        R::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies the value by the specified `i32` coefficient. Panics if the coefficient cannot be converted into the target type.
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }
}

/// Trait that has to be implemented for types to be used as floating points values in the context of the library (e.g. for coordinates, density values)
pub trait Real: RealField + FromPrimitive + ToPrimitive + Debug + Default + ThreadSafe {
    fn try_convert<T: Real>(self) -> Option<T> {
        Some(T::from_f64(self.to_f64()?)?)
    }

    fn try_convert_vec_from<R, D>(vec: &VectorN<R, D>) -> Option<VectorN<Self, D>>
    where
        R: Real,
        D: DimName,
        DefaultAllocator: Allocator<R, D>,
        DefaultAllocator: Allocator<Self, D>,
    {
        let mut converted = VectorN::<Self, D>::zeros();
        for i in 0..D::dim() {
            converted[i] = vec[i].try_convert()?
        }
        Some(converted)
    }

    /// Converts the value to the specified [Index] type. If the value cannot be represented by the target type, `None` is returned.
    fn to_index<I: Index>(self) -> Option<I> {
        I::from_f64(self.to_f64()?)
    }

    /// Converts the value to the specified [Index] type, panics if the value cannot be represented by the target type.
    fn to_index_unchecked<I: Index>(self) -> I {
        I::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies the value by the specified `i32` coefficient. Panics if the coefficient cannot be converted into the target type.
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }

    /// Multiplies the value by the specified `f64` coefficient. Panics if the coefficient cannot be converted into the target type.
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
        + ThreadSafe
        + 'static
{
}

impl<T: RealField + FromPrimitive + ToPrimitive + Debug + Default + ThreadSafe + 'static> Real
    for T
{
}
