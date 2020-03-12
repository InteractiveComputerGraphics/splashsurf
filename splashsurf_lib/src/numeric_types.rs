use std::fmt::{Debug, Display};
use std::hash::Hash;

use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, RealField, VectorN};
use num::{Bounded, CheckedAdd, CheckedMul, CheckedSub, FromPrimitive, Integer, ToPrimitive};

pub trait ThreadSafe: Sync + Send + 'static {}
impl<T> ThreadSafe for T where T: Sync + Send + 'static {}

pub trait Index:
    Copy
    + Hash
    + Integer
    + Bounded
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + ThreadSafe
{
    fn to_real<R: Real>(self) -> Option<R> {
        R::from_f64(self.to_f64()?)
    }

    fn to_real_unchecked<R: Real>(self) -> R {
        R::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies the real value by the specified coefficient, panics if the coefficient cannot be converted into the index type
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }
}

pub trait Real: RealField + FromPrimitive + ToPrimitive + Debug + ThreadSafe {
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

    fn to_index<I: Index>(self) -> Option<I> {
        I::from_f64(self.to_f64()?)
    }

    fn to_index_unchecked<I: Index>(self) -> I {
        I::from_f64(self.to_f64().unwrap()).unwrap()
    }

    /// Multiplies the real value by the specified coefficient, panics if the coefficient cannot be converted into the real type
    fn times(self, n: i32) -> Self {
        self.mul(Self::from_i32(n).unwrap())
    }

    /// Multiplies the real value by the specified coefficient, panics if the coefficient cannot be converted into the real type
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
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + ThreadSafe
{
}

impl<T: RealField + FromPrimitive + ToPrimitive + Debug + ThreadSafe> Real for T {}
