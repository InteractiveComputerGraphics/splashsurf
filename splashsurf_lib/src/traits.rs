use bytemuck::Pod;
use nalgebra::{RealField, SMatrix};
use num_integer::Integer;
use num_traits::{
    Bounded, CheckedAdd, CheckedMul, CheckedSub, FromPrimitive, NumCast, SaturatingSub, ToPrimitive,
};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Convenience trait that combines `Send` and `Sync`
pub trait ThreadSafe: Sync + Send {}
impl<T> ThreadSafe for T where T: Sync + Send {}

pub struct IndexRange<I: Index> {
    start: I,
    end: I,
}

impl<I: Index> IndexRange<I> {
    fn new(start: I, end: I) -> Self {
        assert!(start <= end, "start must be less or equal to end");
        Self { start, end }
    }

    pub fn iter(self) -> impl Iterator<Item = I> {
        let end = self.end;
        let mut counter = self.start;
        std::iter::from_fn(move || {
            let current = counter;
            if current < end {
                counter += I::one();
                Some(current)
            } else {
                None
            }
        })
    }
}

/// Trait that has to be implemented for types to be used as background grid cell indices in the context of the library
pub trait Index:
    Copy
    + Hash
    + Integer
    + Bounded
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + SaturatingSub
    + AddAssign
    + SubAssign
    + MulAssign
    + FromPrimitive
    + ToPrimitive
    + NumCast
    + Default
    + Debug
    + Display
    + Pod
    + ThreadSafe
    + 'static
{
    fn range(start: Self, end: Self) -> IndexRange<Self> {
        IndexRange::new(start, end)
    }

    fn two() -> Self {
        Self::one() + Self::one()
    }

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

    /// Returns the squared value of this value.
    fn squared(self) -> Self {
        self * self
    }

    /// Returns the cubed value of this value.
    fn cubed(self) -> Self {
        self * self * self
    }

    fn checked_cubed(self) -> Option<Self> {
        self.checked_mul(&self)
            .and_then(|val| val.checked_mul(&self))
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
+ NumCast
+ Debug
+ Default
+ Pod
+ ThreadSafe
{
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
        + SaturatingSub
        + AddAssign
        + SubAssign
        + MulAssign
        + FromPrimitive
        + ToPrimitive
        + NumCast
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
        + NumCast
        + Debug
        + Default
        + Pod
        + ThreadSafe
        + 'static,
> Real for T
{
}

/// Trait for converting values, matrices, etc. from one `Real` type to another.
pub trait RealConvert: Sized {
    type Out<To>
    where
        To: Real;

    /// Tries to convert this value to the target type, returns `None` if value cannot be represented by the target type
    fn try_convert<To: Real>(self) -> Option<Self::Out<To>>;
    /// Converts this value to the target type, panics if value cannot be represented by the target type
    fn convert<To: Real>(self) -> Self::Out<To> {
        self.try_convert().expect("failed to convert")
    }
}

impl<From: Real> RealConvert for &From {
    type Out<To>
        = To
    where
        To: Real;

    fn try_convert<To: Real>(self) -> Option<To> {
        <To as NumCast>::from(*self)
    }
}

impl<From: Real, const R: usize, const C: usize> RealConvert for SMatrix<From, R, C> {
    type Out<To>
        = SMatrix<To, R, C>
    where
        To: Real;

    fn try_convert<To: Real>(self) -> Option<SMatrix<To, R, C>> {
        let mut m_out: SMatrix<To, R, C> = SMatrix::zeros();
        m_out
            .iter_mut()
            .zip(self.iter())
            .try_for_each(|(x_out, x_in)| {
                *x_out = <To as NumCast>::from(*x_in)?;
                Some(())
            })?;
        Some(m_out)
    }
}
