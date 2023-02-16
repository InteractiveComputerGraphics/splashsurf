//! Helper types for cartesian coordinate system topology

use num_traits::{CheckedAdd, CheckedSub, One};
use std::ops::{Add, Sub};

/// Direction on a number line/coordinate axis or identifiers for the end points of a line
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Direction {
    Negative = 0,
    Positive = 1,
}

/// Abbreviated type alias for cartesian coordinate axes in 3D
pub type Axis = CartesianAxis3d;

/// The cartesian coordinate axes in 3D
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CartesianAxis3d {
    /// The x-axis
    X = 0,
    /// The y-axis
    Y = 1,
    /// The z-axis
    Z = 2,
}

/// Identifies a direction along a specific cartesian axis
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DirectedAxis {
    pub axis: Axis,
    pub direction: Direction,
}

/// Collection that stores one value per unique [`DirectedAxis`], can be used e.g. to store neighbors in a cartesian grid
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub struct DirectedAxisArray<T> {
    data: [T; 6],
}

impl Direction {
    /// Returns a reference to an array containing all possible directions
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert!(Direction::all_possible().iter().any(|d| d.is_positive()));
    /// assert!(Direction::all_possible().iter().any(|d| d.is_negative()));
    /// assert_eq!(Direction::all_possible().iter().count(), 2);
    /// ```
    pub const fn all_possible() -> &'static [Direction; 2] {
        &ALL_DIRECTIONS
    }

    /// Constructs a new positive or negative direction depending on the flag
    #[inline(always)]
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::new_positive(true), Direction::Positive);
    /// assert_eq!(Direction::new_positive(false), Direction::Negative);
    /// ```
    pub const fn new_positive(is_positive: bool) -> Self {
        if is_positive {
            Direction::Positive
        } else {
            Direction::Negative
        }
    }

    /// Returns the opposite direction
    pub const fn opposite(&self) -> Self {
        match self {
            Direction::Positive => Direction::Negative,
            Direction::Negative => Direction::Positive,
        }
    }

    /// Adds or subtracts the given step from the value depending on the direction
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::Positive.apply_step(27, 3), 30);
    /// assert_eq!(Direction::Negative.apply_step(27, 3), 24);
    /// ```
    #[inline(always)]
    pub fn apply_step<N: Add<Output = N> + Sub<Output = N>>(&self, n: N, step: N) -> N {
        if self.is_positive() {
            n + step
        } else {
            n - step
        }
    }

    /// Same as `apply_step` but uses `checked_add` and `checked_sub`, i.e. returns `None` on overflow
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::Negative.checked_apply_step(0 as i32, 10), Some(-10));
    /// assert_eq!(Direction::Negative.checked_apply_step(0 as u32, 10), None);
    /// ```
    #[inline(always)]
    pub fn checked_apply_step<N: CheckedAdd<Output = N> + CheckedSub<Output = N>>(
        &self,
        n: N,
        step: N,
    ) -> Option<N> {
        if self.is_positive() {
            n.checked_add(&step)
        } else {
            n.checked_sub(&step)
        }
    }

    /// Adds or subtracts a 3D step from the array depending on the direction, returns `None` on overflow
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::Negative.checked_apply_step_ijk(&[1, 2, 3], &[0, 1, 2]), Some([1, 1, 1]));
    /// ```
    #[inline(always)]
    pub fn checked_apply_step_ijk<N: CheckedAdd<Output = N> + CheckedSub<Output = N>>(
        &self,
        start: &[N; 3],
        step: &[N; 3],
    ) -> Option<[N; 3]> {
        Some(if self.is_positive() {
            [
                start[0].checked_add(&step[0])?,
                start[1].checked_add(&step[1])?,
                start[2].checked_add(&step[2])?,
            ]
        } else {
            [
                start[0].checked_sub(&step[0])?,
                start[1].checked_sub(&step[1])?,
                start[2].checked_sub(&step[2])?,
            ]
        })
    }

    /// Returns whether the direction is positive
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::Positive.is_positive(), true);
    /// assert_eq!(Direction::Negative.is_positive(), false);
    /// ```
    #[inline(always)]
    pub const fn is_positive(&self) -> bool {
        match self {
            Direction::Positive => true,
            Direction::Negative => false,
        }
    }

    /// Returns whether the direction is negative
    /// ```
    /// use crate::splashsurf_lib::topology::Direction;
    /// assert_eq!(Direction::Positive.is_negative(), false);
    /// assert_eq!(Direction::Negative.is_negative(), true);
    /// ```
    #[inline(always)]
    pub const fn is_negative(&self) -> bool {
        !self.is_positive()
    }
}

const ALL_DIRECTIONS: [Direction; 2] = [Direction::Negative, Direction::Positive];

impl CartesianAxis3d {
    /// Returns a reference to an array containing all 3D cartesian axes
    /// ```
    /// use crate::splashsurf_lib::topology::Axis;
    /// assert_eq!(Axis::all_possible()[0], Axis::X);
    /// assert_eq!(Axis::all_possible()[2], Axis::Z);
    /// assert_eq!(Axis::all_possible().len(), 3);
    /// ```
    #[inline(always)]
    pub const fn all_possible() -> &'static [Axis; 3] {
        &ALL_AXES
    }

    /// Converts the cartesian axis into the corresponding 3D dimension index (X=0, Y=1, Z=2)
    /// ```
    /// use crate::splashsurf_lib::topology::Axis;
    /// assert_eq!(Axis::X.dim(), 0);
    /// assert_eq!(Axis::Y.dim(), 1);
    /// assert_eq!(Axis::Z.dim(), 2);
    /// ```
    #[inline(always)]
    pub const fn dim(self) -> usize {
        self as usize
    }

    /// Returns the other two axes that are orthogonal to the current axis
    /// ```
    /// use crate::splashsurf_lib::topology::Axis;
    /// assert_eq!(Axis::X.orthogonal_axes(), [Axis::Y, Axis::Z]);
    /// ```
    #[inline(always)]
    pub const fn orthogonal_axes(&self) -> [Self; 2] {
        ORTHOGONAL_AXES[self.dim()]
    }

    /// Combines this coordinate axis with a direction into a DirectedAxis
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(Axis::X.with_direction(Direction::Positive), DirectedAxis::new(Axis::X, Direction::Positive));
    /// ```
    #[inline(always)]
    pub const fn with_direction(self, direction: Direction) -> DirectedAxis {
        DirectedAxis::new(self, direction)
    }
}

const ALL_AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

const AXES_ORTHOGONAL_TO_X: [Axis; 2] = [Axis::Y, Axis::Z];
const AXES_ORTHOGONAL_TO_Y: [Axis; 2] = [Axis::Z, Axis::X];
const AXES_ORTHOGONAL_TO_Z: [Axis; 2] = [Axis::X, Axis::Y];
const ORTHOGONAL_AXES: [[Axis; 2]; 3] = [
    AXES_ORTHOGONAL_TO_X,
    AXES_ORTHOGONAL_TO_Y,
    AXES_ORTHOGONAL_TO_Z,
];

#[test]
fn test_orthogonal_axes() {
    assert_eq!(
        CartesianAxis3d::X.orthogonal_axes(),
        [CartesianAxis3d::Y, CartesianAxis3d::Z]
    );
    assert_eq!(
        CartesianAxis3d::Y.orthogonal_axes(),
        [CartesianAxis3d::Z, CartesianAxis3d::X]
    );
    assert_eq!(
        CartesianAxis3d::Z.orthogonal_axes(),
        [CartesianAxis3d::X, CartesianAxis3d::Y]
    );
}

impl DirectedAxis {
    /// Returns a reference to an array of all possible directed axes in 3D
    #[inline(always)]
    pub const fn all_possible() -> &'static [DirectedAxis; 6] {
        &ALL_DIRECTED_AXES
    }

    /// Constructs a new directed axis
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(DirectedAxis::new(Axis::X, Direction::Positive),
    ///            Axis::X.with_direction(Direction::Positive));
    /// ```
    #[inline(always)]
    pub const fn new(axis: Axis, direction: Direction) -> Self {
        Self { axis, direction }
    }

    /// Returns a directed axis with the opposite direction
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(DirectedAxis::new(Axis::X, Direction::Positive)
    ///                 .opposite(), DirectedAxis::new(Axis::X, Direction::Negative));
    /// assert_eq!(DirectedAxis::new(Axis::Z, Direction::Negative)
    ///                 .opposite(), DirectedAxis::new(Axis::Z, Direction::Positive));
    /// ```
    #[inline(always)]
    pub const fn opposite(&self) -> Self {
        Self::new(self.axis, self.direction.opposite())
    }

    /// Converts the directed axis into a unique index in the range `(0..=5)`
    #[inline(always)]
    pub const fn to_usize(&self) -> usize {
        self.axis as usize + (self.direction as usize * 3)
    }

    /// Converts an index in the range `(0..=5)` to the corresponding directed axis, panics if the index is out of range
    #[inline(always)]
    pub const fn from_usize(n: usize) -> Self {
        Self::all_possible()[n]
    }

    /// Applies an increment of `1` in the direction of this directed axis to the given index array, returns `None` on overflow
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(DirectedAxis::new(Axis::X, Direction::Positive)
    ///                 .apply_single_step(&[1,2,3]), Some([2,2,3]));
    /// ```
    #[inline(always)]
    pub fn apply_single_step<N: Clone + CheckedAdd<Output = N> + CheckedSub<Output = N> + One>(
        &self,
        index: &[N; 3],
    ) -> Option<[N; 3]> {
        self.checked_apply_step(index, N::one())
    }

    /// Applies the given step in the direction of this directed axis to the given index array, returns `None` on overflow
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(DirectedAxis::new(Axis::X, Direction::Positive)
    ///                 .checked_apply_step(&[1,2,3], 6), Some([7,2,3]));
    /// assert_eq!(DirectedAxis::new(Axis::Z, Direction::Negative)
    ///                 .checked_apply_step(&[1,2,3], 10), Some([1,2,-7]));
    /// ```
    #[inline(always)]
    pub fn checked_apply_step<N: Clone + CheckedAdd<Output = N> + CheckedSub<Output = N>>(
        &self,
        index: &[N; 3],
        step: N,
    ) -> Option<[N; 3]> {
        let mut index = index.clone();
        index[self.axis.dim()] = self
            .direction
            .checked_apply_step(index[self.axis.dim()].clone(), step)?;
        Some(index)
    }

    /// Applies the corresponding component of the step in the direction of this directed axis to the given index array, returns `None` on overflow
    /// ```
    /// use crate::splashsurf_lib::topology::{Axis, DirectedAxis, Direction};
    /// assert_eq!(DirectedAxis::new(Axis::X, Direction::Positive)
    ///                 .checked_apply_step_ijk(&[1,2,3], &[6,8,10]), Some([7,2,3]));
    /// assert_eq!(DirectedAxis::new(Axis::Z, Direction::Negative)
    ///                 .checked_apply_step_ijk(&[1,2,3], &[6,8,10]), Some([1,2,-7]));
    /// ```
    #[inline(always)]
    pub fn checked_apply_step_ijk<N: Clone + CheckedAdd<Output = N> + CheckedSub<Output = N>>(
        &self,
        index: &[N; 3],
        step: &[N; 3],
    ) -> Option<[N; 3]> {
        let mut index = index.clone();
        index[self.axis.dim()] = self.direction.checked_apply_step(
            index[self.axis.dim()].clone(),
            step[self.axis.dim()].clone(),
        )?;
        Some(index)
    }
}

const ALL_DIRECTED_AXES: [DirectedAxis; 6] = [
    DirectedAxis::new(Axis::X, Direction::Negative),
    DirectedAxis::new(Axis::Y, Direction::Negative),
    DirectedAxis::new(Axis::Z, Direction::Negative),
    DirectedAxis::new(Axis::X, Direction::Positive),
    DirectedAxis::new(Axis::Y, Direction::Positive),
    DirectedAxis::new(Axis::Z, Direction::Positive),
];

#[test]
fn test_directed_axis_usize_conversion() {
    for i in 0..6 {
        assert_eq!(DirectedAxis::from_usize(i).to_usize(), i);
    }
}

#[test]
fn test_directed_axis_all_possible_consistency() {
    let all_directed_axes = DirectedAxis::all_possible();
    for (i, ax) in all_directed_axes.iter().enumerate() {
        assert_eq!(ax.to_usize(), i);
        assert_eq!(*ax, DirectedAxis::from_usize(i));
    }
}

impl<T> DirectedAxisArray<T> {
    /// Constructs a new array and fills it with values produced by the given closure
    pub fn new_with<F: FnMut(&DirectedAxis) -> T>(f: F) -> Self {
        let mut f = f;
        Self {
            data: [
                f(&DirectedAxis::all_possible()[0]),
                f(&DirectedAxis::all_possible()[1]),
                f(&DirectedAxis::all_possible()[2]),
                f(&DirectedAxis::all_possible()[3]),
                f(&DirectedAxis::all_possible()[4]),
                f(&DirectedAxis::all_possible()[5]),
            ],
        }
    }

    /// Returns a reference to the value stored for the given axis
    pub fn get(&self, axis: &DirectedAxis) -> &T {
        &self.data[axis.to_usize()]
    }

    /// Returns a mutable reference to the value stored for the given axis
    pub fn get_mut(&mut self, axis: &DirectedAxis) -> &mut T {
        &mut self.data[axis.to_usize()]
    }

    /// Returns an iterator of all unique directed axes and references to the corresponding stored value
    pub fn iter(&self) -> impl Iterator<Item = (&DirectedAxis, &T)> {
        DirectedAxis::all_possible().iter().zip(self.data.iter())
    }

    /// Returns an iterator of all unique directed axes and mutable references to the corresponding stored value
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&DirectedAxis, &mut T)> {
        DirectedAxis::all_possible()
            .iter()
            .zip(self.data.iter_mut())
    }

    /// Returns an iterator over references of all stored values
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}
