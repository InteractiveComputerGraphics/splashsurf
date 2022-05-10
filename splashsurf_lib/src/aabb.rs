//! Axis-aligned bounding boxes

use std::fmt;
use std::fmt::Debug;

use nalgebra::SVector;
use rayon::prelude::*;

use crate::{Real, ThreadSafe};

/// Type representing an axis aligned bounding box in arbitrary dimensions
#[derive(Clone, Eq, PartialEq)]
pub struct AxisAlignedBoundingBox<R: Real, const D: usize> {
    min: SVector<R, D>,
    max: SVector<R, D>,
}

/// Convenience type alias for an AABB in two dimensions
pub type AxisAlignedBoundingBox2d<R> = AxisAlignedBoundingBox<R, 2>;
/// Convenience type alias for an AABB in three dimensions
pub type AxisAlignedBoundingBox3d<R> = AxisAlignedBoundingBox<R, 3>;

impl<R, const D: usize> AxisAlignedBoundingBox<R, D>
where
    R: Real,
    SVector<R, D>: ThreadSafe,
{
    /// Constructs the smallest AABB fitting around all the given points, parallel version
    pub fn par_from_points(points: &[SVector<R, D>]) -> Self {
        if points.is_empty() {
            Self::zeros()
        } else if points.len() == 1 {
            Self::from_point(points[0].clone())
        } else {
            let initial_aabb = Self::from_point(points[0].clone());
            points[1..]
                .par_iter()
                .fold(
                    || initial_aabb.clone(),
                    |mut aabb, next_point| {
                        aabb.join_with_point(next_point);
                        aabb
                    },
                )
                .reduce(
                    || initial_aabb.clone(),
                    |mut final_aabb, aabb| {
                        final_aabb.join(&aabb);
                        final_aabb
                    },
                )
        }
    }
}

impl<R, const D: usize> AxisAlignedBoundingBox<R, D>
where
    R: Real,
{
    /// Constructs a degenerate AABB with min and max set to zero
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_point(SVector::zeros())
    }

    /// Constructs an AABB with the given min and max bounding points
    #[inline(always)]
    pub fn new(min: SVector<R, D>, max: SVector<R, D>) -> Self {
        Self { min, max }
    }

    /// Constructs a degenerate AABB with zero extents centered at the given point
    #[inline(always)]
    pub fn from_point(point: SVector<R, D>) -> Self {
        Self {
            min: point.clone(),
            max: point,
        }
    }

    /// Constructs the smallest AABB fitting around all the given points
    pub fn from_points(points: &[SVector<R, D>]) -> Self {
        let mut point_iter = points.iter();
        if let Some(first_point) = point_iter.next().cloned() {
            let mut aabb = Self::from_point(first_point);
            for next_point in point_iter {
                aabb.join_with_point(next_point)
            }
            aabb
        } else {
            Self::zeros()
        }
    }

    /// Tries to convert the AABB from one real type to another real type, returns None if conversion fails
    pub fn try_convert<T>(&self) -> Option<AxisAlignedBoundingBox<T, D>>
    where
        T: Real,
    {
        Some(AxisAlignedBoundingBox::new(
            T::try_convert_vec_from(&self.min)?,
            T::try_convert_vec_from(&self.max)?,
        ))
    }

    /// Returns the min coordinate of the bounding box
    #[inline(always)]
    pub fn min(&self) -> &SVector<R, D> {
        &self.min
    }

    /// Returns the max coordinate of the bounding box
    #[inline(always)]
    pub fn max(&self) -> &SVector<R, D> {
        &self.max
    }

    /// Returns whether the AABB is consistent, i.e. `aabb.min()[i] <= aabb.max()[i]` for all `i`
    pub fn is_consistent(&self) -> bool {
        for i in 0..D {
            if !(self.min[i] <= self.max[i]) {
                return false;
            }
        }
        return true;
    }

    /// Returns whether the AABB is degenerate in any dimension, i.e. `aabb.min()[i] == aabb.max()[i]` for any `i`
    pub fn is_degenerate(&self) -> bool {
        for i in 0..D {
            if self.min[i] == self.max[i] {
                return true;
            }
        }
        return false;
    }

    /// Returns the extents of the bounding box (vector connecting min and max point of the box)
    #[inline(always)]
    pub fn extents(&self) -> SVector<R, D> {
        &self.max - &self.min
    }

    /// Returns the smallest scalar extent of the AABB over all of its dimensions
    #[inline(always)]
    pub fn min_extent(&self) -> R {
        let extents = self.extents();
        // Use imin indirectly, because min is broken in nalgebra
        extents[extents.imin()]
    }

    /// Returns the largest scalar extent of the AABB over all of its dimensions
    #[inline(always)]
    pub fn max_extent(&self) -> R {
        let extents = self.extents();
        // Use imax indirectly, because max is broken in nalgebra
        extents[extents.imax()]
    }

    /// Returns the geometric centroid of the AABB (mean of the corner points)
    pub fn centroid(&self) -> SVector<R, D> {
        &self.min + (self.extents() / (R::one() + R::one()))
    }

    /// Checks if the given AABB is inside of the AABB, the AABB is considered to be half-open to its max coordinate
    pub fn contains_aabb(&self, other: &Self) -> bool {
        for i in 0..D {
            if other.min[i] < self.min[i] || other.min[i] >= self.max[i] {
                return false;
            }

            if other.max[i] < self.min[i] || other.max[i] >= self.max[i] {
                return false;
            }
        }

        return true;
    }

    /// Checks if the given point is inside of the AABB, the AABB is considered to be half-open to its max coordinate
    pub fn contains_point(&self, point: &SVector<R, D>) -> bool {
        for i in 0..D {
            if point[i] < self.min[i] || point[i] >= self.max[i] {
                return false;
            }
        }

        return true;
    }

    /// Translates the AABB by the given vector
    pub fn translate(&mut self, vector: &SVector<R, D>) {
        self.min += vector;
        self.max += vector;
    }

    /// Translates the AABB to center it at the coordinate origin (moves the centroid to the coordinate origin)
    pub fn center_at_origin(&mut self) {
        self.translate(&(self.centroid() * R::one().neg()));
    }

    /// Multiplies a uniform, local scaling to the AABB (i.e. multiplying its extents as if it was centered at the origin)
    pub fn scale_uniformly(&mut self, scaling: R) {
        let center = self.centroid();
        self.translate(&(&center * R::one().neg()));
        self.min *= scaling;
        self.max *= scaling;
        self.translate(&center);
    }

    /// Enlarges this AABB to the smallest AABB enclosing both itself and another AABB
    pub fn join(&mut self, other: &Self) {
        for i in 0..D {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Enlarges this AABB to the smallest AABB enclosing both itself and another point
    pub fn join_with_point(&mut self, point: &SVector<R, D>) {
        for i in 0..D {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    /// Grows this AABB uniformly in all directions by the given scalar margin (i.e. adding the margin to min/max extents)
    pub fn grow_uniformly(&mut self, margin: R) {
        self.min = &self.min - &SVector::repeat(margin);
        self.max = &self.max + &SVector::repeat(margin);
    }

    /// Returns the smallest cubical AABB with the same center that encloses this AABB
    pub fn enclosing_cube(&self) -> Self {
        let center = self.centroid();
        let half_max_extent = self.max_extent() / (R::one() + R::one());

        let mut cube = Self::new(
            SVector::repeat(half_max_extent.neg()),
            SVector::repeat(half_max_extent),
        );
        cube.translate(&center);
        cube
    }
}

impl<R, const D: usize> Debug for AxisAlignedBoundingBox<R, D>
where
    R: Real,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AxisAlignedBoundingBox {{ min: [{:.7}, {:.7}, {:.7}], max: [{:.7}, {:.7}, {:.7}] }}",
            self.min[0], self.min[1], self.min[2], self.max[0], self.max[1], self.max[2]
        )
    }
}

#[test]
fn test_aabb_contains_point() {
    use crate::nalgebra::Vector3;
    let aabb = AxisAlignedBoundingBox3d::<f64>::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    );

    assert!(aabb.contains_point(&Vector3::new(0.5, 0.5, 0.5)));
    assert!(aabb.contains_point(&Vector3::new(0.0, 0.5, 0.5)));
    assert!(aabb.contains_point(&Vector3::new(0.5, 0.0, 0.5)));
    assert!(aabb.contains_point(&Vector3::new(0.5, 0.5, 0.0)));
    assert!(aabb.contains_point(&Vector3::new(0.0, 0.0, 0.0)));
    assert!(!aabb.contains_point(&Vector3::new(1.0, 0.0, 0.0)));
    assert!(!aabb.contains_point(&Vector3::new(0.0, 1.0, 0.0)));
    assert!(!aabb.contains_point(&Vector3::new(0.0, 0.0, 1.0)));
    assert!(!aabb.contains_point(&Vector3::new(1.0, 1.0, 1.0)));
}
