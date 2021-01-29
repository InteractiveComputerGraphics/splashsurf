//! Axis-aligned bounding boxes

use std::fmt;
use std::fmt::Debug;

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, VectorN, U2, U3};
use rayon::prelude::*;

use crate::{Real, ThreadSafe};

/// Type representing an axis aligned bounding box in arbitrary dimensions
#[derive(Clone, Eq, PartialEq)]
pub struct AxisAlignedBoundingBox<R: Real, D: DimName>
where
    DefaultAllocator: Allocator<R, D>,
{
    min: VectorN<R, D>,
    max: VectorN<R, D>,
}

/// Convenience type alias for an AABB in two dimensions
pub type AxisAlignedBoundingBox2d<R> = AxisAlignedBoundingBox<R, U2>;
/// Convenience type alias for an AABB in three dimensions
pub type AxisAlignedBoundingBox3d<R> = AxisAlignedBoundingBox<R, U3>;

impl<R, D> AxisAlignedBoundingBox<R, D>
where
    R: Real,
    D: DimName,
    DefaultAllocator: Allocator<R, D>,
    VectorN<R, D>: ThreadSafe,
{
    /// Constructs the smallest AABB fitting around all the given points, parallel version
    pub fn from_points_par(points: &[VectorN<R, D>]) -> Self {
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

impl<R, D> AxisAlignedBoundingBox<R, D>
where
    R: Real,
    D: DimName,
    DefaultAllocator: Allocator<R, D>,
{
    /// Constructs a degenerate AABB with min and max set to zero
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_point(VectorN::zeros())
    }

    /// Constructs an AABB with the given min and max bounding points
    #[inline(always)]
    pub fn new(min: VectorN<R, D>, max: VectorN<R, D>) -> Self {
        Self { min, max }
    }

    /// Constructs a degenerate AABB with zero extents centered at the given point
    #[inline(always)]
    pub fn from_point(point: VectorN<R, D>) -> Self {
        Self {
            min: point.clone(),
            max: point,
        }
    }

    /// Constructs the smallest AABB fitting around all the given points
    pub fn from_points(points: &[VectorN<R, D>]) -> Self {
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
        DefaultAllocator: Allocator<T, D>,
    {
        Some(AxisAlignedBoundingBox::new(
            T::try_convert_vec_from(&self.min)?,
            T::try_convert_vec_from(&self.max)?,
        ))
    }

    /// Returns the min coordinate of the bounding box
    #[inline(always)]
    pub fn min(&self) -> &VectorN<R, D> {
        &self.min
    }

    /// Returns the max coordinate of the bounding box
    #[inline(always)]
    pub fn max(&self) -> &VectorN<R, D> {
        &self.max
    }

    /// Returns whether the AABB is consistent, i.e. `aabb.min()[i] <= aabb.max()[i]` for all `i`
    pub fn is_consistent(&self) -> bool {
        for i in 0..D::dim() {
            if !(self.min[i] <= self.max[i]) {
                return false;
            }
        }
        return true;
    }

    /// Returns whether the AABB is degenerate in any dimension, i.e. `aabb.min()[i] == aabb.max()[i]` for any `i`
    pub fn is_degenerate(&self) -> bool {
        for i in 0..D::dim() {
            if self.min[i] == self.max[i] {
                return true;
            }
        }
        return false;
    }

    /// Returns the extents of the bounding box (vector connecting min and max point of the box)
    #[inline(always)]
    pub fn extents(&self) -> VectorN<R, D> {
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
    pub fn centroid(&self) -> VectorN<R, D> {
        &self.min + (self.extents() / (R::one() + R::one()))
    }

    /// Checks if the given point is inside or on the boundary of the AABB
    pub fn contains_point(&self, point: &VectorN<R, D>) -> bool {
        for i in 0..D::dim() {
            if point[i] <= self.min[i] || point[i] >= self.max[i] {
                return false;
            }
        }

        return true;
    }

    /// Translates the AABB by the given vector
    pub fn translate(&mut self, vector: &VectorN<R, D>) {
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
        for i in 0..D::dim() {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Enlarges this AABB to the smallest AABB enclosing both itself and another point
    pub fn join_with_point(&mut self, point: &VectorN<R, D>) {
        for i in 0..D::dim() {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    /// Grows this AABB uniformly in all directions by the given scalar margin (i.e. adding the margin to min/max extents)
    pub fn grow_uniformly(&mut self, margin: R) {
        self.min = &self.min - &VectorN::repeat(margin);
        self.max = &self.max + &VectorN::repeat(margin);
    }

    /// Returns the smallest cubical AABB with the same center that encloses this AABB
    pub fn enclosing_cube(&self) -> Self {
        let center = self.centroid();
        let half_max_extent = self.max_extent() / (R::one() + R::one());

        let mut cube = Self::new(
            VectorN::repeat(half_max_extent.neg()),
            VectorN::repeat(half_max_extent),
        );
        cube.translate(&center);
        cube
    }
}

impl<R, D> Debug for AxisAlignedBoundingBox<R, D>
where
    R: Real,
    D: DimName,
    DefaultAllocator: Allocator<R, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AxisAlignedBoundingBox {{ min: [{:.7}, {:.7}, {:.7}], max: [{:.7}, {:.7}, {:.7}] }}",
            self.min[0], self.min[1], self.min[2], self.max[0], self.max[1], self.max[2]
        )
    }
}
