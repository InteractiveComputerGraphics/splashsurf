use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{PyResult, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{Aabb3d, nalgebra::Vector3};

macro_rules! create_aabb3d_interface {
    ($name: ident, $type: ident) => {
        /// Aabb3d wrapper
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: Aabb3d<$type>,
        }

        impl $name {
            pub fn new(data: Aabb3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(min: [$type; 3], max: [$type; 3]) -> PyResult<Self> {
                Ok($name::new(Aabb3d::<$type>::new(
                    Vector3::from_column_slice(&min),
                    Vector3::from_column_slice(&max),
                )))
            }

            /// Constructs the smallest AABB fitting around all the given points
            #[staticmethod]
            fn from_points<'py>(points: &Bound<'py, PyArray2<$type>>) -> PyResult<$name> {
                let points: PyReadonlyArray2<$type> = points.extract()?;
                let points = points.as_slice()?;
                let points: &[Vector3<$type>] = bytemuck::cast_slice(points);

                Ok($name::new(Aabb3d::from_points(points)))
            }

            /// Constructs the smallest AABB fitting around all the given points, parallel version
            #[staticmethod]
            fn par_from_points<'py>(points: &Bound<'py, PyArray2<$type>>) -> PyResult<$name> {
                let points: PyReadonlyArray2<$type> = points.extract()?;
                let points = points.as_slice()?;
                let points: &[Vector3<$type>] = bytemuck::cast_slice(points);

                Ok($name::new(Aabb3d::par_from_points(points)))
            }

            /// Constructs a degenerate AABB with min and max set to zero
            #[staticmethod]
            fn zeros() -> $name {
                $name::new(Aabb3d::zeros())
            }

            /// Constructs a degenerate AABB with zero extents centered at the given point
            #[staticmethod]
            fn from_point(point: [$type; 3]) -> Self {
                $name::new(Aabb3d::from_point(Vector3::from_column_slice(&point)))
            }

            /// Returns the min coordinate of the bounding box
            fn min<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$type>> {
                let min: &[$type] = self.inner.min().as_slice();
                PyArray::from_slice(py, min)
            }

            /// Returns the max coordinate of the bounding box
            fn max<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$type>> {
                let max: &[$type] = self.inner.max().as_slice();
                PyArray::from_slice(py, max)
            }

            /// Returns whether the AABB is consistent, i.e. `aabb.min()[i] <= aabb.max()[i]` for all `i`
            fn is_consistent(&self) -> bool {
                self.inner.is_consistent()
            }

            /// Returns whether the AABB is degenerate in any dimension, i.e. `aabb.min()[i] == aabb.max()[i]` for any `i`
            fn is_degenerate(&self) -> bool {
                self.inner.is_degenerate()
            }

            /// Returns the extents of the bounding box (vector connecting min and max point of the box)
            fn extents<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$type>> {
                let extents = self.inner.extents();
                PyArray::from_slice(py, extents.as_slice())
            }

            /// Returns the smallest scalar extent of the AABB over all of its dimensions
            fn min_extent(&self) -> $type {
                self.inner.min_extent()
            }

            /// Returns the largest scalar extent of the AABB over all of its dimensions
            fn max_extent(&self) -> $type {
                self.inner.max_extent()
            }

            /// Returns the geometric centroid of the AABB (mean of the corner points)
            fn centroid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$type>> {
                let centroid = self.inner.centroid();
                PyArray::from_slice(py, centroid.as_slice())
            }

            /// Checks if the given AABB is inside of the AABB, the AABB is considered to be half-open to its max coordinate
            fn contains_aabb(&self, other: &$name) -> bool {
                self.inner.contains_aabb(&other.inner)
            }

            /// Checks if the given point is inside of the AABB, the AABB is considered to be half-open to its max coordinate
            fn contains_point(&self, point: [$type; 3]) -> bool {
                self.inner
                    .contains_point(&Vector3::from_column_slice(&point))
            }

            /// Translates the AABB by the given vector
            fn translate(&mut self, vector: [$type; 3]) {
                self.inner.translate(&Vector3::from_column_slice(&vector));
            }

            /// Translates the AABB to center it at the coordinate origin (moves the centroid to the coordinate origin)
            fn center_at_origin(&mut self) {
                self.inner.center_at_origin();
            }

            /// Multiplies a uniform, local scaling to the AABB (i.e. multiplying its extents as if it was centered at the origin)
            fn scale_uniformly(&mut self, scaling: $type) {
                self.inner.scale_uniformly(scaling);
            }

            /// Enlarges this AABB to the smallest AABB enclosing both itself and another AABB
            fn join(&mut self, other: &$name) {
                self.inner.join(&other.inner);
            }

            /// Enlarges this AABB to the smallest AABB enclosing both itself and another point
            fn join_with_point(&mut self, point: [$type; 3]) {
                self.inner
                    .join_with_point(&Vector3::from_column_slice(&point));
            }

            /// Grows this AABB uniformly in all directions by the given scalar margin (i.e. adding the margin to min/max extents)
            fn grow_uniformly(&mut self, margin: $type) {
                self.inner.grow_uniformly(margin);
            }

            /// Returns the smallest cubical AABB with the same center that encloses this AABB
            fn enclosing_cube(&self) -> $name {
                $name::new(self.inner.enclosing_cube())
            }
        }
    };
}

create_aabb3d_interface!(Aabb3dF64, f64);
create_aabb3d_interface!(Aabb3dF32, f32);
