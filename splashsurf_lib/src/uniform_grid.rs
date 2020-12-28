use std::iter::Iterator;
use std::ops::{Add, Sub};

use itertools::iproduct;
use nalgebra::Vector3;
use num::{CheckedAdd, CheckedSub, One};
use thiserror::Error as ThisError;

use crate::{AxisAlignedBoundingBox3d, Index, Real};

// TODO: Reduce mess with all array and scalar indexing functions

/*
 * Cube description:
 *         7 ________ 6           _____6__
 *         /|       /|         7/|       /|
 *       /  |     /  |        /  |     /5 |
 *   4 /_______ /    |      /__4____ /    10
 *    |     |  |5    |     |    11  |     |
 *    |    3|__|_____|2    |     |__|__2__|
 *    |    /   |    /      8   3/   9    /
 *    |  /     |  /        |  /     |  /1
 *    |/_______|/          |/___0___|/
 *   0          1
 */

/// An index triplet of a point or vertex in a 3D cartesian grid (index along each axis)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct PointIndex<I: Index> {
    index: [I; 3],
}

/// An index triplet of a cell in a 3D cartesian grid (index along each axis)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct CellIndex<I: Index> {
    index: [I; 3],
}

/// Abbreviated type alias for cartesian coordinate axes in 3D
pub type Axis = CartesianAxis3d;

/// Enum for the cartesian coordinate axes in 3D
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CartesianAxis3d {
    X = 0,
    Y = 1,
    Z = 2,
}

/// Indicates a direction on a number line or coordinate axis
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Direction {
    Negative = 0,
    Positive = 1,
}

/// Identifies a direction along a cartesian axis
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct DirectedAxis {
    axis: Axis,
    direction: Direction,
}

/// Full neighborhood information of a point (denoted as origin point of the neighborhood)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Neighborhood<'a, I: Index> {
    origin: &'a PointIndex<I>,
    neighbors: [Option<PointIndex<I>>; 6],
}

/// An edge that connects a primary point and a neighbor point
pub struct NeighborEdge<'a, 'b: 'a, I: Index> {
    /// Neighborhood of the origin point of this edge
    neighborhood: &'a Neighborhood<'b, I>,
    /// Valid grid index triplet of a neighboring point
    neighbor: &'a PointIndex<I>,
    /// Edge connectivity from the origin point to the neighboring point
    connectivity: DirectedAxis,
}

/// Abbreviated type alias for a uniform cartesian cube grid in 3D
///
/// The underlying type is not public because its methods are only useful internally.
pub type UniformGrid<I, R> = UniformCartesianCubeGrid3d<I, R>;

/// Helper type for connectivity information on a cartesian grid in 3D based on uniform cubes
///
/// This type represents a virtual cartesian grid in 3D based on uniform cubes.
/// It provides helper functions to access connectivity of points (vertices), edges and cells on
/// the virtual grid.
///
/// **Grid construction** The origin of the grid is placed on the min coordinates of its AABB.
/// Then, the AABB is filled with uniformly sized cubes. The final size of the grid may be larger than
/// the AABB, as the last layer of cubes that may not fully fit into the upper extents of the AABB
/// is still considered part of the grid.
///
/// **Connectivity information** The grid then provides helper functions e.g. to find valid neighbors
/// of points, edges and cells on the grid. These entities can either be indexed using index triplets `(i,j,k)` along
/// the cartesian axes or using a flattened index `i*n_x + j*n_y + k*n_z` that is used e.g. for lookup
/// in the [DensityMap](crate::DensityMap). This struct provides helper functions to convert between these representations.
/// For some functions, strongly typed [PointIndex] and [CellIndex] indices are used, these can be
/// obtained using the [get_point](UniformCartesianCubeGrid3d::get_point) and [get_cell](UniformCartesianCubeGrid3d::get_cell) functions respectively.
/// These functions check if the specified indices are in the valid index range of the grid (as computed during construction based
/// on the extents of the grid).
#[derive(Clone, PartialEq, Debug)]
pub struct UniformCartesianCubeGrid3d<I: Index, R: Real> {
    /// AABB of the grid. Note that currently, the grid may extend beyond the max coordinate of the AABB.
    aabb: AxisAlignedBoundingBox3d<R>,
    /// The edge length of the cubes in the grid
    cell_size: R,

    /// The number of vertices of the grid in each cartesian direction
    n_points_per_dim: [I; 3],
    /// The number of cells of the grid in each cartesian direction
    n_cells_per_dim: [I; 3],
}

/// Error type for the construction of a [UniformGrid]
#[rustfmt::skip]
#[derive(Copy, Clone, Eq, PartialEq, Debug, ThisError)]
pub enum GridConstructionError<I: Index, R: Real> {
    /// The cell size is invalid, it has to be larger than zero
    #[error("invalid cell size `{0}` supplied, cell size has to be larger than zero")]
    InvalidCellSize(R),
    /// The AABB is degenerate, every dimension of the AABB has to have non-zero extents
    #[error("degenerate AABB supplied, every dimension of the AABB has to have non-zero extents")]
    DegenerateAabb,
    /// The AABB is inconsistent, every dimension of the AABB has to have an extent larger than zero
    #[error("inconsistent AABB supplied, every dimension of the AABB has to have an extent larger than zero")]
    InconsistentAabb,
    /// The index type is too small to index the number of cells in each dimension of the domain
    #[error("index type is too small to index number of cells per dimension of the domain (max index: {})", I::max_value())]
    IndexTypeTooSmallCellsPerDim,
    /// The index type is too small to index the number of points in each dimension of the domain
    #[error("index type is too small to index number of points per dimension of the domain (max index: {})", I::max_value())]
    IndexTypeTooSmallPointsPerDim,
    /// The index type is too small to index the total number of points in the whole domain (nx * ny * nz)
    #[error("index type is too small to index the total number of points in the whole domain ({0}x{1}x{2}, max index: {})", I::max_value())]
    IndexTypeTooSmallTotalPoints(I, I, I),
    /// The real type is too small to store the coordinates of all possible points in the domain
    #[error("real type is too small to store the coordinates of all points in the domain (max value: {})", R::max_value())]
    RealTypeTooSmallDomainSize,
}

impl<I: Index, R: Real> UniformCartesianCubeGrid3d<I, R> {
    /// Construct a new grid enclosing the given AABB
    ///
    /// The grid will at least contain the AABB but may be larger depending on the cell size.
    pub fn from_aabb(
        aabb: &AxisAlignedBoundingBox3d<R>,
        cell_size: R,
    ) -> Result<Self, GridConstructionError<I, R>> {
        if !(cell_size > R::zero()) {
            return Err(GridConstructionError::InvalidCellSize(cell_size));
        }

        if aabb.is_degenerate() {
            return Err(GridConstructionError::DegenerateAabb);
        }

        if !aabb.is_consistent() {
            return Err(GridConstructionError::InconsistentAabb);
        }

        let n_cells_real = aabb.extents() / cell_size;
        let n_cells_per_dim = Self::checked_n_cells_per_dim(&n_cells_real)
            .ok_or(GridConstructionError::IndexTypeTooSmallCellsPerDim)?;

        Self::new(aabb.min(), &n_cells_per_dim, cell_size)
    }

    /// Constructs a new grid extending in positive cartesian axes direction from the min coordinate by the specified number of cubes of the given size
    pub fn new(
        min: &Vector3<R>,
        n_cells_per_dim: &[I; 3],
        cell_size: R,
    ) -> Result<Self, GridConstructionError<I, R>> {
        let n_cells_per_dim = n_cells_per_dim.clone();
        let n_points_per_dim = Self::checked_n_points_per_dim(&n_cells_per_dim)
            .ok_or(GridConstructionError::IndexTypeTooSmallPointsPerDim)?;

        // Compute the bounding box max value
        let aabb = Self::checked_aabb(min, &n_cells_per_dim, cell_size)
            .ok_or(GridConstructionError::RealTypeTooSmallDomainSize)?;

        // Check that the total number of points can be represented by the index type
        let _ = Self::checked_num_points(&n_points_per_dim).ok_or(
            GridConstructionError::IndexTypeTooSmallTotalPoints(
                n_points_per_dim[0],
                n_points_per_dim[1],
                n_points_per_dim[2],
            ),
        )?;

        Ok(Self {
            aabb,
            cell_size,
            n_points_per_dim,
            n_cells_per_dim,
        })
    }

    /// Create a new zeroed grid
    pub(crate) fn new_zero() -> Self {
        Self {
            aabb: AxisAlignedBoundingBox3d::new(Vector3::zeros(), Vector3::zeros()),
            cell_size: R::zero(),
            n_points_per_dim: [I::zero(); 3],
            n_cells_per_dim: [I::zero(); 3],
        }
    }

    /// Returns the bounding box of the grid
    #[inline(always)]
    pub fn aabb(&self) -> &AxisAlignedBoundingBox3d<R> {
        &self.aabb
    }

    /// Returns the cell size used by the grid
    #[inline(always)]
    pub fn cell_size(&self) -> R {
        self.cell_size
    }

    /// Returns the number of grid points per dimension of the grid
    #[inline(always)]
    pub fn points_per_dim(&self) -> &[I; 3] {
        &self.n_points_per_dim
    }

    /// Returns the number of grid cells per dimension of the grid
    #[inline(always)]
    pub fn cells_per_dim(&self) -> &[I; 3] {
        &self.n_cells_per_dim
    }

    /// Converts a point index triplet into a strongly typed index, returns `None` if the corresponding point is not part of the grid
    #[inline(always)]
    pub fn get_point(&self, ijk: &[I; 3]) -> Option<PointIndex<I>> {
        if self.point_exists(ijk) {
            Some(PointIndex::from_ijk(ijk.clone()))
        } else {
            None
        }
    }

    /// Converts a cell index triplet into a strongly typed index, returns `None` if the corresponding cell is not part of the grid
    #[inline(always)]
    pub fn get_cell(&self, ijk: &[I; 3]) -> Option<CellIndex<I>> {
        if self.cell_exists(ijk) {
            Some(CellIndex::from_ijk(ijk.clone()))
        } else {
            None
        }
    }

    /// Returns whether a point exists in the grid
    #[inline(always)]
    pub fn point_exists(&self, point_ijk: &[I; 3]) -> bool {
        (point_ijk[0] < self.n_points_per_dim[0]
            && point_ijk[1] < self.n_points_per_dim[1]
            && point_ijk[2] < self.n_points_per_dim[2])
            && (point_ijk[0] >= I::zero() && point_ijk[1] >= I::zero() && point_ijk[2] >= I::zero())
    }

    /// Returns whether a cell exists that has the given point index as its origin point (i.e. lower corner)
    #[inline(always)]
    pub fn cell_exists(&self, cell_min_point_ijk: &[I; 3]) -> bool {
        (cell_min_point_ijk[0] < self.n_cells_per_dim[0]
            && cell_min_point_ijk[1] < self.n_cells_per_dim[1]
            && cell_min_point_ijk[2] < self.n_cells_per_dim[2])
            && (cell_min_point_ijk[0] >= I::zero()
                && cell_min_point_ijk[1] >= I::zero()
                && cell_min_point_ijk[2] >= I::zero())
    }

    /// Flattens the grid point index triplet to a single index
    #[inline(always)]
    pub fn flatten_point_indices(&self, i: I, j: I, k: I) -> I {
        let np = &self.n_points_per_dim;
        i * np[1] * np[2] + j * np[2] + k
    }

    /// Flattens the grid point index triplet array to a single index
    #[inline(always)]
    pub fn flatten_point_index_array(&self, ijk: &[I; 3]) -> I {
        self.flatten_point_indices(ijk[0], ijk[1], ijk[2])
    }

    /// Flattens the grid point to a single index
    #[inline(always)]
    pub fn flatten_point_index(&self, point: &PointIndex<I>) -> I {
        self.flatten_point_index_array(point.index())
    }

    /// Flattens the grid cell index triplet to a single index
    #[inline(always)]
    pub fn flatten_cell_indices(&self, i: I, j: I, k: I) -> I {
        let nc = &self.n_cells_per_dim;
        i * nc[1] * nc[2] + j * nc[2] + k
    }

    /// Flattens the grid cell index triplet array to a single index
    #[inline(always)]
    pub fn flatten_cell_index_array(&self, ijk: &[I; 3]) -> I {
        self.flatten_cell_indices(ijk[0], ijk[1], ijk[2])
    }

    /// Flattens the grid point to a single index
    #[inline(always)]
    pub fn flatten_cell_index(&self, cell: &CellIndex<I>) -> I {
        self.flatten_cell_index_array(cell.index())
    }

    /// Converts a flat point index value back to a point index triplet, does not check if the point is part of the grid
    #[inline(always)]
    fn unflatten_point_index(&self, point_index: I) -> [I; 3] {
        let np = &self.n_points_per_dim;

        let i = point_index / (np[1] * np[2]);
        let j = (point_index - i * np[1] * np[2]) / np[2];
        let k = point_index - i * np[1] * np[2] - j * np[2];

        [i, j, k]
    }

    /// Converts a flat point index value back to a strongly typed point index, returns `None` if the point index is not part of the grid
    #[inline(always)]
    pub fn try_unflatten_point_index(&self, point_index: I) -> Option<PointIndex<I>> {
        let point_ijk = self.unflatten_point_index(point_index);
        self.get_point(&point_ijk)
    }

    /// Converts a flat cell index value back to a cell index triplet, does not check if the cell is part of the grid
    #[inline(always)]
    fn unflatten_cell_index(&self, cell_index: I) -> [I; 3] {
        let nc = &self.n_cells_per_dim;

        let i = cell_index / (nc[1] * nc[2]);
        let j = (cell_index - i * nc[1] * nc[2]) / nc[2];
        let k = cell_index - i * nc[1] * nc[2] - j * nc[2];

        [i, j, k]
    }

    /// Converts a flat cell index value back to a strongly typed cell index, returns `None` if the cell index is not part of the grid
    #[inline(always)]
    pub fn try_unflatten_cell_index(&self, cell_index: I) -> Option<CellIndex<I>> {
        let cell_ijk = self.unflatten_cell_index(cell_index);
        self.get_cell(&cell_ijk)
    }

    /// Returns the real-valued coordinates of a grid point in space
    #[inline(always)]
    pub fn point_coordinates_indices(&self, i: I, j: I, k: I) -> Vector3<R> {
        self.aabb.min()
            + Vector3::new(
                i.to_real_unchecked::<R>() * self.cell_size,
                j.to_real_unchecked::<R>() * self.cell_size,
                k.to_real_unchecked::<R>() * self.cell_size,
            )
    }

    /// Returns the real-valued coordinates of a grid point in space
    #[inline(always)]
    pub fn point_coordinates_array(&self, ijk: &[I; 3]) -> Vector3<R> {
        self.point_coordinates_indices(ijk[0], ijk[1], ijk[2])
    }

    /// Returns the real-valued coordinates of a grid point in space
    #[inline(always)]
    pub fn point_coordinates(&self, point: &PointIndex<I>) -> Vector3<R> {
        self.point_coordinates_array(point.index())
    }

    /// Returns the grid cell index triplet of the cell enclosing a point with the given coordinates in space
    #[inline(always)]
    pub fn enclosing_cell(&self, coord: &Vector3<R>) -> [I; 3] {
        let normalized_coord = (coord - self.aabb.min()) / self.cell_size;
        [
            normalized_coord[0].floor().to_index_unchecked(),
            normalized_coord[1].floor().to_index_unchecked(),
            normalized_coord[2].floor().to_index_unchecked(),
        ]
    }

    /// If part of the grid, returns the neighbor of a point following the given directed axis along the grid
    #[inline(always)]
    pub fn get_point_neighbor(
        &self,
        point: &PointIndex<I>,
        direction: DirectedAxis,
    ) -> Option<PointIndex<I>> {
        let point_ijk = point.index();
        let DirectedAxis { axis, direction } = &direction;
        let dim = axis.dim();

        if ((point_ijk[dim] == I::zero()) && direction.is_negative())
            || ((point_ijk[dim] == self.n_points_per_dim[dim] - I::one())
                && direction.is_positive())
        {
            return None;
        }

        let mut neighbor_ijk = point_ijk.clone();
        neighbor_ijk[dim] = direction.apply_step(neighbor_ijk[dim], I::one());
        Some(PointIndex::from_ijk(neighbor_ijk))
    }

    #[inline(always)]
    fn get_point_neighbor_unchecked(&self, point_ijk: &[I; 3], direction: DirectedAxis) -> [I; 3] {
        let DirectedAxis { axis, direction } = direction;
        let dim = axis.dim();

        let mut neighbor_ijk = point_ijk.clone();
        neighbor_ijk[dim] = direction.apply_step(neighbor_ijk[dim], I::one());
        neighbor_ijk
    }

    /// Returns full neighborhood information of a point on the grid
    pub fn get_point_neighborhood<'a>(&self, point: &'a PointIndex<I>) -> Neighborhood<'a, I> {
        let neighbors = [
            self.get_point_neighbor(point, DirectedAxis::from_usize(0)),
            self.get_point_neighbor(point, DirectedAxis::from_usize(1)),
            self.get_point_neighbor(point, DirectedAxis::from_usize(2)),
            self.get_point_neighbor(point, DirectedAxis::from_usize(3)),
            self.get_point_neighbor(point, DirectedAxis::from_usize(4)),
            self.get_point_neighbor(point, DirectedAxis::from_usize(5)),
        ];

        Neighborhood {
            origin: point,
            neighbors,
        }
    }

    /// Returns an array of all cells that may be adjacent to the specified edge
    pub fn cells_adjacent_to_edge<'a, 'b>(
        &self,
        edge: &NeighborEdge<'a, 'b, I>,
    ) -> [Option<CellIndex<I>>; 4] {
        // Each cell has the same index as the point in its lower corner, its 'origin point'.
        // To get all cells adjacent to the given edge, all corresponding origin points have to be found
        //
        //  -> the first origin point is given by the point of the edge with the lower index
        //  -> two other points can be obtained by starting from the first point, and respectively
        //     making a step in negative coordinate direction along each axis that is orthogonal to the edge
        //  -> the last origin point is obtained by combining the two separate steps from the two
        //     previous points (i.e. making a diagonal step)

        // Obtain the point with on the negative end of the edge (first origin point)
        let (edge_start_point, _) = edge.ascending_point_order();
        // Obtain the axes that are orthogonal to the edge (used to make steps to the other points)
        let orthogonal_axes = edge.connectivity.axis.orthogonal_axes();

        // Obtain directed axes towards possible origin points of adjacent cells
        let step_dir1 = orthogonal_axes[0].with_direction(Direction::Negative);
        let step_dir3 = orthogonal_axes[1].with_direction(Direction::Negative);

        // Try to obtain all points that might be the origin or lower corner of a cell
        // Some of them (p1, p3 or both) might not exist if the edge is at the boundary of the grid
        let p0 = Some(edge_start_point.clone());
        let p1 = self.get_point_neighbor(edge_start_point, step_dir1);
        let p3 = self.get_point_neighbor(edge_start_point, step_dir3);
        // Get the last origin point by combining both steps
        let p2 = match (&p1, &p3) {
            // Only if both neighbors exists, the diagonal neighbor also exists
            (Some(p1), Some(_)) => Some(PointIndex::from_ijk(
                self.get_point_neighbor_unchecked(p1.index(), step_dir3),
            )),
            _ => None,
        };

        // The cell indices have to be checked again, as a neighboring cell might not be part of the grid
        [
            p0.filter(|p| self.cell_exists(p.index()))
                .map(CellIndex::from_point),
            p1.filter(|p| self.cell_exists(p.index()))
                .map(CellIndex::from_point),
            p2.filter(|p| self.cell_exists(p.index()))
                .map(CellIndex::from_point),
            p3.filter(|p| self.cell_exists(p.index()))
                .map(CellIndex::from_point),
        ]
    }

    /// Returns an array of all cells that contain the point which is the origin point of the given neighborhood
    pub fn cells_adjacent_to_point<'a>(
        &self,
        neighborhood: &Neighborhood<'a, I>,
    ) -> [Option<CellIndex<I>>; 8] {
        let cells_above = neighborhood
            .get_neighbor_edge(Axis::Z.with_direction(Direction::Positive))
            .map(|edge| self.cells_adjacent_to_edge(&edge));
        let cells_below = neighborhood
            .get_neighbor_edge(Axis::Z.with_direction(Direction::Negative))
            .map(|edge| self.cells_adjacent_to_edge(&edge));

        match (cells_above, cells_below) {
            (Some(cells_above), Some(cells_below)) => [
                cells_above[0],
                cells_above[1],
                cells_above[2],
                cells_above[3],
                cells_below[0],
                cells_below[1],
                cells_below[2],
                cells_below[3],
            ],
            (Some(cells_above), None) => [
                cells_above[0],
                cells_above[1],
                cells_above[2],
                cells_above[3],
                None,
                None,
                None,
                None,
            ],
            (None, Some(cells_below)) => [
                None,
                None,
                None,
                None,
                cells_below[0],
                cells_below[1],
                cells_below[2],
                cells_below[3],
            ],
            (None, None) => [None, None, None, None, None, None, None, None],
        }
    }

    /// Iterator over all 26 (27-1) valid cells that are adjacent to the given cell
    pub fn cells_adjacent_to_cell<'a>(
        &'a self,
        cell: &'a CellIndex<I>,
    ) -> impl Iterator<Item = CellIndex<I>> + 'a {
        let index = cell.index();
        let steps = Direction::all();
        iproduct!(steps.iter(), steps.iter(), steps.iter()).filter_map(
            move |(step_x, step_y, step_z)| {
                let neighbor_cell_ijk = [
                    step_x.checked_apply_step(index[0], I::one())?,
                    step_y.checked_apply_step(index[1], I::one())?,
                    step_z.checked_apply_step(index[2], I::one())?,
                ];
                self.get_cell(&neighbor_cell_ijk)
            },
        )
    }

    // Helper functions for construction of the SparseGrid struct

    fn checked_n_cells_per_dim(n_cells_real: &Vector3<R>) -> Option<[I; 3]> {
        // TODO: Replace ceil by floor, so that the grid AABB is actually a bounding box of the grid
        //  Then, if one dimension contains zero cells, return an error
        Some([
            I::one().max(n_cells_real[0].ceil().to_index()?),
            I::one().max(n_cells_real[1].ceil().to_index()?),
            I::one().max(n_cells_real[2].ceil().to_index()?),
        ])
    }

    fn checked_n_points_per_dim(n_cells_per_dim: &[I; 3]) -> Option<[I; 3]> {
        Some([
            n_cells_per_dim[0].checked_add(&I::one())?,
            n_cells_per_dim[1].checked_add(&I::one())?,
            n_cells_per_dim[2].checked_add(&I::one())?,
        ])
    }

    fn checked_aabb(
        min: &Vector3<R>,
        n_cells_per_dim: &[I; 3],
        cell_size: R,
    ) -> Option<AxisAlignedBoundingBox3d<R>> {
        let max = min
            + Vector3::new(
                cell_size * n_cells_per_dim[0].to_real()?,
                cell_size * n_cells_per_dim[1].to_real()?,
                cell_size * n_cells_per_dim[2].to_real()?,
            );

        Some(AxisAlignedBoundingBox3d::new(min.clone(), max))
    }

    fn checked_num_points(n_points_per_dim: &[I; 3]) -> Option<I> {
        n_points_per_dim[0]
            .checked_mul(&n_points_per_dim[1])?
            .checked_mul(&n_points_per_dim[2])
    }
}

impl<I: Index> PointIndex<I> {
    #[inline(always)]
    fn from_ijk(point_ijk: [I; 3]) -> Self {
        Self { index: point_ijk }
    }

    /// Returns the index array of the point
    #[inline(always)]
    pub fn index(&self) -> &[I; 3] {
        &self.index
    }
}

impl<I: Index> CellIndex<I> {
    #[inline(always)]
    fn from_ijk(cell_ijk: [I; 3]) -> Self {
        Self { index: cell_ijk }
    }

    #[inline(always)]
    fn from_point(point_index: PointIndex<I>) -> Self {
        Self {
            index: point_index.index,
        }
    }

    /// Returns the index array of the cell
    #[inline(always)]
    pub fn index(&self) -> &[I; 3] {
        &self.index
    }

    /// Returns an array containing all local edge indices that are parallel to the given axis in CCW ordering
    #[inline(always)]
    pub fn local_edges_parallel_to(axis: Axis) -> &'static [usize; 4] {
        &CELL_LOCAL_EDGES[axis.dim()]
    }

    /// Returns the local index inside of the cube of the given point
    #[inline(always)]
    pub fn local_point_index_of(&self, ijk: &[I; 3]) -> Option<usize> {
        let delta = [
            ijk[0].checked_sub(&self.index[0])?.to_usize()?,
            ijk[1].checked_sub(&self.index[1])?.to_usize()?,
            ijk[2].checked_sub(&self.index[2])?.to_usize()?,
        ];
        let flat_index = delta[2] * 4 + delta[1] * 2 + delta[0];
        CELL_LOCAL_POINTS.get(flat_index).copied()
    }

    #[inline(always)]
    pub fn local_edge_index_of<'a, 'b>(&self, edge: &NeighborEdge<'a, 'b, I>) -> Option<usize> {
        let (start_point, _) = edge.ascending_point_order();
        let start_point_local = self.local_point_index_of(start_point.index())?;
        let edge_dim = edge.connectivity.axis.dim();
        CELL_LOCAL_EDGES_FROM_LOCAL_POINT[start_point_local][edge_dim]
    }

    /// Converts the given local point index (0 to 7) to a global point index
    #[inline(always)]
    pub fn global_index_of(&self, local_index: usize) -> Option<PointIndex<I>> {
        let local_coords = CELL_LOCAL_POINT_COORDS.get(local_index)?;
        Some(PointIndex::from_ijk([
            self.index[0] + I::from_i32(local_coords[0])?,
            self.index[1] + I::from_i32(local_coords[1])?,
            self.index[2] + I::from_i32(local_coords[2])?,
        ]))
    }
}

#[test]
fn test_cube_cell_local_point_index() {
    let cube: CellIndex<i32> = CellIndex { index: [1, 1, 1] };

    assert_eq!(cube.local_point_index_of(&[1, 1, 1]), Some(0));
    assert_eq!(cube.local_point_index_of(&[2, 1, 1]), Some(1));
    assert_eq!(cube.local_point_index_of(&[2, 2, 1]), Some(2));
    assert_eq!(cube.local_point_index_of(&[1, 2, 1]), Some(3));
    assert_eq!(cube.local_point_index_of(&[1, 1, 2]), Some(4));
    assert_eq!(cube.local_point_index_of(&[2, 1, 2]), Some(5));
    assert_eq!(cube.local_point_index_of(&[2, 2, 2]), Some(6));
    assert_eq!(cube.local_point_index_of(&[1, 2, 2]), Some(7));
    assert_eq!(cube.local_point_index_of(&[0, 2, 2]), None);
    assert_eq!(cube.local_point_index_of(&[1, 2, 3]), None);
}

const CELL_LOCAL_POINTS: [usize; 8] = [0, 1, 3, 2, 4, 5, 7, 6];

const CELL_LOCAL_POINT_COORDS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// Maps from a local point in a cell and an axis direction originating from this point to the local edge index
#[rustfmt::skip]
const CELL_LOCAL_EDGES_FROM_LOCAL_POINT: [[Option<usize>; 3]; 8] = [
    [Some(0), Some(3), Some(8) ],
    [None   , Some(1), Some(9) ],
    [None   , None   , Some(10)],
    [Some(2), None   , Some(11)],
    [Some(4), Some(7), None    ],
    [None   , Some(5), None    ],
    [None   , None   , None    ],
    [Some(6), None   , None    ],
];

const LOCAL_EDGES_PARALLEL_TO_X_AXIS: [usize; 4] = [0, 8, 10, 2];
const LOCAL_EDGES_PARALLEL_TO_Y_AXIS: [usize; 4] = [3, 1, 9, 11];
const LOCAL_EDGES_PARALLEL_TO_Z_AXIS: [usize; 4] = [4, 5, 6, 7];

const CELL_LOCAL_EDGES: [[usize; 4]; 3] = [
    LOCAL_EDGES_PARALLEL_TO_X_AXIS,
    LOCAL_EDGES_PARALLEL_TO_Y_AXIS,
    LOCAL_EDGES_PARALLEL_TO_Z_AXIS,
];

impl Direction {
    pub fn all() -> &'static [Direction; 2] {
        &ALL_DIRECTIONS
    }

    /// Adds or subtracts the given step from the value depending on the direction
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::Direction;
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
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::Direction;
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

    /// Returns whether the direction is positive
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::Direction;
    /// assert_eq!(Direction::Positive.is_positive(), true);
    /// assert_eq!(Direction::Negative.is_positive(), false);
    /// ```
    #[inline(always)]
    pub const fn is_positive(&self) -> bool {
        *self as i32 == 1
    }

    /// Returns whether the direction is negative
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::Direction;
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
    /// Converts the cartesian axis into the 3D dimension index (X=0, Y=1, Z=2)
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::CartesianAxis3d as Axis;
    /// assert_eq!(Axis::X.dim(), 0);
    /// assert_eq!(Axis::Y.dim(), 1);
    /// assert_eq!(Axis::Z.dim(), 2);
    /// ```
    #[inline(always)]
    pub const fn dim(self) -> usize {
        self as usize
    }

    /// Returns the other two axes that are orthogonal to the current axis
    /// ```ignore
    /// use crate::splashsurf_lib::uniform_grid::CartesianAxis3d as Axis;
    /// assert_eq!(Axis::X.orthogonal_axes(), [Axis::Y, Axis::Z]);
    /// ```
    #[inline(always)]
    pub const fn orthogonal_axes(&self) -> [Self; 2] {
        ORTHOGONAL_AXES[self.dim()]
    }

    /// Combines this coordinate axis with a direction into a DirectedAxis
    #[inline(always)]
    pub const fn with_direction(self, direction: Direction) -> DirectedAxis {
        DirectedAxis {
            axis: self,
            direction,
        }
    }
}

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
        &ALL_GRID_NEIGHBOR_LOCATIONS
    }

    #[inline(always)]
    pub const fn from_usize(n: usize) -> Self {
        Self::all_possible()[n]
    }

    #[inline(always)]
    pub const fn to_usize(&self) -> usize {
        (self.direction as usize * 3) + self.axis as usize
    }

    #[inline(always)]
    pub fn apply_step<N: Clone + CheckedAdd<Output = N> + CheckedSub<Output = N> + One>(
        &self,
        index: &[N; 3],
    ) -> Option<[N; 3]> {
        let mut index = index.clone();
        index[self.axis().dim()] = self
            .direction
            .checked_apply_step(index[self.axis().dim()].clone(), N::one())?;
        Some(index)
    }

    /// Cartesian axis
    #[inline(always)]
    pub const fn axis(&self) -> Axis {
        self.axis
    }

    /// Direction along the axis
    #[inline(always)]
    pub const fn direction(&self) -> Direction {
        self.direction
    }
}

#[test]
fn test_directed_axis_usize_conversion() {
    for i in 0..6 {
        debug_assert_eq!(DirectedAxis::from_usize(i).to_usize(), i);
    }
}

#[test]
fn test_directed_axis_all_possible_consistency() {
    let all_directed_axes = DirectedAxis::all_possible();
    for (i, ax) in all_directed_axes.iter().enumerate() {
        debug_assert_eq!(ax.to_usize(), i);
        debug_assert_eq!(*ax, DirectedAxis::from_usize(i));
    }
}

impl<'a, I: Index> Neighborhood<'a, I> {
    /// Returns if the origin point has a valid neighbor following the specified directed axis
    #[inline(always)]
    pub fn has_neighbor(&self, direction: DirectedAxis) -> bool {
        self.neighbors[direction.to_usize()].is_some()
    }

    /// Get a specific neighbor in the given direction from the origin point of the neighborhood
    #[inline(always)]
    pub fn get_neighbor(&self, direction: DirectedAxis) -> Option<PointIndex<I>> {
        self.neighbors[direction.to_usize()]
    }

    /// Get the edge to a specific neighbor in the given direction from the origin point of the neighborhood
    #[inline(always)]
    pub fn get_neighbor_edge<'b>(
        &'b self,
        direction: DirectedAxis,
    ) -> Option<NeighborEdge<'b, 'b, I>> {
        self.neighbors[direction.to_usize()]
            .as_ref()
            .map(|neighbor| self.new_neighbor_edge(neighbor, direction))
    }

    /// Iterate over all valid neighbor points and the corresponding directed axis from the origin to the neighbor
    pub fn neighbor_edge_iter<'b>(&'b self) -> impl Iterator<Item = NeighborEdge<'b, 'b, I>> {
        self.neighbors
            .iter()
            .enumerate()
            .filter_map(move |(i, optional_neighbor)| {
                optional_neighbor
                    .as_ref()
                    .map(|neighbor| self.new_neighbor_edge(neighbor, DirectedAxis::from_usize(i)))
            })
    }

    fn new_neighbor_edge<'b>(
        &'b self,
        neighbor: &'b PointIndex<I>,
        connectivity: DirectedAxis,
    ) -> NeighborEdge<'b, 'b, I> {
        NeighborEdge {
            neighborhood: self,
            neighbor,
            connectivity,
        }
    }
}

impl<'a, 'b, I: Index> NeighborEdge<'a, 'b, I> {
    /// Returns the origin of this neighbor edge
    #[inline(always)]
    pub fn origin_index(&self) -> &PointIndex<I> {
        &self.neighborhood.origin
    }

    /// Returns the neighbor of the origin node connected by this neighbor edge
    #[inline(always)]
    pub fn neighbor_index(&self) -> &PointIndex<I> {
        &self.neighbor
    }

    /// Returns the connectivity between the origin point and its neighbor in terms of a directed axis
    #[inline(always)]
    pub fn connectivity(&self) -> DirectedAxis {
        self.connectivity
    }

    /// Returns references to the point indices of the edge, ordered in such a way that they are in ascending point index order.
    /// That means that the first point index is connected to the second point index with an edge along an axis in positive direction.
    #[inline(always)]
    pub fn ascending_point_order(&self) -> (&PointIndex<I>, &PointIndex<I>) {
        if self.connectivity.direction.is_positive() {
            (self.origin_index(), &self.neighbor_index())
        } else {
            (&self.neighbor_index(), self.origin_index())
        }
    }
}

const ORTHOGONAL_TO_X: [Axis; 2] = [Axis::Y, Axis::Z];
const ORTHOGONAL_TO_Y: [Axis; 2] = [Axis::Z, Axis::X];
const ORTHOGONAL_TO_Z: [Axis; 2] = [Axis::X, Axis::Y];
const ORTHOGONAL_AXES: [[Axis; 2]; 3] = [ORTHOGONAL_TO_X, ORTHOGONAL_TO_Y, ORTHOGONAL_TO_Z];

const ALL_GRID_NEIGHBOR_LOCATIONS: [DirectedAxis; 6] = [
    DirectedAxis {
        axis: Axis::X,
        direction: Direction::Negative,
    },
    DirectedAxis {
        axis: Axis::Y,
        direction: Direction::Negative,
    },
    DirectedAxis {
        axis: Axis::Z,
        direction: Direction::Negative,
    },
    DirectedAxis {
        axis: Axis::X,
        direction: Direction::Positive,
    },
    DirectedAxis {
        axis: Axis::Y,
        direction: Direction::Positive,
    },
    DirectedAxis {
        axis: Axis::Z,
        direction: Direction::Positive,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_grid<I: Index, R: Real>() -> UniformGrid<I, R> {
        let origin = Vector3::new(R::zero(), R::zero(), R::zero());
        let n_cubes_per_dim = [I::one(), I::one(), I::one()];
        let cube_size = R::one();

        UniformGrid::new(&origin, &n_cubes_per_dim, cube_size).unwrap()
    }

    #[test]
    fn test_basic_uniform_grid_features() {
        let grid = unit_grid::<i32, f64>();

        assert_eq!(grid.aabb().max(), &Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(grid.cell_size(), 1.0);

        let points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ];

        for point in points.iter() {
            assert!(grid.point_exists(point));
        }

        assert!(grid.cell_exists(&[0, 0, 0]));

        let origin = grid.get_point(&points[0]);
        assert!(origin.is_some());
        let origin = origin.unwrap();

        assert_eq!(
            grid.get_point_neighbor(&origin, Axis::X.with_direction(Direction::Positive))
                .unwrap()
                .index(),
            &[1, 0, 0]
        );
        assert_eq!(
            grid.get_point_neighbor(&origin, Axis::Y.with_direction(Direction::Positive))
                .unwrap()
                .index(),
            &[0, 1, 0]
        );
        assert_eq!(
            grid.get_point_neighbor(&origin, Axis::Z.with_direction(Direction::Positive))
                .unwrap()
                .index(),
            &[0, 0, 1]
        );
        assert!(grid
            .get_point_neighbor(&origin, Axis::X.with_direction(Direction::Negative))
            .is_none());
        assert!(grid
            .get_point_neighbor(&origin, Axis::Y.with_direction(Direction::Negative))
            .is_none());
        assert!(grid
            .get_point_neighbor(&origin, Axis::Z.with_direction(Direction::Negative))
            .is_none());
    }
}
