use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::marching_cubes::{CellData, MarchingCubesInput};
use crate::mesh::TriMesh3d;
use crate::topology::Axis;
use crate::uniform_grid::{DummySubdomain, GridBoundaryFaceFlags, Subdomain, UniformGrid};
use crate::{profile, Index, Real};
use anyhow::Context;
use log::trace;
use nalgebra::Vector3;
use std::marker::PhantomData;
use thiserror::Error as ThisError;

/// Error enum for the marching cubes triangulation stage
#[derive(Debug, ThisError)]
pub enum TriangulationError {
    /// Error that occurred while generating the triangle connectivity for a case
    #[error("triangle connectivity error: {0}")]
    TriangleConnectivityError(anyhow::Error),
}

/// Trait that is used by the marching cubes [triangulate_with_criterion] function to query whether a cell should be triangulated
pub(crate) trait TriangulationCriterion<I: Index, R: Real, S: Subdomain<I, R>> {
    /// Returns whether the given cell should be triangulated
    fn triangulate_cell(&self, subdomain: &S, flat_cell_index: I, cell_data: &CellData) -> bool;
}

/// An identity triangulation criterion that accepts all cells
pub(crate) struct TriangulationIdentityCriterion;
/// A triangulation criterion that ensures that every cell is part of the subdomain but skips one layer of boundary cells
pub(crate) struct TriangulationSkipBoundaryCells;
/// A triangulation criterion that ensures that only the interior of the stitching domain is triangulated (boundary layer except in stitching direction is skipped)
pub(crate) struct TriangulationStitchingInterior {
    stitching_axis: Axis,
}

/// Trait that is used by the marching cubes [triangulate_with_criterion] function to convert a marching cubes triangulation to actual triangle-vertex connectivity
pub(crate) trait TriangleGenerator<I: Index, R: Real, S: Subdomain<I, R>> {
    fn triangle_connectivity(
        &self,
        subdomain: &S,
        flat_cell_index: I,
        cell_data: &CellData,
        edge_indices: [i32; 3],
    ) -> Result<[usize; 3], anyhow::Error>;
}

/// Maps the edges indices directly to the vertex indices in the cell data
pub(crate) struct DefaultTriangleGenerator;
/// Tries to map the edge indices to the vertex indices in the cell data, returns an error with debug information if vertices are missing
pub(crate) struct DebugTriangleGenerator;

/// Converts the marching cubes input cell data into a triangle surface mesh, appends triangles to existing mesh
#[inline(never)]
pub(crate) fn triangulate<I: Index, R: Real>(
    input: MarchingCubesInput<I>,
    mesh: &mut TriMesh3d<R>,
) -> Result<(), TriangulationError> {
    triangulate_with_criterion(
        &DummySubdomain::new(&UniformGrid::new_zero()),
        input,
        mesh,
        TriangulationIdentityCriterion,
        DebugTriangleGenerator,
    )
}

/// Converts the marching cubes input cell data into a triangle surface mesh, appends triangles to existing mesh with custom criterion to filter out cells during triangulation
#[inline(never)]
pub(crate) fn triangulate_with_criterion<
    I: Index,
    R: Real,
    S: Subdomain<I, R>,
    C: TriangulationCriterion<I, R, S>,
    G: TriangleGenerator<I, R, S>,
>(
    subdomain: &S,
    input: MarchingCubesInput<I>,
    mesh: &mut TriMesh3d<R>,
    triangulation_criterion: C,
    triangle_generator: G,
) -> Result<(), TriangulationError> {
    profile!("triangulate_with_criterion");

    let MarchingCubesInput { cell_data } = input;

    trace!(
        "Starting marching cubes triangulation (Input: cell data map with {} cells, surface mesh with {} triangles and {} vertices)",
        cell_data.len(),
        mesh.triangles.len(),
        mesh.vertices.len(),
    );

    // Triangulate affected cells
    for (&flat_cell_index, cell_data) in &cell_data {
        // Skip cells that don't fulfill triangulation criterion
        if !triangulation_criterion.triangulate_cell(subdomain, flat_cell_index, cell_data) {
            continue;
        }

        // TODO: Replace `are_vertices_above_unchecked` with something that can return an error
        for triangle in marching_cubes_triangulation_iter(&cell_data.are_vertices_above_unchecked())
        {
            // TODO: Allow user to set option to skip invalid triangles?
            let global_triangle = triangle_generator
                .triangle_connectivity(subdomain, flat_cell_index, cell_data, triangle)
                .map_err(|e| TriangulationError::TriangleConnectivityError(e))?;
            mesh.triangles.push(global_triangle);
        }
    }

    trace!(
        "Triangulation done. (Output: surface mesh with {} triangles and {} vertices)",
        mesh.triangles.len(),
        mesh.vertices.len()
    );

    Ok(())
}

/// Forwards to the wrapped triangulation criterion but first makes some assertions on the cell data
struct DebugTriangulationCriterion<
    I: Index,
    R: Real,
    S: Subdomain<I, R>,
    C: TriangulationCriterion<I, R, S>,
> {
    triangulation_criterion: C,
    phantom: PhantomData<(I, R, S)>,
}

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangulationCriterion<I, R, S>
    for TriangulationIdentityCriterion
{
    #[inline(always)]
    fn triangulate_cell(&self, _: &S, _: I, _: &CellData) -> bool {
        true
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangulationCriterion<I, R, S>
    for TriangulationSkipBoundaryCells
{
    #[inline(always)]
    fn triangulate_cell(&self, subdomain: &S, flat_cell_index: I, _: &CellData) -> bool {
        let global_cell = subdomain
            .global_grid()
            .try_unflatten_cell_index(flat_cell_index)
            .unwrap();
        let local_cell = subdomain
            .map_cell(&global_cell)
            .expect("Cell is not part of the subdomain");
        let cell_grid_boundaries =
            GridBoundaryFaceFlags::classify_cell(subdomain.subdomain_grid(), &local_cell);

        return cell_grid_boundaries.is_empty();
    }
}

impl TriangulationStitchingInterior {
    pub(crate) fn new(stitching_axis: Axis) -> Self {
        Self { stitching_axis }
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangulationCriterion<I, R, S>
    for TriangulationStitchingInterior
{
    #[inline(always)]
    fn triangulate_cell(&self, subdomain: &S, flat_cell_index: I, _: &CellData) -> bool {
        let global_cell = subdomain
            .global_grid()
            .try_unflatten_cell_index(flat_cell_index)
            .unwrap();

        let local_cell = subdomain
            .map_cell(&global_cell)
            .expect("Cell is not part of the subdomain");

        let subdomain_grid = subdomain.subdomain_grid();
        let index = local_cell.index();

        // Skip only boundary cells in directions orthogonal to the stitching axis
        !self.stitching_axis.orthogonal_axes().iter().any(|&axis| {
            index[axis.dim()] == I::zero()
                || index[axis.dim()] == subdomain_grid.cells_per_dim()[axis.dim()] - I::one()
        })
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>, C: TriangulationCriterion<I, R, S>>
    DebugTriangulationCriterion<I, R, S, C>
{
    #[allow(unused)]
    fn new(triangulation_criterion: C) -> Self {
        Self {
            triangulation_criterion,
            phantom: Default::default(),
        }
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>, C: TriangulationCriterion<I, R, S>>
    TriangulationCriterion<I, R, S> for DebugTriangulationCriterion<I, R, S, C>
{
    #[inline(always)]
    fn triangulate_cell(&self, subdomain: &S, flat_cell_index: I, cell_data: &CellData) -> bool {
        assert!(
            !cell_data
                .corner_above_threshold
                .iter()
                .any(|c| c.is_indeterminate()),
            "{}",
            cell_debug_string(subdomain, flat_cell_index, cell_data)
        );

        self.triangulation_criterion
            .triangulate_cell(subdomain, flat_cell_index, cell_data)
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangleGenerator<I, R, S>
    for DefaultTriangleGenerator
{
    #[inline(always)]
    fn triangle_connectivity(
        &self,
        _subdomain: &S,
        _flat_cell_index: I,
        cell_data: &CellData,
        edge_indices: [i32; 3],
    ) -> Result<[usize; 3], anyhow::Error> {
        // Note: If the one of the following expect calls causes a panic, it is probably because
        //  a cell was added improperly to the marching cubes input, e.g. a cell was added to the
        //  cell data map that is not part of the domain. This results in only those edges of the cell
        //  that are neighboring to the domain having correct iso surface vertices. The remaining
        //  edges would have missing iso-surface vertices and overall this results in an invalid triangulation
        //
        //  If this happens, it's a bug in the cell data map generation.
        get_triangle(cell_data, edge_indices)
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangleGenerator<I, R, S> for DebugTriangleGenerator {
    #[inline(always)]
    fn triangle_connectivity(
        &self,
        subdomain: &S,
        flat_cell_index: I,
        cell_data: &CellData,
        edge_indices: [i32; 3],
    ) -> Result<[usize; 3], anyhow::Error> {
        get_triangle(cell_data, edge_indices)
            .with_context(|| cell_debug_string(subdomain, flat_cell_index, cell_data))
    }
}

/// Helper function that extracts vertex indices from the [`CellData`] for the triangle with the given edge indices
fn get_triangle(cell_data: &CellData, edge_indices: [i32; 3]) -> Result<[usize; 3], anyhow::Error> {
    let [edge_idx_0, edge_idx_1, edge_idx_2] = edge_indices;

    Ok([
        cell_data
            .iso_surface_vertices
            .get(edge_idx_0 as usize)
            .with_context(|| "Invalid edge index. This is a bug.")?
            .with_context(|| {
                format!(
                    "Missing iso surface vertex at edge {}. This is a bug.",
                    edge_idx_0
                )
            })?,
        cell_data
            .iso_surface_vertices
            .get(edge_idx_1 as usize)
            .with_context(|| "Invalid edge index. This is a bug.")?
            .with_context(|| {
                format!(
                    "Missing iso surface vertex at edge {}. This is a bug.",
                    edge_idx_1
                )
            })?,
        cell_data
            .iso_surface_vertices
            .get(edge_idx_2 as usize)
            .with_context(|| "Invalid edge index. This is a bug.")?
            .with_context(|| {
                format!(
                    "Missing iso surface vertex at edge {}. This is a bug.",
                    edge_idx_2
                )
            })?,
    ])
}

/// Helper function that returns a formatted string to debug triangulation failures
fn cell_debug_string<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
    flat_cell_index: I,
    cell_data: &CellData,
) -> String {
    let global_cell_index = subdomain
        .global_grid()
        .try_unflatten_cell_index(flat_cell_index)
        .expect("Failed to get cell index");

    let point_index = subdomain
        .global_grid()
        .get_point(*global_cell_index.index())
        .expect("Unable to get point index of cell");
    let cell_center = subdomain.global_grid().point_coordinates(&point_index)
        + &Vector3::repeat(subdomain.global_grid().cell_size().times_f64(0.5));

    format!(
        "Unable to construct triangle for cell {:?}, with center coordinates {:?} and edge length {}.\n{:?}\nStitching domain: (offset: {:?}, cells_per_dim: {:?})",
        global_cell_index.index(),
        cell_center,
        subdomain.global_grid().cell_size(),
        cell_data,
        subdomain.subdomain_offset(),
        subdomain.subdomain_grid().cells_per_dim(),
    )
}
