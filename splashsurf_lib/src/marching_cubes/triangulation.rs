use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::marching_cubes::{CellData, MarchingCubesInput};
use crate::mesh::TriMesh3d;
use crate::uniform_grid::{DummySubdomain, Subdomain, UniformGrid};
use crate::{profile, Index, Real};
use anyhow::Context;
use log::trace;
use thiserror::Error as ThisError;

/// Error enum for the marching cubes triangulation stage
#[derive(Debug, ThisError)]
pub enum TriangulationError {
    /// Error that occurred while generating the triangle connectivity for a case
    #[error("triangle connectivity error")]
    TriangleConnectivityError(
        #[source]
        #[from]
        anyhow::Error,
    ),
}

/// Trait that is used by the marching cubes [triangulate_with_criterion] function to query whether a cell should be triangulated
pub(crate) trait TriangulationCriterion<I: Index, R: Real, S: Subdomain<I, R>> {
    /// Returns whether the given cell should be triangulated
    fn triangulate_cell(&self, subdomain: &S, flat_cell_index: I, cell_data: &CellData) -> bool;
}

/// An identity triangulation criterion that accepts all cells
pub(crate) struct TriangulationIdentityCriterion;

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
        DefaultTriangleGenerator,
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

impl<I: Index, R: Real, S: Subdomain<I, R>> TriangulationCriterion<I, R, S>
    for TriangulationIdentityCriterion
{
    #[inline(always)]
    fn triangulate_cell(&self, _: &S, _: I, _: &CellData) -> bool {
        true
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
