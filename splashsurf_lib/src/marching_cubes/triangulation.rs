use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::marching_cubes::{CellData, MarchingCubesInput};
use crate::mesh::TriMesh3d;
use crate::{Index, Real, profile};
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

/// Converts the marching cubes input cell data into a triangle surface mesh, appends triangles to existing mesh
#[inline(never)]
pub(crate) fn triangulate<I: Index, R: Real>(
    input: MarchingCubesInput<I>,
    mesh: &mut TriMesh3d<R>,
) -> Result<(), TriangulationError> {
    profile!("triangulate");

    let MarchingCubesInput { cell_data } = input;

    trace!(
        "Starting marching cubes triangulation (Input: cell data map with {} cells, surface mesh with {} triangles and {} vertices)",
        cell_data.len(),
        mesh.triangles.len(),
        mesh.vertices.len(),
    );

    // Triangulate affected cells
    for (&_flat_cell_index, cell_data) in &cell_data {
        // TODO: Replace `are_vertices_above_unchecked` with something that can return an error
        for triangle in marching_cubes_triangulation_iter(&cell_data.are_vertices_above_unchecked())
        {
            // TODO: Allow user to set option to skip invalid triangles?
            let global_triangle = get_triangle(cell_data, triangle)
                .map_err(TriangulationError::TriangleConnectivityError)?;
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
