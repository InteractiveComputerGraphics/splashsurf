use crate::marching_cubes::{CellData, MarchingCubesInput, RelativeToThreshold};
use crate::uniform_grid::CellIndex;
use crate::{DensityMap, Index, MapType, Real, UniformGrid, profile};
use log::trace;
use nalgebra::Vector3;

/// Returns the marching cubes input data for the narrow band of a single contiguous domain without support for stitching
pub(crate) fn construct_mc_input<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    vertices: &mut Vec<Vector3<R>>,
) -> MarchingCubesInput<I> {
    let mut marching_cubes_data = MarchingCubesInput::default();
    interpolate_points_to_cell_data_generic::<I, R>(
        grid,
        density_map,
        iso_surface_threshold,
        vertices,
        &mut marching_cubes_data,
    );

    update_cell_data_threshold_flags(
        grid,
        density_map,
        iso_surface_threshold,
        true,
        &mut marching_cubes_data,
    );

    marching_cubes_data
}

/// Generates input data for performing the actual marching cubes triangulation
///
/// This function interpolates iso-surface vertices for cells in the narrow band around the iso-surface.
/// The interpolated vertices are appended to the given vertex vector.
///
/// Furthermore, it collects marching cubes input data containing data about each cell in the narrow
/// band around the iso-surface. Each of these cells has to be triangulated in a following step.
///
/// For each cell, this function collects
///   1) an array with a flag per corner vertex, indicating whether it's above/below the iso-surface threshold
///   2) an array with an optional index per edge, referring to the interpolated vertex if the edge crosses the iso-surface
///
/// Note: The threshold flags in the resulting cell data are not complete and still have to be updated after
/// this procedure using the [update_cell_data_threshold_flags] function.
///
/// Note: This functions assumes that the default value for missing point data is zero.
#[inline(never)]
fn interpolate_points_to_cell_data_generic<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    vertices: &mut Vec<Vector3<R>>,
    marching_cubes_data: &mut MarchingCubesInput<I>,
) {
    profile!("interpolate_points_to_cell_data_generic");
    trace!(
        "Starting interpolation of cell data for marching cubes (excluding boundary layer)... (Input: {} existing vertices)",
        vertices.len()
    );

    // Map from flat cell index to all data that is required per cell for the marching cubes triangulation
    let cell_data: &mut MapType<I, CellData> = &mut marching_cubes_data.cell_data;

    // Generate iso-surface vertices and identify affected cells & edges
    {
        profile!("generate_iso_surface_vertices");
        density_map.for_each(|flat_point_index, point_value| {
            let point = grid.try_unflatten_point_index(flat_point_index).unwrap();

            // We want to find edges that cross the iso-surface, i.e. edges where one point is above
            // and the other below the threshold. To not process an edge twice, we skip all points
            // that are already below the iso-surface threshold. Note that we cannot do it the other
            // way around, as some points below the threshold might not be part of the density map
            // at all (e.g. points outside the kernel evaluation radius). This could lead to missing
            // edges that go directly from above the threshold to e.g. zero.
            if point_value < iso_surface_threshold {
                return;
            }

            let neighborhood = grid.get_point_neighborhood(&point);
            // Iterate over all neighbors of the point to find edges crossing the iso-surface
            for neighbor_edge in neighborhood.neighbor_edge_iter() {
                let neighbor = neighbor_edge.neighbor_index();
                // Get flat index of neighbor on global grid
                let flat_neighbor_index = grid.flatten_point_index(neighbor);

                // Try to read out the function value at the neighboring point
                let neighbor_value = if let Some(v) = density_map.get(flat_neighbor_index) {
                    v
                } else {
                    // Neighbors that are not in the point-value map were outside the kernel evaluation radius.
                    // Assume zero density for these points.
                    R::zero()
                };

                // Skip edges that don't cross the iso-surface
                if !(neighbor_value < iso_surface_threshold) {
                    continue;
                }

                // Interpolate iso-surface vertex on the edge
                let alpha = (iso_surface_threshold - point_value) / (neighbor_value - point_value);
                let point_coords = grid.point_coordinates(&point);
                let neighbor_coords = grid.point_coordinates(neighbor);
                let interpolated_coords =
                    (point_coords) * (R::one() - alpha) + neighbor_coords * alpha;

                // Store interpolated vertex and remember its index
                let vertex_index = vertices.len();
                vertices.push(interpolated_coords);

                // Store the data required for the marching cubes triangulation for
                // each cell adjacent to the edge crossing the iso-surface.
                // This includes the above/below iso-surface flags and the interpolated vertex index.
                for cell in grid.cells_adjacent_to_edge(&neighbor_edge).iter().flatten() {
                    let flat_cell_index = grid.flatten_cell_index(cell);

                    let cell_data_entry = cell_data.entry(flat_cell_index).or_default();

                    // Store the index of the interpolated vertex on the corresponding local edge of the cell
                    let local_edge_index = cell.local_edge_index_of(&neighbor_edge).unwrap();

                    // Ensure that there is not already data stored for this edge
                    assert!(
                        cell_data_entry.iso_surface_vertices[local_edge_index].is_none(),
                        "Overwriting already existing vertex. This is a bug."
                    );
                    cell_data_entry.iso_surface_vertices[local_edge_index] = Some(vertex_index);

                    // Mark the corner of the current point as above the iso-surface threshold
                    let local_vertex_index = cell.local_point_index_of(point.index()).unwrap();
                    cell_data_entry.corner_above_threshold[local_vertex_index] =
                        RelativeToThreshold::Above;
                }
            }
        });
    }

    trace!(
        "Cell data interpolation done. (Output: cell data for marching cubes with {} cells and {} vertices)",
        cell_data.len(),
        vertices.len()
    );
}

/// Loops through all corner vertices in the given marching cubes input and updates the above/below threshold flags
fn update_cell_data_threshold_flags<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    skip_points_above_threshold: bool,
    marching_cubes_input: &mut MarchingCubesInput<I>,
) {
    // Cell corner points above the iso-surface threshold which are only surrounded by neighbors that
    // are also above the threshold were not marked as `corner_above_threshold = true` before, because they
    // don't have any adjacent edge crossing the iso-surface (and thus were never touched by the point data loop).
    // This can happen in a configuration where e.g. only one corner is below the threshold.
    //
    // Therefore, we have to loop over all corner points of all cells that were collected for marching cubes
    // and check their density value again.
    //
    // Note, that we would also have this problem if we flipped the default/initial value of corner_above_threshold
    // to false. In this case we could also move this into the point data loop (which might increase performance).
    // However, we would have to special case cells without point data, which are currently skipped.
    // Similarly, they have to be treated in a second pass because we don't want to initialize cells only
    // consisting of missing points and points below the surface.
    profile!("relative_to_threshold_postprocessing");

    let update_corner_flags =
        |cell: &CellIndex<I>, local_point_index: usize, flag: &mut RelativeToThreshold| {
            // Otherwise try to look up its value and potentially mark it as above the threshold
            let point = cell.global_point_index_of(local_point_index).unwrap();
            let flat_point_index = grid.flatten_point_index(&point);
            // Update flag depending on value in density map
            *flag = {
                if let Some(point_value) = density_map.get(flat_point_index) {
                    if point_value > iso_surface_threshold {
                        RelativeToThreshold::Above
                    } else {
                        RelativeToThreshold::Below
                    }
                } else {
                    // Points that are not in the density map are assumed to be below the iso-surface threshold.
                    // This is done to avoid unnecessarily large domains where the density map has to provide values
                    // that are trivially below the iso-surface threshold.
                    RelativeToThreshold::Below
                }
            }
        };

    if skip_points_above_threshold {
        for (&flat_cell_index, cell_data) in marching_cubes_input.cell_data.iter_mut() {
            let cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();
            cell_data
                .corner_above_threshold
                .iter_mut()
                .enumerate()
                // If the point is already marked as above we can ignore it
                .filter(|(_, flag)| **flag != RelativeToThreshold::Above)
                .for_each(|(local_point_index, flag)| {
                    update_corner_flags(&cell, local_point_index, flag)
                });
        }
    } else {
        for (&flat_cell_index, cell_data) in marching_cubes_input.cell_data.iter_mut() {
            let cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();
            cell_data
                .corner_above_threshold
                .iter_mut()
                .enumerate()
                .for_each(|(local_point_index, flag)| {
                    update_corner_flags(&cell, local_point_index, flag)
                });
        }
    }
}
