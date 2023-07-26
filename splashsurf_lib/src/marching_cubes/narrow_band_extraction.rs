use crate::marching_cubes::stitching::{collect_boundary_cell_data, BoundaryData};
use crate::marching_cubes::{CellData, MarchingCubesInput, RelativeToThreshold};
use crate::topology::{Axis, DirectedAxisArray};
use crate::uniform_grid::{CellIndex, GridBoundaryFaceFlags, PointIndex, Subdomain};
use crate::{profile, DensityMap, Index, MapType, Real};
use log::trace;
use nalgebra::Vector3;

/// Trait used by [interpolate_points_to_cell_data_generic] to filter out points and edges during construction of iso-surface vertices in narrow-band cells
trait DensityMapFilter<I: Index, R: Real, S: Subdomain<I, R>> {
    /// Returns whether the given point should be considered for the density map to cell data conversion
    fn process_point(
        &mut self,
        density_map: &DensityMap<I, R>,
        subdomain: &S,
        flat_point_index: I,
        subdomain_point: &PointIndex<I>,
        point_value: R,
    ) -> bool;

    /// Returns whether the given edge should be considered for the density map to cell data conversion
    fn process_edge(
        &mut self,
        density_map: &DensityMap<I, R>,
        subdomain: &S,
        flat_point_index: I,
        subdomain_point: &PointIndex<I>,
        flat_neighbor_index: I,
        subdomain_neighbor: &PointIndex<I>,
    ) -> bool;
}

/// Cell data interpolation filter that accepts all points and edges
struct IdentityDensityMapFilter;

/// Cell data interpolation filter that skips one boundary layer of cells of the domain and builds the boundary density maps
struct SkipBoundaryLayerFilter<I: Index, R: Real> {
    boundary_density_maps: DirectedAxisArray<MapType<I, R>>,
}

/// Cell data interpolation filter for processing the stitching domain between two reconstructed surface patches
struct StitchingDomainNarrowBandFilter {
    stitching_axis: Axis,
}

/// Returns the marching cubes input data for the narrow band of a single contiguous domain without support for stitching
pub(crate) fn construct_mc_input<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    vertices: &mut Vec<Vector3<R>>,
) -> MarchingCubesInput<I> {
    let mut marching_cubes_data = MarchingCubesInput::default();
    let _ = interpolate_points_to_cell_data_generic::<I, R, _, _>(
        subdomain,
        density_map,
        iso_surface_threshold,
        vertices,
        &mut marching_cubes_data,
        IdentityDensityMapFilter,
    );

    update_cell_data_threshold_flags(
        subdomain,
        density_map,
        iso_surface_threshold,
        true,
        &mut marching_cubes_data,
    );

    marching_cubes_data
}

/// Returns marching cubes input data for the narrow band of the subdomain and collects all data on the boundaries required for stitching of this domain to neighboring domains
pub(crate) fn construct_mc_input_with_stitching_data<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    vertices: &mut Vec<Vector3<R>>,
) -> (MarchingCubesInput<I>, DirectedAxisArray<BoundaryData<I, R>>) {
    let mut marching_cubes_data = MarchingCubesInput::default();
    let boundary_filter = interpolate_points_to_cell_data_generic(
        subdomain,
        density_map,
        iso_surface_threshold,
        vertices,
        &mut marching_cubes_data,
        SkipBoundaryLayerFilter::new(),
    );

    update_cell_data_threshold_flags(
        subdomain,
        density_map,
        iso_surface_threshold,
        true,
        &mut marching_cubes_data,
    );

    let mut boundary_density_maps = boundary_filter.into_inner();
    let mut boundary_cell_data = collect_boundary_cell_data(subdomain, &mut marching_cubes_data);

    let boundary_data = DirectedAxisArray::new_with(|axis| BoundaryData {
        boundary_density_map: std::mem::take(boundary_density_maps.get_mut(axis)),
        boundary_cell_data: std::mem::take(boundary_cell_data.get_mut(axis)),
    });

    (marching_cubes_data, boundary_data)
}

/// Updates the marching cubes input data in the narrow band of a stitching domain between two reconstructed patches
pub(crate) fn update_mc_input_for_stitching_domain<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    stitching_axis: Axis,
    vertices: &mut Vec<Vector3<R>>,
    marching_cubes_input: &mut MarchingCubesInput<I>,
) {
    let _ = interpolate_points_to_cell_data_generic(
        subdomain,
        density_map,
        iso_surface_threshold,
        vertices,
        marching_cubes_input,
        StitchingDomainNarrowBandFilter::new(stitching_axis),
    );

    update_cell_data_threshold_flags(
        subdomain,
        density_map,
        iso_surface_threshold,
        false,
        marching_cubes_input,
    );
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
/// Note: The threshold flags in the resulting cell data are not complete and still have to be updated after
/// this procedure using the [update_cell_data_threshold_flags] function.
///
/// Note: This functions assumes that the default value for missing point data is below the iso-surface threshold.
#[inline(never)]
fn interpolate_points_to_cell_data_generic<
    I: Index,
    R: Real,
    S: Subdomain<I, R>,
    F: DensityMapFilter<I, R, S>,
>(
    subdomain: &S,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    vertices: &mut Vec<Vector3<R>>,
    marching_cubes_data: &mut MarchingCubesInput<I>,
    mut filter: F,
) -> F {
    let grid = subdomain.global_grid();
    let subdomain_grid = subdomain.subdomain_grid();

    profile!("interpolate_points_to_cell_data_skip_boundary");
    trace!("Starting interpolation of cell data for marching cubes (excluding boundary layer)... (Input: {} existing vertices)", vertices.len());

    // Map from flat cell index to all data that is required per cell for the marching cubes triangulation
    let cell_data: &mut MapType<I, CellData> = &mut marching_cubes_data.cell_data;

    // Generate iso-surface vertices and identify affected cells & edges
    {
        profile!("generate_iso_surface_vertices");
        density_map.for_each(|flat_point_index, point_value| {
            let global_point = grid.try_unflatten_point_index(flat_point_index).unwrap();
            let point = subdomain
                .map_point(&global_point)
                .expect("Point cannot be mapped into subdomain.");

            // First check if the point should be processed, this can also be used to store certain
            // values in a separate storage, e.g. to filter out points on the boundary but store them
            // in a separate list to process them later.
            if !filter.process_point(
                density_map,
                subdomain,
                flat_point_index,
                &point,
                point_value,
            ) {
                return;
            }

            // We want to find edges that cross the iso-surface,
            // therefore we can choose to either skip all points above or below the threshold.
            //
            // In most scenes, the sparse density map should contain more entries above than
            // below the threshold, as it contains the whole fluid interior, whereas areas completely
            // devoid of fluid are not part of the density map.
            //
            // Skip points with densities above the threshold to improve efficiency
            if point_value > iso_surface_threshold {
                return;
            }

            let neighborhood = subdomain_grid.get_point_neighborhood(&point);
            // Iterate over all neighbors of the point to find edges crossing the iso-surface
            for neighbor_edge in neighborhood.neighbor_edge_iter() {
                let neighbor = neighbor_edge.neighbor_index();

                // Get flat index of neighbor on global grid
                let global_neighbor = subdomain.inv_map_point(neighbor).unwrap();
                let flat_neighbor_index = grid.flatten_point_index(&global_neighbor);

                // Try to read out the function value at the neighboring point
                let neighbor_value = if let Some(v) = density_map.get(flat_neighbor_index) {
                    v
                } else {
                    // Neighbors that are not in the point-value map were outside of the kernel evaluation radius.
                    // This should only happen for cells that are completely outside of the compact support of a particle.
                    // The point-value map has to be consistent such that for each cell, where at least one point-value
                    // is missing like this, the cell has to be completely below the iso-surface threshold.
                    continue;
                };

                // Skip edges that don't cross the iso-surface
                if !(neighbor_value > iso_surface_threshold) {
                    continue;
                }

                if !filter.process_edge(
                    density_map,
                    subdomain,
                    flat_point_index,
                    &point,
                    flat_neighbor_index,
                    neighbor,
                ) {
                    continue;
                }

                // Interpolate iso-surface vertex on the edge
                let alpha = (iso_surface_threshold - point_value) / (neighbor_value - point_value);
                let point_coords = subdomain_grid.point_coordinates(&point);
                let neighbor_coords = subdomain_grid.point_coordinates(neighbor);
                let interpolated_coords =
                    (point_coords) * (R::one() - alpha) + neighbor_coords * alpha;

                // Store interpolated vertex and remember its index
                let vertex_index = vertices.len();
                vertices.push(interpolated_coords);

                // Store the data required for the marching cubes triangulation for
                // each cell adjacent to the edge crossing the iso-surface.
                // This includes the above/below iso-surface flags and the interpolated vertex index.
                for cell in subdomain_grid
                    .cells_adjacent_to_edge(&neighbor_edge)
                    .iter()
                    .flatten()
                {
                    let global_cell = subdomain.inv_map_cell(cell).unwrap();
                    let flat_cell_index = grid.flatten_cell_index(&global_cell);

                    let cell_data_entry = cell_data
                        .entry(flat_cell_index)
                        .or_insert_with(CellData::default);

                    // Store the index of the interpolated vertex on the corresponding local edge of the cell
                    let local_edge_index = cell.local_edge_index_of(&neighbor_edge).unwrap();

                    // Ensure that there is not already data stored for this edge
                    assert!(
                        cell_data_entry.iso_surface_vertices[local_edge_index].is_none(),
                        "Overwriting already existing vertex. This is a bug."
                    );
                    cell_data_entry.iso_surface_vertices[local_edge_index] = Some(vertex_index);

                    // Mark the neighbor as above the iso-surface threshold
                    let local_vertex_index = cell.local_point_index_of(neighbor.index()).unwrap();
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

    filter
}

/// Loops through all corner vertices in the given marching cubes input and updates the above/below threshold flags
fn update_cell_data_threshold_flags<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
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

    let grid = subdomain.global_grid();

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

impl<I: Index, R: Real, S: Subdomain<I, R>> DensityMapFilter<I, R, S> for IdentityDensityMapFilter {
    #[inline(always)]
    fn process_point(
        &mut self,
        _density_map: &DensityMap<I, R>,
        _subdomain: &S,
        _flat_point_index: I,
        _subdomain_point: &PointIndex<I>,
        _point_value: R,
    ) -> bool {
        return true;
    }

    #[inline(always)]
    fn process_edge(
        &mut self,
        _density_map: &DensityMap<I, R>,
        _subdomain: &S,
        _flat_point_index: I,
        _subdomain_point: &PointIndex<I>,
        _flat_neighbor_index: I,
        _subdomain_neighbor: &PointIndex<I>,
    ) -> bool {
        return true;
    }
}

impl<I: Index, R: Real> SkipBoundaryLayerFilter<I, R> {
    /// Constructs a new empty filter
    fn new() -> Self {
        Self {
            boundary_density_maps: Default::default(),
        }
    }

    /// Consumes self and returns the collected boundary density maps
    fn into_inner(self) -> DirectedAxisArray<MapType<I, R>> {
        self.boundary_density_maps
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> DensityMapFilter<I, R, S>
    for SkipBoundaryLayerFilter<I, R>
{
    #[inline(always)]
    fn process_point(
        &mut self,
        density_map: &DensityMap<I, R>,
        subdomain: &S,
        flat_point_index: I,
        subdomain_point: &PointIndex<I>,
        point_value: R,
    ) -> bool {
        let grid = subdomain.global_grid();
        let subdomain_grid = subdomain.subdomain_grid();

        // Skip points directly at the boundary but add them to the respective boundary density map
        {
            let point_boundary_flags =
                GridBoundaryFaceFlags::classify_point(subdomain_grid, &subdomain_point);
            if !point_boundary_flags.is_empty() {
                // Insert the point into each boundary density map it belongs to
                for boundary in point_boundary_flags.iter_individual() {
                    let boundary_map = self.boundary_density_maps.get_mut(&boundary);
                    boundary_map.insert(flat_point_index, point_value);

                    // Also insert second row neighbor, if present
                    if let Some(flat_neighbor_index) = subdomain_grid
                        // Get neighbor in subdomain
                        .get_point_neighbor(&subdomain_point, boundary.opposite())
                        // Map neighbor from subdomain to global grid
                        .and_then(|neighbor| subdomain.inv_map_point(&neighbor))
                        // Flatten on global grid
                        .map(|global_neighbor| grid.flatten_point_index(&global_neighbor))
                    {
                        if let Some(density_value) = density_map.get(flat_neighbor_index) {
                            boundary_map.insert(flat_neighbor_index, density_value);
                        }
                    }
                }
                // Skip this point for interpolation
                return false;
            }
        }

        // Otherwise the point can be processed
        return true;
    }

    #[inline(always)]
    fn process_edge(
        &mut self,
        _density_map: &DensityMap<I, R>,
        subdomain: &S,
        _flat_point_index: I,
        _subdomain_point: &PointIndex<I>,
        _flat_neighbor_index: I,
        subdomain_neighbor: &PointIndex<I>,
    ) -> bool {
        let subdomain_grid = subdomain.subdomain_grid();

        // Check if the neighbor is on the boundary of the subdomain
        let point_boundary_flags =
            GridBoundaryFaceFlags::classify_point(subdomain_grid, subdomain_neighbor);
        let point_is_on_outer_boundary = !point_boundary_flags.is_empty();

        // Skip edges that go into the boundary layer
        if point_is_on_outer_boundary {
            return false;
        }

        return true;
    }
}

impl StitchingDomainNarrowBandFilter {
    /// Constructs a new filter for the narrow band of a stitching domain
    fn new(stitching_axis: Axis) -> Self {
        StitchingDomainNarrowBandFilter { stitching_axis }
    }

    /// Detects points that are on the positive/negative side of the stitching domain, along the stitching axis
    fn point_is_on_stitching_surface<I: Index, R: Real, S: Subdomain<I, R>>(
        &self,
        subdomain: &S,
        p: &PointIndex<I>,
    ) -> bool {
        let subdomain_grid = subdomain.subdomain_grid();

        let index = p.index();
        index[self.stitching_axis.dim()] == I::zero()
            || index[self.stitching_axis.dim()]
                == subdomain_grid.points_per_dim()[self.stitching_axis.dim()] - I::one()
    }

    /// Detects points that are on a boundary other than the stitching surfaces
    fn point_is_outside_stitching_volume<I: Index, R: Real, S: Subdomain<I, R>>(
        &self,
        subdomain: &S,
        p: &PointIndex<I>,
    ) -> bool {
        let subdomain_grid = subdomain.subdomain_grid();

        let index = p.index();
        self.stitching_axis
            .orthogonal_axes()
            .iter()
            .copied()
            .any(|axis| {
                index[axis.dim()] == I::zero()
                    || index[axis.dim()] == subdomain_grid.points_per_dim()[axis.dim()] - I::one()
            })
    }
}

impl<I: Index, R: Real, S: Subdomain<I, R>> DensityMapFilter<I, R, S>
    for StitchingDomainNarrowBandFilter
{
    #[inline(always)]
    fn process_point(
        &mut self,
        _density_map: &DensityMap<I, R>,
        subdomain: &S,
        _flat_point_index: I,
        subdomain_point: &PointIndex<I>,
        _point_value: R,
    ) -> bool {
        // Skip points on the outside of the stitching domain (except if they are on the stitching surface)
        if self.point_is_outside_stitching_volume(subdomain, subdomain_point) {
            return false;
        }

        true
    }

    #[inline(always)]
    fn process_edge(
        &mut self,
        _density_map: &DensityMap<I, R>,
        subdomain: &S,
        _flat_point_index: I,
        subdomain_point: &PointIndex<I>,
        _flat_neighbor_index: I,
        subdomain_neighbor: &PointIndex<I>,
    ) -> bool {
        // Skip edges that are on the stitching surface (were already triangulated by the patches)
        if self.point_is_on_stitching_surface(subdomain, subdomain_point)
            && self.point_is_on_stitching_surface(subdomain, subdomain_neighbor)
        {
            return false;
        }

        // Skip edges that go out of the stitching domain
        if self.point_is_outside_stitching_volume(subdomain, subdomain_neighbor) {
            return false;
        }

        true
    }
}
