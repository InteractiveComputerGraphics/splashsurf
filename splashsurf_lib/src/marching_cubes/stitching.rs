use crate::marching_cubes::narrow_band_extraction::update_mc_input_for_stitching_domain;
use crate::marching_cubes::triangulation::{
    triangulate_with_criterion, DefaultTriangleGenerator, TriangulationStitchingInterior,
};
use crate::marching_cubes::{CellData, MarchingCubesInput};
use crate::mesh::TriMesh3d;
use crate::topology::{Axis, DirectedAxis, DirectedAxisArray, Direction};
use crate::uniform_grid::{GridBoundaryFaceFlags, OwningSubdomainGrid, Subdomain};
use crate::{Index, MapType, Real, UniformGrid};
use log::{debug, trace};

/// Stitches the two given surface patches by triangulating the domain between them
pub(crate) fn stitch_surface_patches<I: Index, R: Real>(
    iso_surface_threshold: R,
    stitching_axis: Axis,
    mut negative_side: SurfacePatch<I, R>,
    mut positive_side: SurfacePatch<I, R>,
) -> SurfacePatch<I, R> {
    assert_eq!(
        negative_side.subdomain.global_grid(),
        positive_side.subdomain.global_grid(),
        "The global grid of the two subdomains that should be stitched is not identical!"
    );

    // Take out boundary data to satisfy borrow checker
    let mut negative_data = std::mem::take(&mut negative_side.data);
    let mut positive_data = std::mem::take(&mut positive_side.data);

    let global_grid = negative_side.subdomain.global_grid();

    debug!(
        "Stitching patches orthogonal to {:?}-axis. (-) side (offset: {:?}, cells_per_dim: {:?}, stitching_level: {:?}), (+) side (offset: {:?}, cells_per_dim: {:?}, stitching_level: {:?})",
        stitching_axis,
        negative_side.subdomain.subdomain_offset(),
        negative_side.subdomain.subdomain_grid().cells_per_dim(),
        negative_side.stitching_level,
        positive_side.subdomain.subdomain_offset(),
        positive_side.subdomain.subdomain_grid().cells_per_dim(),
        positive_side.stitching_level,
    );

    // Construct domain for the triangulation of the boundary layer between the sides
    let stitching_subdomain = compute_stitching_domain(
        stitching_axis,
        global_grid,
        &negative_side.subdomain,
        &positive_side.subdomain,
    );

    // Merge the two input meshes structures and get vertex offset for all vertices of the positive side
    let (mut output_mesh, negative_vertex_offset, positive_vertex_offset) = {
        let mut negative_mesh = std::mem::take(&mut negative_side.mesh);
        let mut positive_mesh = std::mem::take(&mut positive_side.mesh);

        if negative_mesh.vertices.len() > positive_mesh.vertices.len() {
            let positive_vertex_offset = negative_mesh.vertices.len();
            negative_mesh.append(&mut positive_mesh);
            (negative_mesh, None, Some(positive_vertex_offset))
        } else {
            let negative_vertex_offset = positive_mesh.vertices.len();
            positive_mesh.append(&mut negative_mesh);
            (positive_mesh, Some(negative_vertex_offset), None)
        }
    };

    // Merge the boundary data at the stitching boundaries of the two patches
    let merged_boundary_data = {
        // On the negative side we need the data of its positive boundary and vice versa
        let negative_data = std::mem::take(
            negative_data.get_mut(&DirectedAxis::new(stitching_axis, Direction::Positive)),
        );
        let positive_data = std::mem::take(
            positive_data.get_mut(&DirectedAxis::new(stitching_axis, Direction::Negative)),
        );

        // Merge the boundary layer density and cell data maps of the two sides
        merge_boundary_data(
            &stitching_subdomain,
            &negative_side.subdomain,
            negative_data,
            negative_vertex_offset,
            &positive_side.subdomain,
            positive_data,
            positive_vertex_offset,
        )
    };

    let BoundaryData {
        boundary_density_map,
        boundary_cell_data,
    } = merged_boundary_data;

    let mut marching_cubes_input = MarchingCubesInput {
        cell_data: boundary_cell_data,
    };

    // Perform marching cubes on the stitching domain
    let mut boundary_cell_data = {
        update_mc_input_for_stitching_domain(
            &stitching_subdomain,
            &boundary_density_map.into(),
            iso_surface_threshold,
            stitching_axis,
            &mut output_mesh.vertices,
            &mut marching_cubes_input,
        );

        // Collect the boundary cell data of the stitching domain
        let boundary_cell_data =
            collect_boundary_cell_data(&stitching_subdomain, &marching_cubes_input);

        triangulate_with_criterion(
            &stitching_subdomain,
            marching_cubes_input,
            &mut output_mesh,
            TriangulationStitchingInterior::new(stitching_axis),
            DefaultTriangleGenerator,
        );

        boundary_cell_data
    };

    // Get domain for the whole stitched domain
    let output_subdomain_grid = compute_stitching_result_domain(
        stitching_axis,
        global_grid,
        &negative_side.subdomain,
        &positive_side.subdomain,
    );

    // Merge all remaining boundary data
    let output_boundary_data = DirectedAxisArray::new_with(|&directed_axis| {
        // The positive and negative sides of the result domain can be taken directly from the inputs
        if directed_axis == stitching_axis.with_direction(Direction::Negative) {
            let data = std::mem::take(negative_data.get_mut(&directed_axis));
            data.map_to_domain(
                &output_subdomain_grid,
                &negative_side.subdomain,
                negative_vertex_offset,
            )
        } else if directed_axis == stitching_axis.with_direction(Direction::Positive) {
            let data = std::mem::take(positive_data.get_mut(&directed_axis));
            data.map_to_domain(
                &output_subdomain_grid,
                &positive_side.subdomain,
                positive_vertex_offset,
            )
        } else {
            // Otherwise, they have to be merged first
            let mut merged_data = merge_boundary_data(
                &output_subdomain_grid,
                &negative_side.subdomain,
                std::mem::take(negative_data.get_mut(&directed_axis)),
                negative_vertex_offset,
                &positive_side.subdomain,
                std::mem::take(positive_data.get_mut(&directed_axis)),
                positive_vertex_offset,
            );

            // Map cell indices from stitching domain to result domain and append to cell data map
            let current_boundary_cell_data =
                std::mem::take(boundary_cell_data.get_mut(&directed_axis));
            for (flat_cell_index, cell_data) in current_boundary_cell_data {
                let global_cell = global_grid
                    .try_unflatten_cell_index(flat_cell_index)
                    .unwrap();

                // Skip cells not part of the output grid
                if output_subdomain_grid.map_cell(&global_cell).is_none() {
                    continue;
                }

                merged_data
                    .boundary_cell_data
                    .entry(flat_cell_index)
                    // Merge with existing cell data
                    .and_modify(|existing_cell_data| {
                        // Should be fine to just replace these values as they will be overwritten anyway in the next stitching process
                        existing_cell_data.corner_above_threshold =
                            cell_data.corner_above_threshold;
                        // For the iso-surface vertices we need the union
                        for (existing_vertex, new_vertex) in existing_cell_data
                            .iso_surface_vertices
                            .iter_mut()
                            .zip(cell_data.iso_surface_vertices.iter())
                        {
                            if existing_vertex != new_vertex {
                                assert!(
                                    existing_vertex.is_none(),
                                    "Overwriting already existing vertex. This is a bug."
                                );
                                *existing_vertex = *new_vertex
                            }
                        }
                    })
                    // Or insert new cell data
                    .or_insert_with(|| cell_data.clone());
            }

            merged_data
        }
    });

    SurfacePatch {
        subdomain: output_subdomain_grid,
        mesh: output_mesh,
        data: output_boundary_data,
        stitching_level: negative_side
            .stitching_level
            .max(positive_side.stitching_level),
    }
}

/// A surface patch representing a local part of a larger surface reconstruction
#[derive(Clone, Debug)]
pub(crate) struct SurfacePatch<I: Index, R: Real> {
    /// The local surface mesh of this side
    pub(crate) mesh: TriMesh3d<R>,
    /// The subdomain of this local mesh
    pub(crate) subdomain: OwningSubdomainGrid<I, R>,
    /// All additional data required for stitching
    pub(crate) data: DirectedAxisArray<BoundaryData<I, R>>,
    /// The maximum number of times parts of this patch where stitched together
    pub(crate) stitching_level: usize,
}

impl<I: Index, R: Real> SurfacePatch<I, R> {
    /// Creates an empty surface patch for the given subdomain
    pub(crate) fn new_empty(subdomain: OwningSubdomainGrid<I, R>) -> Self {
        Self {
            mesh: Default::default(),
            subdomain,
            data: Default::default(),
            stitching_level: 0,
        }
    }
}

/// Stitching data per boundary
#[derive(Clone, Default, Debug)]
pub(crate) struct BoundaryData<I: Index, R: Real> {
    /// The density map for all vertices of this boundary
    pub(crate) boundary_density_map: MapType<I, R>,
    /// The cell data for all cells of this boundary
    pub(crate) boundary_cell_data: MapType<I, CellData>,
}

/// Extracts the cell data of all cells on the boundary of the subdomain
#[inline(never)]
pub(crate) fn collect_boundary_cell_data<I: Index, R: Real, S: Subdomain<I, R>>(
    subdomain: &S,
    input: &MarchingCubesInput<I>,
) -> DirectedAxisArray<MapType<I, CellData>> {
    let mut boundary_cell_data: DirectedAxisArray<MapType<I, CellData>> = Default::default();

    for (&flat_cell_index, cell_data) in &input.cell_data {
        let global_cell = subdomain
            .global_grid()
            .try_unflatten_cell_index(flat_cell_index)
            .unwrap();
        let local_cell = subdomain.map_cell(&global_cell).unwrap();

        // Check which grid boundary faces this cell is part of
        let cell_grid_boundaries =
            GridBoundaryFaceFlags::classify_cell(subdomain.subdomain_grid(), &local_cell);
        // Only process cells that are part of some boundary
        if !cell_grid_boundaries.is_empty() {
            for boundary in cell_grid_boundaries.iter_individual() {
                boundary_cell_data
                    .get_mut(&boundary)
                    .insert(flat_cell_index, cell_data.clone());
            }
        }
    }

    boundary_cell_data
}

impl<I: Index, R: Real> BoundaryData<I, R> {
    /// Maps this boundary data to another domain by converting all indices to the new subdomain
    fn map_to_domain(
        mut self,
        target_domain: &OwningSubdomainGrid<I, R>,
        source_domain: &OwningSubdomainGrid<I, R>,
        source_offset: Option<usize>,
    ) -> Self {
        assert_eq!(
            target_domain.global_grid(),
            source_domain.global_grid(),
            "The global grid of target and source domain has to match!"
        );
        let grid = target_domain.global_grid();

        // Process density map
        {
            let mut points_to_remove = Vec::new();
            for flat_point_index in self.boundary_density_map.keys().copied() {
                let global_point = grid.try_unflatten_point_index(flat_point_index).unwrap();

                // If point is not part of the target domain, it is marked to be removed
                if target_domain.map_point(&global_point).is_none() {
                    points_to_remove.push(flat_point_index)
                }
            }

            // Remove all points not part of the target domain
            for flat_point_index in points_to_remove {
                self.boundary_density_map.remove(&flat_point_index);
            }
        }

        // Process cell data
        {
            let mut cells_to_remove = Vec::new();
            for flat_cell_index in self.boundary_cell_data.keys().copied() {
                let global_cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();

                // If cell is not part of the target domain, it is marked to be removed
                if target_domain.map_cell(&global_cell).is_none() {
                    cells_to_remove.push(flat_cell_index);
                }
            }

            // Remove all cells not part of the target domain
            for flat_cell_index in cells_to_remove {
                self.boundary_cell_data.remove(&flat_cell_index);
            }
        }

        // Apply vertex offset if required
        self.apply_cell_data_vertex_offset(source_offset);

        self
    }

    /// Merges another boundary data object into this object
    fn merge_with(
        mut self,
        target_domain: &OwningSubdomainGrid<I, R>,
        mut other: BoundaryData<I, R>,
        other_domain: &OwningSubdomainGrid<I, R>,
        other_vertex_offset: Option<usize>,
    ) -> Self {
        assert_eq!(
            target_domain.global_grid(),
            other_domain.global_grid(),
            "The global grid of target and source domain has to match!"
        );
        let grid = target_domain.global_grid();

        // Apply vertex offset if required
        other.apply_cell_data_vertex_offset(other_vertex_offset);

        let BoundaryData {
            boundary_density_map: other_boundary_density_map,
            boundary_cell_data: other_boundary_cell_data,
        } = other;

        // Process density map
        for (flat_point_index, density_contribution) in other_boundary_density_map {
            let global_point = grid.try_unflatten_point_index(flat_point_index).unwrap();

            // Skip points that are not part of the target domain
            if target_domain.map_point(&global_point).is_none() {
                continue;
            }

            // TODO: For a proper average we would have to keep track how often this point is merged with other domains
            //  and use an incremental average over this number of contributions
            self.boundary_density_map
                .entry(flat_point_index)
                // Compute average with existing value
                .and_modify(|density| {
                    *density += density_contribution;
                    *density /= R::one() + R::one();
                })
                // Or just insert the new value
                .or_insert(density_contribution);
        }

        // Process cell data
        for (flat_cell_index, cell_data) in other_boundary_cell_data {
            let global_cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();

            // Skip points that are not part of the target domain
            if target_domain.map_cell(&global_cell).is_none() {
                continue;
            }

            self.boundary_cell_data
                .entry(flat_cell_index)
                // The cell data interpolation function should only populate cells that are part of their subdomain
                .and_modify(|_| {
                    panic!("Merge conflict: there is duplicate cell data for this cell index")
                })
                // Otherwise insert the additional cell data
                .or_insert(cell_data);
        }

        self
    }

    /// Adds an offset to all iso-surface vertex indices stored in the cell data map
    fn apply_cell_data_vertex_offset(&mut self, vertex_offset: Option<usize>) {
        // Apply the vertex offset
        if let Some(vertex_offset) = vertex_offset {
            for v in self
                .boundary_cell_data
                .iter_mut()
                // Iterate over all vertices in the cell data
                .flat_map(|(_, cell_data)| cell_data.iso_surface_vertices.iter_mut())
                // Skip missing vertices
                .flatten()
            {
                *v += vertex_offset;
            }
        }
    }
}

/// Merges boundary such that only density values and cell data in the result subdomain are part of the result
fn merge_boundary_data<I: Index, R: Real>(
    target_subdomain: &OwningSubdomainGrid<I, R>,
    negative_subdomain: &OwningSubdomainGrid<I, R>,
    negative_data: BoundaryData<I, R>,
    negative_vertex_offset: Option<usize>,
    positive_subdomain: &OwningSubdomainGrid<I, R>,
    positive_data: BoundaryData<I, R>,
    positive_vertex_offset: Option<usize>,
) -> BoundaryData<I, R> {
    trace!("Merging boundary data. Size of containers: (-) side (density map: {}, cell data map: {}), (+) side (density map: {}, cell data map: {})", negative_data.boundary_density_map.len(), negative_data.boundary_cell_data.len(), positive_data.boundary_density_map.len(), positive_data.boundary_cell_data.len());

    let negative_len =
        negative_data.boundary_density_map.len() + negative_data.boundary_cell_data.len();
    let positive_len =
        positive_data.boundary_density_map.len() + positive_data.boundary_cell_data.len();

    let merged_boundary_data = if negative_len > positive_len {
        let merged_boundary_data = negative_data.map_to_domain(
            target_subdomain,
            negative_subdomain,
            negative_vertex_offset,
        );
        merged_boundary_data.merge_with(
            target_subdomain,
            positive_data,
            positive_subdomain,
            positive_vertex_offset,
        )
    } else {
        let merged_boundary_data = positive_data.map_to_domain(
            target_subdomain,
            positive_subdomain,
            positive_vertex_offset,
        );
        merged_boundary_data.merge_with(
            target_subdomain,
            negative_data,
            negative_subdomain,
            negative_vertex_offset,
        )
    };

    trace!("Finished merging boundary data. Size of containers: result (density map: {}, cell data map: {})", merged_boundary_data.boundary_density_map.len(), merged_boundary_data.boundary_cell_data.len());

    merged_boundary_data
}

/// Computes the [OwningSubdomainGrid] for stitching region between the two sides that has to be triangulated
fn compute_stitching_domain<I: Index, R: Real>(
    stitching_axis: Axis,
    global_grid: &UniformGrid<I, R>,
    negative_subdomain: &OwningSubdomainGrid<I, R>,
    positive_subdomain: &OwningSubdomainGrid<I, R>,
) -> OwningSubdomainGrid<I, R> {
    // Ensure that global grids are equivalent
    assert_eq!(
        negative_subdomain.global_grid(),
        global_grid,
        "The global grid of the two subdomains that should be stitched is not identical!"
    );
    assert_eq!(
        positive_subdomain.global_grid(),
        global_grid,
        "The global grid of the two subdomains that should be stitched is not identical!"
    );

    // Check that the two domains actually meet
    {
        // Starting at the offset of the negative subdomain and going along the stitching axis...
        let lower_corner_end = stitching_axis
            .with_direction(Direction::Positive)
            .checked_apply_step_ijk(
                negative_subdomain.subdomain_offset(),
                negative_subdomain.subdomain_grid().cells_per_dim(),
            )
            .expect("Index type out of range?");

        // ...we should arrive at the lower corner of the positive side
        assert_eq!(
            lower_corner_end,
            *positive_subdomain.subdomain_offset(),
            "The two subdomains that should be stitched do not meet directly!"
        );
    }

    // Get the number of cells of the stitching domain
    let n_cells_per_dim = {
        let mut n_cells_per_dim_neg = negative_subdomain.subdomain_grid().cells_per_dim().clone();
        let mut n_cells_per_dim_pos = positive_subdomain.subdomain_grid().cells_per_dim().clone();

        // Between the two subdomains are only two layers of cells
        n_cells_per_dim_neg[stitching_axis.dim()] = I::one() + I::one();
        n_cells_per_dim_pos[stitching_axis.dim()] = I::one() + I::one();

        // Ensure that the stitching domain is identical from both sides
        assert_eq!(
            n_cells_per_dim_neg, n_cells_per_dim_pos,
            "The cross sections of the two subdomains that should be stitched is not identical!"
        );

        let n_cells_per_dim = n_cells_per_dim_neg;
        n_cells_per_dim
    };

    // Obtain the index of the lower corner of the stitching domain
    let stitching_grid_offset = {
        let axis_index = stitching_axis.dim();

        // Start at offset of negative domain
        let mut stitching_grid_offset = negative_subdomain.subdomain_offset().clone();

        // Go to the end of the negative domain along the stitching axis
        stitching_grid_offset[axis_index] +=
            negative_subdomain.subdomain_grid().cells_per_dim()[axis_index];
        // Subtract one because the stitching domain starts in the boundary layer
        stitching_grid_offset[axis_index] -= I::one();
        stitching_grid_offset
    };
    // Get coordinates of offset point
    let lower_corner_coords = global_grid.point_coordinates_array(&stitching_grid_offset);

    // Build the grid for the stitching domain
    let stitching_grid = UniformGrid::new(
        &lower_corner_coords,
        &n_cells_per_dim,
        global_grid.cell_size(),
    )
    .expect("Unable to construct stitching domain grid");

    trace!(
        "Constructed domain for stitching. offset: {:?}, cells_per_dim: {:?}",
        stitching_grid_offset,
        n_cells_per_dim
    );

    OwningSubdomainGrid::new(global_grid.clone(), stitching_grid, stitching_grid_offset)
}

/// Computes the [OwningSubdomainGrid] for the final combined domain of the two sides
fn compute_stitching_result_domain<I: Index, R: Real>(
    stitching_axis: Axis,
    global_grid: &UniformGrid<I, R>,
    negative_subdomain: &OwningSubdomainGrid<I, R>,
    positive_subdomain: &OwningSubdomainGrid<I, R>,
) -> OwningSubdomainGrid<I, R> {
    // Get the number of cells of the result domain by adding all cells in stitching direction
    let n_cells_per_dim = {
        let length_neg = negative_subdomain.subdomain_grid().cells_per_dim()[stitching_axis.dim()];
        let length_pos = positive_subdomain.subdomain_grid().cells_per_dim()[stitching_axis.dim()];

        let mut n_cells_per_dim = negative_subdomain.subdomain_grid().cells_per_dim().clone();
        n_cells_per_dim[stitching_axis.dim()] = length_neg + length_pos;

        n_cells_per_dim
    };

    // Construct the grid
    let subdomain_grid = UniformGrid::new(
        &negative_subdomain.subdomain_grid().aabb().min(),
        &n_cells_per_dim,
        global_grid.cell_size(),
    )
    .expect("Unable to construct stitching domain grid");

    OwningSubdomainGrid::new(
        global_grid.clone(),
        subdomain_grid,
        negative_subdomain.subdomain_offset().clone(),
    )
}
