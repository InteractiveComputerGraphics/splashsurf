use coarse_prof::profile;
use log::info;
use na::Vector3;

use crate::marching_cubes_lut::get_marching_cubes_triangulation;
use crate::mesh::TriMesh3d;
use crate::{new_map, DensityMap, Index, MapType, Real, UniformGrid};

pub fn triangulate_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
) -> TriMesh3d<R> {
    profile!("triangulate_density_map");
    let marching_cubes_data =
        interpolate_points_to_cell_data::<I, R>(&grid, &density_map, iso_surface_threshold);
    triangulate::<I, R>(marching_cubes_data)
}

#[derive(Copy, Clone, Debug)]
enum RelativeToThreshold {
    Below,
    Indeterminate,
    Above,
}

impl RelativeToThreshold {
    fn is_above(&self) -> bool {
        match self {
            RelativeToThreshold::Below => false,
            RelativeToThreshold::Above => true,
            // TODO: Replace with error?
            RelativeToThreshold::Indeterminate => panic!(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CellData {
    iso_surface_vertices: [Option<usize>; 12],
    corner_above_threshold: [RelativeToThreshold; 8],
}

impl CellData {
    fn are_vertices_above(&self) -> [bool; 8] {
        [
            self.corner_above_threshold[0].is_above(),
            self.corner_above_threshold[1].is_above(),
            self.corner_above_threshold[2].is_above(),
            self.corner_above_threshold[3].is_above(),
            self.corner_above_threshold[4].is_above(),
            self.corner_above_threshold[5].is_above(),
            self.corner_above_threshold[6].is_above(),
            self.corner_above_threshold[7].is_above(),
        ]
    }
}

impl Default for CellData {
    fn default() -> Self {
        CellData {
            iso_surface_vertices: [None; 12],
            corner_above_threshold: [RelativeToThreshold::Indeterminate; 8],
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MarchingCubesInput<I: Index, R: Real> {
    vertices: Vec<Vector3<R>>,
    cell_data: MapType<I, CellData>,
}

#[inline(never)]
pub(crate) fn interpolate_points_to_cell_data<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
) -> MarchingCubesInput<I, R> {
    profile!("interpolate_points_to_cell_data");

    // Note: This functions assumes that the default value for missing point data is below the iso-surface threshold
    info!("Starting interpolation of cell data for marching cubes...");

    // Map from flat cell index to all data that is required per cell for the marching cubes triangulation
    let mut cell_data: MapType<I, CellData> = new_map();
    // Storage for vertices that are created on edges crossing the iso-surface
    let mut vertices = Vec::new();

    // Generate iso-surface vertices and identify affected cells & edges
    {
        profile!("generate_iso_surface_vertices");
        for (flat_point_index, point_value) in density_map.iter() {
            // Skip grid points with values above the surface threshold
            if point_value > iso_surface_threshold {
                continue;
            }

            let point = grid.try_unflatten_point_index(flat_point_index).unwrap();
            let neighborhood = grid.get_point_neighborhood(&point);

            // Iterate over all neighbors of the point
            for neighbor_edge in neighborhood.neighbor_edge_iter() {
                let neighbor = neighbor_edge.neighbor_index();

                let flat_neighbor_index = grid.flatten_point_index(neighbor);
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

                // Check if an edge crossing the iso-surface was found
                if neighbor_value > iso_surface_threshold {
                    // Interpolate iso-surface vertex
                    let alpha =
                        (iso_surface_threshold - point_value) / (neighbor_value - point_value);
                    let point_coords = grid.point_coordinates(&point);
                    let neighbor_coords = grid.point_coordinates(neighbor);
                    let interpolated_coords =
                        (point_coords) * (R::one() - alpha) + neighbor_coords * alpha;

                    // Store interpolated vertex and remember its index
                    let vertex_index = vertices.len();
                    vertices.push(interpolated_coords);

                    // Iterate over all cells adjacent to the edge
                    for cell in grid.cells_adjacent_to_edge(&neighbor_edge).iter().flatten() {
                        let flat_cell_index = grid.flatten_cell_index(cell);

                        let mut cell_data_entry = cell_data
                            .entry(flat_cell_index)
                            .or_insert_with(CellData::default);

                        // Store the index of the interpolated vertex on the corresponding local edge of the cell
                        let local_edge_index = cell.local_edge_index_of(&neighbor_edge).unwrap();
                        cell_data_entry.iso_surface_vertices[local_edge_index] = Some(vertex_index);

                        // Mark the neighbor as above the iso-surface threshold
                        let local_vertex_index =
                            cell.local_point_index_of(neighbor.index()).unwrap();
                        cell_data_entry.corner_above_threshold[local_vertex_index] =
                            RelativeToThreshold::Above;
                    }
                }
            }
        }
    }

    // Cell corner points above the iso-surface threshold which are only surrounded by neighbors that
    // are also above the threshold were not marked as `corner_above_threshold = true` before, because they
    // don't have any adjacent edge crossing the iso-surface (and thus were never touched by the point data loop).
    //
    // Therefore, we have to loop over all corner points of all cells and check again their grid density value
    //
    // Note that we would also have this problem if we flipped the default value of corner_above_threshold
    // to false. In this case we could also move this into the point data loop (which might increase performance).
    // However, we would have to special case cells without point data, which are currently skipped.
    // Similarly, they have to be treated in a second pass because we don't want to initialize cells only
    // consisting of missing points and points below the surface.
    {
        profile!("relative_to_threshold_postprocessing");
        for (&flat_cell_index, cell_data) in cell_data.iter_mut() {
            let cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();
            for (local_point_index, flag_above) in
                cell_data.corner_above_threshold.iter_mut().enumerate()
            {
                // If the point is already marked as above we can ignore it
                if let RelativeToThreshold::Above = flag_above {
                    continue;
                }

                // Otherwise try to look up its value and potentially mark it as above the threshold
                let point = cell.global_index_of(local_point_index).unwrap();
                let flat_point_index = grid.flatten_point_index(&point);
                if let Some(point_value) = density_map.get(flat_point_index) {
                    if point_value > iso_surface_threshold {
                        *flag_above = RelativeToThreshold::Above;
                    } else {
                        *flag_above = RelativeToThreshold::Below;
                    }
                } else {
                    *flag_above = RelativeToThreshold::Below;
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    assert_cell_data_point_data_consistency(density_map, &cell_data, grid, iso_surface_threshold);

    info!(
        "Generated cell data for marching cubes with {} cells and {} vertices.",
        cell_data.len(),
        vertices.len()
    );
    info!("Interpolation done.");

    MarchingCubesInput {
        vertices,
        cell_data,
    }
}

#[inline(never)]
pub(crate) fn triangulate<I: Index, R: Real>(input: MarchingCubesInput<I, R>) -> TriMesh3d<R> {
    profile!("triangulate");

    let MarchingCubesInput {
        vertices,
        cell_data,
    } = input;

    info!(
        "Starting marching cubes triangulation of {} cells...",
        cell_data.len()
    );

    let mut mesh = TriMesh3d {
        vertices,
        triangles: Vec::new(),
    };

    // Triangulate affected cells
    for (&_flat_cell_index, cell_data) in &cell_data {
        let triangles = get_marching_cubes_triangulation(&cell_data.are_vertices_above());

        for triangle in triangles.iter().flatten() {
            let global_triangle = [
                cell_data.iso_surface_vertices[triangle[0] as usize].unwrap(),
                cell_data.iso_surface_vertices[triangle[1] as usize].unwrap(),
                cell_data.iso_surface_vertices[triangle[2] as usize].unwrap(),
            ];
            mesh.triangles.push(global_triangle);
        }
    }

    info!(
        "Generated surface mesh with {} triangles and {} vertices.",
        mesh.triangles.len(),
        mesh.vertices.len()
    );
    info!("Triangulation done.");

    mesh
}

#[allow(unused)]
#[inline(never)]
fn assert_cell_data_point_data_consistency<I: Index, R: Real>(
    density_map: &DensityMap<I, R>,
    cell_data: &MapType<I, CellData>,
    grid: &UniformGrid<I, R>,
    iso_surface_threshold: R,
) {
    // Check for each cell that if it has a missing point value, then it is has no other
    // iso-surface crossing edges / vertices
    for (&flat_cell_index, cell_data) in cell_data {
        let mut has_missing_point_data = false;
        let mut has_point_data_above_threshold = false;

        let cell = grid.try_unflatten_cell_index(flat_cell_index).unwrap();
        for i in 0..8 {
            let point = cell.global_index_of(i).unwrap();
            let flat_point_index = grid.flatten_point_index(&point);
            if let Some(point_value) = density_map.get(flat_point_index) {
                if point_value > iso_surface_threshold {
                    has_point_data_above_threshold = true;
                }
            } else {
                has_missing_point_data = true;
            }
        }

        assert!(!(has_missing_point_data && has_point_data_above_threshold));

        let mut has_point_above_threshold = false;
        for &flag_above in cell_data.corner_above_threshold.iter() {
            if let RelativeToThreshold::Above = flag_above {
                has_point_above_threshold = true;
            }
        }

        assert!(!(has_missing_point_data && has_point_above_threshold));

        let mut has_iso_surface_vertex = false;
        for vertex in cell_data.iso_surface_vertices.iter() {
            if vertex.is_some() {
                has_iso_surface_vertex = true;
            }
        }

        assert!(!(has_missing_point_data && has_iso_surface_vertex));
    }
}

#[test]
fn test_interpolate_cell_data() {
    let iso_surface_threshold = 0.25;
    let default_value = 0.0;

    let origin = Vector3::new(0.0, 0.0, 0.0);
    let n_cubes_per_dim = [1, 1, 1];
    let cube_size = 1.0;

    let grid = UniformGrid::<i32, f64>::new(&origin, &n_cubes_per_dim, cube_size);

    assert_eq!(grid.aabb().max(), &Vector3::new(1.0, 1.0, 1.0));

    let mut sparse_data = MapType::new();

    let marching_cubes_data =
        interpolate_points_to_cell_data(&sparse_data, &grid, iso_surface_threshold, default_value);

    assert_eq!(marching_cubes_data.vertices.len(), 0);
    assert_eq!(marching_cubes_data.cell_data.len(), 0);

    let points = vec![
        ([0, 0, 0], 0.0),
        ([1, 0, 0], 0.75),
        ([1, 1, 0], 1.0),
        ([0, 1, 0], 0.5),
        ([0, 0, 1], 0.0),
        ([1, 0, 1], 0.0),
        ([1, 1, 1], 1.0),
        ([0, 1, 1], 0.0),
    ];

    for (ijk, val) in points {
        sparse_data.insert(grid.flatten_point_index_array(&ijk), val);
    }

    let marching_cubes_data =
        interpolate_points_to_cell_data(&sparse_data, &grid, iso_surface_threshold, default_value);

    assert_eq!(marching_cubes_data.cell_data.len(), 1);
    // Check that the correct number of vertices was created
    assert_eq!(marching_cubes_data.vertices.len(), 6);

    let cell = &marching_cubes_data.cell_data[&0];

    // Check that the correct vertices were marked as being below the iso-surface
    assert_eq!(
        cell.corner_above_threshold,
        [false, true, true, true, false, false, true, false]
    );

    // Check that vertices were instered at the correct edges
    assert!(cell.iso_surface_vertices[0].is_some());
    assert!(cell.iso_surface_vertices[3].is_some());
    assert!(cell.iso_surface_vertices[5].is_some());
    assert!(cell.iso_surface_vertices[6].is_some());
    assert!(cell.iso_surface_vertices[9].is_some());
    assert!(cell.iso_surface_vertices[11].is_some());

    // TODO: Continue writing test
    let _mesh = triangulate(marching_cubes_data);
    //println!("{:?}", mesh)
}
