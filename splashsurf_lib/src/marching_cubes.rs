use crate::marching_cubes::narrow_band_extraction::{
    construct_mc_input, construct_mc_input_with_stitching_data,
};
use crate::marching_cubes::triangulation::{
    triangulate, triangulate_with_criterion, DefaultTriangleGenerator,
    TriangulationSkipBoundaryCells,
};
use crate::mesh::TriMesh3d;
use crate::uniform_grid::{DummySubdomain, OwnedSubdomainGrid, Subdomain};
use crate::{new_map, DensityMap, Index, MapType, Real, UniformGrid};

pub mod marching_cubes_lut;
mod narrow_band_extraction;
mod stitching;
mod triangulation;

pub(crate) use stitching::{stitch_surface_patches, SurfacePatch};

/// Input data required by the marching cubes triangulation
#[derive(Clone, Debug)]
pub(crate) struct MarchingCubesInput<I: Index> {
    /// Data for all cells that are supposed to be triangulated by marching cubes
    pub(crate) cell_data: MapType<I, CellData>,
}

/// Flag indicating whether a vertex is above or below the iso-surface threshold
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum RelativeToThreshold {
    Below,
    Indeterminate,
    Above,
}

impl RelativeToThreshold {
    /// Returns if the value is above the iso-surface threshold, panics if the value is indeterminate
    fn is_above(&self) -> bool {
        match self {
            RelativeToThreshold::Below => false,
            RelativeToThreshold::Above => true,
            // TODO: Replace with error?
            RelativeToThreshold::Indeterminate => {
                panic!("Trying to evaluate cell data with indeterminate data!")
            }
        }
    }

    /// Returns if the value is above the iso-surface threshold or `None` if the value is indeterminate
    fn is_indeterminate(&self) -> bool {
        if let RelativeToThreshold::Indeterminate = self {
            true
        } else {
            false
        }
    }
}

/// Data for a single cell required by marching cubes
#[derive(Clone, Debug)]
pub(crate) struct CellData {
    /// The interpolated iso-surface vertex per local edge if the edge crosses the iso-surface
    pub(crate) iso_surface_vertices: [Option<usize>; 12],
    /// Flags indicating whether a corner vertex of the cell is above or below the iso-surface threshold
    pub(crate) corner_above_threshold: [RelativeToThreshold; 8],
}

impl CellData {
    /// Returns an boolean array indicating for each corner vertex of the cell whether it's above the iso-surface threshold
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

impl<I: Index> Default for MarchingCubesInput<I> {
    fn default() -> Self {
        Self {
            cell_data: new_map(),
        }
    }
}

/// Performs a marching cubes triangulation of a density map on the given background grid
pub fn triangulate_density_map<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
) -> TriMesh3d<R> {
    profile!("triangulate_density_map");

    let mut mesh = TriMesh3d::default();
    triangulate_density_map_append(grid, None, density_map, iso_surface_threshold, &mut mesh);
    mesh
}

/// Performs a marching cubes triangulation of a density map on the given background grid, appends triangles to the given mesh
pub fn triangulate_density_map_append<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    subdomain: Option<&OwnedSubdomainGrid<I, R>>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
    mesh: &mut TriMesh3d<R>,
) {
    profile!("triangulate_density_map_append");

    let marching_cubes_data = if let Some(subdomain) = subdomain {
        construct_mc_input(
            subdomain,
            &density_map,
            iso_surface_threshold,
            &mut mesh.vertices,
        )
    } else {
        let subdomain = DummySubdomain::new(grid);
        construct_mc_input(
            &subdomain,
            &density_map,
            iso_surface_threshold,
            &mut mesh.vertices,
        )
    };

    triangulate::<I, R>(marching_cubes_data, mesh);
}

/// Performs triangulation of the given density map to a surface patch
pub(crate) fn triangulate_density_map_to_surface_patch<I: Index, R: Real>(
    subdomain: &OwnedSubdomainGrid<I, R>,
    density_map: &DensityMap<I, R>,
    iso_surface_threshold: R,
) -> SurfacePatch<I, R> {
    profile!("triangulate_density_map_append");

    let mut mesh = TriMesh3d::default();
    let subdomain = subdomain.clone();

    assert!(
        subdomain.subdomain_grid().cells_per_dim().iter().all(|&n_cells| n_cells > I::one() + I::one()),
        "Interpolation procedure with stitching support only works on grids & subdomains with more than 2 cells in each dimension!"
    );

    let (marching_cubes_data, boundary_data) = construct_mc_input_with_stitching_data(
        &subdomain,
        &density_map,
        iso_surface_threshold,
        &mut mesh.vertices,
    );

    triangulate_with_criterion(
        &subdomain,
        marching_cubes_data,
        &mut mesh,
        TriangulationSkipBoundaryCells,
        DefaultTriangleGenerator,
    );

    SurfacePatch {
        mesh,
        subdomain,
        data: boundary_data,
        stitching_level: 0,
    }
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
            let point = cell.global_point_index_of(i).unwrap();
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
    use nalgebra::Vector3;
    let iso_surface_threshold = 0.25;
    //let default_value = 0.0;

    let mut trimesh = crate::TriMesh3d::default();
    let origin = Vector3::new(0.0, 0.0, 0.0);
    let n_cubes_per_dim = [1, 1, 1];
    let cube_size = 1.0;

    let grid = UniformGrid::<i32, f64>::new(&origin, &n_cubes_per_dim, cube_size).unwrap();

    assert_eq!(grid.aabb().max(), &Vector3::new(1.0, 1.0, 1.0));

    let mut sparse_data = new_map();

    let marching_cubes_data = {
        let subdomain = DummySubdomain::new(&grid);
        construct_mc_input(
            &subdomain,
            &sparse_data.clone().into(),
            iso_surface_threshold,
            &mut trimesh.vertices,
        )
    };

    assert_eq!(trimesh.vertices.len(), 0);
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

    let marching_cubes_data = {
        let subdomain = DummySubdomain::new(&grid);
        construct_mc_input(
            &subdomain,
            &sparse_data.clone().into(),
            iso_surface_threshold,
            &mut trimesh.vertices,
        )
    };

    assert_eq!(marching_cubes_data.cell_data.len(), 1);
    // Check that the correct number of vertices was created
    assert_eq!(trimesh.vertices.len(), 6);

    let cell = &marching_cubes_data.cell_data[&0];

    // Check that the correct vertices were marked as being below the iso-surface
    assert_eq!(
        cell.corner_above_threshold
            .iter()
            .map(|r| r.is_above())
            .collect::<Vec<_>>(),
        vec![false, true, true, true, false, false, true, false]
    );

    // Check that vertices were inserted at the correct edges
    assert!(cell.iso_surface_vertices[0].is_some());
    assert!(cell.iso_surface_vertices[3].is_some());
    assert!(cell.iso_surface_vertices[5].is_some());
    assert!(cell.iso_surface_vertices[6].is_some());
    assert!(cell.iso_surface_vertices[9].is_some());
    assert!(cell.iso_surface_vertices[11].is_some());

    // TODO: Continue writing test
    let _mesh = triangulate(marching_cubes_data, &mut trimesh);
    //println!("{:?}", mesh)
}
