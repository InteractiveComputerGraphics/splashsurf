//! Example with a naive marching cubes implementation that reconstructs a level set (e.g. zero level from an SDF)

use anyhow::anyhow;
use sdfu::SDF;
use splashsurf_lib::Aabb3d;
use splashsurf_lib::Real;
use splashsurf_lib::io;
use splashsurf_lib::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use splashsurf_lib::mesh::TriMesh3d;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::uniform_grid::UniformCartesianCubeGrid3d;
use std::collections::HashMap;
use ultraviolet::vec::Vec3;

pub enum LevelSetSign {
    Inside,
    Outside,
}

/// Interface that has to be provided by a level set to be reconstructed using marching cubes
pub trait MarchingCubesLevelSet<R: Real> {
    /// Returns whether the level set may be supported within the given AABB.
    ///
    /// This function is used by the marching cubes algorithm to quickly discard regions of the domain
    /// that will not contain any triangles in the final triangulation (i.e. regions that are not part
    /// of the support of the level set).
    ///
    /// Therefore an implementation should conservatively return `true` and only return `false` if it
    /// can be guaranteed that the region is not part of the support. Consequently an implementation
    /// should:
    ///  - return `true` if part of the region is guaranteed to be part of the support
    ///  - return `true` if it *cannot* be ruled out that the region contains parts of the support of the level set
    ///  - return `false` if it can be guaranteed that level set is not supported at all in the region
    ///
    fn is_region_supported(&self, aabb: &Aabb3d<R>) -> bool;
    /// Returns the sign of the level set function at the given coordinate
    fn evaluate_sign(&self, coordinate: &Vector3<R>) -> LevelSetSign;
    /// Returns the value of the level set function at the given coordinate
    fn evaluate(&self, coordinate: &Vector3<R>) -> R;
}

/// Reconstructs a triangle mesh from a level set function
pub fn marching_cubes<R: Real, L: MarchingCubesLevelSet<R>>(
    level_set: L,
    domain: &Aabb3d<R>,
    cube_size: R,
) -> Result<TriMesh3d<R>, anyhow::Error> {
    let mut mesh = TriMesh3d::default();

    let is_supported = level_set.is_region_supported(domain);
    if !is_supported {
        return Ok(mesh);
    }

    let mut edge_to_vertex = HashMap::new();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    let grid = UniformCartesianCubeGrid3d::from_aabb(domain, cube_size)?;
    let n_cells = grid.cells_per_dim();
    for i in 0..n_cells[0] {
        for j in 0..n_cells[1] {
            for k in 0..n_cells[2] {
                let cell = grid.get_cell([i, j, k]).ok_or(anyhow!("Missing cell!"))?;

                let mut vertices_inside = [true; 8];
                for local_point_index in 0..8 {
                    let point = cell
                        .global_point_index_of(local_point_index)
                        .ok_or_else(|| anyhow!("Missing point!"))?;
                    let coords = grid.point_coordinates(&point);

                    match level_set.evaluate_sign(&coords) {
                        LevelSetSign::Inside => vertices_inside[local_point_index] = true,
                        LevelSetSign::Outside => vertices_inside[local_point_index] = false,
                    }
                }

                for triangle in
                    marching_cubes_triangulation_iter(&vertices_inside).collect::<Vec<_>>()
                {
                    let mut global_triangle = [0; 3];
                    for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
                        let edge = cell
                            .global_edge_index_of(local_edge_index as usize)
                            .ok_or_else(|| anyhow!("Missing edge!"))?;
                        let vertex_index = *edge_to_vertex.entry(edge).or_insert_with(|| {
                            let vertex_index = vertices.len();

                            let origin_coords = grid.point_coordinates(edge.origin());
                            let target_coords = grid.point_coordinates(&edge.target());

                            let origin_value = level_set.evaluate(&origin_coords);
                            let target_value = level_set.evaluate(&target_coords);

                            let diff = target_value - origin_value;
                            let c = (origin_value / diff).abs();
                            let p = origin_coords + (target_coords - origin_coords) * c;

                            let vertex_coords = p;
                            vertices.push(vertex_coords);
                            vertex_index
                        });
                        global_triangle[v_idx] = vertex_index;
                    }
                    triangles.push(global_triangle);
                }
            }
        }
    }

    mesh.vertices = vertices;
    mesh.triangles = triangles;

    Ok(mesh)
}

/// Wrapper for an SDF from sdfu
pub struct SdfuLevelSet<S: SDF<f32, Vec3>> {
    sdf: S,
}

/// Converts an nalgebra Vector to an ultraviolet Vec3
fn vec_na_to_uv(vec: &Vector3<f32>) -> Vec3 {
    Vec3::new(vec.x, vec.y, vec.z)
}

impl<S> MarchingCubesLevelSet<f32> for SdfuLevelSet<S>
where
    S: SDF<f32, Vec3>,
{
    fn is_region_supported(&self, aabb: &Aabb3d<f32>) -> bool {
        let min = aabb.min();
        let max = aabb.max();
        let diag = (max - min).norm();

        // Check distance value of the SDF at the center of the AABB
        // If it's less than the AABBs diagonal, the level set is supported inside of the AABB
        let centroid = aabb.centroid();
        let center_dist = self.sdf.dist(vec_na_to_uv(&centroid));

        center_dist <= diag / 2.0
    }

    fn evaluate_sign(&self, coordinate: &Vector3<f32>) -> LevelSetSign {
        let dist = self
            .sdf
            .dist(Vec3::new(coordinate.x, coordinate.y, coordinate.z));
        if dist > 0.0 {
            LevelSetSign::Outside
        } else {
            LevelSetSign::Inside
        }
    }

    fn evaluate(&self, coordinate: &Vector3<f32>) -> f32 {
        self.sdf.dist(vec_na_to_uv(coordinate))
    }
}

/// Builds an example level set function to reconstruct
fn example_level_set() -> SdfuLevelSet<impl SDF<f32, Vec3>> {
    let sdf = sdfu::Sphere::new(0.5f32)
        .union_smooth(
            sdfu::Sphere::new(0.3).translate(Vec3::new(-0.3, 0.3, 0.0)),
            0.1,
        )
        .subtract(
            sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(-0.3, 0.3, 0.0)),
        );
    SdfuLevelSet { sdf }
}

fn main() -> Result<(), anyhow::Error> {
    let level_set = example_level_set();

    let domain = Aabb3d::new(Vector3::new(-1.5, -1.5, -1.5), Vector3::new(1.5, 1.5, 1.5));
    let cube_size = 0.05;

    let mesh = marching_cubes(level_set, &domain, cube_size)?;

    println!(
        "Vertices: {} triangles: {}",
        mesh.vertices.len(),
        mesh.triangles.len()
    );

    #[cfg(feature = "io")]
    io::vtk_format::write_vtk(&mesh, "out/sdf_test.vtk", "mesh")?;

    Ok(())
}
