use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use splashsurf_lib::{mesh::TriMesh3d, nalgebra::Vector3, postprocessing::marching_cubes_cleanup, uniform_grid::UniformCartesianCubeGrid3d};


#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup")]
#[pyo3(signature = (triangles, vertices, grid_info, *, max_iter, keep_vertices))]
pub fn marching_cubes_cleanup_py<'py>(
    triangles: PyReadonlyArray2<usize>,
    vertices: PyReadonlyArray2<f64>,
    grid_info: ([f64; 3], [f64; 3], f64, [i64; 3], [i64; 3]),
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    let mut mesh = {
        let mut res = TriMesh3d::<f64>::default();

        let verts = {
            let binding = vertices.as_array();
            let verts: ndarray::ArrayBase<ndarray::CowRepr<'_, f64>, ndarray::Dim<[usize; 1]>> =
                binding.flatten();
            let mut res = Vec::new();
            let mut i = 0;
            while i < verts.len() {
                res.push(Vector3::new(verts[i], verts[i + 1], verts[i + 2]));
                i += 3;
            }
            res
        };

        let tris = {
            let binding = triangles.as_array();
            let tris = binding.flatten();
            let mut res = Vec::new();
            let mut i = 0;
            while i < tris.len() {
                res.push([tris[i], tris[i + 1], tris[i + 2]]);
                i += 3;
            }
            res
        };

        res.vertices = verts;
        res.triangles = tris;

        res
    };

    let grid = UniformCartesianCubeGrid3d::<i64, f64>::new(
        &Vector3::new(grid_info.0[0], grid_info.0[1], grid_info.0[2]),
        &grid_info.4,
        grid_info.2,
    )
    .expect("returned grid");

    let verts_map = marching_cubes_cleanup(&mut mesh, &grid, max_iter, keep_vertices);

    verts_map
}