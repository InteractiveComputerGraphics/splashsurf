use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView3, ScalarOperand, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use splashsurf_lib::mesh::TriMesh3d;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::uniform_grid::UniformCartesianCubeGrid3d;
use std::collections::HashMap;
use splashsurf_lib::density_map::DensityMap;
use splashsurf_lib::{nalgebra, Aabb3d, Real, SpatialDecomposition};
use splashsurf_lib::marching_cubes::triangulate_density_map;
use splashsurf_lib::{
    reconstruct_surface,
    Parameters,
    UniformGrid,
    postprocessing::{
        marching_cubes_cleanup,
        par_laplacian_smoothing_inplace,
        decimation
    }
};

// fn reconstruct_from_uniform_grid<T: Real + ScalarOperand>(scalar_field: ArrayView3<'_, T>, base: ArrayView1<'_, T>, cell_size: T, threshold: T) -> (Array2<usize>, Array2<T>){
//     assert_eq!(base.shape()[0], 3 as usize);

//     // pad with -1s around original array
//     let y_shape = [scalar_field.shape()[0] + 2, scalar_field.shape()[1] + 2, scalar_field.shape()[2] + 2];
//     let mut y = Array3::<T>::ones(y_shape) * (threshold + T::one());
//     y.slice_mut(s![1..-1, 1..-1, 1..-1]).assign(&scalar_field);

//     #[cfg(debug_assertions)]
//     println!("x: {:?}", scalar_field.dim());

//     //let mut hashmap = HashMap::<i64, f64, fxhash::FxBuildHasher>::new();
//     let mut hashmap = HashMap::<usize, T, fxhash::FxBuildHasher>::with_hasher(fxhash::FxBuildHasher::default());

//     if cfg!(debug_assertions){
//         let y = scalar_field.shape();
//         println!("dyn dims: {:?}",y);
//     }

//     // Subtract cell size to account for the padding of the original array
//     let base = nalgebra::Vector3::<T>::new(base[0]-cell_size, base[1]-cell_size, base[2]-cell_size);
//     let res = [y.shape()[0], y.shape()[1], y.shape()[2]];
//     let grid = UniformGrid::new(&base, &res, cell_size).unwrap();

//     for (i, value) in y.indexed_iter(){
//         let fixed_size_index   = &[i.0, i.1, i.2];
//         hashmap.insert(grid.flatten_point_index_array(fixed_size_index), *value);
//     }

//     #[cfg(debug_assertions)]
//     println!("Hashmap length: {}", hashmap.len());

//     let densitymap = DensityMap::from(hashmap);

//     let surface = triangulate_density_map(&grid, &densitymap, threshold).unwrap();

//     let points : Vec<T> = surface.vertices.iter().flatten().copied().collect();
//     let tris : Vec<usize> = surface.triangles.iter().flatten().copied().collect();
//     let triangles = ndarray::Array2::from_shape_vec((surface.triangles.len(), 3), tris).unwrap();
//     let vertices = ndarray::Array2::from_shape_vec((surface.vertices.len(), 3), points).unwrap();

//     (triangles, vertices)
// }

// /// Reconstruct the surface from grid data
// #[pyfunction]
// #[pyo3(name="reconstruct_from_uniform_grid_f64")]
// #[pyo3(text_signature = "(scalar_field, base, cell_size, /)")]
// fn reconstruct_from_uniform_grid_py_f64<'py>(py: Python<'py>, scalar_field: PyReadonlyArray3<f64>, base: PyReadonlyArray1<f64>, cell_size: f64, threshold: f64) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f64>>){
//     let x = scalar_field.as_array();
//     let (tris, vertices) = reconstruct_from_uniform_grid(x, base.as_array(), cell_size, threshold);
//     (tris.into_pyarray(py), vertices.into_pyarray(py))
// }

// /// Reconstruct the surface from grid data
// #[pyfunction]
// #[pyo3(name="reconstruct_from_uniform_grid_f32")]
// #[pyo3(text_signature = "(scalar_field, base, cell_size, /)")]
// fn reconstruct_from_uniform_grid_py_f32<'py>(py: Python<'py>, scalar_field: PyReadonlyArray3<f32>, base: PyReadonlyArray1<f32>, cell_size: f32, threshold: f32) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>){
//     let x = scalar_field.as_array();
//     let (tris, vertices) = reconstruct_from_uniform_grid(x, base.as_array(), cell_size, threshold);
//     (tris.into_pyarray(py), vertices.into_pyarray(py))
// }

/// Reconstruct the surface from only particle positions
#[pyfunction]
#[pyo3(name="reconstruct_surface")]
#[pyo3(signature = (particles, *, particle_radius=0.025, rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, iso_surface_threshold=0.6, enable_multi_threading=false, global_neighborhood_list=false))]
fn reconstruct_surface_py<'py>(
    py: Python<'py>, 
    particles: Vec<[f64; 3]>, 
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    //spatial_decomposition: Option<SpatialDecomposition<f64>>
) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f64>>, ([f64; 3], [f64; 3], f64, [i64; 3], [i64; 3])) {
    let particle_positions: Vec<Vector3<f64>> = particles.iter().map(|v| Vector3::new(v[0], v[1], v[2])).collect();

    let mut min = particle_positions[0];
    let mut max = particle_positions[0];
    for v in particle_positions.clone() {
        if v < min {
            min = v;
        }

        if v > max {
            max = v;
        }
    }

    let params = Parameters::<f64> {
        particle_radius,
        rest_density,
        compact_support_radius: smoothing_length * 2.0 * particle_radius,
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb: Some(Aabb3d::new(min, max)),
        enable_multi_threading,
        spatial_decomposition: None,
        global_neighborhood_list,
    };
    
    let surface = reconstruct_surface::<i64,f64>(&particle_positions, &params).expect("Surface Reconstruction");
    let grid = surface.grid().clone();
    let aabb = grid.aabb().clone();

    let grid_info = {
        let min = [aabb.min()[0], aabb.min()[1], aabb.min()[2]];
        let max = [aabb.max()[0], aabb.max()[1], aabb.max()[2]];
        (
            min, 
            max, 
            grid.cell_size().clone(), 
            grid.points_per_dim().clone(), 
            grid.cells_per_dim().clone()
        )
    };

    let mesh = surface.mesh();

    let points : Vec<f64> = mesh.vertices.iter().flatten().copied().collect();
    let tris : Vec<usize> = mesh.triangles.iter().flatten().copied().collect();
    let triangles = ndarray::Array2::from_shape_vec((mesh.triangles.len(), 3), tris).unwrap();
    let vertices = ndarray::Array2::from_shape_vec((mesh.vertices.len(), 3), points).unwrap();

    (
        triangles.into_pyarray(py),
        vertices.into_pyarray(py),
        grid_info
    )
}

#[pyfunction]
#[pyo3(name="marching_cubes_cleanup")]
#[pyo3(signature = (triangles, vertices, grid_info, *, max_iter, keep_vertices))]
fn marching_cubes_cleanup_py<'py>(
    py: Python<'py>, 
    triangles: PyReadonlyArray2<usize>,
    vertices: PyReadonlyArray2<f64>,
    grid_info: ([f64; 3], [f64; 3], f64, [i64; 3], [i64; 3]),
    max_iter: usize,
    keep_vertices: bool
) -> Vec<Vec<usize>> {
    let mut mesh = {
        let mut res = TriMesh3d::<f64>::default();

        let verts = {
            let binding = vertices.as_array();
            let verts: ndarray::ArrayBase<ndarray::CowRepr<'_, f64>, ndarray::Dim<[usize; 1]>> = binding.flatten();
            let mut res = Vec::new();
            let mut i = 0;
            while i < verts.len() {
                res.push(Vector3::new(verts[i], verts[i+1], verts[i+2]));
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
                res.push([tris[i], tris[i+1], tris[i+2]]);
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
        grid_info.2
    ).expect("returned grid");

    let verts_map = marching_cubes_cleanup(&mut mesh, &grid, max_iter, keep_vertices);

    verts_map
}

/// High-Level Bindings of the splashsurf surface reconstruction implementation.
/// Support reconstructing Level-Set surfaces from particle clouds or from regular grids.
#[pymodule]
fn pysplashsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_function(wrap_pyfunction!(reconstruct_surface_py, m)?);
    let _ = m.add_function(wrap_pyfunction!(marching_cubes_cleanup_py, m)?);
    Ok(())
}
