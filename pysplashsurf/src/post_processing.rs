use ndarray::ArrayViewMut2;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use splashsurf_lib::nalgebra::Vector3;

use crate::mesh::{
    MeshType, PyMeshWithData, PyMixedTriQuadMesh3d, PyTriMesh3d, TriMesh3dF32, TriMesh3dF64,
    TriMeshWithDataF32, TriMeshWithDataF64,
};
use crate::uniform_grid::PyUniformGrid;

/// Merges triangles sharing an edge to quads if they fulfill the given criteria
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "convert_tris_to_quads")]
#[pyo3(signature = (mesh, *, non_squareness_limit = 1.75, normal_angle_limit = 10.0, max_interior_angle = 135.0))]
#[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MixedTriQuadMesh3d]", imports=()))]
pub fn convert_tris_to_quads<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[MixedTriQuadMesh3d, MeshWithData]", imports=()))]
    mesh: Bound<'py, PyAny>,
    non_squareness_limit: f64,
    normal_angle_limit: f64,
    max_interior_angle: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = mesh.py();

    let normal_angle_limit = normal_angle_limit.to_radians();
    let max_interior_angle = max_interior_angle.to_radians();

    if let Ok(mesh) = mesh.downcast::<PyTriMesh3d>() {
        let mesh = mesh.borrow();
        if let Some(mesh) = mesh.as_f32() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                mesh,
                non_squareness_limit as f32,
                normal_angle_limit as f32,
                max_interior_angle as f32,
            );
            PyMixedTriQuadMesh3d::from(quad_mesh).into_bound_py_any(py)
        } else if let Some(mesh) = mesh.as_f64() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                mesh,
                non_squareness_limit,
                normal_angle_limit,
                max_interior_angle,
            );
            PyMixedTriQuadMesh3d::from(quad_mesh).into_bound_py_any(py)
        } else {
            Err(PyTypeError::new_err(
                "unsupported mesh scalar data type, only f32 and f64 are supported",
            ))
        }
    } else if let Ok(mesh) = mesh.downcast::<PyMeshWithData>() {
        let mesh = mesh.borrow();
        if let Some(mesh) = mesh.as_tri_f32() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                &mesh.mesh,
                non_squareness_limit as f32,
                normal_angle_limit as f32,
                max_interior_angle as f32,
            );
            let mut quad_mesh = splashsurf_lib::mesh::MeshWithData::new(quad_mesh);
            quad_mesh.point_attributes = mesh.point_attributes.clone();
            PyMeshWithData::from(quad_mesh).into_bound_py_any(py)
        } else if let Some(mesh) = mesh.as_tri_f64() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                &mesh.mesh,
                non_squareness_limit,
                normal_angle_limit,
                max_interior_angle,
            );
            let mut quad_mesh = splashsurf_lib::mesh::MeshWithData::new(quad_mesh);
            quad_mesh.point_attributes = mesh.point_attributes.clone();
            PyMeshWithData::from(quad_mesh).into_bound_py_any(py)
        } else {
            Err(PyTypeError::new_err(
                "unsupported mesh scalar data type, only f32 and f64 are supported",
            ))
        }
    } else {
        Err(PyTypeError::new_err(
            "unsupported mesh type, only triangle meshes are supported",
        ))
    }
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f64")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f64<'py>(
    py: Python,
    mesh: PyObject,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f64,
    weights: Vec<f64>, // ToDo: Same here
) -> PyResult<()> {
    if mesh.downcast_bound::<TriMesh3dF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMesh3dF64>(py).unwrap();
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            &mut mesh.borrow_mut().inner,
            &vertex_connectivity,
            iterations,
            beta,
            &weights,
        );
        Ok(())
    } else if mesh.downcast_bound::<TriMeshWithDataF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMeshWithDataF64>(py).unwrap();
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            &mut mesh.borrow_mut().inner.mesh,
            &vertex_connectivity,
            iterations,
            beta,
            &weights,
        );
        Ok(())
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_inplace_f32")]
#[pyo3(signature = (mesh, vertex_connectivity, iterations, beta, weights))]
pub fn par_laplacian_smoothing_inplace_py_f32<'py>(
    py: Python,
    mesh: PyObject,
    vertex_connectivity: Vec<Vec<usize>>, // ToDo: only take reference to data here
    iterations: usize,
    beta: f32,
    weights: Vec<f32>, // ToDo: Same here
) -> PyResult<()> {
    if mesh.downcast_bound::<TriMesh3dF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMesh3dF32>(py).unwrap();
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            &mut mesh.borrow_mut().inner,
            &vertex_connectivity,
            iterations,
            beta,
            &weights,
        );
        Ok(())
    } else if mesh.downcast_bound::<TriMeshWithDataF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMeshWithDataF32>(py).unwrap();
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            &mut mesh.borrow_mut().inner.mesh,
            &vertex_connectivity,
            iterations,
            beta,
            &weights,
        );
        Ok(())
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_normals_inplace_f32")]
#[pyo3(signature = (normals, vertex_connectivity, iterations))]
pub fn par_laplacian_smoothing_normals_inplace_py_f32<'py>(
    normals: &Bound<'py, PyArray2<f32>>,
    vertex_connectivity: Vec<Vec<usize>>,
    iterations: usize,
) {
    let mut normals: ArrayViewMut2<f32> = unsafe { normals.as_array_mut() };
    let mut normals_vec: Vec<Vector3<f32>> =
        bytemuck::cast_vec(normals.as_slice().unwrap().to_vec()); // Copies data temporarily into a vec
    splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
        &mut normals_vec,
        &vertex_connectivity,
        iterations,
    );
    normals
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(&bytemuck::cast_slice(normals_vec.as_slice())); // Copy back to numpy array
}

#[pyfunction]
#[pyo3(name = "par_laplacian_smoothing_normals_inplace_f64")]
#[pyo3(signature = (normals, vertex_connectivity, iterations))]
pub fn par_laplacian_smoothing_normals_inplace_py_f64<'py>(
    normals: &Bound<'py, PyArray2<f64>>,
    vertex_connectivity: Vec<Vec<usize>>,
    iterations: usize,
) {
    let mut normals: ArrayViewMut2<f64> = unsafe { normals.as_array_mut() };
    let mut normals_vec: Vec<Vector3<f64>> =
        bytemuck::cast_vec(normals.as_slice().unwrap().to_vec()); // Copies data temporarily into a vec
    splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
        &mut normals_vec,
        &vertex_connectivity,
        iterations,
    );
    normals
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(&bytemuck::cast_slice(normals_vec.as_slice())); // Copy back to numpy array
}

#[pyfunction]
#[pyo3(name = "decimation_f64")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f64<'py>(
    py: Python,
    mesh: PyObject,
    keep_vertices: bool,
) -> PyResult<Vec<Vec<usize>>> {
    if mesh.downcast_bound::<TriMesh3dF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMesh3dF64>(py).unwrap();
        Ok(splashsurf_lib::postprocessing::decimation(
            &mut mesh.borrow_mut().inner,
            keep_vertices,
        ))
    } else if mesh.downcast_bound::<TriMeshWithDataF64>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMeshWithDataF64>(py).unwrap();
        Ok(splashsurf_lib::postprocessing::decimation(
            &mut mesh.borrow_mut().inner.mesh,
            keep_vertices,
        ))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

#[pyfunction]
#[pyo3(name = "decimation_f32")]
#[pyo3(signature = (mesh, *, keep_vertices))]
pub fn decimation_py_f32<'py>(
    py: Python,
    mesh: PyObject,
    keep_vertices: bool,
) -> PyResult<Vec<Vec<usize>>> {
    if mesh.downcast_bound::<TriMesh3dF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMesh3dF32>(py).unwrap();
        Ok(splashsurf_lib::postprocessing::decimation(
            &mut mesh.borrow_mut().inner,
            keep_vertices,
        ))
    } else if mesh.downcast_bound::<TriMeshWithDataF32>(py).is_ok() {
        let mesh = mesh.downcast_bound::<TriMeshWithDataF32>(py).unwrap();
        Ok(splashsurf_lib::postprocessing::decimation(
            &mut mesh.borrow_mut().inner.mesh,
            keep_vertices,
        ))
    } else {
        Err(PyErr::new::<PyValueError, _>("Invalid mesh type"))
    }
}

/// Mesh simplification designed for marching cubes surfaces meshes inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup")]
#[pyo3(signature = (mesh, grid, *, max_rel_snap_dist = None, max_iter = 5, keep_vertices = false))]
pub fn marching_cubes_cleanup<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: Bound<'py, PyAny>,
    grid: &PyUniformGrid,
    max_rel_snap_dist: Option<f64>,
    max_iter: usize,
    keep_vertices: bool,
) -> PyResult<()> {
    let max_rel_snap_dist_f32 = max_rel_snap_dist.map(|d| d as f32);
    use splashsurf_lib::postprocessing::marching_cubes_cleanup as cleanup;

    if let Ok(mesh) = mesh.downcast::<PyTriMesh3d>() {
        let mut mesh = mesh.borrow_mut();
        if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_f32_mut()) {
            cleanup(mesh, grid, max_rel_snap_dist_f32, max_iter, keep_vertices);
        } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_f64_mut()) {
            cleanup(mesh, grid, max_rel_snap_dist, max_iter, keep_vertices);
        } else {
            return Err(PyTypeError::new_err(
                "invalid combination of grid and mesh scalar data types",
            ));
        }
    } else if let Ok(mesh) = mesh.downcast::<PyMeshWithData>()
        && let mut mesh = mesh.borrow_mut()
        && mesh.mesh_cell_type() == MeshType::Tri3d
    {
        if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_tri_f32_mut()) {
            let mesh = &mut mesh.mesh;
            cleanup(mesh, grid, max_rel_snap_dist_f32, max_iter, keep_vertices);
        } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_tri_f64_mut()) {
            let mesh = &mut mesh.mesh;
            cleanup(mesh, grid, max_rel_snap_dist, max_iter, keep_vertices);
        } else {
            return Err(PyTypeError::new_err(
                "invalid combination of grid and mesh scalar data types",
            ));
        }
    } else {
        return Err(PyTypeError::new_err(
            "unsupported mesh type for, only triangle meshes are supported",
        ));
    }

    Ok(())
}
