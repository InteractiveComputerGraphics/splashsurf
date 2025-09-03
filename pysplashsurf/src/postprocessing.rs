use numpy as np;
use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArray};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use splashsurf_lib::nalgebra::Vector3;

use crate::mesh::{
    PyMeshAttribute, PyMeshWithData, PyMixedTriQuadMesh3d, PyVertexVertexConnectivity,
    get_triangle_mesh_generic,
};
use crate::uniform_grid::PyUniformGrid;
use crate::utils::*;

/// Converts triangles to quads by merging triangles sharing an edge if they fulfill the given criteria
///
/// This operation creates a new mesh and does not modify the input mesh.
/// Angles are specified in degrees.
///
/// Parameters
/// ----------
/// mesh
///     The triangle mesh to convert to a mixed triangle-quad mesh.
/// non_squareness_limit
///     Maximum allowed ratio of quad edge lengths to its diagonals to merge two triangles to a quad (inverse is used for minimum).
/// normal_angle_limit
///     Maximum allowed angle (in degrees) between triangle normals to merge them to a quad.
/// max_interior_angle
///     Maximum allowed vertex interior angle (in degrees) inside a quad to merge two triangles to a quad.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "convert_tris_to_quads")]
#[pyo3(signature = (mesh, *, non_squareness_limit = 1.75, normal_angle_limit = 10.0, max_interior_angle = 135.0))]
#[gen_stub(override_return_type(type_repr="typing.Union[MixedTriQuadMesh3d, MeshWithData]", imports=()))]
pub fn convert_tris_to_quads<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: Bound<'py, PyAny>,
    non_squareness_limit: f64,
    normal_angle_limit: f64,
    max_interior_angle: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = mesh.py();

    let normal_angle_limit = normal_angle_limit.to_radians();
    let max_interior_angle = max_interior_angle.to_radians();

    let quad_mesh = {
        // Try to extract the triangle mesh;
        let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(pyerr_only_triangle_mesh)?;
        let mesh = mesh.borrow(py);

        if let Some(mesh) = mesh.as_f32() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                mesh,
                non_squareness_limit as f32,
                normal_angle_limit as f32,
                max_interior_angle as f32,
            );
            Ok(PyMixedTriQuadMesh3d::from(quad_mesh))
        } else if let Some(mesh) = mesh.as_f64() {
            let quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
                mesh,
                non_squareness_limit,
                normal_angle_limit,
                max_interior_angle,
            );
            Ok(PyMixedTriQuadMesh3d::from(quad_mesh))
        } else {
            Err(pyerr_unsupported_scalar())
        }
    }?;

    if let Ok(mesh) = mesh.downcast::<PyMeshWithData>() {
        let mut data_mesh = PyMeshWithData::try_from_pymesh(py, quad_mesh)?;
        data_mesh.point_attributes = mesh
            .borrow()
            .point_attributes
            .iter()
            .map(|attr| {
                let attr_clone: PyMeshAttribute = attr.borrow(py).clone();
                attr_clone.into_pyobject(py).map(Py::from)
            })
            .collect::<Result<_, _>>()?;
        data_mesh.into_bound_py_any(py)
    } else {
        quad_mesh.into_bound_py_any(py)
    }
}

/// Laplacian smoothing of mesh vertices with feature weights
///
/// The smoothing is performed inplace and modifies the vertices of the given mesh.
///
/// Parameters
/// ----------
/// mesh
///     The triangle mesh to smooth.
/// vertex_connectivity
///     The vertex-vertex connectivity of the mesh, required for efficient smoothing.
/// iterations
///     The number of smoothing iterations to perform.
/// beta
///     Factor used for blending the original vertex position with the smoothed position.
/// weights
///     A one-dimensional array of weights per vertex that influence the smoothing.
///     The weight is multiplied with beta.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "laplacian_smoothing_parallel")]
#[pyo3(signature = (mesh, vertex_connectivity, *, iterations, beta = 1.0, weights))]
pub fn laplacian_smoothing_parallel<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: &Bound<'py, PyAny>,
    vertex_connectivity: &Bound<'py, PyVertexVertexConnectivity>,
    iterations: usize,
    beta: f64,
    weights: &Bound<'py, PyUntypedArray>,
) -> PyResult<()> {
    let py = mesh.py();

    // Try to extract the triangle mesh;
    let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(pyerr_only_triangle_mesh)?;
    let mut mesh = mesh.borrow_mut(py);

    if let Some(mesh) = mesh.as_f32_mut() {
        let weights = weights.downcast::<PyArray1<f32>>()?.try_readonly()?;
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            mesh,
            &vertex_connectivity.borrow().connectivity,
            iterations,
            beta as f32,
            weights.as_slice()?,
        );
    } else if let Some(mesh) = mesh.as_f64_mut() {
        let weights = weights.downcast::<PyArray1<f64>>()?.try_readonly()?;
        splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
            mesh,
            &vertex_connectivity.borrow().connectivity,
            iterations,
            beta,
            weights.as_slice()?,
        );
    } else {
        return Err(pyerr_unsupported_scalar());
    }

    Ok(())
}

/// Laplacian smoothing of a normal field
///
/// The smoothing is performed inplace and modifies the given normal array.
///
/// Parameters
/// ----------
/// normals
///     A two-dimensional array of shape (N, 3) containing the normals to smooth.
/// vertex_connectivity
///     The vertex-vertex connectivity of the mesh, required for efficient smoothing.
/// iterations
///     The number of smoothing iterations to perform.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "laplacian_smoothing_normals_parallel")]
#[pyo3(signature = (normals, vertex_connectivity, *, iterations))]
pub fn laplacian_smoothing_normals_parallel<'py>(
    normals: &Bound<'py, PyUntypedArray>,
    vertex_connectivity: &Bound<'py, PyVertexVertexConnectivity>,
    iterations: usize,
) -> PyResult<()> {
    let py = normals.py();
    let element_type = normals.dtype();
    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        let mut normals = normals.downcast::<PyArray2<f32>>()?.try_readwrite()?;
        let normals_vec3: &mut [Vector3<f32>] = bytemuck::cast_slice_mut(normals.as_slice_mut()?);
        splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
            normals_vec3,
            &vertex_connectivity.borrow().connectivity,
            iterations,
        );
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        let mut normals = normals.downcast::<PyArray2<f64>>()?.try_readwrite()?;
        let normals_vec3: &mut [Vector3<f64>] = bytemuck::cast_slice_mut(normals.as_slice_mut()?);
        splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
            normals_vec3,
            &vertex_connectivity.borrow().connectivity,
            iterations,
        );
    } else {
        return Err(pyerr_unsupported_scalar());
    }

    Ok(())
}

/// Performs specialized decimation on the given mesh to prevent "barnacles" when applying weighted Laplacian smoothing
///
/// The decimation is performed inplace and modifies the given mesh.
/// Returns the vertex-vertex connectivity of the decimated mesh which can be used for other
/// post-processing steps.
///
/// Parameters
/// ----------
/// mesh
///     The triangle mesh to decimate.
/// keep_vertices
///     Flag to retain any vertices without connectivity resulting from decimation instead of filtering them out.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "barnacle_decimation")]
#[pyo3(signature = (mesh, *, keep_vertices))]
#[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
pub fn barnacle_decimation<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: Bound<'py, PyAny>,
    keep_vertices: bool,
) -> PyResult<PyVertexVertexConnectivity> {
    use splashsurf_lib::postprocessing::decimation;
    let py = mesh.py();

    // Try to extract the triangle mesh;
    let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(pyerr_only_triangle_mesh)?;
    let mut mesh = mesh.borrow_mut(py);

    if let Some(mesh) = mesh.as_f32_mut() {
        Ok(PyVertexVertexConnectivity::new(decimation(
            mesh,
            keep_vertices,
        )))
    } else if let Some(mesh) = mesh.as_f64_mut() {
        Ok(PyVertexVertexConnectivity::new(decimation(
            mesh,
            keep_vertices,
        )))
    } else {
        Err(pyerr_unsupported_scalar())
    }
}

/// Performs simplification on the given mesh inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren
///
/// The simplification is performed inplace and modifies the given mesh.
/// The method is designed specifically for meshes generated by Marching Cubes.
/// See Moore and Warren: `Mesh Displacement: An Improved Contouring Method for Trivariate Data <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf>`_ (1991)
/// or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
///
/// Parameters
/// ----------
/// mesh
///     The triangle mesh to simplify.
/// grid
///     The uniform grid that was used for the marching cubes triangulation of the input mesh.
/// max_rel_snap_dist
///     Optional maximum relative snapping distance (relative to the grid cell size) to merge close vertices.
/// max_iter
///     The maximum number of iterations of cleanup to perform.
/// keep_vertices
///     Flag to retain any vertices without connectivity resulting from simplification instead of filtering them out.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "marching_cubes_cleanup")]
#[pyo3(signature = (mesh, grid, *, max_rel_snap_dist = None, max_iter = 5, keep_vertices = false))]
#[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
pub fn marching_cubes_cleanup<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: &Bound<'py, PyAny>,
    grid: &PyUniformGrid,
    max_rel_snap_dist: Option<f64>,
    max_iter: usize,
    keep_vertices: bool,
) -> PyResult<()> {
    use splashsurf_lib::postprocessing::marching_cubes_cleanup as cleanup;
    let py = mesh.py();
    let max_rel_snap_dist_f32 = max_rel_snap_dist.map(|d| d as f32);

    // Try to extract the triangle mesh;
    let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(pyerr_only_triangle_mesh)?;
    let mut mesh = mesh.borrow_mut(py);

    if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_f32_mut()) {
        cleanup(mesh, grid, max_rel_snap_dist_f32, max_iter, keep_vertices);
    } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_f64_mut()) {
        cleanup(mesh, grid, max_rel_snap_dist, max_iter, keep_vertices);
    } else {
        return Err(pyerr_scalar_type_mismatch());
    }

    Ok(())
}
