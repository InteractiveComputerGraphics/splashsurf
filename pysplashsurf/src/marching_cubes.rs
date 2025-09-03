use numpy::prelude::*;
use numpy::{Element, PyArray3, PyUntypedArray};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{DensityMap, Real, UniformGrid};

use crate::mesh::{PyTriMesh3d, get_triangle_mesh_generic};
use crate::uniform_grid::PyUniformGrid;
use crate::utils;
use crate::utils::IndexT;

/// Checks the consistency of a reconstructed surface mesh (watertightness, manifoldness), optionally returns a string with details, if problems are found
///
/// Parameters
/// ----------
/// mesh
///     The triangle mesh to check for consistency.
/// grid
///     The uniform grid that was used for the marching cubes triangulation of the input mesh.
/// check_closed
///     Flag to enable checking if the mesh is closed (watertight).
/// check_manifold
///     Flag to enable checking if the mesh is manifold (i.e. has no non-manifold vertices & edges).
/// debug
///     Flag to enable additional debug output during the consistency checks.
///
/// Returns
/// -------
///     An optional string with details about the problems found during the consistency checks.
///     If no problems are found, None is returned.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "check_mesh_consistency")]
#[pyo3(signature = (mesh, grid, *, check_closed = true, check_manifold = true, debug = false))]
pub fn check_mesh_consistency<'py>(
    #[gen_stub(override_type(type_repr="typing.Union[TriMesh3d, MeshWithData]", imports=()))]
    mesh: &Bound<'py, PyAny>,
    grid: &PyUniformGrid,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
) -> PyResult<Option<String>> {
    let py = mesh.py();

    // Try to extract the triangle mesh;
    let mesh = get_triangle_mesh_generic(&mesh).ok_or_else(utils::pyerr_only_triangle_mesh)?;
    let mesh = mesh.borrow(py);

    if let (Some(grid), Some(mesh)) = (grid.as_f32(), mesh.as_f32()) {
        Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
            grid,
            mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .err())
    } else if let (Some(grid), Some(mesh)) = (grid.as_f64(), mesh.as_f64()) {
        Ok(splashsurf_lib::marching_cubes::check_mesh_consistency(
            grid,
            mesh,
            check_closed,
            check_manifold,
            debug,
        )
        .err())
    } else {
        Err(utils::pyerr_scalar_type_mismatch())
    }
}

/// Performs a standard marching cubes triangulation of a 3D array of values
///
/// The array of values has to be a contiguous array with shape ``(nx, ny, nz)``.
/// The iso-surface threshold defines which value is considered to be "on" the surface.
/// The cube size and translation parameters define the scaling and translation of the resulting
/// mesh. Without translation, the value ``values[0, 0, 0]`` is located at coordinates ``(0, 0, 0)``.
///
/// The values are interpreted as a "density field", meaning that values higher than the iso-surface
/// threshold are considered to be "inside" the surface and values lower than the threshold are
/// considered to be "outside" the surface. This is the opposite convention to an SDF (signed distance field).
/// However, even if values of an SDF are provided as an input, the marching cubes algorithm
/// will still work and produce a watertight surface mesh (if the surface is fully contained in the
/// array).
///
/// If ``return_grid`` is set to ``True``, the function will return a tuple of the mesh and the
/// uniform grid that was used for the triangulation. This can be used for other functions such as
/// :py:func:`check_mesh_consistency`. Otherwise, only the mesh is returned.
///
/// The function is currently single-threaded. The SPH surface reconstruction functions :py:func:`reconstruction_pipeline`
/// and :py:func:`reconstruct_surface` improve performance by processing multiple patches in parallel.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///    A three-dimensional numpy array of shape (nx, ny, nz) containing the scalar values at the vertices
///    of the marching cubes grid.
/// iso_surface_threshold
///    The iso-surface threshold value used to determine the surface.
/// cube_size
///    The size of each cube/voxel of the marching cubes grid. Determines the scaling of the resulting mesh.
/// translation
///    An optional translation vector [tx, ty, tz] applied to the entire mesh after scaling.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "marching_cubes")]
#[pyo3(signature = (values, *, iso_surface_threshold, cube_size, translation = None, return_grid = false))]
#[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, tuple[TriMesh3d, UniformGrid]]", imports=()))]
pub fn marching_cubes<'py>(
    values: &Bound<'py, PyUntypedArray>,
    iso_surface_threshold: f64,
    cube_size: f64,
    translation: Option<[f64; 3]>,
    return_grid: bool,
) -> PyResult<Py<PyAny>> {
    assert_eq!(values.shape().len(), 3, "values must be a 3D array");

    fn triangulate_density_map_generic<'py, R: Real + Element>(
        values: &Bound<'py, PyArray3<R>>,
        iso_surface_threshold: R,
        cube_size: R,
        translation: Option<[R; 3]>,
        return_grid: bool,
    ) -> PyResult<Py<PyAny>> {
        let py = values.py();
        let shape = values.shape();
        let translation = Vector3::from(translation.unwrap_or([R::zero(); 3]));
        let n_cells_per_dim = [
            shape[0] as IndexT - 1,
            shape[1] as IndexT - 1,
            shape[2] as IndexT - 1,
        ];

        let grid = UniformGrid::new(&translation, &n_cells_per_dim, cube_size)
            .map_err(anyhow::Error::from)?;

        let values = values.try_readonly()?;
        let density_map = DensityMap::from(values.as_slice()?);

        let mesh = splashsurf_lib::marching_cubes::triangulate_density_map(
            &grid,
            &density_map,
            iso_surface_threshold,
        )
        .map_err(anyhow::Error::from)?;

        let mesh = PyTriMesh3d::try_from_generic(mesh)?;
        let grid = PyUniformGrid::try_from_generic(grid)?;

        if return_grid {
            (mesh, grid).into_py_any(py)
        } else {
            mesh.into_py_any(py)
        }
    }

    if let Ok(values) = values.downcast::<PyArray3<f32>>() {
        triangulate_density_map_generic(
            &values,
            iso_surface_threshold as f32,
            cube_size as f32,
            translation.map(|t| t.map(|t| t as f32)),
            return_grid,
        )
    } else if let Ok(values) = values.downcast::<PyArray3<f64>>() {
        triangulate_density_map_generic(
            &values,
            iso_surface_threshold,
            cube_size,
            translation,
            return_grid,
        )
    } else {
        Err(utils::pyerr_unsupported_scalar())
    }
}
