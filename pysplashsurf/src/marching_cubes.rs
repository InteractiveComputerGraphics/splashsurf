use numpy::prelude::*;
use numpy::{Element, PyArray3, PyUntypedArray};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{DensityMap, Real, UniformGrid};

use crate::mesh::{PyTriMesh3d, get_triangle_mesh_generic};
use crate::uniform_grid::PyUniformGrid;
use crate::utils;
use crate::utils::IndexT;

/// Checks the consistency of a reconstructed surface mesh (watertightness, manifoldness), optionally returns a string with details if problems are found
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
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "marching_cubes")]
#[pyo3(signature = (values, *, cube_size, iso_surface_threshold, translation = None))]
pub fn marching_cubes<'py>(
    values: &Bound<'py, PyUntypedArray>,
    cube_size: f64,
    iso_surface_threshold: f64,
    translation: Option<[f64; 3]>,
) -> PyResult<(PyTriMesh3d, PyUniformGrid)> {
    assert_eq!(values.shape().len(), 3, "values must be a 3D array");

    fn triangulate_density_map_generic<'py, R: Real + Element>(
        values: &Bound<'py, PyArray3<R>>,
        cube_size: R,
        iso_surface_threshold: R,
        translation: Option<[R; 3]>,
    ) -> PyResult<(PyTriMesh3d, PyUniformGrid)> {
        let shape = values.shape();
        let translation = Vector3::from(translation.unwrap_or([R::zero(); 3]));
        let n_cells_per_dim = [
            shape[0] as IndexT - 1,
            shape[1] as IndexT - 1,
            shape[2] as IndexT - 1,
        ];

        let grid = UniformGrid::new(&translation, &n_cells_per_dim, cube_size)
            .map_err(anyhow::Error::from)?;

        // TODO: Replace with borrow
        let values = values.try_readonly()?.as_slice()?.to_vec();
        let density_map = DensityMap::from(values);

        let mesh = splashsurf_lib::marching_cubes::triangulate_density_map(
            &grid,
            &density_map,
            iso_surface_threshold,
        )
        .map_err(anyhow::Error::from)?;
        Ok((
            PyTriMesh3d::try_from_generic(mesh)?,
            PyUniformGrid::try_from_generic(grid)?,
        ))
    }

    if let Ok(values) = values.downcast::<PyArray3<f32>>() {
        triangulate_density_map_generic(
            &values,
            cube_size as f32,
            iso_surface_threshold as f32,
            translation.map(|t| t.map(|t| t as f32)),
        )
    } else if let Ok(values) = values.downcast::<PyArray3<f64>>() {
        triangulate_density_map_generic(&values, cube_size, iso_surface_threshold, translation)
    } else {
        Err(utils::pyerr_unsupported_scalar())
    }
}
