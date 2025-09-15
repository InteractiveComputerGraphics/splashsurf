use numpy as np;
use numpy::prelude::*;
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition,
    mesh::{AttributeData, MeshAttribute},
    nalgebra::Vector3,
};
use std::borrow::Cow;

use crate::mesh::PyMeshWithData;
use crate::reconstruction::PySurfaceReconstruction;
use crate::utils::{IndexT, pyerr_unsupported_scalar};

/// Runs the surface reconstruction pipeline for the given particle positions with optional post-processing
///
/// Note that smoothing length and cube size are given in multiples of the particle radius.
///
/// Parameters
/// ----------
/// particles : numpy.ndarray
///     A two-dimensional numpy array of shape (N, 3) containing the positions of the particles.
/// attributes_to_interpolate
///     Dictionary containing all attributes to interpolate. The keys are the attribute names and the values are the corresponding 1D/2D arrays.
///     The arrays must have the same length as the number of particles.
///     Supported array types are 2D float32/float64 arrays for vector attributes and 1D uint64/float32/float64 arrays for scalar attributes.
/// particle_radius
///     Particle radius.
/// rest_density
///     Rest density of the fluid.
/// smoothing_length
///     Smoothing length of the SPH kernel in multiples of the particle radius (compact support radius of SPH kernel will be twice the smoothing length).
/// cube_size
///     Size of the cubes (voxels) used for the marching cubes grid in multiples of the particle radius.
/// iso_surface_threshold
///     Threshold of the SPH interpolation of the "color field" where the iso surface should be extracted.
/// aabb_min
///     Lower corner [x,y,z] of the AABB of particles that are active in the reconstruction.
/// aabb_max
///     Upper corner [x,y,z] of the AABB of particles to consider in the reconstruction.
/// multi_threading
///     Flag to enable multi-threading for the reconstruction and post-processing steps.
/// simd
///    Flag to enable SIMD vectorization for the reconstruction if supported by the CPU architecture.
/// subdomain_grid
///     Flag to enable spatial decomposition by dividing the domain into subdomains with dense marching cube grids for efficient multi-threading.
/// subdomain_grid_auto_disable
///     Flag to automatically disable the subdomain grid if the global domain is too small.
/// subdomain_num_cubes_per_dim
///     Number of marching cubes voxels along each coordinate axis in each subdomain if the subdomain grid is enabled.
/// check_mesh_closed
///     Flag to enable checking the final mesh for holes.
/// check_mesh_manifold
///     Flag to enable checking the final mesh for non-manifold edges and vertices.
/// check_mesh_orientation
///     Flag to enable checking the final mesh for inverted triangles (compares angle between vertex normals and adjacent face normals).
/// check_mesh_debug
///     Flag to enable additional debug output for the check-mesh operations (has no effect if no other check-mesh option is enabled).
/// mesh_cleanup
///     Flag to enable marching cubes mesh cleanup. This implements the method from "Compact isocontours from sampled data" (Moore, Warren; 1992).
/// mesh_cleanup_snap_dist
///     If marching cubes mesh cleanup is enabled, this limits vertex snapping to the specified distance relative to the MC edge length (should be in range of [0.0,0.5]).
/// decimate_barnacles
///     Flag to perform barnacle decimation. For details see "Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids" (Löschner, Böttcher, Jeske, Bender; 2023).
/// keep_vertices
///     Flag to retain any vertices without connectivity resulting from mesh cleanup or decimation step instead of filtering them out.
/// compute_normals
///     Flag to enable computation of vertex normals on the final mesh.
/// sph_normals
///     Flag to enable computation of normals using SPH interpolation (gradient of the color field) instead of geometry-based normals.
/// normals_smoothing_iters
///     Number of Laplacian smoothing iterations to perform on the normal field.
/// mesh_smoothing_iters
///     Number of Laplacian smoothing iterations to perform on the mesh vertices.
/// mesh_smoothing_weights
///     Flag to enable computation and usage of mesh smoothing weights according to "Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids" (Löschner, Böttcher, Jeske, Bender; 2023).
/// mesh_smoothing_weights_normalization
///     Normalization value for the mesh smoothing weights.
/// generate_quads
///     Flag to enable conversion of triangles to quads depending on geometric criteria.
/// quad_max_edge_diag_ratio
///     Maximum allowed ratio of quad edge lengths to its diagonals to merge two triangles to a quad (inverse is used for minimum).
/// quad_max_normal_angle
///     Maximum allowed angle (in degrees) between triangle normals to merge them to a quad.
/// quad_max_interior_angle
///     Maximum allowed vertex interior angle (in degrees) inside a quad to merge two triangles to a quad.
/// output_mesh_smoothing_weights
///     Flag to attach and return the smoothing weights as a mesh attribute if smoothing weights are computed.
/// output_raw_normals
///     Flag to output the raw normals in addition to smoothed normals if smoothing of normals is enabled.
/// output_raw_mesh
///     Flag to return a copy of the raw mesh before any post-processing steps (inside the returned reconstruction object).
/// mesh_aabb_min
///     Lower corner [x,y,z] of the axis-aligned bounding box for the mesh, triangles fully outside this box will be removed.
/// mesh_aabb_max
///     Upper corner [x,y,z] of the axis-aligned bounding box for the mesh, triangles fully outside this box will be removed.
/// mesh_aabb_clamp_vertices
///     Flag to clamp the vertices of the mesh to the AABB in addition to removing triangles outside the AABB.
///
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "reconstruction_pipeline")]
#[pyo3(signature = (particles, *, attributes_to_interpolate = None,
    particle_radius, rest_density = 1000.0, smoothing_length, cube_size, iso_surface_threshold = 0.6,
    aabb_min = None, aabb_max = None, multi_threading = true, simd = true,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64,
    check_mesh_closed = false, check_mesh_manifold = false, check_mesh_orientation = false, check_mesh_debug = false,
    mesh_cleanup = false, mesh_cleanup_snap_dist = None, decimate_barnacles = false, keep_vertices = false, compute_normals = false, sph_normals = false,
    normals_smoothing_iters = None, mesh_smoothing_iters = None, mesh_smoothing_weights = true, mesh_smoothing_weights_normalization = 13.0,
    generate_quads = false, quad_max_edge_diag_ratio = 1.75, quad_max_normal_angle = 10.0, quad_max_interior_angle = 135.0,
    output_mesh_smoothing_weights = false, output_raw_normals = false, output_raw_mesh = false,
    mesh_aabb_min = None, mesh_aabb_max = None, mesh_aabb_clamp_vertices = true
))]
pub fn reconstruction_pipeline<'py>(
    particles: &Bound<'py, PyUntypedArray>,
    attributes_to_interpolate: Option<Bound<'py, PyDict>>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
    multi_threading: bool,
    simd: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    check_mesh_closed: bool,
    check_mesh_manifold: bool,
    check_mesh_orientation: bool,
    check_mesh_debug: bool,
    mesh_cleanup: bool,
    mesh_cleanup_snap_dist: Option<f64>,
    decimate_barnacles: bool,
    keep_vertices: bool,
    compute_normals: bool,
    sph_normals: bool,
    normals_smoothing_iters: Option<usize>,
    mesh_smoothing_iters: Option<usize>,
    mesh_smoothing_weights: bool,
    mesh_smoothing_weights_normalization: f64,
    generate_quads: bool,
    quad_max_edge_diag_ratio: f64,
    quad_max_normal_angle: f64,
    quad_max_interior_angle: f64,
    output_mesh_smoothing_weights: bool,
    output_raw_normals: bool,
    output_raw_mesh: bool,
    mesh_aabb_min: Option<[f64; 3]>,
    mesh_aabb_max: Option<[f64; 3]>,
    mesh_aabb_clamp_vertices: bool,
    //dtype: Option<Bound<'py, PyArrayDescr>>,
) -> PyResult<(PyMeshWithData, PySurfaceReconstruction)> {
    let py = particles.py();
    let element_type = particles.dtype();

    let dtype: Option<Bound<'py, PyArrayDescr>> = None;
    if let Some(target_dtype) = dtype
        && !target_dtype.is_equiv_to(&element_type)
    {
        unimplemented!("Casting to different dtype is not implemented yet");
    }

    let particle_aabb = aabb_min
        .zip(aabb_max)
        .map(|(min, max)| Aabb3d::new(Vector3::from(min), Vector3::from(max)));

    let mesh_aabb = mesh_aabb_min
        .zip(mesh_aabb_max)
        .map(|(min, max)| Aabb3d::new(Vector3::from(min), Vector3::from(max)));

    let spatial_decomposition = if subdomain_grid {
        SpatialDecomposition::UniformGrid(GridDecompositionParameters {
            subdomain_num_cubes_per_dim,
            auto_disable: subdomain_grid_auto_disable,
        })
    } else {
        SpatialDecomposition::None
    };

    let parameters = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        compact_support_radius: 2.0 * smoothing_length * particle_radius,
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb,
        enable_multi_threading: multi_threading,
        enable_simd: simd,
        spatial_decomposition,
        global_neighborhood_list: false,
    };

    let postprocessing_args = splashsurf::reconstruct::ReconstructionPostprocessingParameters {
        check_mesh_closed,
        check_mesh_manifold,
        check_mesh_orientation,
        check_mesh_debug,
        mesh_cleanup,
        mesh_cleanup_snap_dist,
        decimate_barnacles,
        keep_vertices,
        compute_normals,
        sph_normals,
        normals_smoothing_iters,
        interpolate_attributes: None,
        mesh_smoothing_iters,
        mesh_smoothing_weights,
        mesh_smoothing_weights_normalization,
        generate_quads,
        quad_max_edge_diag_ratio,
        quad_max_normal_angle,
        quad_max_interior_angle,
        output_mesh_smoothing_weights,
        output_raw_normals,
        output_raw_mesh,
        mesh_aabb,
        mesh_aabb_clamp_vertices,
    };

    fn reconstruction_to_pymesh<'py, R: Real + Element>(
        py: Python<'py>,
        reconstruction: splashsurf::reconstruct::ReconstructionResult<IndexT, R>,
    ) -> PyResult<(PyMeshWithData, PySurfaceReconstruction)> {
        let mesh_with_data = if let Some(tri_mesh) = reconstruction.tri_mesh {
            PyMeshWithData::try_from_mesh_with_data(py, tri_mesh)?
        } else if let Some(tri_quad_mesh) = reconstruction.tri_quad_mesh {
            PyMeshWithData::try_from_mesh_with_data(py, tri_quad_mesh)?
        } else {
            return Err(PyRuntimeError::new_err("reconstruction returned no mesh"));
        };
        let rec = PySurfaceReconstruction::try_from_generic(py, reconstruction.raw_reconstruction)?;
        Ok((mesh_with_data, rec))
    }

    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        let particles = particles.downcast::<PyArray2<f32>>()?;
        let reconstruction = reconstruction_pipeline_generic_impl::<IndexT, _>(
            particles,
            attributes_to_interpolate,
            &parameters
                .try_convert()
                .expect("failed to convert reconstruction parameters to f32"),
            &postprocessing_args,
        )?;
        reconstruction_to_pymesh(py, reconstruction)
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        let particles = particles.downcast::<PyArray2<f64>>()?;
        let reconstruction = reconstruction_pipeline_generic_impl::<IndexT, _>(
            particles,
            attributes_to_interpolate,
            &parameters,
            &postprocessing_args,
        )?;
        reconstruction_to_pymesh(py, reconstruction)
    } else {
        Err(pyerr_unsupported_scalar())
    }
}

fn reconstruction_pipeline_generic_impl<'py, I: Index, R: Real + Element>(
    particles: &Bound<'py, PyArray2<R>>,
    attributes_to_interpolate: Option<Bound<'py, PyDict>>,
    parameters: &splashsurf_lib::Parameters<R>,
    postprocessing_args: &splashsurf::reconstruct::ReconstructionPostprocessingParameters,
) -> Result<splashsurf::reconstruct::ReconstructionResult<I, R>, anyhow::Error> {
    let particles: PyReadonlyArray2<R> = particles.try_readonly()?;
    let particle_positions: &[Vector3<R>] = bytemuck::cast_slice(particles.as_slice()?);

    enum AttributePyView<'a, R: Real + Element> {
        U64(PyReadonlyArray1<'a, u64>),
        Float(PyReadonlyArray1<'a, R>),
        FloatVec3(PyReadonlyArray2<'a, R>),
    }

    let mut attr_names = Vec::new();
    let mut attr_views = Vec::new();

    // Collect readonly views of all attribute arrays
    for (key, value) in attributes_to_interpolate.iter().flatten() {
        let key_str: String = key
            .downcast::<PyString>()
            .expect("attribute key has to be a string")
            .extract()?;

        if let Ok(value) = value.downcast::<PyArray1<u64>>() {
            attr_views.push(AttributePyView::U64(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray1<R>>() {
            attr_views.push(AttributePyView::Float(value.readonly()));
            attr_names.push(key_str);
        } else if let Ok(value) = value.downcast::<PyArray2<R>>() {
            attr_views.push(AttributePyView::FloatVec3(value.readonly()));
            attr_names.push(key_str);
        } else {
            println!("Failed to downcast attribute {} to valid type", &key_str);
        }
    }

    // Get slices from attribute views and construct borrowed MeshAttributes
    let attributes = attr_names
        .into_iter()
        .zip(attr_views.iter())
        .map(|(name, view)| -> Result<MeshAttribute<R>, anyhow::Error> {
            let data = match view {
                AttributePyView::U64(view) => {
                    AttributeData::ScalarU64(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::Float(view) => {
                    AttributeData::ScalarReal(Cow::Borrowed(view.as_slice()?.into()))
                }
                AttributePyView::FloatVec3(view) => {
                    let vec3_slice: &[Vector3<R>] = bytemuck::cast_slice(view.as_slice()?);
                    AttributeData::Vector3Real(Cow::Borrowed(vec3_slice.into()))
                }
            };
            Ok(MeshAttribute::new(name, data))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut postprocessing_args = postprocessing_args.clone();
    postprocessing_args.interpolate_attributes =
        (!attributes.is_empty()).then(|| attributes.iter().map(|a| a.name.clone()).collect());

    splashsurf::reconstruct::reconstruction_pipeline(
        particle_positions,
        &attributes,
        &parameters,
        &postprocessing_args,
    )
}
