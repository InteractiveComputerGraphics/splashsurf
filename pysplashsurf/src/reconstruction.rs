use crate::mesh::PyTriMesh3d;
use crate::neighborhood_search::PyNeighborhoodLists;
use crate::uniform_grid::PyUniformGrid;
use crate::utils;
use anyhow::anyhow;
use ndarray::ArrayView1;
use numpy as np;
use numpy::prelude::*;
use numpy::{Element, PyArray1, PyArray2, PyUntypedArray};
use pyo3::{Bound, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Real, SpatialDecomposition, SurfaceReconstruction,
    nalgebra::Vector3,
};
use utils::{IndexT, PyFloatVecWrapper};

/// Result returned by surface reconstruction functions with surface mesh and other data
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "SurfaceReconstruction")]
pub struct PySurfaceReconstruction {
    grid: Py<PyUniformGrid>,
    particle_densities: Option<PyFloatVecWrapper>,
    particle_inside_aabb: Option<Vec<bool>>,
    particle_neighbors: Option<Py<PyNeighborhoodLists>>,
    mesh: Py<PyTriMesh3d>,
}

impl PySurfaceReconstruction {
    pub fn try_from_generic<'py, R: Real + Element>(
        py: Python<'py>,
        reconstruction: SurfaceReconstruction<IndexT, R>,
    ) -> PyResult<Self> {
        Ok(Self {
            grid: Py::new(py, PyUniformGrid::try_from_generic(reconstruction.grid)?)?,
            particle_densities: reconstruction
                .particle_densities
                .map(PyFloatVecWrapper::try_from_generic)
                .transpose()?,
            particle_inside_aabb: reconstruction.particle_inside_aabb,
            particle_neighbors: reconstruction
                .particle_neighbors
                .map(|n| Py::new(py, PyNeighborhoodLists::from(n)))
                .transpose()?,
            mesh: Py::new(py, PyTriMesh3d::try_from_generic(reconstruction.mesh)?)?,
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySurfaceReconstruction {
    /// The marching cubes grid parameters used for the surface reconstruction
    #[getter]
    fn grid<'py>(this: Bound<'py, Self>) -> Py<PyUniformGrid> {
        this.borrow().grid.clone_ref(this.py())
    }

    /// The global array of particle densities (`None` if they were only computed locally)
    #[getter]
    fn particle_densities<'py>(
        this: Bound<'py, Self>,
    ) -> PyResult<Option<Bound<'py, PyUntypedArray>>> {
        this.borrow()
            .particle_densities
            .as_ref()
            .map(|p| p.view(this.into_any()))
            .transpose()
    }

    /// A boolean array indicating whether each particle was inside the AABB used for the reconstruction (`None` if no AABB was set)
    #[getter]
    fn particle_inside_aabb<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, PyUntypedArray>> {
        this.borrow().particle_inside_aabb.as_ref().map(|p| {
            let array: ArrayView1<bool> = ArrayView1::from(p.as_slice());
            let pyarray = unsafe { PyArray1::borrow_from_array(&array, this.into_any()) };
            pyarray
                .into_any()
                .downcast_into::<PyUntypedArray>()
                .expect("downcast should not fail")
        })
    }

    /// The global neighborhood lists per particle (`None` if they were only computed locally)
    #[getter]
    fn particle_neighbors<'py>(this: Bound<'py, Self>) -> Option<Py<PyNeighborhoodLists>> {
        this.borrow()
            .particle_neighbors
            .as_ref()
            .map(|p| p.clone_ref(this.py()))
    }

    /// The reconstructed triangle mesh
    #[getter]
    fn mesh<'py>(this: Bound<'py, Self>) -> Py<PyTriMesh3d> {
        this.borrow().mesh.clone_ref(this.py())
    }
}

/// Performs a surface reconstruction from the given particles without additional post-processing
///
/// Note that all parameters use absolute distance units and are not relative to the particle radius.
///
/// Parameters
/// ----------
/// particles : numpy.ndarray
///     A two-dimensional numpy array of shape (N, 3) containing the positions of the particles.
/// particle_radius
///     Particle radius.
/// rest_density
///     Rest density of the fluid.
/// smoothing_length
///     Smoothing length of the SPH kernel in absolute distance units (compact support radius of SPH kernel will be twice the smoothing length).
/// cube_size
///     Size of the cubes (voxels) used for the marching cubes grid in absolute distance units.
/// iso_surface_threshold
///     Threshold of the SPH interpolation of the "color field" where the iso surface should be extracted.
/// aabb_min
///     Lower corner of the AABB of particles to consider in the reconstruction.
/// aabb_max
///     Upper corner of the AABB of particles to consider in the reconstruction.
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
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "reconstruct_surface")]
#[pyo3(signature = (particles, *,
    particle_radius, rest_density = 1000.0, smoothing_length, cube_size, iso_surface_threshold = 0.6,
    aabb_min = None, aabb_max = None,
    multi_threading = true, simd = true, global_neighborhood_list = false,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64
))]
pub fn reconstruct_surface<'py>(
    particles: &Bound<'py, PyUntypedArray>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
    multi_threading: bool,
    simd: bool,
    global_neighborhood_list: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
) -> PyResult<PySurfaceReconstruction> {
    let py = particles.py();

    let particle_aabb = aabb_min
        .zip(aabb_max)
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
        global_neighborhood_list,
    };

    let element_type = particles.dtype();
    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        let particles = particles.downcast::<PyArray2<f32>>()?.try_readonly()?;
        let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particles.as_slice()?);
        let reconstruction = splashsurf_lib::reconstruct_surface::<IndexT, _>(
            particle_positions,
            &parameters
                .try_convert()
                .expect("failed to convert reconstruction parameters to f32"),
        )
        .map_err(|e| anyhow!(e))?;
        PySurfaceReconstruction::try_from_generic(py, reconstruction)
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        let particles = particles.downcast::<PyArray2<f64>>()?.try_readonly()?;
        let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particles.as_slice()?);
        let reconstruction =
            splashsurf_lib::reconstruct_surface::<IndexT, _>(particle_positions, &parameters)
                .map_err(|e| anyhow!(e))?;
        PySurfaceReconstruction::try_from_generic(py, reconstruction)
    } else {
        Err(utils::pyerr_unsupported_scalar())
    }
}
