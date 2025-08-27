use crate::mesh::PyTriMesh3d;
use crate::utils::*;
use crate::{
    mesh::{TriMesh3dF32, TriMesh3dF64},
    uniform_grid::{UniformGridF32, UniformGridF64},
};
use anyhow::anyhow;
use numpy as np;
use numpy::prelude::*;
use numpy::{Element, PyArray2, PyUntypedArray};
use pyo3::exceptions::PyTypeError;
use pyo3::{Bound, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Real, SpatialDecomposition, SurfaceReconstruction,
    UniformGrid, nalgebra::Vector3,
};

macro_rules! create_reconstruction_interface {
    ($name: ident, $type: ident, $mesh_class: ident, $grid_class: ident) => {
        /// SurfaceReconstruction wrapper
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: SurfaceReconstruction<i64, $type>,
        }

        impl $name {
            pub fn new(data: SurfaceReconstruction<i64, $type>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            /// PyTrimesh3d clone of the contained mesh
            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh().clone())
            }

            /// PyUniformGrid clone of the contained grid
            #[getter]
            fn grid(&self) -> $grid_class {
                $grid_class::new(self.inner.grid().clone())
            }

            // Doesn't work because SurfaceReconstruction.mesh() only returns an immutable reference
            // /// Returns PyTrimesh3dF32/F64 without copying the mesh data, removes the mesh from the object
            // fn take_mesh(&mut self) -> $mesh_class {
            //     let mesh = std::mem::take(&mut self.inner.mesh());
            //     $mesh_class::new(mesh)
            // }

            /// Returns a reference to the global particle density vector if computed during the reconstruction (currently, all reconstruction approaches return this)
            fn particle_densities(&self) -> &Vec<$type> {
                self.inner
                    .particle_densities()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Surface Reconstruction did not return particle densities")
                    })
                    .unwrap()
            }

            /// Returns a reference to the global list of per-particle neighborhood lists if computed during the reconstruction (`None` if not specified in the parameters)
            fn particle_neighbors(&self) -> Option<&Vec<Vec<usize>>> {
                self.inner.particle_neighbors()
            }
        }
    };
}

create_reconstruction_interface!(SurfaceReconstructionF64, f64, TriMesh3dF64, UniformGridF64);
create_reconstruction_interface!(SurfaceReconstructionF32, f32, TriMesh3dF32, UniformGridF32);

enum PyUniformGridData {
    F32(UniformGrid<u64, f32>),
    F64(UniformGrid<u64, f64>),
}

/// Struct containing the parameters of a uniform grid used for the surface reconstruction
#[gen_stub_pyclass]
#[pyclass]
pub struct PyUniformGrid {
    inner: PyUniformGridData,
}

impl_from_mesh!(PyUniformGrid, UniformGrid<u64, f32> => PyUniformGridData::F32);
impl_from_mesh!(PyUniformGrid, UniformGrid<u64, f64> => PyUniformGridData::F64);

impl PyUniformGrid {
    pub(crate) fn as_f32(&self) -> Option<&UniformGrid<u64, f32>> {
        match &self.inner {
            PyUniformGridData::F32(grid) => Some(grid),
            _ => None,
        }
    }

    pub(crate) fn as_f64(&self) -> Option<&UniformGrid<u64, f64>> {
        match &self.inner {
            PyUniformGridData::F64(grid) => Some(grid),
            _ => None,
        }
    }
}

enum PySurfaceReconstructionData {
    F32(SurfaceReconstruction<u64, f32>),
    F64(SurfaceReconstruction<u64, f64>),
}

/// Struct containing results of the surface reconstruction including the mesh, grid parameters and optional particle data
#[gen_stub_pyclass]
#[pyclass]
pub struct PySurfaceReconstruction {
    inner: PySurfaceReconstructionData,
}

impl_from_mesh!(PySurfaceReconstruction, SurfaceReconstruction<u64, f32> => PySurfaceReconstructionData::F32);
impl_from_mesh!(PySurfaceReconstruction, SurfaceReconstruction<u64, f64> => PySurfaceReconstructionData::F64);

impl PySurfaceReconstruction {
    pub fn try_from_generic<R: Real + Element>(
        mut reconstruction: SurfaceReconstruction<u64, R>,
    ) -> PyResult<Self> {
        transmute_take_into::<_, SurfaceReconstruction<u64, f32>, _>(&mut reconstruction)
            .or_else(|| {
                transmute_take_into::<_, SurfaceReconstruction<u64, f64>, _>(&mut reconstruction)
            })
            .ok_or_else(|| PyTypeError::new_err("unsupported type of reconstruction, only u64 for Index and f32 and f64 for Real type are supported"))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySurfaceReconstruction {
    /// Returns a copy of the surface mesh of the reconstruction
    fn copy_mesh(&self) -> PyResult<PyTriMesh3d> {
        match &self.inner {
            PySurfaceReconstructionData::F32(reconstruction) => {
                Ok(PyTriMesh3d::from(reconstruction.mesh().clone()))
            }
            PySurfaceReconstructionData::F64(reconstruction) => {
                Ok(PyTriMesh3d::from(reconstruction.mesh().clone()))
            }
        }
    }

    /// Returns a copy of the uniform grid parameters used for the reconstruction
    fn copy_grid(&self) -> PyUniformGrid {
        match &self.inner {
            PySurfaceReconstructionData::F32(reconstruction) => {
                PyUniformGrid::from(reconstruction.grid().clone())
            }
            PySurfaceReconstructionData::F64(reconstruction) => {
                PyUniformGrid::from(reconstruction.grid().clone())
            }
        }
    }

    /// Returns a copy of the particle densities computed during the reconstruction
    fn copy_particle_densities<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyUntypedArray>> {
        match &self.inner {
            PySurfaceReconstructionData::F32(reconstruction) => Some(
                reconstruction
                    .particle_densities()?
                    .to_pyarray(py)
                    .into_any()
                    .downcast_into::<PyUntypedArray>()
                    .expect("downcasting should not fail"),
            ),
            PySurfaceReconstructionData::F64(reconstruction) => Some(
                reconstruction
                    .particle_densities()?
                    .to_pyarray(py)
                    .into_any()
                    .downcast_into::<PyUntypedArray>()
                    .expect("downcasting should not fail"),
            ),
        }
    }

    /// Returns a copy of the per-particle neighborhood lists computed during the reconstruction if available
    ///
    /// The neighborhood lists are only available if the flag for global neighborhood list was set in the reconstruction parameters.
    fn copy_particle_neighbors(&self) -> Option<Vec<Vec<usize>>> {
        match &self.inner {
            PySurfaceReconstructionData::F32(reconstruction) => reconstruction
                .particle_neighbors()
                .map(|neighbors| neighbors.clone()),
            PySurfaceReconstructionData::F64(reconstruction) => reconstruction
                .particle_neighbors()
                .map(|neighbors| neighbors.clone()),
        }
    }
}

/// Performs a surface reconstruction from the given particles without additional post-processing
///
/// Note that all parameters use absolute distance units and are not relative to the particle radius.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "reconstruct_surface")]
#[pyo3(signature = (particles, *,
    particle_radius, rest_density = 1000.0, smoothing_length, cube_size, iso_surface_threshold = 0.6,
    multi_threading = true, global_neighborhood_list = false,
    subdomain_grid = true, subdomain_grid_auto_disable = true, subdomain_num_cubes_per_dim = 64,
    aabb_min = None, aabb_max = None
))]
pub fn reconstruct_surface<'py>(
    particles: &Bound<'py, PyUntypedArray>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    multi_threading: bool,
    global_neighborhood_list: bool,
    subdomain_grid: bool,
    subdomain_grid_auto_disable: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
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
        spatial_decomposition,
        global_neighborhood_list,
    };

    let element_type = particles.dtype();
    if element_type.is_equiv_to(&np::dtype::<f32>(py)) {
        let particles = particles.downcast::<PyArray2<f32>>()?.readonly();
        let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particles.as_slice()?);
        let reconstruction = splashsurf_lib::reconstruct_surface::<u64, _>(
            particle_positions,
            &parameters
                .try_convert()
                .expect("failed to convert reconstruction parameters to f32"),
        )
        .map_err(|e| anyhow!(e))?;
        PySurfaceReconstruction::try_from_generic(reconstruction)
    } else if element_type.is_equiv_to(&np::dtype::<f64>(py)) {
        let particles = particles.downcast::<PyArray2<f64>>()?.readonly();
        let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particles.as_slice()?);
        let reconstruction =
            splashsurf_lib::reconstruct_surface::<u64, _>(particle_positions, &parameters)
                .map_err(|e| anyhow!(e))?;
        PySurfaceReconstruction::try_from_generic(reconstruction)
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported scalar type {} for reconstruction, only float32 and float64 are supported",
            element_type
        )))
    }
}
