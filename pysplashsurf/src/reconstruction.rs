use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{Bound, prelude::*};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Index, Real, SpatialDecomposition, SurfaceReconstruction,
    nalgebra::Vector3, reconstruct_surface,
};

use crate::{
    mesh::{TriMesh3dF32, TriMesh3dF64},
    uniform_grid::{UniformGridF32, UniformGridF64},
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

/// Reconstruct the surface from only particle positions
pub fn reconstruct_surface_py<I: Index, R: Real>(
    particles: &[Vector3<R>],
    particle_radius: R,
    rest_density: R,
    smoothing_length: R,
    cube_size: R,
    iso_surface_threshold: R,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[R; 3]>,
    aabb_max: Option<[R; 3]>,
) -> SurfaceReconstruction<I, R> {
    let aabb;
    if let (Some(aabb_min), Some(aabb_max)) = (aabb_min, aabb_max) {
        // Convert the min and max arrays to Vector3
        aabb = Some(Aabb3d::new(
            Vector3::from(aabb_min),
            Vector3::from(aabb_max),
        ));
    } else {
        aabb = None;
    }

    let spatial_decomposition;
    if use_custom_grid_decomposition {
        let mut grid_params = GridDecompositionParameters::default();
        grid_params.subdomain_num_cubes_per_dim = subdomain_num_cubes_per_dim;
        spatial_decomposition = Some(SpatialDecomposition::UniformGrid(grid_params));
    } else {
        spatial_decomposition = None;
    }

    let params = splashsurf_lib::Parameters {
        particle_radius,
        rest_density,
        // Compact support is twice the smoothing length
        compact_support_radius: (smoothing_length * particle_radius) * R::from_float(2.0),
        cube_size: cube_size * particle_radius,
        iso_surface_threshold,
        particle_aabb: aabb,
        enable_multi_threading,
        spatial_decomposition,
        global_neighborhood_list,
    };

    let surface = reconstruct_surface(&particles, &params).unwrap();

    surface
}

#[pyfunction]
#[pyo3(name = "reconstruct_surface_f32")]
#[pyo3(signature = (particles, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading=false,
    global_neighborhood_list=false, use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64,
    aabb_min = None, aabb_max = None
))]
pub fn reconstruct_surface_py_f32<'py>(
    particles: &Bound<'py, PyArray2<f32>>,
    particle_radius: f32,
    rest_density: f32,
    smoothing_length: f32,
    cube_size: f32,
    iso_surface_threshold: f32,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[f32; 3]>,
    aabb_max: Option<[f32; 3]>,
) -> PyResult<SurfaceReconstructionF32> {
    let particles: PyReadonlyArray2<f32> = particles.extract()?;

    let particle_positions = particles.as_slice()?;
    let particle_positions: &[Vector3<f32>] = bytemuck::cast_slice(particle_positions);

    let reconstruction = reconstruct_surface_py::<i64, f32>(
        particle_positions,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        enable_multi_threading,
        global_neighborhood_list,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        aabb_min,
        aabb_max,
    );

    Ok(SurfaceReconstructionF32::new(reconstruction.to_owned()))
}

#[pyfunction]
#[pyo3(name = "reconstruct_surface_f64")]
#[pyo3(signature = (particles, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading=false,
    global_neighborhood_list=false, use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64,
    aabb_min = None, aabb_max = None
))]
pub fn reconstruct_surface_py_f64<'py>(
    particles: &Bound<'py, PyArray2<f64>>,
    particle_radius: f64,
    rest_density: f64,
    smoothing_length: f64,
    cube_size: f64,
    iso_surface_threshold: f64,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[f64; 3]>,
    aabb_max: Option<[f64; 3]>,
) -> PyResult<SurfaceReconstructionF64> {
    let particles: PyReadonlyArray2<f64> = particles.extract()?;

    let particle_positions = particles.as_slice()?;
    let particle_positions: &[Vector3<f64>] = bytemuck::cast_slice(particle_positions);

    let reconstruction = reconstruct_surface_py::<i64, f64>(
        particle_positions,
        particle_radius,
        rest_density,
        smoothing_length,
        cube_size,
        iso_surface_threshold,
        enable_multi_threading,
        global_neighborhood_list,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        aabb_min,
        aabb_max,
    );

    Ok(SurfaceReconstructionF64::new(reconstruction.to_owned()))
}
