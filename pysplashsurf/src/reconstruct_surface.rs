use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, Bound};
use splashsurf_lib::{
    nalgebra::Vector3, reconstruct_surface, Aabb3d, GridDecompositionParameters, Index, Real,
    SpatialDecomposition, SurfaceReconstruction,
};

use crate::structs::{PySurfaceReconstructionF32, PySurfaceReconstructionF64};

/// Reconstruct the surface from only particle positions
fn reconstruct_surface_py<I: Index, R: Real>(
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
    if aabb_min == None || aabb_max == None {
        aabb = None;
    } else {
        aabb = Some(Aabb3d::new(
            Vector3::from(aabb_min.unwrap()),
            Vector3::from(aabb_max.unwrap()),
        ));
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
        compact_support_radius: (smoothing_length * particle_radius).times_f64(2.0),
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
) -> PySurfaceReconstructionF32 {
    let particles: PyReadonlyArray2<f32> = particles.extract().unwrap();

    let particle_positions = particles.as_slice().unwrap();
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

    PySurfaceReconstructionF32::new(reconstruction.to_owned())
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
) -> PySurfaceReconstructionF64 {
    let particles: PyReadonlyArray2<f64> = particles.extract().unwrap();

    let particle_positions = particles.as_slice().unwrap();
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

    PySurfaceReconstructionF64::new(reconstruction.to_owned())
}
