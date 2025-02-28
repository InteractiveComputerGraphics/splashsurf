use numpy::{Element, PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyFloat, Bound, IntoPyObjectExt, Py, PyAny, Python};
use splashsurf_lib::{
    nalgebra::Vector3, reconstruct_surface, Aabb3d, GridDecompositionParameters, Index, Real,
    SpatialDecomposition, SurfaceReconstruction,
};

use crate::structs::{PySurfaceReconstructionF32, PySurfaceReconstructionF64};

/// Reconstruct the surface from only particle positions
fn reconstruct_surface_py<I: Index, R: Real>(
    particles: Vec<Vector3<R>>,
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

fn reconstruct_surface_py_interface<'py, R: Real + Element>(
    py: Python<'py>,
    particles: &Bound<'py, PyArray2<R>>,
    particle_radius: Py<PyFloat>,
    rest_density: Py<PyFloat>,
    smoothing_length: Py<PyFloat>,
    cube_size: Py<PyFloat>,
    iso_surface_threshold: Py<PyFloat>,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[Py<PyFloat>; 3]>,
    aabb_max: Option<[Py<PyFloat>; 3]>,
) -> SurfaceReconstruction<i64, R> {
    let particles: PyReadonlyArray2<R> = particles.extract().unwrap();
    let particle_positions: Vec<Vector3<R>> = particles
        .as_array()
        .outer_iter()
        .map(|row| Vector3::new(row[0], row[1], row[2]))
        .collect();

    let aabb_min: Option<[R; 3]> = aabb_min.map(|x| {
        let mut res = [R::zero(); 3];
        for i in 0..3 {
            res[i] = R::from_f64(x[i].extract::<f64>(py).unwrap()).unwrap();
        }
        res
    });

    let aabb_max: Option<[R; 3]> = aabb_max.map(|x| {
        let mut res = [R::zero(); 3];
        for i in 0..3 {
            res[i] = R::from_f64(x[i].extract::<f64>(py).unwrap()).unwrap();
        }
        res
    });

    let reconstruction = reconstruct_surface_py::<i64, R>(
        particle_positions,
        R::from_f64(particle_radius.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(rest_density.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(smoothing_length.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(cube_size.extract::<f64>(py).unwrap()).unwrap(),
        R::from_f64(iso_surface_threshold.extract::<f64>(py).unwrap()).unwrap(),
        enable_multi_threading,
        global_neighborhood_list,
        use_custom_grid_decomposition,
        subdomain_num_cubes_per_dim,
        aabb_min,
        aabb_max,
    );

    reconstruction
}

#[pyfunction]
#[pyo3(name = "reconstruct_surface")]
#[pyo3(signature = (particles, *, particle_radius, rest_density,
    smoothing_length, cube_size, iso_surface_threshold, enable_multi_threading=false,
    global_neighborhood_list=false, use_custom_grid_decomposition=false, subdomain_num_cubes_per_dim=64,
    aabb_min = None, aabb_max = None
))]
pub fn reconstruct_surface_py_dynamic<'py>(
    py: Python<'py>,
    particles: &Bound<'py, PyAny>,
    particle_radius: Py<PyFloat>,
    rest_density: Py<PyFloat>,
    smoothing_length: Py<PyFloat>,
    cube_size: Py<PyFloat>,
    iso_surface_threshold: Py<PyFloat>,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: u32,
    aabb_min: Option<[Py<PyFloat>; 3]>,
    aabb_max: Option<[Py<PyFloat>; 3]>,
) -> Bound<'py, PyAny> {
    if let Ok(particles) = particles.downcast::<PyArray2<f32>>() {
        let reconstruction = reconstruct_surface_py_interface::<f32>(
            py,
            particles,
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

        PySurfaceReconstructionF32::new(reconstruction.clone())
            .into_bound_py_any(py)
            .unwrap()
        
    } else if let Ok(particles) = particles.downcast::<PyArray2<f64>>() {
        let reconstruction = reconstruct_surface_py_interface::<f64>(
            py,
            particles,
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

        PySurfaceReconstructionF64::new(reconstruction.clone())
            .into_bound_py_any(py)
            .unwrap()

    } else {
        panic!("Couldn't convert particles to f32 or f64 array!")
    }
}
