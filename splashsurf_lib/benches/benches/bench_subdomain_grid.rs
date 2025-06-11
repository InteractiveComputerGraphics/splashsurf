use criterion::{Criterion, SamplingMode, criterion_group};
use nalgebra::Vector3;
use splashsurf_lib::io::particles_from_file;
use splashsurf_lib::{
    GridDecompositionParameters, Parameters, SpatialDecomposition, SurfaceReconstruction,
    reconstruct_surface,
};
use std::time::Duration;

static CANYON_PATH: &str = "C:\\canyon.xyz";

fn parameters_canyon() -> Parameters<f32> {
    let particle_radius = 0.011;
    let compact_support_radius = 4.0 * particle_radius;
    let cube_size = 1.5 * particle_radius;

    Parameters {
        particle_radius,
        rest_density: 1000.0,
        compact_support_radius,
        cube_size,
        iso_surface_threshold: 0.6,
        particle_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: Some(SpatialDecomposition::UniformGrid(
            GridDecompositionParameters {
                subdomain_num_cubes_per_dim: 32,
                ..Default::default()
            },
        )),
        global_neighborhood_list: false,
    }
}

pub fn grid_canyon(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> = &particles_from_file(CANYON_PATH).unwrap();
    let parameters = parameters_canyon();

    let mut group = c.benchmark_group("grid_canyon");
    group.sample_size(12);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(180));

    let mut reconstruction = SurfaceReconstruction::default();

    group.bench_function("canyon_c_1_50_nc_32", |b| {
        b.iter(|| {
            let mut parameters = parameters.clone();
            parameters.cube_size = 1.5 * parameters.particle_radius;
            parameters.spatial_decomposition = Some(SpatialDecomposition::UniformGrid(
                GridDecompositionParameters {
                    subdomain_num_cubes_per_dim: 32,
                    ..Default::default()
                },
            ));
            reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.bench_function("canyon_c_1_00_nc_48", |b| {
        b.iter(|| {
            let mut parameters = parameters.clone();
            parameters.cube_size = 1.0 * parameters.particle_radius;
            parameters.spatial_decomposition = Some(SpatialDecomposition::UniformGrid(
                GridDecompositionParameters {
                    subdomain_num_cubes_per_dim: 48,
                    ..Default::default()
                },
            ));
            reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.bench_function("canyon_c_0_75_nc_64", |b| {
        b.iter(|| {
            let mut parameters = parameters.clone();
            parameters.cube_size = 0.75 * parameters.particle_radius;
            parameters.spatial_decomposition = Some(SpatialDecomposition::UniformGrid(
                GridDecompositionParameters {
                    subdomain_num_cubes_per_dim: 64,
                    ..Default::default()
                },
            ));
            reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });
}

pub fn grid_optimal_num_cubes_canyon(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> = &particles_from_file(CANYON_PATH).unwrap();
    let mut parameters = parameters_canyon();

    let mut with_cube_factor = |cube_factor: f32, num_cubes: &[u32]| {
        parameters.cube_size = cube_factor * parameters.particle_radius;

        let mut group = c.benchmark_group(format!(
            "grid_optimal_num_cubes_canyon_c_{}",
            format!("{:.2}", cube_factor).replace('.', "_")
        ));
        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);
        group.warm_up_time(Duration::from_secs(10));
        group.measurement_time(Duration::from_secs(120));

        let mut reconstruction = SurfaceReconstruction::default();

        let mut gen_test = |num_cubes: u32| {
            group.bench_function(format!("subdomain_num_cubes_{}", num_cubes), |b| {
                b.iter(|| {
                    let mut parameters = parameters.clone();
                    parameters.spatial_decomposition = Some(SpatialDecomposition::UniformGrid(
                        GridDecompositionParameters {
                            subdomain_num_cubes_per_dim: num_cubes,
                            ..Default::default()
                        },
                    ));
                    reconstruction =
                        reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters)
                            .unwrap()
                })
            });
        };

        for &n in num_cubes {
            gen_test(n);
        }

        group.finish();
    };

    with_cube_factor(1.5, &[18, 24, 28, 32, 40, 48, 56, 64]); // Ideal: 32
    with_cube_factor(1.0, &[28, 32, 40, 48, 56, 64, 68, 72]); // Ideal: 48
    with_cube_factor(0.75, &[40, 48, 56, 64, 68, 72, 80]); // Ideal: 64
    with_cube_factor(0.5, &[48, 56, 64, 68, 72, 78, 82, 96, 104, 112]); // Ideal: 82
}

criterion_group!(
    bench_subdomain_grid,
    grid_canyon,
    grid_optimal_num_cubes_canyon
);
