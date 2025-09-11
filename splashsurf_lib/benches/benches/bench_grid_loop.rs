use criterion::{Criterion, criterion_group};
use std::time::Duration;
use splashsurf_lib::dense_subdomains::{density_grid_loop, DensityGridLoopParameters};
use splashsurf_lib::kernel::CubicSplineKernel;

pub fn grid_loop_no_simd(c: &mut Criterion) {
    let params: DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("/Users/floeschner/programming/splashsurf_private/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let mut group = c.benchmark_group("mesh");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("grid_loop_no_simd", |b| {
        b.iter(|| {
            let kernel = CubicSplineKernel::new(params.compact_support_radius);
            let mut params = params.clone();
            density_grid_loop(
                params.levelset_grid.as_mut_slice(),
                params.subdomain_particles.as_slice(),
                params.subdomain_particle_densities.as_slice(),
                &params.subdomain_mc_grid,
                &params.subdomain_ijk,
                &params.global_mc_grid,
                params.cube_radius,
                params.squared_support_with_margin,
                params.particle_rest_mass,
                &kernel,
            );
        })
    });

    group.finish();
}

criterion_group!(bench_grid_loop, grid_loop_no_simd);
