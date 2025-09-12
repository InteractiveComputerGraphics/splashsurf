use criterion::{Criterion, criterion_group};
use splashsurf_lib::dense_subdomains;
use splashsurf_lib::kernel::CubicSplineKernel;
use std::time::Duration;

pub fn grid_loop_no_simd(c: &mut Criterion) {
    let params: dense_subdomains::DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("../data/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let mut group = c.benchmark_group("grid_loop");
    group.sample_size(400);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("grid_loop_no_simd", |b| {
        b.iter(|| {
            let kernel = CubicSplineKernel::new(params.compact_support_radius);
            let mut params = params.clone();
            dense_subdomains::density_grid_loop(
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

pub fn grid_loop_neon(c: &mut Criterion) {
    #[cfg(not(target_feature = "neon"))]
    {
        println!("Skipping NEON benchmark because the target does not support NEON.");
        return;
    }

    let params: dense_subdomains::DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("../data/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let reference = {
        let kernel = CubicSplineKernel::new(params.compact_support_radius);
        let mut params = params.clone();
        dense_subdomains::density_grid_loop(
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
        params.levelset_grid
    };

    #[cfg(target_feature = "neon")]
    {
        let neon_result = unsafe {
            let kernel = CubicSplineKernel::new(params.compact_support_radius);
            let mut params = params.clone();
            dense_subdomains::density_grid_loop_neon(
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
            params.levelset_grid
        };

        println!("{:?}", &reference[0..10]);
        println!("{:?}", &neon_result[0..10]);

        assert!(
            neon_result
                .iter()
                .zip(reference.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0)
        );
    }

    let mut group = c.benchmark_group("grid_loop");
    group.sample_size(600);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));

    #[cfg(target_feature = "neon")]
    unsafe {
        group.bench_function("grid_loop_neon", |b| {
            b.iter(|| {
                let kernel = CubicSplineKernel::new(params.compact_support_radius);
                let mut params = params.clone();
                dense_subdomains::density_grid_loop_neon(
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
    }

    group.finish();
}

pub fn grid_loop_avx2(c: &mut Criterion) {
    #[cfg(not(target_feature = "avx2"))]
    {
        println!("Skipping AVX2 benchmark because the target does not support AVX2.");
        return;
    }

    let params: dense_subdomains::DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("../data/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let reference = {
        let kernel = CubicSplineKernel::new(params.compact_support_radius);
        let mut params = params.clone();
        dense_subdomains::density_grid_loop(
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
        params.levelset_grid
    };

    #[cfg(target_feature = "avx2")]
    {
        let neon_result = unsafe {
            let kernel = CubicSplineKernel::new(params.compact_support_radius);
            let mut params = params.clone();
            dense_subdomains::density_grid_loop_avx(
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
            params.levelset_grid
        };

        println!("{:?}", &reference[0..10]);
        println!("{:?}", &neon_result[0..10]);

        assert!(
            neon_result
                .iter()
                .zip(reference.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0)
        );
    }

    let mut group = c.benchmark_group("grid_loop");
    group.sample_size(600);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));

    #[cfg(target_feature = "avx2")]
    unsafe {
        group.bench_function("grid_loop_avx2", |b| {
            b.iter(|| {
                let kernel = CubicSplineKernel::new(params.compact_support_radius);
                let mut params = params.clone();
                dense_subdomains::density_grid_loop_avx(
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
    }

    group.finish();
}

criterion_group!(
    bench_grid_loop,
    grid_loop_no_simd,
    grid_loop_neon,
    grid_loop_avx2
);
