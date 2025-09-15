use criterion::{Criterion, criterion_group};
use nalgebra::{Scalar, Vector3};
use serde_derive::{Deserialize, Serialize};
use splashsurf_lib::dense_subdomains;
use splashsurf_lib::kernel::CubicSplineKernel;
use splashsurf_lib::uniform_grid::UniformCartesianCubeGrid3d;
use std::time::Duration;

// Code to generate the input data for the benchmark from dense_subdomains.rs
/*
if subdomain_particles.len() > 6800 {
    println!("{}", subdomain_particles.len());
    let arguments = DensityGridLoopParameters {
        levelset_grid: levelset_grid.clone(),
        subdomain_particles: subdomain_particles.clone(),
        subdomain_particle_densities: subdomain_particle_densities.clone(),
        subdomain_mc_grid: mc_grid.clone(),
        subdomain_ijk: *subdomain_idx.index(),
        global_mc_grid: parameters.global_marching_cubes_grid.clone(),
        cube_radius,
        squared_support_with_margin,
        particle_rest_mass: parameters.particle_rest_mass,
        compact_support_radius: parameters.compact_support_radius,
    };
    serde_json::to_writer(
        std::fs::File::create(format!(
            "density_grid_loop_subdomain_{}.json",
            flat_subdomain_idx
        ))
        .unwrap(),
        &arguments,
    )
    .unwrap();
}
 */

#[derive(Clone, Serialize, Deserialize)]
pub struct DensityGridLoopParameters<I: Scalar, R: Scalar> {
    pub levelset_grid: Vec<R>,
    pub subdomain_particles: Vec<Vector3<R>>,
    pub subdomain_particle_densities: Vec<R>,
    pub subdomain_mc_grid: UniformCartesianCubeGrid3d<I, R>,
    pub subdomain_ijk: [I; 3],
    pub global_mc_grid: UniformCartesianCubeGrid3d<i64, R>,
    pub cube_radius: I,
    pub squared_support_with_margin: R,
    pub particle_rest_mass: R,
    pub compact_support_radius: R,
}

pub fn grid_loop_no_simd(c: &mut Criterion) {
    let params: DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
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
            dense_subdomains::density_grid_loop_scalar(
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

#[allow(unreachable_code)]
#[allow(unused_variables)]
pub fn grid_loop_neon(c: &mut Criterion) {
    #[cfg(not(target_feature = "neon"))]
    {
        println!("Skipping NEON benchmark because the target does not support NEON.");
        return;
    }

    let params: DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("../data/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let reference = {
        let kernel = CubicSplineKernel::new(params.compact_support_radius);
        let mut params = params.clone();
        dense_subdomains::density_grid_loop_scalar(
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

        let eps = f32::EPSILON * 10.0;
        println!("{:?}", params.subdomain_mc_grid);
        for (i, (a, b)) in neon_result.iter().zip(reference.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff >= eps {
                let ijk = params.subdomain_mc_grid.try_unflatten_point_index(i as i64);
                println!(
                    "Mismatch at index {}: neon = {}, reference = {}, diff = {}, ijk = {:?}",
                    i, a, b, diff, ijk
                );
            }
        }

        println!("reference head: {:?}", &reference[0..10]);
        println!("neon head     : {:?}", &reference[0..10]);

        println!(
            "reference tail: {:?}",
            &neon_result[neon_result.len() - 10..]
        );
        println!(
            "neon tail     : {:?}",
            &neon_result[neon_result.len() - 10..]
        );

        assert!(
            neon_result
                .iter()
                .zip(reference.iter())
                .all(|(a, b)| (a - b).abs() < eps)
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

#[allow(unreachable_code)]
#[allow(unused_variables)]
pub fn grid_loop_avx2(c: &mut Criterion) {
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        println!("Skipping AVX2 benchmark because the target does not support AVX2 and FMA.");
        return;
    }

    let params: DensityGridLoopParameters<i64, f32> = serde_json::from_reader(
        std::fs::File::open("../data/density_grid_loop_subdomain_33.json").unwrap(),
    )
    .unwrap();

    let reference = {
        let kernel = CubicSplineKernel::new(params.compact_support_radius);
        let mut params = params.clone();
        dense_subdomains::density_grid_loop_scalar(
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

    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        let avx_result = unsafe {
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
        println!("{:?}", &avx_result[0..10]);

        assert!(
            avx_result
                .iter()
                .zip(reference.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON * 100.0)
        );
    }

    let mut group = c.benchmark_group("grid_loop");
    group.sample_size(600);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));

    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
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
