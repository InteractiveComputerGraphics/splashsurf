use criterion::{criterion_group, Criterion};
use nalgebra::Vector3;
use splashsurf_lib::{
    reconstruct_surface, reconstruct_surface_inplace, Parameters, SpatialDecompositionParameters,
    SubdivisionCriterion, SurfaceReconstruction,
};
use std::time::Duration;

use super::io::vtk::particles_from_vtk;
use super::io::xyz::particles_from_xyz;

#[allow(dead_code)]
use super::io::vtk::write_vtk;

pub fn surface_reconstruction_canyon(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_xyz("../data/canyon_13353401_particles.xyz").unwrap();

    let particle_radius = 0.011;
    let kernel_radius = 4.0 * particle_radius;
    let cube_size = 1.5 * particle_radius;

    let parameters = Parameters {
        particle_radius,
        rest_density: 1000.0,
        kernel_radius,
        splash_detection_radius: None,
        cube_size,
        iso_surface_threshold: 0.6,
        domain_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: Some(SpatialDecompositionParameters {
            subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
            ghost_particle_safety_factor: Some(1.0),
        }),
    };

    let mut group = c.benchmark_group("full surface reconstruction");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("surface_reconstruction_canyon", move |b| {
        b.iter(|| {
            reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.finish();
}

pub fn surface_reconstruction_dam_break(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_vtk("../data/dam_break_frame_23_24389_particles.vtk").unwrap();

    let particle_radius = 0.025;
    let kernel_radius = 4.0 * particle_radius;
    let cube_size = 0.3 * particle_radius;

    let parameters = Parameters {
        particle_radius,
        rest_density: 1000.0,
        kernel_radius,
        splash_detection_radius: None,
        cube_size,
        iso_surface_threshold: 0.6,
        domain_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: Some(SpatialDecompositionParameters {
            subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
            ghost_particle_safety_factor: Some(1.0),
        }),
    };

    let mut group = c.benchmark_group("full surface reconstruction");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(15));

    let mut reconstruction = SurfaceReconstruction::default();
    group.bench_function("surface_reconstruction_dam_break", |b| {
        b.iter(|| {
            reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.finish();

    /*
    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_dam_break.vtk",
        "mesh",
    )
    .unwrap();
     */
}

pub fn surface_reconstruction_double_dam_break(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_vtk("../data/double_dam_break_frame_26_4732_particles.vtk").unwrap();

    let particle_radius = 0.025;
    let kernel_radius = 4.0 * particle_radius;
    let cube_size = 0.3 * particle_radius;

    let parameters = Parameters {
        particle_radius,
        rest_density: 1000.0,
        kernel_radius,
        splash_detection_radius: None,
        cube_size,
        iso_surface_threshold: 0.6,
        domain_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: Some(SpatialDecompositionParameters {
            subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
            ghost_particle_safety_factor: Some(1.0),
        }),
    };

    let mut group = c.benchmark_group("full surface reconstruction");
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(25));

    let mut reconstruction = SurfaceReconstruction::default();
    group.bench_function("surface_reconstruction_double_dam_break", |b| {
        b.iter(|| {
            reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.finish();

    /*
    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_double_dam_break.vtk",
        "mesh",
    )
    .unwrap();
     */
}

pub fn surface_reconstruction_double_dam_break_inplace(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_vtk("../data/double_dam_break_frame_26_4732_particles.vtk").unwrap();

    let particle_radius = 0.025;
    let kernel_radius = 4.0 * particle_radius;
    let cube_size = 0.3 * particle_radius;

    let parameters = Parameters {
        particle_radius,
        rest_density: 1000.0,
        kernel_radius,
        splash_detection_radius: None,
        cube_size,
        iso_surface_threshold: 0.6,
        domain_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: Some(SpatialDecompositionParameters {
            subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
            ghost_particle_safety_factor: Some(1.0),
        }),
    };

    let mut group = c.benchmark_group("full surface reconstruction");
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(25));

    let mut reconstruction = SurfaceReconstruction::default();
    group.bench_function("surface_reconstruction_double_dam_break_inplace", |b| {
        b.iter(|| {
            reconstruct_surface_inplace::<i64, _>(
                particle_positions.as_slice(),
                &parameters,
                &mut reconstruction,
            )
            .unwrap()
        })
    });

    group.finish();

    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_double_dam_break.vtk",
        "mesh",
    )
    .unwrap();
}

criterion_group!(
    bench_full,
    surface_reconstruction_double_dam_break,
    surface_reconstruction_double_dam_break_inplace,
    surface_reconstruction_dam_break,
    surface_reconstruction_canyon
);
