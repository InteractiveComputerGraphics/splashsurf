use criterion::{criterion_group, Criterion};
use nalgebra::Vector3;
use splashsurf_lib::{
    reconstruct_surface, Parameters, SpatialDecompositionParameters, SubdivisionCriterion,
};
use std::time::Duration;

use super::io::particles_from_xyz;

pub fn surface_reconstruction(c: &mut Criterion) {
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

    group.bench_function("reconstruct_surface", move |b| {
        b.iter(|| {
            reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap()
        })
    });

    group.finish();
}

criterion_group!(bench_full, surface_reconstruction);
