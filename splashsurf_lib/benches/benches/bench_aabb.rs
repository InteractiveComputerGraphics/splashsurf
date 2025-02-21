use criterion::{Criterion, criterion_group};
use nalgebra::Vector3;
use splashsurf_lib::Aabb3d;
use splashsurf_lib::io::particles_from_file;
use std::time::Duration;

pub fn aabb_from_points(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_file("../data/hilbert_46843_particles.bgeo").unwrap();

    let mut group = c.benchmark_group("aabb");
    group.sample_size(200);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("aabb_from_points", move |b| {
        b.iter(|| Aabb3d::from_points(particle_positions))
    });

    group.finish();
}

pub fn aabb_from_points_par(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_file("../data/hilbert_46843_particles.bgeo").unwrap();

    let mut group = c.benchmark_group("aabb");
    group.sample_size(500);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("aabb_from_points_par", move |b| {
        b.iter(|| Aabb3d::par_from_points(particle_positions))
    });

    group.finish();
}

criterion_group!(bench_aabb, aabb_from_points, aabb_from_points_par);
