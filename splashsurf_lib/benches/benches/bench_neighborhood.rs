use criterion::{criterion_group, Criterion};
use nalgebra::Vector3;
use splashsurf_lib::io;
use splashsurf_lib::{neighborhood_search, AxisAlignedBoundingBox3d};
use std::time::Duration;

static PARTICLE_RADIUS: f64 = 0.025;
static COMPACT_SUPPORT_RADIUS: f64 = 4.0 * PARTICLE_RADIUS;

//static NUM_PARTICLES: Option<usize> = Some(800);
static NUM_PARTICLES: Option<usize> = None;

static PARTICLE_FILE: &'static str = "../data/bunny_frame_14_7705_particles.vtk";

fn particle_subset(particle_positions: &[Vector3<f32>]) -> &[Vector3<f32>] {
    if let Some(n_particles) = NUM_PARTICLES.clone() {
        &particle_positions[0..n_particles]
    } else {
        &particle_positions[..]
    }
}

pub fn neighborhood_search_naive(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &io::vtk_format::particles_from_vtk(PARTICLE_FILE).unwrap();
    let particle_positions = particle_subset(particle_positions.as_slice());

    let mut group = c.benchmark_group("neighborhood_search");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let mut neighborhood_lists = Vec::with_capacity(particle_positions.len());
    group.bench_function("neighborhood_search_naive", move |b| {
        b.iter(|| {
            neighborhood_lists.clear();
            neighborhood_search::neighborhood_search_naive(
                &particle_positions,
                COMPACT_SUPPORT_RADIUS as f32,
                &mut neighborhood_lists,
            );
        })
    });

    group.finish();
}

pub fn neighborhood_search_spatial_hashing(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &io::vtk_format::particles_from_vtk(PARTICLE_FILE).unwrap();
    let particle_positions = particle_subset(particle_positions.as_slice());

    let domain = AxisAlignedBoundingBox3d::from_points(particle_positions)
        .grow_uniformly(COMPACT_SUPPORT_RADIUS as f32);

    let mut group = c.benchmark_group("neighborhood_search");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let mut neighborhood_lists = Vec::with_capacity(particle_positions.len());
    group.bench_function("neighborhood_search_spatial_hashing", move |b| {
        b.iter(|| {
            neighborhood_lists.clear();
            neighborhood_search::neighborhood_search_spatial_hashing::<i32, f32>(
                &domain,
                &particle_positions,
                COMPACT_SUPPORT_RADIUS as f32,
                &mut neighborhood_lists,
            );
        })
    });

    group.finish();
}

pub fn neighborhood_search_spatial_hashing_parallel(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f32>> =
        &io::vtk_format::particles_from_vtk(PARTICLE_FILE).unwrap();
    let particle_positions = particle_subset(particle_positions.as_slice());

    let domain = AxisAlignedBoundingBox3d::from_points(particle_positions)
        .grow_uniformly(COMPACT_SUPPORT_RADIUS as f32);

    let mut group = c.benchmark_group("neighborhood_search");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let mut neighborhood_lists = Vec::with_capacity(particle_positions.len());
    group.bench_function("neighborhood_search_spatial_hashing_parallel", move |b| {
        b.iter(|| {
            neighborhood_lists.clear();
            neighborhood_search::neighborhood_search_spatial_hashing_parallel::<i32, f32>(
                &domain,
                &particle_positions,
                COMPACT_SUPPORT_RADIUS as f32,
                &mut neighborhood_lists,
            );
        })
    });

    group.finish();
}

criterion_group!(
    bench_neighborhood,
    neighborhood_search_naive,
    neighborhood_search_spatial_hashing,
    neighborhood_search_spatial_hashing_parallel,
);
