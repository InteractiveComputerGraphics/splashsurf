use criterion::{criterion_group, BatchSize, Criterion};
use nalgebra::Vector3;
use splashsurf_lib::octree::Octree;
use splashsurf_lib::{grid_for_reconstruction, UniformGrid};

use super::io::particles_from_xyz;
use bitflags::_core::time::Duration;

pub fn subdivide_recursively_benchmark(c: &mut Criterion) {
    let particle_positions: &Vec<Vector3<f64>> =
        &particles_from_xyz("../data/canyon_13353401_particles.xyz").unwrap();
    let particles_per_cell = 20000;

    let particle_radius = 0.011;
    let kernel_radius = 4.0 * particle_radius;

    let cube_size = 1.5 * particle_radius;
    let grid: &UniformGrid<i64, f64> = &grid_for_reconstruction(
        particle_positions.as_slice(),
        particle_radius,
        cube_size,
        None,
    )
    .unwrap();

    let mut group = c.benchmark_group("Octree subdivision");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    let get_tree = || Octree::new(&grid, particle_positions.len());

    group.bench_function("subdivide_recursively_margin_par", move |b| {
        b.iter_batched(
            get_tree,
            |mut tree| {
                tree.subdivide_recursively_margin_par(
                    grid,
                    particle_positions.as_slice(),
                    particles_per_cell,
                    kernel_radius,
                )
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(bench_octree, subdivide_recursively_benchmark);
