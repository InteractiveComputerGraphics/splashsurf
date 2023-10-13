mod benches;

use criterion::criterion_main;

use benches::bench_aabb::bench_aabb;
use benches::bench_full::bench_full;
use benches::bench_mesh::bench_mesh;
use benches::bench_neighborhood::bench_neighborhood;
use benches::bench_subdomain_grid::bench_subdomain_grid;

criterion_main!(
    bench_aabb,
    bench_mesh,
    bench_full,
    bench_neighborhood,
    bench_subdomain_grid,
);
