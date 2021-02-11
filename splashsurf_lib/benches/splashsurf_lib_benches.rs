mod benches;

use criterion::criterion_main;

use benches::bench_aabb::bench_aabb;
use benches::bench_full::bench_full;
use benches::bench_mesh::bench_mesh;
use benches::bench_octree::bench_octree;

criterion_main!(bench_aabb, bench_mesh, bench_octree, bench_full);
