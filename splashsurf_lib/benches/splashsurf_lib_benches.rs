mod benches;

use benches::bench_octree::bench_octree;
use criterion::criterion_main;

criterion_main!(bench_octree);
