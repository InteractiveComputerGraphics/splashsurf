mod benches;

use criterion::criterion_main;

use benches::bench_octree::bench_octree;
use benches::bench_full::bench_full;

criterion_main!(bench_octree, bench_full);
