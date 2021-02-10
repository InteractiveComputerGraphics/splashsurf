use splashsurf_lib::generic_tree::VisitableTree;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::octree::Octree;
use splashsurf_lib::{grid_for_reconstruction, Index, Real, SubdivisionCriterion, UniformGrid};
use std::path::Path;

use super::io;

/*
#[allow(dead_code)]
fn particles_to_file<P: AsRef<Path>, R: Real>(particles: Vec<Vector3<R>>, path: P) {
    let points = PointCloud3d { points: particles };
    io::vtk::write_vtk(
        UnstructuredGridPiece::from(&points),
        path.as_ref(),
        "particles",
    )
    .unwrap();
}
 */

#[allow(dead_code)]
fn octree_to_file<P: AsRef<Path>, I: Index, R: Real>(
    octree: &Octree<I, R>,
    grid: &UniformGrid<I, R>,
    path: P,
) {
    let mesh = octree.hexmesh(&grid, false);
    io::vtk::write_vtk(mesh.to_dataset(), path.as_ref(), "octree").unwrap();
}

#[test]
fn build_octree() {
    let particles_per_dim = Vector3::new(20, 10, 6);
    let distance = 0.05;
    let start = Vector3::new(-0.5, -0.5, -0.5);

    let mut particles =
        Vec::with_capacity(particles_per_dim.x * particles_per_dim.y * particles_per_dim.z);

    for i in 0..particles_per_dim.x {
        for j in 0..particles_per_dim.y {
            if ((i as f64) > 0.25 * particles_per_dim.x as f64)
                && ((i as f64) < 0.75 * particles_per_dim.x as f64)
                && ((j as f64) > 0.5 * particles_per_dim.y as f64)
            {
                continue;
            }

            for k in 0..particles_per_dim.z {
                let particle = Vector3::new(
                    start.x + (i + 1) as f64 * distance,
                    start.y + (j + 1) as f64 * distance,
                    start.z + (k + 1) as f64 * distance,
                );
                particles.push(particle);
            }
        }
    }

    let extents = Vector3::new(
        (particles_per_dim.x + 2) as f64,
        (particles_per_dim.y + 2) as f64,
        (particles_per_dim.z + 2) as f64,
    ) * distance;

    let lower_corner = start - Vector3::repeat(distance * 0.11);

    let cube_size = distance * 0.3;
    let n_cubes = extents * (2.0 / cube_size);
    let n_cubes = [
        n_cubes.x.ceil() as i64,
        n_cubes.y.ceil() as i64,
        n_cubes.z.ceil() as i64,
    ];

    let grid = UniformGrid::new(&lower_corner, &n_cubes, cube_size).unwrap();
    //println!("{:?}", grid);

    let mut octree = Octree::new(&grid, particles.as_slice().len());
    octree.subdivide_recursively_margin(
        &grid,
        particles.as_slice(),
        SubdivisionCriterion::MaxParticleCount(30),
        0.0,
        false,
    );

    /*
    let root = octree.root();

    println!("min: {:?}, max: {:?}", root.min_corner(), root.max_corner());
    println!(
        "min coord: {:?}, max coord: {:?}",
        grid.point_coordinates(root.min_corner()),
        grid.point_coordinates(root.max_corner())
    );
    */

    //particles_to_file(particles, "U:\\particles.vtk");
    //octree_to_file(&octree, &grid, "U:\\octree.vtk");
}

/// Loads particles from a VTK file and checks that octree contains all particles and that each particle is in the correct leaf
#[test]
fn build_octree_from_vtk() {
    let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
    let particles = io::vtk::particles_from_vtk::<f64, _>(file).unwrap();
    //println!("Loaded {} particles from {}", particles.len(), file);

    let grid = grid_for_reconstruction::<i64, _>(
        particles.as_slice(),
        0.025,
        4.0 * 0.025,
        0.2,
        None,
        true,
    )
    .unwrap();
    //println!("{:?}", grid);

    let mut octree = Octree::new(&grid, particles.as_slice().len());
    octree.subdivide_recursively_margin(
        &grid,
        particles.as_slice(),
        SubdivisionCriterion::MaxParticleCount(60),
        0.0,
        false,
    );

    // Sum the number of particles per leaf
    let mut particle_count = 0;
    for node in octree.root().dfs_iter() {
        if let Some(particle_set) = node.data().particle_set() {
            let node_particles = &particle_set.particles;
            //println!("Leaf with: {} particles", node_particles.len());
            particle_count += node_particles.len();

            // Ensure that all particles are within extents of octree cell
            let aabb = node.aabb();
            for &idx in node_particles.iter() {
                let particle = particles[idx];
                assert!(aabb.contains_point(&particle));
            }
        }
    }
    // Ensure that every particle was exactly in one cell
    assert_eq!(particle_count, particles.len());

    //octree_to_file(&octree, &grid, "U:\\double_dam_break_frame_26_4732_particles_octree.vtk");
}

struct TestParameters<R: Real> {
    particle_radius: R,
    compact_support_radius: R,
    cube_size: R,
    margin: Option<R>,
    max_particles_per_cell: Option<usize>,
    compare_seq_par: bool,
}

impl<R: Real> Default for TestParameters<R> {
    fn default() -> Self {
        let particle_radius = R::from_f64(0.025).unwrap();
        let compact_support_radius = particle_radius.times_f64(4.0);
        let cube_size = particle_radius.times_f64(0.5);

        Self {
            particle_radius,
            compact_support_radius,
            cube_size,
            margin: None,
            max_particles_per_cell: None,
            compare_seq_par: true,
        }
    }
}

impl<R: Real> TestParameters<R> {
    fn new(particle_radius: f64, compact_support_factor: f64, cube_size_factor: f64) -> Self {
        let params = Self::default();
        params.with_parameters(particle_radius, compact_support_factor, cube_size_factor)
    }

    fn with_margin(mut self, margin: Option<f64>) -> Self {
        self.margin = margin.map(|m| self.particle_radius.times_f64(m));
        self
    }
    fn with_max_particles_per_cell(mut self, max_particles_per_cell: Option<usize>) -> Self {
        self.max_particles_per_cell = max_particles_per_cell;
        self
    }

    fn with_compare_seq_par(mut self, compare_seq_par: bool) -> Self {
        self.compare_seq_par = compare_seq_par;
        self
    }

    fn with_parameters(
        mut self,
        particle_radius: f64,
        compact_support_factor: f64,
        cube_size_factor: f64,
    ) -> Self {
        self.particle_radius = R::from_f64(particle_radius).unwrap();
        self.compact_support_radius = self.particle_radius.times_f64(compact_support_factor);
        self.cube_size = self.particle_radius.times_f64(cube_size_factor);
        self
    }

    fn build_grid<I: Index>(&self, particle_positions: &[Vector3<R>]) -> UniformGrid<I, R> {
        grid_for_reconstruction(
            particle_positions,
            self.particle_radius,
            self.compact_support_radius,
            self.cube_size,
            None,
            true,
        )
        .unwrap()
    }
}

/// Returns a vector containing per particle how often it is a non-ghost particle in the octree
fn count_non_ghost_particles<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    octree: &Octree<I, R>,
) -> Vec<usize> {
    let mut non_ghost_particles = vec![0; particle_positions.len()];
    for node in octree.root().dfs_iter() {
        if let Some(particle_set) = node.data().particle_set() {
            for &idx in particle_set.particles.iter() {
                if node.aabb().contains_point(&particle_positions[idx]) {
                    non_ghost_particles[idx] += 1;
                }
            }
        }
    }

    non_ghost_particles
}

/// Asserts whether each particle has a unique octree node where it is not a ghost particle
fn assert_unique_node_per_particle<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    octree: &Octree<I, R>,
) {
    let non_ghost_particles_counts = count_non_ghost_particles(particle_positions, octree);
    let no_unique_node_particles: Vec<_> = non_ghost_particles_counts
        .into_iter()
        .enumerate()
        .filter(|&(_, count)| count != 1)
        .collect();

    assert_eq!(
        no_unique_node_particles,
        vec![],
        "There are nodes that don't have a unique octree node where they are not ghost particles!"
    );
}

/// Asserts whether both trees have equivalent particle sets in the same octree nodes
fn assert_tree_equivalence<I: Index, R: Real>(left_tree: &Octree<I, R>, right_tree: &Octree<I, R>) {
    for (node_seq, node_par) in left_tree
        .root()
        .dfs_iter()
        .zip(right_tree.root().dfs_iter())
    {
        match (
            node_seq.data().particle_set(),
            node_par.data().particle_set(),
        ) {
            (Some(particles_seq), Some(particles_par)) => {
                let particles_seq = &particles_seq.particles;
                let particles_par = &particles_par.particles;

                // Ensure that we have the same number of particles for sequential and parallel result
                assert_eq!(particles_seq.len(), particles_par.len(), "Found corresponding leaves in the trees that don't have the same number of particles!");

                // Sort the particle lists
                let mut particles_seq = particles_seq.to_vec();
                let mut particles_par = particles_par.to_vec();
                particles_seq.sort_unstable();
                particles_par.sort_unstable();

                // Ensure that the particle lists are identical
                assert_eq!(particles_seq, particles_par, "Found corresponding leaves in the trees that don't have the same sorted particle lists!");
            }
            (None, None) => {
                // Both nodes are leaves
                continue;
            }
            _ => {
                // Parallel and sequential nodes do not match (one is leaf, one has children)
                panic!("Encountered a node where one octree has a particle set but the other does not!");
            }
        }
    }
}

fn build_octree_par_consistency<I: Index, R: Real, P: AsRef<Path>>(
    file: P,
    parameters: TestParameters<R>,
) {
    let particles = io::vtk::particles_from_vtk::<R, _>(file).unwrap();

    let grid = parameters.build_grid::<I>(particles.as_slice());

    let octree_seq = if parameters.compare_seq_par {
        let mut octree_seq = Octree::new(&grid, particles.as_slice().len());
        octree_seq.subdivide_recursively_margin(
            &grid,
            particles.as_slice(),
            parameters
                .max_particles_per_cell
                .map(|n| SubdivisionCriterion::MaxParticleCount(n))
                .unwrap_or(SubdivisionCriterion::MaxParticleCountAuto),
            parameters.margin.unwrap_or(R::zero()),
            false,
        );

        assert_unique_node_per_particle(particles.as_slice(), &octree_seq);

        Some(octree_seq)
    } else {
        None
    };

    let mut octree_par = Octree::new(&grid, particles.as_slice().len());
    octree_par.subdivide_recursively_margin_par(
        &grid,
        particles.as_slice(),
        parameters
            .max_particles_per_cell
            .map(|n| SubdivisionCriterion::MaxParticleCount(n))
            .unwrap_or(SubdivisionCriterion::MaxParticleCountAuto),
        parameters.margin.unwrap_or(R::zero()),
        false,
    );

    assert_unique_node_per_particle(particles.as_slice(), &octree_par);

    if let Some(octree_seq) = octree_seq {
        assert_tree_equivalence(&octree_seq, &octree_par)
    }
}

#[test]
fn build_octree_cube() {
    build_octree_par_consistency::<i64, f64, _>(
        "../data/cube_2366_particles.vtk",
        TestParameters::default()
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(70)),
    );
}

#[test]
fn build_octree_double_dam_break() {
    build_octree_par_consistency::<i64, f64, _>(
        "../data/double_dam_break_frame_26_4732_particles.vtk",
        TestParameters::default()
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(200)),
    );
}

#[test]
fn build_octree_dam_break() {
    build_octree_par_consistency::<i64, f64, _>(
        "../data/dam_break_frame_23_24389_particles.vtk",
        TestParameters::default()
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(1000)),
    );
}

#[test]
fn build_octree_bunny() {
    build_octree_par_consistency::<i64, f64, _>(
        "../data/bunny_frame_14_7705_particles.vtk",
        TestParameters::default()
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(200)),
    );
}

#[test]
fn build_octree_hilbert() {
    build_octree_par_consistency::<i64, f64, _>(
        "../data/hilbert_46843_particles.vtk",
        TestParameters::default()
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(1000)),
    );
}

/*
#[test]
fn build_octree_canyon() {
    build_octree_par_consistency::<i64, f64, _>(
        "../../canyon_13353401_particles.vtk",
        TestParameters::new(0.011, 4.0, 1.5)
            .with_margin(Some(1.0))
            .with_max_particles_per_cell(Some(52161))
            .with_compare_seq_par(false),
    );
}
*/
