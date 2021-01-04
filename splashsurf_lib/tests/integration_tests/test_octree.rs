use std::path::Path;

use splashsurf_lib::mesh::PointCloud3d;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::octree::Octree;
use splashsurf_lib::{grid_for_reconstruction, Index, Real, UniformGrid};

use vtkio::model::UnstructuredGridPiece;

use super::io;

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

#[allow(dead_code)]
fn octree_to_file<P: AsRef<Path>, I: Index, R: Real>(
    octree: &Octree<I>,
    grid: &UniformGrid<I, R>,
    path: P,
) {
    let mesh = octree.hexmesh(&grid, false);
    io::vtk::write_vtk(UnstructuredGridPiece::from(&mesh), path.as_ref(), "octree").unwrap();
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
    println!("{:?}", grid);

    let mut octree = Octree::new(&grid, particles.as_slice().len());
    octree.subdivide_recursively(&grid, particles.as_slice(), 30);

    let root = octree.root();
    println!("min: {:?}, max: {:?}", root.min_corner(), root.max_corner());
    println!(
        "min coord: {:?}, max coord: {:?}",
        grid.point_coordinates(root.min_corner()),
        grid.point_coordinates(root.max_corner())
    );

    //particles_to_file(particles, "U:\\particles.vtk");
    //octree_to_file(&octree, &grid, "U:\\octree.vtk");
}

/// Loads particles from a VTK file and checks that octree contains all particles and that each particle is in the correct leaf
#[test]
fn build_octree_from_vtk() {
    let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
    let particles = io::vtk::particles_from_vtk::<f64, _>(file).unwrap();
    println!("Loaded {} particles from {}", particles.len(), file);

    let grid = grid_for_reconstruction::<i64, _>(particles.as_slice(), 0.025, 0.2, None).unwrap();
    println!("{:?}", grid);

    let mut octree = Octree::new(&grid, particles.as_slice().len());
    octree.subdivide_recursively(&grid, particles.as_slice(), 60);

    // Sum the number of particles per leaf
    let mut particle_count = 0;
    for node in octree.depth_first_iter() {
        if let Some(node_particles) = node.particles() {
            println!("Leaf with: {} particles", node_particles.len());
            particle_count += node_particles.len();

            // Ensure that all particles are within extents of octree cell
            let aabb = node.aabb(&grid);
            for idx in node_particles.iter().copied() {
                let particle = particles[idx];
                assert!(aabb.contains_point(&particle));
            }
        }
    }
    // Ensure that every particle was exactly in one cell
    assert_eq!(particle_count, particles.len());

    //octree_to_file(&octree, &grid, "U:\\double_dam_break_frame_26_4732_particles_octree.vtk");
}

#[test]
fn build_octree_par_consistency() {
    let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
    let particles = io::vtk::particles_from_vtk::<f64, _>(file).unwrap();
    println!("Loaded {} particles from {}", particles.len(), file);

    let grid = grid_for_reconstruction::<i64, _>(particles.as_slice(), 0.025, 0.2, None).unwrap();

    let mut octree_seq = Octree::new(&grid, particles.as_slice().len());
    octree_seq.subdivide_recursively(&grid, particles.as_slice(), 20);

    let mut octree_par = Octree::new(&grid, particles.as_slice().len());
    octree_par.subdivide_recursively(&grid, particles.as_slice(), 20);

    let mut particle_count = 0;
    for (node_seq, node_par) in octree_seq
        .depth_first_iter()
        .zip(octree_par.depth_first_iter())
    {
        match (node_seq.particles(), node_par.particles()) {
            (Some(particles_seq), Some(particles_par)) => {
                // Ensure that we have the same number of particles for sequential and parallel result
                assert_eq!(particles_seq.len(), particles_par.len());
                particle_count += particles_seq.len();

                // Sort the particle lists
                let mut particles_seq = particles_seq.to_vec();
                let mut particles_par = particles_par.to_vec();
                particles_seq.sort_unstable();
                particles_par.sort_unstable();

                // Ensure that the particle lists are identical
                assert_eq!(particles_seq, particles_par);
            }
            (None, None) => {
                // Both nodes are leaves
                continue;
            }
            _ => {
                // Parallel and sequential node do not match (one is leaf, one has children)
                assert!(false);
            }
        }
    }

    // Check if all particles were visited once
    assert_eq!(particle_count, particles.len());
}
