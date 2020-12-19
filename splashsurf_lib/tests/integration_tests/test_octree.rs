use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::octree::Octree;
use splashsurf_lib::{grid_for_reconstruction, Real};

use super::io;

#[test]
fn octree_from_vtk() {
    let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
    let particles = io::vtk::particles_from_vtk::<f64, _>(file).unwrap();
    println!("Loaded {} particles from {}", particles.len(), file);

    let grid = grid_for_reconstruction::<i64, _>(particles.as_slice(), 0.025, 0.2, None).unwrap();

    println!("{:?}", grid);

    let octree = Octree::new(&grid, particles.as_slice(), 60);

    let mut particle_count = 0;
    for node in octree.depth_first_iter() {
        if let Some(particles) = node.particles() {
            println!("Leaf with: {} particles", particles.len());
            particle_count += particles.len();
        }
    }
    assert_eq!(particle_count, particles.len());

    let mesh = octree.into_hexmesh(&grid);
    use vtkio::model::UnstructuredGridPiece;
    io::vtk::write_vtk(
        UnstructuredGridPiece::from(&mesh),
        "U:\\octree.vtk",
        "octree",
    )
    .unwrap();
}

/*
#[test]
fn octree_from_random() {
    let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
    let particles = vtk::particles_from_vtk::<f64, _>(file).unwrap();
    println!("Loaded {} particles from {}", particles.len(), file);

    let grid =
        crate::grid_for_reconstruction::<i64, _>(particles.as_slice(), 0.025, 0.2, None).unwrap();

    println!("{:?}", grid);

    let octree = Octree::new(&grid, particles.as_slice(), 60);

    let mut particle_count = 0;
    for node in octree.depth_first_iter() {
        if let Some(particles) = node.particles() {
            println!("Leaf with: {} particles", particles.len());
            particle_count += particles.len();
        }
    }
    assert_eq!(particle_count, particles.len());

    let mesh = octree.into_hexmesh(&grid);
    use vtkio::model::UnstructuredGridPiece;
    vtk::write_vtk(
        UnstructuredGridPiece::from(&mesh),
        "U:\\octree.vtk",
        "octree",
    )
    .unwrap();
}
*/
