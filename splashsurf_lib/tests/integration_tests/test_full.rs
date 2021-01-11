use nalgebra::Vector3;
use splashsurf_lib::{reconstruct_surface, Parameters};

use super::io::vtk::{particles_from_vtk, write_vtk};

#[test]
pub fn surface_reconstruction_bunny() {
    let particle_positions: &Vec<Vector3<f32>> =
        &particles_from_vtk("../data/bunny_frame_14_7705_particles.vtk").unwrap();

    let particle_radius = 0.025;
    let kernel_radius = 4.0 * particle_radius;
    let cube_size = 0.75 * particle_radius;

    let parameters = Parameters {
        particle_radius,
        rest_density: 1000.0,
        kernel_radius,
        splash_detection_radius: None,
        cube_size,
        iso_surface_threshold: 0.6,
        domain_aabb: None,
        enable_multi_threading: true,
        spatial_decomposition: None,
    };

    let reconstruction =
        reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

    // TODO: Compare with a solution file
    // TODO: Test with a fixed grid?

    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_bunny.vtk",
        "mesh",
    )
    .unwrap();
}
