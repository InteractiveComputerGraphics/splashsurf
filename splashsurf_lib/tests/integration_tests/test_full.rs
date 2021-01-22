use nalgebra::Vector3;
use splashsurf_lib::{
    reconstruct_surface, Parameters, SpatialDecompositionParameters, SubdivisionCriterion,
};

use super::io::vtk::{particles_from_vtk, write_vtk};

#[test]
pub fn surface_reconstruction_bunny_global() {
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

    // Ensure that the mesh does not have a boundary
    assert!(reconstruction.mesh().find_boundary_edges().is_empty());

    // TODO: Compare with a solution file
    // TODO: Test with a fixed grid?

    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_bunny_par_global.vtk",
        "mesh",
    )
    .unwrap();
}

#[test]
pub fn surface_reconstruction_bunny_stitching() {
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
        spatial_decomposition: Some(SpatialDecompositionParameters {
            subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
            ghost_particle_safety_factor: Some(1.0),
            enable_stitching: true,
        }),
    };

    let reconstruction =
        reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

    // Ensure that the mesh does not have a boundary
    assert!(reconstruction.mesh().find_boundary_edges().is_empty());

    // TODO: Compare with a solution file
    // TODO: Test with a fixed grid?

    write_vtk(
        reconstruction.mesh(),
        "../out/reconstruct_surface_bunny_par_stitching.vtk",
        "mesh",
    )
    .unwrap();
}
