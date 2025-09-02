use splashsurf_lib::io::vtk_format;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{Aabb3d, GridDecompositionParameters, SpatialDecomposition};

use all_asserts::assert_range;

#[test]
fn test_basic_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let particles =
        splashsurf_lib::io::particles_from_file("../data/bunny_frame_14_7705_particles.vtk")?;

    let particle_radius = 0.025;
    let mut params = splashsurf_lib::Parameters::new_relative(particle_radius, 4.0, 1.0);
    params.spatial_decomposition = SpatialDecomposition::None;

    let mut postprocessing = splashsurf::reconstruct::ReconstructionPostprocessingParameters::new();
    postprocessing.output_raw_mesh = true;

    let reconstruction = splashsurf::reconstruct::reconstruction_pipeline::<i64, _>(
        &particles,
        &[],
        &params,
        &postprocessing,
    )?;
    let mesh = &reconstruction
        .tri_mesh
        .as_ref()
        .expect("reconstruction should produce a triangle mesh")
        .mesh;
    let raw_mesh = &reconstruction.raw_reconstruction.mesh;
    vtk_format::write_vtk(mesh, "../out/bunny_test_basic_pipeline.vtk", "mesh")?;

    // Compare raw and final mesh
    assert_eq!(
        mesh.triangles.len(),
        raw_mesh.triangles.len(),
        "Number of triangles should match between raw and final mesh"
    );
    assert_eq!(
        mesh.vertices.len(),
        raw_mesh.vertices.len(),
        "Number of vertices should match between raw and final mesh"
    );
    assert_eq!(
        mesh.triangles, raw_mesh.triangles,
        "Triangle indices should match between raw and final mesh"
    );
    // Check number of triangles and vertices
    assert_range!(
        (35000..40000),
        raw_mesh.triangles.len(),
        "Number of triangles should be in expected range"
    );
    assert_range!(
        (15000..20000),
        raw_mesh.vertices.len(),
        "Number of vertices should be in expected range"
    );
    // Check manifoldness
    let mesh_manifold_information = mesh.compute_manifold_information();
    assert!(
        mesh_manifold_information.is_closed(),
        "Mesh should be closed"
    );
    assert!(
        mesh_manifold_information.is_manifold(),
        "Mesh should be manifold"
    );

    Ok(())
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn test_basic_pipeline_fine() -> Result<(), Box<dyn std::error::Error>> {
    let particles =
        splashsurf_lib::io::particles_from_file("../data/bunny_frame_14_7705_particles.vtk")?;

    let particle_radius = 0.025;
    let mut params = splashsurf_lib::Parameters::new_relative(particle_radius, 4.0, 0.2);
    params.spatial_decomposition = SpatialDecomposition::UniformGrid(GridDecompositionParameters {
        subdomain_num_cubes_per_dim: 48,
        auto_disable: false,
    });

    let postprocessing = splashsurf::reconstruct::ReconstructionPostprocessingParameters::new();

    let reconstruction = splashsurf::reconstruct::reconstruction_pipeline::<i64, _>(
        &particles,
        &[],
        &params,
        &postprocessing,
    )?;
    let mesh = &reconstruction
        .tri_mesh
        .as_ref()
        .expect("reconstruction should produce a triangle mesh")
        .mesh;
    vtk_format::write_vtk(mesh, "../out/bunny_test_basic_pipeline_fine.vtk", "mesh")?;

    assert_range!(
        (920000..930000),
        mesh.triangles.len(),
        "Number of triangles should be in expected range"
    );
    assert_range!(
        (455000..465000),
        mesh.vertices.len(),
        "Number of vertices should be in expected range"
    );

    let mesh_manifold_information = mesh.compute_manifold_information();
    assert!(
        mesh_manifold_information.is_closed(),
        "Mesh should be closed"
    );
    assert!(
        mesh_manifold_information.is_manifold(),
        "Mesh should be manifold"
    );

    Ok(())
}

#[test]
fn test_basic_pipeline_particle_aabb() -> Result<(), Box<dyn std::error::Error>> {
    let particles =
        splashsurf_lib::io::particles_from_file("../data/bunny_frame_14_7705_particles.vtk")?;
    let particle_radius = 0.025;

    let mut params = splashsurf_lib::Parameters::new_relative(particle_radius, 4.0, 1.0);
    params.particle_aabb = Some(Aabb3d::new(
        Vector3::new(-0.7, 0.6, -0.2),
        Vector3::new(0.8, 2.1, 0.7),
    ));

    let postprocessing = splashsurf::reconstruct::ReconstructionPostprocessingParameters::new();

    let reconstruction = splashsurf::reconstruct::reconstruction_pipeline::<i64, _>(
        &particles,
        &[],
        &params,
        &postprocessing,
    )?;
    let mesh = &reconstruction
        .tri_mesh
        .as_ref()
        .expect("reconstruction should produce a triangle mesh")
        .mesh;
    vtk_format::write_vtk(
        mesh,
        "../out/bunny_test_basic_pipeline_particle_aabb.vtk",
        "mesh",
    )?;

    assert_range!(
        (19000..23000),
        mesh.triangles.len(),
        "Number of triangles should be in expected range"
    );
    assert_range!(
        (9500..11500),
        mesh.vertices.len(),
        "Number of vertices should be in expected range"
    );

    let mesh_manifold_information = mesh.compute_manifold_information();
    assert!(
        mesh_manifold_information.is_closed(),
        "Mesh should be closed"
    );
    assert!(
        mesh_manifold_information.is_manifold(),
        "Mesh should be manifold"
    );

    Ok(())
}

#[test]
fn test_basic_pipeline_postprocessing() -> Result<(), Box<dyn std::error::Error>> {
    let particles =
        splashsurf_lib::io::particles_from_file("../data/bunny_frame_14_7705_particles.vtk")?;
    let particle_radius = 0.025;

    let params = splashsurf_lib::Parameters::new_relative(particle_radius, 4.0, 1.0);

    let mut postprocessing = splashsurf::reconstruct::ReconstructionPostprocessingParameters::new();
    postprocessing.mesh_cleanup = true;
    postprocessing.mesh_smoothing_weights = true;
    postprocessing.mesh_smoothing_iters = Some(5);
    postprocessing.output_mesh_smoothing_weights = true;
    postprocessing.output_raw_mesh = true;

    let reconstruction = splashsurf::reconstruct::reconstruction_pipeline::<i64, _>(
        &particles,
        &[],
        &params,
        &postprocessing,
    )?;
    let mesh = &reconstruction
        .tri_mesh
        .as_ref()
        .expect("reconstruction should produce a triangle mesh")
        .mesh;
    let raw_mesh = &reconstruction.raw_reconstruction.mesh;
    vtk_format::write_vtk(
        mesh,
        "../out/bunny_test_basic_pipeline_postprocessing.vtk",
        "mesh",
    )?;

    assert_ne!(
        mesh.triangles.len(),
        raw_mesh.triangles.len(),
        "Number of triangles should differ between raw and postprocessed mesh"
    );
    assert_ne!(
        mesh.vertices.len(),
        raw_mesh.vertices.len(),
        "Number of vertices should differ between raw and postprocessed mesh"
    );

    assert_eq!(
        reconstruction
            .tri_mesh
            .as_ref()
            .unwrap()
            .point_attributes
            .len(),
        2,
        "There should be two point attributes (number of neighbors and smoothing weights)"
    );

    assert_range!(
        (18000..22000),
        mesh.triangles.len(),
        "Number of triangles should be in expected range"
    );
    assert_range!(
        (8000..12000),
        mesh.vertices.len(),
        "Number of vertices should be in expected range"
    );

    let mesh_manifold_information = mesh.compute_manifold_information();
    assert!(
        mesh_manifold_information.is_closed(),
        "Mesh should be closed"
    );
    assert!(
        mesh_manifold_information.is_manifold(),
        "Mesh should be manifold"
    );

    Ok(())
}

#[test]
fn test_basic_pipeline_postprocessing_with_aabb() -> Result<(), Box<dyn std::error::Error>> {
    let particles =
        splashsurf_lib::io::particles_from_file("../data/bunny_frame_14_7705_particles.vtk")?;
    let particle_radius = 0.025;

    let mut params = splashsurf_lib::Parameters::new_relative(particle_radius, 4.0, 1.0);
    params.particle_aabb = Some(Aabb3d::new(
        Vector3::new(-0.7, 0.6, -0.2),
        Vector3::new(0.8, 2.1, 0.7),
    ));

    let mut postprocessing = splashsurf::reconstruct::ReconstructionPostprocessingParameters::new();
    postprocessing.mesh_cleanup = true;
    postprocessing.mesh_smoothing_weights = true;
    postprocessing.mesh_smoothing_iters = Some(5);
    postprocessing.output_mesh_smoothing_weights = true;
    postprocessing.output_raw_mesh = true;

    let reconstruction = splashsurf::reconstruct::reconstruction_pipeline::<i64, _>(
        &particles,
        &[],
        &params,
        &postprocessing,
    )?;
    let mesh = &reconstruction
        .tri_mesh
        .as_ref()
        .expect("reconstruction should produce a triangle mesh")
        .mesh;
    let raw_mesh = &reconstruction.raw_reconstruction.mesh;
    vtk_format::write_vtk(
        mesh,
        "../out/bunny_test_basic_pipeline_postprocessing_aabb.vtk",
        "mesh",
    )?;

    assert_ne!(
        mesh.triangles.len(),
        raw_mesh.triangles.len(),
        "Number of triangles should differ between raw and postprocessed mesh"
    );
    assert_ne!(
        mesh.vertices.len(),
        raw_mesh.vertices.len(),
        "Number of vertices should differ between raw and postprocessed mesh"
    );

    assert_eq!(
        reconstruction
            .tri_mesh
            .as_ref()
            .unwrap()
            .point_attributes
            .len(),
        2,
        "There should be two point attributes (number of neighbors and smoothing weights)"
    );

    assert_range!(
        (11000..13000),
        mesh.triangles.len(),
        "Number of triangles should be in expected range"
    );
    assert_range!(
        (5000..7000),
        mesh.vertices.len(),
        "Number of vertices should be in expected range"
    );

    let mesh_manifold_information = mesh.compute_manifold_information();
    assert!(
        mesh_manifold_information.is_closed(),
        "Mesh should be closed"
    );
    assert!(
        mesh_manifold_information.is_manifold(),
        "Mesh should be manifold"
    );

    Ok(())
}
