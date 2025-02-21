use nalgebra::Vector3;
use splashsurf_lib::marching_cubes::check_mesh_consistency;
use splashsurf_lib::{
    Aabb3d, GridDecompositionParameters, Parameters, Real, SpatialDecomposition,
    reconstruct_surface,
};

enum Strategy {
    Global,
    SubdomainGrid,
}

fn params_with_aabb<R: Real>(
    particle_radius: R,
    compact_support_radius: R,
    cube_size: R,
    iso_surface_threshold: R,
    domain_aabb: Option<Aabb3d<R>>,
    strategy: Strategy,
) -> Parameters<R> {
    let compact_support_radius = particle_radius * compact_support_radius;
    let cube_size = particle_radius * cube_size;

    let mut parameters = Parameters {
        particle_radius,
        rest_density: R::from_f64(1000.0).unwrap(),
        compact_support_radius,
        cube_size,
        iso_surface_threshold,
        particle_aabb: domain_aabb,
        enable_multi_threading: false,
        spatial_decomposition: None,
        global_neighborhood_list: false,
    };

    match strategy {
        Strategy::Global => {}
        Strategy::SubdomainGrid => {
            parameters.spatial_decomposition = Some(SpatialDecomposition::UniformGrid(
                GridDecompositionParameters {
                    subdomain_num_cubes_per_dim: 64,
                },
            ))
        }
    }

    parameters
}

fn params<R: Real>(
    particle_radius: R,
    compact_support_radius: R,
    cube_size: R,
    iso_surface_threshold: R,
    strategy: Strategy,
) -> Parameters<R> {
    params_with_aabb(
        particle_radius,
        compact_support_radius,
        cube_size,
        iso_surface_threshold,
        None,
        strategy,
    )
}

/// This tests ensures that a surface is reconstructed properly if a single edge is above the threshold
/// on one side but outside the compact support on the other side.
#[test]
fn test_edge_above_threshold_to_outside_of_compact_support_global() {
    let particle_positions: Vec<Vector3<f32>> = vec![Vector3::new(0.01, 0.0, 0.0)];
    let parameters = params(1.0, 1.0, 1.0, 0.1, Strategy::Global);
    let reconstruction =
        reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

    println!(
        "Reconstructed mesh from {} particles has {} triangles.",
        particle_positions.len(),
        reconstruction.mesh().triangles.len()
    );

    assert_eq!(
        reconstruction.mesh().vertices.len(),
        6,
        "Number of vertices"
    );
    assert_eq!(
        reconstruction.mesh().triangles.len(),
        8,
        "Number of triangles"
    );

    // Ensure that the mesh does not have a boundary
    if let Err(e) = check_mesh_consistency(
        reconstruction.grid(),
        reconstruction.mesh(),
        true,
        true,
        true,
    ) {
        eprintln!("{}", e);
        panic!("Mesh contains topological/manifold errors");
    }
}

#[test]
fn test_edge_above_threshold_to_outside_of_compact_support_subdomains() {
    let particle_positions: Vec<Vector3<f32>> = vec![Vector3::new(0.01, 0.0, 0.0)];
    let parameters = params(1.0, 1.0, 1.0, 0.1, Strategy::SubdomainGrid);
    let reconstruction =
        reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

    println!(
        "Reconstructed mesh from {} particles has {} triangles.",
        particle_positions.len(),
        reconstruction.mesh().triangles.len()
    );

    assert_eq!(
        reconstruction.mesh().vertices.len(),
        6,
        "Number of vertices"
    );
    assert_eq!(
        reconstruction.mesh().triangles.len(),
        8,
        "Number of triangles"
    );

    // Ensure that the mesh does not have a boundary
    if let Err(e) = check_mesh_consistency(
        reconstruction.grid(),
        reconstruction.mesh(),
        true,
        true,
        true,
    ) {
        eprintln!("{}", e);
        panic!("Mesh contains topological/manifold errors");
    }
}
