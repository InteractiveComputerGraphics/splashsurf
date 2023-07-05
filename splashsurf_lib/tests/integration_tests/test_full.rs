use nalgebra::Vector3;
use splashsurf_lib::io::vtk_format::{particles_from_vtk, write_vtk};
use splashsurf_lib::marching_cubes::check_mesh_consistency;
use splashsurf_lib::{
    reconstruct_surface, Aabb3d, Parameters, ParticleDensityComputationStrategy, Real,
    SpatialDecompositionParameters, SubdivisionCriterion,
};
use std::path::Path;

// TODO: Compare with a solution file
// TODO: Test with a fixed grid?

enum Strategy {
    Global,
    Octree,
    OctreeStitching,
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
        domain_aabb,
        enable_multi_threading: false,
        subdomain_num_cubes_per_dim: None,
        spatial_decomposition: None,
    };

    match strategy {
        Strategy::Global => {}
        Strategy::Octree => {
            parameters.spatial_decomposition = Some(SpatialDecompositionParameters {
                subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
                ghost_particle_safety_factor: Some(R::one()),
                enable_stitching: false,
                particle_density_computation:
                    ParticleDensityComputationStrategy::SynchronizeSubdomains,
            });
        }
        Strategy::OctreeStitching => {
            parameters.spatial_decomposition = Some(SpatialDecompositionParameters {
                subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
                ghost_particle_safety_factor: Some(R::one() + R::one()),
                enable_stitching: true,
                particle_density_computation:
                    ParticleDensityComputationStrategy::SynchronizeSubdomains,
            });
        }
        Strategy::SubdomainGrid => {
            parameters.subdomain_num_cubes_per_dim = Some(64);
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

fn default_params_with<R: Real>(strategy: Strategy) -> Parameters<R> {
    params(
        R::from_f64(0.025).unwrap(),
        R::from_f64(4.0).unwrap(),
        R::from_f64(0.75).unwrap(),
        R::from_f64(0.6).unwrap(),
        strategy,
    )
}

fn default_params<R: Real>() -> Parameters<R> {
    default_params_with(Strategy::Global)
}

fn test_for_boundary<R: Real>(params: &Parameters<R>) -> bool {
    params
        .spatial_decomposition
        .as_ref()
        .map(|s| s.enable_stitching)
        .unwrap_or(true)
}

macro_rules! generate_test {
    ($floating_point_type:ty, $test_name:ident, $input_file:literal => $output_file:literal, $parameters:expr, $min_triangles:literal, $max_triangles:literal $(, $attr:meta)?) => {
        #[test]
        $(#[$attr])?
        fn $test_name() {
            let input_file = Path::new("../data/").join($input_file);
            let output_file = Path::new("../out/").join($output_file);

            let particle_positions: &Vec<Vector3<$floating_point_type>> =
                &particles_from_vtk(input_file).unwrap();

            let parameters = $parameters;

            let reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

            write_vtk(reconstruction.mesh(), output_file, "mesh").unwrap();

            println!(
                "Reconstructed mesh from particle file '{}' with {} particles has {} triangles.",
                $input_file,
                particle_positions.len(),
                reconstruction.mesh().triangles.len()
            );

            // Ensure that the number of triangles is roughly correct
            assert!(
                reconstruction.mesh().triangles.len() > $min_triangles,
                "Mesh has probably too few triangles (min expected: {}, is: {})",
                $min_triangles,
                reconstruction.mesh().triangles.len()
            );
            assert!(
                reconstruction.mesh().triangles.len() < $max_triangles,
                "Mesh has probably too many triangles (max expected: {}, is: {})",
                $max_triangles,
                reconstruction.mesh().triangles.len()
            );

            if test_for_boundary(&parameters) {
                // Ensure that the mesh does not have a boundary
                if let Err(e) = check_mesh_consistency(reconstruction.grid(), reconstruction.mesh())
                {
                    eprintln!("{}", e);
                    panic!("Mesh contains boundary edges");
                }
            }
        }
    };
}

generate_test!(f32, surface_reconstruction_bunny_global, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_global.vtk", default_params(), 60000, 80000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_bunny_no_stitching, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_no_stitching.vtk", default_params_with(Strategy::Octree), 60000, 80000);
generate_test!(f32, surface_reconstruction_bunny_stitching, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 60000, 80000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_bunny_grid, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_grid.vtk", default_params_with(Strategy::SubdomainGrid), 60000, 80000, cfg_attr(debug_assertions, ignore));

generate_test!(f32, surface_reconstruction_hexecontahedron_stitching, "pentagonal_hexecontahedron_32286_particles.vtk" => "reconstruct_surface_pentagonal_hexecontahedron_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 550000, 650000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_hexecontahedron_grid, "pentagonal_hexecontahedron_32286_particles.vtk" => "reconstruct_surface_pentagonal_hexecontahedron_par_grid.vtk", default_params_with(Strategy::SubdomainGrid), 550000, 650000, cfg_attr(debug_assertions, ignore));

generate_test!(f32, surface_reconstruction_hilbert_stitching, "hilbert_46843_particles.vtk" => "reconstruct_surface_hilbert_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 360000, 400000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_hilbert2_stitching, "hilbert2_7954_particles.vtk" => "reconstruct_surface_hilbert2_par_stitching.vtk", params(0.025, 4.0, 1.1, 0.6, Strategy::OctreeStitching), 90000, 100000);
generate_test!(f32, surface_reconstruction_octocat_stitching, "octocat_32614_particles.vtk" => "reconstruct_surface_octocat_par_stitching.vtk", params(0.025, 4.0, 0.75, 0.6, Strategy::OctreeStitching), 140000, 180000, cfg_attr(debug_assertions, ignore));

generate_test!(f32, surface_reconstruction_hilbert_grid, "hilbert_46843_particles.vtk" => "reconstruct_surface_hilbert_par_grid.vtk", default_params_with(Strategy::SubdomainGrid), 360000, 400000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_hilbert2_grid, "hilbert2_7954_particles.vtk" => "reconstruct_surface_hilbert2_par_grid.vtk", params(0.025, 4.0, 1.1, 0.6, Strategy::SubdomainGrid), 90000, 100000);
generate_test!(f32, surface_reconstruction_octocat_grid, "octocat_32614_particles.vtk" => "reconstruct_surface_octocat_par_grid.vtk", params(0.025, 4.0, 0.75, 0.6, Strategy::SubdomainGrid), 140000, 180000, cfg_attr(debug_assertions, ignore));

generate_test!(f32, surface_reconstruction_knot_global, "sailors_knot_19539_particles.vtk" => "reconstruct_surface_knot_par_global.vtk", params(0.025, 4.0, 1.1, 0.6, Strategy::Global), 40000, 70000, cfg_attr(debug_assertions, ignore));
generate_test!(f32, surface_reconstruction_knot_stitching, "sailors_knot_19539_particles.vtk" => "reconstruct_surface_knot_par_stitching.vtk", params(0.025, 4.0, 1.1, 0.6, Strategy::OctreeStitching), 40000, 70000);
generate_test!(f32, surface_reconstruction_knot_grid, "sailors_knot_19539_particles.vtk" => "reconstruct_surface_knot_par_grid.vtk", params(0.025, 4.0, 1.1, 0.6, Strategy::SubdomainGrid), 40000, 70000);

generate_test!(f32, surface_reconstruction_free_particles_01, "free_particles_1000_particles.vtk" => "reconstruct_surface_free_particles_01_global.vtk", params(0.5, 4.0, 1.5, 0.45, Strategy::Global), 21000, 25000);
generate_test!(f32, surface_reconstruction_free_particles_02, "free_particles_125_particles.vtk" => "reconstruct_surface_free_particles_02_global.vtk", params_with_aabb(0.5, 4.0, 1.5, 0.45, Some(Aabb3d::new(Vector3::new(-10.0, -10.0, -10.0), Vector3::new(210.0, 210.0, 210.0))), Strategy::Global), 1500, 1600);
