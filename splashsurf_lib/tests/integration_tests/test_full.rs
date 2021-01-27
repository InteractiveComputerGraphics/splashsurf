use nalgebra::Vector3;
use splashsurf_lib::{
    reconstruct_surface, Parameters, SpatialDecompositionParameters, SubdivisionCriterion,
};
use std::path::Path;

use super::io::vtk::{particles_from_vtk, write_vtk};

// TODO: Compare with a solution file
// TODO: Test with a fixed grid?

enum Strategy {
    Global,
    Octree,
    OctreeStitching,
}

fn params(
    particle_radius: f32,
    kernel_radius: f32,
    cube_size: f32,
    strategy: Strategy,
) -> Parameters<f32> {
    let kernel_radius = particle_radius * kernel_radius;
    let cube_size = particle_radius * cube_size;

    let mut parameters = Parameters {
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

    match strategy {
        Strategy::Global => {}
        Strategy::Octree => {
            parameters.spatial_decomposition = Some(SpatialDecompositionParameters {
                subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
                ghost_particle_safety_factor: Some(1.0),
                enable_stitching: false,
            });
        }
        Strategy::OctreeStitching => {
            parameters.spatial_decomposition = Some(SpatialDecompositionParameters {
                subdivision_criterion: SubdivisionCriterion::MaxParticleCountAuto,
                ghost_particle_safety_factor: Some(1.0),
                enable_stitching: true,
            });
        }
    }

    parameters
}

fn default_params_with(strategy: Strategy) -> Parameters<f32> {
    params(0.025, 4.0, 0.75, strategy)
}

fn default_params() -> Parameters<f32> {
    default_params_with(Strategy::Global)
}

fn test_for_boundary(params: &Parameters<f32>) -> bool {
    params.spatial_decomposition.as_ref().map(|s| s.enable_stitching).unwrap_or(true)
}

macro_rules! generate_test {
    ($test_name:ident, $input_file:literal => $output_file:literal, $parameters:expr, $min_triangles:literal, $max_triangles:literal) => {
        #[test]
        fn $test_name() {
            let input_file = Path::new("../data/").join($input_file);
            let output_file = Path::new("../out/").join($output_file);

            let particle_positions: &Vec<Vector3<f32>> = &particles_from_vtk(input_file).unwrap();

            let parameters = $parameters;

            let reconstruction =
                reconstruct_surface::<i64, _>(particle_positions.as_slice(), &parameters).unwrap();

            write_vtk(
                reconstruction.mesh(),
                output_file,
                "mesh",
            )
            .unwrap();

            // Ensure that the number of triangles is roughly correct
            assert!(
                reconstruction.mesh().triangles.len() > $min_triangles,
                "Mesh has probably too few triangles"
            );
            assert!(
                reconstruction.mesh().triangles.len() < $max_triangles,
                "Mesh has probably too many triangles"
            );

            if test_for_boundary(&parameters) {
                // Ensure that the mesh does not have a boundary
                assert_eq!(reconstruction.mesh().find_boundary_edges(), vec![], "Mesh boundary is not empty");
            }
        }
    };
}

generate_test!(surface_reconstruction_bunny_global, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_global.vtk", default_params(), 60000, 80000);
generate_test!(surface_reconstruction_bunny_no_stitching, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_no_stitching.vtk", default_params_with(Strategy::Octree), 60000, 80000);
generate_test!(surface_reconstruction_bunny_stitching, "bunny_frame_14_7705_particles.vtk" => "reconstruct_surface_bunny_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 60000, 80000);

generate_test!(surface_reconstruction_hexecontahedron_stitching, "pentagonal_hexecontahedron_32286_particles.vtk" => "reconstruct_surface_pentagonal_hexecontahedron_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 600000, 700000);
generate_test!(surface_reconstruction_hilbert_stitching, "hilbert_46843_particles.vtk" => "reconstruct_surface_hilbert_par_stitching.vtk", default_params_with(Strategy::OctreeStitching), 380000, 430000);
generate_test!(surface_reconstruction_hilbert2_stitching, "hilbert2_7954_particles.vtk" => "reconstruct_surface_hilbert2_par_stitching.vtk", params(0.025, 4.0, 1.1, Strategy::OctreeStitching), 80000, 130000);
generate_test!(surface_reconstruction_octocat_stitching, "octocat_32614_particles.vtk" => "reconstruct_surface_octocat_par_stitching.vtk", params(0.025, 4.0, 0.75, Strategy::OctreeStitching), 150000, 210000);

//generate_test!(surface_reconstruction_knot_global, "sailors_knot_19539_particles.vtk" => "reconstruct_surface_knot_par_global.vtk", params(0.025, 4.0, 1.1, Strategy::Global), 50000, 100000);
//generate_test!(surface_reconstruction_knot_stitching, "sailors_knot_19539_particles.vtk" => "reconstruct_surface_knot_par_stitching.vtk", params(0.025, 4.0, 1.1, Strategy::OctreeStitching), 50000, 100000);
