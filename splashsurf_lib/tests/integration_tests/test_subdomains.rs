use all_asserts::assert_range;
use nalgebra::Vector3;
use splashsurf_lib::GridDecompositionParameters;
#[cfg(feature = "io")]
use splashsurf_lib::io::vtk_format::write_vtk;
use splashsurf_lib::marching_cubes::check_mesh_consistency;
use std::path::Path;
use splashsurf_lib::kernel::KernelType;

macro_rules! generate_single_particle_test {
    ($test_name:ident, $output_file:literal, cube_size = $cube_size_rel:literal, tris = $range_tri:expr, verts = $range_vert:expr, subdomains = $range_subdomains:expr) => {
        #[test]
        fn $test_name() -> Result<(), Box<dyn std::error::Error>> {
            let output_file = Path::new("../out/").join($output_file);

            let particle_positions = vec![Vector3::new(0.0, 0.0, 0.0)];

            let particle_radius = 0.025;
            let parameters = splashsurf_lib::Parameters {
                particle_radius,
                compact_support_radius: 4.0 * particle_radius,
                cube_size: $cube_size_rel * particle_radius,
                iso_surface_threshold: 0.6,
                particle_aabb: None,
                enable_multi_threading: true,
                enable_simd: true,
                spatial_decomposition: splashsurf_lib::SpatialDecomposition::UniformGrid(
                    GridDecompositionParameters {
                        subdomain_num_cubes_per_dim: 64,
                        auto_disable: false,
                    },
                ),
                rest_density: 1000.0,
                global_neighborhood_list: false,
                kernel_type: KernelType::CubicSpline,
            };

            let reconstruction =
                splashsurf_lib::reconstruct_surface::<i64, f32>(&particle_positions, &parameters)?;
            #[cfg(feature = "io")]
            {
                write_vtk(&reconstruction.mesh, &output_file, "mesh")?;
            }
            #[cfg(not(feature = "io"))]
            {
                let _ = &output_file;
            }

            assert_range!(
                $range_tri,
                reconstruction.mesh.triangles.len(),
                "number of triangles should be in range"
            );
            assert_range!(
                $range_vert,
                reconstruction.mesh.vertices.len(),
                "number of vertices should be in range"
            );

            if let Err(e) =
                check_mesh_consistency(&reconstruction.grid, &reconstruction.mesh, true, true, true)
            {
                eprintln!("{}", e);
                panic!("mesh contains topological/manifold errors");
            }

            let subdomain_grid = reconstruction
                .subdomain_grid
                .as_ref()
                .expect("subdomain grid should be present");
            assert_range!(
                $range_subdomains,
                subdomain_grid.cells_per_dim().iter().product::<i64>(),
                "number of subdomains should be in range"
            );

            Ok(())
        }
    };
}

generate_single_particle_test!(
    test_single_particle_c0_5,
    "single_particle_c0_5.vtk",
    cube_size = 0.5,
    tris = (240..260),
    verts = (120..135),
    subdomains = (1..=1)
);

generate_single_particle_test!(
    test_single_particle_c0_1,
    "single_particle_c0_1.vtk",
    cube_size = 0.1,
    tris = (5700..6000),
    verts = (2800..3000),
    subdomains = (7..=9)
);

generate_single_particle_test!(
    test_single_particle_c0_025,
    "single_particle_c0_025.vtk",
    cube_size = 0.025,
    tris = (90000..100000),
    verts = (45000..48000),
    subdomains = (330..350)
);
