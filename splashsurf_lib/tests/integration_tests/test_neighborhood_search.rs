use nalgebra::Vector3;
use splashsurf_lib::Aabb3d;
use splashsurf_lib::neighborhood_search::*;

fn sort_neighborhood_lists(neighborhood_list: &mut Vec<Vec<usize>>) {
    for neighbors in neighborhood_list.iter_mut() {
        neighbors.sort_unstable();
    }
}

fn generate_simple_test_cases(search_radius: f32) -> Vec<(Vec<Vector3<f32>>, Vec<Vec<usize>>)> {
    vec![
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(
                    1.0 + search_radius,
                    1.0 + search_radius,
                    1.0 + search_radius,
                ),
            ],
            vec![Vec::new(), Vec::new()],
        ),
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0 + 0.9999 * search_radius, 1.0, 1.0),
            ],
            vec![vec![1], vec![0]],
        ),
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0 + search_radius, 1.0, 1.0),
            ],
            vec![vec![1], vec![0]],
        ),
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0 + search_radius * 1.0001, 1.0, 1.0),
            ],
            vec![Vec::new(), Vec::new()],
        ),
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0 + 0.9 * search_radius, 1.0, 1.0),
                Vector3::new(1.0 - 0.9 * search_radius, 1.0, 1.0),
                Vector3::new(1.0, 1.0 + 0.2 * search_radius, 1.0),
            ],
            vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]],
        ),
        (
            vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0 + 0.9 * search_radius, 1.0, 1.0),
                Vector3::new(1.0 - 0.9 * search_radius, 1.0, 1.0),
                Vector3::new(1.0 - 0.8 * search_radius, 1.0, -0.2 * search_radius),
                Vector3::new(1.0, 1.0 + 0.2 * search_radius, 1.0),
                Vector3::new(1.0, 1.0 - 0.2 * search_radius, 1.0),
                Vector3::new(1.0, 1.0 + 0.2 * search_radius, 1.0 + 0.2 * search_radius),
                Vector3::new(1.0, 1.0 - 0.2 * search_radius, 1.0 - 0.2 * search_radius),
            ],
            vec![
                vec![1, 2, 4, 5, 6, 7],
                vec![0, 4, 5, 6, 7],
                vec![0, 4, 5, 6, 7],
                vec![],
                vec![0, 1, 2, 5, 6, 7],
                vec![0, 1, 2, 4, 6, 7],
                vec![0, 1, 2, 4, 5, 7],
                vec![0, 1, 2, 4, 5, 6],
            ],
        ),
    ]
}

#[test]
fn test_neighborhood_search_naive_simple() {
    let search_radius: f32 = 0.3;

    for (particles, mut solution) in generate_simple_test_cases(search_radius) {
        let mut nl = Vec::new();
        neighborhood_search_naive(particles.as_slice(), search_radius, &mut nl);

        sort_neighborhood_lists(&mut nl);
        sort_neighborhood_lists(&mut solution);

        assert_eq!(
            nl, solution,
            "neighborhood_search_naive failed. Search radius: {}, input: {:?}",
            search_radius, particles
        );
    }
}

#[test]
fn test_neighborhood_search_spatial_hashing_simple() {
    let search_radius: f32 = 0.3;

    for (particles, mut solution) in generate_simple_test_cases(search_radius) {
        let mut nl = Vec::new();
        let mut domain = Aabb3d::from_points(particles.as_slice());
        domain.grow_uniformly(search_radius);
        neighborhood_search_spatial_hashing::<i32, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl,
        );

        sort_neighborhood_lists(&mut nl);
        sort_neighborhood_lists(&mut solution);

        assert_eq!(
            nl, solution,
            "neighborhood_search_spatial_hashing failed. Search radius: {}, input: {:?}",
            search_radius, particles
        );
    }
}

#[test]
fn test_neighborhood_search_spatial_hashing_flat_simple() {
    let search_radius: f32 = 0.3;

    for (particles, mut solution) in generate_simple_test_cases(search_radius) {
        let mut nl = FlatNeighborhoodList::default();
        let mut domain = Aabb3d::from_points(particles.as_slice());
        domain.grow_uniformly(search_radius);
        neighborhood_search_spatial_hashing_flat::<i32, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl,
        );
        let mut nl = nl.to_vec_vec();

        sort_neighborhood_lists(&mut nl);
        sort_neighborhood_lists(&mut solution);

        assert_eq!(
            nl, solution,
            "neighborhood_search_spatial_hashing failed. Search radius: {}, input: {:?}",
            search_radius, particles
        );
    }
}

#[test]
fn test_neighborhood_search_spatial_hashing_parallel_simple() {
    let search_radius: f32 = 0.3;

    for (particles, mut solution) in generate_simple_test_cases(search_radius) {
        let mut nl = Vec::new();
        let mut domain = Aabb3d::from_points(particles.as_slice());
        domain.grow_uniformly(search_radius);
        neighborhood_search_spatial_hashing_parallel::<i32, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl,
        );

        sort_neighborhood_lists(&mut nl);
        sort_neighborhood_lists(&mut solution);

        assert_eq!(
            nl, solution,
            "neighborhood_search_spatial_hashing failed. Search radius: {}, input: {:?}",
            search_radius, particles
        );
    }
}

#[cfg(feature = "io")]
mod tests_from_files {
    use super::*;
    use splashsurf_lib::Aabb3d;
    use splashsurf_lib::io;

    #[test]
    fn test_compare_free_particles_125() {
        let search_radius: f32 = 2.0 * 0.5;

        let file = "../data/free_particles_125_particles.vtk";
        let particles = io::vtk_format::particles_from_vtk::<f32, _>(file).unwrap();

        let mut domain = Aabb3d::par_from_points(particles.as_slice());
        domain.scale_uniformly(1.5);

        let mut nl_naive = Vec::new();
        let mut nl_hashed = Vec::new();
        let mut nl_hashed_par = Vec::new();
        let mut nl_hashed_flat = FlatNeighborhoodList::default();

        neighborhood_search_naive(particles.as_slice(), search_radius, &mut nl_naive);
        neighborhood_search_spatial_hashing::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed,
        );
        neighborhood_search_spatial_hashing_parallel::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_par,
        );
        neighborhood_search_spatial_hashing_flat::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_flat,
        );

        sort_neighborhood_lists(&mut nl_naive);
        sort_neighborhood_lists(&mut nl_hashed);
        sort_neighborhood_lists(&mut nl_hashed_par);
        let mut nl_hashed_flat = nl_hashed_flat.to_vec_vec();
        sort_neighborhood_lists(&mut nl_hashed_flat);

        assert_eq!(
            nl_hashed, nl_naive,
            "result of neighborhood_search_spatial_hashing does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_par, nl_naive,
            "result of neighborhood_search_spatial_hashing_parallel does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_flat, nl_naive,
            "result of neighborhood_search_spatial_hashing_flat does not match neighborhood_search_naive, file: {:?}",
            file
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_compare_free_particles_1000() {
        let search_radius: f32 = 2.0 * 0.5;

        let file = "../data/free_particles_1000_particles.vtk";
        let particles = io::vtk_format::particles_from_vtk::<f32, _>(file).unwrap();

        let mut domain = Aabb3d::par_from_points(particles.as_slice());
        domain.scale_uniformly(1.5);

        let mut nl_naive = Vec::new();
        let mut nl_hashed = Vec::new();
        let mut nl_hashed_par = Vec::new();
        let mut nl_hashed_flat = FlatNeighborhoodList::default();

        neighborhood_search_naive(particles.as_slice(), search_radius, &mut nl_naive);
        neighborhood_search_spatial_hashing::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed,
        );
        neighborhood_search_spatial_hashing_parallel::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_par,
        );
        neighborhood_search_spatial_hashing_flat::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_flat,
        );

        sort_neighborhood_lists(&mut nl_naive);
        sort_neighborhood_lists(&mut nl_hashed);
        sort_neighborhood_lists(&mut nl_hashed_par);
        let mut nl_hashed_flat = nl_hashed_flat.to_vec_vec();
        sort_neighborhood_lists(&mut nl_hashed_flat);

        assert_eq!(
            nl_hashed, nl_naive,
            "result of neighborhood_search_spatial_hashing does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_par, nl_naive,
            "result of neighborhood_search_spatial_hashing_parallel does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_flat, nl_naive,
            "result of neighborhood_search_spatial_hashing_flat does not match neighborhood_search_naive, file: {:?}",
            file
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_compare_cube_2366() {
        let search_radius: f32 = 4.0 * 0.025;

        let file = "../data/cube_2366_particles.vtk";
        let particles = io::vtk_format::particles_from_vtk::<f32, _>(file).unwrap();

        let mut domain = Aabb3d::par_from_points(particles.as_slice());
        domain.scale_uniformly(1.5);

        let mut nl_naive = Vec::new();
        let mut nl_hashed = Vec::new();
        let mut nl_hashed_par = Vec::new();
        let mut nl_hashed_flat = FlatNeighborhoodList::default();

        neighborhood_search_naive(particles.as_slice(), search_radius, &mut nl_naive);
        neighborhood_search_spatial_hashing::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed,
        );
        neighborhood_search_spatial_hashing_parallel::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_par,
        );
        neighborhood_search_spatial_hashing_flat::<i64, f32>(
            &domain,
            particles.as_slice(),
            search_radius,
            &mut nl_hashed_flat,
        );

        sort_neighborhood_lists(&mut nl_naive);
        sort_neighborhood_lists(&mut nl_hashed);
        sort_neighborhood_lists(&mut nl_hashed_par);
        let mut nl_hashed_flat = nl_hashed_flat.to_vec_vec();
        sort_neighborhood_lists(&mut nl_hashed_flat);

        assert_eq!(
            nl_hashed, nl_naive,
            "result of neighborhood_search_spatial_hashing does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_par, nl_naive,
            "result of neighborhood_search_spatial_hashing_parallel does not match neighborhood_search_naive, file: {:?}",
            file
        );
        assert_eq!(
            nl_hashed_flat, nl_naive,
            "result of neighborhood_search_spatial_hashing_flat does not match neighborhood_search_naive, file: {:?}",
            file
        );
    }
}
