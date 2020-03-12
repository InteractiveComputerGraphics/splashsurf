use splashsurf_lib::uniform_grid::{Axis, Direction, UniformGrid};
use splashsurf_lib::{Index, Real};

use nalgebra::Vector3;

fn unit_grid<I: Index, R: Real>() -> UniformGrid<I, R> {
    let origin = Vector3::new(R::zero(), R::zero(), R::zero());
    let n_cubes_per_dim = [I::one(), I::one(), I::one()];
    let cube_size = R::one();

    UniformGrid::new(&origin, &n_cubes_per_dim, cube_size)
}

#[test]
fn test_basic_uniform_grid_features() {
    let grid = unit_grid::<i32, f64>();

    assert_eq!(grid.aabb().max(), &Vector3::new(1.0, 1.0, 1.0));
    assert_eq!(grid.cell_size(), 1.0);

    let points = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];

    for point in points.iter() {
        assert!(grid.point_exists(point));
    }

    assert!(grid.cell_exists(&[0, 0, 0]));

    let origin = grid.get_point(&points[0]);
    assert!(origin.is_some());
    let origin = origin.unwrap();

    assert_eq!(
        grid.get_point_neighbor(&origin, Axis::X.with_direction(Direction::Positive))
            .unwrap()
            .index(),
        &[1, 0, 0]
    );
    assert_eq!(
        grid.get_point_neighbor(&origin, Axis::Y.with_direction(Direction::Positive))
            .unwrap()
            .index(),
        &[0, 1, 0]
    );
    assert_eq!(
        grid.get_point_neighbor(&origin, Axis::Z.with_direction(Direction::Positive))
            .unwrap()
            .index(),
        &[0, 0, 1]
    );
    assert!(grid
        .get_point_neighbor(&origin, Axis::X.with_direction(Direction::Negative))
        .is_none());
    assert!(grid
        .get_point_neighbor(&origin, Axis::Y.with_direction(Direction::Negative))
        .is_none());
    assert!(grid
        .get_point_neighbor(&origin, Axis::Z.with_direction(Direction::Negative))
        .is_none());
}
