use std::cell::RefCell;

use nalgebra::Vector3;

use crate::uniform_grid::PointIndex;
use crate::{Index, Real, UniformGrid};

#[derive(Clone, Debug)]
struct OctreeNode<'a, I: Index, R: Real> {
    pub parent: Option<&'a OctreeNode<'a, I, R>>,
    pub lower_corner: PointIndex<I>,
    pub upper_corner: PointIndex<I>,
    pub body: RefCell<NodeBody<'a, I, R>>,
}

impl<'a, I: Index, R: Real> OctreeNode<'a, I, R> {
    pub fn new_root(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        let n_points = grid.points_per_dim();
        let min_point = [I::zero(), I::zero(), I::zero()];
        let max_point = [
            n_points[0] - I::zero(),
            n_points[1] - I::zero(),
            n_points[2] - I::zero(),
        ];

        Self {
            parent: None,
            lower_corner: grid.get_point(&min_point).unwrap(),
            upper_corner: grid.get_point(&max_point).unwrap(),
            body: RefCell::new(NodeBody::new_leaf((0..n_particles).collect())),
        }
    }

    pub fn subdivide(&mut self, _grid: &UniformGrid<I, R>, _particle_positions: &[Vector3<R>]) {
        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles } = &*self.body.borrow() {
            let particles = particles.clone();

            NodeBody::Leaf { particles }
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        *self.body.borrow_mut() = new_body;
    }
}

fn get_split_coordinate<I: Index, R: Real>(grid: UniformGrid<I, R>, lower: PointIndex<I>, upper: PointIndex<I>, component: VectorComponent) -> R {
    
}

#[derive(Clone, Debug)]
enum NodeBody<'a, I: Index, R: Real> {
    Children {
        children: [Box<OctreeNode<'a, I, R>>; 8],
    },
    Leaf {
        particles: Vec<usize>,
    },
}

impl<'a, I: Index, R: Real> NodeBody<'a, I, R> {
    pub fn new_leaf(indices: Vec<usize>) -> Self {
        NodeBody::Leaf { particles: indices }
    }
}

#[allow(dead_code)]
pub fn build_octree<R: Real>(
    particle_positions: &[Vector3<R>],
    particle_neighbor_lists: &[Vec<usize>],
    kernel_radius: R,
    max_particles_per_leaf: usize,
) {
    let mut indices: Vec<_> = (0..particle_positions.len()).collect();

    let first_level = octree_subdivision(&mut indices[..], particle_positions);
}

fn octree_subdivision<'a, R: Real>(
    indices: &'a mut [usize],
    particle_positions: &[Vector3<R>],
) -> Vec<&'a mut [usize]> {
    let x_splits = split_all(vec![indices], particle_positions, VectorComponent::X);
    assert!(x_splits.len() == 2);
    let y_splits = split_all(x_splits, particle_positions, VectorComponent::Y);
    assert!(y_splits.len() == 4);
    let z_splits = split_all(y_splits, particle_positions, VectorComponent::Z);
    assert!(z_splits.len() == 8);

    z_splits
}

fn split_all<'a, R: Real, Iter: IntoIterator<Item = &'a mut [usize]>>(
    index_slices: Iter,
    particle_positions: &[Vector3<R>],
    component: VectorComponent,
) -> Vec<&'a mut [usize]> {
    let mut slices = Vec::new();
    for indices in index_slices {
        let (lower_indices, upper_indices) = half_split(indices, particle_positions, component);
        slices.push(lower_indices);
        slices.push(upper_indices)
    }
    slices
}

fn half_split<'a, R: Real>(
    indices: &'a mut [usize],
    particle_positions: &[Vector3<R>],
    component: VectorComponent,
) -> (&'a mut [usize], &'a mut [usize]) {
    sort_by_component(indices, particle_positions, component);
    indices.split_at_mut(indices.len() / 2)
}

#[derive(Copy, Clone, Debug)]
enum VectorComponent {
    X = 0,
    Y = 1,
    Z = 2,
}

fn sort_by_component<R: Real>(
    indices: &mut [usize],
    particle_positions: &[Vector3<R>],
    component: VectorComponent,
) {
    let comp = component as usize;
    indices.sort_unstable_by(|&idx_a, &idx_b| {
        particle_positions[idx_a][comp]
            .partial_cmp(&particle_positions[idx_b][comp])
            .unwrap()
    })
}
