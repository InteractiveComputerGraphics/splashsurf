use nalgebra::Vector3;

use crate::uniform_grid::{Direction, PointIndex};
use crate::{Index, Real, UniformGrid};

#[cfg(test)]
mod test_octree {
    use super::*;

    mod vtk {
        use super::*;

        use std::fs::create_dir_all;
        use std::path::Path;

        use anyhow::{anyhow, Context};

        use vtkio::model::{ByteOrder, DataSet, Version, Vtk};
        use vtkio::{export_be, import_be, IOBuffer};

        pub fn particles_from_vtk<R: Real, P: AsRef<Path>>(
            vtk_file: P,
        ) -> Result<Vec<Vector3<R>>, anyhow::Error> {
            let sph_dataset = read_vtk(vtk_file)?;
            particles_from_dataset(sph_dataset)
        }

        pub fn read_vtk<P: AsRef<Path>>(filename: P) -> Result<DataSet, vtkio::Error> {
            let filename = filename.as_ref();
            import_be(filename).map(|vtk| vtk.data)
        }

        pub fn particles_from_coords<RealOut: Real, RealIn: Real>(
            coords: &Vec<RealIn>,
        ) -> Result<Vec<Vector3<RealOut>>, anyhow::Error> {
            if coords.len() % 3 != 0 {
                anyhow!(
                    "The number of values in the particle data point buffer is not divisible by 3"
                );
            }

            let num_points = coords.len() / 3;
            let mut positions = Vec::with_capacity(num_points);
            for i in 0..num_points {
                positions.push(Vector3::new(
                    RealOut::from_f64(coords[3 * i + 0].to_f64().unwrap()).unwrap(),
                    RealOut::from_f64(coords[3 * i + 1].to_f64().unwrap()).unwrap(),
                    RealOut::from_f64(coords[3 * i + 2].to_f64().unwrap()).unwrap(),
                ))
            }

            Ok(positions)
        }

        pub fn particles_from_dataset<R: Real>(
            dataset: DataSet,
        ) -> Result<Vec<Vector3<R>>, anyhow::Error> {
            if let DataSet::UnstructuredGrid { pieces, .. } = dataset {
                if let Some(piece) = pieces.into_iter().next() {
                    let points = piece
                        .load_piece_data()
                        .context("Failed to load unstructured grid piece")?
                        .points;

                    match points {
                        IOBuffer::F64(coords) => particles_from_coords(&coords),
                        IOBuffer::F32(coords) => particles_from_coords(&coords),
                        _ => Err(anyhow!(
                            "Point coordinate IOBuffer does not contain f32 or f64 values"
                        )),
                    }
                } else {
                    Err(anyhow!(
                        "Loaded dataset does not contain an unstructured grid piece"
                    ))
                }
            } else {
                Err(anyhow!(
                    "Loaded dataset does not contain an unstructured grid"
                ))
            }
        }
    }

    #[test]
    fn build_octree() {
        let file = "../data/double_dam_break_frame_26_4732_particles.vtk";
        let particles = vtk::particles_from_vtk::<f64, _>(file).unwrap();
        println!("Loaded {} particles from {}", particles.len(), file);

        let grid = crate::grid_for_reconstruction::<i64, _>(particles.as_slice(), 0.025, 0.2, None)
            .unwrap();

        println!("{:?}", grid);

        let octree = Octree::new(&grid, particles.as_slice(), 60);

        let mut particle_count = 0;
        for node in octree.depth_first_iter() {
            if let Some(particles) = node.particles() {
                println!("Leaf with: {} particles", particles.len());
                particle_count += particles.len();
            }
        }
        assert_eq!(particle_count, particles.len());
    }
}

#[derive(Clone, Debug)]
struct Octree<I: Index> {
    root: OctreeNode<I>,
}

impl<I: Index> Octree<I> {
    pub fn new<R: Real>(
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) -> Self {
        let mut root = OctreeNode::new_root(grid, particle_positions.len());
        root.subdivide_recursively(grid, particle_positions, particles_per_cell);
        Self { root }
    }

    pub fn depth_first_iter(&self) -> impl Iterator<Item = &OctreeNode<I>> {
        let mut queue = Vec::new();
        let mut visited_parents = Vec::new();
        queue.push(&self.root);

        let mut iter = move || -> Option<&OctreeNode<I>> {
            while let Some(&next) = queue.last() {
                // Check if the node has children
                if let Some(children) = next.children() {
                    // Check if all children of this node were already yielded
                    if let Some(&last_parent) = visited_parents.last() {
                        if std::ptr::eq(next, last_parent) {
                            // We already yielded all children of this node, so it can be yielded itself
                            let _ = queue.pop();
                            let _ = visited_parents.pop();
                            return Some(next);
                        }
                    }

                    // Enqueue all children
                    for child in children.iter().rev() {
                        queue.push(child);
                    }
                    visited_parents.push(next);
                }
                // Leaf reached, remove form queue and yield it
                else {
                    let _ = queue.pop();
                    return Some(next);
                }
            }

            None
        };

        std::iter::from_fn(iter)
    }
}

#[derive(Clone, Debug)]
struct OctreeNode<I: Index> {
    pub lower_corner: PointIndex<I>,
    pub upper_corner: PointIndex<I>,
    pub body: NodeBody<I>,
}

#[derive(Clone, Debug)]
enum NodeBody<I: Index> {
    Children { children: Vec<Box<OctreeNode<I>>> },
    Leaf { particles: Vec<usize> },
}

impl<I: Index> NodeBody<I> {
    pub fn new_leaf(indices: Vec<usize>) -> Self {
        NodeBody::Leaf { particles: indices }
    }

    pub fn new_with_children(children: Vec<Box<OctreeNode<I>>>) -> Self {
        assert_eq!(children.len(), 8);
        NodeBody::Children { children }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            NodeBody::Leaf { .. } => true,
            _ => false,
        }
    }

    pub fn particles(&self) -> Option<&[usize]> {
        match self {
            NodeBody::Leaf { particles } => Some(particles.as_slice()),
            _ => None,
        }
    }

    pub fn children(&self) -> Option<&[Box<OctreeNode<I>>]> {
        match self {
            NodeBody::Children { children } => Some(children.as_slice()),
            _ => None,
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut [Box<OctreeNode<I>>]> {
        match self {
            NodeBody::Children { children } => Some(children.as_mut_slice()),
            _ => None,
        }
    }
}

impl<I: Index> OctreeNode<I> {
    fn new_root<R: Real>(grid: &UniformGrid<I, R>, n_particles: usize) -> Self {
        let n_points = grid.points_per_dim();
        let min_point = [I::zero(), I::zero(), I::zero()];
        let max_point = [
            n_points[0] - I::one(),
            n_points[1] - I::one(),
            n_points[2] - I::one(),
        ];

        Self {
            lower_corner: grid
                .get_point(&min_point)
                .expect("Cannot get lower corner of grid"),
            upper_corner: grid
                .get_point(&max_point)
                .expect("Cannot get upper corner of grid"),
            body: NodeBody::new_leaf((0..n_particles).collect()),
        }
    }

    fn new_leaf(
        lower_corner: PointIndex<I>,
        upper_corner: PointIndex<I>,
        particles: Vec<usize>,
    ) -> Self {
        Self {
            lower_corner,
            upper_corner,
            body: NodeBody::new_leaf(particles),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.body.is_leaf()
    }

    pub fn particles(&self) -> Option<&[usize]> {
        self.body.particles()
    }

    pub fn children(&self) -> Option<&[Box<OctreeNode<I>>]> {
        self.body.children()
    }

    fn subdivide_recursively<R: Real>(
        &mut self,
        grid: &UniformGrid<I, R>,
        particle_positions: &[Vector3<R>],
        particles_per_cell: usize,
    ) {
        if let Some(particles) = self.body.particles() {
            if particles.len() < particles_per_cell {
                return;
            }
        } else {
            return;
        }

        // TODO: Replace recursion using tree visitor?
        self.subdivide(grid, particle_positions);
        if let Some(children) = self.body.children_mut() {
            for child_node in children {
                child_node.subdivide_recursively(grid, particle_positions, particles_per_cell);
            }
        }
    }

    fn subdivide<R: Real>(&mut self, grid: &UniformGrid<I, R>, particle_positions: &[Vector3<R>]) {
        if !can_split(&self.lower_corner, &self.upper_corner) {
            return;
        }

        // Convert node body from Leaf to Children
        let new_body = if let NodeBody::Leaf { particles } = &self.body {
            let split_point = get_split_point(grid, &self.lower_corner, &self.upper_corner)
                .expect("Failed to get split point of octree node");
            let split_coordinates = grid.point_coordinates(&split_point);

            println!(
                "Split point: {:?} (lower: {:?}, upper: {:?})",
                split_point, self.lower_corner, self.upper_corner
            );

            let particles = particles.clone();
            let mut octants = vec![OctantFlags::default(); particles.len()];
            let mut counters: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

            assert_eq!(particles.len(), octants.len());
            for (particle, octant) in particles.iter().copied().zip(octants.iter_mut()) {
                let relative_pos = split_coordinates - particle_positions[particle];
                *octant = OctantFlags::classify(&relative_pos);
                counters[Octant::from_flags(*octant) as usize] += 1;
            }

            println!(
                "Subdivision into octants with {:?} particles each",
                counters
            );

            let mut children = Vec::with_capacity(8);
            for (octant_flags, octant_particle_count) in OctantFlags::all()
                .iter()
                .copied()
                .zip(counters.iter().copied())
            {
                let lower_corner = octant_flags
                    .combine_point_index(grid, &self.lower_corner, &split_point)
                    .expect("Failed to get corner point of octree subcell");
                let upper_corner = octant_flags
                    .combine_point_index(grid, &split_point, &self.upper_corner)
                    .expect("Failed to get corner point of octree subcell");

                let mut octant_particles = Vec::with_capacity(octant_particle_count);
                for (i, octant_i) in octants.iter().copied().enumerate() {
                    if octant_i == octant_flags {
                        octant_particles.push(i);
                    }
                }

                let child = Box::new(OctreeNode::new_leaf(
                    lower_corner,
                    upper_corner,
                    octant_particles,
                ));

                children.push(child);
            }

            NodeBody::new_with_children(children)
        } else {
            panic!("Cannot subdivide a non-leaf octree node");
        };

        self.body = new_body;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct OctantFlags {
    x_axis: Direction,
    y_axis: Direction,
    z_axis: Direction,
}

impl OctantFlags {
    #[inline(always)]
    pub const fn all() -> &'static [OctantFlags; 8] {
        &ALL_OCTANT_FLAGS
    }

    #[inline(always)]
    pub const fn from_bool(x_positive: bool, y_positive: bool, z_positive: bool) -> Self {
        Self {
            x_axis: Direction::from_bool(x_positive),
            y_axis: Direction::from_bool(y_positive),
            z_axis: Direction::from_bool(z_positive),
        }
    }

    pub const fn from_octant(octant: Octant) -> Self {
        match octant {
            Octant::NegNegNeg => Self::from_bool(false, false, false),
            Octant::PosNegNeg => Self::from_bool(true, false, false),
            Octant::NegPosNeg => Self::from_bool(false, true, false),
            Octant::PosPosNeg => Self::from_bool(true, true, false),
            Octant::NegNegPos => Self::from_bool(false, false, true),
            Octant::PosNegPos => Self::from_bool(true, false, true),
            Octant::NegPosPos => Self::from_bool(false, true, true),
            Octant::PosPosPos => Self::from_bool(true, true, true),
        }
    }

    /// Classifies a point relative to zero into the corresponding octant
    #[inline(always)]
    pub fn classify<R: Real>(point: &Vector3<R>) -> Self {
        Self::from_bool(
            point[0].is_positive(),
            point[1].is_positive(),
            point[2].is_positive(),
        )
    }

    /// Combines two vectors by choosing between their components depending on the octant
    pub fn combine_point_index<I: Index, R: Real>(
        &self,
        grid: &UniformGrid<I, R>,
        lower: &PointIndex<I>,
        upper: &PointIndex<I>,
    ) -> Option<PointIndex<I>> {
        let lower = lower.index();
        let upper = upper.index();

        let combined_index = [
            if self.x_axis.is_positive() {
                upper[0]
            } else {
                lower[0]
            },
            if self.y_axis.is_positive() {
                upper[1]
            } else {
                lower[1]
            },
            if self.z_axis.is_positive() {
                upper[2]
            } else {
                lower[2]
            },
        ];

        grid.get_point(&combined_index)
    }

    /// Combines two vectors by choosing between their components depending on the octant
    pub fn combine<R: Real>(&self, lower: &Vector3<R>, upper: &Vector3<R>) -> Vector3<R> {
        Vector3::new(
            if self.x_axis.is_positive() {
                upper[0]
            } else {
                lower[0]
            },
            if self.y_axis.is_positive() {
                upper[1]
            } else {
                lower[1]
            },
            if self.z_axis.is_positive() {
                upper[2]
            } else {
                lower[2]
            },
        )
    }
}

impl From<Octant> for OctantFlags {
    fn from(octant: Octant) -> Self {
        Self::from_octant(octant)
    }
}

impl Default for OctantFlags {
    fn default() -> Self {
        OctantFlags {
            x_axis: Direction::Negative,
            y_axis: Direction::Negative,
            z_axis: Direction::Negative,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Octant {
    NegNegNeg = 0,
    PosNegNeg = 1,
    NegPosNeg = 2,
    PosPosNeg = 3,
    NegNegPos = 4,
    PosNegPos = 5,
    NegPosPos = 6,
    PosPosPos = 7,
}

impl Octant {
    #[inline(always)]
    pub const fn all() -> &'static [Octant; 8] {
        &ALL_OCTANTS
    }

    #[inline(always)]
    pub const fn from_flags(flags: OctantFlags) -> Self {
        use Direction::*;
        let OctantFlags {
            x_axis,
            y_axis,
            z_axis,
        } = flags;
        match (x_axis, y_axis, z_axis) {
            (Negative, Negative, Negative) => Octant::NegNegNeg,
            (Positive, Negative, Negative) => Octant::PosNegNeg,
            (Negative, Positive, Negative) => Octant::NegPosNeg,
            (Positive, Positive, Negative) => Octant::PosPosNeg,
            (Negative, Negative, Positive) => Octant::NegNegPos,
            (Positive, Negative, Positive) => Octant::PosNegPos,
            (Negative, Positive, Positive) => Octant::NegPosPos,
            (Positive, Positive, Positive) => Octant::PosPosPos,
        }
    }
}

impl From<OctantFlags> for Octant {
    fn from(flags: OctantFlags) -> Self {
        Self::from_flags(flags)
    }
}

const ALL_OCTANTS: [Octant; 8] = [
    Octant::NegNegNeg,
    Octant::PosNegNeg,
    Octant::NegPosNeg,
    Octant::PosPosNeg,
    Octant::NegNegPos,
    Octant::PosNegPos,
    Octant::NegPosPos,
    Octant::PosPosPos,
];

const ALL_OCTANT_FLAGS: [OctantFlags; 8] = [
    OctantFlags::from_octant(Octant::NegNegNeg),
    OctantFlags::from_octant(Octant::PosNegNeg),
    OctantFlags::from_octant(Octant::NegPosNeg),
    OctantFlags::from_octant(Octant::PosPosNeg),
    OctantFlags::from_octant(Octant::NegNegPos),
    OctantFlags::from_octant(Octant::PosNegPos),
    OctantFlags::from_octant(Octant::NegPosPos),
    OctantFlags::from_octant(Octant::PosPosPos),
];

#[cfg(test)]
mod test_octant {
    use super::*;

    #[test]
    fn test_octant_iter_all_consistency() {
        for (i, octant) in Octant::all().iter().copied().enumerate() {
            assert_eq!(octant as usize, i);
            assert_eq!(octant, unsafe {
                std::mem::transmute::<u8, Octant>(i as u8)
            });
        }
    }

    #[test]
    fn test_octant_flags_iter_all_consistency() {
        assert_eq!(Octant::all().len(), OctantFlags::all().len());
        for (octant, octant_flags) in Octant::all()
            .iter()
            .copied()
            .zip(OctantFlags::all().iter().copied())
        {
            assert_eq!(octant, Octant::from(octant_flags));
            assert_eq!(octant_flags, OctantFlags::from(octant));
        }
    }
}

fn can_split<I: Index>(lower: &PointIndex<I>, upper: &PointIndex<I>) -> bool {
    let lower = lower.index();
    let upper = upper.index();

    upper[0] - lower[0] > I::one()
        && upper[1] - lower[1] > I::one()
        && upper[2] - lower[2] > I::one()
}

fn get_split_point<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    lower: &PointIndex<I>,
    upper: &PointIndex<I>,
) -> Option<PointIndex<I>> {
    let two = I::one() + I::one();

    let lower = lower.index();
    let upper = upper.index();

    let mid_indices = [
        (lower[0] + upper[0]) / two,
        (lower[1] + upper[1]) / two,
        (lower[2] + upper[2]) / two,
    ];

    grid.get_point(&mid_indices)
}
