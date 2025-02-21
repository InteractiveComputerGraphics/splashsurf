//! Functions for interpolating quantities (e.g. normals, scalar fields) by evaluating SPH sums

use crate::Real;
use crate::kernel::SymmetricKernel3d;
use crate::profile;
use crate::{ThreadSafe, kernel};
use nalgebra::{SVector, Unit, Vector3};
use rayon::prelude::*;
use rstar::RTree;
use rstar::primitives::GeomWithData;
use std::ops::AddAssign;

/// Acceleration structure for interpolating field quantities of the fluid to arbitrary points using SPH interpolation
pub struct SphInterpolator<R: Real> {
    compact_support_radius: R,
    tree: RTree<Particle<R>>,
}

/// Particle type that is stored in the R-tree for fast SPH neighbor queries
type Particle<R> = GeomWithData<[R; 3], ParticleData<R>>;

/// Data associated with each particle that is stored in the R-tree
struct ParticleData<R: Real> {
    /// Index of the particle in the global particle list
    index: usize,
    /// Volume associated with each particle which is needed to evaluate the SPH density field
    volume: R,
}

/// Trait for per-particle quantities that can be interpolated using SPH
trait InterpolationQuantity<R: Real>: Clone + AddAssign + ThreadSafe {
    /// Initializes a value of zero
    fn zero() -> Self;
    /// Scales this quantity by the given factor
    fn scale(&self, factor: R) -> Self;
}

impl<R: Real> InterpolationQuantity<R> for R {
    fn zero() -> R {
        R::zero()
    }

    fn scale(&self, factor: R) -> R {
        *self * factor
    }
}

impl<R: Real, const D: usize> InterpolationQuantity<R> for SVector<R, D> {
    fn zero() -> Self {
        Self::zeros()
    }

    fn scale(&self, factor: R) -> Self {
        Self::scale(self, factor)
    }
}

impl<R: Real> SphInterpolator<R> {
    /// Initializes the acceleration structure for interpolating values of the given fluid particles, this is a relatively expensive operation (builds an R-tree)
    pub fn new(
        particle_positions: &[Vector3<R>],
        particle_densities: &[R],
        particle_rest_mass: R,
        compact_support_radius: R,
    ) -> Self {
        assert_eq!(particle_positions.len(), particle_densities.len());

        let tree = build_rtree(particle_positions, particle_densities, particle_rest_mass);

        Self {
            compact_support_radius,
            tree,
        }
    }

    /// Interpolates surface normals (i.e. normalized SPH gradient of the indicator function) of the fluid to the given points using SPH interpolation, appends to the given vector
    pub fn interpolate_normals_inplace(
        &self,
        interpolation_points: &[Vector3<R>],
        normals: &mut Vec<Unit<Vector3<R>>>,
    ) {
        profile!("interpolate_normals_inplace");

        let squared_support = self.compact_support_radius * self.compact_support_radius;
        let kernel = kernel::CubicSplineKernel::new(self.compact_support_radius);

        interpolation_points
            .par_iter()
            .map(|x_i| {
                // Compute the gradient of the particle density field which points in the same direction as surface normals
                let mut density_grad = Vector3::zeros();

                // SPH: Iterate over all other particles within the squared support radius
                let query_point = bytemuck::cast::<_, [R; 3]>(*x_i);
                for p_j in self
                    .tree
                    .locate_within_distance(query_point, squared_support)
                {
                    // Volume of the neighbor particle
                    let vol_j = p_j.data.volume;
                    // Position of the neighbor particle
                    let x_j = bytemuck::cast_ref::<_, Vector3<R>>(p_j.geom());

                    // Relative position `dx` and distance `r` of the neighbor particle
                    let dx = x_j - x_i;
                    let r = dx.norm();

                    // Compute the contribution of the neighbor to the gradient of the density field
                    // TODO: Replace this by a discrete gradient norm evaluation
                    let kernel_grad = dx.unscale(r) * kernel.evaluate_gradient_norm(r);
                    density_grad += kernel_grad * vol_j;
                }

                // Normalize the gradient to get the surface normal
                Unit::new_normalize(density_grad)
            })
            .collect_into_vec(normals);
    }

    /// Interpolates surface normals (i.e. normalized SPH gradient of the indicator function) of the fluid to the given points using SPH interpolation
    pub fn interpolate_normals(
        &self,
        interpolation_points: &[Vector3<R>],
    ) -> Vec<Unit<Vector3<R>>> {
        let mut normals = Vec::with_capacity(interpolation_points.len());
        self.interpolate_normals_inplace(interpolation_points, &mut normals);
        normals
    }

    /// Interpolates a scalar per particle quantity to the given points, panics if the there are less per-particles values than particles, appends to the given vector
    #[allow(non_snake_case)]
    fn interpolate_scalar_quantity_inplace(
        &self,
        particle_quantity: &[R],
        interpolation_points: &[Vector3<R>],
        interpolated_values: &mut Vec<R>,
        first_order_correction: bool,
    ) {
        self.interpolate_quantity_inplace(
            particle_quantity,
            interpolation_points,
            interpolated_values,
            first_order_correction,
        )
    }

    /// Interpolates a scalar per particle quantity to the given points, panics if the there are less per-particles values than particles
    pub fn interpolate_scalar_quantity(
        &self,
        particle_quantity: &[R],
        interpolation_points: &[Vector3<R>],
        first_order_correction: bool,
    ) -> Vec<R> {
        let mut values = Vec::with_capacity(interpolation_points.len());
        self.interpolate_scalar_quantity_inplace(
            particle_quantity,
            interpolation_points,
            &mut values,
            first_order_correction,
        );
        values
    }

    /// Interpolates a vectorial per particle quantity to the given points, panics if the there are less per-particles values than particles, appends to the given vector
    #[allow(non_snake_case)]
    fn interpolate_vector_quantity_inplace<const D: usize>(
        &self,
        particle_quantity: &[SVector<R, D>],
        interpolation_points: &[Vector3<R>],
        interpolated_values: &mut Vec<SVector<R, D>>,
        first_order_correction: bool,
    ) {
        self.interpolate_quantity_inplace(
            particle_quantity,
            interpolation_points,
            interpolated_values,
            first_order_correction,
        )
    }

    /// Interpolates a vectorial per particle quantity to the given points, panics if the there are less per-particles values than particles
    pub fn interpolate_vector_quantity<const D: usize>(
        &self,
        particle_quantity: &[SVector<R, D>],
        interpolation_points: &[Vector3<R>],
        first_order_correction: bool,
    ) -> Vec<SVector<R, D>> {
        let mut values = Vec::with_capacity(interpolation_points.len());
        self.interpolate_vector_quantity_inplace(
            particle_quantity,
            interpolation_points,
            &mut values,
            first_order_correction,
        );
        values
    }

    /// Interpolates a per particle quantity to the given points, panics if the there are less per-particles values than particles, appends to the given vector
    #[allow(non_snake_case)]
    fn interpolate_quantity_inplace<T: InterpolationQuantity<R>>(
        &self,
        particle_quantity: &[T],
        interpolation_points: &[Vector3<R>],
        interpolated_values: &mut Vec<T>,
        first_order_correction: bool,
    ) {
        profile!("interpolate_quantity_inplace");
        assert_eq!(particle_quantity.len(), self.tree.size());

        let squared_support = self.compact_support_radius * self.compact_support_radius;
        let kernel = kernel::CubicSplineKernel::new(self.compact_support_radius);

        let enable_correction = if first_order_correction {
            R::one()
        } else {
            R::zero()
        };

        interpolation_points
            .par_iter()
            .map(|x_i| {
                let mut interpolated_value = T::zero();
                let mut correction = R::zero();

                // SPH: Iterate over all other particles within the squared support radius
                let query_point = bytemuck::cast::<_, [R; 3]>(*x_i);
                for p_j in self
                    .tree
                    .locate_within_distance(query_point, squared_support)
                {
                    // Volume of the neighbor particle
                    let vol_j = p_j.data.volume;
                    // Position of the neighbor particle
                    let x_j = bytemuck::cast_ref::<_, Vector3<R>>(p_j.geom());

                    // Relative position `dx` and distance `r` of the neighbor particle
                    let dx = x_j - x_i;
                    let r = dx.norm();

                    // Unchecked access is fine as we asserted before that the slice has the correct length
                    let A_j = unsafe { particle_quantity.get_unchecked(p_j.data.index).clone() };
                    let W_ij = kernel.evaluate(r);

                    interpolated_value += A_j.scale(vol_j * W_ij);
                    correction += vol_j * W_ij;
                }

                let correction_factor =
                    enable_correction * correction.recip() + (R::one() - enable_correction);
                interpolated_value.scale(correction_factor)
            })
            .collect_into_vec(interpolated_values);
    }
}

/// Constructs an R-Tree for the given particles, storing each particle's index and volume in the tree
fn build_rtree<R: Real>(
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    particle_rest_mass: R,
) -> RTree<Particle<R>> {
    assert_eq!(particle_positions.len(), particle_densities.len());

    // Prepare data for R-tree insertion:
    // Collect all particles with their position and compute their volume
    let particles = particle_positions
        .iter()
        .zip(particle_densities.iter().copied())
        .enumerate()
        .map(|(i, (p, rho_i))| {
            let data = ParticleData {
                index: i,
                volume: particle_rest_mass / rho_i,
            };
            Particle::new(bytemuck::cast(*p), data)
        })
        .collect();

    // Build the R-tree to accelerate SPH neighbor queries near the interpolation points

    {
        profile!("build R-tree");
        RTree::bulk_load(particles)
    }
}
