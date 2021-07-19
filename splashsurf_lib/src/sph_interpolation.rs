//! Functions for interpolating quantities (e.g. normals, scalar fields) by evaluating SPH sums

use crate::kernel;
use crate::kernel::SymmetricKernel3d;
use crate::profile;
use crate::Real;
use nalgebra::{Unit, Vector3};
use rayon::prelude::*;
use rstar::primitives::PointWithData;
use rstar::RTree;

/// Particle type that is stored in the R-tree for fast SPH neighbor queries
type Particle<R> = PointWithData<ParticleData<R>, [R; 3]>;

/// Data associated with each particle that is stored in the R-tree
struct ParticleData<R: Real> {
    /// Volume associated with each particle which is needed to evaluate the SPH density field
    volume: R,
}

/// Computes the surface normals at the specified interpolation points using the gradient of the SPH density field of the given particles
pub fn compute_sph_normals<R: Real>(
    interpolation_points: &[Vector3<R>],
    particle_positions: &[Vector3<R>],
    particle_densities: &[R],
    particle_rest_mass: R,
    compact_support_radius: R,
) -> Vec<Unit<Vector3<R>>> {
    profile!("compute_sph_normals");

    let squared_support = compact_support_radius * compact_support_radius;

    // Prepare data for R-tree insertion:
    // Collect all particles with their position and compute their volume
    let particles = particle_positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let data = ParticleData {
                volume: particle_rest_mass / particle_densities[i],
            };
            Particle::new(data, [p.x, p.y, p.z])
        })
        .collect();

    // Build the R-tree to accelerate SPH neighbor queries near the interpolation points
    // This is faster than a full neighborhood search if there are less interpolation points than particles
    let tree = {
        profile!("build R-tree");
        RTree::bulk_load(particles)
    };

    let kernel = kernel::CubicSplineKernel::new(compact_support_radius);

    let normals = {
        profile!("evaluate normals with SPH sum");
        let mut normals = Vec::with_capacity(interpolation_points.len());
        interpolation_points
            .par_iter()
            .map(|x_i| {
                // Compute the gradient of the particle density field which points in the same direction as surface normals
                let mut density_grad = Vector3::zeros();

                // SPH: Iterate over all other particles within the squared support radius
                for p_j in tree.locate_within_distance([x_i.x, x_i.y, x_i.z], squared_support) {
                    // Volume of the neighbor particle
                    let vol_j = p_j.data.volume;
                    // Position of the neighbor particle
                    let x_j = Vector3::new(p_j.position()[0], p_j.position()[1], p_j.position()[2]);

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
            .collect_into_vec(&mut normals);
        normals
    };

    normals
}
