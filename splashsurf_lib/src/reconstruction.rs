use log::info;
use nalgebra::Vector3;

use crate::dense_subdomains::{
    compute_global_density_vector, decomposition, initialize_parameters, reconstruction, stitching,
    subdomain_classification::GhostMarginClassifier,
};
use crate::{profile, Index, Parameters, Real, SurfaceReconstruction};

/// Performs a surface reconstruction with a regular grid for domain decomposition
pub(crate) fn reconstruct_surface_subdomain_grid<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction subdomain-grid");

    let parameters = initialize_parameters(parameters, &particle_positions, output_surface)?;

    // Filter "narrow band"
    /*
    let narrow_band_particles = extract_narrow_band(&parameters, &particles);
    let particles = narrow_band_particles;
     */

    let subdomains =
        decomposition::<I, R, GhostMarginClassifier<I>>(&parameters, &particle_positions)?;

    /*
    {
        use super::dense_subdomains::debug::*;
        subdomain_stats(&parameters, &particle_positions, &subdomains);
        info!(
            "Number of subdomains with only ghost particles: {}",
            count_no_owned_particles_subdomains(&parameters, &particle_positions, &subdomains)
        );
    }
     */

    let particle_densities =
        compute_global_density_vector(&parameters, &particle_positions, &subdomains);

    let surface_patches = reconstruction(
        &parameters,
        &particle_positions,
        &particle_densities,
        &subdomains,
    );

    let global_mesh = stitching(surface_patches);
    info!(
        "Global mesh has {} vertices and {} triangles.",
        global_mesh.vertices.len(),
        global_mesh.triangles.len()
    );

    output_surface.mesh = global_mesh;
    output_surface.particle_densities = Some(particle_densities);
    Ok(())
}
