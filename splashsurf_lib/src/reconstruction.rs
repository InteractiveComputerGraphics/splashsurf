use crate::dense_subdomains::{
    compute_global_densities_and_neighbors, decomposition, initialize_parameters, reconstruction,
    stitching, subdomain_classification::GhostMarginClassifier,
};
use crate::mesh::TriMesh3d;
use crate::uniform_grid::UniformGrid;
use crate::workspace::LocalReconstructionWorkspace;
use crate::{
    density_map, marching_cubes, neighborhood_search, profile, Index, Parameters, Real,
    ReconstructionError, SurfaceReconstruction,
};
use anyhow::Context;
use log::{info, trace};
use nalgebra::Vector3;

/// Performs a surface reconstruction with a regular grid for domain decomposition
pub(crate) fn reconstruct_surface_subdomain_grid<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction subdomain-grid");

    let internal_parameters =
        initialize_parameters(parameters, &particle_positions, output_surface)?;
    output_surface.grid = internal_parameters
        .global_marching_cubes_grid()
        .context("failed to convert global marching cubes grid")?;

    // Filter "narrow band"
    /*
    let narrow_band_particles = extract_narrow_band(&parameters, &particles);
    let particles = narrow_band_particles;
     */

    let subdomains =
        decomposition::<I, R, GhostMarginClassifier<I>>(&internal_parameters, &particle_positions)?;

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

    let (particle_densities, particle_neighbors) = compute_global_densities_and_neighbors(
        &internal_parameters,
        &particle_positions,
        &subdomains,
    );

    let surface_patches = reconstruction(
        &internal_parameters,
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
    if parameters.global_neighborhood_list {
        output_surface.particle_neighbors = Some(particle_neighbors);
    }
    Ok(())
}

/// Performs a global surface reconstruction without domain decomposition
pub(crate) fn reconstruct_surface_global<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_global");

    // Multiple local workspaces are only needed for processing different subdomains in parallel.
    // However, in this global surface reconstruction without domain decomposition, each step in the
    // reconstruction pipeline manages its memory on its own.
    let mut workspace = output_surface
        .workspace
        .get_local_with_capacity(particle_positions.len())
        .borrow_mut();

    // Reuse allocated memory: swap particle densities from output object into the workspace if the former has a larger capacity
    if let Some(output_densities) = output_surface.particle_densities.as_ref() {
        if output_densities.capacity() > output_surface.workspace.densities().capacity() {
            std::mem::swap(
                output_surface.particle_densities.as_mut().unwrap(),
                &mut workspace.particle_densities,
            );
        }
    }

    // Clear the current mesh, as reconstruction will be appended to output
    output_surface.mesh.clear();
    // Perform global reconstruction without domain decomposition
    reconstruct_single_surface_append(
        &mut *workspace,
        &output_surface.grid,
        particle_positions,
        parameters,
        &mut output_surface.mesh,
    )?;

    output_surface.particle_densities = Some(std::mem::take(&mut workspace.particle_densities));

    Ok(())
}

/// Performs global neighborhood search and computes per particle densities
pub(crate) fn compute_particle_densities_and_neighbors<I: Index, R: Real>(
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    particle_neighbor_lists: &mut Vec<Vec<usize>>,
    densities: &mut Vec<R>,
) {
    profile!("compute_particle_densities_and_neighbors");

    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    trace!("Starting neighborhood search...");
    neighborhood_search::search_inplace::<I, R>(
        &grid.aabb(),
        particle_positions,
        parameters.compact_support_radius,
        parameters.enable_multi_threading,
        particle_neighbor_lists,
    );

    trace!("Computing particle densities...");
    density_map::compute_particle_densities_inplace::<I, R>(
        particle_positions,
        particle_neighbor_lists.as_slice(),
        parameters.compact_support_radius,
        particle_rest_mass,
        parameters.enable_multi_threading,
        densities,
    );
}

/// Reconstruct a surface, appends triangulation to the given mesh
pub(crate) fn reconstruct_single_surface_append<'a, I: Index, R: Real>(
    workspace: &mut LocalReconstructionWorkspace<I, R>,
    grid: &UniformGrid<I, R>,
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_mesh: &'a mut TriMesh3d<R>,
) -> Result<(), ReconstructionError<I, R>> {
    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    let particle_densities = {
        compute_particle_densities_and_neighbors(
            grid,
            particle_positions,
            parameters,
            &mut workspace.particle_neighbor_lists,
            &mut workspace.particle_densities,
        );
        workspace.particle_densities.as_slice()
    };

    // Create a new density map, reusing memory with the workspace is bad for cache efficiency
    // Alternatively one could reuse memory with a custom caching allocator
    let mut density_map = Default::default();
    density_map::generate_sparse_density_map(
        grid,
        particle_positions,
        particle_densities,
        None,
        particle_rest_mass,
        parameters.compact_support_radius,
        parameters.cube_size,
        parameters.enable_multi_threading,
        &mut density_map,
    )?;

    marching_cubes::triangulate_density_map_append(
        grid,
        &density_map,
        parameters.iso_surface_threshold,
        output_mesh,
    )?;

    Ok(())
}
