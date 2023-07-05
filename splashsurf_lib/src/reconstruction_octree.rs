//! Helper functions calling the individual steps of the reconstruction pipeline

use crate::generic_tree::*;
use crate::marching_cubes::SurfacePatch;
use crate::mesh::TriMesh3d;
use crate::octree::{NodeData, Octree, OctreeNode};
use crate::uniform_grid::{OwningSubdomainGrid, Subdomain, UniformGrid};
use crate::workspace::LocalReconstructionWorkspace;
use crate::{
    density_map, marching_cubes, neighborhood_search, new_map, profile, utils, Index, Parameters,
    ParticleDensityComputationStrategy, Real, ReconstructionError, SpatialDecompositionParameters,
    SurfaceReconstruction,
};
use log::{debug, info, trace};
use nalgebra::Vector3;
use num_traits::Bounded;
use parking_lot::Mutex;

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
    // Perform global reconstruction without octree
    reconstruct_single_surface_append(
        &mut *workspace,
        &output_surface.grid,
        None,
        particle_positions,
        None,
        parameters,
        &mut output_surface.mesh,
    )?;

    // TODO: Set this correctly
    output_surface.density_map = None;
    output_surface.particle_densities = Some(std::mem::take(&mut workspace.particle_densities));

    Ok(())
}

/// Performs a surface reconstruction with an octree for domain decomposition
pub(crate) fn reconstruct_surface_domain_decomposition<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
) -> Result<(), ReconstructionError<I, R>> {
    profile!("reconstruct_surface_domain_decomposition");

    OctreeBasedSurfaceReconstruction::new(particle_positions, parameters, output_surface)
        .expect("Unable to construct octree. Missing/invalid decomposition parameters?")
        .run(particle_positions, output_surface)?;

    Ok(())
}

/// Helper type for performing an octree based surface reconstruction
struct OctreeBasedSurfaceReconstruction<I: Index, R: Real> {
    /// General parameters for the surface reconstruction
    parameters: Parameters<R>,
    /// Spatial decomposition specific parameters extracted from the general parameters
    spatial_decomposition: SpatialDecompositionParameters<R>,
    /// The implicit global grid for the entire surface reconstruction
    grid: UniformGrid<I, R>,
    /// Octree containing the individual particle subdomains, built during the construction of this helper type
    octree: Octree<I, R>,
}

// TODO: Make this less object oriented?
impl<I: Index, R: Real> OctreeBasedSurfaceReconstruction<I, R> {
    /// Initializes the octree based surface reconstruction by constructing the corresponding octree
    fn new(
        global_particle_positions: &[Vector3<R>],
        parameters: &Parameters<R>,
        output_surface: &SurfaceReconstruction<I, R>,
    ) -> Option<Self> {
        // The grid was already generated by the calling public function
        let grid = output_surface.grid.clone();

        // Construct the octree
        let octree = if let Some(decomposition_parameters) = &parameters.spatial_decomposition {
            let margin_factor = decomposition_parameters
                .ghost_particle_safety_factor
                .unwrap_or(R::one());

            Octree::new_subdivided(
                &grid,
                global_particle_positions,
                decomposition_parameters.subdivision_criterion.clone(),
                parameters.compact_support_radius * margin_factor,
                parameters.enable_multi_threading,
                decomposition_parameters.enable_stitching,
            )
        } else {
            // TODO: Use default values instead?

            // If there are no decomposition parameters, we cannot construct an octree.
            return None;
        };

        // Disable all multi-threading in sub-tasks for now (instead, entire sub-tasks are processed in parallel)
        let parameters = {
            let mut p = parameters.clone();
            p.enable_multi_threading = false;
            p
        };

        Some(Self {
            octree,
            spatial_decomposition: parameters.spatial_decomposition.as_ref().unwrap().clone(),
            grid,
            parameters,
        })
    }

    /// Runs the surface reconstruction on the initialized octree
    fn run(
        self,
        global_particle_positions: &[Vector3<R>],
        output_surface: &mut SurfaceReconstruction<I, R>,
    ) -> Result<(), ReconstructionError<I, R>> {
        // Reuse allocated memory: swap particle densities from output object into the workspace if the former has a larger capacity
        if let Some(output_densities) = output_surface.particle_densities.as_ref() {
            if output_densities.capacity() > output_surface.workspace.densities().capacity() {
                std::mem::swap(
                    output_surface.particle_densities.as_mut().unwrap(),
                    output_surface.workspace.densities_mut(),
                );
            }
        }

        // Compute particle densities depending on the selected strategy
        let global_particle_densities_vec =
            match self.spatial_decomposition.particle_density_computation {
                // Strategy 1: compute particle densities globally
                ParticleDensityComputationStrategy::Global => {
                    Some(Self::compute_particle_densities_global(
                        global_particle_positions,
                        &self.grid,
                        &self.parameters,
                        output_surface,
                    ));
                    Some(std::mem::take(output_surface.workspace.densities_mut()))
                }
                // Strategy 2: compute and merge particle densities per subdomain
                ParticleDensityComputationStrategy::SynchronizeSubdomains => {
                    Some(Self::compute_particle_densities_local(
                        global_particle_positions,
                        &self.grid,
                        &self.octree,
                        &self.parameters,
                        output_surface,
                    ));
                    Some(std::mem::take(output_surface.workspace.densities_mut()))
                }
                // Strategy 3: each subdomain will compute densities later on its own
                // (can only work correctly if margin is large enough)
                ParticleDensityComputationStrategy::IndependentSubdomains => None,
            };

        {
            let global_particle_densities =
                global_particle_densities_vec.as_ref().map(|v| v.as_slice());

            // Run surface reconstruction
            if self.spatial_decomposition.enable_stitching {
                self.run_with_stitching(
                    global_particle_positions,
                    global_particle_densities,
                    output_surface,
                )?;
            } else {
                self.run_without_stitching(
                    global_particle_positions,
                    global_particle_densities,
                    output_surface,
                )?;
            }

            info!(
                "Global mesh has {} triangles and {} vertices.",
                output_surface.mesh.triangles.len(),
                output_surface.mesh.vertices.len()
            );
        }

        output_surface.octree = Some(self.octree);
        output_surface.density_map = None;
        output_surface.particle_densities = global_particle_densities_vec;

        Ok(())
    }

    /// Computes the particle densities globally on all particles without any domain decomposition
    fn compute_particle_densities_global(
        global_particle_positions: &[Vector3<R>],
        grid: &UniformGrid<I, R>,
        parameters: &Parameters<R>,
        output_surface: &mut SurfaceReconstruction<I, R>,
    ) {
        let mut densities = std::mem::take(output_surface.workspace.densities_mut());

        {
            let mut workspace = output_surface.workspace.get_local().borrow_mut();
            compute_particle_densities_and_neighbors(
                grid,
                global_particle_positions,
                parameters,
                &mut workspace.particle_neighbor_lists,
                &mut densities,
            );
        }

        *output_surface.workspace.densities_mut() = densities;
    }

    /// Computes the particles densities per subdomain followed by merging them into a global vector
    fn compute_particle_densities_local(
        global_particle_positions: &[Vector3<R>],
        grid: &UniformGrid<I, R>,
        octree: &Octree<I, R>,
        parameters: &Parameters<R>,
        output_surface: &mut SurfaceReconstruction<I, R>,
    ) {
        profile!(
            parent_scope,
            "parallel subdomain particle density computation"
        );
        info!("Starting computation of particle densities.");

        // Take the global density storage from workspace to move it behind a mutex
        let mut global_densities = std::mem::take(output_surface.workspace.densities_mut());
        utils::resize_and_fill(
            &mut global_densities,
            global_particle_positions.len(),
            <R as Bounded>::min_value(),
            parameters.enable_multi_threading,
        );
        let global_densities = Mutex::new(global_densities);

        let tl_workspaces = &output_surface.workspace;

        octree
            .root()
            .par_visit_bfs(|octree_node: &OctreeNode<I, R>| {
                profile!(
                    "visit octree node for density computation",
                    parent = parent_scope
                );

                let node_particles = if let Some(particle_set) = octree_node.data().particle_set() {
                    &particle_set.particles
                } else {
                    // Skip non-leaf nodes
                    return;
                };

                let mut tl_workspace_ref_mut = tl_workspaces
                    .get_local_with_capacity(node_particles.len())
                    .borrow_mut();
                let tl_workspace = &mut *tl_workspace_ref_mut;

                Self::collect_node_particle_positions(
                    node_particles,
                    global_particle_positions,
                    &mut tl_workspace.particle_positions,
                );

                compute_particle_densities_and_neighbors(
                    grid,
                    tl_workspace.particle_positions.as_slice(),
                    parameters,
                    &mut tl_workspace.particle_neighbor_lists,
                    &mut tl_workspace.particle_densities,
                );

                {
                    profile!("update global density values");

                    let mut global_densities = global_densities.lock();
                    for (&global_idx, (&density, position)) in node_particles.iter().zip(
                        tl_workspace
                            .particle_densities
                            .iter()
                            .zip(tl_workspace.particle_positions.iter()),
                    ) {
                        // Check if the particle is actually inside of the cell and not a ghost particle
                        if octree_node.aabb().contains_point(position) {
                            global_densities[global_idx] = density;
                        }
                    }
                }
            });

        // Unpack densities from mutex and move back into workspace
        *output_surface.workspace.densities_mut() = global_densities.into_inner();
    }

    /// Performs surface reconstruction without stitching by visiting all octree leaf nodes
    fn run_without_stitching(
        &self,
        global_particle_positions: &[Vector3<R>],
        global_particle_densities: Option<&[R]>,
        output_surface: &mut SurfaceReconstruction<I, R>,
    ) -> Result<(), ReconstructionError<I, R>> {
        // Clear all local meshes
        {
            let tl_workspaces = &mut output_surface.workspace;
            // Clear all thread local meshes
            tl_workspaces
                .local_workspaces_mut()
                .iter_mut()
                .for_each(|local_workspace| {
                    local_workspace.borrow_mut().mesh.clear();
                });
        }

        // Perform individual surface reconstructions on all non-empty leaves of the octree
        {
            let tl_workspaces = &output_surface.workspace;

            profile!(parent_scope, "parallel subdomain surf. rec.");
            info!("Starting triangulation of surface patches.");

            self.octree
                .root()
                .try_par_visit_bfs(|octree_node: &OctreeNode<I, R>| -> Result<(), ReconstructionError<I, R>> {
                    let particles = if let Some(particle_set) = octree_node.data().particle_set() {
                        &particle_set.particles
                    } else {
                        // Skip non-leaf nodes: as soon as all leaves are processed all work is done
                        return Ok(());
                    };

                    profile!("visit octree node for reconstruction", parent = parent_scope);
                    trace!("Processing octree leaf with {} particles", particles.len());

                    if particles.is_empty() {
                        return Ok(());
                    } else {
                        let subdomain_grid = self.extract_node_subdomain(octree_node);

                        debug!(
                            "Surface reconstruction of local patch with {} particles. (offset: {:?}, cells_per_dim: {:?})",
                            particles.len(),
                            subdomain_grid.subdomain_offset(),
                            subdomain_grid.subdomain_grid().cells_per_dim());

                        let mut tl_workspace = tl_workspaces
                            .get_local_with_capacity(particles.len())
                            .borrow_mut();

                        // Take particle position storage from workspace and fill it with positions of the leaf
                        let mut node_particle_positions = std::mem::take(&mut tl_workspace.particle_positions);
                        Self::collect_node_particle_positions(particles, global_particle_positions, &mut node_particle_positions);

                        // Take particle density storage from workspace and fill it with densities of the leaf
                        let node_particle_densities = if let Some(global_particle_densities) = global_particle_densities {
                            let mut node_particle_densities = std::mem::take(&mut tl_workspace.particle_densities);
                            Self::collect_node_particle_densities(particles, global_particle_densities, &mut node_particle_densities);
                            Some(node_particle_densities)
                        } else {
                            None
                        };

                        // Take the thread local mesh and append to it without clearing
                        let mut node_mesh = std::mem::take(&mut tl_workspace.mesh);

                        reconstruct_single_surface_append(
                            &mut *tl_workspace,
                            &self.grid,
                            Some(&subdomain_grid),
                            node_particle_positions.as_slice(),
                            node_particle_densities.as_ref().map(|v| v.as_slice()),
                            &self.parameters,
                            &mut node_mesh,
                        )?;

                        trace!("Surface patch successfully processed.");

                        // Put back everything taken from the workspace
                        tl_workspace.particle_positions = node_particle_positions;
                        tl_workspace.mesh = node_mesh;
                        if let Some(node_particle_densities) = node_particle_densities {
                            tl_workspace.particle_densities = node_particle_densities;
                        }

                        Ok(())
                    }
                })?;
        };

        // Append all thread local meshes to global mesh
        {
            let tl_workspaces = &mut output_surface.workspace;
            tl_workspaces.local_workspaces_mut().iter_mut().fold(
                &mut output_surface.mesh,
                |global_mesh, local_workspace| {
                    global_mesh.append(&mut local_workspace.borrow_mut().mesh);
                    global_mesh
                },
            );
        }

        Ok(())
    }

    /// Performs surface reconstruction with stitching on octree using DFS visitation: reconstruct leaf nodes first, then stitch the parent node as soon as all children are processed
    fn run_with_stitching(
        &self,
        global_particle_positions: &[Vector3<R>],
        global_particle_densities: Option<&[R]>,
        output_surface: &mut SurfaceReconstruction<I, R>,
    ) -> Result<(), ReconstructionError<I, R>> {
        let mut octree = self.octree.clone();

        // Perform individual surface reconstructions on all non-empty leaves of the octree
        {
            let tl_workspaces = &output_surface.workspace;

            profile!(
                parent_scope,
                "parallel domain decomposed surf. rec. with stitching"
            );
            info!("Starting triangulation of surface patches.");

            octree
                .root_mut()
                // Use DFS visitation as we can only start stitching after all child nodes of one node are reconstructed/stitched.
                .try_par_visit_mut_dfs_post(|octree_node: &mut OctreeNode<I, R>| -> Result<(), ReconstructionError<I, R>> {
                    profile!("visit octree node (reconstruct or stitch)", parent = parent_scope);

                    // Extract the set of particles of the current node
                    let particles = if let Some(particle_set) = octree_node.data().particle_set() {
                        &particle_set.particles
                    } else {
                        // If node has no particle set, its children were already processed so it can be stitched
                        octree_node.stitch_surface_patches(self.parameters.iso_surface_threshold)?;
                        // After stitching we can directly continue visting the next node
                        return Ok(());
                    };

                    trace!("Processing octree leaf with {} particles", particles.len());

                    let subdomain_grid = self.extract_node_subdomain(octree_node);
                    let surface_patch = if particles.is_empty() {
                        SurfacePatch::new_empty(subdomain_grid)
                    } else {
                        debug!(
                            "Reconstructing surface of local patch with {} particles. (offset: {:?}, cells_per_dim: {:?})",
                            particles.len(),
                            subdomain_grid.subdomain_offset(),
                            subdomain_grid.subdomain_grid().cells_per_dim()
                        );

                        let mut tl_workspace = tl_workspaces
                            .get_local_with_capacity(particles.len())
                            .borrow_mut();

                        // Take particle position storage from workspace and fill it with positions of the leaf
                        let mut node_particle_positions = std::mem::take(&mut tl_workspace.particle_positions);
                        Self::collect_node_particle_positions(particles, global_particle_positions, &mut node_particle_positions);

                        // Take particle density storage from workspace and fill it with densities of the leaf
                        let node_particle_densities = if let Some(global_particle_densities) = global_particle_densities {
                            let mut node_particle_densities = std::mem::take(&mut tl_workspace.particle_densities);
                            Self::collect_node_particle_densities(particles, global_particle_densities, &mut node_particle_densities);
                            Some(node_particle_densities)
                        } else {
                            None
                        };

                        let surface_patch = reconstruct_surface_patch(
                            &mut *tl_workspace,
                            &subdomain_grid,
                            node_particle_positions.as_slice(),
                            node_particle_densities.as_ref().map(|v| v.as_slice()),
                            &self.parameters,
                        );

                        // Put back everything taken from the workspace
                        tl_workspace.particle_positions = node_particle_positions;
                        if let Some(node_particle_densities) = node_particle_densities {
                            tl_workspace.particle_densities = node_particle_densities;
                        }

                        surface_patch?
                    };

                    trace!("Surface patch successfully processed.");

                    // Store triangulation in the leaf
                    octree_node
                        .data_mut()
                        .replace(NodeData::SurfacePatch(surface_patch.into()));

                    Ok(())
                })?;

            info!("Generation of surface patches is done.");
        };

        // Move stitched mesh out of octree
        {
            let surface_path = octree
                .root_mut()
                .data_mut()
                .take()
                .into_surface_patch()
                .expect("Cannot extract stitched mesh from root node")
                .patch;
            output_surface.mesh = surface_path.mesh;
        }

        Ok(())
    }

    /// Computes the subdomain grid for the given octree node
    fn extract_node_subdomain(&self, octree_node: &OctreeNode<I, R>) -> OwningSubdomainGrid<I, R> {
        let grid = &self.grid;

        let subdomain_grid = octree_node
            .grid(octree_node.aabb().min(), grid.cell_size())
            .expect("Unable to construct Octree node grid");
        let subdomain_offset = octree_node.min_corner();
        subdomain_grid.log_grid_info();

        OwningSubdomainGrid::new(grid.clone(), subdomain_grid, *subdomain_offset.index())
    }

    /// Collects the particle positions of all particles in the node
    fn collect_node_particle_positions(
        node_particles: &[usize],
        global_particle_positions: &[Vector3<R>],
        node_particle_positions: &mut Vec<Vector3<R>>,
    ) {
        node_particle_positions.clear();
        utils::reserve_total(node_particle_positions, node_particles.len());

        // Extract the particle positions of the leaf
        node_particle_positions.extend(
            node_particles
                .iter()
                .map(|&idx| global_particle_positions[idx]),
        );
    }

    /// Collects the density values of all particles in the node
    fn collect_node_particle_densities(
        node_particles: &[usize],
        global_particle_densities: &[R],
        node_particle_densities: &mut Vec<R>,
    ) {
        node_particle_densities.clear();
        utils::reserve_total(node_particle_densities, node_particles.len());

        // Extract the particle densities of the leaf
        node_particle_densities.extend(
            node_particles
                .iter()
                .map(|&idx| global_particle_densities[idx]),
        );
    }
}

/// Computes per particle densities into the workspace, also performs the required neighborhood search
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
    subdomain_grid: Option<&OwningSubdomainGrid<I, R>>,
    particle_positions: &[Vector3<R>],
    particle_densities: Option<&[R]>,
    parameters: &Parameters<R>,
    output_mesh: &'a mut TriMesh3d<R>,
) -> Result<(), ReconstructionError<I, R>> {
    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    let particle_densities = if let Some(particle_densities) = particle_densities {
        assert_eq!(particle_densities.len(), particle_positions.len());
        particle_densities
    } else {
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
    let mut density_map = new_map().into();
    density_map::generate_sparse_density_map(
        grid,
        subdomain_grid,
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
        subdomain_grid,
        &density_map,
        parameters.iso_surface_threshold,
        output_mesh,
    )?;

    Ok(())
}

/// Reconstruct a surface, appends triangulation to the given mesh
pub(crate) fn reconstruct_surface_patch<I: Index, R: Real>(
    workspace: &mut LocalReconstructionWorkspace<I, R>,
    subdomain_grid: &OwningSubdomainGrid<I, R>,
    particle_positions: &[Vector3<R>],
    particle_densities: Option<&[R]>,
    parameters: &Parameters<R>,
) -> Result<SurfacePatch<I, R>, ReconstructionError<I, R>> {
    profile!("reconstruct_surface_patch");

    let particle_rest_density = parameters.rest_density;
    let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
        * parameters.particle_radius.powi(3);
    let particle_rest_mass = particle_rest_volume * particle_rest_density;

    let particle_densities = if let Some(particle_densities) = particle_densities {
        assert_eq!(particle_densities.len(), particle_positions.len());
        particle_densities
    } else {
        compute_particle_densities_and_neighbors(
            subdomain_grid.global_grid(),
            particle_positions,
            parameters,
            &mut workspace.particle_neighbor_lists,
            &mut workspace.particle_densities,
        );
        workspace.particle_densities.as_slice()
    };

    // Create a new density map, reusing memory with the workspace is bad for cache efficiency
    // Alternatively, one could reuse memory with a custom caching allocator
    let mut density_map = new_map().into();
    density_map::generate_sparse_density_map(
        subdomain_grid.global_grid(),
        Some(subdomain_grid),
        particle_positions,
        particle_densities,
        None,
        particle_rest_mass,
        parameters.compact_support_radius,
        parameters.cube_size,
        parameters.enable_multi_threading,
        &mut density_map,
    )?;

    // Run marching cubes and get boundary data
    let patch = marching_cubes::triangulate_density_map_to_surface_patch::<I, R>(
        subdomain_grid,
        &density_map,
        parameters.iso_surface_threshold,
    )?;

    Ok(patch)
}