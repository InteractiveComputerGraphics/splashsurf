use nalgebra::Vector3;
use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use thread_local::ThreadLocal;

use crate::mesh::TriMesh3d;
use crate::{new_map, DensityMap, Index, Real};

/// Collection of all thread local workspaces used to reduce allocations on subsequent surface reconstructions
#[derive(Default)]
pub struct ReconstructionWorkspace<I: Index, R: Real> {
    local_workspaces: ThreadLocal<RefCell<LocalReconstructionWorkspace<I, R>>>,
}

impl<I: Index, R: Real> ReconstructionWorkspace<I, R> {
    /// Returns a reference to a thread local workspace, initializes it with the given capacity if not already initialized
    pub fn get_local_with_capacity(
        &self,
        capacity: usize,
    ) -> &RefCell<LocalReconstructionWorkspace<I, R>> {
        self.local_workspaces
            .get_or(|| RefCell::new(LocalReconstructionWorkspace::with_capacity(capacity)))
    }

    /// Returns a mutable reference to the thread local workspaces
    pub fn local_workspaces_mut(
        &mut self,
    ) -> &mut ThreadLocal<RefCell<LocalReconstructionWorkspace<I, R>>> {
        &mut self.local_workspaces
    }
}

impl<I: Index, R: Real> Clone for ReconstructionWorkspace<I, R> {
    /// Returns a new default workspace without any allocated memory
    fn clone(&self) -> Self {
        ReconstructionWorkspace::default()
    }
}

impl<I: Index, R: Real> Debug for ReconstructionWorkspace<I, R> {
    /// Only print the name of type to the formatter
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReconstructionWorkspace").finish()
    }
}

/// Workspace used by [crate::reconstruct_single_surface] for the reconstruction of a single surface patch
pub struct LocalReconstructionWorkspace<I: Index, R: Real> {
    /// Storage for the particle positions (only used in octree based approach)
    pub particle_positions: Vec<Vector3<R>>,
    /// Storage for per particle neighbor lists
    pub particle_neighbor_lists: Vec<Vec<usize>>,
    /// Storage for per particle densities
    pub particle_densities: Vec<R>,
    /// Storage for the mesh
    pub mesh: TriMesh3d<R>,
    /// Storage for the density map
    pub density_map: DensityMap<I, R>,
}

impl<I: Index, R: Real> Default for LocalReconstructionWorkspace<I, R> {
    /// Constructs a workspace without allocating additional memory
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Index, R: Real> LocalReconstructionWorkspace<I, R> {
    /// Constructs a workspace without allocating additional memory
    pub fn new() -> Self {
        Self {
            particle_positions: Default::default(),
            particle_neighbor_lists: Default::default(),
            particle_densities: Default::default(),
            mesh: Default::default(),
            density_map: new_map().into(),
        }
    }

    /// Constructs a workspace with capacity for the given number of particles
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            particle_positions: Vec::with_capacity(capacity),
            particle_neighbor_lists: Vec::with_capacity(capacity),
            particle_densities: Vec::with_capacity(capacity),
            mesh: Default::default(),
            density_map: new_map().into(),
        }
    }
}
