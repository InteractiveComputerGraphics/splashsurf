//! Workspace for reusing allocated memory between multiple surface reconstructions

use crate::Real;
use nalgebra::Vector3;
use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use thread_local::ThreadLocal;

/// Collection of all thread local workspaces used to reduce allocations on subsequent surface reconstructions
#[derive(Default)]
pub struct ReconstructionWorkspace<R: Real> {
    /// Temporary storage for storing a filtered set of the user provided particles
    filtered_particles: Vec<Vector3<R>>,
    local_workspaces: ThreadLocal<RefCell<LocalReconstructionWorkspace<R>>>,
}

impl<R: Real> ReconstructionWorkspace<R> {
    /// Returns a mutable reference to the global filtered particles vector
    pub(crate) fn filtered_particles_mut(&mut self) -> &mut Vec<Vector3<R>> {
        &mut self.filtered_particles
    }

    /// Returns a reference to a thread local workspace, initializes it with the given capacity if not already initialized
    pub(crate) fn get_local_with_capacity(
        &self,
        capacity: usize,
    ) -> &RefCell<LocalReconstructionWorkspace<R>> {
        self.local_workspaces
            .get_or(|| RefCell::new(LocalReconstructionWorkspace::with_capacity(capacity)))
    }
}

impl<R: Real> Clone for ReconstructionWorkspace<R> {
    /// Returns a new default workspace without any allocated memory
    fn clone(&self) -> Self {
        ReconstructionWorkspace::default()
    }
}

impl<R: Real> Debug for ReconstructionWorkspace<R> {
    /// Only print the name of type to the formatter
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReconstructionWorkspace").finish()
    }
}

/// Workspace used by [`reconstruct_surface_inplace`] internally to re-use allocated memory
pub(crate) struct LocalReconstructionWorkspace<R: Real> {
    /// Storage for per particle neighbor lists
    pub particle_neighbor_lists: Vec<Vec<usize>>,
    /// Storage for per particle densities
    pub particle_densities: Vec<R>,
}

impl<R: Real> Default for LocalReconstructionWorkspace<R> {
    /// Constructs a workspace without allocating additional memory
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Real> LocalReconstructionWorkspace<R> {
    /// Constructs a workspace without allocating additional memory
    pub(crate) fn new() -> Self {
        Self {
            particle_neighbor_lists: Default::default(),
            particle_densities: Default::default(),
        }
    }

    /// Constructs a workspace with capacity for the given number of particles
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            particle_neighbor_lists: Vec::with_capacity(capacity),
            particle_densities: Vec::with_capacity(capacity),
        }
    }
}
