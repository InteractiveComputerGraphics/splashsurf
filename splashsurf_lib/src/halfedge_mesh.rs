//! Basic implementation of a half-edge based triangle mesh
//!
//! See [`HalfEdgeTriMesh`] for more information.

use crate::mesh::{Mesh3d, TriMesh3d, TriMesh3dExt};
use crate::{Real, SetType, profile};
use nalgebra::Vector3;
use rayon::prelude::*;
use thiserror::Error as ThisError;

impl<R: Real> TriMesh3dExt<R> for HalfEdgeTriMesh<R> {
    fn tri_vertices(&self) -> &[Vector3<R>] {
        &self.vertices
    }
}

/// A half-edge in a [`HalfEdgeTriMesh`]
#[derive(Copy, Clone, Debug, Default)]
pub struct HalfEdge {
    /// Unique global index of this half-edge in the mesh
    pub idx: usize,
    /// Vertex this half-edge points to
    pub to: usize,
    /// Enclosed face of this half-edge loop (or `None` if boundary)
    pub face: Option<usize>,
    /// The next half-edge along the half-edge loop (or `None` if boundary)
    pub next: Option<usize>,
    /// Index of the half-edge going into the opposite direction
    pub opposite: usize,
}

impl HalfEdge {
    /// Returns whether the given half-edge is a boundary edge
    pub fn is_boundary(&self) -> bool {
        self.face.is_none()
    }
}

/// A half-edge based triangle mesh data structure
///
/// The main purpose of this data structure is to provide methods to perform consistent collapses of
/// half-edges for decimation procedures.
///
/// As [`splashsurf_lib`](crate) is focused on closed meshes, handling of holes is not specifically tested.
/// In particular, it is not directly possible to walk along a mesh boundary using the half-edges of
/// this implementation.
///
/// A [`HalfEdgeTriMesh`] can be easily constructed from a [`TriMesh3d`] using a [`From`](HalfEdgeTriMesh::from::<TriMesh3d>) implementation.
///
/// Note that affected vertex/face/half-edge indices become "invalid" after half-edge collapse is performed.
/// The corresponding data still exist (i.e. they can be retrieved from the mesh) but following these
/// indices amounts to following outdated connectivity.
/// Therefore, it should be checked if an index was marked as removed after a collapse using the
/// [`is_valid_vertex`](HalfEdgeTriMesh::is_valid_vertex)/[`is_valid_triangle`](HalfEdgeTriMesh::is_valid_triangle)/[`is_valid_half_edge`](HalfEdgeTriMesh::is_valid_half_edge)
/// methods.
#[derive(Clone, Debug, Default)]
pub struct HalfEdgeTriMesh<R: Real> {
    /// All vertices in the mesh
    pub vertices: Vec<Vector3<R>>,
    /// All triangles in the mesh
    pub triangles: Vec<[usize; 3]>,
    /// All half-edges in the mesh
    pub half_edges: Vec<HalfEdge>,
    /// Connectivity map of all vertices to their connected neighbors
    vertex_half_edge_map: Vec<Vec<usize>>,
    /// Set of all vertices marked for removal
    removed_vertices: SetType<usize>,
    /// Set of all triangles marked for removal
    removed_triangles: SetType<usize>,
    /// Set of all half edges marked for removal
    removed_half_edges: SetType<usize>,
}

/// Error indicating why a specific half-edge collapse is illegal
#[derive(Copy, Clone, Debug, Eq, PartialEq, ThisError)]
pub enum IllegalHalfEdgeCollapse {
    /// Trying to collapse an edge with boundary vertices at both ends
    #[error("trying to collapse an edge with boundary vertices at both ends")]
    BoundaryCollapse,
    /// Trying to collapse an edge with vertices that share incident vertices other than the vertices directly opposite to the edge
    #[error(
        "trying to collapse an edge with vertices that share incident vertices other than the vertices directly opposite to the edge"
    )]
    IntersectionOfOneRing,
    /// Trying to collapse an edge without faces
    #[error("trying to collapse an edge without faces")]
    FacelessEdge,
}

impl<R: Real> HalfEdgeTriMesh<R> {
    /// Converts this mesh into a simple triangle mesh and a vertex-vertex connectivity map
    pub fn into_parts(mut self, keep_vertices: bool) -> (TriMesh3d<R>, Vec<Vec<usize>>) {
        Self::compute_vertex_vertex_connectivity(&mut self.vertex_half_edge_map, &self.half_edges);
        self.garbage_collection_for_trimesh(keep_vertices);
        let mesh = TriMesh3d {
            vertices: self.vertices,
            triangles: self.triangles,
        };
        (mesh, self.vertex_half_edge_map)
    }

    /// Returns the valence of a vertex (size of its one-ring)
    pub fn vertex_one_ring_len(&self, vertex: usize) -> usize {
        self.vertex_half_edge_map[vertex].len()
    }

    /// Returns the index of the `i`-th vertex from the one-ring of the given vertex
    pub fn vertex_one_ring_ith(&self, vertex: usize, i: usize) -> usize {
        self.half_edges[self.vertex_half_edge_map[vertex][i]].to
    }

    /// Iterator over the one-ring vertex neighbors of the given vertex
    pub fn vertex_one_ring(&self, vertex: usize) -> impl Iterator<Item = usize> + '_ {
        self.vertex_half_edge_map[vertex]
            .iter()
            .copied()
            .map(|he_i| self.half_edges[he_i].to)
    }

    /// Iterator over the outgoing half-edges of the given vertex
    pub fn outgoing_half_edges(&self, vertex: usize) -> impl Iterator<Item = HalfEdge> + '_ {
        self.vertex_half_edge_map[vertex]
            .iter()
            .copied()
            .map(|he_i| self.half_edges[he_i])
    }

    /// Iterator over all incident faces of the given vertex
    pub fn incident_faces(&self, vertex: usize) -> impl Iterator<Item = usize> + '_ {
        self.outgoing_half_edges(vertex).filter_map(|he| he.face)
    }

    /// Returns the half-edge between the "from" and "to" vertex if it exists in the mesh
    pub fn half_edge(&self, from: usize, to: usize) -> Option<HalfEdge> {
        let from_edges = self
            .vertex_half_edge_map
            .get(from)
            .expect("vertex must be part of the mesh");
        for &he_idx in from_edges {
            let he = &self.half_edges[he_idx];
            if he.to == to {
                return Some(*he);
            }
        }

        None
    }

    /// Returns whether the given half-edge or its opposite half-edge is a boundary edge
    pub fn is_boundary_edge(&self, half_edge: HalfEdge) -> bool {
        half_edge.is_boundary() || self.opposite(half_edge).is_boundary()
    }

    /// Returns whether the given vertex is a boundary vertex
    pub fn is_boundary_vertex(&self, vert_idx: usize) -> bool {
        let hes = self
            .vertex_half_edge_map
            .get(vert_idx)
            .expect("vertex must be part of the mesh");
        hes.iter()
            .copied()
            .any(|he_idx| self.half_edges[he_idx].is_boundary())
    }

    /// Returns whether the given triangle is valid (i.e. not marked as removed)
    pub fn is_valid_triangle(&self, triangle_idx: usize) -> bool {
        !self.removed_triangles.contains(&triangle_idx)
    }

    /// Returns whether the given vertex is valid (i.e. not marked as removed)
    pub fn is_valid_vertex(&self, vertex_idx: usize) -> bool {
        !self.removed_vertices.contains(&vertex_idx)
    }

    /// Returns whether the given vertex is valid (i.e. not marked as removed)
    pub fn is_valid_half_edge(&self, half_edge_idx: usize) -> bool {
        !self.removed_half_edges.contains(&half_edge_idx)
    }

    /// Returns the next half-edge in the loop of the given half-edge, panics if there is none
    pub fn next(&self, half_edge: HalfEdge) -> HalfEdge {
        self.half_edges[half_edge
            .next
            .expect("half edge must have a next reference")]
    }

    /// Returns the next half-edge in the loop of the given half-edge if it exists
    pub fn try_next(&self, half_edge: HalfEdge) -> Option<HalfEdge> {
        half_edge.next.map(|n| self.half_edges[n])
    }

    /// Returns the opposite half-edge of the given half-edge
    pub fn opposite(&self, half_edge: HalfEdge) -> HalfEdge {
        self.half_edges[half_edge.opposite]
    }

    /// Returns a mutable reference to the opposite half-edge of the given half-edge
    pub fn opposite_mut(&mut self, half_edge: usize) -> &mut HalfEdge {
        let opp_idx = self.half_edges[half_edge].opposite;
        &mut self.half_edges[opp_idx]
    }

    /// Checks if the collapse of the given half-edge is topologically legal
    pub fn is_collapse_ok(&self, half_edge: HalfEdge) -> Result<(), IllegalHalfEdgeCollapse> {
        // Based on PMP library:
        // https://github.com/pmp-library/pmp-library/blob/86099e4e274c310d23e8c46c4829f881242814d3/src/pmp/SurfaceMesh.cpp#L755

        let v0v1 = half_edge;
        let v1v0 = self.opposite(v0v1);

        let v0 = v1v0.to; // From vertex
        let v1 = v0v1.to; // To vertex

        // Checks if edges to opposite vertex of half-edge are boundary edges and returns opposite vertex
        let check_opposite_vertex =
            |he: HalfEdge| -> Result<Option<usize>, IllegalHalfEdgeCollapse> {
                if !he.is_boundary() {
                    let h1 = self.next(he);
                    let h2 = self.next(h1);

                    if self.opposite(h1).is_boundary() && self.opposite(h2).is_boundary() {
                        return Err(IllegalHalfEdgeCollapse::BoundaryCollapse);
                    }

                    // Return the opposite vertex
                    Ok(Some(h1.to))
                } else {
                    Ok(None)
                }
            };

        // Notation:
        //   v_pos -> vertex opposite to the half-edge to collapse (v0v1)
        //   v_neg -> vertex opposite to the opposite half-edge to collapse (v1v0)
        let v_pos = check_opposite_vertex(v0v1)?;
        let v_neg = check_opposite_vertex(v1v0)?;

        if v_pos.is_none() || v_neg.is_none() {
            return Err(IllegalHalfEdgeCollapse::FacelessEdge);
        }

        // Test intersection of the one-rings of v0 and v1
        for &he in &self.vertex_half_edge_map[v0] {
            let he = &self.half_edges[he];
            let vv = he.to;
            if vv != v1
                && Some(vv) != v_pos
                && Some(vv) != v_neg
                && self.half_edge(vv, v1).is_some()
            {
                return Err(IllegalHalfEdgeCollapse::IntersectionOfOneRing);
            }
        }

        Ok(())
    }

    pub fn try_half_edge_collapse(
        &mut self,
        half_edge: HalfEdge,
    ) -> Result<(), IllegalHalfEdgeCollapse> {
        self.is_collapse_ok(half_edge)?;

        self.half_edge_collapse(half_edge);
        Ok(())
    }

    pub fn half_edge_collapse(&mut self, half_edge: HalfEdge) {
        let he = half_edge;
        let he_o = self.opposite(he);

        let v_from = he_o.to;
        let v_to = he.to;

        // TODO: Support collapse of boundary edges

        let he_n = self
            .try_next(he)
            .expect("encountered boundary (missing opposite vertex)");
        let he_nn = self.next(he_n);

        let he_on = self
            .try_next(he_o)
            .expect("encountered boundary (missing opposite vertex)");
        let he_onn = self.next(he_on);

        // Vertices opposite to the edge to collapse
        let v_pos = he_n.to;
        let v_neg = he_on.to;

        let conn_from = self.vertex_half_edge_map[v_from].clone();
        let mut conn_to = self.vertex_half_edge_map[v_to].clone();

        // Mark faces and vertex for removal
        {
            he.face.map(|f| self.removed_triangles.insert(f));
            he_o.face.map(|f| self.removed_triangles.insert(f));
            self.removed_vertices.insert(v_from);
        }

        // Collect half-edges to delete (inside collapsed triangles)
        self.removed_half_edges.insert(he.idx);
        self.removed_half_edges.insert(he_n.idx);
        self.removed_half_edges.insert(he_nn.idx);
        self.removed_half_edges.insert(he_o.idx);
        self.removed_half_edges.insert(he_on.idx);
        self.removed_half_edges.insert(he_onn.idx);

        // Handle case of two opposite but coincident faces
        if v_pos == v_neg {
            // Faces were already marked for removal above
            // Half-edges were already marked for removal above

            // Mark other vertices of triangles for removal
            self.removed_vertices.insert(v_to);
            self.removed_vertices.insert(v_pos);
            // Clear all connectivity of removed vertices
            self.vertex_half_edge_map[v_from].clear();
            self.vertex_half_edge_map[v_to].clear();
            self.vertex_half_edge_map[v_pos].clear();

            return;
        }

        // Update the faces referencing the removed vertex
        for &he_idx in &conn_from {
            if let Some(f) = self.half_edges[he_idx].face {
                self.triangles[f].iter_mut().for_each(|i| {
                    if *i == v_from {
                        *i = v_to;
                    }
                })
            }
        }

        // Update the half-edges around the collapsed triangles (they become opposites)
        {
            let he_no = self.opposite(he_n);
            let he_nno = self.opposite(he_nn);

            self.half_edges[he_no.idx].opposite = he_nno.idx;
            self.half_edges[he_nno.idx].opposite = he_no.idx;

            let he_ono = self.opposite(he_on);
            let he_onno = self.opposite(he_onn);

            self.half_edges[he_ono.idx].opposite = he_onno.idx;
            self.half_edges[he_onno.idx].opposite = he_ono.idx;
        }

        // Remove collapsed half-edges from connectivity of target vertex
        conn_to.retain(|he_i| *he_i != he_n.idx && *he_i != he_o.idx);
        // Transfer half-edge connectivity from collapsed to target vertex
        for &he_i in &conn_from {
            if he_i != he.idx && he_i != he_on.idx {
                conn_to.push(he_i);
            }
        }
        // Update the targets of half-edges pointing to collapsed vertex
        for &he_i in &conn_to {
            let opp = self.opposite_mut(he_i);
            if opp.to == v_from {
                opp.to = v_to;
            }
        }
        self.vertex_half_edge_map[v_to] = conn_to;
        // Clear all connectivity of the collapsed vertex
        self.vertex_half_edge_map[v_from].clear();

        // Remove collapsed half-edges from connectivity of vertices opposite to collapsed edge
        self.vertex_half_edge_map[v_pos].retain(|he_i| *he_i != he_nn.idx);
        self.vertex_half_edge_map[v_neg].retain(|he_i| *he_i != he_onn.idx);
    }

    /// Computes the largest angle in radians by which a face normals rotates of triangles affect by the given half edge collapse, assumes that the given half edge is valid
    pub fn half_edge_collapse_max_normal_change(&self, half_edge: HalfEdge) -> R {
        let he = half_edge;
        let he_o = self.opposite(he);

        let v_to = he.to;
        let v_from = he_o.to;

        let mut max_normal_change_angle = R::zero();
        for face in self.incident_faces(v_from) {
            let tri_old = self.triangles[face];
            let tri_new = tri_old.map(|i| if i == v_from { v_to } else { i });

            // Skip faces that will be collapsed
            if tri_new.iter().copied().filter(|i| *i == v_to).count() > 1 {
                continue;
            }

            let new_area = self.tri_area_ijk::<R>(&tri_new);
            if new_area > R::default_epsilon() {
                let old_normal = self.tri_normal_ijk::<R>(&tri_old);
                let new_normal = self.tri_normal_ijk::<R>(&tri_new);

                let alpha = old_normal.dot(&new_normal).acos();
                max_normal_change_angle = max_normal_change_angle.max(alpha);
            }
        }

        max_normal_change_angle
    }

    /// Computes the largest ratio (`new/old`) of triangle aspect ratio of triangles affect by the given half edge collapse, assumes that the given half edge is valid
    pub fn half_edge_collapse_max_aspect_ratio_change(&self, half_edge: HalfEdge) -> R {
        let he = half_edge;
        let he_o = self.opposite(he);

        let v_to = he.to;
        let v_from = he_o.to;

        let mut max_aspect_ratio_change = R::one();
        for face in self.incident_faces(v_from) {
            let tri_old = self.triangles[face];
            let tri_new = tri_old.map(|i| if i == v_from { v_to } else { i });

            // Skip faces that will be collapsed
            if tri_new.iter().copied().filter(|i| *i == v_to).count() > 1 {
                continue;
            }

            let old_aspect_ratio = self.tri_aspect_ratio::<R>(&tri_old);
            let new_aspect_ratio = self.tri_aspect_ratio::<R>(&tri_new);

            max_aspect_ratio_change =
                max_aspect_ratio_change.max(new_aspect_ratio / old_aspect_ratio)
        }

        max_aspect_ratio_change
    }

    fn compute_vertex_vertex_connectivity(
        vertex_half_edge_map: &mut [Vec<usize>],
        half_edges: &[HalfEdge],
    ) {
        vertex_half_edge_map.par_iter_mut().for_each(|hes| {
            for he in hes {
                *he = half_edges[*he].to;
            }
        });
    }

    /// Clean mesh of deleted elements (vertices, faces) for conversion into a triangle mesh (does not clean-up half-edges and their references)
    fn garbage_collection_for_trimesh(&mut self, keep_vertices: bool) {
        // Filter and update triangles
        let filtered_triangles = self
            .triangles
            .par_iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| !self.removed_triangles.contains(i))
            .map(|(_, tri)| tri)
            .collect();
        self.triangles = filtered_triangles;

        if !keep_vertices {
            let mut new_vertex_indices = vec![0; self.vertices.len()];
            let mut filtered_vertices =
                Vec::with_capacity(self.vertices.len() - self.removed_vertices.len());
            let mut filtered_vertex_map =
                Vec::with_capacity(self.vertices.len() - self.removed_vertices.len());

            // Filter vertices and assign new indices
            let mut index_counter = 0;
            for (i, new_index) in new_vertex_indices.iter_mut().enumerate() {
                if !self.removed_vertices.contains(&i) {
                    *new_index = index_counter;
                    index_counter += 1;
                    filtered_vertices.push(self.vertices[i]);
                    filtered_vertex_map.push(std::mem::take(&mut self.vertex_half_edge_map[i]));
                }
            }

            // Update vertex maps
            filtered_vertex_map.iter_mut().for_each(|m| {
                for v in m {
                    *v = new_vertex_indices[*v];
                }
            });

            self.vertices = filtered_vertices;
            self.vertex_half_edge_map = filtered_vertex_map;

            // Update triangles
            self.triangles.par_iter_mut().for_each(|tri| {
                tri[0] = new_vertex_indices[tri[0]];
                tri[1] = new_vertex_indices[tri[1]];
                tri[2] = new_vertex_indices[tri[2]];
            });

            self.removed_vertices.clear();
        }

        self.removed_triangles.clear();
    }
}

impl<R: Real> From<TriMesh3d<R>> for HalfEdgeTriMesh<R> {
    fn from(mesh: TriMesh3d<R>) -> Self {
        profile!("construct_half_edge_mesh");

        let mut he_mesh = HalfEdgeTriMesh::default();
        he_mesh.vertex_half_edge_map = vec![Vec::with_capacity(5); mesh.vertices().len()];
        he_mesh.vertices = mesh.vertices;

        for (tri_idx, tri) in mesh.triangles.iter().copied().enumerate() {
            // Storage for inner half-edge indices
            let mut tri_hes = [0, 0, 0];

            // Loop over inner-half edges
            for i in 0..3 {
                let from = tri[i];
                let to = tri[(i + 1) % 3];

                // Check if half-edge exists already
                if let Some(he) = he_mesh.half_edge(from, to) {
                    // Store the current half-edge for later use
                    tri_hes[i] = he.idx;
                    // Update the face of the half-edge
                    he_mesh.half_edges[he.idx].face = Some(tri_idx);
                } else {
                    let he_idx = he_mesh.half_edges.len();
                    // Inner (counter-clockwise) edge
                    let he_ccw = HalfEdge {
                        idx: he_idx,
                        to,
                        face: Some(tri_idx),
                        next: None,
                        opposite: he_idx + 1,
                    };
                    // Outer (counter-clockwise) edge
                    let he_cw = HalfEdge {
                        idx: he_idx + 1,
                        to: from,
                        face: None,
                        next: None,
                        opposite: he_idx,
                    };
                    tri_hes[i] = he_idx;

                    // Store half-edges
                    he_mesh.half_edges.push(he_ccw);
                    he_mesh.half_edges.push(he_cw);
                    // Update vertex connectivity
                    he_mesh.vertex_half_edge_map[from].push(he_idx);
                    he_mesh.vertex_half_edge_map[to].push(he_idx + 1);
                }
            }

            // Update inner half-edge next pointers
            for i in 0..3 {
                let j = (i + 1) % 3;
                he_mesh.half_edges[tri_hes[i]].next = Some(tri_hes[j]);
            }
        }

        he_mesh.triangles = mesh.triangles;
        he_mesh
    }
}

#[test]
fn test_half_edge_mesh() {
    let tri_mesh = TriMesh3d::<f32> {
        vertices: vec![
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 0.0),
            Vector3::new(2.0, 1.0, 0.0),
        ],
        triangles: vec![[0, 1, 2], [1, 3, 2]],
    };

    let he_mesh = crate::halfedge_mesh::HalfEdgeTriMesh::from(tri_mesh.clone());

    let vert_map = tri_mesh.vertex_vertex_connectivity();
    let (_new_tri_mesh, new_vert_map) = he_mesh.into_parts(true);

    let mut a = vert_map;
    let mut b = new_vert_map;

    a.par_iter_mut().for_each(|l| l.sort_unstable());
    b.par_iter_mut().for_each(|l| l.sort_unstable());

    for (la, lb) in a.iter().zip(b.iter()) {
        assert_eq!(la, lb);
    }
}
