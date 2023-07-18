//! Basic mesh types used by the library and implementation of VTK export
//!
//! This modules provides four basic types of meshes embedded in three dimensional spaces used
//! by the library:
//!  - [`TriMesh3d`]: triangle surface mesh in 3D space
//!  - [`MixedTriQuadMesh3d`]: surface mesh in 3D space with triangle and/or quadrilateral cells
//!  - [`PointCloud3d`]: points without connectivity in 3D space
//!  - [`HexMesh3d`]: mesh with volumetric hexahedral cells
//!
//! Furthermore, it provides the [`MeshWithData`] type that can be used to attach additional
//! attributes to vertices (e.g. normals) or cells (e.g. areas/aspect ratios) of the mesh.
//!
//! If the `vtk_extras` feature is enabled, this module also provides traits to convert these
//! meshes to [`vtkio`] data types for serialization to `VTK` files. For example:
//!  - [`IntoVtkUnstructuredGridPiece`] to convert basic meshes and meshes with attached attributes to the
//!  - [`IntoVtkDataSet`] for all meshes implementing [`IntoVtkUnstructuredGridPiece`] to directly save a mesh as a VTK file

use crate::{new_map, profile, Aabb3d, MapType, Real, RealConvert};
use bytemuck_derive::{Pod, Zeroable};
use nalgebra::{Unit, Vector3};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt::Debug;
use thread_local::ThreadLocal;
#[cfg(feature = "vtk_extras")]
use vtkio::model::{Attribute, UnstructuredGridPiece};

#[cfg(feature = "vtk_extras")]
pub use crate::mesh::vtk_helper::{IntoVtkDataSet, IntoVtkUnstructuredGridPiece};

/// A named attribute with data that can be attached to the vertices or cells of a mesh
#[derive(Clone, Debug)]
pub struct MeshAttribute<R: Real> {
    /// Name of the attribute
    pub name: String,
    /// Data of the attribute
    pub data: AttributeData<R>,
}

/// Data of an [`MeshAttribute`] that can be attached to the vertices or cells of a mesh
///
/// One value in the data-set corresponds is associated to a point or cell of the mesh.
#[derive(Clone, Debug)]
pub enum AttributeData<R: Real> {
    ScalarU64(Vec<u64>),
    ScalarReal(Vec<R>),
    Vector3Real(Vec<Vector3<R>>),
}

/// A triangle (surface) mesh in 3D
#[derive(Clone, Debug, Default)]
pub struct TriMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The triangles of the mesh identified by their vertex indices
    pub triangles: Vec<[usize; 3]>,
}

/// Cell type for [`MixedTriQuadMesh3d`]
#[derive(Clone, Debug)]
pub enum TriangleOrQuadCell {
    /// Vertex indices representing a triangle
    Tri([usize; 3]),
    /// Vertex indices representing a quadrilateral
    Quad([usize; 4]),
}

impl TriangleOrQuadCell {
    /// Returns the number of actual number of vertices of this cell (3 if triangle, 4 if quad)
    fn num_vertices(&self) -> usize {
        match self {
            TriangleOrQuadCell::Tri(_) => 3,
            TriangleOrQuadCell::Quad(_) => 4,
        }
    }

    /// Returns the slice of vertex indices of this cell
    fn vertices(&self) -> &[usize] {
        match self {
            TriangleOrQuadCell::Tri(v) => v,
            TriangleOrQuadCell::Quad(v) => v,
        }
    }
}

/// A surface mesh in 3D containing cells representing either triangles or quadrilaterals
#[derive(Clone, Debug, Default)]
pub struct MixedTriQuadMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// All triangle cells of the mesh
    pub cells: Vec<TriangleOrQuadCell>,
}

/// A hexahedral (volumetric) mesh in 3D
#[derive(Clone, Debug, Default)]
pub struct HexMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The hexahedral cells of the mesh identified by their vertex indices
    pub cells: Vec<[usize; 8]>,
}

/// A point cloud in 3D
#[derive(Clone, Debug, Default)]
pub struct PointCloud3d<R: Real> {
    /// Coordinates of all points in the point cloud
    points: Vec<Vector3<R>>,
    /// Indices of the points (for `CellConnectivity`)
    indices: Vec<usize>,
}

impl<R: Real> PointCloud3d<R> {
    pub fn new(points: impl Into<Vec<Vector3<R>>>) -> Self {
        let points = points.into();
        let indices = (0..points.len()).collect();
        Self { points, indices }
    }
}

/// Basic functionality that is provided by all meshes of the library
///
/// Meshes consist of vertices and cells. Cells identify their associated vertices using indices
/// into the mesh's slice of vertices.
pub trait Mesh3d<R: Real> {
    /// The cell connectivity type of the mesh
    type Cell: CellConnectivity;

    /// Returns a slice of all vertices of the mesh
    fn vertices(&self) -> &[Vector3<R>];
    /// Returns a slice of all cells of the mesh
    fn cells(&self) -> &[Self::Cell];

    /// Returns a mapping of all mesh vertices to the set of the cells they belong to
    fn vertex_cell_connectivity(&self) -> Vec<Vec<usize>> {
        profile!("vertex_cell_connectivity");
        let mut connectivity_map: Vec<Vec<usize>> = vec![Vec::new(); self.vertices().len()];
        for (cell_idx, cell) in self.cells().iter().enumerate() {
            cell.for_each_vertex(|v_i| {
                if !connectivity_map[v_i].contains(&cell_idx) {
                    connectivity_map[v_i].push(cell_idx);
                }
            })
        }

        connectivity_map
    }
}

/// Basic interface for mesh cells consisting of a collection of vertex indices
pub trait CellConnectivity {
    /// Returns the number of vertices per cell
    fn num_vertices(&self) -> usize {
        Self::expected_num_vertices()
    }
    /// Returns the expected number of vertices per cell (helpful for connectivities with a constant number of vertices to reserve storage)
    fn expected_num_vertices() -> usize;
    /// Calls the given closure with each vertex index that is part of this cell, stopping at the first error and returning that error
    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E>;
    /// Calls the given closure with each vertex index that is part of this cell
    fn for_each_vertex<F: FnMut(usize)>(&self, mut f: F) {
        self.try_for_each_vertex::<(), _>(move |i| {
            f(i);
            Ok(())
        })
        .unwrap();
    }
}

/// Cell type for [`TriMesh3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct TriangleCell(pub [usize; 3]);
/// Cell type for [`HexMesh3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct HexCell(pub [usize; 8]);
/// Cell type for [`PointCloud3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct PointCell(pub usize);

impl CellConnectivity for TriangleCell {
    fn expected_num_vertices() -> usize {
        3
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        self.0.iter().copied().try_for_each(f)
    }
}

impl CellConnectivity for TriangleOrQuadCell {
    fn expected_num_vertices() -> usize {
        4
    }

    fn num_vertices(&self) -> usize {
        return self.num_vertices();
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        self.vertices().iter().copied().try_for_each(f)
    }
}

impl CellConnectivity for HexCell {
    fn expected_num_vertices() -> usize {
        8
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        self.0.iter().copied().try_for_each(f)
    }
}

impl CellConnectivity for PointCell {
    fn expected_num_vertices() -> usize {
        1
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, mut f: F) -> Result<(), E> {
        f(self.0)
    }
}

impl<R: Real> Mesh3d<R> for TriMesh3d<R> {
    type Cell = TriangleCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn cells(&self) -> &[TriangleCell] {
        self.triangle_cells()
    }
}

impl<R: Real> Mesh3d<R> for MixedTriQuadMesh3d<R> {
    type Cell = TriangleOrQuadCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn cells(&self) -> &[TriangleOrQuadCell] {
        &self.cells
    }
}

impl<R: Real> Mesh3d<R> for HexMesh3d<R> {
    type Cell = HexCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn cells(&self) -> &[HexCell] {
        bytemuck::cast_slice::<[usize; 8], HexCell>(self.cells.as_slice())
    }
}

impl<R: Real> Mesh3d<R> for PointCloud3d<R> {
    type Cell = PointCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.points.as_slice()
    }

    fn cells(&self) -> &[PointCell] {
        bytemuck::cast_slice::<usize, PointCell>(self.indices.as_slice())
    }
}

impl<R: Real, MeshT: Mesh3d<R>> Mesh3d<R> for &MeshT {
    type Cell = MeshT::Cell;
    fn vertices(&self) -> &[Vector3<R>] {
        (*self).vertices()
    }
    fn cells(&self) -> &[MeshT::Cell] {
        (*self).cells()
    }
}

impl<'a, R: Real, MeshT: Mesh3d<R> + ToOwned> Mesh3d<R> for std::borrow::Cow<'a, MeshT> {
    type Cell = MeshT::Cell;
    fn vertices(&self) -> &[Vector3<R>] {
        (*self.as_ref()).vertices()
    }
    fn cells(&self) -> &[MeshT::Cell] {
        (*self.as_ref()).cells()
    }
}

impl TriangleCell {
    /// Returns an iterator over all edges of this triangle
    pub fn edges<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        (0..3).map(|i| (self.0[i], self.0[(i + 1) % 3]))
    }
}

impl<R: Real> TriMesh3d<R> {
    /// Returns a slice of all triangles of the mesh as `TriangleCell`s
    pub fn triangle_cells(&self) -> &[TriangleCell] {
        bytemuck::cast_slice::<[usize; 3], TriangleCell>(self.triangles.as_slice())
    }

    /// Clears the vertex and triangle storage, preserves allocated memory
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.triangles.clear();
    }

    /// Appends the other mesh to this mesh
    ///
    /// This operation appends the content of the other mesh's vertex and triangle storage tho this mesh.
    /// The vertex indices of the appended triangles are adjusted accordingly.
    /// The other mesh will be empty after this operation.
    pub fn append(&mut self, other: &mut TriMesh3d<R>) {
        let TriMesh3d {
            vertices: ref mut new_verts,
            triangles: ref mut new_tris,
        } = other;

        let vertex_offset = self.vertices.len();
        let tri_offset = self.triangles.len();

        self.vertices.append(new_verts);
        self.triangles.append(new_tris);

        // Update the vertex indices per triangle
        for tri in self.triangles.iter_mut().skip(tri_offset) {
            tri[0] += vertex_offset;
            tri[1] += vertex_offset;
            tri[2] += vertex_offset;
        }
    }

    /// Returns a new triangle mesh containing only the specified triangles and removes all unreferenced vertices
    pub fn keep_tris(&self, cell_indices: &[usize]) -> Self {
        // Each entry is true if this vertex should be kept, false otherwise
        let vertex_keep_table = {
            let mut table = vec![false; self.vertices.len()];
            for &cell_index in cell_indices {
                let cell_connectivity = &self.triangles[cell_index];

                for &vertex_index in cell_connectivity {
                    table[vertex_index] = true;
                }
            }
            table
        };

        let old_to_new_label_map = {
            let mut label_map = MapType::default();
            let mut next_label = 0;
            for (i, keep) in vertex_keep_table.iter().enumerate() {
                if *keep {
                    label_map.insert(i, next_label);
                    next_label += 1;
                }
            }
            label_map
        };

        let relabeled_cells: Vec<_> = cell_indices
            .iter()
            .map(|&i| self.triangles[i].clone())
            .map(|mut cell| {
                for index in &mut cell {
                    *index = *old_to_new_label_map
                        .get(index)
                        .expect("Index must be in map");
                }
                cell
            })
            .collect();

        let relabeled_vertices: Vec<_> = vertex_keep_table
            .iter()
            .enumerate()
            .filter_map(|(i, should_keep)| if *should_keep { Some(i) } else { None })
            .map(|index| self.vertices[index].clone())
            .collect();

        Self {
            vertices: relabeled_vertices,
            triangles: relabeled_cells,
        }
    }

    /// Removes all triangles from the mesh that are completely outside of the given AABB and clamps the remaining triangles to the boundary
    pub fn clamp_with_aabb(&mut self, aabb: &Aabb3d<R>) {
        // Find all triangles with at least one vertex inside of AABB
        let triangles_to_keep = self
            .triangles
            .par_iter()
            .enumerate()
            .filter(|(_, tri)| tri.iter().any(|&v| aabb.contains_point(&self.vertices[v])))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        // Remove all other triangles from mesh
        let new_mesh = self.keep_tris(&triangles_to_keep);
        // Replace current mesh
        let TriMesh3d {
            vertices,
            triangles,
        } = new_mesh;
        self.vertices = vertices;
        self.triangles = triangles;

        // Clamp remaining vertices to AABB
        self.vertices.par_iter_mut().for_each(|v| {
            let min = aabb.min();
            let max = aabb.max();
            v.x = v.x.clamp(min.x, max.x);
            v.y = v.y.clamp(min.y, max.y);
            v.z = v.z.clamp(min.z, max.z);
        })
    }

    /// Returns a mapping of all mesh vertices to the set of their connected neighbor vertices
    pub fn vertex_vertex_connectivity(&self) -> Vec<Vec<usize>> {
        profile!("vertex_vertex_connectivity");

        let mut connectivity_map: Vec<Vec<usize>> =
            vec![Vec::with_capacity(4); self.vertices().len()];
        for tri in &self.triangles {
            for &i in tri {
                for &j in tri {
                    if i != j && !connectivity_map[i].contains(&j) {
                        connectivity_map[i].push(j);
                    }
                }
            }
        }

        connectivity_map
    }

    /// Same as [`Self::vertex_normal_directions_inplace`] but assumes that the output is already zeroed
    fn vertex_normal_directions_inplace_assume_zeroed(&self, normal_directions: &mut [Vector3<R>]) {
        assert_eq!(normal_directions.len(), self.vertices.len());

        for tri_verts in self.triangles.iter() {
            let v0 = &self.vertices[tri_verts[0]];
            let v1 = &self.vertices[tri_verts[1]];
            let v2 = &self.vertices[tri_verts[2]];
            let normal = (v1 - v0).cross(&(v2 - v1));

            normal_directions[tri_verts[0]] += normal;
            normal_directions[tri_verts[1]] += normal;
            normal_directions[tri_verts[2]] += normal;
        }
    }

    /// Same as [`Self::vertex_normal_directions_inplace`] but assumes that the output is already zeroed
    fn par_vertex_normal_directions_inplace_assume_zeroed(
        &self,
        normal_directions: &mut [Vector3<R>],
    ) {
        assert_eq!(normal_directions.len(), self.vertices.len());

        let tl_normals = ThreadLocal::new();
        let init_tl_normals = || RefCell::new(vec![Vector3::zeros(); normal_directions.len()]);

        self.triangles.par_chunks(256).for_each(|tri_chunk| {
            let mut tl_normals_ref = tl_normals.get_or(init_tl_normals).borrow_mut();
            let tl_normals = &mut *tl_normals_ref;

            for tri_verts in tri_chunk.iter() {
                let v0 = &self.vertices[tri_verts[0]];
                let v1 = &self.vertices[tri_verts[1]];
                let v2 = &self.vertices[tri_verts[2]];
                let normal = (v1 - v0).cross(&(v2 - v1));

                tl_normals[tri_verts[0]] += normal;
                tl_normals[tri_verts[1]] += normal;
                tl_normals[tri_verts[2]] += normal;
            }
        });

        let tl_normals = tl_normals
            .into_iter()
            .map(|cell| cell.into_inner())
            .collect::<Vec<_>>();

        normal_directions
            .par_iter_mut()
            .with_min_len(256)
            .enumerate()
            .for_each(|(i, out_normal)| {
                for tl_normals in tl_normals.iter() {
                    *out_normal += tl_normals[i];
                }
            });
    }

    /// Computes the mesh's vertex normal directions inplace using an area weighted average of the adjacent triangle faces
    ///
    /// Note that this function only computes the normal directions, these vectors are **not normalized**!
    /// See [`Self::vertex_normals_inplace`] if actual normal vectors are needed.
    ///
    /// The method will panic if the length of the output slice is different from the number of vertices of the mesh.
    ///
    /// The method does not make any assumptions about the values in the output slice.
    pub fn vertex_normal_directions_inplace(&self, normal_directions: &mut [Vector3<R>]) {
        assert_eq!(normal_directions.len(), self.vertices.len());

        for normal in normal_directions.iter_mut() {
            normal.fill(R::zero());
        }

        self.vertex_normal_directions_inplace_assume_zeroed(normal_directions);
    }

    /// Computes the mesh's vertex normal directions using an area weighted average of the adjacent triangle faces
    ///
    /// Note that this function only computes the normal directions, these vectors are **not normalized**!
    /// See [`Self::vertex_normals`] if actual normal vectors are needed.
    pub fn vertex_normal_directions(&self) -> Vec<Vector3<R>> {
        let mut normal_directions = vec![Vector3::zeros(); self.vertices.len()];
        self.vertex_normal_directions_inplace_assume_zeroed(normal_directions.as_mut_slice());
        normal_directions
    }

    /// Same as [`Self::vertex_normals_inplace`] but assumes that the output is already zeroed
    fn vertex_normals_inplace_assume_zeroed<'a>(&self, normals: &'a mut [Unit<Vector3<R>>]) {
        assert_eq!(normals.len(), self.vertices.len());

        // First, compute the directions of the normals...
        {
            let normal_directions =
                bytemuck::cast_slice_mut::<Unit<Vector3<R>>, Vector3<R>>(normals);
            self.vertex_normal_directions_inplace_assume_zeroed(normal_directions);
        }

        // ...then actually normalize them.
        for normal in normals.iter_mut() {
            let norm = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
            *normal.as_mut_unchecked() /= norm;
            //normal.renormalize();
        }
    }

    /// Same as [`Self::par_vertex_normals_inplace`] but assumes that the output is already zeroed
    fn par_vertex_normals_inplace_assume_zeroed<'a>(&self, normals: &'a mut [Unit<Vector3<R>>]) {
        assert_eq!(normals.len(), self.vertices.len());

        // First, compute the directions of the normals...
        {
            let normal_directions =
                bytemuck::cast_slice_mut::<Unit<Vector3<R>>, Vector3<R>>(normals);
            self.par_vertex_normal_directions_inplace_assume_zeroed(normal_directions);
        }

        // ...then actually normalize them.
        normals.par_chunks_mut(256).for_each(|normal_chunk| {
            for normal in normal_chunk.iter_mut() {
                let norm = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
                *normal.as_mut_unchecked() /= norm;
                //normal.renormalize();
            }
        });
    }

    /// Computes the mesh's vertex normals inplace using an area weighted average of the adjacent triangle faces
    ///
    /// The method will panic if the length of the output slice is different from the number of vertices of the mesh.
    ///
    /// The method does not make any assumptions about the values in the output slice.
    pub fn vertex_normals_inplace(&self, normals: &mut [Unit<Vector3<R>>]) {
        assert_eq!(normals.len(), self.vertices.len());

        for normal in normals.iter_mut() {
            normal.as_mut_unchecked().fill(R::zero());
        }

        self.vertex_normals_inplace_assume_zeroed(normals);
    }

    /// Computes the mesh's vertex normals using an area weighted average of the adjacent triangle faces
    pub fn vertex_normals(&self) -> Vec<Unit<Vector3<R>>> {
        let mut normals = vec![Unit::new_unchecked(Vector3::zeros()); self.vertices.len()];
        self.vertex_normals_inplace_assume_zeroed(normals.as_mut_slice());
        normals
    }

    /// Computes the mesh's vertex normals inplace using an area weighted average of the adjacent triangle faces (parallelized version)
    ///
    /// The method will panic if the length of the output slice is different from the number of vertices of the mesh.
    ///
    /// The method does not make any assumptions about the values in the output slice.
    pub fn par_vertex_normals_inplace(&self, normals: &mut [Unit<Vector3<R>>]) {
        assert_eq!(normals.len(), self.vertices.len());

        normals.par_chunks_mut(256).for_each(|normal_chunk| {
            for normal in normal_chunk {
                normal.as_mut_unchecked().fill(R::zero())
            }
        });

        self.par_vertex_normals_inplace_assume_zeroed(normals);
    }

    /// Computes the mesh's vertex normals using an area weighted average of the adjacent triangle faces (parallelized version)
    pub fn par_vertex_normals(&self) -> Vec<Unit<Vector3<R>>> {
        let mut normals = vec![Unit::new_unchecked(Vector3::zeros()); self.vertices.len()];
        self.par_vertex_normals_inplace_assume_zeroed(normals.as_mut_slice());
        normals
    }

    /// Returns all boundary edges of the mesh
    ///
    /// Returns edges which are only connected to exactly one triangle, along with the connected triangle
    /// index and the local index of the edge within that triangle.
    ///
    /// Note that the output order is not necessarily deterministic due to the internal use of hashmaps.
    pub fn find_boundary_edges(&self) -> Vec<([usize; 2], usize, usize)> {
        let mut sorted_edges = Vec::new();
        let mut edge_info = Vec::new();

        // Local indices into the triangle connectivity to obtain all edges
        let tri_edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

        // For each triangle collect
        //  - each edge (with sorted vertices to use as unique key)
        //  - each edge with the index of the triangle and local index in the triangle
        for (tri_idx, tri_conn) in self.triangles.iter().enumerate() {
            for (local_idx, (v0, v1)) in tri_edges
                .iter()
                .copied()
                .map(|(i0, i1)| (tri_conn[i0], tri_conn[i1]))
                .enumerate()
            {
                // Sort the edge
                if v0 < v1 {
                    sorted_edges.push([v0, v1])
                } else {
                    sorted_edges.push([v1, v0])
                };

                edge_info.push(([v0, v1], tri_idx, local_idx));
            }
        }

        // Count the number of occurrences of "equivalent" edges (in the sense that they refer
        // to the same vertex indices).
        let mut edge_counts = new_map();
        for (edge_idx, edge) in sorted_edges.iter().copied().enumerate() {
            edge_counts
                .entry(edge)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((edge_idx, 1));
        }

        // Take only the faces which have a count of 1, which correspond to boundary faces
        edge_counts
            .into_iter()
            .map(|(_edge, value)| value)
            .filter(|&(_, count)| count == 1)
            .map(move |(edge_idx, _)| edge_info[edge_idx].clone())
            .collect()
    }
}

#[test]
fn test_find_boundary() {
    // TODO: Needs a test with a real mesh
    let mesh = TriMesh3d::<f64> {
        vertices: vec![
            Vector3::new_random(),
            Vector3::new_random(),
            Vector3::new_random(),
        ],
        triangles: vec![[0, 1, 2]],
    };

    let mut boundary = mesh.find_boundary_edges();
    boundary.sort_unstable();

    assert_eq!(
        boundary,
        vec![
            ([0usize, 1usize], 0, 0),
            ([1usize, 2usize], 0, 1),
            ([2usize, 0usize], 0, 2),
        ]
    );
}

/// Wrapper type for meshes with attached point or cell data
#[derive(Clone, Debug)]
pub struct MeshWithData<R: Real, MeshT: Mesh3d<R>> {
    /// The mesh geometry itself
    pub mesh: MeshT,
    /// Data attached to each vertex or point of the mesh
    pub point_attributes: Vec<MeshAttribute<R>>,
    /// Data attached to each cell of the mesh
    pub cell_attributes: Vec<MeshAttribute<R>>,
}

impl<R: Real, MeshT: Mesh3d<R>> Mesh3d<R> for MeshWithData<R, MeshT> {
    type Cell = MeshT::Cell;
    fn vertices(&self) -> &[Vector3<R>] {
        self.mesh.vertices()
    }
    fn cells(&self) -> &[MeshT::Cell] {
        self.mesh.cells()
    }
}

/// Returns an mesh data wrapper with a default mesh and without attached attributes
impl<R: Real, MeshT: Mesh3d<R> + Default> Default for MeshWithData<R, MeshT> {
    fn default() -> Self {
        Self::new(MeshT::default())
    }
}

impl<R: Real, MeshT: Mesh3d<R>> MeshWithData<R, MeshT> {
    /// Wraps the given mesh such that point and cell data can be attached
    pub fn new(mesh: MeshT) -> Self {
        Self {
            mesh,
            point_attributes: Vec::new(),
            cell_attributes: Vec::new(),
        }
    }

    /// Replaces the mesh but keeps the data
    pub fn with_mesh<NewMeshT: Mesh3d<R>>(self, new_mesh: NewMeshT) -> MeshWithData<R, NewMeshT> {
        if !self.point_attributes.is_empty() {
            assert_eq!(
                self.mesh.vertices().len(),
                new_mesh.vertices().len(),
                "number of vertices should match if there are point attributes"
            );
        }
        if !self.cell_attributes.is_empty() {
            assert_eq!(
                self.mesh.cells().len(),
                new_mesh.cells().len(),
                "number of cells should match if there are cell attributes"
            )
        }
        MeshWithData {
            mesh: new_mesh,
            point_attributes: self.point_attributes,
            cell_attributes: self.cell_attributes,
        }
    }

    /// Attaches an attribute to the points of the mesh, panics if the length of the data does not match the mesh's number of points
    pub fn with_point_data(mut self, point_attribute: impl Into<MeshAttribute<R>>) -> Self {
        let point_attribute = point_attribute.into();
        assert_eq!(point_attribute.data.len(), self.mesh.vertices().len());
        self.point_attributes.push(point_attribute);
        self
    }

    /// Attaches an attribute to the cells of the mesh, panics if the length of the data does not match the mesh's number of cells
    pub fn with_cell_data(mut self, cell_attribute: impl Into<MeshAttribute<R>>) -> Self {
        let cell_attribute = cell_attribute.into();
        assert_eq!(cell_attribute.data.len(), self.mesh.cells().len());
        self.cell_attributes.push(cell_attribute);
        self
    }
}

impl<R: Real> MeshAttribute<R> {
    /// Creates a new named mesh attribute with the given data
    pub fn new<S: Into<String>>(name: S, data: impl Into<AttributeData<R>>) -> Self {
        Self {
            name: name.into(),
            data: data.into(),
        }
    }

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`](crate::Real) trait
    pub fn new_real_scalar<S: Into<String>>(name: S, data: impl Into<Vec<R>>) -> Self {
        Self {
            name: name.into(),
            data: AttributeData::ScalarReal(data.into()),
        }
    }

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`](crate::Real) trait
    pub fn new_real_vector3<S: Into<String>>(name: S, data: impl Into<Vec<Vector3<R>>>) -> Self {
        Self {
            name: name.into(),
            data: AttributeData::Vector3Real(data.into()),
        }
    }

    /// Converts the mesh attribute to a [`vtkio::model::Attribute`])
    #[cfg(feature = "vtk_extras")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    fn to_vtk_attribute(&self) -> Attribute {
        match &self.data {
            AttributeData::ScalarU64(u64_vec) => {
                Attribute::scalars(&self.name, 1).with_data(u64_vec.clone())
            }
            AttributeData::ScalarReal(real_vec) => {
                Attribute::scalars(&self.name, 1).with_data(real_vec.clone())
            }
            AttributeData::Vector3Real(vec3r_vec) => Attribute::scalars(&self.name, 3)
                .with_data(vec3r_vec.iter().flatten().copied().collect::<Vec<R>>()),
        }
    }
}

impl<R: Real> AttributeData<R> {
    /// Returns the number of entries in the data set
    fn len(&self) -> usize {
        match self {
            AttributeData::ScalarU64(v) => v.len(),
            AttributeData::ScalarReal(v) => v.len(),
            AttributeData::Vector3Real(v) => v.len(),
        }
    }
}

impl<R: Real, V: Into<Vec<u64>>> From<V> for AttributeData<R> {
    fn from(data: V) -> Self {
        Self::ScalarU64(data.into())
    }
}

#[cfg(feature = "vtk_extras")]
impl<R, MeshT> MeshWithData<R, MeshT>
where
    R: Real,
    MeshT: Mesh3d<R>,
    for<'a> &'a MeshT: IntoVtkUnstructuredGridPiece,
{
    /// Creates a [`vtkio::model::UnstructuredGridPiece`] representing this mesh including its attached [`MeshAttribute`]s
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    fn unstructured_grid(&self) -> UnstructuredGridPiece {
        let mut grid_piece: UnstructuredGridPiece = (&self.mesh).into_unstructured_grid();
        for point_attribute in &self.point_attributes {
            grid_piece
                .data
                .point
                .push(point_attribute.to_vtk_attribute())
        }
        for cell_attribute in &self.cell_attributes {
            grid_piece.data.cell.push(cell_attribute.to_vtk_attribute())
        }
        grid_piece
    }
}

macro_rules! impl_into_vtk {
    ($name:tt) => {
        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for $name<R> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(&self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for &$name<R> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<'a, R: Real> IntoVtkUnstructuredGridPiece for std::borrow::Cow<'a, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(&self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<'a, R: Real> IntoVtkUnstructuredGridPiece for &std::borrow::Cow<'a, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(&self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for &MeshWithData<R, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                self.unstructured_grid()
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for MeshWithData<R, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                self.unstructured_grid()
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
        impl<'a, R: Real> IntoVtkUnstructuredGridPiece
            for &MeshWithData<R, std::borrow::Cow<'a, $name<R>>>
        {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                self.unstructured_grid()
            }
        }
    };
}

impl_into_vtk!(TriMesh3d);
impl_into_vtk!(MixedTriQuadMesh3d);
impl_into_vtk!(HexMesh3d);
impl_into_vtk!(PointCloud3d);

/// Trait implementations to convert meshes into types supported by [`vtkio`]
#[cfg(feature = "vtk_extras")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
pub mod vtk_helper {
    use vtkio::model::{
        Attributes, CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers,
    };
    use vtkio::IOBuffer;

    use super::{
        CellConnectivity, HexCell, Mesh3d, PointCell, Real, TriangleCell, TriangleOrQuadCell,
    };

    /// Trait that can be implemented by mesh cells to return the corresponding [`vtkio::model::CellType`]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub trait HasVtkCellType {
        /// Returns the corresponding [`vtkio::model::CellType`] of the cell
        fn vtk_cell_type(&self) -> CellType;
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for TriangleCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Triangle
        }
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for TriangleOrQuadCell {
        fn vtk_cell_type(&self) -> CellType {
            match self {
                TriangleOrQuadCell::Tri(_) => CellType::Triangle,
                TriangleOrQuadCell::Quad(_) => CellType::Quad,
            }
        }
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for HexCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Hexahedron
        }
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for PointCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Vertex
        }
    }

    /// Conversion of meshes into a [`vtkio::model::UnstructuredGridPiece`]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub trait IntoVtkUnstructuredGridPiece {
        fn into_unstructured_grid(self) -> UnstructuredGridPiece;
    }

    /// Direct conversion of meshes into a full [`vtkio::model::DataSet`]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub trait IntoVtkDataSet {
        fn into_dataset(self) -> DataSet;
    }

    impl IntoVtkUnstructuredGridPiece for UnstructuredGridPiece {
        fn into_unstructured_grid(self) -> UnstructuredGridPiece {
            self
        }
    }

    impl<T: IntoVtkUnstructuredGridPiece> IntoVtkDataSet for T {
        fn into_dataset(self) -> DataSet {
            DataSet::inline(self.into_unstructured_grid())
        }
    }

    impl IntoVtkDataSet for DataSet {
        fn into_dataset(self) -> DataSet {
            self
        }
    }

    /// Converts any supported mesh to a [`vtkio::model::UnstructuredGridPiece`]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub fn mesh_to_unstructured_grid<'a, R, MeshT>(mesh: &'a MeshT) -> UnstructuredGridPiece
    where
        R: Real,
        MeshT: Mesh3d<R> + 'a,
        <MeshT as Mesh3d<R>>::Cell: HasVtkCellType,
    {
        let points = {
            let mut points: Vec<R> = Vec::with_capacity(mesh.vertices().len() * 3);
            points.extend(mesh.vertices().iter().map(|p| p.as_slice()).flatten());
            points
        };

        let vertices = {
            let mut vertices =
                Vec::with_capacity(mesh.cells().len() * (MeshT::Cell::expected_num_vertices() + 1));
            for cell in mesh.cells().iter() {
                vertices.push(cell.num_vertices() as u32);
                cell.for_each_vertex(|v| vertices.push(v as u32));
            }
            vertices
        };

        let mut cell_types = Vec::with_capacity(mesh.cells().len());
        cell_types.extend(mesh.cells().iter().map(|c| c.vtk_cell_type()));

        new_unstructured_grid_piece(points, vertices, cell_types)
    }

    fn new_unstructured_grid_piece<B: Into<IOBuffer>>(
        points: B,
        vertices: Vec<u32>,
        cell_types: Vec<CellType>,
    ) -> UnstructuredGridPiece {
        UnstructuredGridPiece {
            points: points.into(),
            cells: Cells {
                cell_verts: VertexNumbers::Legacy {
                    num_cells: cell_types.len() as u32,
                    vertices,
                },
                types: cell_types,
            },
            data: Attributes::new(),
        }
    }
}
