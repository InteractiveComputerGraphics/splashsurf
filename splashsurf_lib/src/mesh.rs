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

use crate::{Aabb3d, MapType, Real, RealConvert, new_map, profile};
use bytemuck_derive::{Pod, Zeroable};
use nalgebra::{Unit, Vector3};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::Debug;
use thread_local::ThreadLocal;
#[cfg(feature = "vtk_extras")]
use vtkio::model::{Attribute, UnstructuredGridPiece};

#[cfg(feature = "vtk_extras")]
pub use crate::mesh::vtk_helper::{IntoVtkDataSet, IntoVtkUnstructuredGridPiece};

/// Computes the unsigned area of the given triangle
pub fn tri_area<RIn: Real, RComp: Real>(
    a: &Vector3<RIn>,
    b: &Vector3<RIn>,
    c: &Vector3<RIn>,
) -> RComp {
    let a = a.convert::<RComp>();
    let b = b.convert::<RComp>();
    let c = c.convert::<RComp>();
    ((b - a).cross(&(c - a)))
        .norm()
        .unscale(RComp::one() + RComp::one())
}

/// Computes the face normal of the given triangle
pub fn tri_normal<RIn: Real, RComp: Real>(
    a: &Vector3<RIn>,
    b: &Vector3<RIn>,
    c: &Vector3<RIn>,
) -> Vector3<RComp> {
    let a = a.convert::<RComp>();
    let b = b.convert::<RComp>();
    let c = c.convert::<RComp>();
    ((b - a).cross(&(c - a))).normalize()
}

/// Computes the angle at vertex `b` of the given triangle
pub fn angle_between<RIn: Real, RComp: Real>(
    a: &Vector3<RIn>,
    b: &Vector3<RIn>,
    c: &Vector3<RIn>,
) -> RComp {
    let a = a.convert::<RComp>();
    let b = b.convert::<RComp>();
    let c = c.convert::<RComp>();
    ((a - b).dot(&(c - b)) / ((a - b).norm() * (c - b).norm())).acos()
}

/// Computes the minimum and maximum angle in the given triangle
pub fn tri_min_max_angles<RIn: Real, RComp: Real>(
    a: &Vector3<RIn>,
    b: &Vector3<RIn>,
    c: &Vector3<RIn>,
) -> (RComp, RComp) {
    let a = a.convert::<RComp>();
    let b = b.convert::<RComp>();
    let c = c.convert::<RComp>();
    let alpha1: RComp = angle_between(&a, &b, &c);
    let alpha2: RComp = angle_between(&b, &c, &a);
    let alpha3 = RComp::pi() - alpha1 - alpha2;

    (
        alpha1.min(alpha2.min(alpha3)),
        alpha1.max(alpha2.max(alpha3)),
    )
}

/// Computes the aspect ratio of the given triangle
///
/// The aspect ratio is computed as the inverse of the "stretch ratio" `S` given by
/// ```txt
/// S = sqrt(12) * (r_in / l_max)
/// ```
/// where `r_in` is the radius of the in-circle and `l_max` is the longest edge of the triangle.
/// See e.g.: <https://www.engmorph.com/2d-element-aspect-ratio-diff-simula>.
pub fn tri_aspect_ratio<RIn: Real, RComp: Real>(
    a: &Vector3<RIn>,
    b: &Vector3<RIn>,
    c: &Vector3<RIn>,
) -> RComp {
    let two = RComp::from_i32(2).unwrap();
    let sqrt_twelve = RComp::from_i32(12).unwrap().sqrt();

    let a = a.convert::<RComp>();
    let b = b.convert::<RComp>();
    let c = c.convert::<RComp>();

    let l0 = (a - b).norm();
    let l1 = (b - c).norm();
    let l2 = (c - a).norm();
    let s = (l0 + l1 + l2) / two;

    let area: RComp = tri_area(&a, &b, &c);
    let r_in = area / s;
    let l_max = l0.max(l1.max(l2));

    l_max / (sqrt_twelve * r_in)
}

/// Utility functions for triangles meshes
pub trait TriMesh3dExt<R: Real> {
    /// Returns the slice of all triangle vertices of the mesh
    fn tri_vertices(&self) -> &[Vector3<R>];

    /// Computes the area of the triangle with the given vertices
    fn tri_area_ijk<RComp: Real>(&self, ijk: &[usize; 3]) -> RComp {
        let v = self.tri_vertices();
        tri_area(&v[ijk[0]], &v[ijk[1]], &v[ijk[2]])
    }

    /// Computes the face normal of the triangle with the given vertices
    fn tri_normal_ijk<RComp: Real>(&self, ijk: &[usize; 3]) -> Vector3<RComp> {
        let v = self.tri_vertices();
        tri_normal(&v[ijk[0]], &v[ijk[1]], &v[ijk[2]])
    }

    /// Computes the minimum and maximum angle in the triangle with the given vertices
    fn tri_min_max_angles_ijk<RComp: Real>(&self, ijk: &[usize; 3]) -> (RComp, RComp) {
        let v = self.tri_vertices();
        tri_min_max_angles(&v[ijk[0]], &v[ijk[1]], &v[ijk[2]])
    }

    /// Computes the aspect ratio of the triangle with the given vertices
    fn tri_aspect_ratio<RComp: Real>(&self, ijk: &[usize; 3]) -> RComp {
        let v = self.tri_vertices();
        tri_aspect_ratio(&v[ijk[0]], &v[ijk[1]], &v[ijk[2]])
    }
}

impl<R: Real> TriMesh3dExt<R> for TriMesh3d<R> {
    fn tri_vertices(&self) -> &[Vector3<R>] {
        &self.vertices
    }
}

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
/// One value in the data-set is associated to a point or cell of the mesh.
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

    /// Returns a reference to the vertex indices of this cell
    fn vertices(&self) -> &[usize] {
        match self {
            TriangleOrQuadCell::Tri(v) => v,
            TriangleOrQuadCell::Quad(v) => v,
        }
    }

    /// Returns a mutable reference to the vertex indices of this cell
    fn vertices_mut(&mut self) -> &mut [usize] {
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
pub trait Mesh3d<R: Real>
where
    Self: Sized,
{
    /// The cell connectivity type of the mesh
    type Cell: CellConnectivity + Clone;

    /// Returns a slice of all vertices of the mesh
    fn vertices(&self) -> &[Vector3<R>];
    /// Returns a mutable slice of all vertices of the mesh
    fn vertices_mut(&mut self) -> &mut [Vector3<R>];
    /// Returns a slice of all cells of the mesh
    fn cells(&self) -> &[Self::Cell];

    /// Constructs a mesh from the given vertices and connectivity (does not check inputs for validity)
    fn from_vertices_and_connectivity(
        vertices: Vec<Vector3<R>>,
        connectivity: Vec<Self::Cell>,
    ) -> Self;

    /// Returns a mapping of all mesh vertices to the set of their connected neighbor vertices
    fn vertex_vertex_connectivity(&self) -> Vec<Vec<usize>> {
        profile!("vertex_vertex_connectivity");

        let mut connectivity_map: Vec<Vec<usize>> =
            vec![Vec::with_capacity(4); self.vertices().len()];
        for cell in self.cells() {
            for &i in cell.vertices() {
                for &j in cell.vertices() {
                    if i != j && !connectivity_map[i].contains(&j) {
                        connectivity_map[i].push(j);
                    }
                }
            }
        }

        connectivity_map
    }

    /// Returns a mapping of all mesh vertices to the set of the cells they belong to
    fn vertex_cell_connectivity(&self) -> Vec<Vec<usize>> {
        profile!("vertex_cell_connectivity");
        let mut connectivity_map: Vec<Vec<usize>> = vec![Vec::new(); self.vertices().len()];
        for (cell_idx, cell) in self.cells().iter().enumerate() {
            for &v_i in cell.vertices() {
                if !connectivity_map[v_i].contains(&cell_idx) {
                    connectivity_map[v_i].push(cell_idx);
                }
            }
        }

        connectivity_map
    }

    /// Returns a new mesh containing only the specified cells and removes all unreferenced vertices
    fn keep_cells(&self, cell_indices: &[usize], keep_vertices: bool) -> Self {
        if keep_vertices {
            keep_cells_impl(self, cell_indices, &[])
        } else {
            let vertex_keep_table = vertex_keep_table(self, cell_indices);
            keep_cells_impl(self, cell_indices, &vertex_keep_table)
        }
    }

    /// Removes all cells from the mesh that are completely outside of the given AABB and clamps the remaining cells to the boundary
    fn par_clamp_with_aabb(
        &self,
        aabb: &Aabb3d<R>,
        clamp_vertices: bool,
        keep_vertices: bool,
    ) -> Self
    where
        Self::Cell: Sync,
    {
        // Find all triangles with at least one vertex inside of AABB
        let vertices = self.vertices();
        let cells_to_keep = self
            .cells()
            .par_iter()
            .enumerate()
            .filter(|(_, cell)| {
                cell.vertices()
                    .iter()
                    .copied()
                    .any(|v| aabb.contains_point(&vertices[v]))
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        // Remove all other cells from mesh
        let mut new_mesh = self.keep_cells(&cells_to_keep, keep_vertices);
        // Clamp remaining vertices to AABB
        if clamp_vertices {
            new_mesh.vertices_mut().par_iter_mut().for_each(|v| {
                let min = aabb.min();
                let max = aabb.max();
                v.x = v.x.clamp(min.x, max.x);
                v.y = v.y.clamp(min.y, max.y);
                v.z = v.z.clamp(min.z, max.z);
            });
        }

        new_mesh
    }
}

/// Returns the list of vertices that should remain in the given mesh after keeping only the given cells
fn vertex_keep_table<R: Real, MeshT: Mesh3d<R>>(mesh: &MeshT, cell_indices: &[usize]) -> Vec<bool> {
    let vertices = mesh.vertices();
    let cells = mesh.cells();

    // Each entry is true if this vertex should be kept, false otherwise

    {
        let mut table = vec![false; vertices.len()];
        for cell in cell_indices.iter().copied().map(|c_i| &cells[c_i]) {
            for &vertex_index in cell.vertices() {
                table[vertex_index] = true;
            }
        }
        table
    }
}

/// Returns a new mesh keeping only the given cells and vertices in the mesh
fn keep_cells_impl<R: Real, MeshT: Mesh3d<R>>(
    mesh: &MeshT,
    cell_indices: &[usize],
    vertex_keep_table: &[bool],
) -> MeshT {
    let vertices = mesh.vertices();
    let cells = mesh.cells();

    if vertex_keep_table.is_empty() {
        MeshT::from_vertices_and_connectivity(
            mesh.vertices().to_vec(),
            cell_indices.iter().map(|&i| &cells[i]).cloned().collect(),
        )
    } else {
        assert_eq!(mesh.vertices().len(), vertex_keep_table.len());

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
            .map(|&i| &cells[i])
            .cloned()
            .map(|mut cell| {
                for index in cell.vertices_mut() {
                    *index = *old_to_new_label_map
                        .get(index)
                        .expect("Index must be in map");
                }
                cell
            })
            .collect();

        let relabeled_vertices: Vec<_> = vertex_keep_table
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, should_keep)| if should_keep { Some(i) } else { None })
            .map(|index| vertices[index])
            .collect();

        MeshT::from_vertices_and_connectivity(relabeled_vertices, relabeled_cells)
    }
}

/// Basic interface for mesh cells consisting of a collection of vertex indices
pub trait CellConnectivity {
    /// Returns the number of vertices per cell (may vary between cells)
    fn num_vertices(&self) -> usize {
        Self::expected_num_vertices()
    }
    /// Returns the expected number of vertices per cell (helpful for connectivities with a constant or upper bound on the number of vertices to reserve storage)
    fn expected_num_vertices() -> usize;
    /// Returns a reference to the vertex indices connected by this cell
    fn vertices(&self) -> &[usize];
    /// Returns a reference to the vertex indices connected by this cell
    fn vertices_mut(&mut self) -> &mut [usize];
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

    fn vertices(&self) -> &[usize] {
        &self.0[..]
    }

    fn vertices_mut(&mut self) -> &mut [usize] {
        &mut self.0[..]
    }
}

impl CellConnectivity for HexCell {
    fn expected_num_vertices() -> usize {
        8
    }

    fn vertices(&self) -> &[usize] {
        &self.0[..]
    }

    fn vertices_mut(&mut self) -> &mut [usize] {
        &mut self.0[..]
    }
}

impl CellConnectivity for TriangleOrQuadCell {
    fn expected_num_vertices() -> usize {
        4
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices()
    }

    fn vertices(&self) -> &[usize] {
        TriangleOrQuadCell::vertices(self)
    }

    fn vertices_mut(&mut self) -> &mut [usize] {
        TriangleOrQuadCell::vertices_mut(self)
    }
}

impl CellConnectivity for PointCell {
    fn expected_num_vertices() -> usize {
        1
    }

    fn vertices(&self) -> &[usize] {
        std::slice::from_ref(&self.0)
    }

    fn vertices_mut(&mut self) -> &mut [usize] {
        std::slice::from_mut(&mut self.0)
    }
}

impl<R: Real> Mesh3d<R> for TriMesh3d<R> {
    type Cell = TriangleCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        self.vertices.as_mut_slice()
    }

    fn cells(&self) -> &[TriangleCell] {
        self.triangle_cells()
    }

    fn from_vertices_and_connectivity(
        vertices: Vec<Vector3<R>>,
        triangles: Vec<Self::Cell>,
    ) -> Self {
        Self {
            vertices,
            triangles: bytemuck::cast_vec::<TriangleCell, [usize; 3]>(triangles),
        }
    }
}

impl<R: Real> Mesh3d<R> for HexMesh3d<R> {
    type Cell = HexCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        self.vertices.as_mut_slice()
    }

    fn cells(&self) -> &[HexCell] {
        bytemuck::cast_slice::<[usize; 8], HexCell>(self.cells.as_slice())
    }

    fn from_vertices_and_connectivity(vertices: Vec<Vector3<R>>, cells: Vec<HexCell>) -> Self {
        Self {
            vertices,
            cells: bytemuck::cast_vec::<HexCell, [usize; 8]>(cells),
        }
    }
}

impl<R: Real> Mesh3d<R> for MixedTriQuadMesh3d<R> {
    type Cell = TriangleOrQuadCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        self.vertices.as_mut_slice()
    }

    fn cells(&self) -> &[TriangleOrQuadCell] {
        &self.cells
    }

    fn from_vertices_and_connectivity(
        vertices: Vec<Vector3<R>>,
        cells: Vec<TriangleOrQuadCell>,
    ) -> Self {
        Self { vertices, cells }
    }
}

impl<R: Real> Mesh3d<R> for PointCloud3d<R> {
    type Cell = PointCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.points.as_slice()
    }

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        self.points.as_mut_slice()
    }

    fn cells(&self) -> &[PointCell] {
        bytemuck::cast_slice::<usize, PointCell>(self.indices.as_slice())
    }

    fn from_vertices_and_connectivity(points: Vec<Vector3<R>>, cells: Vec<PointCell>) -> Self {
        Self {
            points,
            indices: bytemuck::cast_vec::<PointCell, usize>(cells),
        }
    }
}

impl<R: Real, MeshT: Mesh3d<R> + Clone> Mesh3d<R> for std::borrow::Cow<'_, MeshT> {
    type Cell = MeshT::Cell;

    fn vertices(&self) -> &[Vector3<R>] {
        (*self.as_ref()).vertices()
    }

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        (self.to_mut()).vertices_mut()
    }

    fn cells(&self) -> &[MeshT::Cell] {
        (*self.as_ref()).cells()
    }

    fn from_vertices_and_connectivity(vertices: Vec<Vector3<R>>, cells: Vec<Self::Cell>) -> Self {
        std::borrow::Cow::Owned(MeshT::from_vertices_and_connectivity(vertices, cells).to_owned())
    }
}

impl TriangleCell {
    /// Returns an iterator over all edges of this triangle
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..3).map(|i| (self.0[i], self.0[(i + 1) % 3]))
    }
}

pub struct MeshEdgeInformation {
    /// Map from sorted edge to `(edge_idx, edge_count)`
    edge_counts: MapType<[usize; 2], (usize, usize)>,
    /// For each edge_idx: (edge, face_idx, local_edge_idx)
    edge_info: Vec<([usize; 2], usize, usize)>,
}

pub struct EdgeInformation {
    /// The vertices of the edge
    pub edge: [usize; 2],
    /// The vertices of the edge in ascending order
    pub edge_sorted: [usize; 2],
    /// Total number of incident faces to the edge
    pub incident_faces: usize,
    /// One representative face that contains the edge with the given index
    pub face: usize,
    /// Local index of this edge in the representative face
    pub local_edge_index: usize,
}

impl MeshEdgeInformation {
    /// Iterator over all edge information stored in this struct
    pub fn iter(&self) -> impl Iterator<Item = EdgeInformation> + '_ {
        self.edge_counts.iter().map(|(e, (edge_idx, count))| {
            let info = &self.edge_info[*edge_idx];
            EdgeInformation {
                edge: info.0,
                edge_sorted: *e,
                incident_faces: *count,
                face: info.1,
                local_edge_index: info.2,
            }
        })
    }

    /// Returns the number of boundary edges (i.e. edges with 1 incident face)
    pub fn count_boundary_edges(&self) -> usize {
        self.edge_counts
            .values()
            .filter(|(_, count)| *count == 1)
            .count()
    }

    /// Returns the number of non-manifold edges (i.e. edges with more than 2 incident face)
    pub fn count_non_manifold_edges(&self) -> usize {
        self.edge_counts
            .values()
            .filter(|(_, count)| *count > 2)
            .count()
    }

    /// Iterator over all boundary edges
    pub fn boundary_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edge_counts
            .values()
            .filter(|(_, count)| *count == 1)
            .map(move |(edge_idx, _)| self.edge_info[*edge_idx].0)
    }

    /// Iterator over all non-manifold edges
    pub fn non_manifold_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edge_counts
            .values()
            .filter(|(_, count)| *count > 2)
            .map(move |(edge_idx, _)| self.edge_info[*edge_idx].0)
    }
}

pub struct MeshManifoldInformation {
    /// List of all edges with only one incident face
    pub boundary_edges: Vec<[usize; 2]>,
    /// List of all non-manifold edges (edges with more than two incident faces)
    pub non_manifold_edges: Vec<[usize; 2]>,
    /// List of all non-manifold vertices (vertices with more than one fan of faces)
    pub non_manifold_vertices: Vec<usize>,
}

impl MeshManifoldInformation {
    /// Returns whether the associated mesh is closed (has no boundary edges)
    pub fn is_closed(&self) -> bool {
        self.boundary_edges.is_empty()
    }

    /// Returns whether the associated mesh is a 2-manifold (no non-manifold edges and no non-manifold vertices)
    pub fn is_manifold(&self) -> bool {
        self.non_manifold_edges.is_empty() && self.non_manifold_vertices.is_empty()
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
            vertices: new_verts,
            triangles: new_tris,
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
    fn vertex_normals_inplace_assume_zeroed(&self, normals: &mut [Unit<Vector3<R>>]) {
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
    fn par_vertex_normals_inplace_assume_zeroed(&self, normals: &mut [Unit<Vector3<R>>]) {
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

    /// Computes a helper struct with information about all edges in the mesh (i.e. number of incident triangles etc.)
    pub fn compute_edge_information(&self) -> MeshEdgeInformation {
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

        MeshEdgeInformation {
            edge_counts,
            edge_info,
        }
    }

    /// Returns all non-manifold vertices of this mesh
    ///
    /// A non-manifold vertex is generated by pinching two surface sheets together at that vertex
    /// such that the vertex is incident to more than one fan of triangles.
    ///
    /// Note: This function assumes that all edges in the mesh are manifold edges! If there are
    ///  non-manifold edges, it is possible to connect two triangle fans using a third fan which is
    ///  not detected by this function.
    pub fn find_non_manifold_vertices(&self) -> Vec<usize> {
        let mut non_manifold_verts = Vec::new();
        let mut tri_fan = Vec::new();

        let sort_edge = |edge: (usize, usize)| -> (usize, usize) {
            if edge.0 > edge.1 {
                (edge.1, edge.0)
            } else {
                edge
            }
        };

        let is_fan_triangle =
            |vert: usize, next_tri: &TriangleCell, current_fan: &[TriangleCell]| -> bool {
                let mut is_fan_tri = false;

                // Check each edge of the tri against all triangles in the fan
                'edge_loop: for edge in next_tri.edges().map(sort_edge) {
                    // Only edges connected to the current vertex are relevant
                    if edge.0 == vert || edge.1 == vert {
                        // Check against all triangles of the current fan
                        for fan_tri in current_fan {
                            for fan_edge in fan_tri.edges().map(sort_edge) {
                                if edge == fan_edge {
                                    // Triangle is part of the current fan
                                    is_fan_tri = true;
                                    break 'edge_loop;
                                }
                            }
                        }
                    }
                }

                is_fan_tri
            };

        let tris = self.triangle_cells();
        let vert_face_connectivity = self.vertex_cell_connectivity();
        for vert in 0..self.vertices.len() {
            let mut remaining_faces = vert_face_connectivity[vert]
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();

            if remaining_faces.len() > 1 {
                // Pick an arbitrary first face of the fan
                tri_fan.push(tris[remaining_faces.pop_first().unwrap()]);
                // Try to match all other faces
                while !remaining_faces.is_empty() {
                    let mut found_next = None;

                    // Check all remaining faces against the current fan
                    for &next_candidate in &remaining_faces {
                        if is_fan_triangle(vert, &tris[next_candidate], &tri_fan) {
                            found_next = Some(next_candidate);
                            break;
                        }
                    }

                    if let Some(next) = found_next {
                        // New fan triangle found
                        tri_fan.push(tris[next]);
                        remaining_faces.remove(&next);
                    } else {
                        // None of the remaining faces are part of the fan
                        break;
                    }
                }

                if !remaining_faces.is_empty() {
                    // At least one triangle is not part of the current fan
                    //  -> Non-manifold vertex was found
                    non_manifold_verts.push(vert);
                }

                tri_fan.clear();
            }
        }

        non_manifold_verts
    }

    /// Returns a struct with lists of all boundary edges, non-manifold edges and non-manifold vertices
    ///
    /// Note that the output order is not necessarily deterministic due to the internal use of hashmaps.
    pub fn compute_manifold_information(&self) -> MeshManifoldInformation {
        let edges = self.compute_edge_information();
        let boundary_edges = edges.boundary_edges().collect();
        let non_manifold_edges = edges.non_manifold_edges().collect();

        let non_manifold_vertices = self.find_non_manifold_vertices();

        MeshManifoldInformation {
            boundary_edges,
            non_manifold_edges,
            non_manifold_vertices,
        }
    }
}

#[cfg(test)]
mod tri_mesh_tests {
    use super::*;

    fn mesh_one_tri() -> TriMesh3d<f64> {
        TriMesh3d::<f64> {
            vertices: vec![
                Vector3::new_random(),
                Vector3::new_random(),
                Vector3::new_random(),
            ],
            triangles: vec![[0, 1, 2]],
        }
    }

    fn mesh_non_manifold_edge() -> TriMesh3d<f64> {
        TriMesh3d::<f64> {
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2], [1, 2, 4]],
        }
    }

    fn mesh_non_manifold_edge_double() -> TriMesh3d<f64> {
        TriMesh3d::<f64> {
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2], [1, 2, 4], [4, 2, 1]],
        }
    }

    fn mesh_non_manifold_vertex() -> TriMesh3d<f64> {
        TriMesh3d::<f64> {
            vertices: vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
                Vector3::new(2.0, 1.0, 1.0),
                Vector3::new(1.0, 2.0, 1.0),
            ],
            triangles: vec![[0, 2, 1], [2, 3, 4]],
        }
    }

    #[test]
    fn test_tri_mesh_edge_info() {
        let mesh = mesh_non_manifold_edge();
        let edges = mesh.compute_edge_information();

        for ei in edges.iter() {
            if ei.edge_sorted == [1, 2] {
                assert_eq!(ei.incident_faces, 3);
            } else {
                assert_eq!(ei.incident_faces, 1);
            }
        }

        assert_eq!(edges.count_boundary_edges(), 6);
        assert_eq!(edges.count_non_manifold_edges(), 1);
    }

    #[test]
    fn test_tri_mesh_non_manifold_vertex_info() {
        let mesh = mesh_non_manifold_vertex();
        let non_manifold_vertes = mesh.find_non_manifold_vertices();
        assert_eq!(non_manifold_vertes, [2]);
    }

    #[test]
    fn test_tri_mesh_manifold_info() {
        {
            let mesh = mesh_one_tri();
            let info = mesh.compute_manifold_information();
            assert!(!info.is_closed());
            assert!(info.is_manifold());
            assert_eq!(info.non_manifold_edges.len(), 0);
            assert_eq!(info.non_manifold_vertices.len(), 0);
        }

        {
            let mesh = mesh_non_manifold_edge();
            let info = mesh.compute_manifold_information();
            assert!(!info.is_closed());
            assert!(!info.is_manifold());
            assert_eq!(info.non_manifold_edges.len(), 1);
            assert_eq!(info.non_manifold_vertices.len(), 0);
        }

        {
            let mesh = mesh_non_manifold_edge_double();
            let info = mesh.compute_manifold_information();
            assert!(!info.is_closed());
            assert!(!info.is_manifold());
            assert_eq!(info.non_manifold_edges.len(), 1);
            assert_eq!(info.non_manifold_vertices.len(), 0);
        }

        {
            let mesh = mesh_non_manifold_vertex();
            let info = mesh.compute_manifold_information();
            assert!(!info.is_closed());
            assert!(!info.is_manifold());
            assert_eq!(info.non_manifold_edges.len(), 0);
            assert_eq!(info.non_manifold_vertices.len(), 1);
        }
    }
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

    fn vertices_mut(&mut self) -> &mut [Vector3<R>] {
        self.mesh.vertices_mut()
    }

    fn cells(&self) -> &[MeshT::Cell] {
        self.mesh.cells()
    }

    fn from_vertices_and_connectivity(
        vertices: Vec<Vector3<R>>,
        connectivity: Vec<Self::Cell>,
    ) -> Self {
        MeshWithData::new(MeshT::from_vertices_and_connectivity(
            vertices,
            connectivity,
        ))
    }

    /// Returns a new mesh containing only the specified cells and removes all unreferenced vertices and attributes
    fn keep_cells(&self, cell_indices: &[usize], keep_all_vertices: bool) -> Self {
        // Filter internal mesh
        let mut new_mesh = if keep_all_vertices {
            let mut new_mesh = keep_cells_impl(self, cell_indices, &[]);
            new_mesh.point_attributes = self.point_attributes.clone();
            new_mesh
        } else {
            let vertex_keep_table = vertex_keep_table(self, cell_indices);
            let mut new_mesh = keep_cells_impl(self, cell_indices, &vertex_keep_table);

            let vertex_indices = vertex_keep_table
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, should_keep)| if should_keep { Some(i) } else { None })
                .collect::<Vec<_>>();

            // Filter the point attributes
            new_mesh.point_attributes = self
                .point_attributes
                .iter()
                .map(|attr| attr.keep_indices(&vertex_indices))
                .collect();

            new_mesh
        };

        // Filter the cell attributes
        let cell_attributes = self
            .cell_attributes
            .iter()
            .map(|attr| attr.keep_indices(cell_indices))
            .collect();
        new_mesh.cell_attributes = cell_attributes;

        new_mesh
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

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`] trait
    pub fn new_real_scalar<S: Into<String>>(name: S, data: impl Into<Vec<R>>) -> Self {
        Self {
            name: name.into(),
            data: AttributeData::ScalarReal(data.into()),
        }
    }

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`] trait
    pub fn new_real_vector3<S: Into<String>>(name: S, data: impl Into<Vec<Vector3<R>>>) -> Self {
        Self {
            name: name.into(),
            data: AttributeData::Vector3Real(data.into()),
        }
    }

    /// Converts the mesh attribute to a [`vtkio::model::Attribute`])
    #[cfg(feature = "vtk_extras")]
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
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

    /// Returns a new attribute keeping only the entries with the given index
    fn keep_indices(&self, indices: &[usize]) -> Self {
        let data = match &self.data {
            AttributeData::ScalarU64(d) => {
                AttributeData::ScalarU64(indices.iter().copied().map(|i| d[i]).collect())
            }
            AttributeData::ScalarReal(d) => {
                AttributeData::ScalarReal(indices.iter().copied().map(|i| d[i]).collect())
            }
            AttributeData::Vector3Real(d) => {
                AttributeData::Vector3Real(indices.iter().copied().map(|i| d[i]).collect())
            }
        };

        Self {
            name: self.name.clone(),
            data,
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
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
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
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for $name<R> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(&self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for &$name<R> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<'a, R: Real> IntoVtkUnstructuredGridPiece for std::borrow::Cow<'a, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(&self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<'a, R: Real> IntoVtkUnstructuredGridPiece for &std::borrow::Cow<'a, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                vtk_helper::mesh_to_unstructured_grid(self)
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for &MeshWithData<R, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                self.unstructured_grid()
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
        impl<R: Real> IntoVtkUnstructuredGridPiece for MeshWithData<R, $name<R>> {
            fn into_unstructured_grid(self) -> UnstructuredGridPiece {
                self.unstructured_grid()
            }
        }

        #[cfg(feature = "vtk_extras")]
        #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
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
#[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
pub mod vtk_helper {
    use vtkio::IOBuffer;
    use vtkio::model::{
        Attributes, CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers,
    };

    use super::{
        CellConnectivity, HexCell, Mesh3d, PointCell, Real, TriangleCell, TriangleOrQuadCell,
    };

    /// Trait that can be implemented by mesh cells to return the corresponding [`vtkio::model::CellType`]
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    pub trait HasVtkCellType {
        /// Returns the corresponding [`vtkio::model::CellType`] of the cell
        fn vtk_cell_type(&self) -> CellType;
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for TriangleCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Triangle
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for TriangleOrQuadCell {
        fn vtk_cell_type(&self) -> CellType {
            match self {
                TriangleOrQuadCell::Tri(_) => CellType::Triangle,
                TriangleOrQuadCell::Quad(_) => CellType::Quad,
            }
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for HexCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Hexahedron
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for PointCell {
        fn vtk_cell_type(&self) -> CellType {
            CellType::Vertex
        }
    }

    /// Conversion of meshes into a [`vtkio::model::UnstructuredGridPiece`]
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    pub trait IntoVtkUnstructuredGridPiece {
        fn into_unstructured_grid(self) -> UnstructuredGridPiece;
    }

    /// Direct conversion of meshes into a full [`vtkio::model::DataSet`]
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
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
    #[cfg_attr(docsrs, doc(cfg(feature = "vtk_extras")))]
    pub fn mesh_to_unstructured_grid<'a, R, MeshT>(mesh: &'a MeshT) -> UnstructuredGridPiece
    where
        R: Real,
        MeshT: Mesh3d<R> + 'a,
        <MeshT as Mesh3d<R>>::Cell: HasVtkCellType,
    {
        let points = {
            let mut points: Vec<R> = Vec::with_capacity(mesh.vertices().len() * 3);
            points.extend(mesh.vertices().iter().flat_map(|p| p.as_slice()));
            points
        };

        let vertices = {
            let mut vertices =
                Vec::with_capacity(mesh.cells().len() * (MeshT::Cell::expected_num_vertices() + 1));
            for cell in mesh.cells().iter() {
                vertices.push(cell.num_vertices() as u32);
                cell.vertices()
                    .iter()
                    .copied()
                    .for_each(|v| vertices.push(v as u32));
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
