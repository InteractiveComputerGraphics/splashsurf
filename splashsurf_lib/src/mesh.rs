//! Basic mesh types used by the library and implementation of VTK export
//!
//! This modules provides three basic types of meshes embedded in three dimensional spaces used
//! by the library:
//!  - [`TriMesh3d`]
//!  - [`HexMesh3d`]
//!  - [`PointCloud3d`]
//!
//! Furthermore, it provides the [`MeshWithData`] type that is used when additional attributes are
//! attached to the vertices (e.g. normals) or cells (e.g. some identifiers) of the mesh.
//!
//! If the `vtk_extras` feature is enabled, this module also provides features for conversion of these
//! meshes to [`vtkio`](https://docs.rs/vtkio/0.6.*/vtkio/index.html) data structures. For example:
//!  - [`MeshWithData::to_unstructured_grid`] to convert a mesh together with all attached attributes
//!  - [`vtk_helper::mesh_to_unstructured_grid`] to convert a basic mesh without additional data
//!  - `From<T> for UnstructuredGridPiece` implementations for the basic mesh types
//!  - `Into<DataSet>` implementations for the basic mesh types

use crate::{new_map, Real};
use bytemuck::{cast_slice, cast_slice_mut};
use bytemuck_derive::{Pod, Zeroable};
use nalgebra::{Unit, Vector3};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt::Debug;
use thread_local::ThreadLocal;
#[cfg(feature = "vtk_extras")]
use vtkio::model::{Attribute, DataSet, UnstructuredGridPiece};

// TODO: Rename/restructure VTK helper implementations

/// A named attribute with data that can be attached to the vertices or cells of a mesh
#[derive(Clone, Debug)]
pub struct MeshAttribute<R: Real> {
    /// Name of the attribute
    pub name: &'static str,
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
}

/// Basic interface for mesh cells consisting of a collection of vertex indices
pub trait CellConnectivity {
    /// Returns the number of vertices per cell
    fn num_vertices() -> usize;
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

/// Cell type for the [`TriMesh3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct TriangleCell(pub [usize; 3]);
/// Cell type for the [`HexMesh3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct HexCell(pub [usize; 8]);
/// Cell type for the [`PointCloud3d`]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(transparent)]
pub struct PointCell(pub usize);

impl CellConnectivity for TriangleCell {
    fn num_vertices() -> usize {
        3
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        self.0.iter().copied().try_for_each(f)
    }
}

impl CellConnectivity for HexCell {
    fn num_vertices() -> usize {
        8
    }

    fn try_for_each_vertex<E, F: FnMut(usize) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        self.0.iter().copied().try_for_each(f)
    }
}

impl CellConnectivity for PointCell {
    fn num_vertices() -> usize {
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
        cast_slice::<[usize; 3], TriangleCell>(self.triangles.as_slice())
    }
}

impl<R: Real> Mesh3d<R> for HexMesh3d<R> {
    type Cell = HexCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.vertices.as_slice()
    }

    fn cells(&self) -> &[HexCell] {
        cast_slice::<[usize; 8], HexCell>(self.cells.as_slice())
    }
}

impl<R: Real> Mesh3d<R> for PointCloud3d<R> {
    type Cell = PointCell;

    fn vertices(&self) -> &[Vector3<R>] {
        self.points.as_slice()
    }

    fn cells(&self) -> &[PointCell] {
        cast_slice::<usize, PointCell>(self.indices.as_slice())
    }
}

impl<R: Real> TriMesh3d<R> {
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
            let normal_directions = cast_slice_mut::<Unit<Vector3<R>>, Vector3<R>>(normals);
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
            let normal_directions = cast_slice_mut::<Unit<Vector3<R>>, Vector3<R>>(normals);
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
    pub fn new(name: &'static str, data: impl Into<AttributeData<R>>) -> Self {
        Self {
            name,
            data: data.into(),
        }
    }

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`](crate::Real) trait
    pub fn new_real_scalar(name: &'static str, data: impl Into<Vec<R>>) -> Self {
        Self {
            name,
            data: AttributeData::ScalarReal(data.into()),
        }
    }

    /// Creates a new named mesh attribute with scalar values implementing the [`Real`](crate::Real) trait
    pub fn new_real_vector3(name: &'static str, data: impl Into<Vec<Vector3<R>>>) -> Self {
        Self {
            name,
            data: AttributeData::Vector3Real(data.into()),
        }
    }

    /// Converts the mesh attribute to a [`vtkio::model::Attribute`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.Attribute.html)
    #[cfg(feature = "vtk_extras")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    fn to_vtk_attribute(&self) -> Attribute {
        match &self.data {
            AttributeData::ScalarU64(u64_vec) => {
                Attribute::scalars(self.name, 1).with_data(u64_vec.clone())
            }
            AttributeData::ScalarReal(real_vec) => {
                Attribute::scalars(self.name, 1).with_data(real_vec.clone())
            }
            AttributeData::Vector3Real(vec3r_vec) => Attribute::scalars(self.name, 3)
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
    for<'a> &'a MeshT: Into<UnstructuredGridPiece>,
{
    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh including its attached [`MeshAttribute`]s
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub fn to_unstructured_grid(&self) -> UnstructuredGridPiece {
        let mut grid_piece: UnstructuredGridPiece = (&self.mesh).into();
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

/// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh with all its attributes and wraps it into a [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
#[cfg(feature = "vtk_extras")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
impl<R, MeshT> Into<DataSet> for &MeshWithData<R, MeshT>
where
    R: Real,
    MeshT: Mesh3d<R>,
    for<'a> &'a MeshT: Into<UnstructuredGridPiece>,
{
    fn into(self) -> DataSet {
        DataSet::inline(self.to_unstructured_grid())
    }
}

/// Trait implementations to convert meshes into types supported by [`vtkio`](https://github.com/elrnv/vtkio)
#[cfg(feature = "vtk_extras")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
pub mod vtk_helper {
    use vtkio::model::{
        Attributes, CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers,
    };
    use vtkio::IOBuffer;

    use super::{
        CellConnectivity, HexCell, HexMesh3d, Mesh3d, PointCell, PointCloud3d, Real, TriMesh3d,
        TriangleCell,
    };

    /// Trait that can be implemented by mesh cells to return the corresponding [`vtkio::model::CellType`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.CellType.html)
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub trait HasVtkCellType {
        /// Returns the corresponding [`vtkio::model::CellType`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.CellType.html) of the cell
        fn vtk_cell_type() -> CellType;
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for TriangleCell {
        fn vtk_cell_type() -> CellType {
            CellType::Triangle
        }
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for HexCell {
        fn vtk_cell_type() -> CellType {
            CellType::Hexahedron
        }
    }

    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl HasVtkCellType for PointCell {
        fn vtk_cell_type() -> CellType {
            CellType::Vertex
        }
    }

    /// Converts any supported mesh to a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html)
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub fn mesh_to_unstructured_grid<'a, R, MeshT>(mesh: &'a MeshT) -> UnstructuredGridPiece
    where
        R: Real,
        MeshT: Mesh3d<R> + 'a,
        <MeshT as Mesh3d<R>>::Cell: HasVtkCellType,
    {
        let points = {
            let mut points: Vec<R> = Vec::with_capacity(mesh.vertices().len() * 3);
            for v in mesh.vertices().iter() {
                points.extend(v.as_slice());
            }
            points
        };

        let vertices_per_cell = MeshT::Cell::num_vertices();
        let vertices = {
            let mut vertices = Vec::with_capacity(mesh.cells().len() * (vertices_per_cell + 1));
            for cell in mesh.cells().iter() {
                vertices.push(3);
                cell.for_each_vertex(|v| vertices.push(v as u32));
            }
            vertices
        };

        let cell_types = vec![<MeshT::Cell as HasVtkCellType>::vtk_cell_type(); mesh.cells().len()];

        new_unstructured_grid_piece(points, vertices, cell_types)
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<R> From<&TriMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &TriMesh3d<R>) -> Self {
            mesh_to_unstructured_grid(mesh)
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<'a, R> From<&'a HexMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a HexMesh3d<R>) -> Self {
            mesh_to_unstructured_grid(mesh)
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this point cloud
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<'a, R> From<&'a PointCloud3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a PointCloud3d<R>) -> Self {
            mesh_to_unstructured_grid(mesh)
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh and wraps it into a [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<R: Real> Into<DataSet> for &TriMesh3d<R> {
        fn into(self) -> DataSet {
            DataSet::inline(UnstructuredGridPiece::from(self))
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh and wraps it into a [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<R: Real> Into<DataSet> for &HexMesh3d<R> {
        fn into(self) -> DataSet {
            DataSet::inline(UnstructuredGridPiece::from(self))
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this point cloud and wraps it into a [`vtkio::model::DataSet`](https://docs.rs/vtkio/0.6.*/vtkio/model/enum.DataSet.html)
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<R: Real> Into<DataSet> for &PointCloud3d<R> {
        fn into(self) -> DataSet {
            DataSet::inline(UnstructuredGridPiece::from(self))
        }
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
