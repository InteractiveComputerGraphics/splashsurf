//! Basic mesh types used by the library and implementation of VTK export

use crate::{new_map, Real};
use nalgebra::{Unit, Vector3};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt::Debug;
use thread_local::ThreadLocal;
#[cfg(feature = "vtk_extras")]
use vtkio::model::{Attribute, UnstructuredGridPiece};

// TODO: Rename/restructure VTK helper implementations

/// A triangle (surface) mesh in 3D
#[derive(Clone, Debug, Default)]
pub struct TriMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The triangles of the mesh identified by their vertex indices
    pub triangles: Vec<[usize; 3]>,
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
            let normal = (v0 - v1).cross(&(v2 - v1));

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
                let normal = (v0 - v1).cross(&(v2 - v1));

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
            let normal_directions = unsafe {
                // This is sound, as Unit<T> has repr(transparent)
                let vector3_ptr = normals.as_mut_ptr() as *mut Vector3<R>;
                let normal_directions: &'a mut _ =
                    std::slice::from_raw_parts_mut(vector3_ptr, normals.len());
                normal_directions
            };
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
            let normal_directions = unsafe {
                // This is sound, as Unit<T> has repr(transparent)
                let vector3_ptr = normals.as_mut_ptr() as *mut Vector3<R>;
                let normal_directions: &'a mut _ =
                    std::slice::from_raw_parts_mut(vector3_ptr, normals.len());
                normal_directions
            };
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
    pub points: Vec<Vector3<R>>,
}

/// A mesh with attached vertex or point data
#[derive(Clone, Debug, Default)]
pub struct MeshWithData<MeshT, PointDataT, CellDataT = ()> {
    /// The mesh geometry itself
    pub mesh: MeshT,
    /// Data attached to each vertex or point of the mesh
    pub point_data: Vec<PointDataT>,
    /// Data attached to each cell of the mesh
    pub cell_data: Vec<CellDataT>,
}

impl<MeshT, PointDataT, CellDataT> MeshWithData<MeshT, PointDataT, CellDataT> {
    /// Creates a new mesh the given point data
    pub fn with_point_data<PointData: Into<Vec<PointDataT>>>(
        mesh: MeshT,
        point_data: PointData,
    ) -> Self {
        Self {
            mesh,
            point_data: point_data.into(),
            cell_data: vec![],
        }
    }

    /// Creates a new mesh the given cell data
    pub fn with_cell_data<CellData: Into<Vec<CellDataT>>>(
        mesh: MeshT,
        cell_data: CellData,
    ) -> Self {
        Self {
            mesh,
            point_data: vec![],
            cell_data: cell_data.into(),
        }
    }
}

#[cfg(feature = "vtk_extras")]
impl<'a, MeshT: 'a, PointDataT> MeshWithData<MeshT, PointDataT, ()>
where
    &'a MeshT: Into<UnstructuredGridPiece>,
    PointDataT: Real,
{
    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub fn to_dataset(&'a self) -> UnstructuredGridPiece {
        let mut grid_piece: UnstructuredGridPiece = (&self.mesh).into();
        grid_piece
            .data
            .point
            .push(Attribute::scalars("density", 1).with_data(self.point_data.clone()));
        grid_piece
    }
}

#[cfg(feature = "vtk_extras")]
impl<'a, MeshT: 'a> MeshWithData<MeshT, (), u64>
where
    &'a MeshT: Into<UnstructuredGridPiece>,
{
    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    pub fn to_dataset(&'a self) -> UnstructuredGridPiece {
        let mut grid_piece: UnstructuredGridPiece = (&self.mesh).into();
        grid_piece
            .data
            .cell
            .push(Attribute::scalars("node_id", 1).with_data(self.cell_data.clone()));
        grid_piece
    }
}

#[cfg(feature = "vtk_extras")]
#[doc(hidden)]
/// Trait implementations to convert meshes into types supported by [`vtkio`](https://github.com/elrnv/vtkio)
pub mod vtk_helper {
    use vtkio::model::{
        Attributes, CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers,
    };
    use vtkio::IOBuffer;

    use super::{HexMesh3d, PointCloud3d, Real, TriMesh3d};

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<R> From<&TriMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &TriMesh3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::with_capacity(mesh.vertices.len() * 3);
                for v in mesh.vertices.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::with_capacity(mesh.triangles.len() * (3 + 1));
                for triangle in mesh.triangles.iter() {
                    vertices.push(3);
                    vertices.extend(triangle.iter().map(|&i| i as u32));
                }
                vertices
            };

            let cell_types = vec![CellType::Triangle; mesh.triangles.len()];

            new_unstructured_grid_piece(points, vertices, cell_types)
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this mesh
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<'a, R> From<&'a HexMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a HexMesh3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::with_capacity(mesh.vertices.len() * 3);
                for v in mesh.vertices.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::with_capacity(mesh.cells.len() * (8 + 1));
                for cell in mesh.cells.iter() {
                    vertices.push(8);
                    vertices.extend(cell.iter().map(|&i| i as u32));
                }
                vertices
            };

            let cell_types = vec![CellType::Hexahedron; mesh.cells.len()];

            new_unstructured_grid_piece(points, vertices, cell_types)
        }
    }

    /// Creates a [`vtkio::model::UnstructuredGridPiece`](https://docs.rs/vtkio/0.6.*/vtkio/model/struct.UnstructuredGridPiece.html) representing this point cloud
    #[cfg_attr(doc_cfg, doc(cfg(feature = "vtk_extras")))]
    impl<'a, R> From<&'a PointCloud3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a PointCloud3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::with_capacity(mesh.points.len() * 3);
                for v in mesh.points.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::with_capacity(mesh.points.len() * (1 + 1));
                for (i, _) in mesh.points.iter().enumerate() {
                    vertices.push(1 as u32);
                    vertices.push(i as u32);
                }
                vertices
            };

            let cell_types = vec![CellType::Vertex; mesh.points.len()];

            new_unstructured_grid_piece(points, vertices, cell_types)
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
