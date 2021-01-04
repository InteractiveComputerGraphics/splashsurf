use std::fmt::Debug;

use nalgebra::{Unit, Vector3};

use crate::Real;

/// A triangle (surface) mesh in 3D
#[derive(Clone, Debug, Default)]
pub struct TriMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The triangles of the mesh identified by their vertex indices
    pub triangles: Vec<[usize; 3]>,
}

impl<R: Real> TriMesh3d<R> {
    /// Appends the other mesh to this mesh by simply appending all vertices and triangles of the other mesh and adjusting indices accordingly
    pub fn append(&mut self, other: TriMesh3d<R>) {
        let TriMesh3d {
            vertices: mut new_verts,
            triangles: mut new_tris,
        } = other;

        let vertex_offset = self.vertices.len();
        let tri_offset = self.triangles.len();

        self.vertices.append(&mut new_verts);
        self.triangles.append(&mut new_tris);

        for tri in self.triangles.iter_mut().skip(tri_offset) {
            tri[0] += vertex_offset;
            tri[1] += vertex_offset;
            tri[2] += vertex_offset;
        }
    }

    /// Same as [Self::vertex_normal_directions_inplace] but assumes that the output is already zeroed
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

    /// Computes the mesh's vertex normal directions inplace using an area weighted average of the adjacent triangle faces
    ///
    /// Note that this function only computes the normal directions, these vectors are **not normalized**!
    /// See [Self::vertex_normals_inplace] if actual normal vectors are needed.
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
    /// See [Self::vertex_normals] if actual normal vectors are needed.
    pub fn vertex_normal_directions(&self) -> Vec<Vector3<R>> {
        let mut normal_directions = vec![Vector3::zeros(); self.vertices.len()];
        self.vertex_normal_directions_inplace_assume_zeroed(normal_directions.as_mut_slice());
        normal_directions
    }

    /// Same as [Self::vertex_normals_inplace] but assumes that the output is already zeroed
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
            normal.renormalize();
        }
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
pub struct MeshWithPointData<MeshT, DataT> {
    /// The mesh geometry itself
    pub mesh: MeshT,
    /// Data attached to each vertex or point of the mesh
    pub data: Vec<DataT>,
}

#[cfg(feature = "vtk_extras")]
use vtkio::model::{Attribute, UnstructuredGridPiece};

#[cfg(feature = "vtk_extras")]
impl<'a, MeshT: 'a, DataT> MeshWithPointData<MeshT, DataT>
where
    &'a MeshT: Into<UnstructuredGridPiece>,
    DataT: Real,
{
    pub fn to_dataset(&'a self) -> UnstructuredGridPiece {
        let mut grid_piece: UnstructuredGridPiece = (&self.mesh).into();
        grid_piece
            .data
            .point
            .push(Attribute::scalars("density", 1).with_data(self.data.clone()));
        grid_piece
    }
}

#[cfg(feature = "vtk_extras")]
pub mod vtk_helper {
    use vtkio::model::{
        Attributes, CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers,
    };
    use vtkio::IOBuffer;

    use super::{HexMesh3d, PointCloud3d, Real, TriMesh3d};

    impl<R> From<&TriMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &TriMesh3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::new();
                points.reserve(mesh.vertices.len() * 3);
                for v in mesh.vertices.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::new();
                vertices.reserve(mesh.triangles.len() * (3 + 1));
                for triangle in mesh.triangles.iter() {
                    vertices.push(3);
                    vertices.extend(triangle.iter().copied().map(|i| i as u32));
                }
                vertices
            };

            let cell_types = vec![CellType::Triangle; mesh.triangles.len()];

            new_unstructured_grid_piece(points, vertices, cell_types)
        }
    }

    impl<'a, R> From<&'a HexMesh3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a HexMesh3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::new();
                points.reserve(mesh.vertices.len() * 3);
                for v in mesh.vertices.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::new();
                vertices.reserve(mesh.cells.len() * (8 + 1));
                for cell in mesh.cells.iter() {
                    vertices.push(8);
                    vertices.extend(cell.iter().copied().map(|i| i as u32));
                }
                vertices
            };

            let cell_types = vec![CellType::Hexahedron; mesh.cells.len()];

            new_unstructured_grid_piece(points, vertices, cell_types)
        }
    }

    impl<'a, R> From<&'a PointCloud3d<R>> for UnstructuredGridPiece
    where
        R: Real,
    {
        fn from(mesh: &'a PointCloud3d<R>) -> Self {
            let points = {
                let mut points: Vec<R> = Vec::new();
                points.reserve(mesh.points.len() * 3);
                for v in mesh.points.iter() {
                    points.extend(v.as_slice());
                }
                points
            };

            let vertices = {
                let mut vertices = Vec::new();
                vertices.reserve(mesh.points.len() * (1 + 1));
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

    impl<R: Real> Into<DataSet> for &TriMesh3d<R> {
        fn into(self) -> DataSet {
            DataSet::inline(UnstructuredGridPiece::from(self))
        }
    }

    impl<R: Real> Into<DataSet> for &HexMesh3d<R> {
        fn into(self) -> DataSet {
            DataSet::inline(UnstructuredGridPiece::from(self))
        }
    }

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
