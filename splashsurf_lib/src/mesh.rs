use std::fmt::Debug;

use nalgebra::Vector3;
use vtkio::model::{Attribute, Attributes, CellType, Cells, DataSet};

use crate::Real;

/// A triangle (surface) mesh in 3D
#[derive(Clone, Debug)]
pub struct TriMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The triangles of the mesh identified by their vertex indices
    pub triangles: Vec<[usize; 3]>,
}

/// A hexahedral (volumetric) mesh in 3D
#[derive(Clone, Debug)]
pub struct HexMesh3d<R: Real> {
    /// Coordinates of all vertices of the mesh
    pub vertices: Vec<Vector3<R>>,
    /// The hexahedral cells of the mesh identified by their vertex indices
    pub cells: Vec<[usize; 8]>,
}

/// A point cloud in 3D
#[derive(Clone, Debug)]
pub struct PointCloud3d<R: Real> {
    /// Coordinates of all points in the point cloud
    pub points: Vec<Vector3<R>>,
}

/// A mesh with attached vertex or point data
#[derive(Clone, Debug)]
pub struct MeshWithPointData<MeshT, DataT> {
    /// The mesh geometry itself
    pub mesh: MeshT,
    /// Data attached to each vertex or point of the mesh
    pub data: Vec<DataT>,
}

impl<'a, MeshT: 'a, DataT> MeshWithPointData<MeshT, DataT>
where
    DataSet: From<&'a MeshT>,
    DataT: Real,
{
    pub fn to_dataset(&'a self) -> DataSet {
        let mut mesh_dataset = DataSet::from(&self.mesh);

        if let DataSet::UnstructuredGrid { ref mut data, .. } = &mut mesh_dataset {
            let attribute = Attribute::Scalars {
                num_comp: 1,
                lookup_table: None,
                data: self.data.clone().into(),
            };

            data.point.push((format!("density"), attribute));
        }

        mesh_dataset
    }
}

impl<'a, R> From<&'a TriMesh3d<R>> for DataSet
where
    R: Real,
{
    fn from(mesh: &'a TriMesh3d<R>) -> Self {
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

        DataSet::UnstructuredGrid {
            points: points.into(),
            cells: Cells {
                num_cells: mesh.triangles.len() as u32,
                vertices,
            },
            cell_types,
            data: Attributes::new(),
        }
    }
}

impl<'a, R> From<&'a HexMesh3d<R>> for DataSet
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

        DataSet::UnstructuredGrid {
            points: points.into(),
            cells: Cells {
                num_cells: mesh.cells.len() as u32,
                vertices,
            },
            cell_types,
            data: Attributes::new(),
        }
    }
}

impl<'a, R> From<&'a PointCloud3d<R>> for DataSet
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

        DataSet::UnstructuredGrid {
            points: points.into(),
            cells: Cells {
                num_cells: mesh.points.len() as u32,
                vertices,
            },
            cell_types,
            data: Attributes::new(),
        }
    }
}
