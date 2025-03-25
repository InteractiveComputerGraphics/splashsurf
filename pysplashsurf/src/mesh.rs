use ndarray::{Array2, ArrayView2, ArrayView};
use numpy::{Element, IntoPyArray, PyReadonlyArray2, PyArray2, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*, IntoPyObjectExt};
use splashsurf_lib::{mesh::{AttributeData, MeshAttribute, MeshWithData, TriMesh3d, Mesh3d, MixedTriQuadMesh3d, TriangleOrQuadCell}, Real, nalgebra::{Vector3, Unit}};

use crate::aabb::{PyAabb3dF32, PyAabb3dF64};

fn get_attribute_with_name<'py, R: Real + Element>(py: Python<'py>, attrs: &[MeshAttribute<R>], name: &str) -> PyResult<PyObject> where R: pyo3::IntoPyObject<'py> {
    let elem = attrs.iter().filter(|x| x.name == name).next();
    match elem {
        Some(attr) => match attr.data.clone() {
            AttributeData::ScalarU64(res) => Ok(res.into_pyobject(py).unwrap().into()),
            AttributeData::ScalarReal(res) => Ok(res.into_pyobject(py).unwrap().into()),
            AttributeData::Vector3Real(res) => {
                let flattened: Vec<R> = bytemuck::cast_vec(res);
                let res: Array2<R> =
                    Array2::from_shape_vec((flattened.len()/3, 3), flattened)
                        .unwrap();
                Ok(res.into_pyarray(py).into_bound_py_any(py).unwrap().into())
            },
        },
        None => Err(PyErr::new::<PyValueError, _>(format!("Attribute with name {} doesn't exist", name)))
    }
}

fn add_attribute_with_name<'py, R: Real + Element>(attrs: &mut Vec<MeshAttribute<R>>, attribute: MeshAttribute<R>) -> PyResult<()> {
    let elem = attrs.iter().filter(|x| x.name == attribute.name).next();
    match elem {
        None => {
            attrs.push(attribute);
            Ok(())
        },
        _ => Err(PyErr::new::<PyValueError, _>(format!("Attribute with name {} already exists", attribute.name)))
    }
}
 
macro_rules! create_mesh_data_interface {
    ($name: ident, $type: ident, $mesh_class: ident, $aabb_class: ident) => {
        /// MeshWithData wrapper
        #[pyclass]
        pub struct $name {
            pub inner: MeshWithData<$type, TriMesh3d<$type>>,
        }

        impl $name {
            pub fn new(data: MeshWithData<$type, TriMesh3d<$type>>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {

            #[new]
            fn py_new(mesh: &$mesh_class) -> PyResult<Self> {
                let meshdata = MeshWithData::new(mesh.inner.clone());
                Ok($name::new(meshdata))
            }

            /// PyTriMesh3dF32/F64 clone of the contained mesh
            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh.clone())
            }

            /// Removes all cells from the mesh that are completely outside of the given AABB and clamps the remaining cells to the boundary
            fn par_clamp_with_aabb(&self, aabb: &$aabb_class, clamp_vertices: bool, keep_vertices: bool) -> $name {
                $name::new(self.inner.par_clamp_with_aabb(&aabb.inner, clamp_vertices, keep_vertices))
            }

            fn push_point_attribute_scalar_u64<'py>(&mut self, name: &str, data: Vec<u64>) -> PyResult<()> {
                add_attribute_with_name::<$type>(&mut self.inner.point_attributes, MeshAttribute::new(name, AttributeData::ScalarU64(data)))
            }

            fn push_point_attribute_scalar_real<'py>(&mut self, name: &str, data: Vec<$type>) -> PyResult<()> {
                add_attribute_with_name::<$type>(&mut self.inner.point_attributes, MeshAttribute::new(name, AttributeData::ScalarReal(data)))
            }

            fn push_point_attribute_vector_real<'py>(&mut self, name: &str, data: &Bound<'py, PyArray2<$type>>) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract().unwrap();
                let data = data.as_slice().unwrap();
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(&mut self.inner.point_attributes, MeshAttribute::new(name, AttributeData::Vector3Real(data.to_vec())))
            }

            fn push_cell_attribute_scalar_u64<'py>(&mut self, name: &str, data: Vec<u64>) -> PyResult<()> {
                add_attribute_with_name::<$type>(&mut self.inner.cell_attributes, MeshAttribute::new(name, AttributeData::ScalarU64(data)))
            }

            fn push_cell_attribute_scalar_real<'py>(&mut self, name: &str, data: Vec<$type>) -> PyResult<()> {
                add_attribute_with_name::<$type>(&mut self.inner.cell_attributes, MeshAttribute::new(name, AttributeData::ScalarReal(data)))
            }

            fn push_cell_attribute_vector_real<'py>(&mut self, name: &str, data: &Bound<'py, PyArray2<$type>>) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract().unwrap();
                let data = data.as_slice().unwrap();
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(&mut self.inner.cell_attributes, MeshAttribute::new(name, AttributeData::Vector3Real(data.to_vec())))
            }

            /// Get mesh vertex attribute by name
            fn get_point_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.point_attributes.as_slice(), name)
            }

            /// Get mesh cell attribute by name
            fn get_cell_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.cell_attributes.as_slice(), name)
            }
        }
    };
}

macro_rules! create_mesh_interface {
    ($name: ident, $type: ident) => {
        /// TriMesh3d wrapper
        #[pyclass]
        pub struct $name {
            pub inner: TriMesh3d<$type>,
        }

        impl $name {
            pub fn new(data: TriMesh3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            /// nx3 array of vertex positions
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points)
                        .unwrap();
                vertices.to_pyarray(py) // seems like at least one copy is necessary here (to_pyarray copies the data)
            }

            /// nx3 array of the vertex indices that make up a triangle
            #[getter]
            fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
                let tris: &[usize] = bytemuck::cast_slice(&self.inner.triangles);
                let triangles: ArrayView2<usize> =
                    ArrayView::from_shape((self.inner.triangles.len(), 3), tris).unwrap();
                triangles.to_pyarray(py)
            }

            /// Computes the mesh's vertex normals using an area weighted average of the adjacent triangle faces (parallelized version)
            fn par_vertex_normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let normals_vec = self.inner.par_vertex_normals();
                let normals_vec = bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);
                //let normals_vec: Vec<$type> = bytemuck::cas

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals)
                        .unwrap();
                
                normals.to_pyarray(py)
            }

            /// Returns a mapping of all mesh vertices to the set of their connected neighbor vertices
            fn vertex_vertex_connectivity(&self) -> Vec<Vec<usize>> {
                self.inner.vertex_vertex_connectivity()
            }
        }
    };
}

macro_rules! create_tri_quad_mesh_interface {
    ($name: ident, $type: ident) => {
        /// MixedTriQuadMesh3d wrapper
        #[pyclass]
        pub struct $name {
            pub inner: MixedTriQuadMesh3d<$type>,
        }

        impl $name {
            pub fn new(data: MixedTriQuadMesh3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            /// nx3 array of vertex positions
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points)
                        .unwrap();
                vertices.to_pyarray(py)
            }

            /// 2D list specifying the vertex indices either for a triangle or a quad
            #[getter]
            fn cells(&self) -> Vec<Vec<usize>> {
                let cells: Vec<Vec<usize>> = self.inner.cells.iter().map(|c| match c {
                    TriangleOrQuadCell::Tri(v) => v.to_vec(),
                    TriangleOrQuadCell::Quad(v) => v.to_vec(),
                }).collect();
                cells
            }
        }
    };
}

create_mesh_interface!(PyTriMesh3dF64, f64);
create_mesh_interface!(PyTriMesh3dF32, f32);

create_tri_quad_mesh_interface!(PyMixedTriQuadMesh3dF64, f64);
create_tri_quad_mesh_interface!(PyMixedTriQuadMesh3dF32, f32);

create_mesh_data_interface!(PyMeshWithDataF64, f64, PyTriMesh3dF64, PyAabb3dF64);
create_mesh_data_interface!(PyMeshWithDataF32, f32, PyTriMesh3dF32, PyAabb3dF32);