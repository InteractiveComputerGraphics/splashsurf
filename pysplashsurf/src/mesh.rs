use ndarray::{Array2, ArrayView, ArrayView2};
use numpy::{Element, IntoPyArray, PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::{
    IntoPyObjectExt,
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::{
    Real,
    mesh::{
        AttributeData, Mesh3d, MeshAttribute, MeshWithData, MixedTriQuadMesh3d, TriMesh3d,
        TriangleOrQuadCell,
    },
    nalgebra::{Unit, Vector3},
};

use crate::aabb::{Aabb3dF32, Aabb3dF64};

fn get_attribute_with_name<'py, R: Real + Element>(
    py: Python<'py>,
    attrs: &[MeshAttribute<R>],
    name: &str,
) -> PyResult<PyObject>
where
    R: pyo3::IntoPyObject<'py>,
{
    let elem = attrs.iter().filter(|x| x.name == name).next();
    match elem {
        Some(attr) => match attr.data.clone() {
            AttributeData::ScalarU64(res) => Ok(res.into_pyobject(py).unwrap().into()),
            AttributeData::ScalarReal(res) => Ok(res.into_pyobject(py).unwrap().into()),
            AttributeData::Vector3Real(res) => {
                let flattened: Vec<R> = bytemuck::cast_vec(res);
                let res: Array2<R> =
                    Array2::from_shape_vec((flattened.len() / 3, 3), flattened).unwrap();
                Ok(res.into_pyarray(py).into_bound_py_any(py).unwrap().into())
            }
        },
        None => Err(PyErr::new::<PyValueError, _>(format!(
            "Attribute with name {} doesn't exist",
            name
        ))),
    }
}

fn add_attribute_with_name<'py, R: Real + Element>(
    attrs: &mut Vec<MeshAttribute<R>>,
    attribute: MeshAttribute<R>,
) -> PyResult<()> {
    let elem = attrs.iter().filter(|x| x.name == attribute.name).next();
    match elem {
        None => {
            attrs.push(attribute);
            Ok(())
        }
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "Attribute with name {} already exists",
            attribute.name
        ))),
    }
}

macro_rules! create_mesh_data_interface {
    ($name: ident, $type: ident, $mesh_class: ident, $pymesh_class: ident, $aabb_class: ident) => {
        /// MeshWithData wrapper
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: MeshWithData<$type, $mesh_class<$type>>,
        }

        impl $name {
            pub fn new(data: MeshWithData<$type, $mesh_class<$type>>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            fn py_new(mesh: &$pymesh_class) -> PyResult<Self> {
                let meshdata = MeshWithData::new(mesh.inner.clone());
                Ok($name::new(meshdata))
            }

            /// Clone of the contained mesh
            #[getter]
            fn mesh(&self) -> $pymesh_class {
                $pymesh_class::new(self.inner.mesh.clone())
            }

            /// Returns mesh without copying the mesh data, removes it from the object
            fn take_mesh(&mut self) -> $pymesh_class {
                let mesh = std::mem::take(&mut self.inner.mesh);
                $pymesh_class::new(mesh)
            }

            /// Removes all cells from the mesh that are completely outside of the given AABB and clamps the remaining cells to the boundary
            fn par_clamp_with_aabb(
                &self,
                aabb: &$aabb_class,
                clamp_vertices: bool,
                keep_vertices: bool,
            ) -> $name {
                $name::new(self.inner.par_clamp_with_aabb(
                    &aabb.inner,
                    clamp_vertices,
                    keep_vertices,
                ))
            }

            fn push_point_attribute_scalar_u64<'py>(
                &mut self,
                name: &str,
                data: Vec<u64>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.point_attributes,
                    MeshAttribute::new(name, AttributeData::ScalarU64(data)),
                )
            }

            fn push_point_attribute_scalar_real<'py>(
                &mut self,
                name: &str,
                data: Vec<$type>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.point_attributes,
                    MeshAttribute::new(name, AttributeData::ScalarReal(data)),
                )
            }

            fn push_point_attribute_vector_real<'py>(
                &mut self,
                name: &str,
                data: &Bound<'py, PyArray2<$type>>,
            ) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract().unwrap();
                let data = data.as_slice().unwrap();
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(
                    &mut self.inner.point_attributes,
                    MeshAttribute::new(name, AttributeData::Vector3Real(data.to_vec())),
                )
            }

            fn push_cell_attribute_scalar_u64<'py>(
                &mut self,
                name: &str,
                data: Vec<u64>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    MeshAttribute::new(name, AttributeData::ScalarU64(data)),
                )
            }

            fn push_cell_attribute_scalar_real<'py>(
                &mut self,
                name: &str,
                data: Vec<$type>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    MeshAttribute::new(name, AttributeData::ScalarReal(data)),
                )
            }

            fn push_cell_attribute_vector_real<'py>(
                &mut self,
                name: &str,
                data: &Bound<'py, PyArray2<$type>>,
            ) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract().unwrap();
                let data = data.as_slice().unwrap();
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    MeshAttribute::new(name, AttributeData::Vector3Real(data.to_vec())),
                )
            }

            /// Get mesh vertex attribute by name
            fn get_point_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.point_attributes.as_slice(), name)
            }

            /// Get mesh cell attribute by name
            fn get_cell_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.cell_attributes.as_slice(), name)
            }

            /// Get all point attributes in a python dictionary
            fn get_point_attributes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
                let res = PyDict::new(py);

                for attr in self.inner.point_attributes.iter() {
                    let data = get_attribute_with_name::<$type>(
                        py,
                        self.inner.point_attributes.as_slice(),
                        &attr.name,
                    );
                    match data {
                        Ok(data) => res.set_item(&attr.name, data).unwrap(),
                        Err(_) => println!("Couldn't embed attribute {} in PyDict", &attr.name),
                    }
                }

                res
            }

            /// Get all cell attributes in a python dictionary
            fn get_cell_attributes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
                let res = PyDict::new(py);

                for attr in self.inner.cell_attributes.iter() {
                    let data = get_attribute_with_name::<$type>(
                        py,
                        self.inner.cell_attributes.as_slice(),
                        &attr.name,
                    );
                    match data {
                        Ok(data) => res.set_item(&attr.name, data).unwrap(),
                        Err(_) => println!("Couldn't embed attribute {} in PyDict", &attr.name),
                    }
                }

                res
            }

            /// Get all registered point attribute names
            fn get_point_attribute_keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
                let mut res: Vec<&str> = vec![];

                for attr in self.inner.point_attributes.iter() {
                    res.push(&attr.name);
                }

                PyList::new(py, res).unwrap()
            }

            /// Get all registered cell attribute names
            fn get_cell_attribute_keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
                let mut res: Vec<&str> = vec![];

                for attr in self.inner.cell_attributes.iter() {
                    res.push(&attr.name);
                }

                PyList::new(py, res).unwrap()
            }
        }
    };
}

macro_rules! create_mesh_interface {
    ($name: ident, $type: ident) => {
        /// TriMesh3d wrapper
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: TriMesh3d<$type>,
        }

        impl $name {
            pub fn new(data: TriMesh3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            /// nx3 array of vertex positions, copies the data
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points).unwrap();
                vertices.to_pyarray(py) // seems like at least one copy is necessary here (to_pyarray copies the data)
            }

            /// nx3 array of the vertex indices that make up a triangle, copies the data
            #[getter]
            fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
                let tris: &[u64] = bytemuck::cast_slice(&self.inner.triangles);
                let triangles: ArrayView2<u64> =
                    ArrayView::from_shape((self.inner.triangles.len(), 3), tris).unwrap();
                triangles.to_pyarray(py)
            }

            /// Returns a tuple of vertices and triangles without copying the data, removes the data in the class
            fn take_vertices_and_triangles<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyTuple> {
                let vertices = std::mem::take(&mut self.inner.vertices);
                let triangles = std::mem::take(&mut self.inner.triangles);

                let n = vertices.len();
                let m = triangles.len();

                let vertices_scalar: Vec<$type> = bytemuck::cast_vec(vertices);
                let vertices_array = PyArray::from_vec(py, vertices_scalar)
                    .reshape([n, 3])
                    .unwrap();

                let triangles_scalar: Vec<usize> = bytemuck::cast_vec(triangles);
                let triangles_array = PyArray::from_vec(py, triangles_scalar)
                    .reshape([m, 3])
                    .unwrap();

                let tup = (vertices_array, triangles_array);
                tup.into_pyobject(py).unwrap()
            }

            /// Computes the mesh's vertex normals using an area weighted average of the adjacent triangle faces (parallelized version)
            fn par_vertex_normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let normals_vec = self.inner.par_vertex_normals();
                let normals_vec =
                    bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals).unwrap();

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
        #[gen_stub_pyclass]
        #[pyclass]
        pub struct $name {
            pub inner: MixedTriQuadMesh3d<$type>,
        }

        impl $name {
            pub fn new(data: MixedTriQuadMesh3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            /// nx3 array of vertex positions, copies data
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points).unwrap();
                vertices.to_pyarray(py)
            }

            /// 2D list specifying the vertex indices either for a triangle or a quad
            #[getter]
            fn cells(&self) -> Vec<Vec<usize>> {
                let cells: Vec<Vec<usize>> = self
                    .inner
                    .cells
                    .iter()
                    .map(|c| match c {
                        TriangleOrQuadCell::Tri(v) => v.to_vec(),
                        TriangleOrQuadCell::Quad(v) => v.to_vec(),
                    })
                    .collect();
                cells
            }

            /// Returns a tuple of vertices and triangles without copying the data, removes the data in the class
            fn take_vertices_and_cells<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyTuple> {
                let vertices = std::mem::take(&mut self.inner.vertices);
                let cells = std::mem::take(&mut self.inner.cells);

                let n = vertices.len();

                let vertices_scalar: Vec<$type> = bytemuck::cast_vec(vertices);
                let vertices_array = PyArray::from_vec(py, vertices_scalar)
                    .reshape([n, 3])
                    .unwrap();

                let cells_list: Vec<Vec<usize>> = cells
                    .into_iter()
                    .map(|c| match c {
                        TriangleOrQuadCell::Tri(v) => v.into(),
                        TriangleOrQuadCell::Quad(v) => v.into(),
                    })
                    .collect();

                let tup = (vertices_array, cells_list);
                tup.into_pyobject(py).unwrap()
            }
        }
    };
}

create_mesh_interface!(TriMesh3dF64, f64);
create_mesh_interface!(TriMesh3dF32, f32);

create_tri_quad_mesh_interface!(MixedTriQuadMesh3dF64, f64);
create_tri_quad_mesh_interface!(MixedTriQuadMesh3dF32, f32);

create_mesh_data_interface!(TriMeshWithDataF64, f64, TriMesh3d, TriMesh3dF64, Aabb3dF64);
create_mesh_data_interface!(TriMeshWithDataF32, f32, TriMesh3d, TriMesh3dF32, Aabb3dF32);

create_mesh_data_interface!(
    MixedTriQuadMeshWithDataF64,
    f64,
    MixedTriQuadMesh3d,
    MixedTriQuadMesh3dF64,
    Aabb3dF64
);
create_mesh_data_interface!(
    MixedTriQuadMeshWithDataF32,
    f32,
    MixedTriQuadMesh3d,
    MixedTriQuadMesh3dF32,
    Aabb3dF32
);
