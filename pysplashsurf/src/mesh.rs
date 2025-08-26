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
        Mesh3d, MeshWithData, MixedTriQuadMesh3d, OwnedAttributeData, OwnedMeshAttribute,
        TriMesh3d, TriangleOrQuadCell,
    },
    nalgebra::{Unit, Vector3},
};

use crate::aabb::{Aabb3dF32, Aabb3dF64};

fn get_attribute_with_name<'py, R: Real + Element>(
    py: Python<'py>,
    attrs: &[OwnedMeshAttribute<R>],
    name: &str,
) -> PyResult<PyObject>
where
    R: pyo3::IntoPyObject<'py>,
{
    let elem = attrs.iter().filter(|x| x.name == name).next();
    match elem {
        Some(attr) => match attr.data.clone() {
            OwnedAttributeData::ScalarU64(res) => Ok(res.into_owned().into_pyobject(py)?.into()),
            OwnedAttributeData::ScalarReal(res) => Ok(res.into_owned().into_pyobject(py)?.into()),
            OwnedAttributeData::Vector3Real(res) => {
                let flattened: Vec<R> = bytemuck::cast_vec(res.into_owned());
                let res: Array2<R> = Array2::from_shape_vec((flattened.len() / 3, 3), flattened)
                    .map_err(anyhow::Error::new)?;
                Ok(res.into_pyarray(py).into_bound_py_any(py)?.into())
            }
        },
        None => Err(PyErr::new::<PyValueError, _>(format!(
            "Attribute with name {} doesn't exist",
            name
        ))),
    }
}

fn add_attribute_with_name<'py, R: Real + Element>(
    attrs: &mut Vec<OwnedMeshAttribute<R>>,
    attribute: OwnedMeshAttribute<R>,
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

            /// Returns a copy of the contained mesh
            fn get_mesh(&self) -> $pymesh_class {
                $pymesh_class::new(self.inner.mesh.clone())
            }

            /// Returns the contained mesh by moving it out of this object (zero copy)
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
                    OwnedMeshAttribute::new(name, OwnedAttributeData::ScalarU64(data.into())),
                )
            }

            fn push_point_attribute_scalar_real<'py>(
                &mut self,
                name: &str,
                data: Vec<$type>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.point_attributes,
                    OwnedMeshAttribute::new(name, OwnedAttributeData::ScalarReal(data.into())),
                )
            }

            fn push_point_attribute_vector_real<'py>(
                &mut self,
                name: &str,
                data: &Bound<'py, PyArray2<$type>>,
            ) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract()?;
                let data = data.as_slice()?;
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(
                    &mut self.inner.point_attributes,
                    OwnedMeshAttribute::new(
                        name,
                        OwnedAttributeData::Vector3Real(data.to_vec().into()),
                    ),
                )
            }

            fn push_cell_attribute_scalar_u64<'py>(
                &mut self,
                name: &str,
                data: Vec<u64>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    OwnedMeshAttribute::new(name, OwnedAttributeData::ScalarU64(data.into())),
                )
            }

            fn push_cell_attribute_scalar_real<'py>(
                &mut self,
                name: &str,
                data: Vec<$type>,
            ) -> PyResult<()> {
                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    OwnedMeshAttribute::new(name, OwnedAttributeData::ScalarReal(data.into())),
                )
            }

            fn push_cell_attribute_vector_real<'py>(
                &mut self,
                name: &str,
                data: &Bound<'py, PyArray2<$type>>,
            ) -> PyResult<()> {
                let data: PyReadonlyArray2<$type> = data.extract()?;
                let data = data.as_slice()?;
                let data: &[Vector3<$type>] = bytemuck::cast_slice(data);

                add_attribute_with_name::<$type>(
                    &mut self.inner.cell_attributes,
                    OwnedMeshAttribute::new(
                        name,
                        OwnedAttributeData::Vector3Real(data.to_vec().into()),
                    ),
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
            fn get_point_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let res = PyDict::new(py);

                for attr in self.inner.point_attributes.iter() {
                    let data = get_attribute_with_name::<$type>(
                        py,
                        self.inner.point_attributes.as_slice(),
                        &attr.name,
                    );
                    match data {
                        Ok(data) => res.set_item(&attr.name, data)?,
                        Err(_) => println!("Couldn't embed attribute {} in PyDict", &attr.name),
                    }
                }

                Ok(res)
            }

            /// Get all cell attributes in a python dictionary
            fn get_cell_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let res = PyDict::new(py);

                for attr in self.inner.cell_attributes.iter() {
                    let data = get_attribute_with_name::<$type>(
                        py,
                        self.inner.cell_attributes.as_slice(),
                        &attr.name,
                    );
                    match data {
                        Ok(data) => res.set_item(&attr.name, data)?,
                        Err(_) => println!("Couldn't embed attribute {} in PyDict", &attr.name),
                    }
                }

                Ok(res)
            }

            /// Get all registered point attribute names
            fn get_point_attribute_keys<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyList>> {
                let mut res: Vec<&str> = vec![];

                for attr in self.inner.point_attributes.iter() {
                    res.push(&attr.name);
                }

                PyList::new(py, res)
            }

            /// Get all registered cell attribute names
            fn get_cell_attribute_keys<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyList>> {
                let mut res: Vec<&str> = vec![];

                for attr in self.inner.cell_attributes.iter() {
                    res.push(&attr.name);
                }

                PyList::new(py, res)
            }
        }
    };
}

macro_rules! create_tri_mesh_interface {
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
            /// Returns a copy of the `Nx3` array of vertex positions
            fn get_vertices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points)
                        .map_err(anyhow::Error::new)?;
                Ok(vertices.to_pyarray(py)) // seems like at least one copy is necessary here (to_pyarray copies the data)
            }

            /// Returns a copy of the `Mx3` array of the vertex indices that make up a triangle
            fn get_triangles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
                let tris: &[u64] = bytemuck::cast_slice(&self.inner.triangles);
                let triangles: ArrayView2<u64> =
                    ArrayView::from_shape((self.inner.triangles.len(), 3), tris)
                        .map_err(anyhow::Error::new)?;
                Ok(triangles.to_pyarray(py))
            }

            /// Alias for `get_triangles`
            fn get_cells<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
                self.get_triangles(py)
            }

            /// Returns the `Nx3` array of vertex positions by moving it out of the mesh (zero copy)
            fn take_vertices<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let vertices = std::mem::take(&mut self.inner.vertices);
                let n = vertices.len();
                let vertices_scalar: Vec<$type> = bytemuck::cast_vec(vertices);
                let vertices_array = PyArray::from_vec(py, vertices_scalar)
                    .reshape([n, 3])
                    .map_err(anyhow::Error::new)?;
                Ok(vertices_array)
            }

            /// Returns the `Mx3` array of the vertex indices that make up the triangles by moving it out of the mesh (zero copy)
            fn take_triangles<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<u64>>> {
                let triangles = std::mem::take(&mut self.inner.triangles);
                let m = triangles.len();
                let triangles_scalar: Vec<u64> = bytemuck::cast_vec(triangles);
                let triangles_array = PyArray::from_vec(py, triangles_scalar)
                    .reshape([m, 3])
                    .map_err(anyhow::Error::new)?;
                Ok(triangles_array)
            }

            /// Alias for `take_triangles`
            fn take_cells<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
                self.take_triangles(py)
            }

            /// Returns a tuple containing the vertices and triangles of the mesh by moving them out of the mesh (zero copy)
            fn take_vertices_and_triangles<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyTuple>> {
                let tup = (self.take_vertices(py)?, self.take_triangles(py)?);
                tup.into_pyobject(py)
            }

            /// Computes the mesh's vertex normals using an area weighted average of the adjacent triangle faces (parallelized version)
            fn par_vertex_normals<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let normals_vec = self.inner.par_vertex_normals();
                let normals_vec =
                    bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals)
                        .map_err(anyhow::Error::new)?;

                Ok(normals.to_pyarray(py))
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
            /// Returns a copy of the `Nx3` array of vertex positions
            fn get_vertices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points)
                        .map_err(anyhow::Error::new)?;
                Ok(vertices.to_pyarray(py))
            }

            /// Returns a 2D list specifying the vertex indices either for a triangle or a quad
            fn get_cells(&self) -> PyResult<Vec<Vec<usize>>> {
                let cells: Vec<Vec<usize>> = self
                    .inner
                    .cells
                    .iter()
                    .map(|c| match c {
                        TriangleOrQuadCell::Tri(v) => v.to_vec(),
                        TriangleOrQuadCell::Quad(v) => v.to_vec(),
                    })
                    .collect();
                Ok(cells)
            }

            /// Returns the `Nx3` array of vertex positions by moving it out of the mesh (zero copy)
            fn take_vertices<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<$type>>> {
                let vertices = std::mem::take(&mut self.inner.vertices);
                let n = vertices.len();
                let vertices_scalar: Vec<$type> = bytemuck::cast_vec(vertices);
                let vertices_array = PyArray::from_vec(py, vertices_scalar)
                    .reshape([n, 3])
                    .map_err(anyhow::Error::new)?;
                Ok(vertices_array)
            }
        }
    };
}

create_tri_mesh_interface!(TriMesh3dF64, f64);
create_tri_mesh_interface!(TriMesh3dF32, f32);

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
