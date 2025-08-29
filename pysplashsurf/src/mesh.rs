use crate::NumpyUsize;
use crate::aabb::{Aabb3dF32, Aabb3dF64};
use crate::utils::*;
use bytemuck::{NoUninit, Pod};
use ndarray::{Array2, ArrayView, ArrayView1, ArrayView2};
use numpy as np;
use numpy::{
    Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayMethods,
    PyReadonlyArray2, PyUntypedArray, ToPyArray,
};
use pyo3::exceptions::PyTypeError;
use pyo3::{
    IntoPyObjectExt,
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;
use splashsurf_lib::mesh::TriangleCell;
use splashsurf_lib::{
    Real,
    mesh::{
        Mesh3d, MeshWithData, MixedTriQuadMesh3d, OwnedAttributeData, OwnedMeshAttribute,
        TriMesh3d, TriangleOrQuadCell,
    },
    nalgebra::{Unit, Vector3},
};

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

fn get_vec_generic<'py, R: Element>(
    values: &[R],
    shape: (usize, usize),
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    assert_eq!(
        shape.0 * shape.1,
        values.len(),
        "shape does not match values length"
    );
    if shape.1 == 1 {
        let array: ArrayView1<R> =
            ArrayView::from_shape((values.len(),), values).map_err(anyhow::Error::new)?;
        let pyarray = unsafe { PyArray1::borrow_from_array(&array, container) };
        Ok(pyarray
            .into_any()
            .downcast_into::<PyUntypedArray>()
            .expect("downcast should not fail"))
    } else {
        let array: ArrayView2<R> =
            ArrayView::from_shape(shape, values).map_err(anyhow::Error::new)?;
        let pyarray = unsafe { PyArray2::borrow_from_array(&array, container) };
        Ok(pyarray
            .into_any()
            .downcast_into::<PyUntypedArray>()
            .expect("downcast should not fail"))
    }
}

fn get_scalar_generic<'py, R: Element>(
    values: &[R],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    get_vec_generic(values, (values.len(), 1), container)
}

fn get_vec3f_generic<'py, R: Real + Element>(
    values: &[Vector3<R>],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let coordinates: &[R] = bytemuck::cast_slice(values);
    get_vec_generic(coordinates, (values.len(), 3), container)
}

fn get_triangles_generic<'py>(
    triangles: &[TriangleCell],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
    let vertex_indices: &[NumpyUsize] = bytemuck::cast_slice(triangles);
    let view = get_vec_generic(vertex_indices, (triangles.len(), 3), container)?.into_any();
    Ok(view.downcast_into::<PyArray2<NumpyUsize>>()?)
}

fn compute_normals_generic<'py, R: Real + Element>(
    py: Python<'py>,
    mesh: &TriMesh3d<R>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let normals_vec = mesh.par_vertex_normals();
    let normals_vec = bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, R>(normals_vec);

    Ok(PyArray::from_vec(py, normals_vec)
        .reshape([mesh.vertices().len(), 3])?
        .into_any()
        .downcast_into::<PyUntypedArray>()
        .expect("downcast should not fail"))
}

pub fn get_triangle_mesh_generic<'py>(mesh: &Bound<'py, PyAny>) -> Option<Py<PyTriMesh3d>> {
    let py = mesh.py();
    if let Ok(mesh) = mesh.downcast::<PyTriMesh3d>() {
        Some(mesh.as_unbound().clone_ref(py))
    } else if let Ok(data_mesh) = mesh.downcast::<PyMeshWithData>()
        && data_mesh.borrow().mesh_type() == MeshType::Tri3d
    {
        data_mesh.borrow().as_tri3d(py)
    } else {
        None
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
            fn get_triangles<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
                let tris: &[NumpyUsize] = bytemuck::cast_slice(&self.inner.triangles);
                let triangles: ArrayView2<NumpyUsize> =
                    ArrayView::from_shape((self.inner.triangles.len(), 3), tris)
                        .map_err(anyhow::Error::new)?;
                Ok(triangles.to_pyarray(py))
            }

            /// Alias for `get_triangles`
            fn get_cells<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
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
            ) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
                let triangles = std::mem::take(&mut self.inner.triangles);
                let m = triangles.len();
                let triangles_scalar: Vec<NumpyUsize> = bytemuck::cast_vec(triangles);
                let triangles_array = PyArray::from_vec(py, triangles_scalar)
                    .reshape([m, 3])
                    .map_err(anyhow::Error::new)?;
                Ok(triangles_array)
            }

            /// Alias for `take_triangles`
            fn take_cells<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
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

/// Vertex-vertex connectivity of a mesh
#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "VertexVertexConnectivity")]
pub struct PyVertexVertexConnectivity {
    pub(crate) connectivity: Vec<Vec<usize>>,
}

impl PyVertexVertexConnectivity {
    pub fn new(connectivity: Vec<Vec<usize>>) -> Self {
        Self { connectivity }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVertexVertexConnectivity {
    /// Returns a copy of the contained connectivity data
    pub fn copy_connectivity(&self) -> Vec<Vec<usize>> {
        self.connectivity.clone()
    }

    /// Returns the contained connectivity data by moving it out of this object (zero copy)
    pub fn take_connectivity(&mut self) -> Vec<Vec<usize>> {
        // TODO: Check if this is actually zero-copy with the conversion to Python lists
        std::mem::take(&mut self.connectivity)
    }
}

#[derive(Clone)]
enum PyTriMesh3dData {
    F32(TriMesh3d<f32>),
    F64(TriMesh3d<f64>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "TriMesh3d")]
#[derive(Clone)]
pub struct PyTriMesh3d {
    inner: PyTriMesh3dData,
}

enum_wrapper_impl_from!(PyTriMesh3d, TriMesh3d<f32> => PyTriMesh3dData::F32);
enum_wrapper_impl_from!(PyTriMesh3d, TriMesh3d<f64> => PyTriMesh3dData::F64);

impl Default for PyTriMesh3d {
    fn default() -> Self {
        Self {
            inner: PyTriMesh3dData::F32(TriMesh3d::default()),
        }
    }
}

impl PyTriMesh3d {
    pub fn try_from_generic<R: Real + Element>(mut mesh: TriMesh3d<R>) -> PyResult<Self> {
        transmute_take_into::<_, TriMesh3d<f32>, _>(&mut mesh)
            .or_else(|| transmute_take_into::<_, TriMesh3d<f64>, _>(&mut mesh))
            .ok_or_else(|| {
                PyTypeError::new_err(
                    "Unsupported scalar type for TriMesh3d. Only f32 and f64 are supported.",
                )
            })
    }

    pub fn as_f32(&self) -> Option<&TriMesh3d<f32>> {
        match &self.inner {
            PyTriMesh3dData::F32(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&TriMesh3d<f64>> {
        match &self.inner {
            PyTriMesh3dData::F64(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut TriMesh3d<f32>> {
        match &mut self.inner {
            PyTriMesh3dData::F32(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_f64_mut(&mut self) -> Option<&mut TriMesh3d<f64>> {
        match &mut self.inner {
            PyTriMesh3dData::F64(mesh) => Some(mesh),
            _ => None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTriMesh3d {
    /// Numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            PyTriMesh3dData::F32(_) => np::dtype::<f32>(py),
            PyTriMesh3dData::F64(_) => np::dtype::<f64>(py),
        }
    }

    /// The `Nx3` array of vertex positions of the mesh
    #[getter]
    pub fn vertices<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &this.borrow().inner {
            PyTriMesh3dData::F32(mesh) => get_vec3f_generic(mesh.vertices(), this.into_any()),
            PyTriMesh3dData::F64(mesh) => get_vec3f_generic(mesh.vertices(), this.into_any()),
        }
    }

    /// The `Mx3` array of vertex indices per triangle
    #[getter]
    pub fn triangles<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
        match &this.borrow().inner {
            PyTriMesh3dData::F32(mesh) => get_triangles_generic(mesh.cells(), this.into_any()),
            PyTriMesh3dData::F64(mesh) => get_triangles_generic(mesh.cells(), this.into_any()),
        }
    }

    /// Computes the vertex normals of the mesh using an area weighted average of the adjacent triangle faces
    pub fn vertex_normals_parallel<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &self.inner {
            PyTriMesh3dData::F32(mesh) => compute_normals_generic(py, mesh),
            PyTriMesh3dData::F64(mesh) => compute_normals_generic(py, mesh),
        }
    }

    /// Computes the vertex-vertex connectivity of the mesh
    pub fn vertex_vertex_connectivity(&self) -> PyVertexVertexConnectivity {
        let connectivity = match &self.inner {
            PyTriMesh3dData::F32(mesh) => mesh.vertex_vertex_connectivity(),
            PyTriMesh3dData::F64(mesh) => mesh.vertex_vertex_connectivity(),
        };
        PyVertexVertexConnectivity::new(connectivity)
    }
}

#[derive(Clone)]
enum PyMixedTriQuadMesh3dData {
    F32(MixedTriQuadMesh3d<f32>),
    F64(MixedTriQuadMesh3d<f64>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "MixedTriQuadMesh3d")]
#[derive(Clone)]
pub struct PyMixedTriQuadMesh3d {
    inner: PyMixedTriQuadMesh3dData,
}

enum_wrapper_impl_from!(PyMixedTriQuadMesh3d, MixedTriQuadMesh3d<f32> => PyMixedTriQuadMesh3dData::F32);
enum_wrapper_impl_from!(PyMixedTriQuadMesh3d, MixedTriQuadMesh3d<f64> => PyMixedTriQuadMesh3dData::F64);

impl PyMixedTriQuadMesh3d {
    pub fn try_from_generic<R: Real + Element>(mut mesh: MixedTriQuadMesh3d<R>) -> PyResult<Self> {
        transmute_take_into::<_, MixedTriQuadMesh3d<f32>, _>(&mut mesh)
            .or_else(|| transmute_take_into::<_, MixedTriQuadMesh3d<f64>, _>(&mut mesh))
            .ok_or_else(|| {
                PyTypeError::new_err(
                    "Unsupported scalar type for MixedTriQuadMesh3d. Only f32 and f64 are supported.",
                )
            })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMixedTriQuadMesh3d {
    /// Numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            PyMixedTriQuadMesh3dData::F32(_) => np::dtype::<f32>(py),
            PyMixedTriQuadMesh3dData::F64(_) => np::dtype::<f64>(py),
        }
    }

    /// The `Nx3` array of vertex positions of the mesh
    #[getter]
    pub fn vertices<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &this.borrow().inner {
            PyMixedTriQuadMesh3dData::F32(mesh) => {
                get_vec3f_generic(mesh.vertices(), this.into_any())
            }
            PyMixedTriQuadMesh3dData::F64(mesh) => {
                get_vec3f_generic(mesh.vertices(), this.into_any())
            }
        }
    }

    /// Returns a copy of all triangle cells of the mesh as an `Nx3` array of vertex indices
    pub fn get_triangles<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
        let cells = match &self.inner {
            PyMixedTriQuadMesh3dData::F32(mesh) => mesh.cells.as_slice(),
            PyMixedTriQuadMesh3dData::F64(mesh) => mesh.cells.as_slice(),
        };

        filter_cells(py, cells, |cell| match cell {
            TriangleOrQuadCell::Tri(tri) => Some(tri.map(|v| v as NumpyUsize)),
            _ => None,
        })
    }

    /// Returns a copy of all quad cells of the mesh as an `Nx3` array of vertex indices
    pub fn get_quads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
        let cells = match &self.inner {
            PyMixedTriQuadMesh3dData::F32(mesh) => mesh.cells.as_slice(),
            PyMixedTriQuadMesh3dData::F64(mesh) => mesh.cells.as_slice(),
        };

        filter_cells(py, cells, |cell| match cell {
            TriangleOrQuadCell::Quad(quad) => Some(quad.map(|v| v as NumpyUsize)),
            _ => None,
        })
    }
}

pub fn filter_cells<'py, C, const N: usize, F>(
    py: Python<'py>,
    cells: &[C],
    filter: F,
) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>>
where
    [NumpyUsize; N]: Pod + NoUninit,
    F: Fn(&C) -> Option<[NumpyUsize; N]>,
{
    let filtered_cells: Vec<[NumpyUsize; N]> = cells.iter().filter_map(filter).collect();
    let n_triangles = filtered_cells.len();
    let vertex_indices: Vec<NumpyUsize> = bytemuck::cast_vec(filtered_cells);
    let array: Array2<NumpyUsize> =
        Array2::from_shape_vec((n_triangles, N), vertex_indices).map_err(anyhow::Error::new)?;
    let pyarray = array.into_pyarray(py);
    Ok(pyarray)
}

/// Enum specifying the type of mesh contained in a `MeshWithData`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[gen_stub_pyclass_enum]
#[pyclass]
pub enum MeshType {
    /// 3D triangle mesh
    Tri3d,
    /// 3D mixed triangle and quad mesh
    MixedTriQuad3d,
}

pub enum PyMesh3dData {
    Tri3d(Py<PyTriMesh3d>),
    MixedTriQuad3d(Py<PyMixedTriQuadMesh3d>),
}

enum_impl_from!(PyMesh3dData, Py<PyTriMesh3d> => PyMesh3dData::Tri3d);
enum_impl_from!(PyMesh3dData, Py<PyMixedTriQuadMesh3d> => PyMesh3dData::MixedTriQuad3d);

#[derive(Clone)]
enum PyMeshAttributeData {
    F32(OwnedMeshAttribute<f32>),
    F64(OwnedMeshAttribute<f64>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "MeshAttribute")]
#[derive(Clone)]
pub struct PyMeshAttribute {
    inner: PyMeshAttributeData,
}

enum_wrapper_impl_from!(PyMeshAttribute, OwnedMeshAttribute<f32> => PyMeshAttributeData::F32);
enum_wrapper_impl_from!(PyMeshAttribute, OwnedMeshAttribute<f64> => PyMeshAttributeData::F64);

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshAttribute {
    /// Numpy dtype of the data stored in the attribute
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            PyMeshAttributeData::F32(attr) => match attr.data {
                OwnedAttributeData::ScalarU64(_) => np::dtype::<u64>(py),
                OwnedAttributeData::ScalarReal(_) => np::dtype::<f32>(py),
                OwnedAttributeData::Vector3Real(_) => np::dtype::<f32>(py),
            },
            PyMeshAttributeData::F64(attr) => match attr.data {
                OwnedAttributeData::ScalarU64(_) => np::dtype::<u64>(py),
                OwnedAttributeData::ScalarReal(_) => np::dtype::<f64>(py),
                OwnedAttributeData::Vector3Real(_) => np::dtype::<f64>(py),
            },
        }
    }

    /// Name of the attribute
    #[getter]
    pub fn name(&self) -> String {
        match &self.inner {
            PyMeshAttributeData::F32(attr) => attr.name.clone(),
            PyMeshAttributeData::F64(attr) => attr.name.clone(),
        }
    }

    /// View of the attribute data as a numpy array
    #[getter]
    pub fn data<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &this.borrow().inner {
            PyMeshAttributeData::F32(attr) => match &attr.data {
                OwnedAttributeData::ScalarU64(data) => get_scalar_generic(data, this.into_any()),
                OwnedAttributeData::ScalarReal(data) => get_scalar_generic(data, this.into_any()),
                OwnedAttributeData::Vector3Real(data) => get_vec3f_generic(data, this.into_any()),
            },
            PyMeshAttributeData::F64(attr) => match &attr.data {
                OwnedAttributeData::ScalarU64(data) => get_scalar_generic(data, this.into_any()),
                OwnedAttributeData::ScalarReal(data) => get_scalar_generic(data, this.into_any()),
                OwnedAttributeData::Vector3Real(data) => get_vec3f_generic(data, this.into_any()),
            },
        }
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "MeshWithData")]
pub struct PyMeshWithData {
    mesh: PyMesh3dData,
    pub(crate) point_attributes: Vec<Py<PyMeshAttribute>>,
    pub(crate) cell_attributes: Vec<Py<PyMeshAttribute>>,
}

impl PyMeshWithData {
    /// Constructs a new mesh with data from an existing mesh object (either `TriMesh3d` or `MixedTriQuadMesh3d`)
    pub fn try_from_pymesh<'py, T>(py: Python<'py>, mesh: T) -> PyResult<Self>
    where
        T: IntoPyObject<'py>,
        T::Output: Into<Py<T::Target>>,
        Py<T::Target>: Into<PyMesh3dData>,
        PyErr: From<T::Error>,
    {
        let mesh_bound = mesh.into_pyobject(py)?;
        let mesh_py: Py<T::Target> = mesh_bound.into();
        let mesh: PyMesh3dData = mesh_py.into();

        Ok(Self {
            mesh,
            point_attributes: vec![],
            cell_attributes: vec![],
        })
    }

    pub fn try_from_mesh_with_data<'py, R: Real + Element, M: Mesh3d<R> + 'static>(
        py: Python<'_>,
        mesh_with_data: MeshWithData<R, M>,
    ) -> PyResult<Self> {
        // Deconstruct the input mesh
        let MeshWithData {
            mut mesh,
            mut point_attributes,
            mut cell_attributes,
        } = mesh_with_data;

        // Convert the inner mesh
        let mut mesh_with_data =
            if let Some(mesh) = transmute_same_take::<M, TriMesh3d<R>>(&mut mesh) {
                PyTriMesh3d::try_from_generic(mesh)
                    .and_then(|tri_mesh| Self::try_from_pymesh(py, tri_mesh))
            } else if let Some(mesh) = transmute_same_take::<M, MixedTriQuadMesh3d<R>>(&mut mesh) {
                PyMixedTriQuadMesh3d::try_from_generic(mesh)
                    .and_then(|quad_mesh| Self::try_from_pymesh(py, quad_mesh))
            } else {
                Err(pyerr_only_tri_and_tri_quad_mesh())
            }?;

        fn try_convert_attribute_vec<'a, In: Real + Element, Out: Real + Element>(
            py: Python<'_>,
            attributes: &mut Vec<OwnedMeshAttribute<In>>,
            dest: &mut Vec<Py<PyMeshAttribute>>,
        ) -> Option<()>
        where
            PyMeshAttribute: From<OwnedMeshAttribute<Out>>,
        {
            transmute_same_take::<Vec<OwnedMeshAttribute<In>>, Vec<OwnedMeshAttribute<Out>>>(
                attributes,
            )
            .map(|a| {
                a.into_iter()
                    .map(|a| {
                        PyMeshAttribute::from(a)
                            .into_pyobject(py)
                            .expect("allocation should not fail")
                            .into()
                    })
                    .collect::<Vec<Py<PyMeshAttribute>>>()
            })
            .and_then(|a| Some(*dest = a))
        }

        if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f32>() {
            try_convert_attribute_vec::<R, f32>(
                py,
                &mut point_attributes,
                &mut mesh_with_data.point_attributes,
            );
            try_convert_attribute_vec::<R, f32>(
                py,
                &mut cell_attributes,
                &mut mesh_with_data.cell_attributes,
            );
        } else if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f64>() {
            try_convert_attribute_vec::<R, f64>(
                py,
                &mut point_attributes,
                &mut mesh_with_data.point_attributes,
            );
            try_convert_attribute_vec::<R, f64>(
                py,
                &mut cell_attributes,
                &mut mesh_with_data.cell_attributes,
            );
        } else {
            return Err(pyerr_unsupported_scalar());
        }

        Ok(mesh_with_data)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshWithData {
    /// Numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.mesh {
            PyMesh3dData::Tri3d(mesh) => mesh.borrow(py).dtype(py),
            PyMesh3dData::MixedTriQuad3d(mesh) => mesh.borrow(py).dtype(py),
        }
    }

    /// Returns the type of the underlying mesh
    #[getter]
    pub fn mesh_type(&self) -> MeshType {
        match &self.mesh {
            PyMesh3dData::Tri3d(_) => MeshType::Tri3d,
            PyMesh3dData::MixedTriQuad3d(_) => MeshType::MixedTriQuad3d,
        }
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr="typing.List[MeshAttribute]", imports=()))]
    pub fn point_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.point_attributes.iter().map(|attr| attr.clone_ref(py)),
        )
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr="typing.List[MeshAttribute]", imports=()))]
    pub fn cell_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.cell_attributes.iter().map(|attr| attr.clone_ref(py)),
        )
    }

    pub fn as_tri3d<'py, 'a>(&'a self, py: Python<'py>) -> Option<Py<PyTriMesh3d>> {
        match &self.mesh {
            PyMesh3dData::Tri3d(mesh) => Some(mesh.clone_ref(py)),
            _ => None,
        }
    }

    pub fn as_mixed_tri_quad3d<'py>(&self, py: Python<'py>) -> Option<Py<PyMixedTriQuadMesh3d>> {
        match &self.mesh {
            PyMesh3dData::MixedTriQuad3d(mesh) => Some(mesh.clone_ref(py)),
            _ => None,
        }
    }

    /// The contained mesh without associated data and attributes
    #[getter]
    #[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MixedTriQuadMesh3d]", imports=()))]
    pub fn mesh<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        match &self.mesh {
            PyMesh3dData::Tri3d(mesh) => mesh.clone_ref(py).into_any(),
            PyMesh3dData::MixedTriQuad3d(mesh) => mesh.clone_ref(py).into_any(),
        }
    }

    /// Returns a copy of the contained mesh without associated data and attributes
    #[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MixedTriQuadMesh3d]", imports=()))]
    pub fn copy_mesh<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.mesh {
            PyMesh3dData::Tri3d(mesh) => mesh.borrow(py).clone().into_bound_py_any(py),
            PyMesh3dData::MixedTriQuad3d(mesh) => mesh.borrow(py).clone().into_bound_py_any(py),
        }
    }
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
