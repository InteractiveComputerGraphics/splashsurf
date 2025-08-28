use crate::NumpyUsize;
use crate::aabb::{Aabb3dF32, Aabb3dF64};
use crate::utils::*;
use ndarray::{Array2, ArrayView, ArrayView2};
use numpy as np;
use numpy::{
    Element, IntoPyArray, PyArray, PyArray2, PyArrayDescr, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArray, ToPyArray,
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

fn get_vertices_generic<'py, R: Real + Element>(
    vertices: &[Vector3<R>],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let coordinates: &[R] = bytemuck::cast_slice(vertices);
    let array: ArrayView2<R> =
        ArrayView::from_shape((vertices.len(), 3), coordinates).map_err(anyhow::Error::new)?;
    let pyarray = unsafe { PyArray2::borrow_from_array(&array, container) };
    Ok(pyarray
        .into_any()
        .downcast_into::<PyUntypedArray>()
        .expect("downcast should not fail"))
}

fn get_triangles_generic<'py>(
    triangles: &[TriangleCell],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
    let vertex_indices: &[NumpyUsize] = bytemuck::cast_slice(triangles);
    let array: ArrayView2<NumpyUsize> =
        ArrayView::from_shape((triangles.len(), 3), vertex_indices).map_err(anyhow::Error::new)?;
    let pyarray = unsafe { PyArray2::borrow_from_array(&array, container) };
    Ok(pyarray)
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

enum PyTriMesh3dData {
    F32(TriMesh3d<f32>),
    F64(TriMesh3d<f64>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "TriMesh3d")]
pub struct PyTriMesh3d {
    inner: PyTriMesh3dData,
}

impl_from_mesh!(PyTriMesh3d, TriMesh3d<f32> => PyTriMesh3dData::F32);
impl_from_mesh!(PyTriMesh3d, TriMesh3d<f64> => PyTriMesh3dData::F64);

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
    /// Returns the numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
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
            PyTriMesh3dData::F32(mesh) => get_vertices_generic(mesh.vertices(), this.into_any()),
            PyTriMesh3dData::F64(mesh) => get_vertices_generic(mesh.vertices(), this.into_any()),
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

enum PyMixedTriQuadMesh3dData {
    F32(MixedTriQuadMesh3d<f32>),
    F64(MixedTriQuadMesh3d<f64>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "MixedTriQuadMesh3d")]
pub struct PyMixedTriQuadMesh3d {
    inner: PyMixedTriQuadMesh3dData,
}

impl_from_mesh!(PyMixedTriQuadMesh3d, MixedTriQuadMesh3d<f32> => PyMixedTriQuadMesh3dData::F32);
impl_from_mesh!(PyMixedTriQuadMesh3d, MixedTriQuadMesh3d<f64> => PyMixedTriQuadMesh3dData::F64);

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
    /// Returns the numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
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
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
            PyMixedTriQuadMesh3dData::F64(mesh) => {
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
        }
    }
}

enum PyMeshWithDataData {
    Tri3dF32(MeshWithData<f32, TriMesh3d<f32>>),
    Tri3dF64(MeshWithData<f64, TriMesh3d<f64>>),
    MixedTriQuadF32(MeshWithData<f32, MixedTriQuadMesh3d<f32>>),
    MixedTriQuadF64(MeshWithData<f64, MixedTriQuadMesh3d<f64>>),
}

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(name = "MeshWithData")]
pub struct PyMeshWithData {
    inner: PyMeshWithDataData,
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

impl_from_mesh!(PyMeshWithData, MeshWithData<f32, TriMesh3d<f32>> => PyMeshWithDataData::Tri3dF32);
impl_from_mesh!(PyMeshWithData, MeshWithData<f64, TriMesh3d<f64>> => PyMeshWithDataData::Tri3dF64);
impl_from_mesh!(PyMeshWithData, MeshWithData<f32, MixedTriQuadMesh3d<f32>> => PyMeshWithDataData::MixedTriQuadF32);
impl_from_mesh!(PyMeshWithData, MeshWithData<f64, MixedTriQuadMesh3d<f64>> => PyMeshWithDataData::MixedTriQuadF64);

impl PyMeshWithData {
    pub fn try_from_generic<R: Real + Element, M: Mesh3d<R> + 'static>(
        mut mesh: MeshWithData<R, M>,
    ) -> PyResult<Self> {
        transmute_take_into::<_, MeshWithData<f32, TriMesh3d<f32>>, _>(&mut mesh)
            .or_else(|| transmute_take_into::<_, MeshWithData<f64, TriMesh3d<f64>>, _>(&mut mesh))
            .or_else(|| transmute_take_into::<_, MeshWithData<f32, MixedTriQuadMesh3d<f32>>, _>(&mut mesh))
            .or_else(|| transmute_take_into::<_, MeshWithData<f64, MixedTriQuadMesh3d<f64>>, _>(&mut mesh))
            .ok_or_else(|| PyTypeError::new_err(
            "Unsupported mesh type for MeshWithData. Only TriMesh3d and MixedTriQuadMesh3d with f32 or f64 scalar types are supported.",
        ))
    }

    pub fn as_tri_f32(&self) -> Option<&MeshWithData<f32, TriMesh3d<f32>>> {
        match &self.inner {
            PyMeshWithDataData::Tri3dF32(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_tri_f64(&self) -> Option<&MeshWithData<f64, TriMesh3d<f64>>> {
        match &self.inner {
            PyMeshWithDataData::Tri3dF64(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_tri_f32_mut(&mut self) -> Option<&mut MeshWithData<f32, TriMesh3d<f32>>> {
        match &mut self.inner {
            PyMeshWithDataData::Tri3dF32(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn as_tri_f64_mut(&mut self) -> Option<&mut MeshWithData<f64, TriMesh3d<f64>>> {
        match &mut self.inner {
            PyMeshWithDataData::Tri3dF64(mesh) => Some(mesh),
            _ => None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshWithData {
    /// Returns the numpy dtype of the underlying scalar type (either `np.float32` or `np.float64`)
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            PyMeshWithDataData::Tri3dF32(_) | PyMeshWithDataData::MixedTriQuadF32(_) => {
                np::dtype::<f32>(py)
            }
            PyMeshWithDataData::Tri3dF64(_) | PyMeshWithDataData::MixedTriQuadF64(_) => {
                np::dtype::<f64>(py)
            }
        }
    }

    /// Returns the type of the underlying mesh
    #[getter]
    pub fn mesh_cell_type(&self) -> MeshType {
        match &self.inner {
            PyMeshWithDataData::Tri3dF32(_) | PyMeshWithDataData::Tri3dF64(_) => MeshType::Tri3d,
            PyMeshWithDataData::MixedTriQuadF32(_) | PyMeshWithDataData::MixedTriQuadF64(_) => {
                MeshType::MixedTriQuad3d
            }
        }
    }

    /// Returns a copy of the contained mesh without associated data and attributes
    #[gen_stub(override_return_type(type_repr="typing.Union[TriMesh3d, MixedTriQuadMesh3d]", imports=()))]
    pub fn copy_mesh<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            PyMeshWithDataData::Tri3dF32(mesh) => {
                PyTriMesh3d::from(mesh.mesh.clone()).into_bound_py_any(py)
            }
            PyMeshWithDataData::Tri3dF64(mesh) => {
                PyTriMesh3d::from(mesh.mesh.clone()).into_bound_py_any(py)
            }
            PyMeshWithDataData::MixedTriQuadF32(mesh) => {
                PyMixedTriQuadMesh3d::from(mesh.mesh.clone()).into_bound_py_any(py)
            }
            PyMeshWithDataData::MixedTriQuadF64(mesh) => {
                PyMixedTriQuadMesh3d::from(mesh.mesh.clone()).into_bound_py_any(py)
            }
        }
    }

    /// The `Nx3` array of vertex positions of the mesh
    #[getter]
    pub fn vertices<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyUntypedArray>> {
        match &this.borrow().inner {
            PyMeshWithDataData::Tri3dF32(mesh) => {
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
            PyMeshWithDataData::Tri3dF64(mesh) => {
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
            PyMeshWithDataData::MixedTriQuadF32(mesh) => {
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
            PyMeshWithDataData::MixedTriQuadF64(mesh) => {
                get_vertices_generic(mesh.vertices(), this.into_any())
            }
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
