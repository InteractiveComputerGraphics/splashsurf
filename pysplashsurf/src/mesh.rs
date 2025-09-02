use bytemuck::{NoUninit, Pod};
use ndarray::Array2;
use numpy as np;
use numpy::prelude::*;
use numpy::{Element, PyArray, PyArray2, PyArrayDescr, PyUntypedArray};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
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

use crate::NumpyUsize;
use crate::utils;
use crate::utils::{enum_impl_from, enum_wrapper_impl_from};

fn view_triangles_generic<'py>(
    triangles: &[TriangleCell],
    container: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
    let vertex_indices: &[NumpyUsize] = bytemuck::cast_slice(triangles);
    let view = utils::view_generic(vertex_indices, &[triangles.len(), 3], container)?.into_any();
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

/// Triangle surface mesh in 3D
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
        utils::transmute_take_into::<_, TriMesh3d<f32>, _>(&mut mesh)
            .or_else(|| utils::transmute_take_into::<_, TriMesh3d<f64>, _>(&mut mesh))
            .ok_or_else(utils::pyerr_unsupported_scalar)
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
    /// Numpy dtype of the underlying scalar type (either ``np.float32`` or ``np.float64``)
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
            PyTriMesh3dData::F32(mesh) => utils::view_vec_generic(mesh.vertices(), this.into_any()),
            PyTriMesh3dData::F64(mesh) => utils::view_vec_generic(mesh.vertices(), this.into_any()),
        }
    }

    /// The `Mx3` array of vertex indices per triangle
    #[getter]
    pub fn triangles<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<NumpyUsize>>> {
        match &this.borrow().inner {
            PyTriMesh3dData::F32(mesh) => view_triangles_generic(mesh.cells(), this.into_any()),
            PyTriMesh3dData::F64(mesh) => view_triangles_generic(mesh.cells(), this.into_any()),
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

/// Mixed triangle and quad surface mesh in 3D
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
        utils::transmute_take_into::<_, MixedTriQuadMesh3d<f32>, _>(&mut mesh)
            .or_else(|| utils::transmute_take_into::<_, MixedTriQuadMesh3d<f64>, _>(&mut mesh))
            .ok_or_else(utils::pyerr_unsupported_scalar)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMixedTriQuadMesh3d {
    /// Numpy dtype of the underlying scalar type (either ``np.float32`` or ``np.float64``)
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
                utils::view_vec_generic(mesh.vertices(), this.into_any())
            }
            PyMixedTriQuadMesh3dData::F64(mesh) => {
                utils::view_vec_generic(mesh.vertices(), this.into_any())
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

    /// Returns a copy of all quad cells of the mesh as an `Nx4` array of vertex indices
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
        use utils::{view_scalar_generic, view_vec_generic};
        match &this.borrow().inner {
            PyMeshAttributeData::F32(attr) => match &attr.data {
                OwnedAttributeData::ScalarU64(data) => view_scalar_generic(data, this.into_any()),
                OwnedAttributeData::ScalarReal(data) => view_scalar_generic(data, this.into_any()),
                OwnedAttributeData::Vector3Real(data) => view_vec_generic(data, this.into_any()),
            },
            PyMeshAttributeData::F64(attr) => match &attr.data {
                OwnedAttributeData::ScalarU64(data) => view_scalar_generic(data, this.into_any()),
                OwnedAttributeData::ScalarReal(data) => view_scalar_generic(data, this.into_any()),
                OwnedAttributeData::Vector3Real(data) => view_vec_generic(data, this.into_any()),
            },
        }
    }
}

/// Mesh with attached point and cell attributes
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
        let mut mesh_with_data = if let Some(mesh) =
            utils::transmute_same_take::<M, TriMesh3d<R>>(&mut mesh)
        {
            PyTriMesh3d::try_from_generic(mesh)
                .and_then(|tri_mesh| Self::try_from_pymesh(py, tri_mesh))
        } else if let Some(mesh) = utils::transmute_same_take::<M, MixedTriQuadMesh3d<R>>(&mut mesh)
        {
            PyMixedTriQuadMesh3d::try_from_generic(mesh)
                .and_then(|quad_mesh| Self::try_from_pymesh(py, quad_mesh))
        } else {
            Err(utils::pyerr_only_tri_and_tri_quad_mesh())
        }?;

        fn try_convert_attribute_vec<'a, In: Real + Element, Out: Real + Element>(
            py: Python<'_>,
            attributes: &mut Vec<OwnedMeshAttribute<In>>,
            dest: &mut Vec<Py<PyMeshAttribute>>,
        ) -> Option<()>
        where
            PyMeshAttribute: From<OwnedMeshAttribute<Out>>,
        {
            utils::transmute_same_take::<Vec<OwnedMeshAttribute<In>>, Vec<OwnedMeshAttribute<Out>>>(
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
            return Err(utils::pyerr_unsupported_scalar());
        }

        Ok(mesh_with_data)
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
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshWithData {
    /// Numpy dtype of the underlying scalar type (either ``np.float32`` or ``np.float64``)
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.mesh {
            PyMesh3dData::Tri3d(mesh) => mesh.borrow(py).dtype(py),
            PyMesh3dData::MixedTriQuad3d(mesh) => mesh.borrow(py).dtype(py),
        }
    }

    /// Type of the underlying mesh
    #[getter]
    pub fn mesh_type(&self) -> MeshType {
        match &self.mesh {
            PyMesh3dData::Tri3d(_) => MeshType::Tri3d,
            PyMesh3dData::MixedTriQuad3d(_) => MeshType::MixedTriQuad3d,
        }
    }

    /// The attributes attached points (vertices) of the mesh
    #[getter]
    #[gen_stub(override_return_type(type_repr="dict[str, numpy.typing.NDArray]", imports=()))]
    pub fn point_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.point_attributes
            .iter()
            .map(|attr| -> PyResult<_> {
                let attr = attr.clone_ref(py).into_bound(py);
                let name = attr.try_borrow()?.name();
                let data = PyMeshAttribute::data(attr)?;
                Ok((name, data))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_py_dict(py)
    }

    /// The attributes attached to the cells (triangles or quads) of the mesh
    #[getter]
    #[gen_stub(override_return_type(type_repr="dict[str, numpy.typing.NDArray]", imports=()))]
    pub fn cell_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.cell_attributes
            .iter()
            .map(|attr| -> PyResult<_> {
                let attr = attr.clone_ref(py).into_bound(py);
                let name = attr.try_borrow()?.name();
                let data = PyMeshAttribute::data(attr)?;
                Ok((name, data))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_py_dict(py)
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
