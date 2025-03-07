use numpy::{Element, IntoPyArray, PyArray2, ToPyArray};
use ndarray::{ArrayView, ArrayView2, Array2};
use pyo3::{prelude::*, PyResult, PyObject, PyErr, IntoPyObjectExt, PyAny};
use splashsurf_lib::{mesh::{AttributeData, MeshAttribute, MeshWithData, TriMesh3d}, Real, SurfaceReconstruction, UniformGrid};

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
        None => Err(PyErr::new::<PyAny, _>(format!("Attribute with name {} doesn't exist", name)))
    }
}
 
macro_rules! create_mesh_data_interface {
    ($name: ident, $type: ident, $mesh_class: ident) => {
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
            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh.clone())
            }

            fn get_point_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.point_attributes.as_slice(), name)
            }

            fn get_cell_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.cell_attributes.as_slice(), name)
            }
        }
    };
}

macro_rules! create_mesh_interface {
    ($name: ident, $type: ident) => {
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
            #[getter]
            fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$type>> {
                let points: &[$type] = bytemuck::cast_slice(&self.inner.vertices);
                let vertices: ArrayView2<$type> =
                    ArrayView::from_shape((self.inner.vertices.len(), 3), points)
                        .unwrap();
                vertices.to_pyarray(py) // seems like at least one copy is necessary here (to_pyarray copies the data)
            }

            #[getter]
            fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
                let tris: &[usize] = bytemuck::cast_slice(&self.inner.triangles);
                let triangles: ArrayView2<usize> =
                    ArrayView::from_shape((self.inner.triangles.len(), 3), tris).unwrap();
                triangles.to_pyarray(py)
            }
        }
    };
}

macro_rules! create_grid_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: UniformGrid<i64, $type>,
        }

        impl $name {
            pub fn new(data: UniformGrid<i64, $type>) -> Self {
                Self { inner: data }
            }
        }
    };
}

macro_rules! create_reconstruction_interface {
    ($name: ident, $type: ident, $mesh_class: ident, $grid_class: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: SurfaceReconstruction<i64, $type>,
        }

        impl $name {
            pub fn new(data: SurfaceReconstruction<i64, $type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh().clone())
            }

            #[getter]
            fn grid(&self) -> $grid_class {
                $grid_class::new(self.inner.grid().clone())
            }
        }
    };
}

create_mesh_interface!(PyTriMesh3dF64, f64);
create_mesh_interface!(PyTriMesh3dF32, f32);

create_grid_interface!(PyUniformGridF64, f64);
create_grid_interface!(PyUniformGridF32, f32);

create_reconstruction_interface!(PySurfaceReconstructionF64, f64, PyTriMesh3dF64, PyUniformGridF64);
create_reconstruction_interface!(PySurfaceReconstructionF32, f32, PyTriMesh3dF32, PyUniformGridF32);

create_mesh_data_interface!(PyMeshWithDataF64, f64, PyTriMesh3dF64);
create_mesh_data_interface!(PyMeshWithDataF32, f32, PyTriMesh3dF32);