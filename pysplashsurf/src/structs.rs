use numpy::{Element, IntoPyArray, PyArray2, ToPyArray, PyReadonlyArray2};
use ndarray::{ArrayView, ArrayView2, Array2};
use pyo3::{prelude::*, PyResult, PyObject, PyErr, IntoPyObjectExt, exceptions::PyValueError};
use splashsurf_lib::{nalgebra::{Unit, Vector3}, mesh::{AttributeData, MeshAttribute, MeshWithData, TriMesh3d, Mesh3d}, Real, SurfaceReconstruction, UniformGrid, sph_interpolation::SphInterpolator, Aabb3d};

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

            #[getter]
            fn mesh(&self) -> $mesh_class {
                $mesh_class::new(self.inner.mesh.clone())
            }

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

            fn get_point_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.point_attributes.as_slice(), name)
            }

            fn get_cell_attribute<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
                get_attribute_with_name::<$type>(py, self.inner.cell_attributes.as_slice(), name)
            }
        }
    };
}

macro_rules! create_sph_interpolator_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: SphInterpolator<$type>,
        }

        impl $name {
            pub fn new(data: SphInterpolator<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(
                particle_positions: &Bound<'py, PyArray2<$type>>,
                particle_densities: Vec<$type>, 
                particle_rest_mass: $type, 
                compact_support_radius: $type
            ) -> PyResult<Self> {
                let particle_positions: PyReadonlyArray2<$type> = particle_positions.extract().unwrap();
                let particle_positions = particle_positions.as_slice().unwrap();
                let particle_positions: &[Vector3<$type>] = bytemuck::cast_slice(particle_positions);

                Ok($name::new(SphInterpolator::new(particle_positions, particle_densities.as_slice(), particle_rest_mass, compact_support_radius)))
            }

            fn interpolate_scalar_quantity<'py>(
                &self, 
                particle_quantity: Vec<$type>, 
                interpolation_points: &Bound<'py, PyArray2<$type>>, 
                first_order_correction: bool
            ) -> Vec<$type> {
                let interpolation_points: PyReadonlyArray2<$type> = interpolation_points.extract().unwrap();
                let interpolation_points = interpolation_points.as_slice().unwrap();
                let interpolation_points: &[Vector3<$type>] = bytemuck::cast_slice(interpolation_points);

                self.inner.interpolate_scalar_quantity(particle_quantity.as_slice(), interpolation_points, first_order_correction)
            }

            fn interpolate_normals<'py>(
                &self, 
                py: Python<'py>,
                interpolation_points: &Bound<'py, PyArray2<$type>>,
            ) -> Bound<'py, PyArray2<$type>> {
                let interpolation_points: PyReadonlyArray2<$type> = interpolation_points.extract().unwrap();
                let interpolation_points = interpolation_points.as_slice().unwrap();
                let interpolation_points: &[Vector3<$type>] = bytemuck::cast_slice(interpolation_points);

                let normals_vec = self.inner.interpolate_normals(interpolation_points);
                let normals_vec = bytemuck::allocation::cast_vec::<Unit<Vector3<$type>>, $type>(normals_vec);
                //let normals_vec: Vec<$type> = bytemuck::cas

                let normals: &[$type] = normals_vec.as_slice();
                let normals: ArrayView2<$type> =
                    ArrayView::from_shape((normals.len() / 3, 3), normals)
                        .unwrap();
                
                normals.to_pyarray(py)
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

            fn vertex_vertex_connectivity(&self) -> Vec<Vec<usize>> {
                self.inner.vertex_vertex_connectivity()
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

            fn particle_densities(&self) -> &Vec<$type> {
                self.inner.particle_densities().ok_or_else( || anyhow::anyhow!("Surface Reconstruction did not return particle densities")).unwrap()
            }

            fn particle_neighbors(&self) -> Option<&Vec<Vec<usize>>> {
                self.inner.particle_neighbors()
            }
        }
    };
}

macro_rules! create_aabb3d_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: Aabb3d<$type>
        }

        impl $name {
            pub fn new(data: Aabb3d<$type>) -> Self {
                Self { inner: data }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn py_new<'py>(min: [$type; 3], max: [$type; 3]) -> PyResult<Self> {
                Ok($name::new(Aabb3d::<$type>::new(Vector3::from_column_slice(&min), Vector3::from_column_slice(&max))))
            }

            #[staticmethod]
            fn from_points<'py>(points: &Bound<'py, PyArray2<$type>>) -> $name {
                let points: PyReadonlyArray2<$type> = points.extract().unwrap();
                let points = points.as_slice().unwrap();
                let points: &[Vector3<$type>] = bytemuck::cast_slice(points);

                $name::new(Aabb3d::from_points(points))
            }

            fn grow_uniformly(&mut self, margin: $type) {
                self.inner.grow_uniformly(margin);
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

create_aabb3d_interface!(PyAabb3dF64, f64);
create_aabb3d_interface!(PyAabb3dF32, f32);

create_mesh_data_interface!(PyMeshWithDataF64, f64, PyTriMesh3dF64, PyAabb3dF64);
create_mesh_data_interface!(PyMeshWithDataF32, f32, PyTriMesh3dF32, PyAabb3dF32);

create_sph_interpolator_interface!(PySphInterpolatorF64, f64);
create_sph_interpolator_interface!(PySphInterpolatorF32, f32);