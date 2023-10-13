use splashsurf_lib::halfedge_mesh::HalfEdgeTriMesh;
use splashsurf_lib::io;
use splashsurf_lib::mesh::{Mesh3d, MeshWithData};

#[test]
fn test_halfedge_ico() -> Result<(), anyhow::Error> {
    let mesh = io::obj_format::surface_mesh_from_obj::<f32, _>("../data/icosphere.obj")?.mesh;
    let mut he_mesh = HalfEdgeTriMesh::from(mesh);

    he_mesh.try_half_edge_collapse(he_mesh.half_edge(12, 0).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(18, 2).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(2, 17).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(17, 0).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(27, 7).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(34, 7).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(39, 7).unwrap())?;
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(26, 7).unwrap())?;

    he_mesh.try_half_edge_collapse(he_mesh.half_edge(0, 16).unwrap())?;

    let (tri_mesh, _vertex_map) = he_mesh.into_parts(true);
    let _tri_vertex_map = tri_mesh.vertex_vertex_connectivity();

    io::obj_format::mesh_to_obj(&MeshWithData::new(tri_mesh), "../out/icosphere_new.obj")?;
    Ok(())
}

#[test]
fn test_halfedge_plane() -> Result<(), anyhow::Error> {
    let mut mesh = io::obj_format::surface_mesh_from_obj::<f32, _>("../data/plane.obj")?.mesh;

    // Make mesh curved
    for v in mesh.vertices.iter_mut() {
        v.y = -0.1 * (v.x * v.x + v.z * v.z);
    }

    io::obj_format::mesh_to_obj(&MeshWithData::new(mesh.clone()), "../out/plane_new_0.obj")?;

    let mut he_mesh = HalfEdgeTriMesh::from(mesh);

    //he_mesh.vertices[246].y = 0.01;
    //he_mesh.vertices[267].y = 0.02;

    dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(224, 223).unwrap()));
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(224, 223).unwrap())?;

    dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(223, 225).unwrap()));
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(223, 225).unwrap())?;

    dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(225, 246).unwrap()));
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(225, 246).unwrap())?;

    dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(246, 267).unwrap()));
    he_mesh.try_half_edge_collapse(he_mesh.half_edge(246, 267).unwrap())?;

    //dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(223, 202).unwrap()));
    //he_mesh.try_half_edge_collapse(he_mesh.half_edge(223, 202).unwrap())?;
    //dbg!(he_mesh.half_edge_collapse_max_normal_change(he_mesh.half_edge(202, 181).unwrap()));
    //he_mesh.try_half_edge_collapse(he_mesh.half_edge(202, 181).unwrap())?;

    let (tri_mesh, _vertex_map) = he_mesh.into_parts(true);
    let _tri_vertex_map = tri_mesh.vertex_vertex_connectivity();

    io::obj_format::mesh_to_obj(&MeshWithData::new(tri_mesh), "../out/plane_new_1.obj")?;
    Ok(())
}
