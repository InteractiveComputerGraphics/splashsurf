//! Functions for post-processing of surface meshes (decimation, smoothing, etc.)

use crate::mesh::{Mesh3d, MixedTriQuadMesh3d, TriMesh3d, TriangleOrQuadCell};
use crate::{SetType, profile, MapType, Real};
use log::info;
use nalgebra::Vector3;
use rayon::prelude::*;

/// Merges triangles sharing an edge to quads if they fulfill the given criteria
pub fn convert_tris_to_quads<R: Real>(
    mesh: &TriMesh3d<R>,
    non_squareness_limit: R,
    normal_angle_limit_rad: R,
    max_interior_angle: R,
) -> MixedTriQuadMesh3d<R> {
    profile!("tri_to_quad");

    let vert_tri_map = mesh.vertex_cell_connectivity();
    let tri_normals = mesh
        .triangles
        .par_iter()
        .map(|tri| {
            let v0 = &mesh.vertices[tri[0]];
            let v1 = &mesh.vertices[tri[1]];
            let v2 = &mesh.vertices[tri[2]];
            (v1 - v0).cross(&(v2 - v1)).normalize()
        })
        .collect::<Vec<_>>();

    let min_dot = normal_angle_limit_rad.cos();
    let max_non_squareness = non_squareness_limit;
    let sqrt_two = R::from_f64(2.0_f64.sqrt()).unwrap();

    let tris_to_quad = |tri_i: &[usize; 3], tri_j: &[usize; 3]| -> [usize; 4] {
        let mut quad = [0, 0, 0, 0];
        let missing_vertex: usize = tri_j.iter().copied().find(|v| !tri_i.contains(v)).unwrap();

        quad[0] = tri_i[0];
        if tri_j.contains(&tri_i[0]) {
            if tri_j.contains(&tri_i[1]) {
                quad[1] = missing_vertex;
                quad[2] = tri_i[1];
                quad[3] = tri_i[2];
            } else {
                quad[1] = tri_i[1];
                quad[2] = tri_i[2];
                quad[3] = missing_vertex;
            }
        } else {
            if tri_j.contains(&tri_i[1]) {
                quad[1] = tri_i[1];
                quad[2] = missing_vertex;
                quad[3] = tri_i[2];
            } else {
                panic!("this should not happen");
            }
        }

        quad
    };

    /// Computes the interior angle of a quad at vertex v_center with the quad given by [v_prev, v_center, v_next, v_opp]
    fn quad_interior_angle<R: Real>(
        v_center: &Vector3<R>,
        v_prev: &Vector3<R>,
        v_next: &Vector3<R>,
        v_opp: &Vector3<R>,
    ) -> R {
        let d_prev = v_prev - v_center;
        let d_middle = v_opp - v_center;
        let d_next = v_next - v_center;

        let l_prev = d_prev.norm();
        let l_middle = d_middle.norm();
        let l_next = d_next.norm();

        let angle_prev = (d_prev.dot(&d_middle)).unscale(l_prev * l_middle).acos();
        let angle_next = (d_middle.dot(&d_next)).unscale(l_middle * l_next).acos();

        angle_prev + angle_next
    }

    let quad_candidates = mesh
        .triangles
        .par_iter()
        .enumerate()
        .filter_map(|(i, tri_i)| {
            for &vert in tri_i {
                // Loop over all triangles of all vertices of the triangle from iterator chain
                for &j in &vert_tri_map[vert] {
                    // Skip triangle from iterator chain and make sure that we process every triangle pair only once
                    if j <= i {
                        continue;
                    } else {
                        let tri_j = &mesh.triangles[j];
                        let mut recurring_verts = 0;
                        let mut other_vert = 0;
                        for &v in tri_j {
                            if tri_i.contains(&v) {
                                recurring_verts += 1;
                                if v != vert {
                                    other_vert = v;
                                }
                            }
                        }

                        // Found triangle pair with shared edge
                        if recurring_verts == 2 {
                            let mut convert = false;
                            let mut quality = R::one();

                            // Check "squareness" of the two triangles
                            {
                                let quad = tris_to_quad(tri_i, tri_j);

                                // Compute diagonal edge length
                                let diag = (mesh.vertices[vert] - mesh.vertices[other_vert]).norm();
                                let max_length = (diag / sqrt_two) * max_non_squareness;
                                let min_length = (diag / sqrt_two) * max_non_squareness.recip();

                                let v0 = mesh.vertices[quad[0]];
                                let v1 = mesh.vertices[quad[1]];
                                let v2 = mesh.vertices[quad[2]];
                                let v3 = mesh.vertices[quad[3]];

                                let d0 = v1 - v0;
                                let d1 = v2 - v1;
                                let d2 = v3 - v2;
                                let d3 = v0 - v3;

                                let edge_ls = [d0.norm(), d1.norm(), d2.norm(), d3.norm()];

                                let angles = [
                                    quad_interior_angle(&v0, &v3, &v1, &v2),
                                    quad_interior_angle(&v1, &v0, &v2, &v3),
                                    quad_interior_angle(&v2, &v3, &v1, &v0),
                                    quad_interior_angle(&v3, &v2, &v0, &v1),
                                ];

                                if edge_ls
                                    .iter()
                                    .all(|&edge_l| edge_l <= max_length && edge_l >= min_length)
                                    && angles.iter().all(|&angle| angle <= max_interior_angle)
                                {
                                    convert = true;

                                    let mut longest = edge_ls[0];
                                    let mut shortest = edge_ls[0];

                                    for l in edge_ls {
                                        longest = longest.max(l);
                                        shortest = shortest.min(l);
                                    }

                                    quality = shortest / longest;
                                }
                            }

                            // Check normal deviation of triangles
                            if convert {
                                let dot_i_j = tri_normals[i].dot(&tri_normals[j]);
                                if dot_i_j >= min_dot {
                                    convert = convert;
                                } else {
                                    convert = false;
                                }
                            }

                            if convert {
                                // Return the two triangle indices that should be merged
                                return Some(((i, j), quality));
                            }
                        }
                    }
                }
            }

            return None;
        })
        .collect::<MapType<_, _>>();

    info!(
        "Number of quad conversion candidates: {}",
        quad_candidates.len()
    );

    //let mut triangles_to_remove = new_map();
    let mut triangles_to_remove = SetType::default();
    let mut filtered_candidates = SetType::default();
    for ((i, j), _q) in quad_candidates {
        // TODO: If triangle already exists in list, compare quality

        if !triangles_to_remove.contains(&i) && !triangles_to_remove.contains(&j) {
            //triangles_to_remove.insert(i, ((i,j), q));
            //triangles_to_remove.insert(j, ((i,j), q));
            triangles_to_remove.insert(i);
            triangles_to_remove.insert(j);

            filtered_candidates.insert((i, j));
        }
    }

    let quads: Vec<[usize; 4]> = filtered_candidates
        .par_iter()
        .copied()
        .map(|(i, j)| {
            let tri_i = &mesh.triangles[i];
            let tri_j = &mesh.triangles[j];
            tris_to_quad(tri_i, tri_j)
        })
        .collect::<_>();

    let filtered_triangles = mesh
        .triangles
        .par_iter()
        .copied()
        .enumerate()
        .filter_map(|(i, tri)| (!triangles_to_remove.contains(&i)).then_some(tri))
        .collect::<Vec<_>>();

    info!("Before conversion: {} triangles", mesh.triangles.len());
    info!(
        "After conversion: {} triangles, {} quads, {} total cells ({:.2}% fewer)",
        filtered_triangles.len(),
        quads.len(),
        filtered_triangles.len() + quads.len(),
        (((mesh.triangles.len() - (filtered_triangles.len() + quads.len())) as f64)
            / (mesh.triangles.len() as f64))
            * 100.0
    );

    let mut cells = Vec::with_capacity(filtered_triangles.len() + quads.len());
    cells.extend(
        filtered_triangles
            .into_iter()
            .map(|tri| TriangleOrQuadCell::Tri(tri)),
    );
    cells.extend(quads.into_iter().map(|quad| TriangleOrQuadCell::Quad(quad)));

    MixedTriQuadMesh3d {
        vertices: mesh.vertices.clone(),
        cells,
    }
}
