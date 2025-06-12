//! Functions for post-processing of surface meshes (decimation, smoothing, etc.)

use crate::halfedge_mesh::{HalfEdgeTriMesh, IllegalHalfEdgeCollapse};
use crate::mesh::{Mesh3d, MixedTriQuadMesh3d, TriMesh3d, TriMesh3dExt, TriangleOrQuadCell};
use crate::topology::{Axis, DirectedAxis, Direction};
use crate::uniform_grid::UniformCartesianCubeGrid3d;
use crate::{Index, MapType, Real, SetType, profile};
use log::{info, warn};
use nalgebra::Vector3;
use rayon::prelude::*;

/// Laplacian Smoothing with feature weights
///
/// Move each vertex towards the mean position of its neighbors.
/// Factor beta in \[0;1] proportional to the amount of smoothing (for beta=1 each vertex is placed at the mean position).
/// Additionally, feature weights can be specified to apply a varying amount of smoothing over the mesh.
pub fn par_laplacian_smoothing_inplace<R: Real>(
    mesh: &mut TriMesh3d<R>,
    vertex_connectivity: &[Vec<usize>],
    iterations: usize,
    beta: R,
    weights: &[R],
) {
    profile!("laplacian_smoothing");

    let mut vertex_buffer = mesh.vertices.clone();

    for _ in 0..iterations {
        profile!("laplacian_smoothing iter");

        std::mem::swap(&mut vertex_buffer, &mut mesh.vertices);

        mesh.vertices
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, vertex_i)| {
                let beta_eff = beta * weights[i];

                // Compute mean position of neighboring vertices
                let mut vertex_sum = Vector3::zeros();
                for j in vertex_connectivity[i].iter() {
                    vertex_sum += vertex_buffer[*j];
                }
                if !vertex_connectivity[i].is_empty() {
                    let n = R::from_usize(vertex_connectivity[i].len()).unwrap();
                    vertex_sum /= n;
                }

                *vertex_i = vertex_i.scale(R::one() - beta_eff) + vertex_sum.scale(beta_eff);
            });
    }
}

/// Laplacian smoothing of a normal field
pub fn par_laplacian_smoothing_normals_inplace<R: Real>(
    normals: &mut Vec<Vector3<R>>,
    vertex_connectivity: &[Vec<usize>],
    iterations: usize,
) {
    profile!("par_laplacian_smoothing_normals_inplace");

    let mut normal_buffer = normals.clone();

    for _ in 0..iterations {
        profile!("smoothing iteration");

        std::mem::swap(&mut normal_buffer, normals);

        normals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, normal_i)| {
                *normal_i = Vector3::zeros();
                for j in vertex_connectivity[i].iter().copied() {
                    let normal_j = normal_buffer[j];
                    *normal_i += normal_j;
                }
                normal_i.normalize_mut();
            });
    }
}

/// Mesh simplification designed for marching cubes surfaces meshes inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren
///
/// See Moore and Warren: ["Mesh Displacement: An Improved Contouring Method for Trivariate Data"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf) (1991)
/// or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
pub fn marching_cubes_cleanup<I: Index, R: Real>(
    mesh: &mut TriMesh3d<R>,
    grid: &UniformCartesianCubeGrid3d<I, R>,
    max_rel_snap_distance: Option<R>,
    max_iter: usize,
    keep_vertices: bool,
) -> Vec<Vec<usize>> {
    profile!("marching_cubes_cleanup");

    let half_dx = grid.cell_size() / (R::one() + R::one());
    let max_snap_distance_sq =
        max_rel_snap_distance.map(|factor| (factor * grid.cell_size()).powi(2));

    let nearest_grid_point = {
        profile!("determine nearest grid points");
        mesh.vertices
            .par_iter()
            .enumerate()
            .map(|(_, v)| {
                // TODO: Move this to uniform grid
                let cell_ijk = grid.enclosing_cell(v);
                let min_point = grid.get_point(cell_ijk).unwrap();
                let min_coord = grid.point_coordinates(&min_point);

                let mut nearest_point = min_point;
                if (v.x - min_coord.x) > half_dx {
                    nearest_point = grid
                        .get_point_neighbor(
                            &nearest_point,
                            DirectedAxis::new(Axis::X, Direction::Positive),
                        )
                        .unwrap()
                }
                if (v.y - min_coord.y) > half_dx {
                    nearest_point = grid
                        .get_point_neighbor(
                            &nearest_point,
                            DirectedAxis::new(Axis::Y, Direction::Positive),
                        )
                        .unwrap()
                }
                if (v.z - min_coord.z) > half_dx {
                    nearest_point = grid
                        .get_point_neighbor(
                            &nearest_point,
                            DirectedAxis::new(Axis::Z, Direction::Positive),
                        )
                        .unwrap()
                }

                grid.flatten_point_index(&nearest_point)
            })
            .collect::<Vec<_>>()
    };

    let (tri_mesh, vertex_map) = {
        profile!("mesh displacement");
        let mut mesh = HalfEdgeTriMesh::from(std::mem::take(mesh));

        // Tracks per vertex how many collapsed vertices contributed to its position
        let mut vertex_sum_count = vec![1_usize; mesh.vertices.len()];
        // Buffer for vertices that should get collapsed
        let mut vertex_buffer = Vec::new();

        for _ in 0..max_iter {
            profile!("mesh displacement iteration");

            let mut collapse_count = 0;
            for v0 in 0..mesh.vertices.len() {
                if !mesh.is_valid_vertex(v0) {
                    continue;
                }

                let grid_point = grid
                    .try_unflatten_point_index(nearest_grid_point[v0])
                    .map(|p| grid.point_coordinates(&p));

                if let Some(max_snap_distance_sq) = max_snap_distance_sq {
                    if let Some(grid_point) = grid_point {
                        // Check if this vertex is close enough to the grid vertex
                        if (mesh.vertices[v0] - grid_point).norm_squared() <= max_snap_distance_sq {
                            // Check for collapse with all neighbors
                            for he in mesh.outgoing_half_edges(v0) {
                                let v1 = he.to;
                                if nearest_grid_point[v0] == nearest_grid_point[v1] {
                                    // Ensure that other vertex is close enough to grid vertex
                                    if (mesh.vertices[v1] - grid_point).norm_squared()
                                        <= max_snap_distance_sq
                                    {
                                        vertex_buffer.push(v1);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Check for collapse with all neighbors
                    for he in mesh.outgoing_half_edges(v0) {
                        let v1 = he.to;
                        if nearest_grid_point[v0] == nearest_grid_point[v1] {
                            vertex_buffer.push(v1);
                        }
                    }
                }

                for &v1 in vertex_buffer.iter() {
                    if mesh.is_valid_vertex(v1) {
                        if let Some(he) = mesh.half_edge(v1, v0) {
                            if mesh.try_half_edge_collapse(he).is_ok() {
                                collapse_count += 1;

                                // Move to averaged position
                                let pos_v0 = mesh.vertices[v0];
                                let pos_v1 = mesh.vertices[v1];

                                let n0 = vertex_sum_count[v0];
                                let n1 = vertex_sum_count[v1];

                                let n_new = n0 + n1;
                                let pos_new = (pos_v0.scale(R::from_usize(n0).unwrap())
                                    + pos_v1.scale(R::from_usize(n1).unwrap()))
                                .unscale(R::from_usize(n_new).unwrap());

                                vertex_sum_count[v0] = n_new;
                                mesh.vertices[v0] = pos_new;
                            }
                        }
                    }
                }

                vertex_buffer.clear();
            }

            if collapse_count == 0 {
                break;
            }
        }

        mesh.into_parts(keep_vertices)
    };

    *mesh = tri_mesh;
    vertex_map
}

pub fn decimation<R: Real>(mesh: &mut TriMesh3d<R>, keep_vertices: bool) -> Vec<Vec<usize>> {
    profile!("decimation");

    let mut half_edge_mesh = HalfEdgeTriMesh::from(std::mem::take(mesh));
    merge_barnacle_configurations(&mut half_edge_mesh);

    {
        profile!("convert mesh back");
        let (new_mesh, vertex_map) = half_edge_mesh.into_parts(keep_vertices);
        *mesh = new_mesh;
        vertex_map
    }
}

pub fn merge_barnacle_configurations<R: Real>(mesh: &mut HalfEdgeTriMesh<R>) {
    profile!("merge_barnacle_configurations");

    merge_single_barnacle_configurations_he(mesh);
    merge_double_barnacle_configurations_he(mesh);
}

#[allow(unused)]
fn find_small_triangles<R: Real>(mesh: &HalfEdgeTriMesh<R>, area_limit: R) -> Vec<usize> {
    profile!("find_small_triangles");
    let zero_sized_triangles = mesh
        .triangles
        .par_iter()
        .enumerate()
        .filter(|(i, _)| mesh.is_valid_triangle(*i))
        .filter(|(_, tri)| mesh.tri_area_ijk::<R>(tri) <= area_limit)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    info!(
        "Found {} small sized triangles (area <= {})",
        zero_sized_triangles.len(),
        area_limit
    );
    zero_sized_triangles
}

#[allow(unused)]
fn find_bad_triangles<R: Real>(mesh: &HalfEdgeTriMesh<R>, max_aspect_ratio: R) -> Vec<usize> {
    profile!("find_bad_triangles");
    let mut ars = mesh
        .triangles
        .par_iter()
        .enumerate()
        .filter(|(i, _)| mesh.is_valid_triangle(*i))
        .filter(|(_, tri)| mesh.tri_area_ijk::<R>(tri) > R::default_epsilon())
        .map(|(i, tri)| {
            let aspect_ratio = mesh.tri_aspect_ratio::<R>(tri);
            (i, aspect_ratio)
        })
        .filter(|(_, ar)| *ar >= max_aspect_ratio)
        .collect::<Vec<_>>();
    ars.par_sort_unstable_by(|a, b| {
        let a: R = a.1;
        let b: R = b.1;
        a.to_f32()
            .unwrap()
            .total_cmp(&b.to_f32().unwrap())
            .reverse()
    });
    dbg!(&ars[0..10.min(ars.len())]);
    println!(
        "Found {} triangles with bad aspect ratio (>= {})",
        ars.len(),
        max_aspect_ratio
    );
    ars.into_iter().map(|(i, _)| i).collect()
}

#[allow(unused)]
fn process_triangle_collapse_queue<R: Real>(
    mesh: &mut HalfEdgeTriMesh<R>,
    triangles: impl Iterator<Item = usize>,
) -> (Vec<usize>, usize) {
    let mut processed = 0;
    let remaining = triangles
        .flat_map(|tri_idx| {
            if !mesh.is_valid_triangle(tri_idx) {
                return None;
            }

            let tri = mesh.triangles[tri_idx];

            let mut last_res = None;
            let mut from = 0;
            let mut to = 0;
            // Try to find an edge of the triangle that can be collapsed
            for i in 0..3 {
                from = tri[i];
                to = tri[(i + 1) % 3];

                if let Some(he) = mesh.half_edge(from, to) {
                    last_res = Some(mesh.try_half_edge_collapse(he));
                    if let Some(Ok(_)) = last_res {
                        processed += 1;
                        return None;
                    }
                } else {
                    warn!(
                        "Invalid collapse: Half-edge missing (from {} to {})",
                        from, to
                    );
                    return None;
                }
            }

            match last_res {
                Some(Err(IllegalHalfEdgeCollapse::IntersectionOfOneRing)) => Some(tri_idx),
                Some(Err(e)) => {
                    warn!("Invalid collapse: {:?} (from {} to {})", e, from, to);
                    None
                }
                _ => None,
            }
        })
        .collect();

    (remaining, processed)
}

#[allow(unused)]
fn process_triangle_collapse_queue_iterative<R: Real>(
    mesh: &mut HalfEdgeTriMesh<R>,
    triangles: impl Iterator<Item = usize>,
) -> usize {
    profile!("process_triangle_collapse_queue_iterative");
    let (mut remaining, mut processed) = process_triangle_collapse_queue(mesh, triangles);
    let mut iter = 1;
    info!(
        "{} collapse operations remaining after pass {}",
        remaining.len(),
        iter
    );
    while !remaining.is_empty() && iter < 5 {
        iter += 1;
        let (remaining_new, processed_new) =
            process_triangle_collapse_queue(mesh, remaining.into_iter());
        remaining = remaining_new;
        processed += processed_new;
        info!(
            "{} collapse operations remaining after pass {}",
            remaining.len(),
            iter
        );
    }

    processed
}

fn process_collapse_queue<R: Real>(
    mesh: &mut HalfEdgeTriMesh<R>,
    collapses: impl Iterator<Item = (usize, usize)>,
) -> SetType<(usize, usize)> {
    collapses
        .flat_map(|(from, to)| {
            if let Some(he) = mesh.half_edge(from, to) {
                match mesh.try_half_edge_collapse(he) {
                    Ok(_) => None,
                    Err(IllegalHalfEdgeCollapse::IntersectionOfOneRing) => Some((from, to)),
                    Err(e) => {
                        warn!("Invalid collapse: {:?} (from {} to {})", e, from, to);
                        None
                    }
                }
            } else {
                warn!(
                    "Invalid collapse: Half-edge missing (from {} to {})",
                    from, to
                );
                None
            }
        })
        .collect()
}

fn process_collapse_queue_iterative<R: Real>(
    mesh: &mut HalfEdgeTriMesh<R>,
    collapses: impl Iterator<Item = (usize, usize)>,
) {
    profile!("process_collapse_queue_iterative");
    let mut remaining = process_collapse_queue(mesh, collapses);
    let mut iter = 1;
    info!(
        "{} collapse operations remaining after pass {}",
        remaining.len(),
        iter
    );
    while !remaining.is_empty() && iter < 5 {
        iter += 1;
        remaining = process_collapse_queue(mesh, remaining.into_iter());
        info!(
            "{} collapse operations remaining after pass {}",
            remaining.len(),
            iter
        );
    }
}

pub fn merge_single_barnacle_configurations_he<R: Real>(mesh: &mut HalfEdgeTriMesh<R>) {
    profile!("merge_single_barnacle_configurations");

    //let vertex_map = &mesh.vertex_map;

    let half_edge_collapses = {
        profile!("find candidates");

        let mut candidates = mesh
            .vertices
            .par_iter()
            .enumerate()
            .filter_map(|(i, _v_i)| {
                (mesh.vertex_one_ring_len(i) == 4
                    && mesh
                        .vertex_one_ring(i)
                        .map(|j| mesh.vertex_one_ring_len(j))
                        .all(|len| (4..=6).contains(&len))
                    && mesh
                        .vertex_one_ring(i)
                        .map(|j| mesh.vertex_one_ring_len(j))
                        .sum::<usize>()
                        == 20)
                    .then_some(i)
            })
            .collect::<SetType<_>>();

        info!("Found {} single barnacle candidates", candidates.len());

        let invalid_candidates = candidates
            .par_iter()
            .copied()
            .filter_map(|c| {
                mesh.vertex_one_ring(c)
                    .any(|i| candidates.contains(&i))
                    .then_some(c)
            })
            .collect::<Vec<_>>();
        info!(
            "Filtered out {} adjacent candidates",
            invalid_candidates.len()
        );
        invalid_candidates.into_iter().for_each(|c| {
            candidates.remove(&c);
        });

        /*
        let mut max_angles = candidates
            .iter()
            .copied()
            .map(|c| {
                let mut max_angle = R::zero();
                for j in mesh.vertex_one_ring(c) {
                    if let Some(he) = mesh.half_edge(j, c) {
                        if let Ok(_) = mesh.is_collapse_ok(he) {
                            max_angle =
                                max_angle.max(mesh.half_edge_collapse_max_normal_change(he));
                            println!("max_angle: {}", max_angle);
                        }
                    }
                }
                println!(
                    "max angle change for config {}: {}\n\n",
                    c,
                    max_angle.to_f32().unwrap().to_degrees()
                );
                (c, max_angle.to_f32().unwrap().to_degrees())
            })
            .collect::<Vec<_>>();
        max_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dbg!(max_angles);
         */

        let half_edge_collapses = candidates
            .iter()
            .copied()
            .flat_map(|c| mesh.vertex_one_ring(c).map(move |i| (i, c)))
            .collect::<MapType<usize, usize>>();

        info!("Enqueued {} collapse operations", half_edge_collapses.len());

        half_edge_collapses
    };

    process_collapse_queue_iterative(mesh, half_edge_collapses.iter().map(|(i, j)| (*i, *j)));
}

pub fn merge_double_barnacle_configurations_he<R: Real>(mesh: &mut HalfEdgeTriMesh<R>) {
    profile!("merge_double_barnacle_configurations");

    let is_center_candidate = |i: usize| -> bool {
        if mesh.vertex_one_ring_len(i) == 5 {
            let mut neighbor_count =
                std::array::from_fn(|j| mesh.vertex_one_ring_len(mesh.vertex_one_ring_ith(i, j)));
            neighbor_count.sort_unstable();
            neighbor_count == [5, 5, 5, 6, 6]
        } else {
            false
        }
    };

    let half_edge_collapses = {
        profile!("find candidates");

        let sorted_pair = |i: usize, j: usize| -> (usize, usize) {
            let a = i.min(j);
            let b = j.max(i);
            (a, b)
        };

        let mut candidate_pairs = mesh
            .vertices
            .par_iter()
            .enumerate()
            .filter_map(|(i, _v_i)| {
                if is_center_candidate(i) {
                    let mut count = 0;
                    let mut other = 0;

                    // Identify the second center vertex of the double config
                    for j in mesh.vertex_one_ring(i) {
                        if is_center_candidate(j) {
                            count += 1;
                            other = j;
                        }
                    }

                    // Sort the candidate pair to avoid duplicates
                    (count == 1).then_some(sorted_pair(i, other))
                } else {
                    None
                }
            })
            .collect::<SetType<_>>();

        info!("Found {} double barnacle candidates", candidate_pairs.len(),);

        // First filter out any candidate pair with one vertex of the pair being part of another pair
        // This allows to make a unique vertex -> pair mapping in the next step
        {
            let is_double_candidate_overlapping = |i: usize, j: usize| {
                let pair = sorted_pair(i, j);
                mesh.vertex_one_ring(i).any(|k| {
                    let other_pair = sorted_pair(i, k);
                    // Only filter out one side of the adjacency
                    let is_smaller = other_pair < pair;
                    k != j && is_smaller && candidate_pairs.contains(&other_pair)
                })
            };

            let invalid_candidates = candidate_pairs
                .par_iter()
                .copied()
                .filter_map(|(i, j)| {
                    (is_double_candidate_overlapping(i, j) || is_double_candidate_overlapping(j, i))
                        .then_some((i, j))
                })
                .collect::<Vec<_>>();
            info!(
                "Filtered out {} overlapping candidates",
                invalid_candidates.len()
            );

            invalid_candidates.into_iter().for_each(|c| {
                candidate_pairs.remove(&c);
            });
        }

        // Filter out any candidate pairs whose neighbors overlap with another candidate pair
        {
            // Collect unique mapping of all center vertices to their candidate pair
            let mut candidate_to_pair_map = MapType::default();
            for (i, j) in candidate_pairs.iter().copied() {
                candidate_to_pair_map.insert(i, sorted_pair(i, j));
                candidate_to_pair_map.insert(j, sorted_pair(i, j));
            }

            // Checks if any neighbor of `i` has a neighbor that belongs to a different candidate pair than `(i,j)`
            let is_double_candidate_adjacent = |i: usize, j: usize| -> bool {
                let pair = sorted_pair(i, j);
                mesh.vertex_one_ring(i)
                    .filter(|k| *k != j)
                    .flat_map(|k| mesh.vertex_one_ring(k).filter(move |&l| l != i && l != j))
                    .any(|l| {
                        if let Some(other_pair) = candidate_to_pair_map.get(&l) {
                            return *other_pair < pair;
                        }
                        false
                    })
            };

            let invalid_candidates = candidate_pairs
                .par_iter()
                .copied()
                .filter_map(|(i, j)| {
                    (is_double_candidate_adjacent(i, j) || is_double_candidate_adjacent(j, i))
                        .then_some((i, j))
                })
                .collect::<Vec<_>>();
            info!(
                "Filtered out {} adjacent neighbor candidates",
                invalid_candidates.len()
            );

            invalid_candidates.into_iter().for_each(|c| {
                candidate_pairs.remove(&c);
            });
        }

        // Collect the actual half-edge collapses to perform
        let mut half_edge_collapses = MapType::default();
        for (i, j) in candidate_pairs {
            let mut insert_replacement = |i: usize, j: usize, k: usize| {
                if k != j {
                    if mesh.vertex_one_ring(k).all(|l| l != j) {
                        half_edge_collapses.insert(k, i);
                    } else if (mesh.vertices[k] - mesh.vertices[i]).norm()
                        <= (mesh.vertices[k] - mesh.vertices[j]).norm()
                    {
                        half_edge_collapses.insert(k, i);
                    } else {
                        half_edge_collapses.insert(k, j);
                    }
                }
            };

            for k in mesh.vertex_one_ring(i) {
                insert_replacement(i, j, k);
            }

            for k in mesh.vertex_one_ring(j) {
                insert_replacement(j, i, k);
            }
        }

        info!("Enqueued {} collapse operations", half_edge_collapses.len());

        half_edge_collapses
    };

    process_collapse_queue_iterative(mesh, half_edge_collapses.iter().map(|(i, j)| (*i, *j)));
}

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
    let sqrt_two = R::from_float(2.0_f64.sqrt());

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
        } else if tri_j.contains(&tri_i[1]) {
            quad[1] = tri_i[1];
            quad[2] = missing_vertex;
            quad[3] = tri_i[2];
        } else {
            panic!("this should not happen");
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
                                convert = dot_i_j >= min_dot;
                            }

                            if convert {
                                // Return the two triangle indices that should be merged
                                return Some(((i, j), quality));
                            }
                        }
                    }
                }
            }

            None
        })
        .collect::<MapType<_, _>>();

    info!(
        "Number of quad conversion candidates: {}",
        quad_candidates.len()
    );

    let mut triangles_to_remove = SetType::default();
    let mut filtered_candidates = SetType::default();
    for ((i, j), _q) in quad_candidates {
        // TODO: If triangle already exists in list, compare quality

        if !triangles_to_remove.contains(&i) && !triangles_to_remove.contains(&j) {
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
    cells.extend(filtered_triangles.into_iter().map(TriangleOrQuadCell::Tri));
    cells.extend(quads.into_iter().map(TriangleOrQuadCell::Quad));

    MixedTriQuadMesh3d {
        vertices: mesh.vertices.clone(),
        cells,
    }
}
