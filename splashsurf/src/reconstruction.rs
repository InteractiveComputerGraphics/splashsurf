use anyhow::{anyhow, Context};
use coarse_prof::profile;
use log::info;
use splashsurf_lib::mesh::PointCloud3d;
use splashsurf_lib::{density_map, Index, Real};

use crate::io;
use crate::{ReconstructionRunnerArgs, ReconstructionRunnerPaths};

// TODO: Detect smallest index type (i.e. check if ok to use i32/u32 as index)

pub(crate) fn entry_point(
    paths: &ReconstructionRunnerPaths,
    args: &ReconstructionRunnerArgs,
) -> Result<(), anyhow::Error> {
    if args.use_double_precision {
        type I = i64;
        type R = f64;

        entry_point_generic::<I, R>(paths, &args.params, &args.io_params)?;
    } else {
        type I = i64;
        type R = f32;

        entry_point_generic::<I, R>(
            paths,
            &args.params.try_convert().ok_or(anyhow!(
                "Unable to convert surface reconstruction parameters from f64 to f32."
            ))?,
            &args.io_params,
        )?;
    }

    Ok(())
}

pub(crate) fn entry_point_generic<I: Index, R: Real>(
    paths: &ReconstructionRunnerPaths,
    params: &splashsurf_lib::Parameters<R>,
    io_params: &io::FormatParameters,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction cli");

    let particle_positions = io::load_particle_positions(&paths.input_file, &io_params.input)
        .with_context(|| {
            format!(
                "Failed to load particle positions from file '{}'",
                paths.input_file.display()
            )
        })?;

    let reconstruction =
        splashsurf_lib::reconstruct_surface::<I, R>(particle_positions.as_slice(), &params)?;

    let grid = reconstruction.grid();
    let density_map = reconstruction.density_map();
    let mesh = reconstruction.mesh();

    {
        profile!("write surface mesh to file");
        info!(
            "Writing surface mesh to \"{}\"...",
            paths.output_file.to_string_lossy()
        );
        io::vtk_format::write_vtk(mesh, &paths.output_file, "mesh").with_context(|| {
            format!(
                "Failed to write reconstructed surface to output file '{}'",
                paths.output_file.to_string_lossy()
            )
        })?;
        info!("Done.");
    }

    if let Some(output_density_map_points_file) = &paths.output_density_map_points_file {
        info!("Constructing density map point cloud...");

        let point_cloud: PointCloud3d<R> = {
            let mut points = Vec::with_capacity(density_map.len());
            for (flat_point_index, _) in density_map.iter() {
                let point = grid.try_unflatten_point_index(flat_point_index).unwrap();
                points.push(grid.point_coordinates(&point));
            }

            PointCloud3d { points }
        };

        info!(
            "Saving density map point cloud to \"{}\"...",
            output_density_map_points_file.to_string_lossy()
        );

        io::vtk_format::write_vtk(
            &point_cloud,
            output_density_map_points_file,
            "density_map_points",
        )?;

        info!("Done.");
    }

    if let Some(output_density_map_grid_file) = &paths.output_density_map_grid_file {
        info!("Constructing density map hex mesh...");

        let density_mesh =
            density_map::sparse_density_map_to_hex_mesh(&density_map, &grid, R::zero());

        info!(
            "Saving density map hex mesh to \"{}\"...",
            output_density_map_grid_file.to_string_lossy()
        );

        io::vtk_format::write_vtk(
            density_mesh.to_dataset(),
            output_density_map_grid_file,
            "density_map",
        )?;

        info!("Done.");
    }

    Ok(())
}
