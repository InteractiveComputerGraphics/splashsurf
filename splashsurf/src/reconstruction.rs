use anyhow::{anyhow, Context};
use coarse_prof::profile;
use log::info;
use splashsurf_lib::mesh::PointCloud3d;
use splashsurf_lib::{density_map, Index, Real};

use crate::io;
use crate::{ReconstructionRunnerArgs, ReconstructionRunnerPaths};

// TODO: Detect smallest index type (i.e. check if ok to use i32/u32 as index)

pub(crate) fn entry_point_f64(
    paths: &ReconstructionRunnerPaths,
    args: &ReconstructionRunnerArgs,
) -> Result<(), anyhow::Error> {
    if args.use_double_precision {
        type I = i64;
        type R = f64;

        entry_point_generic::<I, R>(paths, &args.params)?;
    } else {
        type I = i64;
        type R = f32;

        entry_point_generic::<I, R>(
            paths,
            &args.params.try_convert().ok_or(anyhow!(
                "Unable to convert surface reconstruction parameters from f64 to f32."
            ))?,
        )?;
    }

    Ok(())
}

pub(crate) fn entry_point_generic<I: Index, R: Real>(
    paths: &ReconstructionRunnerPaths,
    params: &splashsurf_lib::Parameters<R>,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction cli");

    info!("Loading dataset from \"{}\"...", paths.input_file.display());
    let particle_positions = if let Some(extension) = paths.input_file.extension() {
        profile!("loading particle positions");

        let extension = extension.to_str().ok_or(anyhow!(
            "Invalid extension of of input file '{}'",
            paths.input_file.display()
        ))?;

        match extension.to_lowercase().as_str() {
            "vtk" => {
                let sph_dataset = io::read_vtk(&paths.input_file)?;
                io::particles_from_dataset(&sph_dataset)?
            }
            "xyz" => io::particles_from_xyz(&paths.input_file)?,
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension '{}' of input file '{}'",
                    extension,
                    paths.input_file.display()
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of input file '{}' (file name has to end with supported extension)",
            paths.input_file.display()
        ));
    };

    info!(
        "Loaded dataset with {} particle positions.",
        particle_positions.len()
    );

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
        io::write_vtk(mesh, &paths.output_file, "mesh").with_context(|| {
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

        io::write_vtk(
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

        io::write_vtk(
            density_mesh.to_dataset(),
            output_density_map_grid_file,
            "density_map",
        )?;

        info!("Done.");
    }

    Ok(())
}
