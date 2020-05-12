# splashsurf
Surface reconstruction for particle data from SPH simulations, written in Rust.

## Usage

```
USAGE:
    splashsurf [FLAGS] [OPTIONS] <input-file> --cube-size <cube-size> --kernel-radius <kernel-radius> --particle-radius <particle-radius> --surface-threshold <surface-threshold>

FLAGS:
    -h, --help            Prints help information
        --mt-files        Whether to allow multi-threading to process multiple input files at once, conflicts with --mt-
                          particles
        --mt-particles    Whether to allow multi-threading internally for a single input file over particles, conflicts
                          with --mt-files
    -d                    Optional filename for writing the intermediate density map grid to disk
    -V, --version         Prints version information

OPTIONS:
        --cube-size <cube-size>
            The marching cubes grid size in multiplies of the particle radius

        --domain-max <domain-max> <domain-max> <domain-max>
            Upper corner of the domain where surface reconstruction should be performed, format:domain-
            max=x_max;y_max;z_max (requires domain-min to be specified)
        --domain-min <domain-min> <domain-min> <domain-min>
            Lower corner of the domain where surface reconstruction should be performed, format: domain-
            min=x_min;y_min;z_min (requires domain-max to be specified)
        --kernel-radius <kernel-radius>
            The kernel radius used for building the density map in multiplies of the particle radius

        --output-dir <output-dir>                              Optional base directory for all output files
        --output-dm-grid <output-dm-grid>
            Optional filename for writing the intermediate density map grid to disk

        --output-dm-points <output-dm-points>
            Optional filename for writing the intermediate density map point cloud to disk

    -o <output-file>                                           Filename for writing the reconstructed surface to disk
        --particle-radius <particle-radius>                    The particle radius of the input data
        --rest-density <rest-density>                          The rest density of the fluid [default: 1000.0]
        --splash-detection-radius <splash-detection-radius>
            If a particle has no neighbors in this radius (in multiplies of the particle radius), it is considered as a
            free particles
        --surface-threshold <surface-threshold>
            The iso-surface threshold for the density, i.e. at what reconstructed density does the fluid start


ARGS:
    <input-file>    Path to the input file where the particle positions are stored (supported formats: VTK, binary
                    XYZ)
```
For example:
```
splashsurf "/home/user/canyon.xyz" --output-dir="/home/user/temp" --particle-radius=0.011 --kernel-radius=4.0 --cube-size=1.5 --surface-threshold=0.6 --mt-particles
```
