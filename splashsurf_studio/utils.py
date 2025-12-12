import pysplashsurf
import numpy as np
import bpy

def get_selected_data_object(context):
    """Get the currently selected object in the scene that is registered with splashsurf_studio."""
    obj = context.active_object
    if not obj:
        return None
    
    for entry in context.scene.splashsurf_studio.data_objs:
        if entry.data_pointer == obj or entry.surface_pointer == obj:
            return entry.data_pointer
        
    return None

def copy_reconstruction_props(source_props, target_props):
    target_props.attributes.clear()
    for attr in source_props.attributes:
        new_attr = target_props.attributes.add()
        new_attr.name = attr.name
        new_attr.interpolate = attr.interpolate
    
    target_props.particle_radius = source_props.particle_radius
    target_props.rest_density = source_props.rest_density
    target_props.cube_size = source_props.cube_size
    target_props.smoothing_length = source_props.smoothing_length
    target_props.iso_surface_threshold = source_props.iso_surface_threshold
    target_props.output_smoothing_weights = source_props.output_smoothing_weights
    target_props.mesh_smoothing_weights_normalization = source_props.mesh_smoothing_weights_normalization
    target_props.mesh_smoothing_iters = source_props.mesh_smoothing_iters
    target_props.normals_smoothing_iters = source_props.normals_smoothing_iters
    target_props.compute_normals = source_props.compute_normals
    target_props.sph_normals = source_props.sph_normals
    target_props.set_split_normals = source_props.set_split_normals
    target_props.mesh_cleanup = source_props.mesh_cleanup
    target_props.mesh_cleanup_limit_snapping_distance = source_props.mesh_cleanup_limit_snapping_distance
    target_props.mesh_cleanup_snapping_distance = source_props.mesh_cleanup_snapping_distance
    target_props.subdomain_grid = source_props.subdomain_grid
    target_props.subdomain_grid_auto_disable = source_props.subdomain_grid_auto_disable
    target_props.subdomain_num_cubes_per_dim = source_props.subdomain_num_cubes_per_dim
    target_props.particle_aabb = source_props.particle_aabb
    target_props.particle_aabb_min = source_props.particle_aabb_min
    target_props.particle_aabb_max = source_props.particle_aabb_max
    target_props.mesh_aabb = source_props.mesh_aabb
    target_props.mesh_aabb_min = source_props.mesh_aabb_min
    target_props.mesh_aabb_max = source_props.mesh_aabb_max

def get_reconstructed_mesh_from_props(data_obj, props):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    data_obj_eval = data_obj.evaluated_get(depsgraph)
    mesh_eval = data_obj_eval.to_mesh()
    
    n = len(mesh_eval.vertices)
    particles = np.empty(n*3, dtype=np.float32)  
    mesh_eval.vertices.foreach_get('co', particles)  
    particles.shape = (n, 3)
    
    # Collect attributes to interpolate
    attributes = {}
    for attr in props.attributes:
        if not attr.interpolate:
            continue

        if attr.name in data_obj.data.attributes:
            attr = data_obj.data.attributes[attr.name]
            if attr.data_type == 'FLOAT_VECTOR':
                data = np.empty(n*3, dtype=np.float32)
                attr.data.foreach_get("vector", data)
                data.shape = (n, 3)
                attributes[attr.name] = data
                
            elif attr.data_type == 'FLOAT':
                data = np.empty(n, dtype=np.float32)
                attr.data.foreach_get("value", data)
                attributes[attr.name] = data
                
            else:
                print(f"Unsupported attribute data type: {attr.data_type}; Only FLOAT_VECTOR and FLOAT are supported.")
                
        else:
            print(f"Attribute '{attr.name}' not found in the mesh data.")
    
    try:
        # Pysplashsurf reconstruction
        mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
            particles,
            attributes_to_interpolate=attributes,
            particle_radius=props.particle_radius,
            rest_density=props.rest_density,
            smoothing_length=props.smoothing_length,
            cube_size=props.cube_size,
            iso_surface_threshold=props.iso_surface_threshold,
            output_mesh_smoothing_weights=props.output_smoothing_weights,
            mesh_smoothing_weights=True if props.mesh_smoothing_iters > 0 else False,
            mesh_smoothing_weights_normalization=props.mesh_smoothing_weights_normalization,
            mesh_smoothing_iters=props.mesh_smoothing_iters,
            normals_smoothing_iters=props.normals_smoothing_iters,
            mesh_cleanup=props.mesh_cleanup,
            mesh_cleanup_snap_dist=props.mesh_cleanup_snapping_distance if props.mesh_cleanup_limit_snapping_distance else None,
            compute_normals=props.compute_normals,
            sph_normals=props.sph_normals,
            subdomain_grid=props.subdomain_grid,
            subdomain_grid_auto_disable=props.subdomain_grid_auto_disable,
            subdomain_num_cubes_per_dim=props.subdomain_num_cubes_per_dim,
            aabb_min= props.particle_aabb_min if props.particle_aabb else None,
            aabb_max= props.particle_aabb_max if props.particle_aabb else None,
            mesh_aabb_min= props.mesh_aabb_min if props.mesh_aabb else None,
            mesh_aabb_max= props.mesh_aabb_max if props.mesh_aabb else None,
            mesh_aabb_clamp_vertices=props.mesh_aabb_clamp_vertices
        )
        
        return mesh_with_data, None
    
    except BaseException as e:
        print(f"Error during reconstruction: {e}")
        return None, str(e)

def build_mesh_from_verts_tris(mesh, verts, tris):
    mesh.clear_geometry()
    mesh.update()
    mesh.validate()
    mesh.vertices.add(len(verts))
    mesh.polygons.add(len(tris))
    mesh.loops.add(len(tris) * 3)
    mesh.vertices.foreach_set("co", np.array(verts).ravel())
    loop_start = np.arange(len(tris)) * 3
    loop_total = np.full(len(tris), 3)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    mesh.polygons.foreach_set("use_smooth", np.full(len(tris), False))
    mesh.loops.foreach_set("vertex_index", np.array(tris).ravel())
    mesh.validate()

def create_empty_surface_object(scene, cloud, entry):
    collection = cloud.users_collection[0]
    
    placeholder_mesh = bpy.data.meshes.new("Mesh")
    surface = bpy.data.objects.new(cloud.name + "_surface", placeholder_mesh)
    collection.objects.link(surface)
    
    entry.surface_pointer = surface

    # Add Copy Transforms constraint
    if scene.splashsurf_studio.match_transforms:
        constraint = surface.constraints.new(type='COPY_TRANSFORMS')
        constraint.target = entry.data_pointer
    
    return surface

def get_valid_surface_object(scene, cloud, entry):
    # Try to get old surface object
    surface = entry.surface_pointer
    
    if surface is None:
        surface = create_empty_surface_object(scene, cloud, entry)
    
    # Only user is this entry, meaning it was deleted and we need to clean up
    if surface.users == 1:
        mesh = surface.data
        bpy.data.objects.remove(surface)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
            
        surface = create_empty_surface_object(scene, cloud, entry)
        
    return surface

def copy_materials(source_obj, target_obj):
    materials = source_obj.materials.values()
    target_obj.materials.clear()
    for mat in materials:
        target_obj.materials.append(mat)

def get_args_from_cli_string(cli_string):
    i = 0
    split_args = cli_string.split()
    args = {}
    while i < len(split_args):
        key = split_args[i]
        if key.startswith("--"):
            key = key[2:]
        elif key.startswith("-"):
            key = key[1:]
        else:
            i += 1
            continue
        
        if "=" in key:
            key, value = key.split("=", 1)
            args[key] = value
        else:
            if key == "a" or key == "interpolate_attribute":
                args["a"] = args.get("a", []) + [split_args[i + 1]] if i + 1 < len(split_args) else []
            elif key in ["particle-aabb-min", "particle-aabb-max", "mesh-aabb-min", "mesh-aabb-max"]:
                args[key] = [float(x) for x in split_args[i + 1:i + 4]] if i + 3 < len(split_args) else []
                i += 2
            else:
                args[key] = split_args[i + 1] if i + 1 < len(split_args) else None
            i += 1
        
        i += 1
    
    return args

def args_to_props(args: dict, props):
    
    attributes_to_interpolate = args.get("a", [])
    for attr in props.attributes:
        if attr.name in attributes_to_interpolate:
            attr.interpolate = True
        else:
            attr.interpolate = False
    
    props.particle_radius = float(args.get("r", 0.025))
    props.rest_density = float(args.get("rest-density", 1000.0))
    props.cube_size = float(args.get("c", 0.5))
    props.smoothing_length = float(args.get("l", 2.0))
    props.iso_surface_threshold = float(args.get("t", 0.6))
    props.output_smoothing_weights = args.get("output-smoothing-weights", "off").lower() == "on"
    props.mesh_smoothing_weights_normalization = float(args.get("mesh-smoothing-weights-normalization", 13.0))
    props.mesh_smoothing_iters = int(args.get("mesh-smoothing-iters", 25))
    props.normals_smoothing_iters = int(args.get("normals-smoothing-iters", 10))
    props.sph_normals = args.get("sph-normals", "off").lower() == "on"
    props.compute_normals = args.get("normals", "off").lower() == "on"
    props.set_split_normals = args.get("normals", "off").lower() == "on"
    props.mesh_cleanup = args.get("mesh-cleanup", "on").lower() == "on"
    props.mesh_cleanup_limit_snapping_distance = True if "mesh-cleanup-snap-dist" in args else False
    props.mesh_cleanup_snapping_distance = float(args.get("mesh-cleanup-snap-dist", 0.5))
    props.subdomain_grid = args.get("subdomain-grid", "on").lower() == "on"
    props.subdomain_grid_auto_disable = args.get("subdomain-grid-auto-disable", "on").lower() == "on"
    props.subdomain_num_cubes_per_dim = int(args.get("subdomain-cubes", 64))
    
    props.particle_aabb = True if "particle-aabb-min" in args else False
    props.particle_aabb_min = args.get("particle-aabb-min", [-1.0, -1.0, -1.0])
    props.particle_aabb_max = args.get("particle-aabb-max", [1.0, 1.0, 1.0])
    
    props.mesh_aabb = True if "mesh-aabb-min" in args else False
    props.mesh_aabb_min = args.get("mesh-aabb-min", [-1.0, -1.0, -1.0])
    props.mesh_aabb_max = args.get("mesh-aabb-max", [1.0, 1.0, 1.0])
    props.mesh_aabb_clamp_vertices = args.get("mesh-aabb-clamp-vertices", "off").lower() == "on"

def props_to_cli_string(props):
    cli_string = ""
    
    cli_string += f" -r {props.particle_radius}"
    cli_string += f" --rest-density {props.rest_density}"
    cli_string += f" -c {props.cube_size}"
    cli_string += f" -l {props.smoothing_length}"
    cli_string += f" -t {props.iso_surface_threshold}"
    
    for attr in props.attributes:
        if attr.interpolate:
            cli_string += f" -a {attr.name}"
    
    cli_string += " --output-smoothing-weights=on" if props.output_smoothing_weights else " --output-smoothing-weights=off"
    cli_string += f" --mesh-smoothing-weights-normalization={props.mesh_smoothing_weights_normalization}"
    cli_string += f" --mesh-smoothing-iters={props.mesh_smoothing_iters}"
    cli_string += f" --normals-smoothing-iters={props.normals_smoothing_iters}"
    cli_string += " --sph-normals=on" if props.sph_normals else " --sph-normals=off"
    cli_string += " --normals=on" if props.compute_normals else " --normals=off"
    cli_string += " --mesh-cleanup=on" if props.mesh_cleanup else " --mesh-cleanup=off"
    cli_string += f" --mesh-cleanup-snap-dist={props.mesh_cleanup_snapping_distance}" if props.mesh_cleanup_limit_snapping_distance else ""
    
    cli_string += " --subdomain-grid=on" if props.subdomain_grid else " --subdomain-grid=off"
    cli_string += " --subdomain-grid-auto-disable=on" if props.subdomain_grid_auto_disable else " --subdomain-grid-auto-disable=off"
    cli_string += f" --subdomain-cubes={props.subdomain_num_cubes_per_dim}"
    
    if props.particle_aabb:
        cli_string += f" --particle-aabb-min {' '.join(map(str, props.particle_aabb_min))}"
        cli_string += f" --particle-aabb-max {' '.join(map(str, props.particle_aabb_max))}"
    
    if props.mesh_aabb:
        cli_string += f" --mesh-aabb-min {' '.join(map(str, props.mesh_aabb_min))}"
        cli_string += f" --mesh-aabb-max {' '.join(map(str, props.mesh_aabb_max))}"
        cli_string += " --mesh-aabb-clamp-vertices=on" if props.mesh_aabb_clamp_vertices else " --mesh-aabb-clamp-vertices=off"
    
    return cli_string

def parse_frame_string(frame_string):
    frames = set()
    parts = frame_string.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            frames.update(range(int(start), int(end) + 1))
        else:
            frames.add(int(part))
    return sorted(frames)

def clear_cache(cached_meshes):
    if not cached_meshes:
        return
    
    print("Clearing cached meshes")
    for frame in cached_meshes:
        for obj in cached_meshes[frame]:
            mesh = cached_meshes[frame][obj]
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(cached_meshes[frame][obj])
    
    cached_meshes.clear()

def draw_attribute_list(layout, props):
    layout.template_list("SPSF_UL_Attribute_List", "", props, "attributes", props, "selected_attribute_num", rows=4)

def reconstruction_properties_panel(layout, props):
    layout.prop(props, "particle_radius")
    layout.prop(props, "rest_density")
    layout.prop(props, "cube_size")
    layout.prop(props, "smoothing_length")
    layout.prop(props, "iso_surface_threshold")
    
    layout.prop(props, "output_smoothing_weights")
    layout.prop(props, "mesh_smoothing_weights_normalization")
    layout.prop(props, "mesh_smoothing_iters")
    
    box = layout.box()
    box.prop(props, "compute_normals")
    if props.compute_normals:
        box.prop(props, "sph_normals")
        box.prop(props, "normals_smoothing_iters")
        box.prop(props, "set_split_normals")
    
    # Mesh cleanup
    box = layout.box()
    box.prop(props, "mesh_cleanup")
    if props.mesh_cleanup:
        box.prop(props, "mesh_cleanup_limit_snapping_distance")
        if props.mesh_cleanup_limit_snapping_distance:
            box.prop(props, "mesh_cleanup_snapping_distance")
    
    # Subdomain grid
    box = layout.box()
    box.prop(props, "subdomain_grid")
    if props.subdomain_grid:
        box.prop(props, "subdomain_grid_auto_disable")
        box.prop(props, "subdomain_num_cubes_per_dim")
    
    # Particle AABB
    box = layout.box()
    box.prop(props, "particle_aabb")
    if props.particle_aabb:
        box.prop(props, "particle_aabb_min")
        box.prop(props, "particle_aabb_max")
        
    # Mesh AABB
    box = layout.box()
    box.prop(props, "mesh_aabb")
    if props.mesh_aabb:
        box.prop(props, "mesh_aabb_min")
        box.prop(props, "mesh_aabb_max")
        box.prop(props, "mesh_aabb_clamp_vertices")
