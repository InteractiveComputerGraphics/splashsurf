import bpy
import numpy as np

from .utils import get_reconstructed_mesh_from_props, build_mesh_from_verts_tris, copy_materials, get_valid_surface_object

cached_meshes = {}
uncached_mesh = None

def update_reconstruction(scene, entry, use_render_props):
    cloud = entry.data_pointer
    if cloud is None:
        print(f"Object '{entry.data_pointer}' not found in the scene.")
        return
    
    surface = get_valid_surface_object(scene, cloud, entry)
    
    if surface.data.cached:
        global uncached_mesh
        old_mesh = surface.data
        if uncached_mesh is not None:
            surface.data = uncached_mesh
        else: # Actually should never happen
            surface.data = bpy.data.meshes.new(name=f"{cloud.name}_Mesh")
        
        copy_materials(old_mesh, surface.data)
            
        uncached_mesh = None
    
    props = cloud.render_reconstruction_properties if use_render_props else cloud.viewport_reconstruction_properties
    
    mesh_with_data, error = get_reconstructed_mesh_from_props(cloud, props)
    
    if error:
        new_error = scene.splashsurf_studio.errors.add()
        new_error.error = error
        new_error.obj = cloud.name
        return
    
    mesh = mesh_with_data.mesh
    verts = mesh.vertices
    tris = mesh.triangles
    
    # Update mesh
    mesh = surface.data
    build_mesh_from_verts_tris(mesh, verts, tris)
    
    # Set normals
    if props.compute_normals and props.set_split_normals:
        normals = mesh_with_data.point_attributes["normals"]
        mesh.normals_split_custom_set_from_vertices(normals)
    
    # Add all attributes to mesh
    for attr_name in mesh_with_data.point_attributes:
        data = mesh_with_data.point_attributes[attr_name]
        if len(data.shape) > 1 and data.shape[1] == 3:
            attribute = mesh.attributes.new(name=attr_name, type='FLOAT_VECTOR', domain='POINT')
            attribute.data.foreach_set("vector", data.ravel())
        elif len(data.shape) == 1 or data.shape[1] == 1:
            if data.dtype == np.float32:
                attribute = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
                attribute.data.foreach_set("value", data)
            elif data.dtype == np.uint64:
                attribute = mesh.attributes.new(name=attr_name, type='INT', domain='POINT')
                attribute.data.foreach_set("value", data)
        else:
            print(f"Unsupported attribute shape {data.shape} or dtype {data.dtype} for attribute '{attr_name}'")
    
    mesh.update()

def update_entries(scene):
    frame = scene.frame_current
    
    # Check if frame should be cached
    cache_frame = False
    for part in scene.splashsurf_studio.cached_frames:
        if frame == part.frame:
            cache_frame = True
            break
    
    # Use correct reconstruction properties
    use_render_props = scene.splashsurf_studio.use_render_for_viewport or scene.splashsurf_studio.rendering
    if cache_frame:
        match scene.splashsurf_studio.cache_settings:
            case 'VIEWPORT':
                use_render_props = False
            case 'RENDER':
                use_render_props = True
    
    use_cache = scene.splashsurf_studio.use_cache_during_render or not scene.splashsurf_studio.rendering
    
    for entry in scene.splashsurf_studio.data_objs:
        if not entry.generate:
            continue
        
        if use_cache and frame in cached_meshes and entry.data_pointer in cached_meshes[frame]:
            cloud = entry.data_pointer
            if cloud is None:
                print(f"Object '{entry.data_pointer}' not found in the scene.")
                continue
            
            # Try to get old surface object
            surface = get_valid_surface_object(scene, cloud, entry)
            
            old_mesh = surface.data
            if old_mesh is not None and not old_mesh.cached:
                global uncached_mesh
                uncached_mesh = old_mesh
            
            surface.data = cached_meshes[frame][entry.data_pointer]
            
            copy_materials(old_mesh, surface.data)
            
        else:
            update_reconstruction(scene, entry, use_render_props)
            
            if use_cache and cache_frame:
                if frame not in cached_meshes:
                    cached_meshes[frame] = {}
                
                mesh = entry.surface_pointer.data
                mesh.cached = True
                
                cached_meshes[frame][entry.data_pointer] = mesh
