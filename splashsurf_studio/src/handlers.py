import bpy
from bpy.app.handlers import persistent

from .utils import get_selected_data_object
from .updater import update_entries, update_reconstruction, cached_meshes

def generate_mesh(scene):
    if not scene.splashsurf_studio.active:
        return
    
    update_entries(scene)

def toggle_rendering_on(scene):
    scene.splashsurf_studio.rendering = True

def toggle_rendering_off(scene):
    scene.splashsurf_studio.rendering = False

@persistent
def ensure_frame_change_handlers(_):
    if generate_mesh not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(generate_mesh)
    if toggle_rendering_on not in bpy.app.handlers.render_init:
        bpy.app.handlers.render_init.append(toggle_rendering_on)
    if toggle_rendering_off not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(toggle_rendering_off)
    if toggle_rendering_off not in bpy.app.handlers.render_cancel:
        bpy.app.handlers.render_cancel.append(toggle_rendering_off)

def property_callback(self, context):
    if context.scene.splashsurf_studio.update_on_change and context.scene.splashsurf_studio.active:
        obj = get_selected_data_object(context)
        scene = context.scene
        frame = scene.frame_current
        
        cache_frame = False
        for part in scene.splashsurf_studio.cached_frames:
            if frame == part.frame:
                cache_frame = True
                break
        
        # Only regenerate if edited property would get used for current reconstruction
        if cache_frame:
            if scene.splashsurf_studio.cache_settings != self.type:
                return
        elif (scene.splashsurf_studio.use_render_for_viewport and self.type == 'VIEWPORT') or (not scene.splashsurf_studio.use_render_for_viewport and self.type == 'RENDER'):
            return
        
        # Find the corresponding data object entry
        for entry in scene.splashsurf_studio.data_objs:
            if entry.data_pointer == obj:
                
                # Use correct reconstruction properties
                use_render_props = scene.splashsurf_studio.use_render_for_viewport or scene.splashsurf_studio.rendering
                if cache_frame:
                    match scene.splashsurf_studio.cache_settings:
                        case 'VIEWPORT':
                            use_render_props = False
                        case 'RENDER':
                            use_render_props = True
                    
                    # Set cached flag to false so that update_reconstruction reuses the mesh data object
                    if entry.surface_pointer is not None and entry.surface_pointer.data.cached:
                        entry.surface_pointer.data.cached = False
                
                update_reconstruction(scene, entry, use_render_props)
                
                if cache_frame:
                    if frame not in cached_meshes:
                        cached_meshes[frame] = {}
                    
                    entry.surface_pointer.data.cached = True
                    cached_meshes[frame][entry.data_pointer] = entry.surface_pointer.data
                
                break

# Callback to update the reconstruction once when the "Update on Change" property is turned on
def update_callback(_, context):
    # Call property callback for both reconstruction properties, property_callback will abort for the not used type
    property_callback(get_selected_data_object(context).viewport_reconstruction_properties, context)
    property_callback(get_selected_data_object(context).render_reconstruction_properties, context)
    
def update_copy_transforms_constraints(self, context):
    if self.match_transforms:
        # Add constraint to all data objects
        for entry in context.scene.splashsurf_studio.data_objs:
            if entry.surface_pointer is not None:
                # Check if constraint already exists
                exists = False
                for constraint in entry.surface_pointer.constraints:
                    if constraint.type == 'COPY_TRANSFORMS' and constraint.target == entry.data_pointer:
                        exists = True
                        break
                
                if not exists:
                    constraint = entry.surface_pointer.constraints.new(type='COPY_TRANSFORMS')
                    constraint.target = entry.data_pointer     
    else:
        # Remove constraints from all data objects
        for entry in context.scene.splashsurf_studio.data_objs:
            if entry.surface_pointer is not None:
                for constraint in entry.surface_pointer.constraints:
                    if constraint.type == 'COPY_TRANSFORMS' and constraint.target == entry.data_pointer:
                        entry.surface_pointer.constraints.remove(constraint)
                        break
                    