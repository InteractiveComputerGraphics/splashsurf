# pyright: reportInvalidTypeForm=false
import bpy
import time
import numpy as np

from .handlers import generate_mesh, cached_meshes
from .utils import *

class SPSF_OT_Register(bpy.types.Operator):
    """Register active object"""
    bl_idname = "splashsurf_studio.register"
    bl_label = "Register Object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or not obj.type == 'MESH':
            return False
        
        return True

    def execute(self, context):
        obj = context.active_object
        
        # Check if object is already registered or is reconstructed mesh od registered object
        for entry in context.scene.splashsurf_studio.data_objs:
            if entry.data_pointer == obj:
                self.report({'WARNING'}, "Object is already registered.")
                return {'CANCELLED'}
            if entry.surface_pointer == obj:
                self.report({'WARNING'}, "Object is reconstructed mesh from a registered object.")
                return {'CANCELLED'}
        
        obj_props = context.scene.splashsurf_studio.data_objs.add()
        obj_props.data_pointer = obj

        obj.viewport_reconstruction_properties.attributes.clear()
        obj.render_reconstruction_properties.attributes.clear()
        obj.render_reconstruction_properties.type = 'RENDER'

        for attr in obj.data.attributes:
            if attr.domain != 'POINT':
                continue
            
            new_attr = obj.viewport_reconstruction_properties.attributes.add()
            new_attr.name = attr.name

            new_attr = obj.render_reconstruction_properties.attributes.add()
            new_attr.name = attr.name
        
        return {'FINISHED'}
    
class SPSF_OT_Unregister(bpy.types.Operator):
    """Unregister selected data object"""
    bl_idname = "splashsurf_studio.unregister"
    bl_label = "Unregister Object"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.splashsurf_studio.data_objs) > 0

    def execute(self, context):
        context.scene.splashsurf_studio.data_objs.remove(context.scene.splashsurf_studio.selected_data_obj_index)
        return {'FINISHED'}

class SPSF_OT_Activate(bpy.types.Operator):
    """Activate surface reconstruction"""
    bl_idname = "splashsurf_studio.activate"
    bl_label = "Activate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return not context.scene.splashsurf_studio.active
    
    def execute(self, context):
        context.scene.render.use_lock_interface = True
        context.scene.splashsurf_studio.rendering = False
        context.scene.splashsurf_studio.active = True
        
        generate_mesh(context.scene)
        
        return {'FINISHED'}
    
class SPSF_OT_Deactivate(bpy.types.Operator):
    """Deactivate surface reconstruction"""
    bl_idname = "splashsurf_studio.deactivate"
    bl_label = "Deactivate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.splashsurf_studio.active
    
    def execute(self, context):
        context.scene.splashsurf_studio.active = False
        return {'FINISHED'}

class SPSF_OT_Copy_To_Render(bpy.types.Operator):
    """Copy viewport settings to render settings"""
    bl_idname = "splashsurf_studio.copy_to_render"
    bl_label = "Copy to Render"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = get_selected_data_object(context)
        
        viewport_props = obj.viewport_reconstruction_properties
        render_props = obj.render_reconstruction_properties
        
        copy_reconstruction_props(viewport_props, render_props)
        
        render_props.type = 'RENDER'
        
        return {'FINISHED'}

class SPSF_OT_Copy_To_Viewport(bpy.types.Operator):
    """Copy viewport settings to render settings"""
    bl_idname = "splashsurf_studio.copy_to_viewport"
    bl_label = "Copy to Viewport"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = get_selected_data_object(context)
        
        viewport_props = obj.viewport_reconstruction_properties
        render_props = obj.render_reconstruction_properties
        
        copy_reconstruction_props(render_props, viewport_props)
        
        return {'FINISHED'}

class SPSF_OT_Test_Render_Params(bpy.types.Operator):
    """Test render parameters and return information"""
    bl_idname = "splashsurf_studio.test_render_params"
    bl_label = "Test Render Params"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = get_selected_data_object(context)
        
        props = obj.render_reconstruction_properties
        
        start_time = time.time()
        
        mesh_with_data, error = get_reconstructed_mesh_from_props(obj, props)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        n_verts = len(mesh_with_data.mesh.vertices)
        n_tris = len(mesh_with_data.mesh.triangles)
        
        memory_bytes = 4 * n_verts * 3 + 4 * n_tris * 3
        
        for _, data in mesh_with_data.point_attributes.items():
            if len(data.shape) > 1 and data.shape[1] == 3:
                memory_bytes += 4 * n_verts * 3  # FLOAT_VECTOR
            elif len(data.shape) == 1 or data.shape[1] == 1:
                if data.dtype == np.float32:
                    memory_bytes += 4 * n_verts  # FLOAT
                elif data.dtype == np.uint64:
                    memory_bytes += 4 * n_verts  # INT
        
        
        self.report({'INFO'}, f"Reconstruction completed in {elapsed_time:.2f} seconds. "
                             f"Vertices: {len(mesh_with_data.mesh.vertices)}, "
                             f"Triangles: {len(mesh_with_data.mesh.triangles)}. "
                             f"Estimated memory size: {memory_bytes / (1024 * 1024):.2f} MB")
        return {'FINISHED'}

class SPSF_OT_Parse_CLI_Args(bpy.types.Operator):
    """Parse CLI argument string and set properties"""
    bl_idname = "splashsurf_studio.parse_cli_string"
    bl_label = "Parse CLI String"
    bl_options = {'REGISTER', 'UNDO'}
    
    cli_string: bpy.props.StringProperty(name="CLI Args", default="")
    is_render_props: bpy.props.BoolProperty(default=False)
    
    def execute(self, context):
        obj = get_selected_data_object(context)
        
        props = obj.render_reconstruction_properties if self.is_render_props else obj.viewport_reconstruction_properties
        
        args = get_args_from_cli_string(self.cli_string)
        
        args_to_props(args, props)
        
        return {"FINISHED"}
    
    def invoke(self, context, event=None):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "cli_string", text="CLI Args")

class SPSF_OT_Get_CLI_String(bpy.types.Operator):
    """Copy CLI argument string from properties to clipboard"""
    bl_idname = "splashsurf_studio.get_cli_string"
    bl_label = "Get CLI String"
    bl_options = {'REGISTER', 'UNDO'}
    
    is_render_props: bpy.props.BoolProperty(default=False)
    
    def execute(self, context):
        obj = get_selected_data_object(context)
        
        props = obj.render_reconstruction_properties if self.is_render_props else obj.viewport_reconstruction_properties
        
        cli_string = props_to_cli_string(props)
        
        self.report({'INFO'}, cli_string)
        return {'FINISHED'}
    
class SPSF_OT_Clear_Errors(bpy.types.Operator):
    """Clear Error Log"""
    bl_idname = "splashsurf_studio.clear_errors"
    bl_label = "Clear Errors"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        context.scene.splashsurf_studio.errors.clear()
        return {'FINISHED'}

class SPSF_OT_Show_Error(bpy.types.Operator):
    """Show Error Log"""
    bl_idname = "splashsurf_studio.show_error"
    bl_label = "Show Error"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        self.report({'ERROR'}, context.scene.splashsurf_studio.errors[context.scene.splashsurf_studio.selected_error_index].error)
        return {'FINISHED'}

class SPSF_OT_Add_Frames(bpy.types.Operator):
    """Add cached frames"""
    bl_idname = "splashsurf_studio.add_frames"
    bl_label = "Add Frames"
    bl_options = {'REGISTER', 'UNDO'}
    
    frame_string: bpy.props.StringProperty(name="Frame", default="0")
    
    def execute(self, context):
        frames = parse_frame_string(self.frame_string)
        if frames == []:
            self.report({'WARNING'}, "No valid frames found in input.")
            return {'CANCELLED'}
        existing_frames = [f.frame for f in context.scene.splashsurf_studio.cached_frames]
        frames = frames+existing_frames
        frames.sort()
        context.scene.splashsurf_studio.cached_frames.clear()
        for frame in frames:
            new_frame = context.scene.splashsurf_studio.cached_frames.add()
            new_frame.frame = frame
        return {'FINISHED'}
    
    def invoke(self, context, event=None):
        self.frame_string = str(context.scene.frame_current)
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "frame_string", text="Frames")

class SPSF_OT_Remove_Frame(bpy.types.Operator):
    """Remove cached frame"""
    bl_idname = "splashsurf_studio.remove_frame"
    bl_label = "Remove Frame"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.splashsurf_studio.cached_frames) > 0
    
    def execute(self, context):
        frame = context.scene.splashsurf_studio.cached_frames[context.scene.splashsurf_studio.selected_cached_frame_index].frame
        
        if frame in cached_meshes:
            for obj in cached_meshes[frame]:
                mesh = cached_meshes[frame][obj]
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(cached_meshes[frame][obj])
            del cached_meshes[frame]
        
        context.scene.splashsurf_studio.cached_frames.remove(context.scene.splashsurf_studio.selected_cached_frame_index)
        return {'FINISHED'}

class SPSF_OT_Clear_Cache(bpy.types.Operator):
    """Clear all cached frames"""
    bl_idname = "splashsurf_studio.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_cache(cached_meshes)
        context.scene.splashsurf_studio.cached_frames.clear()
        return {'FINISHED'}

class SPSF_OT_Regenerate_Cache(bpy.types.Operator):
    """Regenerate all cached frames (May take a while)"""
    bl_idname = "splashsurf_studio.regenerate_cache"
    bl_label = "Regenerate Cache"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.splashsurf_studio.cached_frames) > 0 and context.scene.splashsurf_studio.active
    
    def execute(self, context):
        clear_cache(cached_meshes)
        
        # The frames are cached in generate_mesh which is called when we set the frame
        current_frame = context.scene.frame_current
        for part in context.scene.splashsurf_studio.cached_frames:
            context.scene.frame_set(part.frame)
        
        context.scene.frame_set(current_frame)
        
        return {'FINISHED'}

class SPSF_OT_Jump_To_Frame(bpy.types.Operator):
    """Jump to cached frame"""
    bl_idname = "splashsurf_studio.jump_to_frame"
    bl_label = "Jump to Frame"
    bl_options = {'REGISTER', 'UNDO'}
    
    frame: bpy.props.IntProperty()
    
    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {'FINISHED'}