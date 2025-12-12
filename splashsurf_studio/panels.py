import bpy
import importlib

from .utils import get_selected_data_object, reconstruction_properties_panel, draw_attribute_list

class SPSF_Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = "UI"
    bl_category = "Splashsurf Studio"
    bl_context = "objectmode"

class SPSF_UL_Attribute_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.prop(item, "interpolate", text=item.name)
        
class SPSF_UL_Data_Object_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row()
        row.prop(item, "generate", text="")
        row.label(text=item.data_pointer.name)

class SPSF_UL_Error_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=f"'{item.obj}': {item.error}")

class SPSF_UL_Frame_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=str(item.frame))
        
        op = layout.operator("splashsurf_studio.jump_to_frame", text="", icon='TRIA_RIGHT', emboss=False)
        op.frame = item.frame

class SPSF_Error_Log_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Error Log"
    bl_idname = "SPSF_PT_error_log"
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.splashsurf_studio.errors) > 0
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.template_list("SPSF_UL_Error_List", "", scene.splashsurf_studio, "errors", scene.splashsurf_studio, "selected_error_index", rows=2)
        
        layout.operator("splashsurf_studio.show_error", icon='INFO', text="Expand Error")
        layout.operator("splashsurf_studio.clear_errors", icon='TRASH')

class SPSF_Selection_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Select Input"
    bl_idname = "SPSF_PT_select"
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text=f"Using pySplashsurf v{importlib.metadata.version('pysplashsurf')}")
        
        row = layout.row()
        row.template_list("SPSF_UL_Data_Object_List", "", context.scene.splashsurf_studio, "data_objs", context.scene.splashsurf_studio, "selected_data_obj_index", rows=2)
        col = row.column(align=True)
        col.operator("splashsurf_studio.register", text="", icon='ADD')
        col.operator("splashsurf_studio.unregister", text="", icon='REMOVE')
        
        layout.operator("splashsurf_studio.activate")
        layout.operator("splashsurf_studio.deactivate")
        
        layout.prop(context.scene.splashsurf_studio, "match_transforms")
        
class SPSF_Viewport_Settings_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Viewport Settings"
    bl_idname = "SPSF_PT_viewport"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        obj = get_selected_data_object(context)
        return obj is not None
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text=f"Editing: {get_selected_data_object(context).name}")
        
        layout.prop(context.scene.splashsurf_studio, "use_render_for_viewport")
        
        layout.operator("splashsurf_studio.copy_to_render")
        
        op = layout.operator("splashsurf_studio.parse_cli_string")
        op.is_render_props = False
        op = layout.operator("splashsurf_studio.get_cli_string")
        op.is_render_props = False
        
        layout.prop(context.scene.splashsurf_studio, "update_on_change")

class SPSF_Viewport_Attributes_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Attributes to Interpolate"
    bl_parent_id = "SPSF_PT_viewport"
    bl_idname = "SPSF_PT_viewport_attrs"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = get_selected_data_object(context)
        
        draw_attribute_list(layout, obj.viewport_reconstruction_properties)

class SPSF_Viewport_Params_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Parameters"
    bl_parent_id = "SPSF_PT_viewport"
    bl_idname = "SPSF_PT_viewport_params"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = get_selected_data_object(context)
        
        reconstruction_properties_panel(layout, obj.viewport_reconstruction_properties)

class SPSF_Render_Settings_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Render Settings"
    bl_idname = "SPSF_PT_render"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        obj = get_selected_data_object(context)
        return obj is not None and obj.type == 'MESH'
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text=f"Editing: {get_selected_data_object(context).name}")
        
        layout.operator("splashsurf_studio.copy_to_viewport")
        layout.operator("splashsurf_studio.test_render_params")
        
        op = layout.operator("splashsurf_studio.parse_cli_string")
        op.is_render_props = True
        op = layout.operator("splashsurf_studio.get_cli_string")
        op.is_render_props = True

class SPSF_Render_Attributes_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Attributes to Interpolate"
    bl_parent_id = "SPSF_PT_render"
    bl_idname = "SPSF_PT_render_attrs"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = get_selected_data_object(context)
        
        draw_attribute_list(layout, obj.render_reconstruction_properties)
        
class SPSF_Render_Params_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Parameters"
    bl_parent_id = "SPSF_PT_render"
    bl_idname = "SPSF_PT_render_params"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = get_selected_data_object(context)
        
        reconstruction_properties_panel(layout, obj.render_reconstruction_properties)

class SPSF_Caching_Panel(SPSF_Panel, bpy.types.Panel):
    bl_label = "Cached Frames"
    bl_idname = "SPSF_PT_caching"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene.splashsurf_studio, "cache_settings")
        
        row = layout.row()
        row.template_list("SPSF_UL_Frame_List", "", scene.splashsurf_studio, "cached_frames", scene.splashsurf_studio, "selected_cached_frame_index", rows=2)
        col = row.column(align=True)
        col.operator("splashsurf_studio.add_frames", text="", icon='ADD')
        col.operator("splashsurf_studio.remove_frame", text="", icon='REMOVE')
        
        layout.operator("splashsurf_studio.clear_cache", icon='TRASH')
        layout.operator("splashsurf_studio.regenerate_cache")
        
        layout.prop(scene.splashsurf_studio, "use_cache_during_render")