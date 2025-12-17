import bpy
from .handlers import generate_mesh, toggle_rendering_off, toggle_rendering_on, ensure_frame_change_handlers
from .properties import *
from .operators import *
from .panels import *
from .updater import cached_meshes

classes = [
    SPSF_Frame_Properties,
    SPSF_Data_Obj_Properties,
    SPSF_Attribute_Properties,
    SPSF_Error_Message_Properties,
    SPSF_Reconstruction_Properties,
    SPSF_Scene_Properties,
    SPSF_OT_Activate,
    SPSF_OT_Deactivate,
    SPSF_OT_Copy_To_Render,
    SPSF_OT_Copy_To_Viewport,
    SPSF_OT_Get_CLI_String,
    SPSF_OT_Parse_CLI_Args,
    SPSF_OT_Register,
    SPSF_OT_Unregister,
    SPSF_OT_Test_Render_Params,
    SPSF_OT_Clear_Errors,
    SPSF_OT_Show_Error,
    SPSF_OT_Add_Frames,
    SPSF_OT_Remove_Frame,
    SPSF_OT_Clear_Cache,
    SPSF_OT_Regenerate_Cache,
    SPSF_OT_Jump_To_Frame,
    SPSF_UL_Attribute_List,
    SPSF_UL_Data_Object_List,
    SPSF_UL_Error_List,
    SPSF_Selection_Panel,
    SPSF_UL_Frame_List,
    SPSF_Caching_Panel,
    SPSF_Error_Log_Panel,
    SPSF_Viewport_Settings_Panel,
    SPSF_Viewport_Attributes_Panel,
    SPSF_Viewport_Params_Panel,
    SPSF_Render_Settings_Panel,
    SPSF_Render_Attributes_Panel,
    SPSF_Render_Params_Panel
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Object.viewport_reconstruction_properties = bpy.props.PointerProperty(type=SPSF_Reconstruction_Properties)
    bpy.types.Object.render_reconstruction_properties = bpy.props.PointerProperty(type=SPSF_Reconstruction_Properties)
    bpy.types.Scene.splashsurf_studio = bpy.props.PointerProperty(type=SPSF_Scene_Properties)
    bpy.types.Mesh.cached = bpy.props.BoolProperty(default=False)
    
    bpy.app.handlers.load_post.append(ensure_frame_change_handlers)

    # Run once to ensure handlers are added when script is reloaded
    ensure_frame_change_handlers(None)
    
def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    bpy.app.handlers.load_post.remove(ensure_frame_change_handlers)
    
    if generate_mesh in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(generate_mesh)
    if toggle_rendering_on in bpy.app.handlers.render_init:
        bpy.app.handlers.render_init.remove(toggle_rendering_on)
    if toggle_rendering_off in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.remove(toggle_rendering_off)
    if toggle_rendering_off in bpy.app.handlers.render_cancel:
        bpy.app.handlers.render_cancel.remove(toggle_rendering_off)
    
    del bpy.types.Object.viewport_reconstruction_properties
    del bpy.types.Object.render_reconstruction_properties
    del bpy.types.Scene.splashsurf_studio
    del bpy.types.Mesh.cached

    # Unload cached meshes, only relevant if addon is reloaded
    clear_cache(cached_meshes)