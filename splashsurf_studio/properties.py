# pyright: reportInvalidTypeForm=false
import bpy

from .handlers import property_callback, update_callback, update_copy_transforms_constraints

class SPSF_Error_Message_Properties(bpy.types.PropertyGroup):
    obj: bpy.props.StringProperty(
        name="Object",
        description="Object for which the error was encountered",
        default=""
    )
    
    error: bpy.props.StringProperty(
        name="Error",
        description="Encountered Error",
        default=""
    )

class SPSF_Attribute_Properties(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Attribute Name",
        description="Name of the vertex attribute",
        default=""
    )

    interpolate: bpy.props.BoolProperty(
        name="Interpolate",
        default=False
    )
    
class SPSF_Data_Obj_Properties(bpy.types.PropertyGroup):
    generate: bpy.props.BoolProperty(
        name="Generate Surface",
        description="Generate surface for this data object",
        default=True
    )
    
    data_pointer: bpy.props.PointerProperty(
        name="Data Object Pointer",
        description="Pointer to the particle data object",
        type=bpy.types.Object
    )
    
    surface_pointer: bpy.props.PointerProperty(
        name="Surface Object Pointer",
        description="Pointer to the generated surface object",
        type=bpy.types.Object
    )
    
class SPSF_Frame_Properties(bpy.types.PropertyGroup):
    frame: bpy.props.IntProperty(
        name="Frame",
        description="Frame number",
        default=0
    )

class SPSF_Scene_Properties(bpy.types.PropertyGroup):
    data_objs: bpy.props.CollectionProperty(
        type=SPSF_Data_Obj_Properties,
    )
    
    selected_data_obj_index: bpy.props.IntProperty(
        name="Selected Data Object Index",
        default=0
    )
    
    rendering: bpy.props.BoolProperty(
        default=False
    )
    
    active: bpy.props.BoolProperty(
        default=False
    )
    
    update_on_change: bpy.props.BoolProperty(
        name="Update Mesh on Change",
        description="Update the reconstructed mesh when properties are changed",
        default=True,
        update=update_callback
    )
    
    errors: bpy.props.CollectionProperty(
        type=SPSF_Error_Message_Properties,
    )
    
    selected_error_index: bpy.props.IntProperty(
        name="Selected Error Index",
        default=0
    )
    
    use_render_for_viewport: bpy.props.BoolProperty(
        name="Use Render Settings for Viewport",
        description="Use the render reconstruction settings for the viewport reconstruction",
        default=False,
        update=update_callback
    )
    
    cached_frames: bpy.props.CollectionProperty(
        type=SPSF_Frame_Properties,
    )
    
    selected_cached_frame_index: bpy.props.IntProperty(
        name="Selected Cached Frame Index",
        default=0
    )
    
    cache_settings: bpy.props.EnumProperty(
        name="Cache Settings",
        description="Whether to use viewport or render settings for caching",
        items=[
            ('VIEWPORT', "Viewport", "Use viewport reconstruction settings for caching"),
            ('RENDER', "Render", "Use render reconstruction settings for caching")
        ],
        default='VIEWPORT'
    )
    
    use_cache_during_render: bpy.props.BoolProperty(
        name="Use Cache During Render",
        description="Use cached frames during rendering",
        default=True
    )
    
    match_transforms: bpy.props.BoolProperty(
        name="Match Transforms",
        description="Add constraints to match the transform of each reconstructed surface mesh to the transform of its input data object",
        default=True,
        update=update_copy_transforms_constraints,
    )

class SPSF_Reconstruction_Properties(bpy.types.PropertyGroup):
    
    type: bpy.props.EnumProperty(
        name="Type",
        items=[
            ('VIEWPORT', "Viewport", "Use viewport reconstruction settings for caching"),
            ('RENDER', "Render", "Use render reconstruction settings for caching")
        ],
        default='VIEWPORT'
    )
    
    attributes: bpy.props.CollectionProperty(
        type=SPSF_Attribute_Properties,
    )
    
    selected_attribute_num: bpy.props.IntProperty(
        name="Selected Attribute Index",
        default=-1    
    )
    
    particle_radius: bpy.props.FloatProperty(
        name="Particle Radius",
        description="Radius of the particles used in the reconstruction",
        default=0.025,
        step=1,
        precision=3,
        min=0.0,
        update=property_callback
    )
    
    rest_density: bpy.props.FloatProperty(
        name="Rest Density",
        description="Rest density of the fluid",
        default=1000.0,
        min=0.0,
        update=property_callback
    )
    
    cube_size: bpy.props.FloatProperty(
        name="Cube Size",
        description="Size of a grid cell for the reconstruction",
        default=0.5,
        min=0.0,
        update=property_callback
    )
    
    smoothing_length: bpy.props.FloatProperty(
        name="Smoothing Length",
        description="Smoothing length for the reconstruction",
        default=2.0,
        min=0.0,
        update=property_callback
    )
    
    iso_surface_threshold: bpy.props.FloatProperty(
        name="Iso Surface Threshold",
        description="Threshold for the iso surface extraction",
        default=0.6,
        min=0.0,
        update=property_callback
    )
    
    mesh_smoothing_iters: bpy.props.IntProperty(
        name="Mesh Smoothing Iterations",
        description="Number of iterations for mesh smoothing",
        default=25,
        min=0,
        update=property_callback
    )
    
    compute_normals: bpy.props.BoolProperty(
        name="Compute Normals",
        description="Compute normals for the reconstructed mesh",
        default=False,
        update=property_callback
    )
    
    normals_smoothing_iters: bpy.props.IntProperty(
        name="Normals Smoothing Iterations",
        description="Number of iterations for normals smoothing",
        default=10,
        min=0,
        update=property_callback
    )
    
    mesh_cleanup: bpy.props.BoolProperty(
        name="Mesh Cleanup",
        description="Toggle for marching cubes mesh cleanup",
        default=True,
        update=property_callback
    )
    
    mesh_cleanup_limit_snapping_distance: bpy.props.BoolProperty(
        name="Limit Snapping Distance",
        description="Limit vertex snapping",
        default=False,
        update=property_callback
    )
    
    mesh_cleanup_snapping_distance: bpy.props.FloatProperty(
        name="Snapping Distance",
        description="Maximum distance relative to MC edge length for vertex snapping during mesh cleanup",
        default=0.5,
        min=0.0,
        max=0.5,
        update=property_callback
    )
    
    output_smoothing_weights: bpy.props.BoolProperty(
        name="Output Smoothing Weights",
        description="Add smoothing weights attribute to mesh",
        default=False,
        update=property_callback
    )
    
    mesh_smoothing_weights_normalization: bpy.props.FloatProperty(
        name="Smoothing Weights Normalization Factor",
        description="Normalization factor for smoothing weights",
        default=13.0,
        update=property_callback
    )
    
    subdomain_grid: bpy.props.BoolProperty(
        name="Subdomain Grid",
        description="Use subdomain grid for reconstruction",
        default=True,
        update=property_callback
    )
    
    subdomain_grid_auto_disable: bpy.props.BoolProperty(
        name="Auto Disable",
        description="Whether to automatically disable the spatial decomposition if the domain is too small",
        default=True,
        update=property_callback
    )
    
    subdomain_num_cubes_per_dim: bpy.props.IntProperty(
        name="# Cubes Per Dim",
        description="Number of cubes per dimension in the subdomain grid",
        default=64,
        min=1,
        update=property_callback
    )
    
    sph_normals: bpy.props.BoolProperty(
        name="SPH Normals",
        description="Output SPH normals",
        default=False,
        update=property_callback
    )
    
    particle_aabb: bpy.props.BoolProperty(
        name="Use Particle AABB",
        description="Filter particles by axis-aligned bounding box",
        default=False,
        update=property_callback
    )
    
    particle_aabb_min: bpy.props.FloatVectorProperty(
        name="Particle AABB Min",
        description="Minimum coordinates of the axis-aligned bounding box for the particles",
        default=(-1.0, -1.0, -1.0),
        size=3,
        update=property_callback
    )
    
    particle_aabb_max: bpy.props.FloatVectorProperty(
        name="Particle AABB Max",
        description="Maximum coordinates of the axis-aligned bounding box for the particles",
        default=(1.0, 1.0, 1.0),
        size=3,
        update=property_callback
    )
    
    mesh_aabb: bpy.props.BoolProperty(
        name="Use Mesh AABB",
        description="Filter reconstructed mesh by axis-aligned bounding box",
        default=False,
        update=property_callback
    )
    
    mesh_aabb_clamp_vertices: bpy.props.BoolProperty(
        name="Clamp Mesh Vertices",
        description="Clamp mesh vertices to boundary of axis-aligned bounding box",
        default=False,
        update=property_callback
    )
    
    mesh_aabb_min: bpy.props.FloatVectorProperty(
        name="Mesh AABB Min",
        description="Minimum coordinates of the axis-aligned bounding box for the reconstructed mesh",
        default=(-1.0, -1.0, -1.0),
        size=3,
        update=property_callback
    )
    
    mesh_aabb_max: bpy.props.FloatVectorProperty(
        name="Mesh AABB Max",
        description="Maximum coordinates of the axis-aligned bounding box for the reconstructed mesh",
        default=(1.0, 1.0, 1.0),
        size=3,
        update=property_callback
    )
    
    set_split_normals: bpy.props.BoolProperty(
        name="Set Split Normals",
        description="Set reconstructed mesh normals as blender split normals",
        default=False,
        update=property_callback
    )