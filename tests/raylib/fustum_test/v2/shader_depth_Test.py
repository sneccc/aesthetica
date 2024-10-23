from pyray import *
from raylib.colors import *
import os
import raylib as rl
# Set up the window
screenWidth = 1280
screenHeight = 720

# Enable anti-aliasing and allow window resizing
set_config_flags(ConfigFlags.FLAG_MSAA_4X_HINT | ConfigFlags.FLAG_WINDOW_RESIZABLE)
init_window(screenWidth, screenHeight, b"raylib [3D Depth Shader]")

# Define camera parameters
# Set up the camera
camera=Camera3D()
camera.position = Vector3(2, 12, 6)
camera.target = Vector3(0, .5, 0)
camera.up = Vector3(0, 1, 0)
camera.fovy = 45
camera.projection = CameraProjection.CAMERA_PERSPECTIVE

# Load depth shader
current_dir = os.path.dirname(os.path.abspath(__file__))
depth_shader_path = os.path.join(current_dir, 'resources/shaders/write_depth.fs')
#vertex_shader_path = os.path.join(current_dir, 'resources/shaders/base.vs')
depth_shader = load_shader(b"", depth_shader_path.encode('utf-8'))

def Unloadrendertexture(target):
    if(target.id > 0):
        unload_texture(target.texture.id)
        unload_texture(target.depth.id)
        rl_unload_framebuffer(target.id)


def LoadRenderTextureDepthTex(width, height):
    target = load_render_texture(width, height)
    target.id = rl_load_framebuffer(width, height)
    
    if (target.id > 0):
        rl_enable_framebuffer(target.id)
        
        #Create color texture
        target.texture.id = rl_load_texture(rl.ffi.new("int *",0),width,height,PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,1)
        target.texture.width = width
        target.texture.height = height
        target.texture.format = PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
        target.texture.mipmaps = 1
        
        #create depth texture buffer
        target.depth.id = rl_load_texture_depth(width,height,False)
        target.depth.width = width
        target.depth.height = height
        target.depth.format = 19
        target.depth.mipmaps = 1
        
        #attack color texture and depth texture to FBO
        rl_framebuffer_attach(target.id, target.texture.id, rl.RL_ATTACHMENT_COLOR_CHANNEL0,rl.RL_ATTACHMENT_TEXTURE2D,0)
        rl_framebuffer_attach(target.id, target.depth.id, rl.RL_ATTACHMENT_DEPTH,rl.RL_ATTACHMENT_TEXTURE2D,0)
        
        #check if fbo is complete
        if rl_framebuffer_complete(target.id):
            trace_log(TraceLogLevel.LOG_INFO, f"FBO: [ID {target.id}] Framebuffer created successfully",target.id)
            rl_disable_framebuffer()
        else:
            trace_log(TraceLogLevel.LOG_ERROR, f"FBO: Framebuffer creation failed",target.id)
    return target

#custom render texture buffer
target = LoadRenderTextureDepthTex(screenWidth, screenHeight)

set_target_fps(60)

disable_cursor()
# Flag to toggle depth shader
use_depth_shader = False

def run_texture_buffer(target):
    begin_texture_mode(target)
    clear_background(WHITE)
    begin_mode_3d(camera)
    begin_shader_mode(depth_shader)
    size = Vector3(1, 1, 1.0)
    draw_cube_wires_v(Vector3(0.0, 0.5, 1.0),size, RED)
    draw_cube_v(Vector3(0.0, 0.5, 1.0),size, PURPLE)
    draw_cube_wires_v(Vector3(0.0, 0.5, -1.0),size, DARKGREEN)
    draw_cube_v(Vector3(0.0, 0.5, -1.0),size, YELLOW)
    draw_grid(10, 1.0)
    end_shader_mode()
    end_mode_3d()
    end_texture_mode()

# Main game loop
while not window_should_close():
    print("Main loop iteration") 

    # Update camera
    update_camera(camera, CameraMode.CAMERA_FREE)
    
    run_texture_buffer(target)
    
    # draw into screen our custom render texture
    begin_drawing()
    clear_background(RAYWHITE)
    draw_texture_rec(target.texture, Rectangle(0, 0, screenWidth, screenHeight), Vector2(0, 0), WHITE)
    draw_fps(10, 10)
    end_drawing()
    
unload_shader(depth_shader)
Unloadrendertexture(target)

close_window()





