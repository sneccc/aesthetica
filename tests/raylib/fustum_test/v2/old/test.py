import random
from pyray import *
import raylib as rl
# Initialize the window
init_window(800, 600, b"Occlusion Culling with Cones")

# Generate cone meshes
def create_cone(x, y, z):
    mesh = gen_mesh_cone(0.4, 1, 16)
    model = load_model_from_mesh(mesh)
    model.transform = matrix_translate(x, y, z)
    return model

def is_occluded(model, camera, depth_texture):
    # Get the bounding box of the model
    bbox = get_model_bounding_box(model)
    
    # Project the bounding box corners to screen space
    corners = [get_world_to_screen(Vector3(x, y, z), camera) for x, y, z in [
        (bbox.min.x, bbox.min.y, bbox.min.z),
        (bbox.max.x, bbox.min.y, bbox.min.z),
        (bbox.min.x, bbox.max.y, bbox.min.z),
        (bbox.max.x, bbox.max.y, bbox.min.z),
        (bbox.min.x, bbox.min.y, bbox.max.z),
        (bbox.max.x, bbox.min.y, bbox.max.z),
        (bbox.min.x, bbox.max.y, bbox.max.z),
        (bbox.max.x, bbox.max.y, bbox.max.z)
    ]]
    
    # Find the bounding rectangle in screen space
    min_x = min(c.x for c in corners)
    min_y = min(c.y for c in corners)
    max_x = max(c.x for c in corners)
    max_y = max(c.y for c in corners)
    
    # Get the depth texture data
    image = load_image_from_texture(depth_texture.texture)
    pixels = load_image_colors(image)
    
    # Check if any pixel in this rectangle is visible in the depth texture
    for y in range(int(min_y), int(max_y) + 1):
        for x in range(int(min_x), int(max_x) + 1):
            index = y * get_screen_width() + x
            depth = pixels[index].r / 255.0  # Normalize to 0-1 range
            if depth < 0.9999:  # Not the far plane (allowing for some float imprecision)
                return True  # Not occluded
    
    return False  # Fully occluded

# Create multiple cones
num_cones = 1000
cones = [create_cone(random.uniform(-10, 10), random.uniform(0, 5), random.uniform(-10, 10)) 
         for _ in range(num_cones)]

# Setup camera
camera = Camera3D(
    Vector3(0.0, 10.0, 10.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    45.0,
    CameraProjection.CAMERA_PERSPECTIVE
)
disable_cursor()

# Create a simple shader for depth pre-pass
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'resources/shaders/depth.fs')
depth_shader = load_shader("", path.encode('utf-8'))

depth_texture = load_render_texture(get_screen_width(), get_screen_height()) #type: RenderTexture

# Main game loop
set_target_fps(60)

while not window_should_close():
    objects_drawn = 0
    # Update camera
    update_camera(camera, CameraMode.CAMERA_FREE)

    # Draw
    begin_drawing()
    clear_background(Color(245, 245, 245, 255))
    
    # Main render pass
    begin_mode_3d(camera)
    for cone in cones:
        if not is_occluded(cone, camera, depth_texture):
            objects_drawn += 1
            draw_model(cone, Vector3(0, 0, 0), 1.0, GREEN)
    end_mode_3d()
    draw_fps(10, 10)
    draw_text(f"Objects Drawn: {objects_drawn}", 10, 30, 20, BLACK)
    end_drawing()

# De-Initialization
for cone in cones:
    unload_model(cone)
unload_shader(depth_shader)
unload_render_texture(depth_texture)
close_window()


