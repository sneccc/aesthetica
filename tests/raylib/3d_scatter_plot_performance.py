from pyray import *
import random
from utils import load_embeddings_and_paths
import numpy as np
import os
import pacmap
import numpy as np
from pyray import *
import random
from utils import load_embeddings_and_paths
import os
import umap
from frustum_culling import Frustum  # Importing the Frustum

#current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
#font = load_font_ex(os.path.join(current_dir, "../../resources/fonts/Roboto-Regular.ttf").encode(), 24, 0, 0)
#load embeddings and paths
data_dir = os.path.join(current_dir, "../../data/normalized_art_categories")
embeddings, thumbnail_paths, labels = load_embeddings_and_paths(data_dir)


import umap
print("🐍 Applying UMAP")
# Apply UMAP to reduce dimensionality to 3D
reducer = umap.UMAP(n_components=3, n_neighbors=5, min_dist=0.3, metric='correlation')
embeddings_3d = reducer.fit_transform(embeddings)


print("🐍 embeddings_3d")
scaling_factor = 1
elevation_factor = 5.0 

# Define the range for normalization
min_range = -50
max_range = 50
range_width = max_range - min_range

# Normalize the 3D embeddings to fit within the specified range
embeddings_3d = (embeddings_3d - embeddings_3d.min()) / (embeddings_3d.max() - embeddings_3d.min()) * range_width + min_range

# Apply scaling factor
embeddings_3d *= scaling_factor
embeddings_3d[:, 1] += elevation_factor

# ==== Create billboard positions from the 3D embeddings and create a rectangle for each billboard and textures ====
billboard_positions = [Vector3(point[0], point[1], point[2]) for point in embeddings_3d]
billboard_rectangles = [Rectangle(point[0]-0.5, point[1]-0.5, 1, 1) for point in embeddings_3d]

# Initialization
screenWidth = 1280
screenHeight = 720
init_window(screenWidth, screenHeight, "raylib [core] example - 3d camera free")

# load textures
textures = [load_texture(path) for path in thumbnail_paths]

# Define the camera to look into our 3d world
camera = Camera3D()
camera.position = Vector3(10.0, 10.0, 10.0)  # Camera position
camera.target = Vector3(0.0, 0.0, 0.0)       # Camera looking at point
camera.up = Vector3(0.0, 1.0, 0.0)           # Camera up vector (rotation towards target)
camera.fovy = 45.0                           # Camera field-of-view Y
camera.projection = CameraProjection.CAMERA_PERSPECTIVE       # Camera projection type

#ray
ray = Ray()
ray.position = Vector3(0.0, 0.0, 0.0)
ray.direction = Vector3(0.0, 0.0, 0.0)
raycollision = RayCollision()


bilboard_bounding_boxes = []
disable_cursor()   

set_target_fps(60)                 # Set our game to run at 60 frames-per-second
# Generate random 3D points
num_points = min(5000, len(billboard_positions))

print("Total number of billboards: ", num_points)

distance = 0
is_wire_mode = False
SCREEN_CENTER = Vector2(screenWidth//2, screenHeight//2)

is_Prespective=True

# Initialize frustum
frustum = Frustum(camera)

# Main game loop
while not window_should_close():   # Detect window close button or ESC key
    # Update
    update_camera(camera, CameraMode.CAMERA_FREE)
    
    # Update the frustum after the camera has been updated
    frustum.camera = camera
    frustum.update_frustum()

    # ==== Draw ====
    begin_drawing()
    clear_background(RAYWHITE)
    
    # ==== Draw 3D ====
    begin_mode_3d(camera)
    draw_grid(100, 2)
    
    if is_key_pressed(KeyboardKey.KEY_P):
        is_Prespective = not is_Prespective
        if is_Prespective:
            camera.projection = CameraProjection.CAMERA_PERSPECTIVE
        else:
            camera.projection = CameraProjection.CAMERA_ORTHOGRAPHIC
    
    # Update camera direction for ray casting
    camera_direction = vector3_subtract(camera.target, camera.position)
    camera_direction_normalized = vector3_normalize(camera_direction)
    
    # Position in front of the camera for the debug sphere
    distance_in_front = 10.0
    infrontpos = vector3_add(camera.position, vector3_scale(camera_direction_normalized, distance_in_front))
    draw_sphere(infrontpos, 0.05, RED)
    
    shortest_distance = float('inf')
    closest_billboard = None
    closest_collision_point = None
    is_hit = [{"hit": False, "label": None}]
    
    # Draw the frustum
    frustum.draw_frustum_lines()

    billboard_drawn = 0
    for i in range(num_points):
        billboard_position = billboard_positions[i]
        texture = textures[i]
        
         # Check if billboard's bounding sphere is inside the frustum
        inside = frustum.contains_sphere(billboard_position, 1)
        
        if not inside:
            continue  # Skip drawing this billboard
        
        billboard_drawn += 1
        # Ray collision detection
        min_bb = Vector3(billboard_position.x - 0.5, billboard_position.y - 0.5, billboard_position.z - 0.5)
        max_bb = Vector3(billboard_position.x + 0.5, billboard_position.y + 0.5, billboard_position.z + 0.5)
        
        ray.position = camera.position
        ray.direction = camera_direction_normalized
        
        collision_info = get_ray_collision_box(ray, BoundingBox(min_bb, max_bb))
        
        if collision_info.hit:
            if collision_info.distance < shortest_distance:
                shortest_distance = collision_info.distance
                closest_billboard = billboard_position
                closest_collision_point = collision_info.point
                closest_index = i  # Keep track of the index
        
        # Apply effect to the closest billboard
        if closest_billboard is not None and collision_info.hit:
            if i == closest_index:
                draw_billboard(camera, textures[i], closest_billboard, 5, WHITE)
                is_hit.append({"hit": True, "label": labels[i]})
                
                # Enlarge the billboard when left mouse button is pressed
                if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
                    distance_to_billboard = vector3_length(vector3_subtract(closest_billboard, camera.position))
                    if distance_to_billboard > 5.5:
                        camera_to_target_direction = vector3_subtract(closest_billboard, camera.position)
                        camera_to_target_direction_normalized = vector3_normalize(camera_to_target_direction)
                        camera.position = vector3_subtract(closest_collision_point, vector3_scale(camera_to_target_direction_normalized, 5))
                        camera.target = closest_billboard
                # Create lines to all billboards of the same label
                if is_mouse_button_down(MouseButton.MOUSE_BUTTON_RIGHT):
                    for j in range(num_points):
                        if labels[j] == labels[i]:
                            draw_line_3d(closest_billboard, billboard_positions[j], BLUE)
        else:
            draw_billboard(camera, textures[i], billboard_position, 1, WHITE)
    
    # Draw debug ray
    #draw_ray(ray, RED)
    

    
    # ==== End 3D ====
    end_mode_3d()
    
    draw_fps(10, 10)
    #total number of billboards
    draw_text(f"Total number of billboards drawn: {billboard_drawn}", 10, 40, 20, RED)
    
    # Check if any billboard is hit and display its label
    hit_items = [item for item in is_hit if item["hit"]]
    if hit_items:
        label = hit_items[0]["label"]
        screen_width = get_screen_width()
        screen_height = get_screen_height()
        text = f"Label: {label}, name: {os.path.basename(thumbnail_paths[closest_index])}"
        text_size = 25
        text_width = measure_text(text, text_size)
        half_width = screen_width // 2
        half_width -= text_width // 2
        draw_text(text, half_width, screen_height - 30, text_size, RED)
    
    # ==== End Draw ====
    end_drawing()

# Unload textures before closing
for texture in textures:
    unload_texture(texture)

# De-Initialization
close_window()  # Close window and OpenGL context