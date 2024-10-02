from pyray import *
import pyray as pr
import random
from utils import load_embeddings_and_paths
import numpy as np
import os
import pacmap

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load embeddings and paths
data_dir = os.path.join(current_dir, "../../data/normalized_3d_test")
embeddings, thumbnail_paths, labels = load_embeddings_and_paths(data_dir)

# Initialize PaCMAP
embedding = pacmap.PaCMAP(n_components=3, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)

# Add error handling for PaCMAP transformation
try:
    embeddings_3d = embedding.fit_transform(embeddings)
    print("PaCMAP transformation completed successfully")
except Exception as e:
    print(f"Error during PaCMAP transformation: {e}")
    exit(1)

print("🐍 embeddings_3d shape:", embeddings_3d.shape)

# Adjust these parameters if needed
scaling_factor = 1
elevation_factor = 5.0 
min_range = -50
max_range = 50

# Normalize and scale embeddings
embeddings_3d = (embeddings_3d - embeddings_3d.min()) / (embeddings_3d.max() - embeddings_3d.min()) * (max_range - min_range) + min_range
embeddings_3d *= scaling_factor
embeddings_3d[:, 1] += elevation_factor

# Create billboard positions from the 3D embeddings
billboard_positions = [Vector3(point[0], point[1], point[2]) for point in embeddings_3d]

# Load images
images = []
for path in thumbnail_paths:
    image = load_image(path)
    images.append(image)

print(f"Loaded {len(images)} images out of {len(thumbnail_paths)} paths")

# Load textures

textures = []
for image in images:
    texture = load_texture_from_image(image)
    textures.append(texture)
print(f"Loaded {len(textures)} textures out of {len(images)} images")



# Initialization
screenWidth, screenHeight = 1280, 720
init_window(screenWidth, screenHeight, "3D Scatter Plot")

# Define the camera to look into our 3d world
camera = Camera3D(
    Vector3(10.0, 10.0, 10.0),  # Camera position
    Vector3(0.0, 0.0, 0.0),     # Camera looking at point
    Vector3(0.0, 1.0, 0.0),     # Camera up vector (rotation towards target)
    45.0,                       # Camera field-of-view Y
    CameraProjection.CAMERA_PERSPECTIVE  # Camera projection type
)

disable_cursor()
set_target_fps(60)

num_points = min(5000, len(billboard_positions))
print("Total number of billboards:", num_points)

is_wire_mode = False

# Main game loop
while not window_should_close():
    update_camera(camera, CameraMode.CAMERA_FREE)
    
    if is_key_pressed(KeyboardKey.KEY_Z):
        is_wire_mode = not is_wire_mode
        if is_wire_mode:
            rl_enable_wire_mode()
        else:
            rl_disable_wire_mode()

    begin_drawing()
    clear_background(RAYWHITE)
    
    begin_mode_3d(camera)
    draw_grid(100, 2)
    
    for i, (position, texture) in enumerate(zip(billboard_positions, textures)):
        if i >= num_points:
            break
        draw_billboard(camera, texture, position, 0.5, WHITE)
    
    end_mode_3d()
    
    draw_fps(10, 10)
    draw_text(f"Billboards: {num_points}", 10, 30, 20, DARKGRAY)
    draw_text("Press Z to toggle wireframe mode", 10, 50, 20, DARKGRAY)
    
    end_drawing()

# Cleanup
for texture in textures:
    unload_texture(texture)
close_window()