from pyray import *
import multiprocessing
import random
from utils import load_embeddings_and_paths
import numpy as np
import os
from frustum import Frustum
import umap
from sklearn.cluster import KMeans

def main():
    #current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #font = load_font_ex(os.path.join(current_dir, "../../resources/fonts/Roboto-Regular.ttf").encode(), 24, 0, 0)
    #load embeddings and paths
    data_dir = os.path.join(current_dir, "../../data/normalized_test")
    embeddings, atlas_dir, atlas_positions, labels = load_embeddings_and_paths(data_dir)

    print("🐍 Applying UMAP")
    # Reshape embeddings if they're 3D
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
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
    
     # After creating billboard_positions
    print("🐍 Clustering billboards")
    n_clusters = len(billboard_positions) // 50  # Adjust this value as needed
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings_3d)
    cluster_centers = kmeans.cluster_centers_

    # Create a dictionary to store billboards for each cluster
    clustered_billboards = {i: [] for i in range(n_clusters)}
    for i, (pos, label) in enumerate(zip(billboard_positions, cluster_labels)):
        clustered_billboards[label].append((pos, atlas_positions[i], labels[i]))

    

    # Initialization
    screenWidth = 1280
    screenHeight = 720
    init_window(screenWidth, screenHeight, "raylib [core] example - 3d camera free")

    # load textures
    # With:
    atlas_textures = []
    for atlas_file in sorted(os.listdir(atlas_dir)):
        if atlas_file.startswith('atlas_chunk_') and atlas_file.endswith('.png'):
            atlas_path = os.path.join(atlas_dir, atlas_file)
            texture = load_texture(atlas_path)
            #mesh=gen_mesh_cubicmap(texture,[1.0,1.0,1.0])
            atlas_textures.append(texture)

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

    frustum = Frustum()

    set_target_fps(60)                 # Set our game to run at 60 frames-per-second
    # Generate random 3D points
    num_points = min(5000, len(billboard_positions))

    print("Total number of billboards: ", num_points)

    distance = 0
    is_wire_mode = False
    SCREEN_CENTER = Vector2(screenWidth//2, screenHeight//2)

    is_Prespective=True
    
    
    
    #LOD
    draw_distance_threshold = 100.0
    # GL_TEXTURE_2D = 0x0DE1
    # GL_TEXTURE_LOD_BIAS = 0x8501
    # LOD_BIAS = -0.5 
    # for texture in atlas_textures:
    #     rl_bind_image_texture(texture.id, 0, texture.format, False)
    #     rl_texture_parameters(texture.id, GL_TEXTURE_LOD_BIAS, int(-16))
        
    
    # Main game loop
    while not window_should_close():   # Detect window close button or ESC key
        
        #distance = vector_3distance(camera.position, billboard_positions[0])
        update_camera(camera,CameraMode.CAMERA_FREE)
        # ==== Draw ====
        begin_drawing()
        clear_background(RAYWHITE)
        
        # ==== Draw 3D ====
        begin_mode_3d(camera)
        frustum.extract_frustum()
        drawn_images = 0
        draw_grid(100, 2)
        
        if is_key_pressed(KeyboardKey.KEY_P):
            is_Prespective=not is_Prespective
            if is_Prespective:
                camera.projection = CameraProjection.CAMERA_PERSPECTIVE
            else:
                camera.projection = CameraProjection.CAMERA_ORTHOGRAPHIC
        #draw a line from oposite of camera position to center of canvas
        
        # ==== debug ray ====
        camera_direction = vector3_subtract(camera.target, camera.position)  # Changed subtraction order
        camera_direction_normalized = vector3_normalize(camera_direction)
        distance_in_front = 10.0  # Adjust this value to place the cube closer or farther
        infrontpos = vector3_add(camera.position, vector3_scale(camera_direction_normalized, distance_in_front))
        draw_sphere(infrontpos, 0.05, RED)
        shortest_distance = float('inf')
        closest_billboard = None
        closest_collision_point = None
        is_hit = [{"hit":False, "label":None}]
        
        # ==== Draw billboards ====
        for i, (billboard_position, atlas_position) in enumerate(zip(billboard_positions, atlas_positions)):
            if not frustum.sphere_in_frustum(billboard_position.x, billboard_position.y, billboard_position.z, 1):
                continue
            distance_to_camera = vector3_length(vector3_subtract(billboard_position, camera.position))
            if distance_to_camera > draw_distance_threshold:
                continue
            
            drawn_images+=1
            
            # size is 1x1x1
            min_point = Vector3(billboard_position.x-0.5, billboard_position.y-0.5, billboard_position.z-0.5)
            max_point = Vector3(billboard_position.x+0.5, billboard_position.y+0.5, billboard_position.z+0.5)
        
            ray.position = camera.position
            ray.direction = camera_direction_normalized

            collision_info = get_ray_collision_box(ray, BoundingBox(min_point, max_point))
            
            if collision_info.hit:
                if collision_info.distance < shortest_distance:
                    shortest_distance = collision_info.distance
                    closest_billboard = billboard_position
                    closest_collision_point = collision_info.point

            # Apply effect to the closest billboard
            if closest_billboard is not None and collision_info.hit:
                i = billboard_positions.index(closest_billboard)
                source_rect = Rectangle(atlas_position[0], atlas_position[1], 250, 250)  # Assuming 250x250 thumbnails
                
                # Get the correct atlas texture based on the chunk index
                chunk_index = atlas_position[2]
                atlas_texture = atlas_textures[chunk_index]
                
                draw_billboard_rec(camera, atlas_texture, source_rect,closest_billboard, Vector2(1,1), WHITE)
                
                is_hit.append({"hit":True, "label":labels[i]})
                
                #Enlarge the billboard when left mouse button is pressed
                if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
                    distance_to_billboard = vector3_length(vector3_subtract(closest_billboard, camera.position))
                    if distance_to_billboard > 5.5:
                        camera_to_target_direction = vector3_subtract(closest_billboard, camera.position)
                        camera_to_target_direction_normalized = vector3_normalize(camera_to_target_direction)
                        camera.position = vector3_subtract(closest_collision_point, vector3_scale(camera_to_target_direction_normalized, 5))
                        camera.target = closest_billboard
                #Create lines from this to all billboards of the same label
                if is_mouse_button_down(MouseButton.MOUSE_BUTTON_RIGHT):
                    for j, (billboard_position, atlas_position) in enumerate(zip(billboard_positions, atlas_positions)):
                        if labels[j] == labels[i]:
                            draw_line_3d(closest_billboard, billboard_position, BLUE)
            else:
                source_rect = Rectangle(atlas_position[0], atlas_position[1], 250, 250)
                
                # Get the correct atlas texture based on the chunk index
                chunk_index = atlas_position[2]
                atlas_texture = atlas_textures[chunk_index]
                
                draw_billboard_rec(camera, atlas_texture, source_rect,billboard_position, Vector2(1,1), WHITE)
                
        # Draw debug ray
        draw_ray(ray, RED)

        # ==== End 3D ====
        end_mode_3d()

        #set_mouse_position(int(mouse_position.x), int(mouse_position.y))
        draw_fps(10, 10)
        draw_text(f"Images Drawn: {drawn_images}/{num_points}", 10, 30, 20, BLACK)

        # Check if any billboard is hit and display its label
        hit_items = [item for item in is_hit if item["hit"]]
        if hit_items:
            label = hit_items[0]["label"]
            screen_width = get_screen_width()
            screen_height = get_screen_height()
            text = f"Label: {label}"
            text_size = 25
            text_width = measure_text(text, text_size)  # Use measure_text instead of text_length
            half_width = screen_width // 2
            half_width -= text_width // 2
            draw_text(text, half_width, screen_height - 30, text_size, RED)

        end_drawing()

    # Unload textures before closing
    for atlas_texture in atlas_textures:
        unload_texture(atlas_texture)

    # De-Initialization
    close_window()  # Close window and OpenGL context


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()