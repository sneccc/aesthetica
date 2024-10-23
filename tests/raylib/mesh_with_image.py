from pyray import *
import raylib as rl
import math
import os
import random
from raylib import *

screenWidth = 800
screenHeight = 450

init_window(screenWidth, screenHeight, "test mesh")

current_dir = os.path.dirname(os.path.abspath(__file__))

texture = load_texture(os.path.join(current_dir, "fustum_test/v2/resources/space.png"))

cube_models = []
number_of_cubes = 10_000

for _ in range(number_of_cubes):
    cube_model = load_model_from_mesh(gen_mesh_cube(1, 1, 1))
    cube_model.materials[0].maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture = texture
    cube_models.append(cube_model)

cubeMesh = gen_mesh_cube(1, 1, 1)
cubeMaterial = load_material_default()
cubeMaterial.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture = texture


cube_positions = [Vector3(random.uniform(-8, 8), random.uniform(-8, 8), random.uniform(-8, 8)) for _ in range(number_of_cubes)]

set_target_fps(60)

camera = Camera3D()
camera.position = Vector3(10.0, 10.0, 10.0)  # Camera position
camera.target = Vector3(0.0, 0.0, 0.0)       # Camera looking at point
camera.up = Vector3(0.0, 1.0, 0.0)           # Camera up vector (rotation towards target)
camera.fovy = 45.0                           # Camera field-of-view Y
camera.projection = CameraProjection.CAMERA_PERSPECTIVE 

disable_cursor()
def calculate_billboard_matrix(model_pos, camera_pos):

    direction = vector3_normalize(vector3_subtract(model_pos, camera_pos))
    world_up = Vector3(0.0, 1.0, 0.0)
    right = vector3_normalize(vector3_cross_product(world_up, direction))
    up = vector3_cross_product(direction, right)
    return Matrix(
        right.x, up.x, direction.x, 0.0,
        right.y, up.y, direction.y, 0.0,
        right.z, up.z, direction.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

while not window_should_close():
    update_camera(camera, CameraMode.CAMERA_FREE)
    begin_drawing()
    clear_background(RAYWHITE)
    begin_mode_3d(camera)
    draw_grid(100, 1.0)  
    for position in cube_positions:
        billboard_matrix = calculate_billboard_matrix(position, camera.position)
        translation_matrix = matrix_translate(position.x, position.y, position.z)
        transform_matrix = matrix_multiply(billboard_matrix, translation_matrix)
        draw_mesh(cubeMesh, cubeMaterial, transform_matrix)  
    end_mode_3d()
    draw_fps(10, 10)
    
    end_drawing()
