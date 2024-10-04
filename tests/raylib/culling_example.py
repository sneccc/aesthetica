from pyray import *
from frustum import Frustum
import random

# Initialize the window
init_window(800, 600, "Frustum Culling Example")
set_target_fps(60)

# Create a free camera
camera = Camera3D(
    Vector3(0.0, 10.0, 10.0),  # position
    Vector3(0.0, 0.0, 0.0),    # target
    Vector3(0.0, 1.0, 0.0),    # up
    45.0,                      # fovy
    CameraProjection.CAMERA_PERSPECTIVE         # projection
)

# Generate random spheres
num_spheres = 1000
spheres = []
for _ in range(num_spheres):
    position = Vector3(
        random.uniform(-50, 50),
        random.uniform(-50, 50),
        random.uniform(-50, 50)
    )
    radius = random.uniform(0.5, 2.0)
    spheres.append((position, radius))

# Create a frustum instance
frustum = Frustum()

# Main game loop
while not window_should_close():
    # Update the camera
    update_camera(pointer(camera), CameraMode.CAMERA_FREE)

    # Begin drawing
    begin_drawing()
    clear_background(RAYWHITE)

    begin_mode_3d(camera)

    # Extract the frustum planes
    frustum.extract_frustum()

    # Draw spheres if inside frustum
    for position, radius in spheres:
        if frustum.sphere_in_frustum(position.x, position.y, position.z, radius):
            draw_sphere(position, radius, RED)
        else:
            draw_sphere_wires(position, radius, 8, 8, GRAY)

    # Draw ground
    draw_plane(Vector3(0.0, 0.0, 0.0), Vector2(50.0, 50.0), LIGHTGRAY)

    end_mode_3d()

    # Draw FPS
    draw_fps(10, 10)

    end_drawing()

# Close window
close_window()