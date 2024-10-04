from pyray import *
from frustum import Frustum
import random

# Initialize the window
init_window(800, 600, "Frustum Culling Example")
set_target_fps(60)
#rl_enable_backface_culling()
rl_enable_depth_test()
# Create a free camera
camera = Camera3D(
    Vector3(0.0, 10.0, 10.0),  # position
    Vector3(0.0, 0.0, 0.0),    # target
    Vector3(0.0, 1.0, 0.0),    # up
    45.0,                      # fovy
    CameraProjection.CAMERA_PERSPECTIVE         # projection
)

# Generate random spheres
num_spheres = 10_000
spheres = []
for _ in range(num_spheres):
    position = Vector3(
        random.uniform(-50, 50),
        random.uniform(-50, 50),
        random.uniform(-50, 50)
    )
    radius = random.uniform(0.5, 2.0)
    spheres.append((position, radius))

disable_cursor()
# Create a frustum instance
frustum = Frustum()

# Main game loop
while not window_should_close():
    # Update the camera
    update_camera(camera, CameraMode.CAMERA_FREE)

    # Begin drawing
    begin_drawing()
    clear_background(RAYWHITE)

    begin_mode_3d(camera)

    # Extract the frustum planes
    frustum.extract_frustum()

    # Initialize the drawn spheres counter
    drawn_spheres_count = 0

    # Draw spheres if inside frustum
    for position, radius in spheres:
        if frustum.sphere_in_frustum(position.x, position.y, position.z, radius):
            draw_sphere(position, radius, RED)
            drawn_spheres_count += 1

    # Draw ground
    draw_plane(Vector3(0.0, 0.0, 0.0), Vector2(50.0, 50.0), LIGHTGRAY)

    end_mode_3d()

    # Draw FPS
    draw_fps(10, 10)

    # Display the number of drawn spheres vs total spheres
    draw_text(f"Spheres Drawn: {drawn_spheres_count}/{num_spheres}", 10, 30, 20, BLACK)

    end_drawing()

# Close window
close_window()