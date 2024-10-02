import pyray as pr
import random

pr.init_window(800, 450, "Hello Pyray")
pr.set_target_fps(60)

camera = pr.Camera3D([18.0, 16.0, 18.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0)

# Generate random 3D points
num_points = 100
points = [pr.Vector3(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_points)]

while not pr.window_should_close():
    pr.update_camera(camera, pr.CAMERA_ORBITAL)
    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)
    pr.begin_mode_3d(camera)
    pr.draw_grid(20, 1.0)
    
    # Draw random 3D points
    for point in points:
        pr.draw_sphere(point, 0.2, pr.RED)
    
    pr.end_mode_3d()
    pr.draw_text("Random 3D Points", 10, 40, 20, pr.DARKGRAY)
    pr.end_drawing()

pr.close_window()