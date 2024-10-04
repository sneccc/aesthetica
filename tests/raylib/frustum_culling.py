# frustum_culling.py

import math
from pyray import *
from typing import List

class Frustum:
    def __init__(self, camera):
        """
        Initialize the frustum based on the provided camera.
        """
        self.camera = camera
        self.planes = []
        self.corners = []
        self.update_frustum()

    def update_frustum(self):
        projection = rl_get_matrix_projection()
        view = get_camera_matrix(self.camera)
        
        # Ensure this order: Clip = Projection * View
        clip = matrix_multiply(projection, view)
        
        self.extract_planes(clip)
        self.calculate_frustum_corners(clip)

    def calculate_frustum_corners(self,clip):
        """
        Calculate the eight corners of the frustum in world space.
        """
        # Inverse of the combined matrix
        clip = matrix_multiply(rl_get_matrix_projection(), get_camera_matrix(self.camera))
        inv_clip = matrix_invert(clip)

        # Normalized Device Coordinates (NDC) for the corners of the frustum
        ndc_corners = [
            Vector3(-1, -1, 0),  # Near Bottom Left
            Vector3(1, -1, 0),   # Near Bottom Right
            Vector3(1, 1, 0),    # Near Top Right
            Vector3(-1, 1, 0),   # Near Top Left
            Vector3(-1, -1, 1),  # Far Bottom Left
            Vector3(1, -1, 1),   # Far Bottom Right
            Vector3(1, 1, 1),    # Far Top Right
            Vector3(-1, 1, 1),   # Far Top Left
        ]

        self.corners = []

        for ndc_corner in ndc_corners:
            # Transform from NDC space to world space
            corner = vector3_transform(ndc_corner, inv_clip)
            self.corners.append(corner)
            print(f"Frustum Corner: ({corner.x:.2f}, {corner.y:.2f}, {corner.z:.2f})")

    def extract_planes(self, clip):
        """
        Extract the six frustum planes from the combined clip matrix.
        """
        self.planes = []

        # Right plane
        self.planes.append(self.normalize_plane([
            clip.m3  - clip.m0,
            clip.m7  - clip.m4,
            clip.m11 - clip.m8,
            clip.m15 - clip.m12
        ]))

        # Left plane
        self.planes.append(self.normalize_plane([
            clip.m3  + clip.m0,
            clip.m7  + clip.m4,
            clip.m11 + clip.m8,
            clip.m15 + clip.m12
        ]))

        # Bottom plane
        self.planes.append(self.normalize_plane([
            clip.m3  + clip.m1,
            clip.m7  + clip.m5,
            clip.m11 + clip.m9,
            clip.m15 + clip.m13
        ]))

        # Top plane
        self.planes.append(self.normalize_plane([
            clip.m3  - clip.m1,
            clip.m7  - clip.m5,
            clip.m11 - clip.m9,
            clip.m15 - clip.m13
        ]))

        # Far plane
        self.planes.append(self.normalize_plane([
            clip.m3  - clip.m2,
            clip.m7  - clip.m6,
            clip.m11 - clip.m10,
            clip.m15 - clip.m14
        ]))

        # Near plane
        self.planes.append(self.normalize_plane([
            clip.m3  + clip.m2,
            clip.m7  + clip.m6,
            clip.m11 + clip.m10,
            clip.m15 + clip.m14
        ]))

        # Debug: Print Plane Equations
        #for i, plane in enumerate(self.planes):
        #    print(f"Plane {i}: Normal=({plane[0]:.2f}, {plane[1]:.2f}, {plane[2]:.2f}), d={plane[3]:.2f}")

    def normalize_plane(self, plane):
        """
        Normalize the plane equation.
        """
        normal = Vector3(plane[0], plane[1], plane[2])
        length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        return [plane[0]/length, plane[1]/length, plane[2]/length, plane[3]/length]

    def contains_sphere(self, center, radius):
        """
        Check if a sphere is inside or intersects the frustum.
        """
        for plane in self.planes:
            distance = (plane[0] * center.x +
                        plane[1] * center.y +
                        plane[2] * center.z +
                        plane[3])
            if distance < -radius:
                return False
        return True

    def draw_frustum_lines(self):
        """
        Draw the frustum for debugging purposes.
        """
        if len(self.corners) != 8:
            print("Frustum corners not properly calculated.")
            return

        # Draw lines between the corners
        # Near plane
        draw_line_3d(self.corners[0], self.corners[1], RED)
        draw_line_3d(self.corners[1], self.corners[2], RED)
        draw_line_3d(self.corners[2], self.corners[3], RED)
        draw_line_3d(self.corners[3], self.corners[0], RED)

        # Far plane
        draw_line_3d(self.corners[4], self.corners[5], RED)
        draw_line_3d(self.corners[5], self.corners[6], RED)
        draw_line_3d(self.corners[6], self.corners[7], RED)
        draw_line_3d(self.corners[7], self.corners[4], RED)

        # Connecting lines
        for i in range(4):
            draw_line_3d(self.corners[i], self.corners[i+4], RED)

    def contains_point(self, point):
        """
        Check if a point is inside the frustum.
        """
        for plane in self.planes:
            distance = (plane[0] * point.x +
                        plane[1] * point.y +
                        plane[2] * point.z +
                        plane[3])
            if distance < 0.0:
                return False
        return True