from pyray import *
from enum import Enum

class FrustumPlanes(Enum):
    RIGHT = 0
    LEFT = 1
    BOTTOM = 2
    TOP = 3
    FAR = 4
    NEAR = 5

class Frustum:
    def __init__(self):
        self.planes = [Vector4(0.0, 0.0, 0.0, 0.0) for _ in range(6)]

    def extract_frustum(self):
        # Extract the view frustum planes from the combined projection and modelview matrices
        proj = rl_get_matrix_projection()
        modl = rl_get_matrix_modelview()
        clip = matrix_multiply(modl, proj)

        # Define a helper function to extract planes
        def set_plane(a, b, c, d):
            plane = Vector4(a, b, c, d)
            # Normalize the plane
            t = (a * a + b * b + c * c) ** 0.5
            plane.x /= t
            plane.y /= t
            plane.z /= t
            plane.w /= t
            return plane

        # Right plane
        self.planes[FrustumPlanes.RIGHT.value] = set_plane(
            clip.m3  - clip.m0,
            clip.m7  - clip.m4,
            clip.m11 - clip.m8,
            clip.m15 - clip.m12
        )

        # Left plane
        self.planes[FrustumPlanes.LEFT.value] = set_plane(
            clip.m3  + clip.m0,
            clip.m7  + clip.m4,
            clip.m11 + clip.m8,
            clip.m15 + clip.m12
        )

        # Bottom plane
        self.planes[FrustumPlanes.BOTTOM.value] = set_plane(
            clip.m3  + clip.m1,
            clip.m7  + clip.m5,
            clip.m11 + clip.m9,
            clip.m15 + clip.m13
        )

        # Top plane
        self.planes[FrustumPlanes.TOP.value] = set_plane(
            clip.m3  - clip.m1,
            clip.m7  - clip.m5,
            clip.m11 - clip.m9,
            clip.m15 - clip.m13
        )

        # Far plane
        self.planes[FrustumPlanes.FAR.value] = set_plane(
            clip.m3  - clip.m2,
            clip.m7  - clip.m6,
            clip.m11 - clip.m10,
            clip.m15 - clip.m14
        )

        # Near plane
        self.planes[FrustumPlanes.NEAR.value] = set_plane(
            clip.m3  + clip.m2,
            clip.m7  + clip.m6,
            clip.m11 + clip.m10,
            clip.m15 + clip.m14
        )

    def point_in_frustum(self, x, y, z):
        for plane in self.planes:
            if (plane.x * x + plane.y * y + plane.z * z + plane.w) <= 0:
                return False
        return True

    def sphere_in_frustum(self, x, y, z, radius):
        for plane in self.planes:
            if (plane.x * x + plane.y * y + plane.z * z + plane.w) < -radius:
                return False
        return True

    def box_in_frustum(self, min_point, max_point):
        for plane in self.planes:
            if (plane.x * min_point.x + plane.y * min_point.y + plane.z * min_point.z + plane.w) > 0:
                continue
            if (plane.x * max_point.x + plane.y * min_point.y + plane.z * min_point.z + plane.w) > 0:
                continue
            if (plane.x * max_point.x + plane.y * max_point.y + plane.z * min_point.z + plane.w) > 0:
                continue
            if (plane.x * min_point.x + plane.y * max_point.y + plane.z * min_point.z + plane.w) > 0:
                continue
            if (plane.x * min_point.x + plane.y * min_point.y + plane.z * max_point.z + plane.w) > 0:
                continue
            if (plane.x * max_point.x + plane.y * min_point.y + plane.z * max_point.z + plane.w) > 0:
                continue
            if (plane.x * max_point.x + plane.y * max_point.y + plane.z * max_point.z + plane.w) > 0:
                continue
            if (plane.x * min_point.x + plane.y * max_point.y + plane.z * max_point.z + plane.w) > 0:
                continue
            return False
        return True