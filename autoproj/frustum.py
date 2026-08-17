"""
Frustum Culling Module - Various frustum culling algorithms for 3D clipping

Supports:
- Liang-Barsky algorithm for pyramid frustum (pinhole camera) line clipping
- Quadratic equation solver for cone frustum (fisheye camera) line clipping
- Sutherland-Hodgman algorithm for polygon clipping
- Point culling (in/out frustum test)
"""

import numpy as np
from typing import Union, Tuple, List, Optional
from enum import Enum


class FrustumType(Enum):
    """Type of frustum for culling."""
    PYRAMID = "pyramid"  # Pyramid frustum for pinhole camera
    CONE = "cone"        # Cone frustum for fisheye camera


class FrustumCuller:
    """
    Frustum culler for 3D points, lines, and polygons.

    Supports both pyramid (pinhole) and cone (fisheye) frustum types.

    The culler operates in camera coordinate system where:
    - x: right
    - y: up
    - z: forward (depth)

    Example usage:
        culler = FrustumCuller(
            frustum_type=FrustumType.PYRAMID,
            near_z=0.1, far_z=100.0,
            fx=1000, fy=1000, cx=960, cy=540, width=1920, height=1080
        )
        clipped = culler.clip_line(p1, p2)
        is_inside = culler.is_point_inside(point)
    """

    def __init__(
        self,
        frustum_type: FrustumType = FrustumType.PYRAMID,
        near_z: float = 0.1,
        far_z: Optional[float] = None,
        expansion_factor: float = 1.0,
        **kwargs
    ):
        """
        Initialize frustum culler.

        视锥由近裁剪面和4个FOV角度面（左/右/上/下）构成，
        **没有远裁剪面** —— FOV在深度方向无限延伸。
        因此只要物体在角度FOV内，无论多远都能完整投影。

        Args:
            frustum_type: Type of frustum (PYRAMID or CONE)
            near_z: Near clipping plane (meters)
            far_z: 保留用于API兼容，但裁剪逻辑中不使用远平面
            expansion_factor: Factor to expand frustum boundaries (default: 1.0, no expansion)
            **kwargs: Additional parameters based on frustum type
                For PYRAMID: fx, fy, cx, cy, width, height (or tan_h, tan_v)
                For CONE: theta_max (maximum incident angle in radians)
        """
        self.frustum_type = frustum_type
        self.near_z = near_z
        self.far_z = far_z
        self.expansion_factor = expansion_factor

        if frustum_type == FrustumType.PYRAMID:
            self._init_pyramid(**kwargs)
        elif frustum_type == FrustumType.CONE:
            self._init_cone(**kwargs)
        else:
            raise ValueError(f"Unknown frustum type: {frustum_type}")

    def _init_pyramid(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        tan_h: Optional[float] = None,
        tan_v: Optional[float] = None
    ):
        """Initialize pyramid frustum parameters."""
        if tan_h is not None and tan_v is not None:
            self.tan_h = tan_h * self.expansion_factor
            self.tan_v = tan_v * self.expansion_factor
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.width = width
            self.height = height

            self.tan_h = (max(cx, (width or 0) - 1 - cx) / fx) * self.expansion_factor
            self.tan_v = (max(cy, (height or 0) - 1 - cy) / fy) * self.expansion_factor
        else:
            raise ValueError("Either tan_h/tan_v or fx/fy/cx/cy must be provided")

    def _init_cone(self, theta_max: float):
        """Initialize cone frustum parameters."""
        self.theta_max = theta_max * self.expansion_factor
        self.tan_theta_max = np.tan(self.theta_max)

    @classmethod
    def from_camera(
        cls,
        camera,
        expansion_factor: float = 1.0
    ) -> 'FrustumCuller':
        """
        Create frustum culler from a camera object.

        Args:
            camera: Camera object (PinholeCamera, KannalaBrandtCamera, or FThetaCamera)
            expansion_factor: Frustum expansion factor (default 1.0, no expansion)

        Returns:
            Configured FrustumCuller instance
        """
        camera_type_name = camera.__class__.__name__

        if camera_type_name == 'PinholeCamera':
            return cls(
                frustum_type=FrustumType.PYRAMID,
                near_z=camera.near_z,
                far_z=camera.far_z,
                expansion_factor=expansion_factor,
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                width=camera.width,
                height=camera.height
            )
        elif camera_type_name == 'KannalaBrandtCamera':
            # Kannala-Brandt: has fx, fy, width, height but no fov_h/fov_v properties
            # Compute FOV from pinhole model as geometric reference
            if hasattr(camera, 'fov_h') and hasattr(camera, 'fov_v'):
                theta_max = np.radians(max(camera.fov_h, camera.fov_v) / 2)
            else:
                # Compute geometric FOV from fx/fy
                fov_h = 2 * np.degrees(np.arctan(camera.width / (2 * camera.fx)))
                fov_v = 2 * np.degrees(np.arctan(camera.height / (2 * camera.fy)))
                theta_max = np.radians(max(fov_h, fov_v) / 2)
            return cls(
                frustum_type=FrustumType.CONE,
                near_z=camera.near_z,
                far_z=camera.far_z,
                expansion_factor=expansion_factor,
                theta_max=theta_max
            )
        elif camera_type_name == 'FThetaCamera':
            if hasattr(camera, '_theta_max'):
                theta_max = camera._theta_max
            else:
                raise ValueError("FThetaCamera must have _theta_max attribute")
            return cls(
                frustum_type=FrustumType.CONE,
                near_z=camera.near_z,
                far_z=camera.far_z,
                expansion_factor=expansion_factor,
                theta_max=theta_max
            )
        else:
            # Generic camera: compute FOV from available attributes
            if hasattr(camera, 'fx') and hasattr(camera, 'fy') and hasattr(camera, 'width') and hasattr(camera, 'height'):
                fov_h = 2 * np.degrees(np.arctan(camera.width / (2 * camera.fx)))
                fov_v = 2 * np.degrees(np.arctan(camera.height / (2 * camera.fy)))
                tan_h = np.tan(np.radians(fov_h / 2))
                tan_v = np.tan(np.radians(fov_v / 2))
                return cls(
                    frustum_type=FrustumType.PYRAMID,
                    near_z=camera.near_z,
                    far_z=camera.far_z,
                    expansion_factor=expansion_factor,
                    tan_h=tan_h,
                    tan_v=tan_v
                )
            elif hasattr(camera, '_theta_max'):
                return cls(
                    frustum_type=FrustumType.CONE,
                    near_z=camera.near_z,
                    far_z=camera.far_z,
                    expansion_factor=expansion_factor,
                    theta_max=camera._theta_max
                )
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def is_point_inside(
        self,
        point: np.ndarray
    ) -> bool:
        """
        Check if a single 3D point is inside the frustum.

        Args:
            point: (3,) array of 3D point in camera coordinates

        Returns:
            True if point is inside frustum
        """
        point = np.asarray(point)
        x, y, z = point

        # 只检查近裁剪面和4个FOV角度面，不检查远裁剪面
        if z < self.near_z:
            return False

        if self.frustum_type == FrustumType.PYRAMID:
            return (abs(x) <= z * self.tan_h) and (abs(y) <= z * self.tan_v)
        else:
            rho = np.sqrt(x**2 + y**2)
            return rho <= z * self.tan_theta_max

    def cull_points(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cull a point cloud, returning only points inside the frustum.

        Args:
            points: (N, 3) array of 3D points in camera coordinates

        Returns:
            (inside_points, mask): Points inside frustum and boolean mask
        """
        points = np.asarray(points)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # 只检查近裁剪面，不检查远裁剪面
        valid_z = z >= self.near_z

        if self.frustum_type == FrustumType.PYRAMID:
            in_h = np.abs(x) <= z * self.tan_h
            in_v = np.abs(y) <= z * self.tan_v
            in_frustum = valid_z & in_h & in_v
        else:
            rho = np.sqrt(x**2 + y**2)
            in_frustum = valid_z & (rho <= z * self.tan_theta_max)

        return points[in_frustum], in_frustum

    def clip_line(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Clip a line segment to the frustum.

        Uses Liang-Barsky algorithm for pyramid frustum and quadratic equation
        solver for cone frustum.

        Args:
            p1: (3,) start point in camera coordinates
            p2: (3,) end point in camera coordinates

        Returns:
            Clipped segment (p1_clipped, p2_clipped) or None if entirely outside.
            Each point is a (3,) numpy array.
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)

        if self.frustum_type == FrustumType.PYRAMID:
            return self._clip_line_pyramid(p1, p2)
        else:
            return self._clip_line_cone(p1, p2)

    def _clip_line_pyramid(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Clip line using Liang-Barsky algorithm for pyramid frustum.

        视锥由5个平面定义（无远裁剪面）：
        - Near:   z = near_z
        - Left:   x = -z * tan_h
        - Right:  x =  z * tan_h
        - Bottom: y = -z * tan_v
        - Top:    y =  z * tan_v
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        t_enter = 0.0
        t_exit = 1.0

        def update_t(numerator, denominator, t_e, t_x):
            if denominator > 0:
                t = numerator / denominator
                if t > t_e:
                    t_e = t
            elif denominator < 0:
                t = numerator / denominator
                if t < t_x:
                    t_x = t
            else:
                if numerator > 0:
                    return None
            return t_e, t_x

        # Near plane: z = near_z
        result = update_t(self.near_z - z1, dz, t_enter, t_exit)
        if result is None:
            return None
        t_enter, t_exit = result

        # 注意：无远裁剪面，FOV在深度方向无限延伸

        # Left plane: x = -z * tan_h  =>  x + z*tan_h = 0
        denom = dx + dz * self.tan_h
        result = update_t(-x1 - z1 * self.tan_h, denom, t_enter, t_exit)
        if result is None:
            return None
        t_enter, t_exit = result

        # Right plane: x = z * tan_h  =>  x - z*tan_h = 0
        # Inside: x <= z*tan_h, so numerator positive means outside (x > z*tan_h)
        denom = dz * self.tan_h - dx
        result = update_t(x1 - z1 * self.tan_h, denom, t_enter, t_exit)
        if result is None:
            return None
        t_enter, t_exit = result

        # Bottom plane: y = -z * tan_v  =>  y + z*tan_v = 0
        denom = dy + dz * self.tan_v
        result = update_t(-y1 - z1 * self.tan_v, denom, t_enter, t_exit)
        if result is None:
            return None
        t_enter, t_exit = result

        # Top plane: y = z * tan_v  =>  y - z*tan_v = 0
        # Inside: y <= z*tan_v, so numerator positive means outside (y > z*tan_v)
        denom = dz * self.tan_v - dy
        result = update_t(y1 - z1 * self.tan_v, denom, t_enter, t_exit)
        if result is None:
            return None
        t_enter, t_exit = result

        if t_enter > t_exit or t_exit < 0 or t_enter > 1:
            return None

        t_enter = max(t_enter, 0.0)
        t_exit = min(t_exit, 1.0)

        p_enter = p1 + t_enter * (p2 - p1)
        p_exit = p1 + t_exit * (p2 - p1)

        return (p_enter, p_exit)

    def _clip_line_cone(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Clip line using quadratic equation for cone frustum.

        锥形视锥（鱼眼）定义：
        x² + y² <= (z * tan_theta_max)²

        加上近裁剪面，无远裁剪面。
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        tan_sq = self.tan_theta_max ** 2

        def cone_value(p):
            x, y, z = p
            return x**2 + y**2 - z**2 * tan_sq

        c1 = cone_value(p1)
        c2 = cone_value(p2)

        inside1 = c1 <= 0
        inside2 = c2 <= 0

        # Handle near plane only (no far plane)
        near_clip = None

        if dz != 0:
            # Near plane intersection
            t_near = (self.near_z - z1) / dz
            if 0 <= t_near <= 1:
                p_near = p1 + t_near * (p2 - p1)
                c_near = cone_value(p_near)
                if c_near <= 0:
                    near_clip = (t_near, p_near)

        # Compute cone intersections
        a = dx**2 + dy**2 - dz**2 * tan_sq
        b = 2 * (x1*dx + y1*dy - z1*dz * tan_sq)
        c = c1

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                # Line is parallel to cone axis or degenerate
                # Check if both endpoints are inside or if there's one intersection
                if inside1 and inside2:
                    return (p1, p2)
                return None

            t_cone = -c / b
            if 0 <= t_cone <= 1:
                p_cone = p1 + t_cone * (p2 - p1)
            else:
                if inside1 and inside2:
                    return (p1, p2)
                return None
        else:
            disc = b**2 - 4*a*c
            if disc < 0:
                # No intersection with cone
                if inside1 and inside2:
                    return (p1, p2)
                return None

            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            t1, t2 = min(t1, t2), max(t1, t2)

            intersections = []
            if 0 <= t1 <= 1:
                intersections.append(t1)
            if 0 <= t2 <= 1 and abs(t2 - t1) > 1e-8:
                intersections.append(t2)

            if not intersections:
                if inside1 and inside2:
                    return (p1, p2)
                return None

            if inside1 and inside2:
                return (p1, p2)

        # Build list of valid parameter intervals
        candidates = []

        if inside1:
            candidates.append((0.0, p1.copy()))
        if inside2:
            candidates.append((1.0, p2.copy()))

        # Add cone intersection points
        if abs(a) < 1e-12:
            if 't_cone' in dir() and 0 <= t_cone <= 1:
                candidates.append((t_cone, p_cone))
        else:
            for t in intersections:
                candidates.append((t, p1 + t * (p2 - p1)))

        # Add near plane intersection points (no far plane)
        if near_clip is not None:
            candidates.append(near_clip)

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        # If both endpoints are inside and the cone is convex, return full segment
        if inside1 and inside2:
            return (p1, p2)

        # Find the longest valid segment (between consecutive candidates)
        # A segment between two consecutive candidates is valid if both endpoints are inside
        best_start = None
        best_end = None
        best_length = -1

        for i in range(len(candidates) - 1):
            t_start, p_start = candidates[i]
            t_end, p_end = candidates[i + 1]

            # Check if the point at mid-interval is inside
            t_mid = (t_start + t_end) / 2
            p_mid = p1 + t_mid * (p2 - p1)
            if cone_value(p_mid) <= 0 and p_mid[2] >= self.near_z:
                length = t_end - t_start
                if length > best_length:
                    best_length = length
                    best_start = (t_start, p_start)
                    best_end = (t_end, p_end)

        if best_start is None or best_end is None:
            return None

        return (best_start[1], best_end[1])

    def clip_polygon(
        self,
        polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Clip a polygon to the frustum using Sutherland-Hodgman algorithm.

        Args:
            polygon: (N, 3) array of polygon vertices in camera coordinates

        Returns:
            Clipped polygon (M, 3) or None if entirely outside
        """
        polygon = np.asarray(polygon, dtype=np.float64)

        if len(polygon) < 3:
            return None

        if self.frustum_type == FrustumType.PYRAMID:
            planes = self._get_pyramid_planes()
            result = polygon
            for plane in planes:
                if len(result) < 3:
                    break
                result = self._clip_polygon_to_plane(result, plane)
            return result if len(result) >= 3 else None
        else:
            return self._clip_polygon_cone(polygon)

    def _get_pyramid_planes(self) -> List[Tuple[float, float, float, float]]:
        """Get pyramid frustum plane equations (a, b, c, d) for ax + by + cz + d = 0.

        视锥由5个平面定义（无远裁剪面）。
        The sign convention: points inside have ax + by + cz + d <= 0.
        Each plane equation is negated from the standard form so that the
        interior of the frustum (facing the camera) corresponds to <= 0.
        """
        planes = []
        # Near plane: z >= near_z  =>  -z + near_z <= 0
        planes.append((0, 0, -1, self.near_z))
        # 注意：无远裁剪面
        # Left plane: x >= -z*tan_h  =>  -x - z*tan_h <= 0
        planes.append((-1, 0, -self.tan_h, 0))
        # Right plane: x <= z*tan_h  =>  x - z*tan_h <= 0
        planes.append((1, 0, -self.tan_h, 0))
        # Bottom plane: y >= -z*tan_v  =>  -y - z*tan_v <= 0
        planes.append((0, -1, -self.tan_v, 0))
        # Top plane: y <= z*tan_v  =>  y - z*tan_v <= 0
        planes.append((0, 1, -self.tan_v, 0))
        return planes

    def _clip_polygon_to_plane(
        self,
        polygon: np.ndarray,
        plane: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """
        Clip polygon to a single plane using Sutherland-Hodgman algorithm.

        Inside is defined as: a*x + b*y + c*z + d <= 0

        Args:
            polygon: (N, 3) vertices
            plane: (a, b, c, d) plane equation

        Returns:
            Clipped polygon or None
        """
        a, b, c, d = plane
        output = []

        for i in range(len(polygon)):
            curr = polygon[i]
            next_p = polygon[(i + 1) % len(polygon)]

            curr_val = a * curr[0] + b * curr[1] + c * curr[2] + d
            next_val = a * next_p[0] + b * next_p[1] + c * next_p[2] + d

            # Current point is inside
            if curr_val <= 0:
                output.append(curr)

            # Crossing detected
            if (curr_val <= 0) != (next_val <= 0):
                denom = a * (next_p[0] - curr[0]) + b * (next_p[1] - curr[1]) + c * (next_p[2] - curr[2])
                if abs(denom) > 1e-10:
                    t = -curr_val / denom
                    if 0 <= t <= 1:
                        intersection = curr + t * (next_p - curr)
                        output.append(intersection)

        return np.array(output) if output else None

    def _clip_polygon_cone(
        self,
        polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        """Clip polygon to cone frustum by clipping each edge."""
        result = []
        n = len(polygon)

        for i in range(n):
            curr = polygon[i]
            next_p = polygon[(i + 1) % n]

            clipped = self.clip_line(curr, next_p)

            if clipped is not None:
                p1_clipped, p2_clipped = clipped
                if len(result) == 0 or not np.allclose(result[-1], p1_clipped):
                    result.append(p1_clipped)
                result.append(p2_clipped)

        if not result:
            return None

        # Remove duplicate consecutive points
        unique_result = []
        for pt in result:
            if not unique_result or not np.allclose(unique_result[-1], pt):
                unique_result.append(pt)

        # Remove closing duplicate if polygon is closed
        if len(unique_result) >= 2 and np.allclose(unique_result[0], unique_result[-1]):
            unique_result = unique_result[:-1]

        return np.array(unique_result) if len(unique_result) >= 3 else None