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

# 可选: Numba JIT 加速 (用于 clip_lines_batch 并行批量裁剪)
try:
    from ._numba_kernels import (
        _clip_lines_pyramid_numba,
        _clip_lines_cone_numba,
        _clip_lines_cone_precise_numba,
        NUMBA_AVAILABLE,
        BATCH_PARALLEL_THRESHOLD,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    BATCH_PARALLEL_THRESHOLD = 32
    _clip_lines_pyramid_numba = None
    _clip_lines_cone_numba = None
    _clip_lines_cone_precise_numba = None

# _frustum_scale 的默认值（与 Camera._DEFAULT_FRUSTUM_SCALE 保持一致）
_DEFAULT_FRUSTUM_SCALE = 1.05


class FrustumType(Enum):
    """Type of frustum for culling."""
    PYRAMID = "pyramid"  # Pyramid frustum for pinhole camera
    CONE = "cone"        # Cone frustum for fisheye camera


class FrustumCuller:
    """
    Frustum culler for 3D points, lines, and polygons.

    Supports both pyramid (pinhole) and cone (fisheye) frustum types.

    The culler operates in the OpenCV camera coordinate system where:
    - x: right
    - y: down  (note: image v grows downward, consistent with OpenCV)
    - z: forward (depth, points in front of camera have z > 0)

    坐标系契约（详见 README "Coordinate System Conventions"）：
        - 相机系：OpenCV 约定（x 右 / y 下 / z 前），原点为光心
        - 像素原点：图像左上角，u 向右增长，v 向下增长
        - 外参 T_to_cam：world→camera 的 4×4 齐次矩阵（p_cam = T @ p_world）
        - 单位：米（near_z/far_z 默认 0.1/1000 米）
        - 若数据来源为 OpenGL/Blender（y 上/z 后）或 ROS REP-103（x 前/y 左/z 上）等约定，
          必须在入口处通过 conventions 模块完成轴变换后再传入

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
        # --- 退化输入验证 ---
        if not (isinstance(near_z, (int, float)) and near_z > 0):
            raise ValueError(f"near_z must be a positive number, got {near_z}")
        if not (isinstance(expansion_factor, (int, float)) and expansion_factor > 0):
            raise ValueError(f"expansion_factor must be a positive number, got {expansion_factor}")

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
        # 所有内部参数规范化为 Python float，避免 CuPy 0维数组污染状态
        if tan_h is not None and tan_v is not None:
            self.tan_h = self._to_python_scalar(tan_h) * self.expansion_factor
            self.tan_v = self._to_python_scalar(tan_v) * self.expansion_factor
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            self.fx = self._to_python_scalar(fx)
            self.fy = self._to_python_scalar(fy)
            self.cx = self._to_python_scalar(cx)
            self.cy = self._to_python_scalar(cy)
            self.width = int(width) if width is not None else None
            self.height = int(height) if height is not None else None

            self.tan_h = (max(self.cx, (self.width or 0) - 1 - self.cx) / self.fx) * self.expansion_factor
            self.tan_v = (max(self.cy, (self.height or 0) - 1 - self.cy) / self.fy) * self.expansion_factor
        else:
            raise ValueError("Either tan_h/tan_v or fx/fy/cx/cy must be provided")

    @staticmethod
    def _to_python_scalar(x):
        """将输入转换为 Python 标量，避免 CuPy/NumPy 0维数组导致运算时类型不一致。

        FrustumCuller 是纯几何工具，所有内部参数（theta_max、cos_theta_max 等）
        均以 Python float 形式存储，保证与任意 backend (NumPy/CuPy) 输入兼容。
        """
        if hasattr(x, 'get'):  # CuPy ndarray（含 0 维）
            x = x.get()
        if isinstance(x, np.ndarray):  # NumPy 0维数组 → Python 标量
            x = x.item()
        return float(x)

    @staticmethod
    def _to_numpy(points):
        """将输入数组转为 NumPy ndarray（CuPy 输入会被拷回 host）。

        FrustumCuller 的所有公开方法接收任意 backend 数组，但内部仅使用
        NumPy 计算。CuPy 数组通过 .get() 拷回 host，避免混合运算报错。
        对于无 .get 属性的对象，按常规 np.asarray 处理。
        """
        if hasattr(points, 'get'):
            points = points.get()
        return np.asarray(points, dtype=np.float64)

    def _init_cone(self, theta_max: float):
        """Initialize cone frustum parameters.

        使用 dot-product（余弦点积）形式判断点是否在视锥内，
        替代 tan-based 形式，支持任意 θ_max（包括 > 90°）：

          cos(θ_max) = z / sqrt(x² + y² + z²)  ← 入射角的方向余弦

        对于常规鱼眼 (theta_max ≤ 90°)：
          cos(80°) = 0.174，条件 z/r ≥ 0.174 → z ≥ 0.174·r

        对于大FOV鱼眼 (theta_max > 90°)：
          cos(98°) = -0.139，条件 z/r ≥ -0.139 → z ≥ -0.139·r
          即允许略带负z（相机微后方）的点通过，覆盖 >180° 视场

        同时存储 sin(theta_max) 供二次方程求解使用。
        """
        # 规范化为 Python float，避免 CuPy 0维数组污染内部状态
        theta_max = self._to_python_scalar(theta_max)
        self.theta_max = theta_max * self.expansion_factor
        self.cos_theta_max = float(np.cos(self.theta_max))
        self.sin_theta_max = float(np.sin(self.theta_max))
        # 保持 tan_theta_max 用于向后兼容
        if abs(self.sin_theta_max) > 1e-12:
            self.tan_theta_max = self.sin_theta_max / self.cos_theta_max
        else:
            self.tan_theta_max = np.inf
        # _large_fov 标记：theta_max > 90° 时锥形条件对几乎所有 z>0 点自动成立
        self._large_fov = self.theta_max >= np.pi / 2 - 1e-6

    @classmethod
    def from_camera(
        cls,
        camera,
        expansion_factor: float = 1.0
    ) -> 'FrustumCuller':
        """
        Create frustum culler from a camera object.

        优先使用相机预计算的 FOV 边界（_frustum_tan / _real_tan / _theta_max），
        此时 expansion_factor 参数被忽略（边界值已包含扩展）。
        仅在无预计算值时（fallback 路径），expansion_factor 作为 tan 乘数使用。

        对于 PinholeCamera:
            1. 优先 _frustum_tan_h/v（已含畸变和 margin）
            2. 次选 _real_tan_h/v * _frustum_scale
            3. fallback: 标称 tan * expansion_factor
        对于 KannalaBrandtCamera/FThetaCamera:
            使用 _theta_max * _frustum_scale（忽略 expansion_factor 参数）

        Args:
            camera: Camera object (PinholeCamera, KannalaBrandtCamera, or FThetaCamera)
            expansion_factor: 仅在 fallback 路径（无预计算值）时作为 tan 乘数使用。
                             若 camera._expansion_factor 已计算且参数为默认 1.0，
                             则自动使用 camera._expansion_factor。

        Returns:
            Configured FrustumCuller instance
        """
        # 自动使用相机的扩展因子（若已计算）
        auto_expansion = getattr(camera, '_expansion_factor', None)
        if auto_expansion is not None and expansion_factor == 1.0:
            expansion_factor = auto_expansion

        camera_type_name = camera.__class__.__name__

        if camera_type_name == 'PinholeCamera':
            # 针孔相机：优先使用 _frustum_tan_h/_frustum_tan_v
            # （由 compute_frustum_tan_bounds 计算的有效 FOV 边界）
            # 这些值已考虑畸变非线性和边界 margin，确保裁剪后的棱线端点
            # 能正确投影到图像范围内。
            
            # 优先级：_frustum_tan > _real_tan > 标称 tan * expansion_factor
            frustum_tan_h = getattr(camera, '_frustum_tan_h', None)
            frustum_tan_v = getattr(camera, '_frustum_tan_v', None)
            
            if frustum_tan_h is not None and frustum_tan_v is not None:
                # 使用已预计算的视锥裁剪边界（包含畸变和 margin）
                return cls(
                    frustum_type=FrustumType.PYRAMID,
                    near_z=camera.near_z,
                    far_z=camera.far_z,
                    expansion_factor=1.0,
                    tan_h=frustum_tan_h,
                    tan_v=frustum_tan_v
                )
            
            # 回退：使用 _real_tan + 缩放因子
            real_tan_h = getattr(camera, '_real_tan_h', None)
            real_tan_v = getattr(camera, '_real_tan_v', None)
            frustum_scale = getattr(camera, '_frustum_scale', None)
            if frustum_scale is None:
                frustum_scale = _DEFAULT_FRUSTUM_SCALE
            
            if real_tan_h is not None and real_tan_v is not None:
                tan_h = real_tan_h * frustum_scale
                tan_v = real_tan_v * frustum_scale
                return cls(
                    frustum_type=FrustumType.PYRAMID,
                    near_z=camera.near_z,
                    far_z=camera.far_z,
                    expansion_factor=1.0,
                    tan_h=tan_h,
                    tan_v=tan_v
                )
            else:
                # 最终回退：使用标称 tan * expansion_factor
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
            # Kannala-Brandt 鱼眼相机：锥形视锥
            #
            # 使用 camera._theta_max（由 _compute_theta_max 预计算的真实
            # 最大物理入射角），该值已包含畸变影响，能够正确反映鱼眼
            # 相机的实际可视范围（通常 80°~100°+ 半角）。
            #
            # 注意：_theta_max 已是真实值，expansion_factor 不再使用。
            # 仅应用 _frustum_scale 以匹配 project() 方法的 FOV 检查
            # （project 使用 theta_max * eff_scale）。
            if not hasattr(camera, '_theta_max'):
                # 回退：手动计算
                r_max = max(
                    getattr(camera, 'cx', 0),
                    getattr(camera, 'width', 0) - 1 - getattr(camera, 'cx', 0),
                    getattr(camera, 'cy', 0),
                    getattr(camera, 'height', 0) - 1 - getattr(camera, 'cy', 0),
                )
                fx = getattr(camera, 'fx', 1.0)
                fy = getattr(camera, 'fy', 1.0)
                theta_d_target = max(r_max / fx, r_max / fy)
                k1 = getattr(camera, 'k1', 0.0)
                k2 = getattr(camera, 'k2', 0.0)
                k3 = getattr(camera, 'k3', 0.0)
                k4 = getattr(camera, 'k4', 0.0)

                def _eval_theta_d(th):
                    return th + k1*th**3 + k2*th**5 + k3*th**7 + k4*th**9
                lo, hi = 0.0, np.pi / 2
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if _eval_theta_d(mid) < theta_d_target:
                        lo = mid
                    else:
                        hi = mid
                camera._theta_max = (lo + hi) / 2

            theta_max = camera._theta_max
            # 仅应用 _frustum_scale，不使用 expansion_factor
            # 因为 _theta_max 已经是真实 FOV（包含畸变），
            # project() 中的 FOV 检查使用 theta_max * eff_scale
            frustum_scale = getattr(camera, '_frustum_scale', None)
            if frustum_scale is None:
                frustum_scale = _DEFAULT_FRUSTUM_SCALE
            return cls(
                frustum_type=FrustumType.CONE,
                near_z=camera.near_z,
                far_z=camera.far_z,
                expansion_factor=frustum_scale,
                theta_max=theta_max
            )
        elif camera_type_name == 'FThetaCamera':
            # FThetaCamera._theta_max 已通过多项式反解获得真实入射角（rad）
            # _theta_max 已是真实值，仅应用 _frustum_scale 匹配 project()
            if hasattr(camera, '_theta_max'):
                theta_max = camera._theta_max
            else:
                raise ValueError("FThetaCamera must have _theta_max attribute")
            frustum_scale = getattr(camera, '_frustum_scale', None)
            if frustum_scale is None:
                frustum_scale = _DEFAULT_FRUSTUM_SCALE
            return cls(
                frustum_type=FrustumType.CONE,
                near_z=camera.near_z,
                far_z=camera.far_z,
                expansion_factor=frustum_scale,
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

        For CONE frustum (fisheye): uses dot-product cosine check:
          z / sqrt(x²+y²+z²) >= cos(theta_max)
        This is numerically stable for any theta_max (including >90°).

        For PYRAMID frustum (pinhole): uses tan-based check.
        """
        # 强制转为 NumPy ndarray（CuPy 输入会被 .get() 拷回 host）
        point = self._to_numpy(point)
        x, y, z = point

        if self.frustum_type == FrustumType.PYRAMID:
            if z < self.near_z:
                return False
            return (abs(x) <= z * self.tan_h) and (abs(y) <= z * self.tan_v)
        else:
            # Cone frustum: use dot-product check
            r = np.sqrt(x**2 + y**2 + z**2)
            if r < self.near_z:
                return False
            return z / max(r, 1e-12) >= self.cos_theta_max

    def cull_points(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cull a point cloud, returning only points inside the frustum.

        For CONE frustum: uses dot-product cosine check r>=near_z AND z/r >= cos(theta_max).
        For PYRAMID frustum: uses z>=near_z and tan-based FOV checks.

        注意：FrustumCuller 内部仅使用 NumPy 计算。若传入 CuPy 数组，
        会通过 _to_numpy 拷回 host 后处理，避免混合 backend 运算错误。
        返回的有效点子集与有效掩码均为 NumPy 类型，调用方如需 CuPy
        可自行 asarray。
        """
        # 强制转为 NumPy ndarray（CuPy 输入会被 .get() 拷回 host）
        points = self._to_numpy(points)

        # 空数组：返回空结果，保持形状一致性
        if points.size == 0:
            empty = points.reshape(0, 3) if points.ndim >= 2 else np.empty((0, 3))
            return empty, np.empty(0, dtype=bool)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # NaN 检查：含 NaN 的点标记为无效
        nan_mask = np.isnan(x) | np.isnan(y) | np.isnan(z)

        if self.frustum_type == FrustumType.PYRAMID:
            valid_z = z >= self.near_z
            in_h = np.abs(x) <= z * self.tan_h
            in_v = np.abs(y) <= z * self.tan_v
            in_frustum = valid_z & in_h & in_v
        else:
            # Cone frustum: dot-product check
            r = np.sqrt(x**2 + y**2 + z**2)
            valid_r = r >= self.near_z
            cos_angle = z / np.maximum(r, 1e-12)
            in_frustum = valid_r & (cos_angle >= self.cos_theta_max)

        # NaN 点强制标记为无效
        in_frustum = in_frustum & ~nan_mask

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
        # 强制转为 NumPy ndarray（CuPy 输入会被 .get() 拷回 host）
        p1 = self._to_numpy(p1)
        p2 = self._to_numpy(p2)

        # 形状验证：端点必须是 (3,) 数组
        if p1.shape != (3,):
            raise ValueError(f"p1 must have shape (3,), got {p1.shape}")
        if p2.shape != (3,):
            raise ValueError(f"p2 must have shape (3,), got {p2.shape}")

        # NaN 检查：端点含 NaN 时线段无效
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            return None

        if self.frustum_type == FrustumType.PYRAMID:
            return self._clip_line_pyramid(p1, p2)
        else:
            return self._clip_line_cone(p1, p2)

    def clip_lines_batch(
        self,
        p1s: np.ndarray,
        p2s: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量裁剪线段集合

        当 N > BATCH_PARALLEL_THRESHOLD 且 Numba 可用时, 启用 JIT 并行加速
        (PYRAMID 类型为精确 Liang-Barsky 并行版, CONE 类型为近似版).
        否则回退到逐条调用 clip_line 的 Python 循环.

        Args:
            p1s: (N, 3) 起点数组 (相机坐标系)
            p2s: (N, 3) 终点数组 (相机坐标系, 与 p1s 长度一致)

        Returns:
            clipped: (N, 2, 3) 裁剪后端点
                    clipped[i, 0] = 第 i 条线裁剪后起点 (3,)
                    clipped[i, 1] = 第 i 条线裁剪后终点 (3,)
                    无效线段对应位置为 NaN
            valid: (N,) bool 有效掩码
                    False = 整段在视锥外 (对应位置为 NaN)
                    True = 裁剪后有效线段
        """
        # 强制 NumPy 类型 (CuPy 输入会被 .get() 拷回 host)
        p1s = self._to_numpy(p1s)
        p2s = self._to_numpy(p2s)

        if p1s.shape != p2s.shape:
            raise ValueError(
                f"p1s and p2s must have same shape, got {p1s.shape} vs {p2s.shape}"
            )
        if p1s.ndim != 2 or p1s.shape[1] != 3:
            raise ValueError(f"p1s must have shape (N, 3), got {p1s.shape}")

        N = p1s.shape[0]
        if N == 0:
            return np.empty((0, 2, 3), dtype=np.float64), np.empty(0, dtype=bool)

        # NaN 检查: 含 NaN 的线段标记为无效
        nan_mask_p1 = np.isnan(p1s).any(axis=1)
        nan_mask_p2 = np.isnan(p2s).any(axis=1)
        nan_mask = nan_mask_p1 | nan_mask_p2

        # 路径选择
        use_numba = (
            NUMBA_AVAILABLE
            and N > BATCH_PARALLEL_THRESHOLD
            and not nan_mask.any()  # 有 NaN 时走 Python 路径处理
        )

        if use_numba and self.frustum_type == FrustumType.PYRAMID:
            # PYRAMID: Numba 精确并行 Liang-Barsky
            clipped, valid = _clip_lines_pyramid_numba(
                p1s, p2s,
                self.near_z, self.tan_h, self.tan_v
            )
            return clipped, valid
        elif use_numba and self.frustum_type == FrustumType.CONE:
            # CONE: Numba 精确并行 (完整二次方程求解 + 最长子区间搜索)
            # 替代旧近似版, 消除上层 needs_precise Python 回退路径 (FTheta 负优化根因)
            clipped, valid = _clip_lines_cone_precise_numba(
                p1s, p2s,
                self.near_z, self.cos_theta_max,
                self.sin_theta_max, self._large_fov
            )
            return clipped, valid
        else:
            # Python 回退路径: 逐条裁剪
            clipped = np.full((N, 2, 3), np.nan, dtype=np.float64)
            valid = np.zeros(N, dtype=bool)
            for i in range(N):
                if nan_mask[i]:
                    continue
                result = self.clip_line(p1s[i], p2s[i])
                if result is not None:
                    p1_c, p2_c = result
                    # 径向距离检查 (与 project_box 中保持一致)
                    near_sq = self.near_z ** 2
                    r1_sq = p1_c[0]**2 + p1_c[1]**2 + p1_c[2]**2
                    r2_sq = p2_c[0]**2 + p2_c[1]**2 + p2_c[2]**2
                    if r1_sq >= near_sq and r2_sq >= near_sq:
                        clipped[i, 0] = p1_c
                        clipped[i, 1] = p2_c
                        valid[i] = True
            return clipped, valid

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
        Clip line using cos/sin formulation for cone frustum.

        锥形视锥（鱼眼）定义（dot-product 形式）：
          z / sqrt(x²+y²+z²) >= cos(theta_max)
        等价于二次方程（仅当 cos(theta_max) >= 0，即 θ_max ≤ 90° 时）：
          (x²+y²)*cos²(θ_max) - z²*sin²(θ_max) <= 0

        cos/sin 形式对 θ_max > 90° 稳定（cos 和 sin 均为有限值），
        不需要 tan(θ) 计算，避免了 θ ≈ 90° 时的数值问题。

        近裁剪面使用距离检查：sqrt(x²+y²+z²) >= near_z

        注意：当 θ_max > 90°（_large_fov 模式）时，cos(theta_max) < 0，
        代数式 (x²+y²)*cos² - z²*sin² <= 0 不再与几何判断等价
        （平方运算不保号）。此时改用直接几何式 z/r >= cos(theta_max)
        进行内外判断，参见 _clip_line_large_fov。
        """
        # 大FOV模式（θ_max > 90°）：代数式失效，改用几何式判断
        if self._large_fov:
            return self._clip_line_large_fov(p1, p2)

        x1, y1, z1 = p1
        x2, y2, z2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        cos_sq = self.cos_theta_max ** 2
        sin_sq = self.sin_theta_max ** 2

        def cone_value(p):
            """(x²+y²)*cos²(θ) - z²*sin²(θ) <= 0 means inside cone."""
            x, y, z = p
            return (x**2 + y**2) * cos_sq - z**2 * sin_sq

        def radius_value(p):
            """sqrt(x²+y²+z²) for near plane check."""
            x, y, z = p
            return np.sqrt(x**2 + y**2 + z**2)

        c1 = cone_value(p1)
        c2 = cone_value(p2)

        inside1 = c1 <= 0 and z1 > 0
        inside2 = c2 <= 0 and z2 > 0

        # Near plane: distance-based check (r >= near_z)
        r1 = radius_value(p1)
        r2 = radius_value(p2)
        r1_ok = r1 >= self.near_z
        r2_ok = r2 >= self.near_z

        # Handle near plane clipping
        near_clip = None
        if (r1_ok and not r2_ok) or (not r1_ok and r2_ok):
            # Line crosses near sphere
            # Parametric: r²(t) = (x1+tdx)² + (y1+tdy)² + (z1+tdz)² = near_z²
            a_r = dx**2 + dy**2 + dz**2
            b_r = 2 * (x1*dx + y1*dy + z1*dz)
            c_r = r1**2 - self.near_z**2
            disc_r = b_r**2 - 4*a_r*c_r
            if disc_r >= 0 and abs(a_r) > 1e-12:
                sqrt_disc_r = np.sqrt(disc_r)
                t1_r = (-b_r - sqrt_disc_r) / (2*a_r)
                t2_r = (-b_r + sqrt_disc_r) / (2*a_r)
                # Pick the valid root that's in [0,1]
                for t_r in [t1_r, t2_r]:
                    if 0 <= t_r <= 1:
                        p_near = p1 + t_r * (p2 - p1)
                        if cone_value(p_near) <= 0 and p_near[2] > 0:
                            near_clip = (t_r, p_near)
                        break

        # Compute cone intersections using cos/sin quadratic
        # (x1+tdx)² + (y1+tdy)²) * cos² - (z1+tdz)² * sin² = 0
        a = (dx**2 + dy**2) * cos_sq - dz**2 * sin_sq
        b = 2 * (x1*dx + y1*dy) * cos_sq - 2 * z1*dz * sin_sq
        c = c1

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                # Degenerate line or parallel to cone axis
                if inside1 and inside2 and r1_ok and r2_ok:
                    return (p1.copy(), p2.copy())
                return None

            t_cone = -c / b
            if 0 <= t_cone <= 1:
                p_cone = p1 + t_cone * (p2 - p1)
                if cone_value(p_cone) <= 0 and radius_value(p_cone) >= self.near_z and p_cone[2] > 0:
                    pass  # valid intersection, add to candidates below
                else:
                    t_cone = None
            elif inside1 and inside2 and r1_ok and r2_ok:
                return (p1.copy(), p2.copy())
        else:
            disc = b**2 - 4*a*c
            if disc < 0:
                if inside1 and inside2 and r1_ok and r2_ok:
                    return (p1.copy(), p2.copy())
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
                if inside1 and inside2 and r1_ok and r2_ok:
                    return (p1.copy(), p2.copy())
                return None

            if inside1 and inside2 and r1_ok and r2_ok:
                return (p1.copy(), p2.copy())

        # Build list of valid parameter intervals
        candidates = []

        if inside1 and r1_ok:
            candidates.append((0.0, p1.copy()))
        if inside2 and r2_ok:
            candidates.append((1.0, p2.copy()))

        # Add cone intersection points
        if abs(a) < 1e-12:
            if 't_cone' in dir() and t_cone is not None and 0 <= t_cone <= 1:
                candidates.append((t_cone, p_cone))
        else:
            for t in intersections:
                p_cone = p1 + t * (p2 - p1)
                # 只保留真实视锥边界交点（z 与 cos(θ_max) 同号），
                # 过滤双锥面"反面"的伪交点
                if p_cone[2] * self.cos_theta_max >= 0:
                    candidates.append((t, p_cone))

        # Add near plane intersection points
        if near_clip is not None:
            candidates.append(near_clip)

        # z=0 平面交点（物理近边界：光线必须从相机前方进入）
        # 一次方程 z1 + t*dz = 0  =>  t = -z1 / dz
        if abs(dz) > 1e-12:
            t_z0 = -z1 / dz
            if 0 <= t_z0 <= 1:
                p_z0 = p1 + t_z0 * (p2 - p1)
                if radius_value(p_z0) >= self.near_z:
                    candidates.append((t_z0, p_z0))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        # Find the longest valid segment
        best_start = None
        best_end = None
        best_length = -1

        for i in range(len(candidates) - 1):
            t_start, p_start = candidates[i]
            t_end, p_end = candidates[i + 1]

            t_mid = (t_start + t_end) / 2
            p_mid = p1 + t_mid * (p2 - p1)
            if cone_value(p_mid) <= 0 and radius_value(p_mid) >= self.near_z and p_mid[2] > 0:
                length = t_end - t_start
                if length > best_length:
                    best_length = length
                    best_start = (t_start, p_start)
                    best_end = (t_end, p_end)

        if best_start is None or best_end is None:
            return None

        return (best_start[1], best_end[1])

    def _clip_line_large_fov(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Clip line for large FOV (theta_max > 90°) cone frustum.

        当 theta_max > 90° 时，cos(theta_max) < 0，代数式
        (x²+y²)*cos² - z²*sin² <= 0 不再与几何判断 z/r >= cos(theta_max)
        等价（平方运算不保号）。因此改用直接几何式判断内外：

          inside(p) := r >= near_z AND z/r >= cos(theta_max)

        锥面方程 (x²+y²)*cos² - z²*sin² = 0 仍然有效（用于求交点），
        但内外判断改为几何式。

        算法流程：
        1. 几何式判断两端点是否在视锥内
        2. 若两端都在内：直接返回
        3. 否则：收集所有候选交点（近球面 + 锥面 + 端点）
        4. 按参数 t 排序，取中点在视锥内的最长子区间
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        d = p2 - p1

        cos_sq = self.cos_theta_max ** 2
        sin_sq = self.sin_theta_max ** 2
        near_z_sq = self.near_z ** 2

        def is_inside(p: np.ndarray) -> bool:
            """几何式内外判断：r >= near_z AND z/r >= cos(theta_max)。"""
            x, y, z = p
            if z <= 0:  # 光线必须从前方进入镜头
                return False
            r_sq = x * x + y * y + z * z
            if r_sq < near_z_sq:
                return False
            r = np.sqrt(r_sq)
            return z / r >= self.cos_theta_max

        def radius_sq(p: np.ndarray) -> float:
            x, y, z = p
            return x * x + y * y + z * z

        inside1 = is_inside(p1)
        inside2 = is_inside(p2)

        # 两端都在视锥内：无需裁剪
        if inside1 and inside2:
            return (p1.copy(), p2.copy())

        # 收集所有候选交点（参数 t, 3D 点）
        candidates = []
        if inside1:
            candidates.append((0.0, p1.copy()))
        if inside2:
            candidates.append((1.0, p2.copy()))

        # 1) 近球面交点（r = near_z）：参数方程 |p1 + t*d|² = near_z²
        a_r = float(d @ d)
        if a_r > 1e-12:
            b_r = 2.0 * float(p1 @ d)
            c_r = float(p1 @ p1) - near_z_sq
            disc_r = b_r * b_r - 4.0 * a_r * c_r
            if disc_r >= 0.0:
                sqrt_disc_r = np.sqrt(disc_r)
                for t_r in ((-b_r - sqrt_disc_r) / (2.0 * a_r),
                            (-b_r + sqrt_disc_r) / (2.0 * a_r)):
                    if 0.0 <= t_r <= 1.0:
                        p_near = p1 + t_r * d
                        # 交点必须同时满足锥面角度条件
                        if is_inside(p_near):
                            candidates.append((t_r, p_near))

        # 2) 锥面交点：(x²+y²)*cos² - z²*sin² = 0
        #    参数方程展开为 a*t² + b*t + c = 0
        x1, y1, z1 = p1
        dx, dy, dz = d
        a = (dx * dx + dy * dy) * cos_sq - dz * dz * sin_sq
        b = 2.0 * (x1 * dx + y1 * dy) * cos_sq - 2.0 * z1 * dz * sin_sq
        c = (x1 * x1 + y1 * y1) * cos_sq - z1 * z1 * sin_sq

        cone_roots = []
        if abs(a) > 1e-12:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = np.sqrt(disc)
                cone_roots = [(-b - sqrt_disc) / (2.0 * a),
                              (-b + sqrt_disc) / (2.0 * a)]
        elif abs(b) > 1e-12:
            # 退化为一元一次方程
            cone_roots = [-c / b]

        for t in cone_roots:
            if 0.0 <= t <= 1.0:
                p_cone = p1 + t * d
                # 锥面交点需满足 r >= near_z（在近球面外）
                if radius_sq(p_cone) >= near_z_sq:
                    # 只保留真实视锥边界交点（z 与 cos(θ_max) 同号），
                    # 过滤双锥面"反面"的伪交点
                    if p_cone[2] * self.cos_theta_max >= 0:
                        candidates.append((t, p_cone))

        # 3) z=0 平面交点（大FOV鱼眼的物理近边界：光线必须从相机前方进入）
        #    一次方程 z1 + t*dz = 0  =>  t = -z1 / dz
        if abs(dz) > 1e-12:
            t_z0 = -z1 / dz
            if 0.0 <= t_z0 <= 1.0:
                p_z0 = p1 + t_z0 * d
                # z=0 平面上的点只需满足 r >= near_z（角度条件在 z=0 处天然满足）
                if radius_sq(p_z0) >= near_z_sq:
                    candidates.append((t_z0, p_z0))

        if len(candidates) < 2:
            return None

        # 去重并按 t 排序
        candidates.sort(key=lambda x: x[0])
        unique = []
        for t, p in candidates:
            if not unique or abs(t - unique[-1][0]) > 1e-9:
                unique.append((t, p))
        candidates = unique

        # 在相邻候选点之间寻找中点在视锥内的最长子区间
        best_start = None
        best_end = None
        best_length = -1.0

        for i in range(len(candidates) - 1):
            t_start, p_start = candidates[i]
            t_end, p_end = candidates[i + 1]
            t_mid = 0.5 * (t_start + t_end)
            p_mid = p1 + t_mid * d
            if is_inside(p_mid):
                length = t_end - t_start
                if length > best_length:
                    best_length = length
                    best_start = (t_start, p_start)
                    best_end = (t_end, p_end)

        if best_start is None:
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