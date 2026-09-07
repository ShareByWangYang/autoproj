import numpy as np
import warnings
from typing import Optional, Tuple, Union, List, Dict, Any
from .camera import Camera, PinholeCamera, KannalaBrandtCamera, FThetaCamera
from .geometry import PointCloud, Box3D, LineSet, Polygon3D
from .frustum import FrustumCuller, FrustumType


class Projector:
    """
    统一 3D→2D 投影接口，支持视锥裁剪

    提供点云、线段、包围盒、多边形等多种几何体的投影能力，
    并可选在相机坐标系下对棱和多边形进行精确视锥裁剪
    （基于 Liang-Barsky / Sutherland-Hodgman 算法），
    避免 OpenCV 畸变模型将 FOV 外点"回折"到图像内的问题。

    坐标系契约（详见 README "Coordinate System Conventions"）：
        - 相机系：OpenCV 约定（x 右 / y 下 / z 前），原点为光心
        - 像素原点：图像左上角，u 向右增长，v 向下增长
        - 外参 T_to_cam：world→camera 的 4×4 齐次矩阵（p_cam = T @ p_world）
          若手头是 c2w（camera→world，SLAM/渲染域常见），需先调用
          conventions.invert_transform(T_c2w) 得到 w2c 再传入
        - 单位：米（near_z/far_z 默认 0.1/1000 米）
        - 非本约定的数据源（OpenGL/Blender、ROS REP-103 等）须在入口处
          通过 conventions 模块完成轴变换后再传入，核心不自动猜测坐标系

    Args:
        camera: 相机对象
        cull_frustum: 是否启用视锥裁剪（默认 True）
        frustum_scale: 视锥缩放因子
            - None: 自动计算（基于相机内参和畸变系数）
            - float: 固定缩放（1.0=不扩展，1.05=扩展5%）
        corner_match_tolerance: 角点匹配容差（像素），用于判断裁剪端点是否为原始角点（默认 3.0）
        soft_clip_ratio: 软裁剪比例（仅针孔相机），None=禁用，
            0.05 表示阈值为 min(width,height)*0.05 像素。
            用于修正矩形棱锥视锥在对角线方向的裁剪偏差：
            当被裁剪端点的原始角点投影仍在图像范围内，且与裁剪点投影
            距离小于阈值时，回退使用原始角点，避免对角线方向的误裁剪。
        health_check: 投影健康检查开关（默认 False）。
            开启后，每次 project_points 完成后统计 valid 比例、z<0 点比例、
            投影超出图像边界的比例，并对 T_to_cam 的 R 做正交性/行列式检查。
            当 valid 率异常低（<5%）时发 warning，列出可能原因（外参方向反了？
            轴向错？单位 mm？）。仅诊断不自动修正。

    向后兼容：
        Projector(camera, 2.0) 等价于 Projector(camera, frustum_scale=2.0)
        Projector(camera, frustum_expansion='auto') 等价于 Projector(camera, frustum_scale=None)
    """

    def __init__(
        self,
        camera: Camera,
        cull_frustum: bool = True,
        frustum_scale: Optional[float] = None,
        corner_match_tolerance: float = 3.0,
        soft_clip_ratio: Optional[float] = None,
        health_check: bool = False,
        **kwargs
    ):
        self.camera = camera
        self._corner_match_tolerance = corner_match_tolerance
        # 软裁剪比例：None=禁用，0.05=使用 min(w,h)*0.05 作为像素阈值
        # 仅对针孔相机（矩形棱锥视锥）生效，修正对角线方向的裁剪偏差
        self._soft_clip_ratio = soft_clip_ratio
        # 健康检查开关：开启后 project_points 末尾统计诊断
        self._health_check = health_check

        # 向后兼容：旧参数名 frustum_expansion
        if 'frustum_expansion' in kwargs:
            old_val = kwargs.pop('frustum_expansion')
            warnings.warn(
                "'frustum_expansion' is deprecated, use 'frustum_scale'. "
                "'auto' → None, float value → same float.",
                DeprecationWarning, stacklevel=2
            )
            if isinstance(old_val, str) and old_val.lower() == 'auto':
                frustum_scale = None  # auto → None
            elif isinstance(old_val, (int, float)):
                frustum_scale = float(old_val)

        # 向后兼容：第二个位置参数可以是 cull_frustum(bool) 或旧版 frustum_expansion(float)
        if isinstance(cull_frustum, (int, float)) and not isinstance(cull_frustum, bool):
            frustum_scale = float(cull_frustum)
            cull_frustum = True
        elif isinstance(cull_frustum, bool):
            pass  # 正常处理
        elif cull_frustum is None:
            cull_frustum = True

        self._cull_frustum = cull_frustum
        self._frustum_scale = frustum_scale

        # 计算有效缩放因子并创建裁剪器
        self._effective_expansion = self._compute_effective_expansion(frustum_scale)

        if cull_frustum:
            self._culler = FrustumCuller.from_camera(camera, self._effective_expansion)
        else:
            self._culler = None

    def _compute_effective_expansion(self, scale: Optional[float]) -> float:
        """
        计算有效视锥缩放因子

        当 scale=None 时，调用 camera.compute_expansion_factor()
        基于相机内参和畸变系数精确计算真实 FOV 扩展范围。
        同时将结果缓存到 camera._frustum_scale 供 _check_fov 使用。

        返回值说明：
            返回 camera._frustum_scale（默认 1.05），而非 _expansion_factor。
            此值传给 FrustumCuller.from_camera 作为 expansion_factor 参数，
            但在主要路径中被忽略（culler 使用预计算的 _frustum_tan / _theta_max），
            仅在 fallback 路径中作为 tan 乘数生效。
        """
        if scale is None:
            # 自动计算：调用相机的扩展因子计算
            if hasattr(self.camera, 'compute_expansion_factor'):
                result = self.camera.compute_expansion_factor()
                # compute_expansion_factor 会设置 camera._frustum_scale
                # 返回该值供 FrustumCuller.from_camera 使用
                return self.camera._frustum_scale or result
            return 1.0
        else:
            # 用户指定了固定缩放：仍需计算 _real_tan 等参数
            if hasattr(self.camera, 'compute_expansion_factor'):
                self.camera.compute_expansion_factor()
            # 覆盖为用户值
            self.camera._frustum_scale = float(scale)
            return float(scale)

    @property
    def cull_frustum(self) -> bool:
        """是否启用视锥裁剪"""
        return self._cull_frustum

    @property
    def culler(self) -> Optional[FrustumCuller]:
        """视锥裁剪器（可直接用于独立裁剪操作）"""
        return self._culler

    def set_frustum_scale(self, scale: Optional[float]) -> None:
        """
        动态更新视锥缩放因子

        Args:
            scale: 新的缩放因子，None 表示自动计算
        """
        self._frustum_scale = scale
        self._effective_expansion = self._compute_effective_expansion(scale)
        if self._cull_frustum:
            self._culler = FrustumCuller.from_camera(self.camera, self._effective_expansion)

    def set_frustum_expansion(self, expansion: Union[float, str]) -> None:
        """
        [DEPRECATED] 动态更新视锥扩展系数，请使用 set_frustum_scale

        Args:
            expansion: 新的扩展系数或 'auto'
        """
        warnings.warn(
            "'set_frustum_expansion' is deprecated, use 'set_frustum_scale'. "
            "'auto' → None, float value → same float.",
            DeprecationWarning, stacklevel=2
        )
        if isinstance(expansion, str) and expansion.lower() == 'auto':
            scale = None
        else:
            scale = float(expansion)
        self.set_frustum_scale(scale)

    def set_cull_frustum(self, enabled: bool) -> None:
        """
        动态启用/禁用视锥裁剪

        Args:
            enabled: 是否启用
        """
        self._cull_frustum = enabled
        if enabled:
            self._culler = FrustumCuller.from_camera(self.camera, self._effective_expansion)
        else:
            self._culler = None

    def _project_raw_pixels(
        self,
        points_cam: np.ndarray
    ) -> np.ndarray:
        """
        原始投影：仅执行 3D→2D 光学投影，不做边界检查、clamp 或置 -1。

        用于视锥裁剪后的棱端点投影——这些点已通过 FrustumCuller 的
        几何视锥检查，即使最终像素落在图像边界外（负值或>=size），
        也是线段与视锥面的真实交点，必须作为截断端点保留。

        与 Camera.project() 主路径的差异：
            - 主路径：返回 int32，无效点置 -1，有效点 clamp 到 [0, w-1]
            - 本路径：返回 float64，不做 -1/clamp，保留原始浮点精度，
              即使超出图像边界也原样返回（供裁剪算法求交使用）
            - 三种相机模型的 θ 计算、多项式、像素映射公式与主路径完全一致

        Args:
            points_cam: (N, 3) 相机坐标系下的 3D 点

        Returns:
            pixels: (N, 2) float64，原始 (u, v) 像素坐标（未 clamp）
        """
        cam = self.camera
        pts = np.asarray(points_cam, dtype=np.float64)

        # 空数组：返回 (0, 2) 空数组
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        n = pts.shape[0]
        x_c, y_c, z_c = pts[:, 0], pts[:, 1], pts[:, 2]

        # safe_z 仅用于针孔 x_norm/y_norm 的透视除法（防 z=0 除零）。
        # 注意：KB 和 FTheta 分支使用原始 z_c（支持 θ>90° 的边缘点 z<0），
        # 不走 safe_z，因此下方注释"clip_line 已保证 z>0"仅对针孔成立。
        safe_z = np.maximum(z_c, 1e-10)
        x_norm = x_c / safe_z
        y_norm = y_c / safe_z

        if isinstance(cam, PinholeCamera):
            # --- Pinhole (OpenCV rational polynomial) ---
            # 畸变系数提取为 Python float，避免 NumPy 标量在向量化运算中引入额外开销
            if cam.has_distortion:
                k1 = float(cam.k1); k2 = float(cam.k2); k3 = float(cam.k3)
                k4 = float(cam.k4); k5 = float(cam.k5); k6 = float(cam.k6)
                p1 = float(cam.p1); p2 = float(cam.p2)
                r2 = x_norm ** 2 + y_norm ** 2
                r4 = r2 ** 2
                r6 = r2 ** 3
                numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
                denom = 1 + k4 * r2 + k5 * r4 + k6 * r6
                denom = np.maximum(denom, 1e-10)
                radial = numerator / denom
                x_dist = x_norm * radial
                y_dist = y_norm * radial
                x_dist += 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
                y_dist += p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm
            else:
                x_dist, y_dist = x_norm, y_norm
            u = float(cam.fx) * x_dist + float(cam.cx)
            v = float(cam.fy) * y_dist + float(cam.cy)

        elif isinstance(cam, KannalaBrandtCamera):
            # --- Kannala-Brandt fisheye (theta polynomial) ---
            # 使用原始 z_c（而非 safe_z）计算入射角，支持 θ > 90° 的大鱼眼
            # （z<0 的边缘点）。与 KannalaBrandtCamera.project() 主路径一致：
            # arctan2(r_xy, z) 等价于 arccos(z/r_3d)，但省去 r_3d 开方与 clip。
            k1 = float(cam.k1); k2 = float(cam.k2)
            k3 = float(cam.k3); k4 = float(cam.k4)
            r_xy = np.sqrt(x_c**2 + y_c**2)
            theta = np.arctan2(r_xy, z_c)
            theta_d = theta + k1 * theta**3 + k2 * theta**5 + \
                      k3 * theta**7 + k4 * theta**9
            # 方向向量（仅取 xy 分量，保持投影方向）
            safe_r_xy = np.maximum(r_xy, 1e-10)
            scale = theta_d / safe_r_xy
            x_dist = x_c * scale
            y_dist = y_c * scale
            u = float(cam.fx) * x_dist + float(cam.cx)
            v = float(cam.fy) * y_dist + float(cam.cy)

        elif isinstance(cam, FThetaCamera):
            # --- F-Theta fisheye (polynomial: r = sum(coeff[i] * theta^i)) ---
            # 使用原始 z_c（而非 safe_z）确保 θ 计算正确处理 z < 0
            theta = np.arctan2(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2), z_c)
            # 多项式系数提取为 Python list，避免 NumPy ndarray 在 polyval 中的类型开销
            fw_poly_list = [float(c) for c in cam.fw_poly]
            # Horner 法则计算多项式，比逐项循环快
            r_dist = np.polynomial.polynomial.polyval(theta, fw_poly_list)
            phi = np.arctan2(pts[:, 1], pts[:, 0])
            u = r_dist * np.cos(phi) + float(cam.cx)
            v = r_dist * np.sin(phi) + float(cam.cy)

        else:
            # Fallback: 针孔模型无畸变
            u = float(cam.fx) * x_norm + float(cam.cx)
            v = float(cam.fy) * y_norm + float(cam.cy)

        return np.stack([u, v], axis=1)

    def project_points(
        self,
        points_3d: Union[np.ndarray, PointCloud],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        当 cull_frustum=True 时，使用 FrustumCuller 的扩展 FOV
        （含 frustum_scale 系数）验证点的有效性，
        替代 camera.project() 中固定 5% 容差的 _check_fov。

        Args:
            points_3d: 3D点云，形状为(N, 3+)，支持额外维度（intensity, gpstime等）
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 点云是否已在相机坐标系中
            preserve_extra: 是否保留额外列（depth + 原始额外列）

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v
            preserve_extra=True:
                result: (N, 3+) float64
            valid: (N,) 有效掩码（基于扩展 FOV 验证）
        """
        if isinstance(points_3d, PointCloud):
            points = points_3d.points
        else:
            points = np.asarray(points_3d)

        # 空数组：返回空结果，保持形状一致性
        if points.size == 0:
            n_cols = points.shape[1] if points.ndim >= 2 else 3
            if preserve_extra:
                empty_result = np.empty((0, n_cols + 1), dtype=np.float64)
            else:
                empty_result = np.empty((0, 2), dtype=np.int32)
            return empty_result, np.empty(0, dtype=bool)

        # NaN 检查：含 NaN 的点标记为无效
        nan_mask = np.any(np.isnan(points[:, :3]), axis=1)

        # 获取相机坐标系下的点
        if not pts_in_cam and T_to_cam is not None:
            # R @ p.T + t 广播，避免 hstack 构造 (N,4) 齐次数组的分配/拷贝
            _R = T_to_cam[:3, :3]
            _t = T_to_cam[:3, 3]
            points_cam = (_R @ points[:, :3].T + _t[:, None]).T
        else:
            points_cam = points[:, :3].copy()

        # 使用 culler 的扩展 FOV 验证点
        if self._cull_frustum and self._culler is not None:
            _, valid = self._culler.cull_points(points_cam)
        else:
            # 无 culler：仅检查深度
            z = points_cam[:, 2]
            valid = (z >= self.camera.near_z)

        # 调用 camera.project 完成畸变投影和像素计算
        # 传 preserve_extra=True 以获取 float64 原始坐标
        if preserve_extra:
            result, _ = self.camera.project(points, T_to_cam, pts_in_cam, preserve_extra=True)
        else:
            result, _ = self.camera.project(points, T_to_cam, pts_in_cam, preserve_extra=False)

        # 合并：culler FOV检查 AND 相机原始有效检查
        # 当启用 culler 时，使用 culler 的扩展 FOV 结果作为有效性依据
        # 不再使用 camera.project 内部的固定 5% 容差过滤
        if self._cull_frustum and self._culler is not None:
            # culler 的 FOV 检查已包含扩展因子，是更准确的几何判断
            # 相机投影仅用于计算像素坐标，不再用其内部的 FOV 检查结果
            # 对 culler 判定有效的点，若相机投影结果为 -1 或被 clamp 到边界，
            # 说明该点因畸变或边界检查被相机内部过滤了。
            # 需要保留 culler 的判断，将像素坐标重新计算
            # （使用 _project_raw_pixels 的原始像素计算）
            cam_invalid = (result[:, 0] < 0) & (result[:, 1] < 0)
            # 额外检测：被 clamp 到图像边界的点（可能是边界裁剪导致的）
            # 对于 preserve_extra=False (int32)：u,v 在 [0, w-1] 内但实际应在 FOV 外
            # 对于 preserve_extra=True (float64)：可以检测 float 值是否等于边界值
            if preserve_extra:
                # float64 模式：检测被 clamp 到边界的点
                cam_clamped = (
                    (result[:, 0] >= 0) & (
                        (result[:, 0] <= 0.5) | (result[:, 0] >= self.camera.width - 1.5) |
                        (result[:, 1] <= 0.5) | (result[:, 1] >= self.camera.height - 1.5)
                    )
                )
            else:
                # int32 模式：检测被 clamp 到边界的点
                cam_clamped = (
                    (result[:, 0] >= 0) & (
                        (result[:, 0] <= 0) | (result[:, 0] >= self.camera.width - 1) |
                        (result[:, 1] <= 0) | (result[:, 1] >= self.camera.height - 1)
                    )
                )

            # 对于 culler 有效但相机无效/clamp 的点，重新计算像素坐标
            culler_valid_cam_invalid = valid & (cam_invalid | cam_clamped)
            if np.any(culler_valid_cam_invalid):
                # 修复：对 culler 判定有效的点使用扩展 FOV 内的原始像素计算
                # 这些点在相机固定 FOV 检查中被过滤/clamp，但应在扩展 FOV 内
                fix_indices = np.where(culler_valid_cam_invalid)[0]
                fix_points = points_cam[fix_indices]
                raw_pixels = self._project_raw_pixels(fix_points)

                # 根据 preserve_extra 设置结果
                if preserve_extra:
                    # float64 模式：直接覆盖坐标（保留原始浮点精度）
                    result[fix_indices, 0] = raw_pixels[:, 0]
                    result[fix_indices, 1] = raw_pixels[:, 1]
                else:
                    # int32 模式：clamp 到 [0, w-1] 并转 int32
                    u_clip = np.clip(raw_pixels[:, 0], 0, self.camera.width - 1)
                    v_clip = np.clip(raw_pixels[:, 1], 0, self.camera.height - 1)
                    result[fix_indices, 0] = u_clip.astype(np.int32)
                    result[fix_indices, 1] = v_clip.astype(np.int32)

            # 最终有效性：完全基于 culler 的扩展 FOV 判断
            valid = valid.copy()

        # NaN 点标记为无效（坐标已由 camera.project 处理，这里仅更新掩码）
        valid = valid & ~nan_mask

        # 投影健康检查：统计诊断，发 warning 辅助定位静默错误
        if self._health_check:
            self._run_health_check(points_cam, result, valid, T_to_cam)

        return result, valid

    def _run_health_check(
        self,
        points_cam: np.ndarray,
        result: np.ndarray,
        valid: np.ndarray,
        T_to_cam: Optional[np.ndarray],
    ) -> None:
        """
        投影结果健康检查，对异常统计发 warning（不抛异常）。

        诊断项：
            - T_to_cam 的 R 正交性 / det(R)（镜像反射提示）
            - z<0 点比例（针孔相机应全部 z>0；若大量 z<0 可能外参方向反了）
            - valid 率（过低提示外参/轴向/单位/模型族错配）
            - 投影超出图像边界的比例
        """
        n_total = len(valid)
        if n_total == 0:
            return

        issues = []

        # 1. 外参合理性（若有）—— check_transform_sanity 会自行发 warning
        if T_to_cam is not None:
            from .conventions import check_transform_sanity
            try:
                check_transform_sanity(T_to_cam, direction='w2c')
            except ValueError:
                pass  # T 维度异常，已在其他路径报错

        # 2. z<0 点比例（针孔相机尤其敏感）
        z = points_cam[:, 2]
        n_z_neg = int(np.sum(z < 0))
        ratio_z_neg = n_z_neg / n_total
        # 鱼眼相机 θ>90° 的边缘点 z<0 可能有效，这里仅做提示
        is_fisheye = hasattr(self.camera, 'sub_type') and self.camera.sub_type in ('kannala', 'ftheta')
        if not is_fisheye and ratio_z_neg > 0.5:
            issues.append(
                f"相机系下 z<0 的点占比 {ratio_z_neg:.1%}（针孔相机应 z>0），"
                "可能外参方向弄反(w2c↔c2w)或轴向错配"
            )

        # 3. valid 率
        valid_ratio = float(np.sum(valid)) / n_total
        if valid_ratio < 0.05 and n_total > 10:
            issues.append(
                f"有效投影比例仅 {valid_ratio:.1%}，可能原因：外参方向反了/"
                "轴向错配/单位为 mm(未转 m)/畸变模型族错配/数据不在相机视野内"
            )

        # 4. 投影超出图像边界的比例（基于 raw pixels，不受 clamp 影响）
        if result.shape[1] >= 2:
            u = result[:, 0].astype(np.float64)
            v = result[:, 1].astype(np.float64)
            w, h = self.camera.width, self.camera.height
            out_of_bounds = (u < 0) | (u >= w) | (v < 0) | (v >= h)
            # 仅统计 valid=True 但落边界外的（invalid 的 -1 坐标不算）
            oob_valid = out_of_bounds & valid
            ratio_oob = float(np.sum(oob_valid)) / n_total
            if ratio_oob > 0.3:
                issues.append(
                    f"valid 点中 {ratio_oob:.1%} 投影超出图像边界，"
                    "可能 cx/cy 主点错配或分辨率未缩放(scale_intrinsics)"
                )

        if issues:
            msg = "[health_check] 投影健康检查发现异常: " + "; ".join(issues)
            warnings.warn(msg, stacklevel=2)

    def project_box(
        self,
        box_input: Union[np.ndarray, Box3D],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None,
        extend_to_boundary: bool = False
    ) -> Dict[str, Any]:
        """
        投影 3D 包围盒到图像平面

        支持两种输入格式：
        1. (8, 3) ndarray: 包围盒的8个角点
        2. Box3D 对象: 使用其 corners 属性获取角点

        当启用视锥裁剪时，会对包围盒的12条棱逐条进行
        Liang-Barsky 裁剪，仅投影可见部分。

        Args:
            box_input: (8, 3) 角点数组 或 Box3D 对象
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 是否已在相机坐标系
            cull_frustum: 覆盖默认裁剪行为
            extend_to_boundary: 是否将裁剪点延长到图像边界（2D视觉优化）

        Returns:
            字典包含：
                'corners': (8, 2+) 角点投影像素
                'valid': (8,) 角点有效掩码
                'edges': List[Dict] 裁剪后的棱线结果，每项包含：
                    'pt1': (2,) 端点1像素坐标（裁剪后，延长后可能更新）
                    'pt2': (2,) 端点2像素坐标（裁剪后，延长后可能更新）
                    'draw_pt1': (2,) 实际绘制用端点1（=pt1，便于调用方直接使用）
                    'draw_pt2': (2,) 实际绘制用端点2（=pt2，便于调用方直接使用）
                    'pt1_is_corner': bool 端点1是否为原始角点
                    'pt2_is_corner': bool 端点2是否为原始角点
                'box': 原始 Box3D 对象或 None
        """
        if isinstance(box_input, Box3D):
            box_obj = box_input
            corners = box_input.get_corners()
        else:
            corners = np.asarray(box_input, dtype=np.float64)
            box_obj = None

        # 形状验证：角点必须是 (8, 3)
        if corners.shape != (8, 3):
            raise ValueError(f"box_input must have 8 corners of shape (8, 3), got {corners.shape}")

        # NaN 检查：角点含 NaN 时抛出异常（3D 包围盒不应有缺失坐标）
        if np.any(np.isnan(corners)):
            raise ValueError("box_input contains NaN values in corner coordinates")

        # 变换到相机坐标系
        if not pts_in_cam and T_to_cam is not None:
            corners_h = np.hstack([corners, np.ones((len(corners), 1))])
            corners_cam = (T_to_cam @ corners_h.T).T[:, :3]
        else:
            corners_cam = corners

        # 投影角点
        pixels, valid = self.camera.project(corners_cam, pts_in_cam=True)

        # 确定裁剪行为
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        # 预投影所有原始角点到2D（不裁剪），用于后续识别裁剪点
        raw_corner_pixels = self._project_raw_pixels(corners_cam)

        # 对12条棱逐条裁剪
        edges = []
        if cull_frustum and self._culler is not None:
            # 预先收集所有棱的裁剪结果
            clipped_edges_cam = []  # List[(p1_clipped_cam, p2_clipped_cam, i, j)]
            for i, j in LineSet.BOX_EDGES:
                clipped = self._culler.clip_line(corners_cam[i], corners_cam[j])
                if clipped is not None:
                    p1_clipped, p2_clipped = clipped
                    near_sq = self.camera.near_z ** 2
                    r1_sq = p1_clipped[0]**2 + p1_clipped[1]**2 + p1_clipped[2]**2
                    r2_sq = p2_clipped[0]**2 + p2_clipped[1]**2 + p2_clipped[2]**2
                    if r1_sq >= near_sq and r2_sq >= near_sq:
                        clipped_edges_cam.append((p1_clipped, p2_clipped, i, j))

            # 批量投影所有裁剪后的端点到像素坐标
            if clipped_edges_cam:
                all_pts = np.array(
                    [list(pair[:2]) for pair in clipped_edges_cam],
                    dtype=np.float64
                ).reshape(-1, 3)
                all_pixels_raw = self._project_raw_pixels(all_pts)

                # 构建角点像素集合（用于容差匹配）
                corner_px_set = set()
                for px, py in raw_corner_pixels:
                    corner_px_set.add((round(px, 1), round(py, 1)))

                idx = 0
                for p1_cam, p2_cam, i, j in clipped_edges_cam:
                    p1_px = all_pixels_raw[idx]
                    p2_px = all_pixels_raw[idx + 1]
                    idx += 2

                    # 判断端点是否为原始角点（使用可配置的容差）
                    p1_is_corner = self._match_corner_pixel(p1_px, corner_px_set, self._corner_match_tolerance)
                    p2_is_corner = self._match_corner_pixel(p2_px, corner_px_set, self._corner_match_tolerance)

                    # 软裁剪：修正针孔相机对角线方向的裁剪偏差
                    # 仅对针孔相机（矩形棱锥视锥）生效，鱼眼相机使用锥形视锥无此问题
                    if (self._soft_clip_ratio is not None and
                            self._culler is not None and
                            self._culler.frustum_type == FrustumType.PYRAMID):
                        pixel_threshold = self._soft_clip_ratio * min(self.camera.width, self.camera.height)
                        # 1像素 margin 防止数值精度导致误拒
                        margin = 1
                        w, h = self.camera.width, self.camera.height

                        if not p1_is_corner:
                            orig_px = raw_corner_pixels[i]
                            if (-margin <= orig_px[0] <= w + margin and
                                -margin <= orig_px[1] <= h + margin):
                                dist = np.linalg.norm(p1_px - orig_px)
                                if dist < pixel_threshold:
                                    p1_px = orig_px
                                    p1_is_corner = True

                        if not p2_is_corner:
                            orig_px = raw_corner_pixels[j]
                            if (-margin <= orig_px[0] <= w + margin and
                                -margin <= orig_px[1] <= h + margin):
                                dist = np.linalg.norm(p2_px - orig_px)
                                if dist < pixel_threshold:
                                    p2_px = orig_px
                                    p2_is_corner = True

                    edge_dict = {
                        'pt1': p1_px,
                        'pt2': p2_px,
                        'draw_pt1': p1_px,
                        'draw_pt2': p2_px,
                        'pt1_is_corner': p1_is_corner,
                        'pt2_is_corner': p2_is_corner,
                    }
                    edges.append(edge_dict)
        else:
            # 无裁剪：直接连接有效角点
            for i, j in LineSet.BOX_EDGES:
                if valid[i] and valid[j]:
                    edges.append({
                        'pt1': pixels[i],
                        'pt2': pixels[j],
                        'draw_pt1': pixels[i],
                        'draw_pt2': pixels[j],
                        'pt1_is_corner': True,
                        'pt2_is_corner': True,
                    })

        # 2D 延长到图像边界
        if extend_to_boundary and edges:
            edges = self.extend_edges_to_boundary(edges, self.camera.width, self.camera.height)

        return {
            'corners': pixels,
            'valid': valid,
            'edges': edges,
            'box': box_obj
        }

    # 批量模式下, 角点数 > 此值时启用向量化路径
    _BATCH_BOX_THRESHOLD = 64

    def project_boxes(
        self,
        boxes: Union[np.ndarray, List, 'Box3D'],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None,
        extend_to_boundary: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量投影 3D 包围盒集合到图像平面

        混合策略 (方案 C):
        - N <= _BATCH_BOX_THRESHOLD: 调用现有 project_box 逐条处理, 保兼容性
        - N >  _BATCH_BOX_THRESHOLD: 走向量化路径,
          一次性变换/投影所有角点 + clip_lines_batch 批量裁剪 + 批量投影裁剪点

        返回格式与 project_box 完全一致 (List[Dict]), 调用方可直接迭代使用.

        Args:
            boxes: 以下任一:
                - (N, 8, 3) ndarray: N 个包围盒的角点数组
                - List[Box3D]: Box3D 对象列表
                - List[np.ndarray]: 每个元素为 (8, 3) 角点数组
            T_to_cam: 外参变换矩阵 (4x4)
            pts_in_cam: 是否已在相机坐标系
            cull_frustum: 覆盖默认裁剪行为
            extend_to_boundary: 是否将裁剪点延长到图像边界

        Returns:
            List[Dict], 长度与输入一致, 每项结构与 project_box() 返回值相同
        """
        from .geometry import Box3D as _Box3D  # 局部导入避免循环依赖

        # 统一输入为 (N, 8, 3) ndarray + 暂存 Box3D 对象
        box_objs: List[Optional[_Box3D]] = []
        if isinstance(boxes, _Box3D):
            # 单个 Box3D 对象, 包装成列表
            box_objs = [boxes]
            corners_arr = np.array([boxes.get_corners()], dtype=np.float64)
        elif isinstance(boxes, np.ndarray):
            if boxes.ndim == 2 and boxes.shape == (8, 3):
                # 单个 (8, 3) ndarray
                corners_arr = boxes.astype(np.float64).reshape(1, 8, 3)
                box_objs = [None]
            elif boxes.ndim == 3 and boxes.shape[1:] == (8, 3):
                # (N, 8, 3) ndarray
                corners_arr = boxes.astype(np.float64)
                box_objs = [None] * boxes.shape[0]
            else:
                raise ValueError(
                    f"boxes ndarray must be (8,3) or (N,8,3), got {boxes.shape}"
                )
        elif isinstance(boxes, list):
            if len(boxes) == 0:
                return []
            # List[Box3D] 或 List[ndarray]
            arr_list = []
            for item in boxes:
                if isinstance(item, _Box3D):
                    box_objs.append(item)
                    arr_list.append(item.get_corners())
                else:
                    box_objs.append(None)
                    arr_list.append(np.asarray(item, dtype=np.float64))
            # 验证形状
            for i, a in enumerate(arr_list):
                if a.shape != (8, 3):
                    raise ValueError(
                        f"box[{i}] must have shape (8, 3), got {a.shape}"
                    )
            corners_arr = np.array(arr_list, dtype=np.float64)
        else:
            raise TypeError(
                f"boxes must be ndarray, List[Box3D], or List[ndarray], got {type(boxes)}"
            )

        N = corners_arr.shape[0]
        if N == 0:
            return []

        # NaN 检查
        if np.any(np.isnan(corners_arr)):
            bad_idx = np.where(np.isnan(corners_arr).any(axis=(1, 2)))[0]
            raise ValueError(
                f"boxes contain NaN values at indices: {bad_idx.tolist()}"
            )

        # 小批量: 直接走单条 project_box (兼容性优先)
        if N <= self._BATCH_BOX_THRESHOLD:
            results = []
            for i in range(N):
                result = self.project_box(
                    corners_arr[i], T_to_cam=T_to_cam, pts_in_cam=pts_in_cam,
                    cull_frustum=cull_frustum, extend_to_boundary=extend_to_boundary
                )
                # 替换 box 引用
                result['box'] = box_objs[i]
                results.append(result)
            return results

        # 大批量向量化路径
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        # 1) 一次性批量变换所有角点 (N*8, 3)
        all_corners_flat = corners_arr.reshape(N * 8, 3)
        if not pts_in_cam and T_to_cam is not None:
            # R @ p.T + t 广播，避免 hstack 构造 (N*8,4) 齐次数组
            _R = T_to_cam[:3, :3]
            _t = T_to_cam[:3, 3]
            all_corners_cam = (_R @ all_corners_flat.T + _t[:, None]).T
        else:
            all_corners_cam = all_corners_flat
        # reshape 回 (N, 8, 3)
        corners_cam = all_corners_cam.reshape(N, 8, 3)

        # 2) 批量投影所有原始角点 (camera.project 一次完成)
        pixels_flat, valid_flat = self.camera.project(
            all_corners_cam, pts_in_cam=True
        )
        # pixels_flat: (N*8, 2+), valid_flat: (N*8,)
        # 提取 2D 像素坐标
        pixels_2d = pixels_flat[:, :2].reshape(N, 8, 2)
        valid_corners = valid_flat.reshape(N, 8)

        # 预投影所有原始角点 (raw, 不裁剪), 用于 corner 匹配
        raw_corner_pixels_flat = self._project_raw_pixels(all_corners_cam)
        raw_corner_pixels = raw_corner_pixels_flat.reshape(N, 8, 2)

        # 3) 构造所有棱 (N*12, 2 端点) —— 全向量化 fancy-indexing
        # BOX_EDGES: List[(i, j)] 12 条；edge_def[k] = (角点i, 角点j)
        from .geometry import LineSet
        edge_def = np.asarray(LineSet.BOX_EDGES, dtype=np.int64)  # (12, 2)
        n_edges_per_box = edge_def.shape[0]
        total_edges = N * n_edges_per_box  # 12N

        # corners_cam: (N,8,3) -> (N,12,3) -> (N*12,3)，索引顺序 idx = b*12+e
        edge_p1s = corners_cam[:, edge_def[:, 0], :].reshape(total_edges, 3)
        edge_p2s = corners_cam[:, edge_def[:, 1], :].reshape(total_edges, 3)

        # 4) 批量裁剪 (走 clip_lines_batch, 内部自动 Numba 加速)
        if cull_frustum and self._culler is not None:
            clipped, edge_valid = self._culler.clip_lines_batch(edge_p1s, edge_p2s)
            # clipped: (total_edges, 2, 3), edge_valid: (total_edges,)
        else:
            # 无裁剪: 直接用原始角点连线，构造"伪裁剪"结果 = 原端点
            clipped = np.empty((total_edges, 2, 3), dtype=np.float64)
            clipped[:, 0] = edge_p1s
            clipped[:, 1] = edge_p2s
            # valid = 两端点都有效（向量化）
            ev1 = valid_corners[:, edge_def[:, 0]]  # (N, 12)
            ev2 = valid_corners[:, edge_def[:, 1]]  # (N, 12)
            edge_valid = (ev1 & ev2).reshape(total_edges)

        # 5) 批量投影所有裁剪端点 (一次完成)
        valid_indices = np.where(edge_valid)[0]
        E = len(valid_indices)

        # 每条有效边对应的 box 索引 / 边内索引 / 两端角点索引
        vb = valid_indices // n_edges_per_box       # (E,)
        ve = valid_indices % n_edges_per_box        # (E,)
        ci = edge_def[ve, 0]                         # (E,)
        cj = edge_def[ve, 1]                         # (E,)

        # 软裁剪参数
        soft_ratio = self._soft_clip_ratio
        soft_threshold = (
            soft_ratio * min(self.camera.width, self.camera.height)
            if soft_ratio is not None else 0.0
        )
        w_cam, h_cam = self.camera.width, self.camera.height
        soft_margin = 1
        corner_tolerance = self._corner_match_tolerance

        # 端点像素 (E,2,2)：ep[k,0]=p1, ep[k,1]=p2
        p1_px = np.zeros((E, 2), dtype=np.float64)
        p2_px = np.zeros((E, 2), dtype=np.float64)
        keep_mask = np.ones(E, dtype=bool)

        if E > 0:
            clip_pts = clipped[valid_indices].reshape(-1, 3)  # (E*2, 3)
            clip_pixels = self._project_raw_pixels(clip_pts)  # (E*2, 2)
            ep = clip_pixels.reshape(E, 2, 2)

            # CONE 类型近似裁剪回退: NaN 标记的边改用 clip_line 精确处理
            needs_precise = np.isnan(clip_pts).any(axis=1).reshape(E, 2).any(axis=1)
            prec_idx = np.where(needs_precise)[0]
            if self._culler is not None and prec_idx.size > 0:
                near_sq = self.camera.near_z ** 2
                for k in prec_idx:
                    b_k, i_k, j_k = vb[k], ci[k], cj[k]
                    precise = self._culler.clip_line(
                        corners_cam[b_k, i_k], corners_cam[b_k, j_k]
                    )
                    if precise is None:
                        keep_mask[k] = False
                        continue
                    p1_c, p2_c = precise
                    if (p1_c[0]**2 + p1_c[1]**2 + p1_c[2]**2 < near_sq or
                            p2_c[0]**2 + p2_c[1]**2 + p2_c[2]**2 < near_sq):
                        keep_mask[k] = False
                        continue
                    precise_px = self._project_raw_pixels(np.array([p1_c, p2_c]))
                    ep[k, 0] = precise_px[0]
                    ep[k, 1] = precise_px[1]

            p1_px = ep[:, 0].copy()
            p2_px = ep[:, 1].copy()

            # --- 角点匹配（向量化）---
            # 与 _match_corner_pixel 语义一致：round(.,1) 后 Chebyshev(L∞) 距离 <= tol
            # box_corners: (E,8,2) 每条边所属框的 8 个原始角点像素
            box_corners = raw_corner_pixels[vb]            # (E,8,2)
            rc = np.round(box_corners, 1)                  # (E,8,2)
            rp1 = np.round(p1_px, 1)[:, None, :]           # (E,1,2)
            rp2 = np.round(p2_px, 1)[:, None, :]           # (E,1,2)
            p1_is_corner = (np.abs(rc - rp1).max(axis=2).min(axis=1)
                            <= corner_tolerance)
            p2_is_corner = (np.abs(rc - rp2).max(axis=2).min(axis=1)
                            <= corner_tolerance)

            # --- 软裁剪（针孔棱锥对角线偏差修正，向量化）---
            if (soft_ratio is not None and self._culler is not None and
                    self._culler.frustum_type == FrustumType.PYRAMID):
                orig_p1 = raw_corner_pixels[vb, ci]        # (E,2)
                orig_p2 = raw_corner_pixels[vb, cj]        # (E,2)

                def _soft_apply(p_px, orig, is_corner):
                    inb = ((orig[:, 0] >= -soft_margin) & (orig[:, 0] <= w_cam + soft_margin) &
                           (orig[:, 1] >= -soft_margin) & (orig[:, 1] <= h_cam + soft_margin))
                    dist = np.linalg.norm(p_px - orig, axis=1)
                    repl = (~is_corner) & inb & (dist < soft_threshold)
                    p_out = np.where(repl[:, None], orig, p_px)
                    return p_out, is_corner | repl

                p1_px, p1_is_corner = _soft_apply(p1_px, orig_p1, p1_is_corner)
                p2_px, p2_is_corner = _soft_apply(p2_px, orig_p2, p2_is_corner)
        else:
            p1_is_corner = np.zeros(0, dtype=bool)
            p2_is_corner = np.zeros(0, dtype=bool)

        # 6) 组装每个 box 的结果字典 (与 project_box 返回格式一致)
        results: List[Dict[str, Any]] = []
        box_edges_dict: List[List[Optional[Dict]]] = [
            [None] * n_edges_per_box for _ in range(N)
        ]
        # 仅保留未被精确裁剪丢弃的边
        keep_idx = np.where(keep_mask)[0]
        for k in keep_idx:
            b, e = int(vb[k]), int(ve[k])
            box_edges_dict[b][e] = {
                'pt1': p1_px[k],
                'pt2': p2_px[k],
                'draw_pt1': p1_px[k],
                'draw_pt2': p2_px[k],
                'pt1_is_corner': bool(p1_is_corner[k]),
                'pt2_is_corner': bool(p2_is_corner[k]),
            }

        # 7) 组装最终结果
        for b in range(N):
            edges = [e for e in box_edges_dict[b] if e is not None]
            if extend_to_boundary and edges:
                edges = self.extend_edges_to_boundary(
                    edges, self.camera.width, self.camera.height
                )
            results.append({
                'corners': pixels_2d[b],
                'valid': valid_corners[b],
                'edges': edges,
                'box': box_objs[b],
            })

        return results

    @staticmethod
    def _match_corner_pixel(px, corner_set, tolerance):
        """判断投影点是否匹配某个原始角点坐标"""
        rx, ry = round(px[0], 1), round(px[1], 1)
        if (rx, ry) in corner_set:
            return True
        for (cx, cy) in corner_set:
            if abs(rx - cx) <= tolerance and abs(ry - cy) <= tolerance:
                return True
        return False

    @staticmethod
    def extend_edges_to_boundary(
        edges: List[Dict],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """将裁剪点沿方向延长到图像边界（2D视觉优化）

        对每条边，若一端为角点、另一端为裁剪点，
        则从角点沿角点→裁剪点方向延长到图像边界。

        延长后会同步更新 draw_pt1/draw_pt2，使调用方可直接使用
        draw_pt1→draw_pt2 绘制最终线段，无需在调用方处理延长逻辑。

        返回的边字典会新增以下元数据字段：
        - is_extended: bool，表示该边是否存在延长段
        - extension_pt: 延长线端点（与延长后的 pt1/pt2 一致）

        Args:
            edges: project_box() 返回的边列表
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            延长后的边列表（修改原字典中的 pt1/pt2/draw_pt1/draw_pt2，并添加 is_extended/extension_pt）
        """
        for edge in edges:
            pt1 = edge['pt1']
            pt2 = edge['pt2']
            p1_corner = edge['pt1_is_corner']
            p2_corner = edge['pt2_is_corner']

            # 初始化元数据
            edge['is_extended'] = False
            edge['extension_pt'] = None

            if p1_corner and not p2_corner:
                boundary_pt = Projector._ray_to_rect_boundary(pt1, pt2, img_width, img_height)
                if boundary_pt is not None:
                    new_pt2 = np.array(boundary_pt, dtype=np.float64)
                    # 检查是否实际产生了延长（边界点与原裁剪点不同）
                    if not np.allclose(new_pt2, pt2, atol=1.0):
                        edge['pt2'] = new_pt2
                        edge['pt2_is_corner'] = False
                        edge['is_extended'] = True
                        edge['extension_pt'] = tuple(boundary_pt)
            elif p2_corner and not p1_corner:
                boundary_pt = Projector._ray_to_rect_boundary(pt2, pt1, img_width, img_height)
                if boundary_pt is not None:
                    new_pt1 = np.array(boundary_pt, dtype=np.float64)
                    # 检查是否实际产生了延长
                    if not np.allclose(new_pt1, pt1, atol=1.0):
                        edge['pt1'] = new_pt1
                        edge['pt1_is_corner'] = False
                        edge['is_extended'] = True
                        edge['extension_pt'] = tuple(boundary_pt)

            # 同步 draw_pt1/draw_pt2（延长后 pt1/pt2 已更新为最终绘制坐标）
            edge['draw_pt1'] = edge['pt1']
            edge['draw_pt2'] = edge['pt2']

        return edges

    @staticmethod
    def _ray_to_rect_boundary(
        pt_origin,
        pt_dir,
        img_width: int,
        img_height: int
    ) -> Optional[Tuple[float, float]]:
        """射线-矩形边界求交

        从 pt_origin 沿 pt_origin→pt_dir 方向延长到图像边界。

        Args:
            pt_origin: 射线起点 (x, y)
            pt_dir: 射线方向参考点 (x, y)
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            边界交点 (x, y)，若射线无法到达边界返回 None
        """
        x1, y1 = float(pt_origin[0]), float(pt_origin[1])
        x2, y2 = float(pt_dir[0]), float(pt_dir[1])

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        candidates = []

        if dx > 1e-6:
            t = (img_width - 1 - x1) / dx
            if t > 0:
                candidates.append((t, img_width - 1, y1 + t * dy))
        elif dx < -1e-6:
            t = (0 - x1) / dx
            if t > 0:
                candidates.append((t, 0, y1 + t * dy))

        if dy > 1e-6:
            t = (img_height - 1 - y1) / dy
            if t > 0:
                candidates.append((t, x1 + t * dx, img_height - 1))
        elif dy < -1e-6:
            t = (0 - y1) / dy
            if t > 0:
                candidates.append((t, x1 + t * dx, 0))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        _, bx, by = candidates[0]

        bx = max(0, min(img_width - 1, bx))
        by = max(0, min(img_height - 1, by))

        return (bx, by)

    def project_lines(
        self,
        lines_input: Union[np.ndarray, LineSet, List[Tuple[np.ndarray, np.ndarray]]],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        投影线段集合到图像平面

        支持三种输入格式：
        1. (N, 2, 3) ndarray: 每个线段的两个端点
        2. LineSet 对象: 使用其线段定义
        3. List[Tuple[ndarray, ndarray]]: 线段端点列表

        Args:
            lines_input: 线段集合
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 是否已在相机坐标系
            cull_frustum: 覆盖默认裁剪行为

        Returns:
            List of ((u1, v1), (u2, v2)) 像素坐标对
            裁剪后完全在视锥外的线段返回 None
        """
        # 统一输入格式为 (N, 2, 3) 数组
        if isinstance(lines_input, LineSet):
            segments = [lines_input.get_segment(i) for i in range(lines_input.num_segments)]
        elif isinstance(lines_input, list):
            segments = lines_input
        else:
            # ndarray (N, 2, 3)
            lines_arr = np.asarray(lines_input, dtype=np.float64)
            segments = [(lines_arr[i, 0], lines_arr[i, 1]) for i in range(lines_arr.shape[0])]

        # 确定裁剪行为
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        n_seg = len(segments)
        if n_seg == 0:
            return []

        # 批量变换所有端点到相机坐标系
        all_pts = np.array(segments, dtype=np.float64).reshape(-1, 3)  # (2*n_seg, 3)
        if not pts_in_cam and T_to_cam is not None:
            ones = np.ones((len(all_pts), 1))
            all_pts_h = np.hstack([all_pts, ones])
            all_pts_cam = (T_to_cam @ all_pts_h.T).T[:, :3]
        else:
            all_pts_cam = all_pts

        p1s = all_pts_cam[0::2]  # (n_seg, 3) 所有起点
        p2s = all_pts_cam[1::2]  # (n_seg, 3) 所有终点

        result = []

        if cull_frustum and self._culler is not None:
            # 视锥裁剪路径：使用 clip_lines_batch 一次性批量裁剪 (内部 Numba JIT 加速)
            # 替代逐条 clip_line 的 Python for 循环, 大规模线段场景显著加速
            clipped, valid = self._culler.clip_lines_batch(p1s, p2s)
            # valid[i]=True 表示该线段裁剪后有效 (含 near_sq 径向距离校验)
            # valid[i]=False 表示整段在视锥外或径向距离过近

            # 批量投影所有裁剪后的有效端点 (一次完成)
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                clip_pts = clipped[valid_indices].reshape(-1, 3)  # (M*2, 3)
                clip_pixels = self._project_raw_pixels(clip_pts)  # (M*2, 2+)

                # 按 seg_idx 顺序填充结果 (保持与输入顺序一致)
                next_valid_idx = 0
                for i in range(n_seg):
                    if next_valid_idx < len(valid_indices) and valid_indices[next_valid_idx] == i:
                        px1 = clip_pixels[next_valid_idx * 2]
                        px2 = clip_pixels[next_valid_idx * 2 + 1]
                        result.append((px1, px2))
                        next_valid_idx += 1
                    else:
                        result.append(None)
            else:
                result = [None] * n_seg
        else:
            # 无裁剪：批量投影所有端点
            all_pixels, all_valid = self.camera.project(all_pts_cam, pts_in_cam=True)
            for i in range(n_seg):
                if all_valid[i * 2] and all_valid[i * 2 + 1]:
                    result.append((all_pixels[i * 2], all_pixels[i * 2 + 1]))
                else:
                    result.append(None)

        return result

    def project_polygon(
        self,
        polygon: Polygon3D,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        投影 3D 多边形到图像平面

        Args:
            polygon: Polygon3D 对象
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 是否已在相机坐标系
            cull_frustum: 覆盖默认裁剪行为

        Returns:
            字典包含：
                'vertices': (N, 2+) 顶点投影像素
                'valid': (N,) 顶点有效掩码
                'edges': List[(pixel0, pixel1)] 裁剪后的边像素对
                'polygon': 原始 Polygon3D 对象
        """
        vertices = polygon.vertices

        # 空多边形：返回空结果
        if len(vertices) == 0:
            return {
                'vertices': np.empty((0, 2), dtype=np.int32),
                'valid': np.empty(0, dtype=bool),
                'edges': [],
                'polygon': polygon
            }

        # 变换到相机坐标系
        if not pts_in_cam and T_to_cam is not None:
            vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
            vertices_cam = (T_to_cam @ vertices_h.T).T[:, :3]
        else:
            vertices_cam = vertices

        # 投影顶点
        pixels, valid = self.camera.project(vertices_cam, pts_in_cam=True)

        # 确定裁剪行为
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        # 对每条边进行裁剪
        edges = []
        n = len(vertices_cam)
        if cull_frustum and self._culler is not None:
            for i in range(n):
                if polygon.is_closed:
                    j = (i + 1) % n
                else:
                    if i >= n - 1:
                        break
                    j = i + 1

                clipped = self._culler.clip_line(vertices_cam[i], vertices_cam[j])
                if clipped is not None:
                    p1_clipped, p2_clipped = clipped
                    # 保护 z>0
                    if p1_clipped[2] <= 1e-6 or p2_clipped[2] <= 1e-6:
                        continue
                    edge_pixels_raw = self._project_raw_pixels(
                        np.array([p1_clipped, p2_clipped])
                    )
                    edges.append((edge_pixels_raw[0], edge_pixels_raw[1]))
        else:
            for i in range(n):
                if polygon.is_closed:
                    j = (i + 1) % n
                else:
                    if i >= n - 1:
                        break
                    j = i + 1
                if valid[i] and valid[j]:
                    edges.append((pixels[i], pixels[j]))

        return {
            'vertices': pixels,
            'valid': valid,
            'edges': edges,
            'polygon': polygon
        }

    # 批量模式下, 多边形数 > 此值时启用向量化路径
    _BATCH_POLYGON_THRESHOLD = 32

    def project_polygons(
        self,
        polygons: Union[List, np.ndarray],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        批量投影 3D 多边形集合到图像平面

        混合策略 (方案 C):
        - N <= _BATCH_POLYGON_THRESHOLD: 调用现有 project_polygon 逐条处理
        - N >  _BATCH_POLYGON_THRESHOLD: 走向量化路径,
          一次性变换/投影所有顶点 + clip_lines_batch 批量裁剪边

        注意: 各多边形顶点数可不同, 因此批量向量化路径需按"边"为单元处理,
        而非按"多边形". CONE 类型近似裁剪回退逻辑同 project_boxes.

        Args:
            polygons: List[Polygon3D] 或 List[(M, 3) ndarray]
                各多边形顶点数可不同
            T_to_cam: 外参变换矩阵 (4x4)
            pts_in_cam: 是否已在相机坐标系
            cull_frustum: 覆盖默认裁剪行为

        Returns:
            List[Dict], 长度与输入一致, 每项结构与 project_polygon() 返回值相同
        """
        if len(polygons) == 0:
            return []

        from .geometry import Polygon3D as _Polygon3D

        # 统一输入: List[(M, 3) ndarray] + 暂存 Polygon3D 对象
        vert_list = []
        poly_objs: List[Optional[_Polygon3D]] = []
        is_closed_list = []
        for p in polygons:
            if isinstance(p, _Polygon3D):
                poly_objs.append(p)
                vert_list.append(np.asarray(p.vertices, dtype=np.float64))
                is_closed_list.append(p.is_closed)
            else:
                poly_objs.append(None)
                v = np.asarray(p, dtype=np.float64)
                if v.ndim != 2 or v.shape[1] != 3:
                    raise ValueError(
                        f"polygon vertices must have shape (M, 3), got {v.shape}"
                    )
                vert_list.append(v)
                is_closed_list.append(False)

        N = len(vert_list)
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        # 小批量: 走单条 project_polygon
        if N <= self._BATCH_POLYGON_THRESHOLD:
            results = []
            for i in range(N):
                # 如果原对象是 Polygon3D, 直接传入
                if poly_objs[i] is not None:
                    result = self.project_polygon(
                        poly_objs[i], T_to_cam=T_to_cam, pts_in_cam=pts_in_cam,
                        cull_frustum=cull_frustum
                    )
                else:
                    # 构造临时 Polygon3D 对象
                    tmp = _Polygon3D(vert_list[i], is_closed=is_closed_list[i])
                    result = self.project_polygon(
                        tmp, T_to_cam=T_to_cam, pts_in_cam=pts_in_cam,
                        cull_frustum=cull_frustum
                    )
                    result['polygon'] = None
                results.append(result)
            return results

        # 大批量向量化路径
        # 1) 收集所有顶点 + 边端点, 一次性变换和投影
        all_verts_flat_list = []
        vert_offsets = []  # 每个多边形顶点在 flat 中的起始位置
        offset = 0
        for v in vert_list:
            vert_offsets.append(offset)
            all_verts_flat_list.append(v)
            offset += len(v)
        all_verts_flat = np.concatenate(all_verts_flat_list, axis=0)  # (sum_M, 3)

        if not pts_in_cam and T_to_cam is not None:
            ones = np.ones((len(all_verts_flat), 1))
            verts_h = np.hstack([all_verts_flat, ones])
            all_verts_cam = (T_to_cam @ verts_h.T).T[:, :3]
        else:
            all_verts_cam = all_verts_flat

        # 2) 批量投影所有顶点
        pixels_flat, valid_flat = self.camera.project(all_verts_cam, pts_in_cam=True)
        pixels_2d_flat = pixels_flat[:, :2]

        # 3) 构造所有边 (按多边形顺序)
        edge_p1s_list = []
        edge_p2s_list = []
        edge_meta = []  # (poly_idx, edge_idx_in_poly, vi, vj)
        for p_idx, v in enumerate(vert_list):
            m = len(v)
            if m == 0:
                continue
            start = vert_offsets[p_idx]
            is_closed = is_closed_list[p_idx]
            n_edges = m if is_closed else m - 1
            for e in range(n_edges):
                i = e
                j = (e + 1) % m if is_closed else e + 1
                edge_p1s_list.append(all_verts_cam[start + i])
                edge_p2s_list.append(all_verts_cam[start + j])
                edge_meta.append((p_idx, e, i, j))

        if len(edge_p1s_list) == 0:
            # 所有多边形为空, 直接返回
            results = []
            for p_idx in range(N):
                start = vert_offsets[p_idx]
                m = len(vert_list[p_idx])
                results.append({
                    'vertices': pixels_2d_flat[start:start + m] if m > 0 else np.empty((0, 2)),
                    'valid': valid_flat[start:start + m] if m > 0 else np.empty(0, dtype=bool),
                    'edges': [],
                    'polygon': poly_objs[p_idx],
                })
            return results

        edge_p1s = np.array(edge_p1s_list, dtype=np.float64)
        edge_p2s = np.array(edge_p2s_list, dtype=np.float64)

        # 4) 批量裁剪
        if cull_frustum and self._culler is not None:
            clipped, edge_valid = self._culler.clip_lines_batch(edge_p1s, edge_p2s)
        else:
            # 无裁剪: 直接连线
            clipped = np.empty((len(edge_p1s), 2, 3), dtype=np.float64)
            clipped[:, 0] = edge_p1s
            clipped[:, 1] = edge_p2s
            edge_valid = np.zeros(len(edge_p1s), dtype=bool)
            for k, (p_idx, e, i, j) in enumerate(edge_meta):
                start = vert_offsets[p_idx]
                edge_valid[k] = valid_flat[start + i] and valid_flat[start + j]

        # 5) 批量投影裁剪端点
        valid_indices = np.where(edge_valid)[0]
        if len(valid_indices) > 0:
            clip_pts = clipped[valid_indices].reshape(-1, 3)
            # CONE 近似版可能产生 NaN, 需过滤后再批量投影
            nan_mask = np.isnan(clip_pts).any(axis=1)
            valid_clip_pts = clip_pts[~nan_mask].reshape(-1, 3)
            if len(valid_clip_pts) > 0:
                valid_clip_pixels = self._project_raw_pixels(valid_clip_pts)
            else:
                valid_clip_pixels = np.empty((0, 2), dtype=np.float64)
            # 建立 clip_pts index 到 valid_clip_pixels index 的映射
            valid_idx_map = np.cumsum(~nan_mask) - 1
        else:
            valid_clip_pixels = np.empty((0, 2), dtype=np.float64)
            valid_idx_map = np.array([], dtype=int)
            nan_mask = np.array([], dtype=bool)

        # 6) 组装每个多边形的 edges
        poly_edges_dict: List[List] = [[] for _ in range(N)]

        for k, edge_k in enumerate(valid_indices):
            p_idx, e, i, j = edge_meta[edge_k]
            # 检查是否为 NaN (CONE 近似版)
            base_clip_idx = k * 2
            if nan_mask[base_clip_idx] or nan_mask[base_clip_idx + 1]:
                # 回退到单条 clip_line 精确处理
                p1_cam = edge_p1s[edge_k]
                p2_cam = edge_p2s[edge_k]
                precise = self._culler.clip_line(p1_cam, p2_cam)
                if precise is None:
                    continue
                p1_c, p2_c = precise
                near_sq = self.camera.near_z ** 2
                r1_sq = p1_c[0]**2 + p1_c[1]**2 + p1_c[2]**2
                r2_sq = p2_c[0]**2 + p2_c[1]**2 + p2_c[2]**2
                if r1_sq < near_sq or r2_sq < near_sq:
                    continue
                precise_px = self._project_raw_pixels(np.array([p1_c, p2_c]))
                px1, px2 = precise_px[0], precise_px[1]
            else:
                # 正常向量化路径
                px1 = valid_clip_pixels[valid_idx_map[base_clip_idx]]
                px2 = valid_clip_pixels[valid_idx_map[base_clip_idx + 1]]
            poly_edges_dict[p_idx].append((px1, px2))

        # 7) 组装最终结果
        results = []
        for p_idx in range(N):
            start = vert_offsets[p_idx]
            m = len(vert_list[p_idx])
            results.append({
                'vertices': pixels_2d_flat[start:start + m] if m > 0 else np.empty((0, 2)),
                'valid': valid_flat[start:start + m] if m > 0 else np.empty(0, dtype=bool),
                'edges': poly_edges_dict[p_idx],
                'polygon': poly_objs[p_idx],
            })

        return results

    def project_point_cloud(
        self,
        cloud: PointCloud,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, 'PointCloud']:
        """
        投影 PointCloud 对象，返回投影结果与过滤后的点云

        使用 FrustumCuller 的扩展 FOV（含 frustum_expansion 系数）
        验证点的有效性，替代 camera.project() 中固定 5% 容差的 _check_fov。

        Args:
            cloud: PointCloud 对象（含 points/colors/intensity）
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 点是否已在相机坐标系

        Returns:
            result: (N, 3+) 投影结果（UV + depth）
            valid: (N,) 有效掩码（基于扩展 FOV 验证）
            filtered_cloud: 按 valid 过滤后的 PointCloud
        """
        result, valid = self.project_points(cloud.points, T_to_cam, pts_in_cam)
        filtered_cloud = cloud.filter_by_mask(valid)
        return result, valid, filtered_cloud

    def __call__(
        self,
        geometry: Union[np.ndarray, PointCloud, Box3D, LineSet, Polygon3D],
        **kwargs
    ):
        """
        便捷方法，自动根据类型调用对应的投影方法

        Args:
            geometry: 支持的几何体类型
            **kwargs: 传递给具体投影方法的参数

        Returns:
            对应几何体的投影结果
        """
        if isinstance(geometry, (np.ndarray, PointCloud)):
            return self.project_points(geometry, **kwargs)
        elif isinstance(geometry, Box3D):
            return self.project_box(geometry, **kwargs)
        elif isinstance(geometry, LineSet):
            return self.project_lines(geometry, **kwargs)
        elif isinstance(geometry, Polygon3D):
            return self.project_polygon(geometry, **kwargs)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")