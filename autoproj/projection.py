import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from .camera import Camera
from .geometry import PointCloud, Box3D, LineSet, Polygon3D
from .frustum import FrustumCuller, FrustumType


class Projector:
    """
    统一 3D→2D 投影接口，支持视锥裁剪

    提供点云、线段、包围盒、多边形等多种几何体的投影能力，
    并可选在相机坐标系下对棱和多边形进行精确视锥裁剪
    （基于 Liang-Barsky / Sutherland-Hodgman 算法），
    避免 OpenCV 畸变模型将 FOV 外点"回折"到图像内的问题。

    Args:
        camera: 相机对象
        cull_frustum: 是否启用视锥裁剪（默认 True）
        frustum_expansion: 视锥扩展系数
            - float: 固定扩展系数（1.0 表示不扩展）
            - 'auto': 根据相机内参自动计算

    向后兼容：
        Projector(camera, 2.0) 等价于 Projector(camera, frustum_expansion=2.0)
    """

    def __init__(
        self,
        camera: Camera,
        cull_frustum: Union[bool, float] = True,
        frustum_expansion: Union[float, str] = 'auto'
    ):
        self.camera = camera

        # 向后兼容：第二个位置参数可以是 cull_frustum(bool) 或 frustum_expansion(float)
        if isinstance(cull_frustum, (int, float)) and not isinstance(cull_frustum, bool):
            frustum_expansion = float(cull_frustum)
            cull_frustum = True
        elif isinstance(cull_frustum, bool):
            pass  # 正常处理
        elif cull_frustum is None:
            cull_frustum = True

        self._cull_frustum = cull_frustum
        self.frustum_expansion = frustum_expansion

        # 计算有效扩展系数并创建裁剪器
        self._effective_expansion = self._compute_effective_expansion(frustum_expansion)

        if cull_frustum:
            self._culler = FrustumCuller.from_camera(camera, self._effective_expansion)
        else:
            self._culler = None

    def _compute_effective_expansion(self, expansion: Union[float, str]) -> float:
        """
        计算有效视锥扩展系数

        当 expansion='auto' 时，调用 camera.compute_expansion_factor()
        基于相机内参和畸变系数精确计算真实 FOV 扩展范围。
        """
        if isinstance(expansion, str) and expansion.lower() == 'auto':
            # 使用相机精确计算的扩展因子（基于内参和畸变系数反解）
            if hasattr(self.camera, 'compute_expansion_factor'):
                return self.camera.compute_expansion_factor()
            # 回退：无畸变信息时使用 1.0（不扩展）
            return 1.0
        elif isinstance(expansion, (int, float)):
            return float(expansion)
        else:
            raise ValueError(f"Invalid frustum_expansion: {expansion}")

    @property
    def cull_frustum(self) -> bool:
        """是否启用视锥裁剪"""
        return self._cull_frustum

    @property
    def culler(self) -> Optional[FrustumCuller]:
        """视锥裁剪器（可直接用于独立裁剪操作）"""
        return self._culler

    def set_frustum_expansion(self, expansion: Union[float, str]) -> None:
        """
        动态更新视锥扩展系数

        Args:
            expansion: 新的扩展系数或 'auto'
        """
        self.frustum_expansion = expansion
        self._effective_expansion = self._compute_effective_expansion(expansion)
        if self._cull_frustum:
            self._culler = FrustumCuller.from_camera(self.camera, self._effective_expansion)

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

        Args:
            points_cam: (N, 3) 相机坐标系下的 3D 点

        Returns:
            pixels: (N, 2) float64，原始 (u, v) 像素坐标（未 clamp）
        """
        np = self.np if hasattr(self, 'np') else __import__('numpy')
        cam = self.camera
        pts = np.asarray(points_cam, dtype=np.float64)
        n = pts.shape[0]
        x_c, y_c, z_c = pts[:, 0], pts[:, 1], pts[:, 2]

        # 防止除零（clip_line 已保证 z > near_z > 0，这里做保护）
        safe_z = np.maximum(z_c, 1e-10)
        x_norm = x_c / safe_z
        y_norm = y_c / safe_z

        cname = cam.__class__.__name__

        if cname == 'PinholeCamera':
            # --- Pinhole (OpenCV rational polynomial) ---
            if cam.has_distortion:
                r2 = x_norm ** 2 + y_norm ** 2
                r4 = r2 ** 2
                r6 = r2 ** 3
                numerator = 1 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6
                denom = 1 + cam.k4 * r2 + cam.k5 * r4 + cam.k6 * r6
                denom = np.maximum(denom, 1e-10)
                radial = numerator / denom
                x_dist = x_norm * radial
                y_dist = y_norm * radial
                x_dist += 2 * cam.p1 * x_norm * y_norm + cam.p2 * (r2 + 2 * x_norm ** 2)
                y_dist += cam.p1 * (r2 + 2 * y_norm ** 2) + 2 * cam.p2 * x_norm * y_norm
            else:
                x_dist, y_dist = x_norm, y_norm
            u = cam.fx * x_dist + cam.cx
            v = cam.fy * y_dist + cam.cy

        elif cname == 'KannalaBrandtCamera':
            # --- Kannala-Brandt fisheye (theta polynomial) ---
            r = np.sqrt(x_norm ** 2 + y_norm ** 2)
            theta = np.arctan(r)
            theta_d = theta + cam.k1 * theta**3 + cam.k2 * theta**5 + \
                      cam.k3 * theta**7 + cam.k4 * theta**9
            safe_r = np.maximum(r, 1e-10)
            scale = theta_d / safe_r
            x_dist = x_norm * scale
            y_dist = y_norm * scale
            u = cam.fx * x_dist + cam.cx
            v = cam.fy * y_dist + cam.cy

        elif cname == 'FThetaCamera':
            # --- F-Theta fisheye (polynomial: r = sum(coeff[i] * theta^i)) ---
            theta = np.arctan2(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2), safe_z)
            r_dist = np.zeros_like(theta)
            for i, coeff in enumerate(cam.fw_poly):
                r_dist += coeff * (theta ** i)
            phi = np.arctan2(pts[:, 1], pts[:, 0])
            u = r_dist * np.cos(phi) + cam.cx
            v = r_dist * np.sin(phi) + cam.cy

        else:
            # Fallback: 针孔模型无畸变
            u = cam.fx * x_norm + cam.cx
            v = cam.fy * y_norm + cam.cy

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
        （含 frustum_expansion 系数）验证点的有效性，
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

        # 获取相机坐标系下的点
        if not pts_in_cam and T_to_cam is not None:
            points_h = np.hstack([points[:, :3], np.ones((len(points), 1))])
            points_cam = (T_to_cam @ points_h.T).T[:, :3]
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
            # 对 culler 判定有效的点，若相机投影结果为 -1，说明该点因畸变
            # 被相机内部过滤了。需要保留 culler 的判断，将像素坐标
            # 重新计算（使用 _project_raw_pixels 的原始像素计算）
            cam_invalid = (result[:, 0] < 0) & (result[:, 1] < 0)
            # 对于 culler 有效但相机无效的点，重新计算像素坐标
            culler_valid_cam_invalid = valid & cam_invalid
            if np.any(culler_valid_cam_invalid):
                # 修复：对 culler 判定有效的点使用扩展 FOV 内的原始像素计算
                # 这些点在相机固定 FOV 检查中被过滤，但应在扩展 FOV 内
                fix_indices = np.where(culler_valid_cam_invalid)[0]
                fix_points = points_cam[fix_indices]
                raw_pixels = self._project_raw_pixels(fix_points)

                # 根据 preserve_extra 设置结果
                if preserve_extra:
                    # float64 模式：直接覆盖坐标
                    result[fix_indices, 0] = raw_pixels[:, 0]
                    result[fix_indices, 1] = raw_pixels[:, 1]
                else:
                    # int32 模式：clamp 到 [0, w-1] 并转 int32
                    u_clip = np.clip(raw_pixels[:, 0], 0, self.camera.width - 1)
                    v_clip = np.clip(raw_pixels[:, 1], 0, self.camera.height - 1)
                    result[fix_indices, 0] = u_clip.astype(np.int32)
                    result[fix_indices, 1] = v_clip.astype(np.int32)

            # 最终有效性：完全基于 culler 的扩展 FOV 判断
            # 不再受 camera.project 内部固定 5% 容差影响
            valid = valid.copy()  # 确保可写
            # 保留 cam_valid=True 且 culler_valid=True 的点
            # 以及刚修复的 culler_valid=True 的点
            # 即：valid 保持为 culler 的判断结果

        return result, valid

    def project_box(
        self,
        box_input: Union[np.ndarray, Box3D],
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        cull_frustum: Optional[bool] = None
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

        Returns:
            字典包含：
                'corners': (8, 2+) 角点投影像素
                'valid': (8,) 角点有效掩码
                'edges': List[(pixel0, pixel1)] 裁剪后的棱线像素对
                'box': 原始 Box3D 对象或 None
        """
        if isinstance(box_input, Box3D):
            box_obj = box_input
            corners = box_input.get_corners()
        else:
            corners = np.asarray(box_input)
            box_obj = None

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

        # 对12条棱逐条裁剪
        edges = []
        # 收集所有裁剪后的端点（相机坐标），用于生成完整截断多边形
        if cull_frustum and self._culler is not None:
            # 预先收集所有棱的裁剪结果，便于后续处理
            clipped_edges_cam = []  # List[(p1_clipped_cam, p2_clipped_cam)]
            for i, j in LineSet.BOX_EDGES:
                clipped = self._culler.clip_line(corners_cam[i], corners_cam[j])
                if clipped is not None:
                    p1_clipped, p2_clipped = clipped
                    # FrustumCuller 已保证 z >= near_z 且几何上在视锥内
                    # 只需额外保护：剔除 z<=0 的异常情况
                    if p1_clipped[2] > 1e-6 and p2_clipped[2] > 1e-6:
                        clipped_edges_cam.append((p1_clipped, p2_clipped))

            # 批量投影所有裁剪后的端点到像素坐标（原始，未 clamp）
            if clipped_edges_cam:
                all_pts = np.vstack([
                    np.array([p1, p2]) for p1, p2 in clipped_edges_cam
                ])
                all_pixels_raw = self._project_raw_pixels(all_pts)
                # 解包回每条边的两个端点
                idx = 0
                for p1_cam, p2_cam in clipped_edges_cam:
                    p1_px = all_pixels_raw[idx]
                    p2_px = all_pixels_raw[idx + 1]
                    idx += 2
                    # 使用 float64 原始坐标，允许负值和超过图像尺寸的值
                    # （表示截断点在图像边界上/外，绘制时 OpenCV 会自动裁剪）
                    edges.append((p1_px, p2_px))
        else:
            # 无裁剪：直接连接有效角点
            for i, j in LineSet.BOX_EDGES:
                if valid[i] and valid[j]:
                    edges.append((pixels[i], pixels[j]))

        return {
            'corners': pixels,
            'valid': valid,
            'edges': edges,
            'box': box_obj
        }

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
        # 统一输入格式
        if isinstance(lines_input, LineSet):
            segments = [lines_input.get_segment(i) for i in range(lines_input.num_segments)]
        elif isinstance(lines_input, list):
            segments = lines_input
        else:
            # ndarray (N, 2, 3)
            lines_arr = np.asarray(lines_input)
            segments = [(lines_arr[i, 0], lines_arr[i, 1]) for i in range(lines_arr.shape[0])]

        # 确定裁剪行为
        if cull_frustum is None:
            cull_frustum = self._cull_frustum

        result = []

        for p1, p2 in segments:
            p1 = np.asarray(p1, dtype=np.float64)
            p2 = np.asarray(p2, dtype=np.float64)

            # 变换到相机坐标系
            if not pts_in_cam and T_to_cam is not None:
                p1_h = np.hstack([p1, np.ones(1)])
                p2_h = np.hstack([p2, np.ones(1)])
                p1_cam = (T_to_cam @ p1_h)[:3]
                p2_cam = (T_to_cam @ p2_h)[:3]
            else:
                p1_cam = p1
                p2_cam = p2

            # 视锥裁剪
            if cull_frustum and self._culler is not None:
                clipped = self._culler.clip_line(p1_cam, p2_cam)
                if clipped is None:
                    result.append(None)
                    continue
                p1_cam, p2_cam = clipped
                # 保护 z>0
                if p1_cam[2] <= 1e-6 or p2_cam[2] <= 1e-6:
                    result.append(None)
                    continue
                # 使用原始投影（不做边界检查）
                pixels_raw = self._project_raw_pixels(
                    np.array([p1_cam, p2_cam])
                )
                result.append((pixels_raw[0], pixels_raw[1]))
            else:
                # 无裁剪：使用标准投影（保持原有行为）
                pixels, valid = self.camera.project(
                    np.array([p1_cam, p2_cam]),
                    pts_in_cam=True
                )
                if valid[0] and valid[1]:
                    result.append((pixels[0], pixels[1]))
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