import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

from .backends import BackendSelector


class Camera(ABC):
    """
    相机基类，定义通用投影接口
    
    Attributes:
        width: 图像宽度（像素）
        height: 图像高度（像素）
        cx: 主点x坐标
        cy: 主点y坐标
        near_z: 近裁剪面深度
        far_z: 远裁剪面深度
        margin: 边界检查的像素余量（默认20），可由 Projector.frustum_expansion 动态调整
        backend: 计算后端（NumPy/CUDA）
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        margin: int = 20,
        backend: Optional['Backend'] = None
    ):
        self.width = width
        self.height = height
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.near_z = near_z
        self.far_z = far_z
        # 边界检查的像素余量，可由 Projector.frustum_expansion 动态调整
        self.margin = margin
        self.backend = backend if backend is not None else BackendSelector.select()
        self.np = self.backend.np
        # 子类型标识，由 CameraFactory 创建时设置，用于配置保存时还原类型信息
        self.sub_type: Optional[str] = None
    
    @abstractmethod
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影3D点到图像平面
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            T_to_cam: 外参变换矩阵（4x4），将点云变换到相机坐标系
            pts_in_cam: 点云是否已在相机坐标系中
        
        Returns:
            (像素坐标, 有效掩码)
        """
        pass
    
    def _transform_to_camera(self, points_3d: np.ndarray, T_to_cam: np.ndarray) -> np.ndarray:
        """将点云变换到相机坐标系"""
        np = self.np
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        # 通过 backend.matmul 统一调用，使 C++/CUDA 后端可加速矩阵运算
        transformed = self.backend.matmul(T_to_cam, points_h.T)
        return transformed.T[:, :3]
    
    def _check_depth_range(self, points_cam: np.ndarray) -> np.ndarray:
        """检查深度范围"""
        np = self.np
        z = points_cam[:, 2]
        return (z > self.near_z) & (z < self.far_z)
    
    def _check_fov(self, x_norm: np.ndarray, y_norm: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于未畸变的归一化坐标检查是否在 FOV 内
        
        在畸变之前检查归一化坐标的几何边界，防止畸变将 FOV 外的
        大角度点映射回图像内。这是真正的几何 FOV 检查。
        
        Args:
            x_norm: 未畸变的 x 归一化坐标 (x/z)
            y_norm: 未畸变的 y 归一化坐标 (y/z)
            tolerance: 容差系数（相对），默认5%，补偿镜头畸变可能
                        扩大的有效视场。设为0则使用严格FOV。
        
        Returns:
            (N,) bool 数组，True 表示在 FOV 内
        """
        np = self.np
        # 归一化坐标边界（基于图像尺寸和焦距）
        # 图像右边缘: u=w -> x_norm_max = (w-cx)/fx
        # 图像左边缘: u=0 -> x_norm_min = -cx/fx
        x_max = (self.width - self.cx) / self.fx
        x_min = -self.cx / self.fx
        y_max = (self.height - self.cy) / self.fy
        y_min = -self.cy / self.fy
        
        # 施加容差
        x_max *= (1.0 + tolerance)
        x_min *= (1.0 + tolerance)
        y_max *= (1.0 + tolerance)
        y_min *= (1.0 + tolerance)
        
        return (x_norm >= x_min) & (x_norm <= x_max) & \
               (y_norm >= y_min) & (y_norm <= y_max)
    
    def _check_bounds(self, u: np.ndarray, v: np.ndarray, margin: Optional[int] = None) -> np.ndarray:
        """
        检查像素边界

        Args:
            u: 像素 x 坐标数组
            v: 像素 y 坐标数组
            margin: 边界余量（像素），None 时使用 self.margin
        """
        np = self.np
        if margin is None:
            margin = self.margin
        return (u >= -margin) & (u < self.width + margin) & \
               (v >= -margin) & (v < self.height + margin)

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        通过反解畸变方程，计算图像边界对应的真实入射角。
        子类应覆盖此方法以提供精确的畸变感知 FOV 计算。

        畸变对 FOV 的影响：
        - 桶形畸变（k1<0）：径向压缩 → 真实 FOV > 标称 FOV
        - 枕形畸变（k1>0）：径向拉伸 → 真实 FOV < 标称 FOV

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # 默认：无畸变，使用标称 FOV
        if hasattr(self, 'fov_h'):
            return self.fov_h, self.fov_v
        np = self.np
        fov_h = 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
        fov_v = 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
        return fov_h, fov_v

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        通过比较考虑畸变的真实 FOV 和标称 FOV，
        计算使视锥边界匹配真实可视范围的扩展因子。

        expansion_factor = tan(θ_real_max) / tan(θ_nominal_max)

        桶形畸变（k1<0）: 真实 FOV > 标称 FOV → expansion > 1.0
        枕形畸变（k1>0）: 真实 FOV < 标称 FOV → expansion < 1.0

        Returns:
            扩展因子（1.0 表示无扩展/无畸变）
        """
        # 默认：无畸变，返回 1.0
        return 1.0


class PinholeCamera(Camera):
    """
    针孔相机模型（支持OpenCV 8参数畸变模型）
    
    Args:
        width: 图像宽度
        height: 图像高度
        fx: x方向焦距
        fy: y方向焦距
        cx: 主点x坐标（默认图像中心）
        cy: 主点y坐标（默认图像中心）
        dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, k4, k5, k6]
        near_z: 近裁剪面
        far_z: 远裁剪面
        margin: 边界检查余量（像素，默认20）
        backend: 计算后端（可选）
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dist_coeffs: Optional[Union[list, np.ndarray]] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        margin: int = 20,
        backend: Optional['Backend'] = None
    ):
        super().__init__(width, height, cx, cy, near_z, far_z, margin, backend)
        self.fx = fx
        self.fy = fy
        
        np = self.np
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(8)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
            if len(self.dist_coeffs) < 8:
                padding = np.zeros(8 - len(self.dist_coeffs))
                self.dist_coeffs = np.concatenate([self.dist_coeffs, padding])
        
        self.k1, self.k2, self.p1, self.p2 = self.dist_coeffs[:4]
        self.k3, self.k4, self.k5, self.k6 = self.dist_coeffs[4:]
        self.has_distortion = np.any(self.dist_coeffs != 0)
    
    @classmethod
    def from_dict(cls, config: dict) -> 'PinholeCamera':
        """
        从字典创建针孔相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fx: x方向焦距
                - fy: y方向焦距
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - dist_coeffs: 畸变系数（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
        
        Returns:
            PinholeCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fx=config['fx'],
            fy=config['fy'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            dist_coeffs=config.get('dist_coeffs'),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0)
        )
    
    @property
    def fov_h(self) -> float:
        """水平视场角（度）"""
        np = self.np
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
    
    @property
    def fov_v(self) -> float:
        """垂直视场角（度）"""
        np = self.np
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
    
    def _apply_distortion(self, x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用畸变校正"""
        np = self.np
        if not self.has_distortion:
            return x_norm, y_norm
        
        r2 = x_norm ** 2 + y_norm ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        
        numerator = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        denom = 1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
        denom = np.maximum(denom, 1e-10)
        radial = numerator / denom
        
        x_dist = x_norm * radial
        y_dist = y_norm * radial
        
        x_dist += 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm ** 2)
        y_dist += self.p1 * (r2 + 2 * y_norm ** 2) + 2 * self.p2 * x_norm * y_norm
        
        return x_dist, y_dist

    def _radial_factor(self, r2: float) -> float:
        """计算径向畸变因子（给定 r²）"""
        r4 = r2 * r2
        r6 = r4 * r2
        numerator = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        denom = 1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
        denom = max(denom, 1e-10)
        return numerator / denom

    def _bisect_radial_1d(self, target: float, y_fixed: float = 0.0,
                          max_iter: int = 100, tol: float = 1e-12) -> float:
        """
        二分法求解一维径向畸变方程：x · radial(x, y_fixed) = target

        用于 FOV 计算等纯径向畸变场景，对强畸变（k1>0 或 k1<0）均稳定收敛。

        Args:
            target: 畸变后的归一化坐标值
            y_fixed: 固定的 y 坐标（水平 FOV 计算时为 0）
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            畸变前的归一化坐标 x
        """
        if abs(target) < 1e-12:
            return 0.0

        def f(x):
            r2 = x * x + y_fixed * y_fixed
            return x * self._radial_factor(r2) - target

        # 二分法寻找有根区间
        if target > 0:
            lo, hi = 0.0, max(1.0, target)
            while f(hi) < 0 and hi < 1e10:
                hi *= 2
        else:
            lo, hi = min(-1.0, target), 0.0
            while f(lo) > 0 and lo > -1e10:
                lo *= 2

        # 二分迭代
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            f_mid = f(mid)
            if f_mid > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break

        return (lo + hi) / 2

    def _undistort_normalized(
        self,
        x_dist: float,
        y_dist: float,
        max_iter: int = 50,
        tol: float = 1e-12
    ) -> Tuple[float, float]:
        """
        反解畸变方程：给定畸变后归一化坐标，求畸变前归一化坐标

        策略：先用二分法求解纯径向畸变（稳定收敛），
        再用固定点迭代加入切向畸变修正。

        对于 FOV 计算场景（y_dist=0 或 x_dist=0），
        二分法直接给出精确解，无需后续迭代。

        Args:
            x_dist: 畸变后 x 归一化坐标
            y_dist: 畸变后 y 归一化坐标
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            (x_norm, y_norm): 畸变前归一化坐标
        """
        if not self.has_distortion:
            return x_dist, y_dist

        # 用二分法求解径向部分（作为初值）
        x = self._bisect_radial_1d(x_dist, y_fixed=0.0)
        y = self._bisect_radial_1d(y_dist, y_fixed=0.0)

        # 如果切向畸变系数为零，二分法已经给出精确解
        if abs(self.p1) < 1e-15 and abs(self.p2) < 1e-15:
            # 验证：检查二分法是否同时满足 x 和 y 方程
            # 对 x 方程：x_dist = x * radial(x, y)
            r2_check = x * x + y * y
            radial_check = self._radial_factor(r2_check)
            x_residual = x * radial_check - x_dist
            y_residual = y * radial_check - y_dist

            # 如果残差较大，说明需要联立求解
            if abs(x_residual) > 1e-10 or abs(y_residual) > 1e-10:
                # 用固定点迭代精炼（对小残差有效）
                for _ in range(max_iter):
                    r2 = x * x + y * y
                    radial = self._radial_factor(r2)
                    x_new = x_dist / radial
                    y_new = y_dist / radial
                    if abs(x_new - x) < tol and abs(y_new - y) < tol:
                        x, y = x_new, y_new
                        break
                    x, y = x_new, y_new
            return x, y

        # 有切向畸变：用固定点迭代精炼（对强畸变可能不收敛，
        # 但二分法已给出较好初值，通常能在几步内收敛）
        for _ in range(max_iter):
            r2 = x * x + y * y
            radial = self._radial_factor(r2)

            x_new = (x_dist - 2 * self.p1 * x * y - self.p2 * (r2 + 2 * x * x)) / radial
            y_new = (y_dist - self.p1 * (r2 + 2 * y * y) - 2 * self.p2 * x * y) / radial

            if abs(x_new - x) < tol and abs(y_new - y) < tol:
                x, y = x_new, y_new
                break

            x, y = x_new, y_new

        return x, y

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        通过反解畸变方程，计算图像边界对应的真实入射角。

        图像边界对应的畸变后归一化坐标：
            x_dist_right = (width - 1 - cx) / fx
            x_dist_left  = -cx / fx
            y_dist_top   = (height - 1 - cy) / fy
            y_dist_bottom= -cy / fy

        反解得到畸变前归一化坐标后，计算入射角：
            θ = arctan(x_norm)

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        if not self.has_distortion:
            return self.fov_h, self.fov_v

        # 图像边界对应的畸变后归一化坐标
        x_dist_right = (self.width - 1 - self.cx) / self.fx
        x_dist_left = -self.cx / self.fx
        y_dist_top = (self.height - 1 - self.cy) / self.fy
        y_dist_bottom = -self.cy / self.fy

        # 反解得到畸变前归一化坐标
        # 水平 FOV：在 y=0 平面计算
        x_norm_right, _ = self._undistort_normalized(x_dist_right, 0.0)
        x_norm_left, _ = self._undistort_normalized(x_dist_left, 0.0)

        # 垂直 FOV：在 x=0 平面计算
        _, y_norm_top = self._undistort_normalized(0.0, y_dist_top)
        _, y_norm_bottom = self._undistort_normalized(0.0, y_dist_bottom)

        # 真实 FOV（度）= arctan(right) + arctan(|left|)
        fov_h_real = np.degrees(np.arctan(x_norm_right) + np.arctan(abs(x_norm_left)))
        fov_v_real = np.degrees(np.arctan(y_norm_top) + np.arctan(abs(y_norm_bottom)))

        return fov_h_real, fov_v_real

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        比较考虑畸变的真实 FOV 和标称 FOV：
            expansion = max(|x_norm_real_max|, |y_norm_real_max|) /
                        max(tan_h_nominal, tan_v_nominal)

        其中：
            tan_h_nominal = max(cx, width-1-cx) / fx
            tan_v_nominal = max(cy, height-1-cy) / fy
            x_norm_real 通过反解畸变方程获得

        桶形畸变（k1<0）: expansion > 1.0
        枕形畸变（k1>0）: expansion < 1.0

        Returns:
            扩展因子
        """
        if not self.has_distortion:
            return 1.0

        # 标称 tan 值（取较大的半边）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy

        # 图像边界对应的畸变后归一化坐标
        x_dist_right = (self.width - 1 - self.cx) / self.fx
        x_dist_left = -self.cx / self.fx
        y_dist_top = (self.height - 1 - self.cy) / self.fy
        y_dist_bottom = -self.cy / self.fy

        # 反解得到畸变前归一化坐标
        x_norm_right, _ = self._undistort_normalized(x_dist_right, 0.0)
        x_norm_left, _ = self._undistort_normalized(x_dist_left, 0.0)
        _, y_norm_top = self._undistort_normalized(0.0, y_dist_top)
        _, y_norm_bottom = self._undistort_normalized(0.0, y_dist_bottom)

        # 真实最大 |tan| 值
        tan_h_real = max(abs(x_norm_right), abs(x_norm_left))
        tan_v_real = max(abs(y_norm_top), abs(y_norm_bottom))

        # 扩展因子：取水平和垂直中较大的（保守策略）
        expansion_h = tan_h_real / tan_h_nominal if tan_h_nominal > 1e-10 else 1.0
        expansion_v = tan_v_real / tan_v_nominal if tan_v_nominal > 1e-10 else 1.0

        return max(expansion_h, expansion_v)
    
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        np = self.np
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        # 提取XYZ进行计算
        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        if hasattr(self.backend, 'project_pinhole') and callable(getattr(self.backend, 'project_pinhole')):
            pixels, valid = self.backend.project_pinhole(
                points_cam, self.fx, self.fy, self.cx, self.cy,
                self.dist_coeffs, self.width, self.height,
                self.near_z, self.far_z
            )

            # 额外：基于未畸变归一化坐标的 FOV 几何检查
            # 防止畸变将 FOV 外的大角度点映射回图像内
            z_c = points_cam[:, 2]
            valid_z = z_c > 1e-10
            x_norm = np.zeros(n_points, dtype=np.float64)
            y_norm = np.zeros(n_points, dtype=np.float64)
            x_norm[valid_z] = points_cam[valid_z, 0] / z_c[valid_z]
            y_norm[valid_z] = points_cam[valid_z, 1] / z_c[valid_z]
            fov_valid = self._check_fov(x_norm, y_norm, tolerance=0.05)
            valid = valid & fov_valid

            # 对有效点 UV 做 clamp 到 [0, w-1]，无效点置 -1
            u_raw = pixels[:, 0].astype(np.float64)
            v_raw = pixels[:, 1].astype(np.float64)
            u_clip = np.clip(u_raw, 0, self.width - 1)
            v_clip = np.clip(v_raw, 0, self.height - 1)
            u_clip[~valid] = -1
            v_clip[~valid] = -1

            if not preserve_extra:
                # 向后兼容：返回 (N, 2) int32
                result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
                return result, valid

            # preserve_extra=True: 返回 (N, 3+) float64
            result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
            result[:, 0] = u_clip.astype(np.int32)
            result[:, 1] = v_clip.astype(np.int32)
            result[:, 2] = points_cam[:, 2]
            if original_shape[1] > 3:
                result[:, 3:] = points_3d[:, 3:]
            return result, valid

        # NumPy 路径
        valid = self._check_depth_range(points_cam)

        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        x_norm = np.zeros_like(x_c, dtype=np.float64)
        y_norm = np.zeros_like(y_c, dtype=np.float64)
        valid_z = z_c > 1e-10
        x_norm[valid_z] = x_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        y_norm[valid_z] = y_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)

        # 基于未畸变归一化坐标的 FOV 几何检查
        # 在畸变之前检查，防止畸变将 FOV 外的点映射回图像内
        fov_valid = self._check_fov(x_norm, y_norm, tolerance=0.05)
        valid = valid & fov_valid

        x_dist, y_dist = self._apply_distortion(x_norm, y_norm)

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        # 对有效点 UV 做 clamp 到 [0, w-1]，无效点置 -1
        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            # 向后兼容：返回 (N, 2) int32
            result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
            return result, valid

        # preserve_extra=True: 返回 (N, 3+) float64
        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        result[:, 0] = u_clip.astype(np.int32)
        result[:, 1] = v_clip.astype(np.int32)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid


class KannalaBrandtCamera(Camera):
    """
    Kannala-Brandt鱼眼相机模型（θ多项式畸变）
    
    Args:
        width: 图像宽度
        height: 图像高度
        fx: x方向焦距
        fy: y方向焦距
        cx: 主点x坐标
        cy: 主点y坐标
        k1, k2, k3, k4: 畸变系数（θ^3, θ^5, θ^7, θ^9系数）
        near_z: 近裁剪面
        far_z: 远裁剪面
        margin: 边界检查余量（像素，默认20）
        backend: 计算后端（可选）
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        margin: int = 20,
        backend: Optional['Backend'] = None
    ):
        super().__init__(width, height, cx, cy, near_z, far_z, margin, backend)
        self.fx = fx
        self.fy = fy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
    
    @classmethod
    def from_dict(cls, config: dict) -> 'KannalaBrandtCamera':
        """
        从字典创建Kannala-Brandt鱼眼相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fx: x方向焦距
                - fy: y方向焦距
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - k1, k2, k3, k4: 畸变系数（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
        
        Returns:
            KannalaBrandtCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fx=config['fx'],
            fy=config['fy'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            k1=config.get('k1', 0.0),
            k2=config.get('k2', 0.0),
            k3=config.get('k3', 0.0),
            k4=config.get('k4', 0.0),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0)
        )

    @property
    def fov_h(self) -> float:
        """水平视场角（度）— 标称值，基于针孔模型近似"""
        np = self.np
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))

    @property
    def fov_v(self) -> float:
        """垂直视场角（度）— 标称值，基于针孔模型近似"""
        np = self.np
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))

    def _undistort_theta(self, theta_d: float, max_iter: int = 100, tol: float = 1e-12) -> float:
        """
        反解 Kannala-Brandt θ 多项式：给定畸变后角度 θ_d，求真实入射角 θ

        畸变模型：θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹

        使用二分法反解（θ 单调递增映射到 θ_d）。

        Args:
            theta_d: 畸变后角度（弧度）
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            θ: 真实入射角（弧度）
        """
        # 无畸变时直接返回
        if self.k1 == 0 and self.k2 == 0 and self.k3 == 0 and self.k4 == 0:
            return theta_d

        # 二分法：θ ∈ [0, π]
        lo, hi = 0.0, np.pi
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            theta_d_pred = mid + self.k1 * mid**3 + self.k2 * mid**5 + \
                           self.k3 * mid**7 + self.k4 * mid**9
            if theta_d_pred < theta_d:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break

        return (lo + hi) / 2

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        鱼眼畸变模型：θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
        图像半径：r = fx · θ_d

        图像边界对应的最大半径 r_max，反解得到真实入射角 θ_max。

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # 图像边界对应的最大半径
        r_max_h = max(self.cx, self.width - 1 - self.cx)
        r_max_v = max(self.cy, self.height - 1 - self.cy)

        # 畸变后角度
        theta_d_h = r_max_h / self.fx
        theta_d_v = r_max_v / self.fy

        # 反解得到真实入射角
        theta_real_h = self._undistort_theta(theta_d_h)
        theta_real_v = self._undistort_theta(theta_d_v)

        # 转为度数（FOV = 2 × θ_max）
        fov_h_real = 2 * np.degrees(theta_real_h)
        fov_v_real = 2 * np.degrees(theta_real_v)

        return fov_h_real, fov_v_real

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        对于圆锥视锥（鱼眼），expansion_factor = θ_real / θ_nominal

        其中：
            θ_nominal = arctan(max(cx, w-1-cx) / fx)  （针孔模型近似）
            θ_real 通过反解 θ 多项式获得

        Returns:
            扩展因子
        """
        # 标称半角（针孔模型近似）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
        theta_nominal_h = np.arctan(tan_h_nominal)
        theta_nominal_v = np.arctan(tan_v_nominal)

        # 真实半角
        fov_h_real, fov_v_real = self.compute_real_fov()
        theta_real_h = np.radians(fov_h_real / 2)
        theta_real_v = np.radians(fov_v_real / 2)

        # 扩展因子（角度比）
        expansion_h = theta_real_h / theta_nominal_h if theta_nominal_h > 1e-10 else 1.0
        expansion_v = theta_real_v / theta_nominal_v if theta_nominal_v > 1e-10 else 1.0

        return max(expansion_h, expansion_v)

    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        np = self.np
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        if hasattr(self.backend, 'project_kannala_brandt') and callable(getattr(self.backend, 'project_kannala_brandt')):
            pixels, valid = self.backend.project_kannala_brandt(
                points_cam, self.fx, self.fy, self.cx, self.cy,
                self.k1, self.k2, self.k3, self.k4,
                self.width, self.height, self.near_z, self.far_z
            )

            # 基于未畸变归一化坐标的 FOV 几何检查
            z_c = points_cam[:, 2]
            valid_z = z_c > 1e-10
            x_norm = np.zeros(n_points, dtype=np.float64)
            y_norm = np.zeros(n_points, dtype=np.float64)
            x_norm[valid_z] = points_cam[valid_z, 0] / z_c[valid_z]
            y_norm[valid_z] = points_cam[valid_z, 1] / z_c[valid_z]
            fov_valid = self._check_fov(x_norm, y_norm, tolerance=0.05)
            valid = valid & fov_valid

            u_raw = pixels[:, 0].astype(np.float64)
            v_raw = pixels[:, 1].astype(np.float64)
            u_clip = np.clip(u_raw, 0, self.width - 1)
            v_clip = np.clip(v_raw, 0, self.height - 1)
            u_clip[~valid] = -1
            v_clip[~valid] = -1

            if not preserve_extra:
                result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
                return result, valid

            result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
            result[:, 0] = u_clip.astype(np.int32)
            result[:, 1] = v_clip.astype(np.int32)
            result[:, 2] = points_cam[:, 2]
            if original_shape[1] > 3:
                result[:, 3:] = points_3d[:, 3:]
            return result, valid

        valid = self._check_depth_range(points_cam)

        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        x_norm = np.zeros_like(x_c, dtype=np.float64)
        y_norm = np.zeros_like(y_c, dtype=np.float64)
        valid_z = z_c > 1e-10
        x_norm[valid_z] = x_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        y_norm[valid_z] = y_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)

        # 基于未畸变归一化坐标的 FOV 检查
        fov_valid = self._check_fov(x_norm, y_norm, tolerance=0.05)
        valid = valid & fov_valid

        r = np.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = np.arctan(r)

        theta_d = theta + self.k1 * theta**3 + self.k2 * theta**5 + \
                  self.k3 * theta**7 + self.k4 * theta**9

        safe_r = np.maximum(r, 1e-10)
        scale = theta_d / safe_r

        x_dist = x_norm * scale
        y_dist = y_norm * scale

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
            return result, valid

        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        result[:, 0] = u_clip.astype(np.int32)
        result[:, 1] = v_clip.astype(np.int32)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid


class FThetaCamera(Camera):
    """
    F-Theta等距鱼眼相机模型（多项式映射）
    
    Args:
        width: 图像宽度
        height: 图像高度
        fw_poly: 焦距多项式系数 [a0, a1, a2, ...]
        cx: 主点x坐标
        cy: 主点y坐标
        near_z: 近裁剪面
        far_z: 远裁剪面
        margin: 边界检查余量（像素，默认20）
        backend: 计算后端（可选）
    """

    def __init__(
        self,
        width: int,
        height: int,
        fw_poly: Union[list, np.ndarray],
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        margin: int = 20,
        backend: Optional['Backend'] = None
    ):
        super().__init__(width, height, cx, cy, near_z, far_z, margin, backend)
        np = self.np
        self.fw_poly = np.array(fw_poly, dtype=np.float64)
        self._theta_max = self._compute_theta_max()
    
    def _compute_theta_max(self) -> float:
        """
        数值求解 FOV 最大角度 θ_max，使得 fw_poly(θ_max) = r_max
        
        使用二分法在 [0, π/2] 范围内搜索。
        """
        np = self.np
        r_max = max(self.cx, self.width - self.cx, self.cy, self.height - self.cy)
        
        def eval_poly(theta):
            result = 0.0
            for i, coeff in enumerate(self.fw_poly):
                result += coeff * (theta ** i)
            return result
        
        lo, hi = 0.0, np.pi / 2
        for _ in range(50):  # 二分50次，精度足够
            mid = (lo + hi) / 2
            if eval_poly(mid) < r_max:
                lo = mid
            else:
                hi = mid
        return lo
    
    def _check_fov_theta(self, theta: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于角度 θ 检查是否在 FOV 内（F-Theta 模型）
        
        Args:
            theta: 入射角数组 (rad)
            tolerance: 容差系数（相对），默认5%
        
        Returns:
            (N,) bool 数组
        """
        np = self.np
        theta_max = self._theta_max * (1.0 + tolerance)
        return theta <= theta_max
    
    def _check_fov_ftheta(self, x_norm: np.ndarray, y_norm: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于未畸变归一化坐标的 FOV 检查（F-Theta 模型）
        
        将归一化坐标转换为角度后与 θ_max 比较。
        
        Args:
            x_norm: 未畸变 x 归一化坐标
            y_norm: 未畸变 y 归一化坐标
            tolerance: 容差系数
        """
        np = self.np
        r_norm = np.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = np.arctan(r_norm)
        return self._check_fov_theta(theta, tolerance)
    
    @classmethod
    def from_dict(cls, config: dict) -> 'FThetaCamera':
        """
        从字典创建F-Theta鱼眼相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fw_poly: 焦距多项式系数
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
        
        Returns:
            FThetaCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fw_poly=config['fw_poly'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0)
        )

    @property
    def fx(self) -> float:
        """等效 x 焦距（取多项式一阶系数）"""
        return self.fw_poly[1] if len(self.fw_poly) > 1 else 1.0

    @property
    def fy(self) -> float:
        """等效 y 焦距（同 fx）"""
        return self.fx

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        F-Theta 模型：r = Σ coeff_i · θ^i
        _theta_max 已通过二分法求解 r(θ) = r_max 获得，即为真实半 FOV。

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # _theta_max 已是真实入射角（通过二分法求解多项式方程获得）
        fov_h_real = 2 * np.degrees(self._theta_max)
        fov_v_real = fov_h_real  # F-Theta 是旋转对称模型

        return fov_h_real, fov_v_real

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        对于圆锥视锥，expansion_factor = θ_real / θ_nominal

        其中：
            θ_nominal = arctan(max(cx, w-1-cx) / fx)  （针孔模型近似）
            θ_real = _theta_max（通过多项式反解获得）

        Returns:
            扩展因子
        """
        # 标称半角（针孔模型近似，使用等效焦距）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
        theta_nominal = max(np.arctan(tan_h_nominal), np.arctan(tan_v_nominal))

        if theta_nominal < 1e-10:
            return 1.0

        # 真实半角
        theta_real = self._theta_max

        return theta_real / theta_nominal

    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        np = self.np
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        if hasattr(self.backend, 'project_ftheta') and callable(getattr(self.backend, 'project_ftheta')):
            pixels, valid = self.backend.project_ftheta(
                points_cam, self.fw_poly, self.cx, self.cy,
                self.width, self.height, self.near_z, self.far_z
            )

            # 基于未畸变归一化坐标的 FOV 几何检查
            z_c = points_cam[:, 2]
            valid_z = z_c > 1e-10
            x_norm = np.zeros(n_points, dtype=np.float64)
            y_norm = np.zeros(n_points, dtype=np.float64)
            x_norm[valid_z] = points_cam[valid_z, 0] / z_c[valid_z]
            y_norm[valid_z] = points_cam[valid_z, 1] / z_c[valid_z]
            fov_valid = self._check_fov_ftheta(x_norm, y_norm, tolerance=0.05)
            valid = valid & fov_valid

            u_raw = pixels[:, 0].astype(np.float64)
            v_raw = pixels[:, 1].astype(np.float64)
            u_clip = np.clip(u_raw, 0, self.width - 1)
            v_clip = np.clip(v_raw, 0, self.height - 1)
            u_clip[~valid] = -1
            v_clip[~valid] = -1

            if not preserve_extra:
                result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
                return result, valid

            result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
            result[:, 0] = u_clip.astype(np.int32)
            result[:, 1] = v_clip.astype(np.int32)
            result[:, 2] = points_cam[:, 2]
            if original_shape[1] > 3:
                result[:, 3:] = points_3d[:, 3:]
            return result, valid

        valid = self._check_depth_range(points_cam)

        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        theta = np.arctan2(np.sqrt(x_c**2 + y_c**2), z_c)

        # 基于角度的 FOV 检查（F-Theta 模型）
        fov_valid = self._check_fov_theta(theta, tolerance=0.05)
        valid = valid & fov_valid

        r = np.zeros_like(theta)
        for i, coeff in enumerate(self.fw_poly):
            r += coeff * (theta ** i)

        phi = np.arctan2(y_c, x_c)
        u = r * np.cos(phi) + self.cx
        v = r * np.sin(phi) + self.cy

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            result = np.stack([u_clip.astype(np.int32), v_clip.astype(np.int32)], axis=1)
            return result, valid

        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        result[:, 0] = u_clip.astype(np.int32)
        result[:, 1] = v_clip.astype(np.int32)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid
