import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple


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
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0
    ):
        self.width = width
        self.height = height
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.near_z = near_z
        self.far_z = far_z
    
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
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        return (T_to_cam @ points_h.T).T[:, :3]
    
    def _check_depth_range(self, points_cam: np.ndarray) -> np.ndarray:
        """检查深度范围"""
        z = points_cam[:, 2]
        return (z > self.near_z) & (z < self.far_z)
    
    def _check_bounds(self, u: np.ndarray, v: np.ndarray, margin: int = 20) -> np.ndarray:
        """检查像素边界"""
        return (u >= -margin) & (u < self.width + margin) & \
               (v >= -margin) & (v < self.height + margin)


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
        far_z: float = 1000.0
    ):
        super().__init__(width, height, cx, cy, near_z, far_z)
        self.fx = fx
        self.fy = fy
        
        # 初始化畸变系数
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
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
    
    @property
    def fov_v(self) -> float:
        """垂直视场角（度）"""
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
    
    def _apply_distortion(self, x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用畸变校正"""
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
    
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        points_3d = np.asarray(points_3d)
        
        # 坐标变换
        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(points_3d, T_to_cam)
        else:
            points_cam = points_3d
        
        # 深度检查
        valid = self._check_depth_range(points_cam)
        
        # 归一化
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        x_norm = np.zeros_like(x_c, dtype=np.float64)
        y_norm = np.zeros_like(y_c, dtype=np.float64)
        valid_z = z_c > 1e-10
        x_norm[valid_z] = x_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        y_norm[valid_z] = y_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        
        # 畸变校正
        x_dist, y_dist = self._apply_distortion(x_norm, y_norm)
        
        # 内参投影
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        
        # 边界检查
        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds
        
        # 裁剪到图像范围
        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1
        
        pixels = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
        return pixels, valid


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
        far_z: float = 1000.0
    ):
        super().__init__(width, height, cx, cy, near_z, far_z)
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
    
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        points_3d = np.asarray(points_3d)
        
        # 坐标变换
        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(points_3d, T_to_cam)
        else:
            points_cam = points_3d
        
        # 深度检查
        valid = self._check_depth_range(points_cam)
        
        # 归一化
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        x_norm = np.zeros_like(x_c, dtype=np.float64)
        y_norm = np.zeros_like(y_c, dtype=np.float64)
        valid_z = z_c > 1e-10
        x_norm[valid_z] = x_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        y_norm[valid_z] = y_c[valid_z].astype(np.float64) / z_c[valid_z].astype(np.float64)
        
        # Kannala-Brandt畸变
        r = np.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = np.arctan(r)
        
        theta_d = theta + self.k1 * theta**3 + self.k2 * theta**5 + \
                  self.k3 * theta**7 + self.k4 * theta**9
        
        safe_r = np.maximum(r, 1e-10)
        scale = theta_d / safe_r
        
        x_dist = x_norm * scale
        y_dist = y_norm * scale
        
        # 内参投影
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        
        # 边界检查
        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds
        
        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1
        
        pixels = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
        return pixels, valid


class FThetaCamera(Camera):
    """
    F-Theta等距鱼眼相机模型（多项式映射）
    
    Args:
        width: 图像宽度
        height: 图像高度
        fw_poly: 焦距多项式系数 [a0, a1, a2, ...]
        cx: 主点x坐标
        cy: 主点y坐标
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        fw_poly: Union[list, np.ndarray],
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0
    ):
        super().__init__(width, height, cx, cy, near_z, far_z)
        self.fw_poly = np.array(fw_poly, dtype=np.float64)
    
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
    
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        points_3d = np.asarray(points_3d)
        
        # 坐标变换
        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(points_3d, T_to_cam)
        else:
            points_cam = points_3d
        
        # 深度检查
        valid = self._check_depth_range(points_cam)
        
        # 计算极坐标
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        theta = np.arctan2(np.sqrt(x_c**2 + y_c**2), z_c)
        
        # F-Theta多项式映射
        r = np.zeros_like(theta)
        for i, coeff in enumerate(self.fw_poly):
            r += coeff * (theta ** i)
        
        # 转换到图像坐标
        phi = np.arctan2(y_c, x_c)
        u = r * np.cos(phi) + self.cx
        v = r * np.sin(phi) + self.cy
        
        # 边界检查
        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds
        
        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1
        
        pixels = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
        return pixels, valid