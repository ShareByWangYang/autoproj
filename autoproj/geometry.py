import numpy as np
from typing import Optional, Union


class PointCloud:
    """
    点云类
    
    Args:
        points: 点云坐标，形状为(N, 3)
        colors: 点云颜色，形状为(N, 3)（可选）
        intensity: 反射强度，形状为(N,)（可选）
    """
    
    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ):
        self.points = np.asarray(points)
        self.colors = np.asarray(colors) if colors is not None else None
        self.intensity = np.asarray(intensity) if intensity is not None else None
        
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        
        if self.colors is not None and self.colors.shape[0] != self.points.shape[0]:
            raise ValueError("colors must have the same number of points")
        
        if self.intensity is not None and self.intensity.shape[0] != self.points.shape[0]:
            raise ValueError("intensity must have the same number of points")
    
    @property
    def num_points(self) -> int:
        """点云数量"""
        return self.points.shape[0]
    
    def transform(self, T: np.ndarray) -> 'PointCloud':
        """
        对点云应用变换
        
        Args:
            T: 4x4变换矩阵
        
        Returns:
            变换后的点云
        """
        points_h = np.hstack([self.points, np.ones((self.num_points, 1))])
        transformed = (T @ points_h.T).T[:, :3]
        
        return PointCloud(transformed, self.colors, self.intensity)
    
    def filter_by_mask(self, mask: np.ndarray) -> 'PointCloud':
        """
        根据掩码过滤点云
        
        Args:
            mask: 布尔掩码，形状为(N,)
        
        Returns:
            过滤后的点云
        """
        mask = np.asarray(mask)
        new_points = self.points[mask]
        new_colors = self.colors[mask] if self.colors is not None else None
        new_intensity = self.intensity[mask] if self.intensity is not None else None
        
        return PointCloud(new_points, new_colors, new_intensity)


class Box3D:
    """
    3D包围盒类
    
    Args:
        center: 中心点坐标 (x, y, z)
        size: 尺寸 (length, width, height)
        yaw: 偏航角（弧度）
    """
    
    def __init__(
        self,
        center: Union[list, np.ndarray],
        size: Union[list, np.ndarray],
        yaw: float = 0.0
    ):
        self.center = np.array(center, dtype=np.float64)
        self.size = np.array(size, dtype=np.float64)
        self.yaw = yaw
        
        if self.center.shape[0] != 3:
            raise ValueError("center must have 3 elements")
        if self.size.shape[0] != 3:
            raise ValueError("size must have 3 elements")
    
    @property
    def corners(self) -> np.ndarray:
        """获取包围盒的8个角点"""
        l, w, h = self.size / 2
        
        # 局部坐标系下的角点
        local_corners = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
        ])
        
        # 旋转矩阵（绕z轴）
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # 变换到世界坐标系
        world_corners = (R @ local_corners.T).T + self.center
        
        return world_corners
    
    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组（中心点+尺寸+角度）"""
        return np.concatenate([self.center, self.size, [self.yaw]])
    
    @classmethod
    def from_numpy(cls, data: np.ndarray) -> 'Box3D':
        """从numpy数组创建包围盒"""
        if len(data) != 7:
            raise ValueError("data must have 7 elements: [x, y, z, l, w, h, yaw]")
        return cls(data[:3], data[3:6], data[6])