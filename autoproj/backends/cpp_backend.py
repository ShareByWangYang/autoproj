from .base import Backend
from typing import Optional
import numpy as np

try:
    from .. import _projection_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class CPythonBackend(Backend):
    """
    CPython后端实现
    
    基于C++扩展的高性能实现，提供比纯Python高10-15倍的性能
    """
    
    def __init__(self):
        if CPP_AVAILABLE:
            self._cpp = _projection_cpp
        self.np = np
    
    def name(self) -> str:
        return 'cpp'
    
    def is_available(self) -> bool:
        if not CPP_AVAILABLE:
            return False
        try:
            test_points = np.array([[1.0, 0.0, 10.0]], dtype=np.float64)
            test_dist = np.zeros(8, dtype=np.float64)
            self._cpp.project_pinhole(test_points, 1000.0, 1000.0, 960.0, 540.0, test_dist, 1920, 1080, 0.1, 1000.0)
            return True
        except Exception:
            return False
    
    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)
    
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)
    
    def arctan(self, x: np.ndarray) -> np.ndarray:
        return np.arctan(x)
    
    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.arctan2(y, x)
    
    def cos(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)
    
    def sin(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)
    
    def maximum(self, x: np.ndarray, y: float) -> np.ndarray:
        return np.maximum(x, y)
    
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(x, min_val, max_val)
    
    def zeros_like(self, x: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return np.zeros_like(x)
        return np.zeros_like(x, dtype=dtype)
    
    def ones(self, shape: tuple, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return np.ones(shape)
        return np.ones(shape, dtype=dtype)
    
    def hstack(self, arrays: list) -> np.ndarray:
        return np.hstack(arrays)
    
    def astype(self, x: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return x.astype(dtype)
    
    def asarray(self, x, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return np.asarray(x, dtype=dtype)
    
    def project_pinhole(
        self,
        points_3d: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        dist_coeffs: np.ndarray,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """
        使用C++扩展进行针孔相机投影
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            fx, fy: 焦距
            cx, cy: 主点坐标
            dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, k4, k5, k6]
            width, height: 图像尺寸
            near_z, far_z: 深度范围
        
        Returns:
            (像素坐标, 有效掩码)
        """
        if not self.is_available():
            raise RuntimeError("CPython backend is not available")
        
        points_3d = np.asarray(points_3d, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        
        pixels, valid = self._cpp.project_pinhole(
            points_3d, fx, fy, cx, cy, dist_coeffs,
            width, height, near_z, far_z
        )
        
        return np.array(pixels), np.array(valid)
    
    def project_kannala_brandt(
        self,
        points_3d: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        k1: float, k2: float, k3: float, k4: float,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """
        使用C++扩展进行Kannala-Brandt鱼眼相机投影
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            fx, fy: 焦距
            cx, cy: 主点坐标
            k1, k2, k3, k4: 畸变系数
            width, height: 图像尺寸
            near_z, far_z: 深度范围
        
        Returns:
            (像素坐标, 有效掩码)
        """
        if not self.is_available():
            raise RuntimeError("CPython backend is not available")
        
        points_3d = np.asarray(points_3d, dtype=np.float64)
        
        pixels, valid = self._cpp.project_kannala_brandt(
            points_3d, fx, fy, cx, cy, k1, k2, k3, k4,
            width, height, near_z, far_z
        )
        
        return np.array(pixels), np.array(valid)
    
    def project_ftheta(
        self,
        points_3d: np.ndarray,
        fw_poly: np.ndarray,
        cx: float, cy: float,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """
        使用C++扩展进行F-Theta鱼眼相机投影
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            fw_poly: 焦距多项式系数
            cx, cy: 主点坐标
            width, height: 图像尺寸
            near_z, far_z: 深度范围
        
        Returns:
            (像素坐标, 有效掩码)
        """
        if not self.is_available():
            raise RuntimeError("CPython backend is not available")
        
        points_3d = np.asarray(points_3d, dtype=np.float64)
        fw_poly = np.asarray(fw_poly, dtype=np.float64)
        
        pixels, valid = self._cpp.project_ftheta(
            points_3d, fw_poly, cx, cy,
            width, height, near_z, far_z
        )
        
        return np.array(pixels), np.array(valid)