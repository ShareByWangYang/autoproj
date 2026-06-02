from .base import Backend
from .build_utils import try_build_cpp_backend
from typing import Optional
import numpy as np

# 全局标记，避免重复构建
_HAS_TRIED_BUILD = False

try:
    from .. import _projection_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class CPythonBackend(Backend):
    """
    CPython后端实现
    
    基于C++扩展的高性能实现，提供比纯Python高10-15倍的性能
    支持自动检测和构建 C++ 扩展
    """
    
    def __init__(self, auto_build: bool = True, auto_install_deps: bool = True):
        """
        初始化 CPython 后端
        
        Args:
            auto_build: 是否自动尝试构建 C++ 扩展
            auto_install_deps: 是否自动安装构建依赖
        """
        self._auto_build = auto_build
        self._auto_install_deps = auto_install_deps
        self._cpp = None
        self._available = False
        
        # 尝试加载或构建
        self._load_or_build()
        self.np = np
    
    def _load_or_build(self) -> None:
        """加载已有的 C++ 扩展，或尝试自动构建"""
        global _HAS_TRIED_BUILD
        
        # 1. 首先尝试直接导入
        try:
            from .. import _projection_cpp
            self._cpp = _projection_cpp
            self._available = True
            return
        except ImportError:
            pass
        
        # 2. 如果启用自动构建且尚未尝试过构建
        if self._auto_build and not _HAS_TRIED_BUILD:
            _HAS_TRIED_BUILD = True
            print("检测到 C++ 后端不可用，尝试自动构建...")
            
            try:
                success = try_build_cpp_backend(
                    auto_install_deps=self._auto_install_deps
                )
                
                if success:
                    # 构建成功后重新导入
                    from .. import _projection_cpp
                    self._cpp = _projection_cpp
                    self._available = True
                    print("C++ 后端构建并加载成功！")
                    return
            except Exception as e:
                print(f"自动构建过程中出错: {e}")
        
        # 3. 如果都失败了
        self._available = False
    
    def name(self) -> str:
        return 'cpp'
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        if not self._available or self._cpp is None:
            return False
        
        try:
            # 简单测试验证功能正常
            test_points = np.array([[1.0, 0.0, 10.0]], dtype=np.float64)
            test_dist = np.zeros(8, dtype=np.float64)
            self._cpp.project_pinhole(
                test_points, 1000.0, 1000.0, 960.0, 540.0, 
                test_dist, 1920, 1080, 0.1, 1000.0
            )
            return True
        except Exception as e:
            print(f"CPython 后端功能测试失败: {e}")
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