from .base import Backend
from typing import Optional
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CUDABackend(Backend):
    """
    CUDA后端实现
    
    基于CuPy的GPU加速实现，需要安装cupy包
    """
    
    def name(self) -> str:
        return 'cuda'
    
    def is_available(self) -> bool:
        if not CUPY_AVAILABLE:
            return False
        try:
            # 尝试创建一个小数组来验证GPU是否可用
            x = cp.array([1, 2, 3])
            return True
        except Exception:
            return False
    
    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cp.dot(a, b).get()
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cp.matmul(a, b).get()
    
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return cp.sqrt(x).get()
    
    def arctan(self, x: np.ndarray) -> np.ndarray:
        return cp.arctan(x).get()
    
    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return cp.arctan2(y, x).get()
    
    def cos(self, x: np.ndarray) -> np.ndarray:
        return cp.cos(x).get()
    
    def sin(self, x: np.ndarray) -> np.ndarray:
        return cp.sin(x).get()
    
    def maximum(self, x: np.ndarray, y: float) -> np.ndarray:
        return cp.maximum(x, y).get()
    
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return cp.clip(x, min_val, max_val).get()
    
    def zeros_like(self, x: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return cp.zeros_like(x).get()
        return cp.zeros_like(x, dtype=dtype).get()
    
    def ones(self, shape: tuple, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return cp.ones(shape).get()
        return cp.ones(shape, dtype=dtype).get()
    
    def hstack(self, arrays: list) -> np.ndarray:
        return cp.hstack(arrays).get()
    
    def astype(self, x: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return x.astype(dtype)
    
    def asarray(self, x, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return cp.asarray(x, dtype=dtype).get()
