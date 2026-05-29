import numpy as np
from .base import Backend
from typing import Optional


class NumPyBackend(Backend):
    """
    NumPy后端实现
    
    基于NumPy的纯Python实现，作为默认后端和fallback
    """
    
    def __init__(self):
        self.np = np
    
    def name(self) -> str:
        return 'numpy'
    
    def is_available(self) -> bool:
        return True
    
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
