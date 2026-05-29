from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Backend(ABC):
    """
    后端抽象基类，定义统一的计算接口
    
    所有后端实现必须继承此类并实现所有抽象方法
    """
    
    @abstractmethod
    def name(self) -> str:
        """返回后端名称"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass
    
    @abstractmethod
    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵乘法"""
        pass
    
    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵乘法"""
        pass
    
    @abstractmethod
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """平方根"""
        pass
    
    @abstractmethod
    def arctan(self, x: np.ndarray) -> np.ndarray:
        """反正切"""
        pass
    
    @abstractmethod
    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """反正切2"""
        pass
    
    @abstractmethod
    def cos(self, x: np.ndarray) -> np.ndarray:
        """余弦"""
        pass
    
    @abstractmethod
    def sin(self, x: np.ndarray) -> np.ndarray:
        """正弦"""
        pass
    
    @abstractmethod
    def maximum(self, x: np.ndarray, y: float) -> np.ndarray:
        """逐元素最大值"""
        pass
    
    @abstractmethod
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """裁剪值范围"""
        pass
    
    @abstractmethod
    def zeros_like(self, x: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """创建与输入形状相同的零数组"""
        pass
    
    @abstractmethod
    def ones(self, shape: tuple, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """创建全1数组"""
        pass
    
    @abstractmethod
    def hstack(self, arrays: list) -> np.ndarray:
        """水平堆叠数组"""
        pass
    
    @abstractmethod
    def astype(self, x: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """类型转换"""
        pass
    
    @abstractmethod
    def asarray(self, x, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """转换为数组"""
        pass
