from .base import Backend
from .numpy_backend import NumPyBackend
from .cuda_backend import CUDABackend
from typing import Optional


class BackendSelector:
    """
    后端选择器，自动检测并选择最佳可用后端
    
    优先级：CUDA > NumPy
    
    Example:
        selector = BackendSelector()
        backend = selector.select()  # 自动选择最佳后端
        backend = selector.select('cuda')  # 强制选择CUDA后端
    """
    
    _backends = {
        'numpy': NumPyBackend(),
        'cuda': CUDABackend()
    }
    
    @classmethod
    def available_backends(cls) -> list:
        """返回所有可用后端列表"""
        return [name for name, backend in cls._backends.items() if backend.is_available()]
    
    @classmethod
    def select(cls, backend_name: Optional[str] = None) -> Backend:
        """
        选择后端
        
        Args:
            backend_name: 指定后端名称，可选值: 'numpy', 'cuda'
                         如果为None，自动选择最佳可用后端
        
        Returns:
            Backend实例
        
        Raises:
            ValueError: 如果指定的后端不可用
        """
        if backend_name is not None:
            backend_name = backend_name.lower()
            if backend_name not in cls._backends:
                available = ', '.join(cls._backends.keys())
                raise ValueError(f"Unknown backend: '{backend_name}'. Available: {available}")
            
            backend = cls._backends[backend_name]
            if not backend.is_available():
                raise ValueError(f"Backend '{backend_name}' is not available")
            
            return backend
        
        # 自动选择：优先CUDA，其次NumPy
        for name in ['cuda', 'numpy']:
            backend = cls._backends[name]
            if backend.is_available():
                return backend
        
        raise RuntimeError("No backend available")
    
    @classmethod
    def get_backend(cls, name: str) -> Backend:
        """获取指定名称的后端实例"""
        if name not in cls._backends:
            available = ', '.join(cls._backends.keys())
            raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
        return cls._backends[name]
