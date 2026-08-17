from .base import Backend
from .numpy_backend import NumPyBackend
from .cuda_backend import CUDABackend
from .cpp_backend import CPythonBackend
from typing import Optional


class BackendSelector:
    """
    后端选择器，自动检测并选择最佳可用后端
    
    优先级：CUDA > CPython > NumPy
    
    支持 C++ 后端的自动检测和构建：
    - 如果 C++ 后端不可用，会尝试自动编译
    - 如果编译失败，自动回退到 NumPy 后端
    
    Example:
        selector = BackendSelector()
        backend = selector.select()  # 自动选择最佳后端
        backend = selector.select('cuda')  # 强制选择CUDA后端
    """
    
    # 懒加载的后端，避免启动时的开销
    _backends = {
        'numpy': None,
        'cpp': None,
        'cuda': None
    }
    
    @classmethod
    def _get_backend(cls, name: str, auto_build: bool = True) -> Backend:
        """懒加载后端实例"""
        if cls._backends[name] is None:
            if name == 'numpy':
                cls._backends[name] = NumPyBackend()
            elif name == 'cpp':
                cls._backends[name] = CPythonBackend(auto_build=auto_build)
            elif name == 'cuda':
                cls._backends[name] = CUDABackend()
        return cls._backends[name]
    
    @classmethod
    def available_backends(cls, auto_build: bool = True) -> list:
        """返回所有可用后端列表"""
        available = []
        for name in ['cuda', 'cpp', 'numpy']:
            backend = cls._get_backend(name, auto_build=auto_build)
            if backend.is_available():
                available.append(name)
        return available
    
    @classmethod
    def select(
        cls, 
        backend_name: Optional[str] = None,
        auto_build: bool = True,
        auto_fallback: bool = True
    ) -> Backend:
        """
        选择后端
        
        Args:
            backend_name: 指定后端名称，可选值: 'numpy', 'cpp', 'cuda'
                         如果为None，自动选择最佳可用后端
            auto_build: 是否自动尝试构建不可用的 C++ 后端
            auto_fallback: 如果指定的后端不可用，是否自动降级
        
        Returns:
            Backend实例
        
        Raises:
            ValueError: 如果指定的后端不可用且 auto_fallback=False
        """
        if backend_name is not None:
            backend_name = backend_name.lower()
            if backend_name not in cls._backends:
                available = ', '.join(cls._backends.keys())
                raise ValueError(f"Unknown backend: '{backend_name}'. Available: {available}")
            
            # 获取并检查指定后端
            backend = cls._get_backend(backend_name, auto_build=auto_build)
            if backend.is_available():
                return backend
            
            # 如果后端不可用且不允许降级
            if not auto_fallback:
                raise ValueError(f"Backend '{backend_name}' is not available")
            
            # 自动降级提示
            print(f"⚠️ 指定的 '{backend_name}' 后端不可用，正在降级...")
        
        # 自动选择：优先CUDA，其次CPython，最后NumPy
        for name in ['cuda', 'cpp', 'numpy']:
            backend = cls._get_backend(name, auto_build=auto_build)
            if backend.is_available():
                if backend_name is not None:
                    print(f"✓ 使用降级后的 '{name}' 后端")
                return backend
        
        raise RuntimeError("No backend available")
    
    @classmethod
    def get_backend(cls, name: str, auto_build: bool = True) -> Backend:
        """获取指定名称的后端实例"""
        return cls._get_backend(name, auto_build=auto_build)
    
    @classmethod
    def reset(cls) -> None:
        """重置所有后端，用于测试"""
        for key in cls._backends:
            cls._backends[key] = None
