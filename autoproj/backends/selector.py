from .base import Backend
from .numpy_backend import NumPyBackend
from typing import Optional, Dict


# 数据量感知选择表
# 保留架构骨架便于未来重新引入 cuda/cpp 后端:
# - 未来新增后端只需在 _BACKEND_THRESHOLDS 加入阈值项
# - 在 _backends 字典加入对应懒加载分支即可
#
# 当前状态: 仅 numpy 可用, 所有 op 永远返回 numpy
# 基准测试结论 (benchmark_results.csv):
# - project_points: numpy 100K 点 ~6ms (10Hz 雷达 100ms 帧预算内)
# - project_boxes: numpy + Numba 批量接口 14x 加速 vs 单条循环
# - project_lines: numpy + Numba 3.95x 加速 vs Python for 循环
_BACKEND_THRESHOLDS: Dict[str, Dict[str, int]] = {
    'project_points': {'numpy': 0},
    'project_box': {'numpy': 0},
    'project_boxes': {'numpy': 0},
    'project_polygon': {'numpy': 0},
    'project_polygons': {'numpy': 0},
    'project_lines': {'numpy': 0},
    'default': {'numpy': 0},
}


class BackendSelector:
    """
    后端选择器, 自动检测并选择最佳可用后端

    当前实现: 仅 NumPy 后端可用 (基于 numpy + numba JIT 加速)
    架构保留: 为未来重新引入 cuda/cpp 后端预留扩展点

    优先级 (未来扩展时): CUDA > CPython > NumPy

    Example:
        backend = BackendSelector.select()  # 自动选择 (当前固定返回 numpy)
        backend = BackendSelector.select('numpy')  # 显式指定
        backend = BackendSelector.select(n_points=1_000_000, operation='project_points')
        # 数据量感知 (当前因仅 numpy 可用, 始终返回 numpy)
    """

    # 懒加载的后端实例
    # 未来扩展: 加入 'cpp': None, 'cuda': None 并在 _get_backend 中实现实例化
    _backends: Dict[str, Optional[Backend]] = {
        'numpy': None,
    }

    # 已计算过的 (operation, n_points) → backend_name 缓存
    _selection_cache: Dict[tuple, str] = {}

    @classmethod
    def _get_backend(
        cls,
        name: str,
        auto_build: bool = True,
        omp_threshold: Optional[int] = None,
    ) -> Backend:
        """懒加载后端实例"""
        if cls._backends[name] is None:
            if name == 'numpy':
                cls._backends[name] = NumPyBackend()
            # 未来扩展点:
            # elif name == 'cpp':
            #     from .cpp_backend import CPythonBackend
            #     kw = {'auto_build': auto_build}
            #     if omp_threshold is not None:
            #         kw['omp_threshold'] = omp_threshold
            #     cls._backends[name] = CPythonBackend(**kw)
            # elif name == 'cuda':
            #     from .cuda_backend import CUDABackend
            #     cls._backends[name] = CUDABackend()
        return cls._backends[name]

    @classmethod
    def available_backends(cls, auto_build: bool = True) -> list:
        """返回所有可用后端列表"""
        available = []
        for name in ['numpy']:  # 未来扩展: ['cuda', 'cpp', 'numpy']
            backend = cls._get_backend(name, auto_build=auto_build)
            if backend.is_available():
                available.append(name)
        return available

    @classmethod
    def select(
        cls,
        backend_name: Optional[str] = None,
        auto_build: bool = True,
        auto_fallback: bool = True,
        omp_threshold: Optional[int] = None,
        n_points: Optional[int] = None,
        operation: Optional[str] = None,
    ) -> Backend:
        """
        选择后端

        Args:
            backend_name: 指定后端名称, 当前仅支持 'numpy'
                         None 表示自动选择 (返回 numpy)
                         若同时传入 n_points + operation, backend_name 优先级更高
            auto_build: 保留参数, 当前仅 numpy 不需要构建
                        (未来 C++ 后端恢复时用于控制自动编译)
            auto_fallback: 指定后端不可用时是否自动降级
            omp_threshold: 保留参数, 未来 C++ 后端 OpenMP 并行阈值
            n_points: 数据量感知选择输入规模提示
                     (当前因仅 numpy 可用, 不影响结果)
            operation: 操作类型 ('project_points' / 'project_box' / ...)

        Returns:
            Backend 实例

        Raises:
            ValueError: backend_name 不在已注册列表中
        """
        # 数据量感知路径
        if backend_name is None and n_points is not None:
            chosen = cls._select_by_data_size(n_points, operation, auto_fallback)
            return cls._get_backend(chosen, auto_build=auto_build,
                                     omp_threshold=omp_threshold)

        if backend_name is not None:
            backend_name = backend_name.lower()
            if backend_name not in cls._backends:
                available = ', '.join(cls._backends.keys())
                raise ValueError(
                    f"Unknown backend: '{backend_name}'. Available: {available}"
                )

            backend = cls._get_backend(backend_name, auto_build=auto_build,
                                       omp_threshold=omp_threshold)
            if backend.is_available():
                return backend

            if not auto_fallback:
                raise ValueError(f"Backend '{backend_name}' is not available")

            print(f"⚠️ 指定的 '{backend_name}' 后端不可用, 正在降级...")

        # 自动选择: 当前仅 numpy 可用
        # 未来扩展时改为: for name in ['cuda', 'cpp', 'numpy']:
        for name in ['numpy']:
            backend = cls._get_backend(name, auto_build=auto_build,
                                        omp_threshold=omp_threshold)
            if backend.is_available():
                if backend_name is not None:
                    print(f"✓ 使用降级后的 '{name}' 后端")
                return backend

        raise RuntimeError("No backend available")

    @classmethod
    def _select_by_data_size(
        cls,
        n_points: int,
        operation: Optional[str],
        auto_fallback: bool = True,
    ) -> str:
        """
        根据 n_points + operation 选择最优后端名称

        当前实现: 仅 numpy 可用, 始终返回 'numpy'
        未来扩展: 重新引入 cuda/cpp 时, 查询 _BACKEND_THRESHOLDS 表决策
        """
        cache_key = (operation, n_points)
        if cache_key in cls._selection_cache:
            return cls._selection_cache[cache_key]

        op = operation or 'default'
        thresholds = _BACKEND_THRESHOLDS.get(op, _BACKEND_THRESHOLDS['default'])

        # 构造候选列表 (按优先级降序)
        # 当前仅 numpy, 未来可在此扩展 cuda/cpp 阈值判断
        candidates = ['numpy']

        chosen = None
        for name in candidates:
            backend = cls._get_backend(name)
            if backend.is_available():
                chosen = name
                break

        if chosen is None:
            if not auto_fallback:
                raise ValueError(
                    f"No backend available for operation='{op}', n_points={n_points}"
                )
            chosen = 'numpy'

        cls._selection_cache[cache_key] = chosen
        return chosen

    @classmethod
    def get_backend(
        cls,
        name: str,
        auto_build: bool = True,
        omp_threshold: Optional[int] = None,
    ) -> Backend:
        """获取指定名称的后端实例"""
        return cls._get_backend(name, auto_build=auto_build,
                                 omp_threshold=omp_threshold)

    @classmethod
    def reset(cls) -> None:
        """重置所有后端, 用于测试"""
        for key in cls._backends:
            cls._backends[key] = None
        cls._selection_cache.clear()
