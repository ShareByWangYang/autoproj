import pytest
import numpy as np
from autoproj import BackendSelector, NumPyBackend


class TestBackendSelector:
    """后端选择器测试"""

    def test_available_backends(self):
        """测试获取可用后端列表 (当前仅 numpy)"""
        available = BackendSelector.available_backends()
        assert isinstance(available, list)
        assert available == ['numpy']

    def test_select_default(self):
        """测试自动选择后端"""
        backend = BackendSelector.select()
        assert backend.is_available()
        assert backend.name() == 'numpy'

    def test_select_numpy(self):
        """测试选择 NumPy 后端"""
        backend = BackendSelector.select('numpy')
        assert isinstance(backend, NumPyBackend)
        assert backend.name() == 'numpy'

    def test_select_unknown_raises(self):
        """指定未注册的后端名应抛出 ValueError"""
        with pytest.raises(ValueError):
            BackendSelector.select('cpp')

    def test_data_aware_returns_numpy(self):
        """数据量感知选择当前始终返回 numpy"""
        backend = BackendSelector.select(n_points=1_000_000, operation='project_points')
        assert backend.name() == 'numpy'

    def test_data_aware_cache(self):
        """(operation, n_points) 缓存命中"""
        BackendSelector.reset()
        b1 = BackendSelector.select(n_points=100, operation='project_box')
        b2 = BackendSelector.select(n_points=100, operation='project_box')
        assert b1 is b2


class TestNumPyBackend:
    """NumPy 后端测试"""

    def test_basic_operations(self):
        """测试基本数学运算"""
        backend = NumPyBackend()

        arr = backend.ones((3,))
        assert arr.shape == (3,)

        x = np.array([1.0, 4.0, 9.0])
        result = backend.sqrt(x)
        expected = np.sqrt(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matrix_operations(self):
        """测试矩阵运算"""
        backend = NumPyBackend()

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        result = backend.matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_array_equal(result, expected)


class TestBackendIntegration:
    """后端集成测试"""

    def test_backend_in_camera(self):
        """测试后端在相机投影中的使用"""
        from autoproj import PinholeCamera

        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )

        points_3d = np.array([[1, 0, 10]], dtype=np.float64)
        result, valid = camera.project(points_3d, pts_in_cam=True, preserve_extra=True)

        assert result.shape == points_3d.shape
        pixels = result[:, :2]
        depths = result[:, 2]

        assert valid[0]
        assert pixels[0][0] == 1060  # fx*(x/z) + cx = 1000*(1/10) + 960
        assert depths[0] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
