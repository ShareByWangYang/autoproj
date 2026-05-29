import pytest
import numpy as np
from autoproj import BackendSelector, NumPyBackend, CUDABackend


class TestBackendSelector:
    """后端选择器测试"""
    
    def test_available_backends(self):
        """测试获取可用后端列表"""
        available = BackendSelector.available_backends()
        assert isinstance(available, list)
        assert 'numpy' in available  # NumPy后端始终可用
    
    def test_select_default(self):
        """测试自动选择后端"""
        backend = BackendSelector.select()
        assert backend.is_available()
    
    def test_select_numpy(self):
        """测试选择NumPy后端"""
        backend = BackendSelector.select('numpy')
        assert isinstance(backend, NumPyBackend)
        assert backend.name() == 'numpy'
    
    def test_select_cuda_unavailable(self):
        """测试选择不可用的CUDA后端"""
        try:
            backend = BackendSelector.select('cuda')
            # 如果CUDA可用，验证它
            assert isinstance(backend, CUDABackend)
        except ValueError:
            # CUDA不可用时应该抛出异常
            pass


class TestNumPyBackend:
    """NumPy后端测试"""
    
    def test_basic_operations(self):
        """测试基本数学运算"""
        backend = NumPyBackend()
        
        # 测试数组创建
        arr = backend.ones((3,))
        assert arr.shape == (3,)
        
        # 测试数学函数
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
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert pixels[0][0] == 1060  # fx*(x/z) + cx = 1000*(1/10) + 960


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
