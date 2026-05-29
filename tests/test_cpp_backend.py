import numpy as np
import pytest
import time

from autoproj import CameraFactory, PinholeCamera, KannalaBrandtCamera, FThetaCamera
from autoproj.backends import CPythonBackend, NumPyBackend, BackendSelector


class TestCPythonBackendAvailability:
    """测试CPython后端可用性"""
    
    def test_cpp_backend_exists(self):
        """测试CPython后端是否存在"""
        backend = CPythonBackend()
        assert backend is not None
    
    def test_cpp_backend_name(self):
        """测试后端名称"""
        backend = CPythonBackend()
        assert backend.name() == 'cpp'
    
    def test_cpp_backend_available(self):
        """测试后端是否可用"""
        backend = CPythonBackend()
        assert backend.is_available() is True
    
    def test_backend_selector_includes_cpp(self):
        """测试选择器包含cpp后端"""
        available = BackendSelector.available_backends()
        assert 'cpp' in available


class TestCPythonBackendProjection:
    """测试CPython后端投影功能"""
    
    def test_pinhole_projection_cpp(self):
        """测试针孔相机使用CPP后端"""
        backend = CPythonBackend()
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            backend=backend
        )
        
        points_3d = np.array([[0, 0, 10], [1, 0, 10], [-1, 0, 10]], dtype=np.float64)
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert pixels[0][0] == 960
        assert pixels[0][1] == 540
    
    def test_kannala_projection_cpp(self):
        """测试Kannala-Brandt鱼眼相机使用CPP后端"""
        backend = CPythonBackend()
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05,
            backend=backend
        )
        
        points_3d = np.array([[1, 0, 10]], dtype=np.float64)
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert 0 <= pixels[0][0] < 1920
        assert 0 <= pixels[0][1] < 1080
    
    def test_ftheta_projection_cpp(self):
        """测试F-Theta鱼眼相机使用CPP后端"""
        backend = CPythonBackend()
        camera = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500],
            cx=960, cy=540,
            backend=backend
        )
        
        points_3d = np.array([[1, 0, 10]], dtype=np.float64)
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert 0 <= pixels[0][0] < 1920
        assert 0 <= pixels[0][1] < 1080


class TestCPythonBackendConsistency:
    """测试CPython后端与NumPy后端结果一致性"""
    
    def test_pinhole_consistency(self):
        """测试针孔相机两种后端结果一致"""
        cpp_backend = CPythonBackend()
        numpy_backend = NumPyBackend()
        
        camera_cpp = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, -0.05, 0.01, -0.01],
            backend=cpp_backend
        )
        
        camera_numpy = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, -0.05, 0.01, -0.01],
            backend=numpy_backend
        )
        
        np.random.seed(42)
        points_3d = np.random.randn(100, 3).astype(np.float64)
        points_3d[:, 2] = np.abs(points_3d[:, 2]) + 1  # 确保正深度
        
        pixels_cpp, valid_cpp = camera_cpp.project(points_3d, pts_in_cam=True)
        pixels_numpy, valid_numpy = camera_numpy.project(points_3d, pts_in_cam=True)
        
        np.testing.assert_array_equal(valid_cpp, valid_numpy)
        np.testing.assert_array_equal(pixels_cpp, pixels_numpy)
    
    def test_kannala_consistency(self):
        """测试Kannala-Brandt相机两种后端结果一致"""
        cpp_backend = CPythonBackend()
        numpy_backend = NumPyBackend()
        
        camera_cpp = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05, k3=0.01, k4=0.005,
            backend=cpp_backend
        )
        
        camera_numpy = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05, k3=0.01, k4=0.005,
            backend=numpy_backend
        )
        
        np.random.seed(42)
        points_3d = np.random.randn(100, 3).astype(np.float64)
        points_3d[:, 2] = np.abs(points_3d[:, 2]) + 1
        
        pixels_cpp, valid_cpp = camera_cpp.project(points_3d, pts_in_cam=True)
        pixels_numpy, valid_numpy = camera_numpy.project(points_3d, pts_in_cam=True)
        
        np.testing.assert_array_equal(valid_cpp, valid_numpy)
        np.testing.assert_array_equal(pixels_cpp, pixels_numpy)
    
    def test_ftheta_consistency(self):
        """测试F-Theta相机两种后端结果一致"""
        cpp_backend = CPythonBackend()
        numpy_backend = NumPyBackend()
        
        camera_cpp = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500, 50],
            cx=960, cy=540,
            backend=cpp_backend
        )
        
        camera_numpy = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500, 50],
            cx=960, cy=540,
            backend=numpy_backend
        )
        
        np.random.seed(42)
        points_3d = np.random.randn(100, 3).astype(np.float64)
        points_3d[:, 2] = np.abs(points_3d[:, 2]) + 1
        
        pixels_cpp, valid_cpp = camera_cpp.project(points_3d, pts_in_cam=True)
        pixels_numpy, valid_numpy = camera_numpy.project(points_3d, pts_in_cam=True)
        
        np.testing.assert_array_equal(valid_cpp, valid_numpy)
        np.testing.assert_array_equal(pixels_cpp, pixels_numpy)


class TestCPythonBackendPerformance:
    """测试CPython后端性能"""
    
    def test_pinhole_performance(self):
        """测试针孔相机性能"""
        cpp_backend = CPythonBackend()
        numpy_backend = NumPyBackend()
        
        camera_cpp = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, -0.05, 0.01, -0.01],
            backend=cpp_backend
        )
        
        camera_numpy = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, -0.05, 0.01, -0.01],
            backend=numpy_backend
        )
        
        np.random.seed(42)
        points_3d = np.random.randn(100000, 3).astype(np.float64)
        points_3d[:, 2] = np.abs(points_3d[:, 2]) + 0.1
        
        # 预热
        for _ in range(3):
            camera_cpp.project(points_3d[:100], pts_in_cam=True)
            camera_numpy.project(points_3d[:100], pts_in_cam=True)
        
        # 测试NumPy后端
        start = time.time()
        for _ in range(10):
            camera_numpy.project(points_3d, pts_in_cam=True)
        numpy_time = time.time() - start
        
        # 测试CPP后端
        start = time.time()
        for _ in range(10):
            camera_cpp.project(points_3d, pts_in_cam=True)
        cpp_time = time.time() - start
        
        speedup = numpy_time / cpp_time
        print(f"\nPinhole Performance Test:")
        print(f"  NumPy time: {numpy_time:.4f}s")
        print(f"  CPP time: {cpp_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 2, f"Expected speedup > 2x, got {speedup:.2f}x"
    
    def test_kannala_performance(self):
        """测试Kannala-Brandt鱼眼相机性能"""
        cpp_backend = CPythonBackend()
        numpy_backend = NumPyBackend()
        
        camera_cpp = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05,
            backend=cpp_backend
        )
        
        camera_numpy = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05,
            backend=numpy_backend
        )
        
        np.random.seed(42)
        points_3d = np.random.randn(100000, 3).astype(np.float64)
        points_3d[:, 2] = np.abs(points_3d[:, 2]) + 0.1
        
        # 预热
        for _ in range(3):
            camera_cpp.project(points_3d[:100], pts_in_cam=True)
            camera_numpy.project(points_3d[:100], pts_in_cam=True)
        
        # 测试NumPy后端
        start = time.time()
        for _ in range(10):
            camera_numpy.project(points_3d, pts_in_cam=True)
        numpy_time = time.time() - start
        
        # 测试CPP后端
        start = time.time()
        for _ in range(10):
            camera_cpp.project(points_3d, pts_in_cam=True)
        cpp_time = time.time() - start
        
        speedup = numpy_time / cpp_time
        print(f"\nKannala-Brandt Performance Test:")
        print(f"  NumPy time: {numpy_time:.4f}s")
        print(f"  CPP time: {cpp_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 2, f"Expected speedup > 2x, got {speedup:.2f}x"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])