import numpy as np
import pytest
from autoproj import CameraFactory, PinholeCamera, KannalaBrandtCamera, FThetaCamera


class TestCameraFactory:
    """相机工厂测试"""
    
    def test_create_pinhole(self):
        """测试创建针孔相机"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        assert isinstance(camera, PinholeCamera)
        assert camera.width == 1920
        assert camera.height == 1080
        assert camera.fx == 1000
    
    def test_create_fisheye_default(self):
        """测试创建默认鱼眼相机（Kannala-Brandt）"""
        camera = CameraFactory.create_fisheye(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540
        )
        assert isinstance(camera, KannalaBrandtCamera)
    
    def test_create_fisheye_ftheta(self):
        """测试创建F-Theta鱼眼相机"""
        camera = CameraFactory.create_fisheye(
            sub_type='ftheta',
            width=1920, height=1080,
            fw_poly=[0, 500]
        )
        assert isinstance(camera, FThetaCamera)
    
    def test_list_categories(self):
        """测试列出相机类别"""
        categories = CameraFactory.list_categories()
        assert len(categories) >= 2
        category_names = [cat['name'] for cat in categories]
        assert 'pinhole' in category_names
        assert 'fisheye' in category_names
    
    def test_list_subtypes(self):
        """测试列出子类型"""
        subtypes = CameraFactory.list_subtypes('fisheye')
        assert 'kannala' in subtypes
        assert 'ftheta' in subtypes


class TestPinholeCamera:
    """针孔相机测试"""
    
    def test_project_simple(self):
        """测试简单投影"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        
        # 相机前方10米处的点
        points_3d = np.array([[0, 0, 10]])
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert pixels[0][0] == 960  # cx
        assert pixels[0][1] == 540  # cy
    
    def test_project_offset(self):
        """测试偏移点投影"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        
        # x方向偏移1米，z=10米
        points_3d = np.array([[1, 0, 10]])
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        expected_u = 1000 * (1/10) + 960  # fx * (x/z) + cx
        assert abs(pixels[0][0] - expected_u) < 1
    
    def test_distortion(self):
        """测试畸变校正"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.2, 0, 0, 0]  # 较大的k1系数
        )
        
        points_3d = np.array([[5, 0, 10]], dtype=np.float64)  # 更大的偏移点
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        # 无畸变时应为 1000*(5/10) + 960 = 1460
        # 正畸变(k1>0)会使点向外偏移，所以u应该大于1460
        expected_undistorted = 1000 * (5/10) + 960
        assert pixels[0][0] > expected_undistorted + 5  # 畸变应导致明显偏移
    
    def test_fov(self):
        """测试FOV计算"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000
        )
        
        fov_h = camera.fov_h
        fov_v = camera.fov_v
        
        # 检查FOV在合理范围内
        assert 60 < fov_h < 120
        assert 30 < fov_v < 90


class TestKannalaBrandtCamera:
    """Kannala-Brandt鱼眼相机测试"""
    
    def test_project(self):
        """测试鱼眼投影"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.1
        )
        
        points_3d = np.array([[1, 0, 10]])
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]
        assert 0 <= pixels[0][0] < 1920
        assert 0 <= pixels[0][1] < 1080


class TestFThetaCamera:
    """F-Theta鱼眼相机测试"""
    
    def test_project(self):
        """测试F-Theta投影"""
        camera = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500],
            cx=960, cy=540
        )
        
        points_3d = np.array([[1, 0, 10]])
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert valid[0]


class TestFromDict:
    """从字典创建相机测试"""
    
    def test_pinhole_from_dict(self):
        """测试从字典创建针孔相机"""
        config = {
            'width': 1920,
            'height': 1080,
            'fx': 1000,
            'fy': 1000,
            'cx': 960,
            'cy': 540,
            'dist_coeffs': [0.1, 0, 0, 0]
        }
        camera = PinholeCamera.from_dict(config)
        assert isinstance(camera, PinholeCamera)
        assert camera.width == 1920
        assert camera.fx == 1000
        assert bool(camera.has_distortion) is True
    
    def test_kannala_from_dict(self):
        """测试从字典创建Kannala-Brandt相机"""
        config = {
            'width': 1920,
            'height': 1080,
            'fx': 500,
            'fy': 500,
            'k1': 0.1,
            'k2': 0.05
        }
        camera = KannalaBrandtCamera.from_dict(config)
        assert isinstance(camera, KannalaBrandtCamera)
        assert camera.k1 == 0.1
        assert camera.k2 == 0.05
    
    def test_ftheta_from_dict(self):
        """测试从字典创建F-Theta相机"""
        config = {
            'width': 1920,
            'height': 1080,
            'fw_poly': [0, 500],
            'cx': 960,
            'cy': 540
        }
        camera = FThetaCamera.from_dict(config)
        assert isinstance(camera, FThetaCamera)
        assert camera.fw_poly[1] == 500
    
    def test_factory_create_from_dict(self):
        """测试工厂从字典创建相机"""
        config = {
            'type': 'pinhole:standard',
            'width': 1920,
            'height': 1080,
            'fx': 1000,
            'fy': 1000
        }
        camera = CameraFactory.create_from_dict(config)
        assert isinstance(camera, PinholeCamera)
    
    def test_factory_create_from_dict_fisheye(self):
        """测试工厂从字典创建鱼眼相机"""
        config = {
            'type': 'fisheye:ftheta',
            'width': 1920,
            'height': 1080,
            'fw_poly': [0, 500]
        }
        camera = CameraFactory.create_from_dict(config)
        assert isinstance(camera, FThetaCamera)


class TestBoundaryCheck:
    """边界检查测试"""
    
    def test_out_of_bounds(self):
        """测试边界外的点"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        
        # 非常远的点会超出边界
        points_3d = np.array([[100, 0, 1]])  # x=100, z=1
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert not valid[0]
        assert pixels[0][0] == -1  # 无效点标记为-1
    
    def test_negative_z(self):
        """测试负深度点"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000
        )
        
        points_3d = np.array([[0, 0, -1]])  # z=-1
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        
        assert not valid[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])