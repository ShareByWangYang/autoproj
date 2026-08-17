import pytest
import numpy as np
import os
import tempfile

from autoproj import ConfigLoader, load_camera, load_camera_from_dict, save_camera_config
from autoproj import PinholeCamera, KannalaBrandtCamera, FThetaCamera


class TestConfigLoaderFromDict:
    """从字典加载配置测试"""
    
    def test_load_pinhole_from_dict(self):
        """测试从字典加载针孔相机"""
        config = {
            'type': 'pinhole',
            'width': 1920,
            'height': 1080,
            'fx': 1000,
            'fy': 1000,
            'cx': 960,
            'cy': 540,
            'dist_coeffs': [0.1, -0.05, 0, 0]
        }
        camera = load_camera_from_dict(config)
        assert isinstance(camera, PinholeCamera)
        assert camera.width == 1920
        assert camera.fx == 1000
    
    def test_load_kannala_from_dict(self):
        """测试从字典加载Kannala-Brandt相机"""
        config = {
            'type': 'fisheye:kannala',
            'width': 1920,
            'height': 1080,
            'fx': 500,
            'fy': 500,
            'k1': 0.1,
            'k2': 0.05
        }
        camera = load_camera_from_dict(config)
        assert isinstance(camera, KannalaBrandtCamera)
        assert camera.k1 == 0.1
    
    def test_load_ftheta_from_dict(self):
        """测试从字典加载F-Theta相机"""
        config = {
            'type': 'fisheye:ftheta',
            'width': 1920,
            'height': 1080,
            'fw_poly': [0, 500],
            'cx': 960,
            'cy': 540
        }
        camera = load_camera_from_dict(config)
        assert isinstance(camera, FThetaCamera)
        assert camera.fw_poly[1] == 500
    
    def test_load_dict_missing_type(self):
        """测试缺少type字段的配置"""
        config = {
            'width': 1920,
            'height': 1080,
            'fx': 1000
        }
        with pytest.raises(ValueError):
            load_camera_from_dict(config)


class TestConfigLoaderFromFile:
    """从文件加载配置测试"""
    
    def test_load_yaml_file(self):
        """测试从YAML文件加载相机配置"""
        config_content = """
type: pinhole
width: 1920
height: 1080
fx: 1000
fy: 1000
cx: 960
cy: 540
dist_coeffs: [0.1, -0.05, 0, 0]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            camera = load_camera(temp_path)
            assert isinstance(camera, PinholeCamera)
            assert camera.width == 1920
        finally:
            os.unlink(temp_path)
    
    def test_load_json_file(self):
        """测试从JSON文件加载相机配置"""
        import json
        config = {
            'type': 'pinhole',
            'width': 1920,
            'height': 1080,
            'fx': 1000,
            'fy': 1000
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            camera = load_camera(temp_path)
            assert isinstance(camera, PinholeCamera)
            assert camera.fx == 1000
        finally:
            os.unlink(temp_path)
    
    def test_load_invalid_file_format(self):
        """测试加载不支持的文件格式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('invalid content')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_camera(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_camera('/nonexistent/path/config.yaml')


class TestConfigLoaderSave:
    """保存配置测试"""
    
    def test_save_and_load_yaml(self):
        """测试保存并重新加载YAML配置"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, 0, 0, 0]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_camera_config(camera, temp_path)
            assert os.path.exists(temp_path)
            
            loaded_camera = load_camera(temp_path)
            assert isinstance(loaded_camera, PinholeCamera)
            assert loaded_camera.width == 1920
            assert loaded_camera.fx == 1000
        finally:
            os.unlink(temp_path)
    
    def test_save_json(self):
        """测试保存为JSON文件"""
        camera = PinholeCamera(
            width=1280, height=720,
            fx=800, fy=800
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_camera_config(camera, temp_path)
            assert os.path.exists(temp_path)
            
            loaded_camera = load_camera(temp_path)
            assert loaded_camera.width == 1280
        finally:
            os.unlink(temp_path)


class TestConfigLoaderIntegration:
    """配置加载器集成测试"""
    
    def test_full_roundtrip(self):
        """测试完整的保存-加载往返流程"""
        # 创建相机
        original = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=[0.1, -0.05, 0.01, -0.01]
        )
        
        # 保存到文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_camera_config(original, temp_path)
            
            # 重新加载
            loaded = load_camera(temp_path)
            
            # 验证参数一致
            assert loaded.width == original.width
            assert loaded.height == original.height
            assert loaded.fx == original.fx
            assert loaded.fy == original.fy
            assert loaded.cx == original.cx
            assert loaded.cy == original.cy
            np.testing.assert_array_almost_equal(loaded.dist_coeffs, original.dist_coeffs)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
