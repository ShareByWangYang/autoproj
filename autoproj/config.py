import yaml
import json
from typing import Optional, Dict, Any, Union
import os

from .camera import Camera, PinholeCamera, KannalaBrandtCamera, FThetaCamera
from .factory import CameraFactory


class ConfigLoader:
    """
    配置加载器，支持从YAML/JSON文件加载相机配置
    
    Example:
        # 从YAML文件加载
        loader = ConfigLoader()
        camera = loader.load_camera('camera_config.yaml')
        
        # 从字典加载
        config = {'type': 'pinhole', 'width': 1920, 'height': 1080, ...}
        camera = loader.load_camera_from_dict(config)
    """
    
    def load_camera(self, config_path: str) -> Camera:
        """
        从配置文件加载相机
        
        Args:
            config_path: 配置文件路径（支持 .yaml, .yml, .json）
        
        Returns:
            Camera实例
        
        Raises:
            ValueError: 如果文件格式不支持
            FileNotFoundError: 如果文件不存在
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 根据文件扩展名选择解析方式
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext in ('.yaml', '.yml'):
            config = self._load_yaml(config_path)
        elif ext == '.json':
            config = self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .yaml, .yml, .json")
        
        return self.load_camera_from_dict(config)
    
    def load_camera_from_dict(self, config: Dict[str, Any]) -> Camera:
        """
        从字典配置加载相机
        
        Args:
            config: 相机配置字典
        
        Returns:
            Camera实例
        
        Raises:
            ValueError: 如果配置不完整或类型错误
        """
        if 'type' not in config:
            raise ValueError("Config must contain 'type' field")
        
        camera_type = config['type']
        
        # 使用工厂创建相机
        return CameraFactory.create_from_dict(config)
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载YAML文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_camera_config(self, camera: Camera, file_path: str) -> None:
        """
        将相机配置保存到文件
        
        Args:
            camera: Camera实例
            file_path: 保存路径（支持 .yaml, .yml, .json）
        """
        config = self._camera_to_dict(camera)
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ('.yaml', '.yml'):
            self._save_yaml(config, file_path)
        elif ext == '.json':
            self._save_json(config, file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .yaml, .yml, .json")
    
    def _camera_to_dict(self, camera: Camera) -> Dict[str, Any]:
        """将相机对象转换为字典"""
        config = {
            'width': camera.width,
            'height': camera.height,
            'cx': camera.cx,
            'cy': camera.cy,
            'near_z': camera.near_z,
            'far_z': camera.far_z
        }
        
        if isinstance(camera, PinholeCamera):
            config['type'] = 'pinhole:standard'
            config['fx'] = camera.fx
            config['fy'] = camera.fy
            config['dist_coeffs'] = camera.dist_coeffs.tolist()
        elif isinstance(camera, KannalaBrandtCamera):
            config['type'] = 'fisheye:kannala'
            config['fx'] = camera.fx
            config['fy'] = camera.fy
            config['k1'] = camera.k1
            config['k2'] = camera.k2
            config['k3'] = camera.k3
            config['k4'] = camera.k4
        elif isinstance(camera, FThetaCamera):
            config['type'] = 'fisheye:ftheta'
            config['fw_poly'] = camera.fw_poly.tolist()
        
        return config
    
    def _save_yaml(self, config: Dict[str, Any], file_path: str) -> None:
        """保存为YAML文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def _save_json(self, config: Dict[str, Any], file_path: str) -> None:
        """保存为JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)


# 便捷函数
def load_camera(config_path: str) -> Camera:
    """
    从配置文件加载相机（便捷函数）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Camera实例
    """
    loader = ConfigLoader()
    return loader.load_camera(config_path)


def load_camera_from_dict(config: Dict[str, Any]) -> Camera:
    """
    从字典配置加载相机（便捷函数）
    
    Args:
        config: 相机配置字典
    
    Returns:
        Camera实例
    """
    loader = ConfigLoader()
    return loader.load_camera_from_dict(config)


def save_camera_config(camera: Camera, file_path: str) -> None:
    """
    保存相机配置到文件（便捷函数）
    
    Args:
        camera: Camera实例
        file_path: 保存路径
    """
    loader = ConfigLoader()
    loader.save_camera_config(camera, file_path)
