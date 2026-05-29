from typing import Optional, Dict, Type, Any, List
from .camera import Camera, PinholeCamera, KannalaBrandtCamera, FThetaCamera


class CameraFactory:
    """
    分层命名相机工厂类
    
    支持按类别创建相机，提供直观的API接口
    
    支持的相机类别：
    - pinhole: 针孔相机
      - standard: 标准针孔相机
      - wide_angle: 广角针孔相机
    - fisheye: 鱼眼相机
      - kannala: Kannala-Brandt模型（默认）
      - ftheta: F-Theta等距模型
    """
    
    _hierarchy: Dict[str, Dict[str, Any]] = {
        'pinhole': {
            'default': 'standard',
            'description': '针孔相机（透视投影）',
            'types': {}
        },
        'fisheye': {
            'default': 'kannala',
            'description': '鱼眼相机（大视场角）',
            'types': {}
        }
    }
    
    @classmethod
    def create(
        cls,
        category: str,
        sub_type: Optional[str] = None,** kwargs
    ) -> Camera:
        """
        分层创建相机
        
        Args:
            category: 相机类别 ('pinhole', 'fisheye')
            sub_type: 子类型（可选，默认为该类别的默认类型）
            **kwargs: 相机参数
        
        Returns:
            Camera实例
        
        Raises:
            ValueError: 如果类别或子类型不存在
        """
        category = category.lower()
        
        if category not in cls._hierarchy:
            available = ', '.join(sorted(cls._hierarchy.keys()))
            raise ValueError(f"Unknown camera category: '{category}'. Available: {available}")
        
        config = cls._hierarchy[category]
        
        # 使用默认子类型
        if sub_type is None:
            sub_type = config['default']
        
        sub_type = sub_type.lower()
        
        if sub_type not in config['types']:
            available = ', '.join(sorted(config['types'].keys()))
            raise ValueError(f"Unknown {category} sub-type: '{sub_type}'. Available: {available}")
        
        camera_class = config['types'][sub_type]
        return camera_class(**kwargs)
    
    @classmethod
    def create_from_dict(cls, config: dict) -> Camera:
        """
        从字典配置创建相机
        
        Args:
            config: 相机配置字典，必须包含以下键：
                - type: 相机类型，格式为 'category:sub_type' 或仅 'category'
                  例如: 'pinhole', 'pinhole:standard', 'fisheye:kannala', 'fisheye:ftheta'
        
        Returns:
            Camera实例
        
        Example:
            config = {
                'type': 'pinhole:standard',
                'width': 1920,
                'height': 1080,
                'fx': 1000,
                'fy': 1000,
                'cx': 960,
                'cy': 540,
                'dist_coeffs': [0.1, -0.05, 0, 0]
            }
            camera = CameraFactory.create_from_dict(config)
        """
        if 'type' not in config:
            raise ValueError("config must contain 'type' key")
        
        type_str = config['type']
        config_copy = config.copy()
        config_copy.pop('type')
        
        # 解析类型字符串
        if ':' in type_str:
            category, sub_type = type_str.split(':', 1)
        else:
            category = type_str
            sub_type = None
        
        return cls.create(category, sub_type, **config_copy)
    
    @classmethod
    def create_pinhole(cls, sub_type: Optional[str] = None, **kwargs) -> Camera:
        """创建针孔相机"""
        return cls.create('pinhole', sub_type, **kwargs)
    
    @classmethod
    def create_fisheye(cls, sub_type: Optional[str] = None, **kwargs) -> Camera:
        """创建鱼眼相机"""
        return cls.create('fisheye', sub_type, **kwargs)
    
    @classmethod
    def list_categories(cls) -> List[Dict[str, str]]:
        """列出所有相机类别"""
        return [
            {'name': k, 'description': v['description']}
            for k, v in cls._hierarchy.items()
        ]
    
    @classmethod
    def list_subtypes(cls, category: str) -> List[str]:
        """列出指定类别的所有子类型"""
        category = category.lower()
        if category not in cls._hierarchy:
            raise ValueError(f"Unknown category: '{category}'")
        return list(cls._hierarchy[category]['types'].keys())
    
    @classmethod
    def register_camera_type(
        cls,
        category: str,
        sub_type: str,
        camera_class: Type[Camera],
        is_default: bool = False
    ) -> None:
        """
        注册新的相机类型
        
        Args:
            category: 类别名称
            sub_type: 子类型名称
            camera_class: 相机类
            is_default: 是否设为该类别的默认类型
        """
        category = category.lower()
        sub_type = sub_type.lower()
        
        if category not in cls._hierarchy:
            cls._hierarchy[category] = {
                'default': sub_type,
                'description': f"{category} camera",
                'types': {}
            }
        
        cls._hierarchy[category]['types'][sub_type] = camera_class
        
        if is_default:
            cls._hierarchy[category]['default'] = sub_type


# 注册内置相机类型
CameraFactory.register_camera_type('pinhole', 'standard', PinholeCamera, is_default=True)
CameraFactory.register_camera_type('pinhole', 'wide_angle', PinholeCamera)
CameraFactory.register_camera_type('fisheye', 'kannala', KannalaBrandtCamera, is_default=True)
CameraFactory.register_camera_type('fisheye', 'ftheta', FThetaCamera)