import numpy as np
from typing import Optional, Dict, Any
from .camera import Camera


class Projector:
    """
    投影器类，提供统一的3D到2D投影接口
    
    Args:
        camera: 相机对象
        frustum_expansion: 视锥体扩展系数（默认1.0，表示不扩展）
    """
    
    def __init__(self, camera: Camera, frustum_expansion: float = 1.0):
        self.camera = camera
        self.frustum_expansion = frustum_expansion
    
    def project_points(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Dict[str, Any]:
        """
        投影点云
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            T_to_cam: 外参变换矩阵（4x4）
            pts_in_cam: 点云是否已在相机坐标系中
        
        Returns:
            投影结果字典，包含像素坐标和有效掩码
        """
        pixels, valid = self.camera.project(points_3d, T_to_cam, pts_in_cam)
        
        return {
            'pixels': pixels,
            'valid': valid,
            'num_points': len(points_3d),
            'num_valid': int(np.sum(valid))
        }
    
    def project_box(
        self,
        box_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        投影3D包围盒
        
        Args:
            box_3d: 3D包围盒的8个角点，形状为(8, 3)
            T_to_cam: 外参变换矩阵
        
        Returns:
            投影结果字典
        """
        pixels, valid = self.camera.project(box_3d, T_to_cam)
        
        return {
            'corners': pixels,
            'valid': valid,
            'all_valid': bool(np.all(valid))
        }
    
    def project_lines(
        self,
        lines: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        投影线段集合
        
        Args:
            lines: 线段端点，形状为(N, 2, 3)，每个线段包含两个端点
            T_to_cam: 外参变换矩阵
        
        Returns:
            投影结果字典
        """
        n_lines = lines.shape[0]
        start_points = lines[:, 0, :]
        end_points = lines[:, 1, :]
        
        start_pixels, start_valid = self.camera.project(start_points, T_to_cam)
        end_pixels, end_valid = self.camera.project(end_points, T_to_cam)
        
        # 两个端点都有效才算有效线段
        line_valid = start_valid & end_valid
        
        return {
            'start_pixels': start_pixels,
            'end_pixels': end_pixels,
            'valid': line_valid,
            'num_lines': n_lines,
            'num_valid': int(np.sum(line_valid))
        }