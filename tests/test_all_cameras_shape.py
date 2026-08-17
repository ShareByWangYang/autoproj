#!/usr/bin/env python3
"""
测试所有相机类型的形状保持功能
"""
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoproj import CameraFactory
from autoproj.projection import Projector

print("=" * 80)
print("测试所有相机类型的形状保持功能")
print("=" * 80)

# 创建不同类型的相机
cameras = [
    CameraFactory.create_pinhole(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540),
    CameraFactory.create_fisheye(sub_type='kannala', width=1920, height=1080, fx=500, fy=500, cx=960, cy=540, k1=0.1, k2=0.05),
    CameraFactory.create_fisheye(sub_type='ftheta', width=1920, height=1080, fw_poly=[0, 500, 50], cx=960, cy=540),
]

camera_names = ['Pinhole', 'Kannala-Brandt', 'F-Theta']

# 测试数据
test_points = np.array([
    [0, 0, 10, 255, 1234567890.123],
    [1, 0, 10, 128, 1234567890.124],
    [0, 1, 10, 64, 1234567890.125],
])

for camera, name in zip(cameras, camera_names):
    print(f"\n【测试 {name} 相机】")
    print("-" * 60)
    
    # 测试基本投影（preserve_extra=True 保留 depth 和额外列）
    result, valid = camera.project(test_points, pts_in_cam=True, preserve_extra=True)
    print(f"输入形状: {test_points.shape}")
    print(f"输出形状: {result.shape}")
    print(f"UV + depth:\n{result[:, :3]}")
    print(f"intensity保留: {result[:, 3]}")
    print(f"gpstime保留: {result[:, 4]}")
    print(f"有效掩码: {valid}")
    
    # 测试 Projector
    projector = Projector(camera)
    proj_result, proj_valid = projector.project_points(test_points, pts_in_cam=True, preserve_extra=True)
    print(f"\nProjector结果:")
    print(f"  result形状: {proj_result.shape}")
    print(f"  pixels形状: {proj_result[:, :2].shape}")
    print(f"  depths形状: {proj_result[:, 2].shape}")

print("\n" + "=" * 80)
print("所有测试完成！")
print("=" * 80)
