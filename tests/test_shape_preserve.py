#!/usr/bin/env python3
"""
测试投影方法形状保持功能
"""
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoproj import CameraFactory

print("=" * 80)
print("测试投影形状保持功能")
print("=" * 80)

# 创建相机
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000.0, fy=1000.0,
    cx=960.0, cy=540.0
)

# 测试1: 基本XYZ点云
print("\n【测试1】基本XYZ点云 (N, 3)")
points_3d = np.array([
    [0, 0, 10],
    [1, 0, 10],
    [0, 1, 10],
])
# preserve_extra=True 保留 depth 和额外列
result, valid = camera.project(points_3d, pts_in_cam=True, preserve_extra=True)
print(f"输入形状: {points_3d.shape}")
print(f"输出形状: {result.shape}")
print(f"结果:\n{result}")
print(f"有效掩码: {valid}")

# 测试2: XYZ+intensity
print("\n【测试2】XYZ+intensity (N, 4)")
points_4d = np.array([
    [0, 0, 10, 255],
    [1, 0, 10, 128],
    [0, 1, 10, 64],
])
result, valid = camera.project(points_4d, pts_in_cam=True, preserve_extra=True)
print(f"输入形状: {points_4d.shape}")
print(f"输出形状: {result.shape}")
print(f"结果:\n{result}")
print(f"intensity保留: {result[:, 3]}")

# 测试3: XYZ+intensity+gpstime
print("\n【测试3】XYZ+intensity+gpstime (N, 5)")
points_5d = np.array([
    [0, 0, 10, 255, 1234567890.123],
    [1, 0, 10, 128, 1234567890.124],
    [0, 1, 10, 64, 1234567890.125],
])
result, valid = camera.project(points_5d, pts_in_cam=True, preserve_extra=True)
print(f"输入形状: {points_5d.shape}")
print(f"输出形状: {result.shape}")
print(f"UV + depth:\n{result[:, :3]}")
print(f"intensity: {result[:, 3]}")
print(f"gpstime: {result[:, 4]}")

# 测试4: 无效点处理
print("\n【测试4】无效点处理")
invalid_points = np.array([
    [0, 0, -1, 255],  # 负深度
    [100, 0, 1, 128],  # 超出FOV
    [0, 0, 10, 64],   # 有效点
])
result, valid = camera.project(invalid_points, pts_in_cam=True, preserve_extra=True)
print(f"结果:\n{result}")
print(f"有效掩码: {valid}")
print(f"无效点UV是否为-1: {result[~valid, :2]}")

# 测试5: 深度信息
print("\n【测试5】深度信息")
depth_points = np.array([
    [0, 0, 5, 255],
    [0, 0, 10, 128],
    [0, 0, 20, 64],
])
result, valid = camera.project(depth_points, pts_in_cam=True, preserve_extra=True)
print(f"输入深度: {depth_points[:, 2]}")
print(f"输出深度: {result[:, 2]}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
