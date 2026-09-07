#!/usr/bin/env python3
"""
测试 Projector 类的形状保持功能和视锥裁剪
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoproj import CameraFactory, Box3D, LineSet, Polygon3D
from autoproj.projection import Projector

print("=" * 80)
print("测试 Projector 类的形状保持功能和视锥裁剪")
print("=" * 80)

# 创建相机
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000.0, fy=1000.0,
    cx=960.0, cy=540.0
)

projector = Projector(camera)

# 测试1: project_points
print("\n【测试1】project_points")
print("-" * 60)
points = np.array([
    [0, 0, 10, 255, 1234567890.123],
    [1, 0, 10, 128, 1234567890.124],
])
print(f"输入形状: {points.shape}")
result, valid = projector.project_points(points, pts_in_cam=True, preserve_extra=True)
print(f"输出形状: {result.shape}")
print(f"形状保持一致: {result.shape == points.shape}")
print(f"UV + depth:\n{result[:, :3]}")
print(f"intensity保留: {result[:, 3]}")
print(f"gpstime保留: {result[:, 4]}")
print(f"有效掩码: {valid}")

# 测试2: project_box - 使用 ndarray 输入
print("\n【测试2】project_box (ndarray)")
print("-" * 60)
box_corners = np.array([
    [-1, -0.5, 10], [1, -0.5, 10],
    [-1, 0.5, 10],  [1, 0.5, 10],
    [-1, -0.5, 12], [1, -0.5, 12],
    [-1, 0.5, 12],  [1, 0.5, 12],
])
print(f"输入形状: {box_corners.shape}")
result_dict = projector.project_box(box_corners, T_to_cam=np.eye(4))
print(f"返回键: {list(result_dict.keys())}")
print(f"corners形状: {result_dict['corners'].shape}")
print(f"valid掩码: {result_dict['valid']}")
print(f"裁剪后棱线数: {len(result_dict['edges'])}")
for i, edge in enumerate(result_dict['edges']):
    print(f"  棱{i}: {edge['pt1']} -> {edge['pt2']} (corner: {edge['pt1_is_corner']}, {edge['pt2_is_corner']})")

# 测试3: project_box - 使用 Box3D 对象
print("\n【测试3】project_box (Box3D 对象)")
print("-" * 60)
box_obj = Box3D(center=[0, 0, 11], size=[2, 1, 2])
result_dict2 = projector.project_box(box_obj, T_to_cam=np.eye(4))
print(f"返回键: {list(result_dict2.keys())}")
print(f"corners形状: {result_dict2['corners'].shape}")
print(f"valid掩码: {result_dict2['valid']}")
print(f"裁剪后棱线数: {len(result_dict2['edges'])}")
print(f"box对象: {type(result_dict2['box']).__name__}")

# 测试4: project_lines
print("\n【测试4】project_lines")
print("-" * 60)
lines = np.array([
    [[0, 0, 10], [1, 0, 10]],
    [[0, 0, 20], [0, 1, 20]],
])
print(f"输入形状: {lines.shape}")
segments = projector.project_lines(lines, T_to_cam=np.eye(4))
print(f"返回类型: List")
print(f"线段1: {segments[0]}")
print(f"线段2: {segments[1]}")

# 测试5: project_lines - 使用 LineSet
print("\n【测试5】project_lines (LineSet)")
print("-" * 60)
line_set = LineSet.from_segments([
    (np.array([0, 0, 10]), np.array([1, 0, 10])),
    (np.array([0, 0, 20]), np.array([0, 1, 20])),
])
segments2 = projector.project_lines(line_set, T_to_cam=np.eye(4))
print(f"线段1: {segments2[0]}")
print(f"线段2: {segments2[1]}")

# 测试6: project_polygon
print("\n【测试6】project_polygon")
print("-" * 60)
polygon = Polygon3D(
    vertices=np.array([[0, 0, 10], [5, 0, 10], [5, 5, 10], [0, 5, 10]]),
    is_closed=True
)
poly_result = projector.project_polygon(polygon, T_to_cam=np.eye(4))
print(f"返回键: {list(poly_result.keys())}")
print(f"vertices形状: {poly_result['vertices'].shape}")
print(f"valid掩码: {poly_result['valid']}")
print(f"裁剪后边数: {len(poly_result['edges'])}")

# 测试7: FrustumCuller 独立使用
print("\n【测试7】FrustumCuller 独立裁剪")
print("-" * 60)
from autoproj import FrustumCuller, FrustumType
culler = FrustumCuller.from_camera(camera, expansion_factor=1.0)

# 测试在FOV内的线段
p1 = np.array([0.0, 0.0, 10.0])
p2 = np.array([2.0, 1.0, 10.0])
clipped = culler.clip_line(p1, p2)
print(f"FOV内线段裁剪: {clipped is not None}")
if clipped:
    print(f"  裁剪后: {clipped[0]} -> {clipped[1]}")

# 测试跨越FOV边界的线段
p3 = np.array([0.0, 0.0, 10.0])
p4 = np.array([100.0, 50.0, 10.0])  # 远超FOV
clipped2 = culler.clip_line(p3, p4)
print(f"跨越FOV边界线段裁剪: {clipped2 is not None}")
if clipped2:
    print(f"  裁剪后: {clipped2[0]} -> {clipped2[1]}")

# 测试完全在FOV外的线段
# 使用 y 值远超 y_max 的线段，确保完全在视锥外
p5 = np.array([-100.0, 100.0, 5.0])
p6 = np.array([100.0, 100.0, 5.0])
clipped3 = culler.clip_line(p5, p6)
print(f"完全在FOV外线段裁剪: {clipped3} (应为 None)")

# 测试点裁剪
print(f"\n点裁剪测试:")
print(f"  [0,0,10] 在视锥内: {culler.is_point_inside(np.array([0, 0, 10]))}")
print(f"  [100,0,10] 在视锥内: {culler.is_point_inside(np.array([100, 0, 10]))}")
print(f"  [0,0,0.01] 在视锥内: {culler.is_point_inside(np.array([0, 0, 0.01]))}")
_, mask = culler.cull_points(np.array([[0, 0, 10], [100, 0, 10], [0, 0, 5]]))
print(f"  批量裁剪掩码: {mask}")

print("\n" + "=" * 80)
print("测试完成！")