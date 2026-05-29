#!/usr/bin/env python3
"""
AutoProj 基础投影示例

演示如何创建相机并投影3D点云到图像平面
"""

import numpy as np
from autoproj import CameraFactory, Projector


def main():
    print("=== AutoProj 基础投影示例 ===\n")
    
    # 创建针孔相机
    print("1. 创建针孔相机")
    camera = CameraFactory.create_pinhole(
        width=1920, height=1080,
        fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[0.1, -0.05, 0.01, -0.01]
    )
    print(f"   相机类型: {type(camera).__name__}")
    print(f"   分辨率: {camera.width}x{camera.height}")
    print(f"   焦距: fx={camera.fx}, fy={camera.fy}")
    print(f"   FOV: {camera.fov_h:.1f}° x {camera.fov_v:.1f}°\n")
    
    # 创建投影器
    print("2. 创建投影器")
    projector = Projector(camera)
    print("   投影器创建成功\n")
    
    # 生成测试点云（相机坐标系）
    print("3. 生成测试点云")
    np.random.seed(42)
    n_points = 100
    points_3d = np.random.uniform(-5, 5, (n_points, 3))
    points_3d[:, 2] = np.random.uniform(5, 50, n_points)  # z方向5-50米
    print(f"   生成 {n_points} 个3D点\n")
    
    # 投影点云
    print("4. 执行投影")
    result = projector.project_points(points_3d, pts_in_cam=True)
    print(f"   总点数: {result['num_points']}")
    print(f"   有效投影点: {result['num_valid']}")
    print(f"   有效率: {result['num_valid']/result['num_points']*100:.1f}%\n")
    
    # 显示部分结果
    print("5. 投影结果示例")
    valid_indices = np.where(result['valid'])[0][:5]
    for i in valid_indices:
        pixel = result['pixels'][i]
        point = points_3d[i]
        print(f"   点 ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) -> 像素 ({pixel[0]}, {pixel[1]})")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
