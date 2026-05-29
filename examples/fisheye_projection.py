#!/usr/bin/env python3
"""
AutoProj 鱼眼相机投影示例

演示如何创建鱼眼相机并投影3D点云
"""

import numpy as np
from autoproj import CameraFactory, Projector


def main():
    print("=== AutoProj 鱼眼相机投影示例 ===\n")
    
    # 创建Kannala-Brandt鱼眼相机
    print("1. 创建Kannala-Brandt鱼眼相机")
    kannala_camera = CameraFactory.create_fisheye(
        sub_type='kannala',
        width=1920, height=1080,
        fx=500, fy=500, cx=960, cy=540,
        k1=0.1, k2=0.05, k3=0.01, k4=0.005
    )
    print(f"   相机类型: {type(kannala_camera).__name__}")
    print(f"   畸变系数: k1={kannala_camera.k1}, k2={kannala_camera.k2}")
    print()
    
    # 创建F-Theta鱼眼相机
    print("2. 创建F-Theta鱼眼相机")
    ftheta_camera = CameraFactory.create_fisheye(
        sub_type='ftheta',
        width=1920, height=1080,
        fw_poly=[0, 500, 50],
        cx=960, cy=540
    )
    print(f"   相机类型: {type(ftheta_camera).__name__}")
    print(f"   焦距多项式: {ftheta_camera.fw_poly}")
    print()
    
    # 使用Kannala-Brandt相机进行投影
    print("3. 使用鱼眼相机投影点云")
    projector = Projector(kannala_camera)
    
    # 生成广角点云（覆盖更大范围）
    np.random.seed(42)
    n_points = 200
    points_3d = np.random.uniform(-20, 20, (n_points, 3))
    points_3d[:, 2] = np.random.uniform(1, 50, n_points)
    
    result = projector.project_points(points_3d, pts_in_cam=True)
    print(f"   总点数: {result['num_points']}")
    print(f"   有效投影点: {result['num_valid']}")
    print(f"   有效率: {result['num_valid']/result['num_points']*100:.1f}%")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
