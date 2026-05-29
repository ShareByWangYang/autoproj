#!/usr/bin/env python3
"""
AutoProj 配置文件加载示例

演示如何从YAML/JSON配置文件加载相机参数
"""

import tempfile
import os
from autoproj import load_camera, save_camera_config, CameraFactory


def main():
    print("=== AutoProj 配置文件加载示例 ===\n")
    
    # 创建临时配置文件
    print("1. 创建临时配置文件")
    config_content = """
type: pinhole
width: 1920
height: 1080
fx: 1000
fy: 1000
cx: 960
cy: 540
dist_coeffs: [0.1, -0.05, 0.01, -0.01]
near_z: 0.1
far_z: 1000.0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    print(f"   配置文件: {config_path}")
    print()
    
    # 从配置文件加载相机
    print("2. 从配置文件加载相机")
    camera = load_camera(config_path)
    print(f"   相机类型: {type(camera).__name__}")
    print(f"   分辨率: {camera.width}x{camera.height}")
    print(f"   焦距: fx={camera.fx}, fy={camera.fy}")
    print(f"   主点: cx={camera.cx}, cy={camera.cy}")
    print()
    
    # 保存相机配置到新文件
    print("3. 保存相机配置到新文件")
    output_path = os.path.join(tempfile.gettempdir(), 'camera_output.yaml')
    save_camera_config(camera, output_path)
    print(f"   保存到: {output_path}")
    
    # 验证保存的配置
    loaded_camera = load_camera(output_path)
    print(f"   验证: 加载的相机分辨率 = {loaded_camera.width}x{loaded_camera.height}")
    print()
    
    # 使用字典创建相机
    print("4. 从字典创建相机")
    config_dict = {
        'type': 'fisheye:kannala',
        'width': 1280,
        'height': 720,
        'fx': 400,
        'fy': 400,
        'k1': 0.15
    }
    from autoproj import load_camera_from_dict
    fisheye_camera = load_camera_from_dict(config_dict)
    print(f"   相机类型: {type(fisheye_camera).__name__}")
    print(f"   分辨率: {fisheye_camera.width}x{fisheye_camera.height}")
    
    # 清理临时文件
    os.unlink(config_path)
    if os.path.exists(output_path):
        os.unlink(output_path)
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
