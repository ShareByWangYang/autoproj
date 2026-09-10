#!/usr/bin/env python3
"""
简化的测试脚本
在项目根目录下直接运行测试
"""
import sys
from pathlib import Path

# 添加项目路径 - 注意：因为有双重嵌套，需要添加到外层 autoproj
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

print("=" * 80)
print("AUTO PROJ - 简化测试")
print("=" * 80)
print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}...")
print()

# 测试导入
print("1. 测试导入...")
try:
    from autoproj import CameraFactory
    print("  ✓ CameraFactory 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    print(f"  堆栈: {traceback.format_exc()}")
    sys.exit(1)

print()

# 创建相机测试
print("2. 测试创建相机...")
try:
    camera = CameraFactory.create_pinhole(
        width=1920,
        height=1080,
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0
    )
    print("  ✓ 相机创建成功")
except Exception as e:
    print(f"  ✗ 相机创建失败: {e}")
    import traceback
    print(f"  堆栈: {traceback.format_exc()}")
    sys.exit(1)

# 测试投影
print()
print("3. 测试投影...")
try:
    # 简单的测试点
    test_points = np.array([
        [0, 0, 10],
        [1, 0, 10],
        [-1, 0, 10]
    ], dtype=np.float64)
    
    pixels, valid = camera.project(test_points, pts_in_cam=True)
    print(f"  ✓ 投影完成")
    print(f"  - 像素坐标:\n{pixels}")
    print(f"  - 有效标记: {valid}")
    
except Exception as e:
    print(f"  ✗ 投影失败: {e}")
    import traceback
    print(f"  堆栈: {traceback.format_exc()}")
    sys.exit(1)

print()
print("=" * 80)
print("✓ 所有基础测试通过！")
print("=" * 80)
