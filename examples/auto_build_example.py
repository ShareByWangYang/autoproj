#!/usr/bin/env python3
"""
AutoProj C++ 后端自动构建示例

演示如何使用自动构建和优雅降级功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoproj import CameraFactory
from autoproj.backends import BackendSelector, try_build_cpp_backend
import numpy as np

print("=" * 80)
print("AutoProj 自动构建和优雅降级示例")
print("=" * 80)

# 场景 1: 自动选择最佳后端（推荐）
print("\n【场景 1】自动选择最佳后端")
print("-" * 60)

# 这会自动：
# 1. 检查 CUDA 是否可用
# 2. 如果不可用，检查 C++ 后端（尝试自动构建）
# 3. 如果都不可用，回退到 NumPy
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000.0, fy=1000.0,
    cx=960.0, cy=540.0
)

print(f"✓ 使用后端: {type(camera.backend).__name__}")

# 场景 2: 显式使用 BackendSelector
print("\n【场景 2】显式使用 BackendSelector")
print("-" * 60)

# 重置状态（可选）
BackendSelector.reset()

# 查看可用后端
available = BackendSelector.available_backends(auto_build=True)
print(f"可用后端: {available}")

# 自动选择
backend = BackendSelector.select()
print(f"自动选择: {type(backend).__name__}")

# 场景 3: 尝试特定后端，自动降级
print("\n【场景 3】尝试特定后端，自动降级")
print("-" * 60)

try:
    # 尝试使用 C++ 后端，如果不可用自动降级
    backend = BackendSelector.select(
        'cpp', 
        auto_build=True, 
        auto_fallback=True
    )
    print(f"最终使用: {type(backend).__name__}")
    
    # 使用此后端创建相机
    camera = CameraFactory.create_pinhole(
        width=1920, height=1080,
        fx=1000.0, fy=1000.0,
        cx=960.0, cy=540.0,
        backend=backend
    )
    print("✓ 相机创建成功")
    
except ValueError as e:
    print(f"错误: {e}")

# 场景 4: 禁用自动构建（仅使用已有的后端）
print("\n【场景 4】禁用自动构建")
print("-" * 60)

# 不尝试构建，只使用已有的后端
backend = BackendSelector.select(auto_build=False)
print(f"使用后端: {type(backend).__name__}")

# 场景 5: 手动构建（高级用户）
print("\n【场景 5】手动构建 C++ 后端")
print("-" * 60)

print("尝试手动构建 C++ 后端...")
success = try_build_cpp_backend(auto_install_deps=True)

if success:
    print("✓ C++ 后端构建成功！")
    # 现在可以使用 C++ 后端
    backend = BackendSelector.select('cpp')
    print(f"使用后端: {type(backend).__name__}")
else:
    print("✗ C++ 后端构建失败，使用 NumPy")
    backend = BackendSelector.select('numpy')

# 测试投影
print("\n【测试】投影 10 万个点")
print("-" * 60)

camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000.0, fy=1000.0,
    cx=960.0, cy=540.0,
    backend=backend
)

# 生成测试点
points = np.random.randn(100000, 3).astype(np.float64)
points[:, 2] = np.abs(points[:, 2]) + 5  # 确保在前方

import time
start = time.time()
pixels, valid = camera.project(points, pts_in_cam=True)
elapsed = time.time() - start

print(f"✓ 投影完成: {np.sum(valid)} / {len(valid)} 个点有效")
print(f"✓ 耗时: {elapsed:.3f}s")
print(f"✓ 吞吐量: {(len(points) / elapsed / 1e6):.2f} Mpts/s")

print("\n" + "=" * 80)
print("示例完成！")
print("=" * 80)
