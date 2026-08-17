#!/usr/bin/env python3
"""
测试 AutoProj C++ 后端自动构建功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("AutoProj C++ 后端自动构建测试")
print("=" * 80)

# 1. 测试构建工具
print("\n1. 测试构建工具...")
from autoproj.backends import BuildEnvironment, AutoBuilder, try_build_cpp_backend

env = BuildEnvironment()
print(f"   C++ 编译器: {env.check_cxx_compiler()}")
print(f"   pybind11: {env.check_pybind11()}")
print(f"   写入权限: {env.has_write_permission(project_root)}")

# 2. 测试 BackendSelector
print("\n2. 测试 BackendSelector...")
from autoproj.backends import BackendSelector

# 重置后端状态
BackendSelector.reset()

# 测试可用后端
available = BackendSelector.available_backends(auto_build=True)
print(f"   可用后端: {available}")

# 测试自动选择
backend = BackendSelector.select(auto_build=True, auto_fallback=True)
print(f"   自动选择: {type(backend).__name__}")

# 3. 测试完整流程
print("\n3. 测试完整投影流程...")
from autoproj import CameraFactory
import numpy as np

camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000.0, fy=1000.0,
    cx=960.0, cy=540.0
)

# 测试投影
points = np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10]], dtype=np.float64)
pixels, valid = camera.project(points, pts_in_cam=True)

print(f"   投影成功:")
for i, (p, v) in enumerate(zip(pixels, valid)):
    print(f"   点{i}: {p}, valid={v}")

# 4. 测试性能
print("\n4. 性能测试...")
large_points = np.random.randn(100000, 3).astype(np.float64)
large_points[:, 2] = np.abs(large_points[:, 2]) + 5

import time
start = time.time()
for _ in range(10):
    camera.project(large_points, pts_in_cam=True)
elapsed = time.time() - start

print(f"   投影 100 万点耗时: {elapsed:.3f}s")
print(f"   吞吐量: {(100000 * 10 / elapsed / 1e6):.2f} Mpts/s")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
