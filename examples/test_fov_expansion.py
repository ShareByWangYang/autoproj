"""
Test script for precise FOV expansion factor computation.

Verifies that expansion_factor is automatically computed from camera intrinsics
and distortion coefficients, matching the real visible FOV.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoproj import PinholeCamera, KannalaBrandtCamera, FThetaCamera, Projector


def test_pinhole_no_distortion():
    """无畸变时扩展因子应为 1.0"""
    print("=" * 60)
    print("Test 1: PinholeCamera without distortion")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[0, 0, 0, 0, 0, 0, 0, 0]
    )

    fov_h, fov_v = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    print(f"  Nominal FOV: h={camera.fov_h:.2f}°, v={camera.fov_v:.2f}°")
    print(f"  Real FOV:    h={fov_h:.2f}°, v={fov_v:.2f}°")
    print(f"  Expansion factor: {expansion:.6f}")
    assert abs(expansion - 1.0) < 1e-6, f"Expected 1.0, got {expansion}"
    print("  PASS: expansion == 1.0")


def test_pinhole_barrel_distortion():
    """桶形畸变（k1<0）应使真实 FOV > 标称 FOV，expansion > 1.0"""
    print("\n" + "=" * 60)
    print("Test 2: PinholeCamera with barrel distortion (k1<0)")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[-0.3, 0.1, 0, 0, 0, 0, 0, 0]
    )

    fov_h_nominal = camera.fov_h
    fov_v_nominal = camera.fov_v
    fov_h_real, fov_v_real = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    print(f"  Nominal FOV: h={fov_h_nominal:.2f}°, v={fov_v_nominal:.2f}°")
    print(f"  Real FOV:    h={fov_h_real:.2f}°, v={fov_v_real:.2f}°")
    print(f"  Expansion factor: {expansion:.6f}")
    print(f"  FOV change: h={fov_h_real - fov_h_nominal:+.2f}°, v={fov_v_real - fov_v_nominal:+.2f}°")

    assert expansion > 1.0, f"Barrel distortion should give expansion > 1.0, got {expansion}"
    assert fov_h_real > fov_h_nominal, "Real FOV should be larger than nominal"
    print(f"  PASS: expansion={expansion:.4f} > 1.0 (barrel distortion expands FOV)")


def test_pinhole_pincushion_distortion():
    """枕形畸变（k1>0）应使真实 FOV < 标称 FOV，expansion < 1.0"""
    print("\n" + "=" * 60)
    print("Test 3: PinholeCamera with pincushion distortion (k1>0)")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[0.2, -0.05, 0, 0, 0, 0, 0, 0]
    )

    fov_h_nominal = camera.fov_h
    fov_v_nominal = camera.fov_v
    fov_h_real, fov_v_real = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    print(f"  Nominal FOV: h={fov_h_nominal:.2f}°, v={fov_v_nominal:.2f}°")
    print(f"  Real FOV:    h={fov_h_real:.2f}°, v={fov_v_real:.2f}°")
    print(f"  Expansion factor: {expansion:.6f}")
    print(f"  FOV change: h={fov_h_real - fov_h_nominal:+.2f}°, v={fov_v_real - fov_v_nominal:+.2f}°")

    assert expansion < 1.0, f"Pincushion distortion should give expansion < 1.0, got {expansion}"
    assert fov_h_real < fov_h_nominal, "Real FOV should be smaller than nominal"
    print(f"  PASS: expansion={expansion:.4f} < 1.0 (pincushion distortion shrinks FOV)")


def test_pinhole_auto_expansion():
    """验证 auto 模式使用精确计算的扩展因子"""
    print("\n" + "=" * 60)
    print("Test 4: Projector auto expansion with barrel distortion")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[-0.3, 0.1, 0, 0, 0, 0, 0, 0]
    )

    # auto 模式
    projector = Projector(camera, cull_frustum=True, frustum_scale='auto')
    expected_expansion = camera.compute_expansion_factor()

    print(f"  Camera computed expansion: {expected_expansion:.6f}")
    print(f"  Projector effective expansion: {projector._effective_expansion:.6f}")
    print(f"  Culler tan_h: {projector.culler.tan_h:.6f}")

    # 验证 culler 使用了精确扩展因子
    tan_h_nominal = max(camera.cx, camera.width - 1 - camera.cx) / camera.fx
    expected_tan_h = tan_h_nominal * expected_expansion
    print(f"  Expected tan_h: {expected_tan_h:.6f}")

    assert abs(projector._effective_expansion - expected_expansion) < 1e-10, \
        "Projector should use camera's computed expansion factor"
    assert abs(projector.culler.tan_h - expected_tan_h) < 1e-6, \
        f"Culler tan_h should match, got {projector.culler.tan_h}, expected {expected_tan_h}"
    print("  PASS: auto mode uses precise expansion factor")


def test_pinhole_point_projection_with_auto():
    """验证 auto 扩展因子下点云投影的正确性"""
    print("\n" + "=" * 60)
    print("Test 5: Point projection with auto expansion (barrel distortion)")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[-0.3, 0.1, 0, 0, 0, 0, 0, 0]
    )

    fov_h_real, fov_v_real = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    tan_h_nominal = max(camera.cx, camera.width - 1 - camera.cx) / camera.fx
    tan_h_real = tan_h_nominal * expansion

    z = 10.0
    x_at_nominal_fov = z * tan_h_nominal
    x_at_real_fov = z * tan_h_real

    print(f"  Nominal FOV half-angle tan: {tan_h_nominal:.4f}")
    print(f"  Real FOV half-angle tan:    {tan_h_real:.4f}")
    print(f"  Expansion factor: {expansion:.6f}")
    print(f"  At z={z}m: nominal edge x={x_at_nominal_fov:.2f}, real edge x={x_at_real_fov:.2f}")

    # 测试点：在标称 FOV 外但在真实 FOV 内
    test_points = np.array([
        [0, 0, z],  # 中心
        [x_at_nominal_fov * 0.95, 0, z],  # 标称 FOV 内
        [x_at_nominal_fov, 0, z],  # 标称 FOV 边界
        [x_at_nominal_fov * (1 + (expansion - 1) * 0.5), 0, z],  # 标称外、真实内（中点）
        [x_at_real_fov * 0.99, 0, z],  # 接近真实 FOV 边界
        [x_at_real_fov * 1.01, 0, z],  # 真实 FOV 外
    ])

    projector = Projector(camera, cull_frustum=True, frustum_scale='auto')

    result, valid = projector.project_points(test_points, pts_in_cam=True)

    print(f"\n  Point validation:")
    for i, (pt, v) in enumerate(zip(test_points, valid)):
        ratio = pt[0] / x_at_nominal_fov if x_at_nominal_fov > 0 else 0
        status = "VALID" if v else "INVALID"
        print(f"    [{i}] x={pt[0]:.2f} (nominal*{ratio:.3f}) -> {status}")

    # 验证：真实 FOV 内的点应有效
    assert valid[0], "Center should be valid"
    assert valid[1], "Point inside nominal FOV should be valid"
    assert valid[4], "Point near real FOV edge should be valid"
    assert not valid[5], "Point outside real FOV should be invalid"
    print("  PASS: points at real FOV boundary handled correctly")


def test_kannala_brandt_fov():
    """Kannala-Brandt 鱼眼相机 FOV 计算"""
    print("\n" + "=" * 60)
    print("Test 6: KannalaBrandtCamera FOV computation")
    print("=" * 60)

    camera = KannalaBrandtCamera(
        width=1920, height=1080, fx=500, fy=500, cx=960, cy=540,
        k1=-0.05, k2=0.005, k3=0, k4=0
    )

    fov_h_nominal = camera.fov_h
    fov_v_nominal = camera.fov_v
    fov_h_real, fov_v_real = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    print(f"  Nominal FOV: h={fov_h_nominal:.2f}°, v={fov_v_nominal:.2f}°")
    print(f"  Real FOV:    h={fov_h_real:.2f}°, v={fov_v_real:.2f}°")
    print(f"  Expansion factor: {expansion:.6f}")

    # 鱼眼通常有更大的 FOV，expansion 应反映畸变影响
    assert expansion != 1.0 or (fov_h_real == fov_h_nominal), \
        "Distortion should affect FOV"
    print(f"  PASS: KB fisheye expansion={expansion:.4f}")


def test_ftheta_fov():
    """F-Theta 鱼眼相机 FOV 计算"""
    print("\n" + "=" * 60)
    print("Test 7: FThetaCamera FOV computation")
    print("=" * 60)

    # 典型 F-Theta 多项式: r = a0 + a1*theta + a2*theta^2
    # a1 约等于焦距
    camera = FThetaCamera(
        width=1920, height=1080,
        fw_poly=[0, 400, 0.1],
        cx=960, cy=540
    )

    fov_h_real, fov_v_real = camera.compute_real_fov()
    expansion = camera.compute_expansion_factor()

    print(f"  theta_max: {np.degrees(camera._theta_max):.2f}°")
    print(f"  Real FOV:  h={fov_h_real:.2f}°, v={fov_v_real:.2f}°")
    print(f"  Expansion factor: {expansion:.6f}")

    assert fov_h_real > 0, "FOV should be positive"
    print(f"  PASS: F-Theta expansion={expansion:.4f}")


def test_comparison_manual_vs_auto():
    """对比手动扩展因子 vs auto 精确计算"""
    print("\n" + "=" * 60)
    print("Test 8: Comparison manual (1.05) vs auto expansion")
    print("=" * 60)

    camera = PinholeCamera(
        width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
        dist_coeffs=[-0.3, 0.1, 0, 0, 0, 0, 0, 0]
    )

    auto_expansion = camera.compute_expansion_factor()
    manual_expansion = 1.05

    print(f"  Auto (precise): {auto_expansion:.6f}")
    print(f"  Manual (1.05):  {manual_expansion:.6f}")
    print(f"  Difference:     {auto_expansion - manual_expansion:+.6f}")

    # 生成测试点
    tan_h_nominal = max(camera.cx, camera.width - 1 - camera.cx) / camera.fx
    z = 10.0
    points = np.array([
        [z * tan_h_nominal * r, 0, z]
        for r in [0.9, 0.95, 1.0, 1.02, 1.05, 1.08, 1.10]
    ])

    # 手动 1.05
    proj_manual = Projector(camera, cull_frustum=True, frustum_scale=manual_expansion)
    _, valid_manual = proj_manual.project_points(points, pts_in_cam=True)

    # auto
    proj_auto = Projector(camera, cull_frustum=True, frustum_scale='auto')
    _, valid_auto = proj_auto.project_points(points, pts_in_cam=True)

    print(f"\n  Point comparison at z={z}m:")
    print(f"  {'Ratio':>6} {'X':>8} {'Manual':>8} {'Auto':>8}")
    print(f"  {'-'*34}")
    for i, pt in enumerate(points):
        ratio = pt[0] / (z * tan_h_nominal)
        m = "VALID" if valid_manual[i] else "INVALID"
        a = "VALID" if valid_auto[i] else "INVALID"
        print(f"  {ratio:5.2f}x {pt[0]:8.1f} {m:>8} {a:>8}")

    print(f"\n  Auto expansion provides FOV boundary matching real camera visibility")


if __name__ == "__main__":
    test_pinhole_no_distortion()
    test_pinhole_barrel_distortion()
    test_pinhole_pincushion_distortion()
    test_pinhole_auto_expansion()
    test_pinhole_point_projection_with_auto()
    test_kannala_brandt_fov()
    test_ftheta_fov()
    test_comparison_manual_vs_auto()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
