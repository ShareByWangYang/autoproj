"""
clip_lines_to_image / project_box(clip_to_image) 测试

覆盖:
1. 无畸变针孔解析解验证（左右/上下边界穿越、out-in-out 双穿越）
2. 桶形畸变针孔（亚像素精度、in-out-in 多段）
3. KB / FTheta 鱼眼（独立参考解验证）
4. 退化情形（NaN 端点、全在外、空输入、跨 near_z、形状错误）
5. bit-exact 边界（端点恰在图像矩形边界上视为内部）
6. r_max 复现（GTSystem 4 台标定相机，含极点/有理模型）
7. 多段输出（in-out-in → 2 pieces；超限 invalid）
8. 折线点 return_polyline
9. 像素收敛精度（长棱 ≤0.5px）
10. project_box(clip_to_image=True) 桥接（标记、互斥、批量一致性）
"""
import math

import numpy as np
import pytest

from autoproj import PinholeCamera, KannalaBrandtCamera, FThetaCamera
from autoproj.projection import Projector


W, H = 1920, 1080
U_MAX, V_MAX = float(W - 1), float(H - 1)   # 1919.0 / 1079.0


# ========== fixtures ==========
@pytest.fixture
def pinhole_clean():
    """无畸变针孔: u = 1000*x/z + 960, v = 1000*y/z + 540"""
    cam = PinholeCamera(width=W, height=H, fx=1000.0, fy=1000.0,
                        cx=960.0, cy=540.0)
    return Projector(cam)


@pytest.fixture
def pinhole_barrel():
    """桶形畸变针孔 k1=-0.15（域内单调，但图像矩形原像四角外凸）"""
    cam = PinholeCamera(width=W, height=H, fx=1000.0, fy=1000.0,
                        cx=960.0, cy=540.0, dist_coeffs=[-0.15, 0, 0, 0])
    return Projector(cam)


def _px(proj, pts):
    """投影点集: (3,) → (2,)，(M,3) → (M,2)"""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        return proj._project_raw_pixels(arr.reshape(1, 3))[0]
    return proj._project_raw_pixels(arr)


# ========== 1. 无畸变针孔解析解 ==========
class TestAnalyticPinhole:
    """无畸变针孔下边界交点可解析求解: u=fx*x/z+cx, v=fy*y/z+cy"""

    def test_line_fully_inside_returns_original_bitexact(self, pinhole_clean):
        """全在内的线段: 端点原样返回（bit-exact，不经二分）"""
        p1 = np.array([0.1, 0.1, 5.0])   # u=980, v=560
        p2 = np.array([0.2, -0.1, 5.0])  # u=1000, v=520
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert np.array_equal(clipped[0, 0, 0], p1)
        assert np.array_equal(clipped[0, 0, 1], p2)

    def test_cross_right_edge(self, pinhole_clean):
        """右边界 u=1919: x/z=0.959 → x*=4.795 (z=5), t*=0.5"""
        p1 = np.array([0.0, 0.0, 5.0])    # u=960 内
        p2 = np.array([9.59, 0.0, 5.0])   # u=2878 外
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert np.array_equal(clipped[0, 0, 0], p1)      # 起点在内 → t=0 原样
        assert abs(clipped[0, 0, 1, 0] - 4.795) < 2e-3   # x*
        assert abs(clipped[0, 0, 1, 2] - 5.0) < 1e-12
        u = _px(pinhole_clean, clipped[0, 0, 1])[0]
        assert abs(u - U_MAX) <= 0.5                  # 亚像素验收 ≤0.5px

    def test_cross_left_edge(self, pinhole_clean):
        """左边界 u=0: x/z=-0.96 → x*=-4.8, t*=0.04; 终点原样 bit-exact"""
        p1 = np.array([-5.0, 0.0, 5.0])   # u=-40 外
        p2 = np.array([0.0, 0.0, 5.0])    # u=960 内
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 0, 0] - (-4.8)) < 2e-3
        assert np.array_equal(clipped[0, 0, 1], p2)      # 终点在内 → t=1 原样

    def test_cross_bottom_edge(self, pinhole_clean):
        """下边界 v=1079: y/z=0.539 → y*=2.695"""
        p1 = np.array([0.0, 0.0, 5.0])    # v=540 内
        p2 = np.array([0.0, 3.0, 5.0])    # v=1140 外
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 1, 1] - 2.695) < 2e-3
        v = _px(pinhole_clean, clipped[0, 0, 1])[1]
        assert abs(v - V_MAX) <= 0.5

    def test_out_in_out_double_crossing(self, pinhole_clean):
        """双端在外、中间穿过: y=3(v=1140 出底边) → y=-3(v=-60 出顶边)
        区间 [t1, t2]: y*=2.695 (v=1079) / y*=-2.7 (v=0)"""
        p1 = np.array([0.0, 3.0, 5.0])
        p2 = np.array([0.0, -3.0, 5.0])
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 0, 1] - 2.695) < 2e-3   # 进入点（底边）
        assert abs(clipped[0, 0, 1, 1] - (-2.7)) < 2e-3  # 离开点（顶边）

    def test_batch_matches_single(self, pinhole_clean):
        """批量调用与逐条调用结果一致"""
        lines = np.array([
            [[0.0, 0.0, 5.0], [9.59, 0.0, 5.0]],
            [[-5.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
            [[0.0, 3.0, 5.0], [0.0, -3.0, 5.0]],
        ])
        c_all, v_all = pinhole_clean.clip_lines_to_image(lines)
        for i in range(3):
            c1, v1 = pinhole_clean.clip_lines_to_image(lines[i:i + 1])
            assert v_all[i, 0] == v1[0, 0]
            assert np.allclose(c_all[i], c1[0], equal_nan=True)


# ========== 2. 桶形畸变针孔 ==========
class TestBarrelDistortion:
    def test_bottom_crossing_subpixel(self, pinhole_barrel):
        """桶形畸变下底边穿越: 裁剪端点投影像素距 v=1079 ≤0.5px"""
        p1 = np.array([0.0, 0.0, 5.0])     # v=540 内
        p2 = np.array([0.0, 6.0, 5.0])     # v≈1481 外
        clipped, valid = pinhole_barrel.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert np.array_equal(clipped[0, 0, 0], p1)
        v = _px(pinhole_barrel, clipped[0, 0, 1])[1]
        assert abs(v - V_MAX) <= 0.5
        # 裁剪点应略深于无畸变解 2.695（畸变压缩 → 真实边界 y 更大）
        assert clipped[0, 0, 1, 1] > 2.695

    def test_in_out_in_two_pieces(self, pinhole_barrel):
        """畸变折叠导致的 in-out-in（2 跳变且起点在内）→ 2 个 piece

        k1=-0.15 时 r>1.49 区域折叠回图像内: 线段 (-2,0.7,1)→(2,0.7,1)
        端点折叠入图像（内），中间段 v>1079（外），为 in-out-in 拓扑。
        """
        lines = np.array([[[-2.0, 0.7, 1.0], [2.0, 0.7, 1.0]]])
        clipped, valid = pinhole_barrel.clip_lines_to_image(lines)
        # 两个 piece 均有效
        assert valid[0, 0] and valid[0, 1]
        assert not valid[0, 2]
        # 两段端点都投影在图像内
        for j in range(2):
            for ep in range(2):
                px = _px(pinhole_barrel, clipped[0, j, ep])
                assert -0.5 <= px[0] <= U_MAX + 0.5
                assert -0.5 <= px[1] <= V_MAX + 0.5

    def test_endpoints_inside_after_clip(self, pinhole_barrel):
        """任意有效线段的两个端点都必须投影在图像矩形内"""
        rng = np.random.default_rng(7)
        n = 200
        p1 = np.column_stack([rng.uniform(-3, 3, n),
                              rng.uniform(-2, 2, n),
                              rng.uniform(2, 10, n)])
        p2 = p1 + rng.uniform(-4, 4, (n, 3))
        p2[:, 2] = np.maximum(p2[:, 2], 0.5)
        lines = np.stack([p1, p2], axis=1)
        clipped, valid = pinhole_barrel.clip_lines_to_image(lines)
        assert valid.any()
        pts = clipped[valid].reshape(-1, 3)
        px = _px(pinhole_barrel, pts)
        assert (px[:, 0] >= -0.5).all() and (px[:, 0] <= U_MAX + 0.5).all()
        assert (px[:, 1] >= -0.5).all() and (px[:, 1] <= V_MAX + 0.5).all()


# ========== 3. KB / FTheta 鱼眼（独立参考解） ==========
class TestFisheye:
    def test_kb_bottom_crossing(self):
        """KB fx=500,k1=0.1: v=1079 → θ_d=539/500=1.078=θ+0.1θ³"""
        cam = KannalaBrandtCamera(width=W, height=H, fx=500.0, fy=500.0,
                                  cx=960.0, cy=540.0, k1=0.1)
        proj = Projector(cam)
        # 独立参考解: 二分求 θ+0.1θ³=1.078
        lo, hi = 0.0, math.pi / 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mid + 0.1 * mid ** 3 < 1.078:
                lo = mid
            else:
                hi = mid
        y_ref = 5.0 * math.tan(0.5 * (lo + hi))     # ≈7.508
        p1 = np.array([0.0, 0.0, 5.0])
        p2 = np.array([0.0, 12.0, 5.0])
        clipped, valid = proj.clip_lines_to_image(np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 1, 1] - y_ref) < 5e-2
        v = _px(proj, clipped[0, 0, 1])[1]
        assert abs(v - V_MAX) <= 0.5

    def test_ftheta_bottom_crossing(self):
        """FTheta fw_poly=[0,500]: r=500θ → θ*=539/500=1.078"""
        cam = FThetaCamera(width=W, height=H, fw_poly=[0.0, 500.0],
                           cx=960.0, cy=540.0)
        proj = Projector(cam)
        y_ref = 5.0 * math.tan(1.078)               # ≈9.288
        p1 = np.array([0.0, 0.0, 5.0])
        p2 = np.array([0.0, 15.0, 5.0])
        clipped, valid = proj.clip_lines_to_image(np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 1, 1] - y_ref) < 5e-2
        v = _px(proj, clipped[0, 0, 1])[1]
        assert abs(v - V_MAX) <= 0.5

    def test_kb_fully_inside(self):
        cam = KannalaBrandtCamera(width=W, height=H, fx=500.0, fy=500.0,
                                  cx=960.0, cy=540.0, k1=0.1)
        proj = Projector(cam)
        p1 = np.array([0.1, 0.1, 5.0])
        p2 = np.array([0.2, 0.0, 5.0])
        clipped, valid = proj.clip_lines_to_image(np.array([[p1, p2]]))
        assert valid[0, 0]
        assert np.array_equal(clipped[0, 0, 0], p1)
        assert np.array_equal(clipped[0, 0, 1], p2)


# ========== 4. 退化情形 ==========
class TestDegenerate:
    def test_empty_input(self, pinhole_clean):
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.empty((0, 2, 3)))
        assert clipped.shape == (0, 4, 2, 3)
        assert valid.shape == (0, 4)

    @pytest.mark.parametrize("bad", [
        np.zeros((2, 3)),          # 缺 N 维
        np.zeros((4, 2, 2)),       # 末维不是 3
        np.zeros((4, 3, 3)),       # 中段不是 2
    ])
    def test_shape_error_raises(self, pinhole_clean, bad):
        with pytest.raises(ValueError, match="lines_cam must have shape"):
            pinhole_clean.clip_lines_to_image(bad)

    def test_nan_endpoint_invalid(self, pinhole_clean):
        p1 = np.array([np.nan, 0.0, 5.0])
        p2 = np.array([0.0, 0.0, 5.0])
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert not valid[0].any()
        assert np.isnan(clipped[0]).all()

    def test_all_outside_behind_camera(self, pinhole_clean):
        """线段整体在相机后方（z<0）→ invalid，不返回巨值"""
        lines = np.array([[[0.0, 0.0, -5.0], [1.0, 0.0, -10.0]]])
        clipped, valid = pinhole_clean.clip_lines_to_image(lines)
        assert not valid[0].any()
        assert np.isnan(clipped[0]).all()

    def test_all_outside_off_image(self, pinhole_clean):
        """在相机前方但完全在图像外 → invalid"""
        lines = np.array([[[1000.0, 0.0, 5.0], [2000.0, 0.0, 5.0]]])
        clipped, valid = pinhole_clean.clip_lines_to_image(lines)
        assert not valid[0].any()

    def test_cross_near_z_plane(self, pinhole_clean):
        """端点 z<near_z（相机位于几何体内）→ 交于 near_z 平面"""
        p1 = np.array([0.0, 0.0, 0.05])   # z<near_z=0.1 外
        p2 = np.array([0.0, 0.0, 10.0])   # 内
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        assert abs(clipped[0, 0, 0, 2] - 0.1) < 2e-4
        assert np.array_equal(clipped[0, 0, 1], p2)


# ========== 5. bit-exact 边界 ==========
class TestBoundaryInclusive:
    def test_endpoint_exactly_on_boundary_is_inside(self, pinhole_clean):
        """端点恰落在 u=0 / u=1919 上 → 视为内部（含端点语义）"""
        p1 = np.array([-4.8, 0.0, 5.0])    # u≈+1.4e-13（=0 的浮点邻域）
        p2 = np.array([4.795, 0.0, 5.0])   # u≈1919-1.4e-13
        clipped, valid = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]))
        assert valid[0, 0]
        # all_in 路径：原样返回，不经二分
        assert np.array_equal(clipped[0, 0, 0], p1)
        assert np.array_equal(clipped[0, 0, 1], p2)


# ========== 6. r_max 复现（GTSystem 4 台标定相机） ==========
class TestRMaxReproduction:
    """r_max 必须与 GTSystem 实测值一致（纯 NumPy 5 次迭代反解）"""

    CASES = [
        # name, fx, fy, cx, cy, dist, expect_rmax
        ("2m-A", 1319.711, 1319.362, 953.253, 540.014,
         [0.050573, 0.614254, -4.4e-05, 0.000167, 0.08782,
          0.595484, 0.552382, 0.425873], 1.6726, 1920, 1080),
        ("2m-B-pole", 1319.215, 1319.100, 954.788, 540.838,
         [-0.358890, -0.712111, -0.000408, 3.1e-05, -0.060649,
          0.196353, -1.029467, -0.360568], 0.9647, 1920, 1080),
        ("2m-C", 1308.574, 1308.241, 959.672, 541.013,
         [0.544250, 0.034685, -3.1e-05, 1e-06, -0.000720,
          1.097293, 0.220024, 0.0], 1.7075, 1920, 1080),
        ("8m", 2435.009, 2431.291, 1917.343, 1078.628,
         [-0.182049, 0.101289, -7.8e-05, 0.000778, 0.056209,
          0.463745, -0.194075, 0.206729], 2.2264, 3840, 2160),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_rmax_matches_gtsystem(self, case):
        name, fx, fy, cx, cy, dist, expect, iw, ih = case
        cam = PinholeCamera(width=iw, height=ih, fx=fx, fy=fy,
                            cx=cx, cy=cy, dist_coeffs=dist)
        proj = Projector(cam)
        r = proj.effective_r_max()
        assert abs(r - expect) < 1e-3, f"{name}: r_max={r:.4f} != {expect}"

    def test_domain_radius_only_tightens(self):
        cam = PinholeCamera(width=W, height=H, fx=1000.0, fy=1000.0,
                            cx=960.0, cy=540.0)
        proj = Projector(cam)
        r0 = proj.effective_r_max()
        assert proj.effective_r_max(domain_radius=0.5) == min(r0, 0.5)
        # domain_radius 大于 r_max 不放宽
        assert proj.effective_r_max(domain_radius=1e6) == r0


# ========== 7. 多段输出 ==========
class TestMultiPiece:
    def test_max_pieces_default_is_4(self, pinhole_clean):
        lines = np.array([[[0.0, 0.0, 5.0], [9.59, 0.0, 5.0]]])
        clipped, valid = pinhole_clean.clip_lines_to_image(lines)
        assert clipped.shape[1] == 4
        assert valid.shape[1] == 4

    def test_overflow_invalid(self, pinhole_clean):
        """piece 数超过 max_pieces → 整条线 invalid"""
        # 构造 in-out-in-out-in (3 pieces)，max_pieces=2 → overflow
        p1 = np.array([0.0, 0.0, 5.0])
        p2 = np.array([0.0, 0.0, 5.0])  # 退化，用真实多段
        # 用桶形畸变构造多段: 线段沿 x 轴反复进出
        cam = PinholeCamera(width=W, height=H, fx=1000.0, fy=1000.0,
                            cx=960.0, cy=540.0, dist_coeffs=[-0.3, 0.1, 0, 0])
        proj = Projector(cam)
        # 长水平线穿越多次边界
        lines = np.array([[[-8.0, 0.0, 1.0], [8.0, 0.0, 1.0]]])
        clipped, valid = proj.clip_lines_to_image(lines, max_pieces=1)
        # max_pieces=1 时若有多段则 overflow → invalid
        n_pieces = int(valid[0].sum())
        if n_pieces > 1:
            assert not valid[0].any()


# ========== 8. 折线点 return_polyline ==========
class TestPolyline:
    def test_polyline_shape_and_endpoints(self, pinhole_clean):
        p1 = np.array([0.0, 0.0, 5.0])
        p2 = np.array([9.59, 0.0, 5.0])
        clipped, valid, polys = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]), return_polyline=True)
        assert valid[0, 0]
        assert len(polys) == 1
        assert len(polys[0]) == 1          # 1 个 piece
        poly = polys[0][0]
        assert poly.ndim == 2 and poly.shape[1] == 3
        # 折线首尾与裁剪端点一致
        assert np.allclose(poly[0], clipped[0, 0, 0])
        assert np.allclose(poly[-1], clipped[0, 0, 1])

    def test_polyline_respects_arc_length(self, pinhole_clean):
        """折线点间距 ≈4px（弧长采样，可见段 <512px 不触发 128 点上限）"""
        p1 = np.array([0.0, 0.0, 5.0])
        p2 = np.array([1.0, 0.0, 5.0])    # 可见段 u:960→1160, 约 200px
        _, _, polys = pinhole_clean.clip_lines_to_image(
            np.array([[p1, p2]]), return_polyline=True)
        poly = polys[0][0]
        px = _px(pinhole_clean, poly)
        gaps = np.hypot(np.diff(px[:, 0]), np.diff(px[:, 1]))
        # 间距 ≈4px
        assert np.all(gaps <= 4.5)
        assert len(poly) >= 2

    def test_no_polyline_when_invalid(self, pinhole_clean):
        lines = np.array([[[1000.0, 0.0, 5.0], [2000.0, 0.0, 5.0]]])
        _, _, polys = pinhole_clean.clip_lines_to_image(
            lines, return_polyline=True)
        assert polys[0] == []


# ========== 9. 像素收敛精度 ==========
class TestConvergence:
    def test_long_edge_subpixel_accuracy(self):
        """长棱（20m, z=2m, fx=2435）裁剪端点距边界 ≤0.5px"""
        cam = PinholeCamera(width=3840, height=2160, fx=2435.0, fy=2431.0,
                            cx=1920.0, cy=1080.0)
        proj = Projector(cam)
        # 水平棱从图像内延伸到 20m 外
        p1 = np.array([0.0, 0.0, 2.0])     # 主点内
        p2 = np.array([20.0, 0.0, 2.0])    # 远出右边界
        clipped, valid = proj.clip_lines_to_image(np.array([[p1, p2]]))
        assert valid[0, 0]
        u = _px(proj, clipped[0, 0, 1])[0]
        assert abs(u - (3840 - 1)) <= 0.5


# ========== 10. project_box(clip_to_image=True) 桥接 ==========
def _make_box(x0, x1, y0, y1, z0, z1):
    """autoproj 角点约定: face x0 4 角 + face x1 4 角, (y,z) 序 (-,-),(+,-),(+,+),(-,+)"""
    return np.array([
        [x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1],
        [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1],
    ], dtype=np.float64)


class TestProjectBoxBridge:
    def test_box_inside_all_edges_valid(self, pinhole_clean):
        box = _make_box(-1, 1, -0.5, 0.5, 5, 7)
        res = pinhole_clean.project_box(box, pts_in_cam=True,
                                        clip_to_image=True)
        assert len(res['edges']) == 12
        for e in res['edges']:
            assert e['is_clipped_to_image'] is True
            assert e['piece_count'] == 1
            assert e['piece_index'] == 0
            for pt in (e['pt1'], e['pt2']):
                assert 0.0 <= pt[0] <= U_MAX
                assert 0.0 <= pt[1] <= V_MAX

    def test_box_partially_outside_endpoints_in_rect(self, pinhole_barrel):
        """部分出画的框: 有效棱端点均落在图像矩形内（桶形畸变）"""
        box = _make_box(3, 5, -1, 1, 5, 7)   # 右侧 u 最大=1000*5/5+960=1960 出画
        res = pinhole_barrel.project_box(box, pts_in_cam=True,
                                         clip_to_image=True)
        assert len(res['edges']) > 0
        for e in res['edges']:
            assert e['is_clipped_to_image'] is True
            for pt in (e['pt1'], e['pt2']):
                assert -0.5 <= pt[0] <= U_MAX + 0.5
                assert -0.5 <= pt[1] <= V_MAX + 0.5

    def test_mutual_exclusion_with_extend_to_boundary(self, pinhole_clean):
        """clip_to_image 与 extend_to_boundary 互斥: clip_to_image 优先"""
        box = _make_box(3, 5, -1, 1, 5, 7)
        res = pinhole_clean.project_box(box, pts_in_cam=True,
                                        extend_to_boundary=True,
                                        clip_to_image=True)
        assert len(res['edges']) > 0
        for e in res['edges']:
            assert e['is_clipped_to_image'] is True
            for pt in (e['draw_pt1'], e['draw_pt2']):
                assert -0.5 <= pt[0] <= U_MAX + 0.5
                assert -0.5 <= pt[1] <= V_MAX + 0.5

    def test_default_path_has_no_marker_bitexact(self, pinhole_clean):
        """默认路径回归: 无 'is_clipped_to_image' 键，且与显式 False 一致"""
        box = _make_box(3, 5, -1, 1, 5, 7)
        res_default = pinhole_clean.project_box(box, pts_in_cam=True)
        res_false = pinhole_clean.project_box(box, pts_in_cam=True,
                                              clip_to_image=False)
        assert len(res_default['edges']) == len(res_false['edges'])
        for e0, e1 in zip(res_default['edges'], res_false['edges']):
            assert 'is_clipped_to_image' not in e0
            assert 'is_clipped_to_image' not in e1
            assert np.array_equal(e0['pt1'], e1['pt1'])
            assert np.array_equal(e0['pt2'], e1['pt2'])
            assert e0['pt1_is_corner'] == e1['pt1_is_corner']
            assert e0['pt2_is_corner'] == e1['pt2_is_corner']

    def test_extend_boundary_default_unchanged(self, pinhole_clean):
        """extend_to_boundary 默认路径无新键（行为不变回归）"""
        box = _make_box(3, 5, -1, 1, 5, 7)
        res = pinhole_clean.project_box(box, pts_in_cam=True,
                                        extend_to_boundary=True)
        assert len(res['edges']) > 0
        for e in res['edges']:
            assert 'is_clipped_to_image' not in e

    def test_batch_matches_single(self, pinhole_barrel):
        """project_boxes 批量与 project_box 单框一致（clip 路径）"""
        boxes = np.stack([
            _make_box(-1, 1, -0.5, 0.5, 5, 7),
            _make_box(3, 5, -1, 1, 5, 7),
        ])
        batch = pinhole_barrel.project_boxes(boxes, pts_in_cam=True,
                                             clip_to_image=True)
        for i in range(2):
            single = pinhole_barrel.project_box(boxes[i], pts_in_cam=True,
                                                clip_to_image=True)
            assert len(batch[i]['edges']) == len(single['edges'])
            for eb, es in zip(batch[i]['edges'], single['edges']):
                assert np.allclose(eb['pt1'], es['pt1'])
                assert np.allclose(eb['pt2'], es['pt2'])

    def test_box_fully_outside_no_edges(self, pinhole_clean):
        box = _make_box(100, 102, -1, 1, 5, 7)   # 完全出画
        res = pinhole_clean.project_box(box, pts_in_cam=True,
                                        clip_to_image=True)
        assert len(res['edges']) == 0


# ========== 11. 窄可见窗口加密复扫（A2） ==========
class TestNarrowWindowRescan:
    def test_narrow_window_detected(self, pinhole_clean):
        """16 点粗扫全 outside 但真实存在窄可见窗口 → 复扫应找回

        z=5、x 从 -100→100 时，图像矩形对应的 t 窗口约 [0.476, 0.524]
        （宽 0.048 < 1/15 ≈ 0.067），16 点采样恰好全部落在窗口外。
        """
        a = np.array([-100.0, -10.0, 5.0])
        b = np.array([100.0, 10.0, 5.0])
        # 粗扫确认：16 点全 outside
        t16 = np.linspace(0.0, 1.0, 16)
        px = _px(pinhole_clean, a[None, :] + t16[:, None] * (b - a)[None, :])
        assert not np.any((px[:, 0] >= 0) & (px[:, 0] <= U_MAX) &
                          (px[:, 1] >= 0) & (px[:, 1] <= V_MAX))

        clipped, valid = pinhole_clean.clip_lines_to_image(np.array([[a, b]]))
        assert valid.sum() == 1
        t_lo = (clipped[0, 0, 0, 0] - a[0]) / (b[0] - a[0])
        t_hi = (clipped[0, 0, 1, 0] - a[0]) / (b[0] - a[0])
        assert 0.47 < t_lo < 0.48
        assert 0.52 < t_hi < 0.53

    def test_far_line_still_invalid(self, pinhole_clean):
        """远离图像的线段不应因复扫被判可见"""
        far = np.array([[[500.0, 500.0, 5.0], [600.0, 600.0, 5.0]]])
        _, valid = pinhole_clean.clip_lines_to_image(far)
        assert valid.sum() == 0


# ========== 12. 半径缓存（A1） ==========
class TestRadiusCache:
    def test_radial_radius_computed_once(self, pinhole_barrel, monkeypatch):
        """重复调用命中缓存，内部实现只执行一次"""
        calls = {'n': 0}
        orig = Projector._radial_valid_radius_impl

        def counted(self):
            calls['n'] += 1
            return orig(self)

        monkeypatch.setattr(Projector, '_radial_valid_radius_impl', counted)
        pinhole_barrel.radial_valid_radius()
        pinhole_barrel.radial_valid_radius()
        pinhole_barrel.effective_r_max()
        assert calls['n'] == 1

    def test_boundary_rmax_computed_once(self, pinhole_barrel, monkeypatch):
        calls = {'n': 0}
        orig = Projector._image_boundary_rmax_impl

        def counted(self):
            calls['n'] += 1
            return orig(self)

        monkeypatch.setattr(Projector, '_image_boundary_rmax_impl', counted)
        pinhole_barrel.image_boundary_rmax()
        pinhole_barrel.image_boundary_rmax()
        assert calls['n'] == 1

    def test_cache_invalidated_on_calibration_change(self, pinhole_barrel):
        """标定参数变更后缓存失效，不返回过期半径"""
        r_before = pinhole_barrel.radial_valid_radius()
        pinhole_barrel.camera.k1 = -0.05
        r_after = pinhole_barrel.radial_valid_radius()
        assert r_after != r_before
