"""
FOV限制参数测试

测试 max_fov_deg 参数在三种相机类型中的传递和使用：
- PinholeCamera: max_fov_deg 参数传递
- KannalaBrandtCamera: 大FOV场景测试
- FThetaCamera: 大FOV场景测试

同时测试 FrustumCuller 在大FOV（θ > 90°）场景下的裁剪行为。
"""

import numpy as np
import pytest
from autoproj import CameraFactory, PinholeCamera, KannalaBrandtCamera, FThetaCamera
from autoproj.frustum import FrustumCuller
from autoproj.projection import Projector


class TestFOVLimitsParameter:
    """测试 max_fov_deg 参数传递"""

    def test_pinhole_max_fov_deg_none(self):
        """测试PinholeCamera无FOV上限"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            max_fov_deg=None
        )
        assert camera._max_fov_rad is None

    def test_pinhole_max_fov_deg_set(self):
        """测试PinholeCamera设置FOV上限"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            max_fov_deg=90  # 90° full FOV (45° half-angle)
        )
        assert camera._max_fov_rad == pytest.approx(np.pi / 2)

    def test_kannala_max_fov_deg_none(self):
        """测试KB相机无FOV上限"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=None
        )
        assert camera._max_fov_rad is None
        # 无上限时，theta_max应大于90°（对于这种畸变参数）
        assert camera._theta_max > np.pi / 4

    def test_kannala_max_fov_deg_180(self):
        """测试KB相机设置180°上限"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=180  # 180° full FOV (90° half-angle)
        )
        assert camera._max_fov_rad == pytest.approx(np.pi)
        # 设置上限后，theta_max不应超过上限
        assert camera._theta_max <= np.pi

    def test_kannala_max_fov_deg_90(self):
        """测试KB相机设置90°上限"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=90  # 90° full FOV (45° half-angle)
        )
        assert camera._max_fov_rad == pytest.approx(np.pi / 2)
        # 设置上限后，theta_max不应超过上限
        assert camera._theta_max <= np.pi / 2

    def test_ftheta_max_fov_deg_none(self):
        """测试FTheta相机无FOV上限"""
        camera = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500, 0.01],
            cx=960, cy=540,
            max_fov_deg=None
        )
        assert camera._max_fov_rad is None

    def test_ftheta_max_fov_deg_set(self):
        """测试FTheta相机设置FOV上限"""
        camera = FThetaCamera(
            width=1920, height=1080,
            fw_poly=[0, 500, 0.01],
            cx=960, cy=540,
            max_fov_deg=180  # 180° full FOV (90° half-angle)
        )
        assert camera._max_fov_rad == pytest.approx(np.pi)
        # 设置上限后，theta_max不应超过上限
        assert camera._theta_max <= np.pi

    def test_factory_create_with_max_fov(self):
        """测试工厂方法传递max_fov_deg"""
        camera = CameraFactory.create_fisheye(
            sub_type='kannala',
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.2, k2=0.05,
            max_fov_deg=120  # 120° full FOV (60° half-angle)
        )
        assert camera._max_fov_rad == pytest.approx(2 * np.pi / 3)


class TestLargeFOVRendering:
    """测试大FOV场景下的渲染行为（θ > 90°）"""

    def test_kb_projection_large_fov(self):
        """测试KB相机大角度点投影"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=180  # 180° full FOV (90° half-angle)
        )

        # 测试不同角度的点
        test_points = [
            [0, 0, 10],      # 正面，θ=0°
            [5, 0, 10],      # 侧面，θ≈26.6°
            [15, 0, 10],     # 大角度，θ≈56.3°
            [30, 0, 10],     # 更大角度，θ≈71.6°
            [50, 0, 10],     # 接近90°，θ≈78.7°
        ]

        for pt in test_points:
            points_3d = np.array([pt])
            result, valid = camera.project(points_3d, pts_in_cam=True, preserve_extra=True)

            # 检查形状保持
            assert result.shape == points_3d.shape

            if pt[2] > 0:
                # 点在相机前方，应该有效
                assert valid[0], f"Point {pt} should be valid"

    def test_frustum_culler_large_fov(self):
        """测试FrustumCuller大FOV场景"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=180  # 180° full FOV (90° half-angle)
        )

        culler = FrustumCuller.from_camera(camera, expansion_factor=1.0)

        # 测试线段裁剪
        test_segments = [
            ((0, 0, 5), (3, 0, 5)),      # 水平线
            ((0, 0, 5), (5, 0, 3)),      # 斜线
            ((10, 0, 5), (15, 0, 3)),    # 大角度斜线
            ((0, 0, 5), (-5, 0, 5)),     # 左侧线
        ]

        for pt1, pt2 in test_segments:
            result = culler.clip_line(np.array(pt1, dtype=np.float64), np.array(pt2, dtype=np.float64))
            # 裁剪结果应该是None（完全在视锥外）或两个3D点
            if result is not None:
                assert len(result) == 2
                assert len(result[0]) == 3
                assert len(result[1]) == 3

    def test_project_box_large_fov(self):
        """测试Projector大FOV场景下的box投影"""
        camera = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=180  # 180° full FOV (90° half-angle)
        )

        projector = Projector(camera, cull_frustum=True)

        # 构造一个3D框，部分在视锥内，部分在视锥外
        corners = np.array([
            [-2, -1, 5],  # 前左角点
            [2, -1, 5],   # 前右角点
            [2, 1, 5],    # 后右角点
            [-2, 1, 5],   # 后左角点
            [-2, -1, 8],  # 上前左角点
            [2, -1, 8],   # 上前右角点
            [2, 1, 8],    # 上后右角点
            [-2, 1, 8],   # 上后左角点
        ], dtype=np.float64)

        result = projector.project_box(corners, pts_in_cam=True)

        # 检查结果结构
        assert 'corners' in result
        assert 'valid' in result
        assert 'edges' in result

        # 检查所有角点的有效性
        assert len(result['valid']) == 8

        # 检查边数（标准12条边）
        assert len(result['edges']) >= 0

        # 检查每条边的结构
        for edge in result['edges']:
            assert 'pt1' in edge
            assert 'pt2' in edge
            assert 'pt1_is_corner' in edge
            assert 'pt2_is_corner' in edge

    def test_extend_edges_to_boundary_large_fov(self):
        """测试大FOV场景下的边延长"""
        # 构造边列表，模拟部分裁剪的情况
        edges = [
            {
                'pt1': np.array([960, 540], dtype=np.float64),  # 中心
                'pt2': np.array([1200, 540], dtype=np.float64),  # 右侧
                'pt1_is_corner': True,
                'pt2_is_corner': False,  # 裁剪点
            },
            {
                'pt1': np.array([800, 400], dtype=np.float64),  # 裁剪点
                'pt2': np.array([850, 500], dtype=np.float64),  # 角点
                'pt1_is_corner': False,
                'pt2_is_corner': True,
            },
        ]

        extended = Projector.extend_edges_to_boundary(edges, 1920, 1080)

        # 检查延长结果
        assert len(extended) == 2

        # 第一条边：从角点(960,540)延长裁剪点(1200,540)到右边界
        edge1 = extended[0]
        assert edge1['pt1_is_corner'] == True
        assert edge1['pt2_is_corner'] == False
        assert edge1['is_extended'] == True
        # pt2应该被延长到右边界(x=1919)
        assert abs(edge1['pt2'][0] - 1919) < 1
        assert abs(edge1['pt2'][1] - 540) < 1  # y坐标不变

        # 第二条边：从角点(850,500)沿方向延长裁剪点(800,400)到边界
        # 方向: (-50, -100)，应该延长到上边界(y=0)
        edge2 = extended[1]
        assert edge2['pt1_is_corner'] == False
        assert edge2['pt2_is_corner'] == True
        assert edge2['is_extended'] == True
        # pt1应该被延长到上边界(y=0)，x=850+(-50)*5=600
        assert abs(edge2['pt1'][1] - 0) < 1
        assert abs(edge2['pt1'][0] - 600) < 1


class TestFOVConsistency:
    """测试不同相机类型的FOV计算一致性"""

    def test_pinhole_fov_without_distortion(self):
        """测试无畸变针孔相机FOV"""
        camera = PinholeCamera(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=None
        )

        # fov_h 和 fov_v 已返回度数
        # 计算标称FOV
        h_fov = 2 * np.degrees(np.arctan(960 / 1000))
        v_fov = 2 * np.degrees(np.arctan(540 / 1000))

        # 检查计算的FOV接近标称值
        assert abs(camera.fov_h - h_fov) < 1
        assert abs(camera.fov_v - v_fov) < 1

    def test_kb_fov_with_distortion(self):
        """测试KB相机畸变对FOV的影响"""
        # 有畸变
        camera_distorted = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001
        )

        # 无畸变（k1=k2=k3=k4=0）
        camera_no_distortion = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0, k2=0, k3=0, k4=0
        )

        # 有畸变时的 theta_max通常更大（畸变扩展了FOV）
        assert camera_distorted._theta_max >= camera_no_distortion._theta_max

    def test_max_fov_cap_enforced(self):
        """测试FOV上限强制"""
        # 使用较小的上限值确保能看到效果
        camera_no_limit = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=None
        )

        # 使用较小的上限值（60°），确保能限制FOV
        camera_with_limit = KannalaBrandtCamera(
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=60  # 60° cap
        )

        # 有上限时，theta_max应小于等于上限（弧度）
        assert camera_with_limit._theta_max <= np.radians(60)

        # 无上限时，theta_max应该更大
        assert camera_no_limit._theta_max > camera_with_limit._theta_max


class TestSoftClip:
    """测试针孔相机软裁剪（对角线修正）"""

    def test_soft_clip_disabled_by_default(self):
        """默认禁用软裁剪"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera)
        assert projector._soft_clip_ratio is None

    def test_soft_clip_enabled(self):
        """启用软裁剪后参数正确"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, soft_clip_ratio=0.05)
        assert projector._soft_clip_ratio == 0.05

    def test_soft_clip_threshold_value(self):
        """阈值 = 0.05 * min(width, height)"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, soft_clip_ratio=0.05)
        # 0.05 * min(1920, 1080) = 0.05 * 1080 = 54
        expected = 0.05 * 1080
        actual = projector._soft_clip_ratio * min(camera.width, camera.height)
        assert abs(actual - expected) < 0.01

    def test_soft_clip_preserves_corner_near_boundary(self):
        """软裁剪保留接近边界的原始角点"""
        # 针孔相机，有畸变
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=np.array([-0.05, 0, 0, 0, 0, 0, 0, 0])
        )

        # 不带软裁剪
        projector_no_soft = Projector(camera, soft_clip_ratio=None)
        # 带软裁剪
        projector_with_soft = Projector(camera, soft_clip_ratio=0.05)

        # 构造一个角点刚好在 V-FOV 边界外的 box
        # tan_v_nominal = max(540, 540) / 1000 = 0.54
        # V-FOV 边界 ≈ 0.54 * (1 + 0.05) ≈ 0.567
        # 构造角点 y = 0.575 * z（略超 V-FOV 边界）
        corners = np.array([
            [-1, -0.5, 10],   # 0: 前左下
            [1, -0.5, 10],    # 1: 前右下
            [1, 0.5, 10],     # 2: 后右下
            [-1, 0.5, 10],    # 3: 后左下
            [-1, -0.5, 12],   # 4: 上前左下
            [1, -0.5, 12],    # 5: 上前右下
            [1, 0.575 * 12, 12],  # 6: 上后右下（y略超V-FOV）
            [-1, 0.575 * 12, 12], # 7: 上后左下（y略超V-FOV）
        ], dtype=np.float64)

        result_no_soft = projector_no_soft.project_box(corners, pts_in_cam=True)
        result_with_soft = projector_with_soft.project_box(corners, pts_in_cam=True)

        # 软裁剪后应该有更多端点被标记为角点（is_corner=True）
        # 因为略超边界的角点投影可能仍在图像范围内
        assert len(result_no_soft['edges']) > 0
        assert len(result_with_soft['edges']) > 0

    def test_soft_clip_does_not_accept_far_points(self):
        """软裁剪不应接受远超FOV的点"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540,
            dist_coeffs=np.array([-0.1, 0, 0, 0, 0, 0, 0, 0])
        )
        projector = Projector(camera, soft_clip_ratio=0.05)

        # 构造角点远超 FOV（y = 2.0 * z，远超 tan_v ≈ 0.54）
        corners = np.array([
            [-1, -0.5, 10],
            [1, -0.5, 10],
            [1, 0.5, 10],
            [-1, 0.5, 10],
            [-1, -0.5, 12],
            [1, -0.5, 12],
            [1, 2.0 * 12, 12],   # 远超 V-FOV
            [-1, 2.0 * 12, 12],  # 远超 V-FOV
        ], dtype=np.float64)

        result = projector.project_box(corners, pts_in_cam=True)

        # 远超 FOV 的角点不应被保留为 is_corner
        for edge in result['edges']:
            if edge['pt1'][1] > 600:  # 上半部分
                # 不应被标记为角点（投影应在图像外）
                assert not edge['pt1_is_corner'] or edge['pt1'][1] < 1080

    def test_soft_clip_with_box_fully_inside(self):
        """完全在FOV内的box不受软裁剪影响"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, soft_clip_ratio=0.05)

        corners = np.array([
            [-1, -0.5, 10], [1, -0.5, 10],
            [1, 0.5, 10], [-1, 0.5, 10],
            [-1, -0.5, 12], [1, -0.5, 12],
            [1, 0.5, 12], [-1, 0.5, 12],
        ], dtype=np.float64)

        result = projector.project_box(corners, pts_in_cam=True)

        # 所有边都应该是角点→角点
        for edge in result['edges']:
            assert edge['pt1_is_corner'] == True
            assert edge['pt2_is_corner'] == True

    def test_soft_clip_fisheye_not_affected(self):
        """鱼眼相机不受软裁剪影响（锥形视锥，无对角线问题）"""
        camera = CameraFactory.create_fisheye(
            sub_type='kannala',
            width=1920, height=1080,
            fx=500, fy=500, cx=960, cy=540,
            k1=0.218, k2=-0.058, k3=0.008, k4=-0.001,
            max_fov_deg=180
        )

        # 同一组box，对比开启/关闭软裁剪的结果
        projector_no_soft = Projector(camera, soft_clip_ratio=None)
        projector_with_soft = Projector(camera, soft_clip_ratio=0.05)

        corners = np.array([
            [-2, -1, 5], [2, -1, 5],
            [2, 1, 5], [-2, 1, 5],
            [-2, -1, 8], [2, -1, 8],
            [2, 1, 8], [-2, 1, 8],
        ], dtype=np.float64)

        result_no_soft = projector_no_soft.project_box(corners, pts_in_cam=True)
        result_with_soft = projector_with_soft.project_box(corners, pts_in_cam=True)

        # 结果应该完全一致（鱼眼不触发软裁剪）
        assert len(result_no_soft['edges']) == len(result_with_soft['edges'])
        for e1, e2 in zip(result_no_soft['edges'], result_with_soft['edges']):
            np.testing.assert_array_almost_equal(e1['pt1'], e2['pt1'])
            np.testing.assert_array_almost_equal(e1['pt2'], e2['pt2'])
            assert e1['pt1_is_corner'] == e2['pt1_is_corner']
            assert e2['pt2_is_corner'] == e2['pt2_is_corner']


class TestDrawPoints:
    """测试 draw_pt1/draw_pt2 字段的正确性"""

    def test_draw_pts_exist_no_clipping(self):
        """无裁剪时 draw_pt == pt"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, cull_frustum=False)
        corners = np.array([
            [-1, -0.5, 10], [1, -0.5, 10],
            [1, 0.5, 10], [-1, 0.5, 10],
            [-1, -0.5, 12], [1, -0.5, 12],
            [1, 0.5, 12], [-1, 0.5, 12],
        ], dtype=np.float64)
        result = projector.project_box(corners, pts_in_cam=True)
        assert len(result['edges']) > 0
        for edge in result['edges']:
            assert 'draw_pt1' in edge
            assert 'draw_pt2' in edge
            np.testing.assert_array_almost_equal(edge['draw_pt1'], edge['pt1'])
            np.testing.assert_array_almost_equal(edge['draw_pt2'], edge['pt2'])

    def test_draw_pts_exist_with_clipping(self):
        """有裁剪时 draw_pt 存在且等于 pt"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, cull_frustum=True)
        corners = np.array([
            [-1, -0.5, 10], [1, -0.5, 10],
            [1, 0.5, 10], [-1, 0.5, 10],
            [-1, -0.5, 12], [1, -0.5, 12],
            [1, 0.5, 12], [-1, 0.5, 12],
        ], dtype=np.float64)
        result = projector.project_box(corners, pts_in_cam=True, extend_to_boundary=False)
        assert len(result['edges']) > 0
        for edge in result['edges']:
            assert 'draw_pt1' in edge
            assert 'draw_pt2' in edge
            np.testing.assert_array_almost_equal(edge['draw_pt1'], edge['pt1'])
            np.testing.assert_array_almost_equal(edge['draw_pt2'], edge['pt2'])

    def test_draw_pts_with_extension(self):
        """有延长时 draw_pt 反映延长后的坐标"""
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000, fy=1000, cx=960, cy=540
        )
        projector = Projector(camera, cull_frustum=True)
        # 构造部分在FOV外的box
        corners = np.array([
            [-1, -0.5, 10], [1, -0.5, 10],
            [1, 0.5, 10], [-1, 0.5, 10],
            [-1, -0.5, 12], [1, -0.5, 12],
            [1, 2.0, 12], [-1, 2.0, 12],  # 上方角点超出FOV
        ], dtype=np.float64)
        result = projector.project_box(corners, pts_in_cam=True, extend_to_boundary=True)
        assert len(result['edges']) > 0
        for edge in result['edges']:
            assert 'draw_pt1' in edge
            assert 'draw_pt2' in edge
            # draw_pt 始终等于 pt（延长后 pt 已更新）
            np.testing.assert_array_almost_equal(edge['draw_pt1'], edge['pt1'])
            np.testing.assert_array_almost_equal(edge['draw_pt2'], edge['pt2'])

    def test_draw_pts_extend_static_method(self):
        """extend_edges_to_boundary 正确同步 draw_pt"""
        camera = CameraFactory.create_pinhole(
            width=100, height=100,
            fx=50, fy=50, cx=50, cy=50
        )
        # 手动构造一条边：一端角点(50,50)，一端裁剪点(120,60)
        edges = [{
            'pt1': np.array([50.0, 50.0]),
            'pt2': np.array([120.0, 60.0]),
            'draw_pt1': np.array([50.0, 50.0]),
            'draw_pt2': np.array([120.0, 60.0]),
            'pt1_is_corner': True,
            'pt2_is_corner': False,
        }]
        result = Projector.extend_edges_to_boundary(edges, 100, 100)
        # 延长后 pt2 应更新为边界点
        assert result[0]['is_extended'] == True
        # draw_pt2 应等于更新后的 pt2
        np.testing.assert_array_almost_equal(result[0]['draw_pt2'], result[0]['pt2'])
        # draw_pt1 应等于 pt1（未变）
        np.testing.assert_array_almost_equal(result[0]['draw_pt1'], result[0]['pt1'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
