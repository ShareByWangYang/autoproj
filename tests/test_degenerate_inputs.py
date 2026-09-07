"""
退化输入处理测试

测试空数组、NaN 值、near_z=0 等边界情况：
- Camera/FrustumCuller 构造参数验证
- project_points 空数组与 NaN
- project_box 形状验证与 NaN
- project_lines 空线段
- project_polygon 空多边形
- _project_raw_pixels 空数组
- cull_points 空数组与 NaN
- clip_line 形状验证与 NaN
"""

import numpy as np
import pytest
from autoproj import CameraFactory, PinholeCamera, KannalaBrandtCamera, FThetaCamera, Projector, Polygon3D
from autoproj.frustum import FrustumCuller, FrustumType


# ============================================================
# 构造参数验证
# ============================================================

class TestCameraConstructionValidation:
    """测试 Camera 构造参数的退化输入验证"""

    def test_near_z_zero_raises(self):
        """near_z=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            PinholeCamera(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540, near_z=0)

    def test_near_z_negative_raises(self):
        """near_z 为负数应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            PinholeCamera(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540, near_z=-1.0)

    def test_near_z_nan_raises(self):
        """near_z 为 NaN 应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            PinholeCamera(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540, near_z=float('nan'))

    def test_near_z_ge_far_z_raises(self):
        """near_z >= far_z 应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z.*must be less than far_z"):
            PinholeCamera(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
                          near_z=100, far_z=100)
        with pytest.raises(ValueError, match="near_z.*must be less than far_z"):
            PinholeCamera(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540,
                          near_z=200, far_z=100)

    def test_width_zero_raises(self):
        """width=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="width must be a positive"):
            PinholeCamera(width=0, height=1080, fx=1000, fy=1000, cx=960, cy=540)

    def test_width_negative_raises(self):
        """width 为负数应抛出 ValueError"""
        with pytest.raises(ValueError, match="width must be a positive"):
            PinholeCamera(width=-1, height=1080, fx=1000, fy=1000, cx=960, cy=540)

    def test_height_zero_raises(self):
        """height=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="height must be a positive"):
            PinholeCamera(width=1920, height=0, fx=1000, fy=1000, cx=960, cy=540)

    def test_fisheye_near_z_zero_raises(self):
        """鱼眼相机 near_z=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            KannalaBrandtCamera(width=1920, height=1080, fx=500, fy=500,
                                cx=960, cy=540, k1=0.1, k2=0.05, k3=0.01, k4=0.005,
                                near_z=0)


class TestFrustumCullerConstructionValidation:
    """测试 FrustumCuller 构造参数的退化输入验证"""

    def test_near_z_zero_raises(self):
        """near_z=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            FrustumCuller(FrustumType.PYRAMID, near_z=0, tan_h=1.0, tan_v=0.5)

    def test_near_z_negative_raises(self):
        """near_z 为负数应抛出 ValueError"""
        with pytest.raises(ValueError, match="near_z must be a positive"):
            FrustumCuller(FrustumType.PYRAMID, near_z=-0.1, tan_h=1.0, tan_v=0.5)

    def test_expansion_factor_zero_raises(self):
        """expansion_factor=0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="expansion_factor must be a positive"):
            FrustumCuller(FrustumType.PYRAMID, near_z=0.1, expansion_factor=0,
                          tan_h=1.0, tan_v=0.5)

    def test_expansion_factor_negative_raises(self):
        """expansion_factor 为负数应抛出 ValueError"""
        with pytest.raises(ValueError, match="expansion_factor must be a positive"):
            FrustumCuller(FrustumType.PYRAMID, near_z=0.1, expansion_factor=-1.0,
                          tan_h=1.0, tan_v=0.5)


# ============================================================
# project_points 退化输入
# ============================================================

class TestProjectPointsDegenerate:
    """测试 project_points 的退化输入处理"""

    @pytest.fixture
    def projector(self):
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        )
        return Projector(camera, frustum_scale=1.0)

    def test_empty_array(self, projector):
        """空数组应返回空结果"""
        empty = np.empty((0, 3))
        result, valid = projector.project_points(empty, pts_in_cam=True)
        assert len(result) == 0
        assert len(valid) == 0

    def test_empty_array_preserve_extra(self, projector):
        """空数组 + preserve_extra 应返回空 float64 结果"""
        empty = np.empty((0, 3))
        result, valid = projector.project_points(empty, pts_in_cam=True, preserve_extra=True)
        assert len(result) == 0
        assert len(valid) == 0
        assert result.dtype == np.float64

    def test_nan_values_filtered(self, projector):
        """含 NaN 的点应标记为无效"""
        points = np.array([
            [1, 0, 10],
            [np.nan, 0, 10],
            [0, np.nan, 10],
            [0, 0, np.nan],
            [-1, 0, 10],
        ])
        result, valid = projector.project_points(points, pts_in_cam=True)
        assert valid[0] == True   # 正常点
        assert valid[1] == False  # x=NaN
        assert valid[2] == False  # y=NaN
        assert valid[3] == False  # z=NaN
        assert valid[4] == True   # 正常点

    def test_all_nan(self, projector):
        """全部 NaN 的点应全部无效"""
        points = np.full((3, 3), np.nan)
        result, valid = projector.project_points(points, pts_in_cam=True)
        assert np.all(~valid)
        assert len(valid) == 3


# ============================================================
# project_box 退化输入
# ============================================================

class TestProjectBoxDegenerate:
    """测试 project_box 的退化输入处理"""

    @pytest.fixture
    def projector(self):
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        )
        return Projector(camera, frustum_scale=1.0)

    def test_wrong_shape_raises(self, projector):
        """形状不是 (8,3) 应抛出 ValueError"""
        with pytest.raises(ValueError, match="must have 8 corners"):
            projector.project_box(np.zeros((6, 3)), pts_in_cam=True)

        with pytest.raises(ValueError, match="must have 8 corners"):
            projector.project_box(np.zeros((8, 2)), pts_in_cam=True)

    def test_nan_corners_raises(self, projector):
        """角点含 NaN 应抛出 ValueError"""
        corners = np.zeros((8, 3))
        corners[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            projector.project_box(corners, pts_in_cam=True)

    def test_empty_box_raises(self, projector):
        """空数组应抛出 ValueError（形状不匹配）"""
        with pytest.raises(ValueError, match="must have 8 corners"):
            projector.project_box(np.empty((0, 3)), pts_in_cam=True)


# ============================================================
# project_lines 退化输入
# ============================================================

class TestProjectLinesDegenerate:
    """测试 project_lines 的退化输入处理"""

    @pytest.fixture
    def projector(self):
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        )
        return Projector(camera, frustum_scale=1.0)

    def test_empty_lines(self, projector):
        """空线段列表应返回空列表"""
        empty = np.empty((0, 2, 3))
        result = projector.project_lines(empty, pts_in_cam=True)
        assert result == []

    def test_empty_list(self, projector):
        """空 list 应返回空列表"""
        result = projector.project_lines([], pts_in_cam=True)
        assert result == []


# ============================================================
# project_polygon 退化输入
# ============================================================

class TestProjectPolygonDegenerate:
    """测试 project_polygon 的退化输入处理"""

    @pytest.fixture
    def projector(self):
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        )
        return Projector(camera, frustum_scale=1.0)

    def test_empty_polygon(self, projector):
        """空多边形应返回空结果"""
        poly = Polygon3D(np.empty((0, 3)))
        result = projector.project_polygon(poly, pts_in_cam=True)
        assert len(result['vertices']) == 0
        assert len(result['valid']) == 0
        assert result['edges'] == []


# ============================================================
# _project_raw_pixels 退化输入
# ============================================================

class TestProjectRawPixelsDegenerate:
    """测试 _project_raw_pixels 的退化输入处理"""

    @pytest.fixture
    def projector(self):
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        )
        return Projector(camera, frustum_scale=1.0)

    def test_empty_array(self, projector):
        """空数组应返回 (0, 2) 空数组"""
        empty = np.empty((0, 3))
        result = projector._project_raw_pixels(empty)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    def test_empty_array_fisheye(self):
        """鱼眼相机空数组应返回 (0, 2) 空数组"""
        camera = CameraFactory.create_fisheye(
            sub_type='kannala',
            width=1920, height=1080, fx=500, fy=500, cx=960, cy=540,
            k1=0.1, k2=0.05, k3=0.01, k4=0.005
        )
        projector = Projector(camera, frustum_scale=1.0)
        empty = np.empty((0, 3))
        result = projector._project_raw_pixels(empty)
        assert result.shape == (0, 2)


# ============================================================
# FrustumCuller.cull_points 退化输入
# ============================================================

class TestCullPointsDegenerate:
    """测试 cull_points 的退化输入处理"""

    @pytest.fixture
    def culler(self):
        return FrustumCuller(
            FrustumType.PYRAMID,
            near_z=0.1,
            tan_h=1.0, tan_v=0.5
        )

    def test_empty_array(self, culler):
        """空数组应返回空结果"""
        empty = np.empty((0, 3))
        points, valid = culler.cull_points(empty)
        assert len(points) == 0
        assert len(valid) == 0

    def test_empty_1d_array(self, culler):
        """1D 空数组应返回空结果"""
        empty = np.array([])
        points, valid = culler.cull_points(empty)
        assert len(points) == 0
        assert len(valid) == 0

    def test_nan_filtered(self, culler):
        """含 NaN 的点应标记为无效"""
        points = np.array([
            [0, 0, 5],
            [np.nan, 0, 5],
            [0, 0, np.nan],
            [0, 0, 5],
        ])
        _, valid = culler.cull_points(points)
        assert valid[0] == True
        assert valid[1] == False
        assert valid[2] == False
        assert valid[3] == True

    def test_all_nan(self, culler):
        """全部 NaN 应全部无效"""
        points = np.full((3, 3), np.nan)
        _, valid = culler.cull_points(points)
        assert np.all(~valid)


# ============================================================
# FrustumCuller.clip_line 退化输入
# ============================================================

class TestClipLineDegenerate:
    """测试 clip_line 的退化输入处理"""

    @pytest.fixture
    def culler(self):
        return FrustumCuller(
            FrustumType.PYRAMID,
            near_z=0.1,
            tan_h=1.0, tan_v=0.5
        )

    def test_wrong_shape_p1_raises(self, culler):
        """p1 形状不是 (3,) 应抛出 ValueError"""
        with pytest.raises(ValueError, match="p1 must have shape"):
            culler.clip_line(np.zeros(2), np.zeros(3))

    def test_wrong_shape_p2_raises(self, culler):
        """p2 形状不是 (3,) 应抛出 ValueError"""
        with pytest.raises(ValueError, match="p2 must have shape"):
            culler.clip_line(np.zeros(3), np.zeros(4))

    def test_nan_endpoint_returns_none(self, culler):
        """端点含 NaN 应返回 None"""
        p1 = np.array([np.nan, 0, 5])
        p2 = np.array([1, 0, 5])
        result = culler.clip_line(p1, p2)
        assert result is None

    def test_nan_both_endpoints_returns_none(self, culler):
        """两端点都含 NaN 应返回 None"""
        p1 = np.array([0, np.nan, 5])
        p2 = np.array([1, 0, np.nan])
        result = culler.clip_line(p1, p2)
        assert result is None

    def test_normal_line_still_works(self, culler):
        """正常线段不应受验证影响"""
        p1 = np.array([0, 0, 5])
        p2 = np.array([1, 0, 5])
        result = culler.clip_line(p1, p2)
        assert result is not None
