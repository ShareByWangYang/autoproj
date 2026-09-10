"""
conventions 模块测试

覆盖坐标系适配工具函数：
- build_transform: 外参构造（R+t / quat+wxyz / quat+xyzw）
- invert_transform: c2w↔w2c 互转（刚体变换精确逆）
- scale_intrinsics: 分辨率变更时内参缩放
- check_transform_sanity: 外参合理性检查（正交性/det/有限性）
- 预设轴变换矩阵正确性
"""
import numpy as np
import pytest
import warnings
from autoproj.conventions import (
    build_transform,
    invert_transform,
    scale_intrinsics,
    check_transform_sanity,
    AXIS_OPENGL_TO_OPENCV,
    AXIS_OPENCV_TO_OPENGL,
    AXIS_ROS_REP103_TO_OPENCV_XFYFZU,
)


class TestBuildTransform:
    """build_transform 外参构造"""

    def test_R_and_t(self):
        """R + t 构造"""
        R = np.eye(3)
        t = [1.0, 2.0, 3.0]
        T = build_transform(R=R, t=t)
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[:3, :3], R)
        np.testing.assert_array_almost_equal(T[:3, 3], t)

    def test_R_no_t(self):
        """R 无 t，默认平移为 0"""
        R = np.eye(3)
        T = build_transform(R=R)
        np.testing.assert_array_almost_equal(T[:3, 3], [0, 0, 0])

    def test_quat_wxyz(self):
        """四元数 wxyz 顺序"""
        # 单位四元数 [1,0,0,0] → 单位旋转
        T = build_transform(quat=[1, 0, 0, 0], quat_order='wxyz')
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_quat_xyzw(self):
        """四元数 xyzw 顺序（ROS/eigen 常用）"""
        # xyzw 的单位四元数 [0,0,0,1]
        T = build_transform(quat=[0, 0, 0, 1], quat_order='xyzw')
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_quat_90deg_z(self):
        """绕 z 轴 90° 旋转的四元数"""
        # wxyz: cos(45°)=0.7071, sin(45°)*z=0.7071
        s = np.sqrt(2) / 2
        T = build_transform(quat=[s, 0, 0, s], quat_order='wxyz')
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_array_almost_equal(T[:3, :3], expected_R)

    def test_quat_normalized(self):
        """非单位四元数自动归一化"""
        q = [2, 0, 0, 0]  # 模长 2，归一后 [1,0,0,0]
        T = build_transform(quat=q, quat_order='wxyz')
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_R_overrides_quat(self):
        """R 和 quat 同时提供时，R 优先"""
        R = np.eye(3) * 2  # 非 rotation，但用于验证优先
        T = build_transform(R=R, quat=[1, 0, 0, 0])
        np.testing.assert_array_almost_equal(T[:3, :3], R)

    def test_no_R_no_quat_raises(self):
        """R 和 quat 同时为 None 应抛 ValueError"""
        with pytest.raises(ValueError, match="必须提供"):
            build_transform()

    def test_bad_R_shape_raises(self):
        """R 非 (3,3) 应抛 ValueError"""
        with pytest.raises(ValueError, match="R 必须为"):
            build_transform(R=np.zeros((2, 2)))

    def test_bad_quat_length_raises(self):
        """quat 长度非 4 应抛 ValueError"""
        with pytest.raises(ValueError, match="quat 必须长度为 4"):
            build_transform(quat=[1, 0, 0])

    def test_bad_quat_order_raises(self):
        """quat_order 非法应抛 ValueError"""
        with pytest.raises(ValueError, match="quat_order"):
            build_transform(quat=[1, 0, 0, 0], quat_order='abcd')

    def test_bad_t_length_raises(self):
        """t 长度非 3 应抛 ValueError"""
        with pytest.raises(ValueError, match="t 必须长度为 3"):
            build_transform(R=np.eye(3), t=[1, 2])

    def test_bad_direction_raises(self):
        """direction 非法应抛 ValueError"""
        with pytest.raises(ValueError, match="direction"):
            build_transform(R=np.eye(3), direction='xyz')

    def test_direction_w2c(self):
        """direction='w2c' 不翻转"""
        R = np.eye(3)
        t = [1, 2, 3]
        T = build_transform(R=R, t=t, direction='w2c')
        np.testing.assert_array_almost_equal(T[:3, 3], t)

    def test_direction_c2w(self):
        """direction='c2w' 也不翻转（仅语义声明）"""
        R = np.eye(3)
        t = [1, 2, 3]
        T = build_transform(R=R, t=t, direction='c2w')
        np.testing.assert_array_almost_equal(T[:3, 3], t)

    def test_zero_quat_raises(self):
        """零四元数应抛 ValueError"""
        with pytest.raises(ValueError, match="模长为 0"):
            build_transform(quat=[0, 0, 0, 0])


class TestInvertTransform:
    """invert_transform c2w↔w2c 互转"""

    def test_identity(self):
        """单位矩阵的逆是单位矩阵"""
        T = np.eye(4)
        T_inv = invert_transform(T)
        np.testing.assert_array_almost_equal(T_inv, np.eye(4))

    def test_pure_translation(self):
        """纯平移的逆"""
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        T_inv = invert_transform(T)
        np.testing.assert_array_almost_equal(T_inv[:3, 3], [-1, -2, -3])

    def test_rotation_plus_translation(self):
        """旋转+平移的逆"""
        # 绕 z 轴 90° + 平移 [1, 0, 0]
        angle = np.pi / 2
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [1, 0, 0]
        T_inv = invert_transform(T)
        # T_inv @ T 应为单位矩阵
        result = T_inv @ T
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_roundtrip(self):
        """T @ invert(T) = I"""
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [5, 3, 2]
        T_inv = invert_transform(T)
        np.testing.assert_array_almost_equal(T @ T_inv, np.eye(4))
        np.testing.assert_array_almost_equal(T_inv @ T, np.eye(4))

    def test_bad_shape_raises(self):
        """非 (4,4) 应抛 ValueError"""
        with pytest.raises(ValueError, match="T 必须为"):
            invert_transform(np.zeros((3, 3)))


class TestScaleIntrinsics:
    """scale_intrinsics 内参缩放"""

    def test_no_scale(self):
        """相同分辨率不缩放"""
        fx, fy, cx, cy = 1000, 1000, 960, 540
        fx2, fy2, cx2, cy2 = scale_intrinsics(fx, fy, cx, cy, (1920, 1080), (1920, 1080))
        np.testing.assert_array_almost_equal([fx2, fy2, cx2, cy2], [fx, fy, cx, cy])

    def test_half_scale(self):
        """半分辨率缩放"""
        fx, fy, cx, cy = 1000, 1000, 960, 540
        fx2, fy2, cx2, cy2 = scale_intrinsics(fx, fy, cx, cy, (1920, 1080), (960, 540))
        assert fx2 == pytest.approx(500)
        assert fy2 == pytest.approx(500)
        assert cx2 == pytest.approx(479.75)  # (960+0.5)*0.5-0.5
        assert cy2 == pytest.approx(269.75)  # (540+0.5)*0.5-0.5

    def test_double_scale(self):
        """2x 分辨率缩放"""
        fx, fy, cx, cy = 500, 500, 480, 270
        fx2, fy2, cx2, cy2 = scale_intrinsics(fx, fy, cx, cy, (960, 540), (1920, 1080))
        assert fx2 == pytest.approx(1000)
        assert fy2 == pytest.approx(1000)
        assert cx2 == pytest.approx(960.5)  # (480+0.5)*2-0.5
        assert cy2 == pytest.approx(540.5)  # (270+0.5)*2-0.5

    def test_pixel_center_alignment(self):
        """像素中心对齐：cx=0 缩放后应为 -0.25"""
        _, _, cx2, _ = scale_intrinsics(1000, 1000, 0, 0, (1920, 1080), (960, 540))
        assert cx2 == pytest.approx(-0.25)  # (0+0.5)*0.5-0.5

    def test_non_square_aspect(self):
        """非方形比例缩放"""
        fx, fy, cx, cy = 1000, 1000, 320, 240
        fx2, fy2, cx2, cy2 = scale_intrinsics(fx, fy, cx, cy, (640, 480), (1280, 720))
        assert fx2 == pytest.approx(2000)    # 2x
        assert fy2 == pytest.approx(1500)   # 1.5x

    def test_bad_src_size_raises(self):
        """源分辨率为 0 应抛 ValueError"""
        with pytest.raises(ValueError, match="分辨率必须为正"):
            scale_intrinsics(1000, 1000, 960, 540, (0, 1080), (1920, 1080))

    def test_bad_dst_size_raises(self):
        """目标分辨率为负应抛 ValueError"""
        with pytest.raises(ValueError, match="分辨率必须为正"):
            scale_intrinsics(1000, 1000, 960, 540, (1920, 1080), (-1, 1080))


class TestCheckTransformSanity:
    """check_transform_sanity 外参合理性检查"""

    def test_valid_transform_no_warning(self):
        """合法变换不应发 warning"""
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_transform_sanity(T)  # 不应抛

    def test_non_orthogonal_R_warns(self):
        """非正交 R 应发 warning"""
        T = np.eye(4)
        T[:3, :3] = np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        with pytest.warns(UserWarning, match="非正交"):
            check_transform_sanity(T)

    def test_negative_det_warns(self):
        """det(R) < 0 应发 warning"""
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, 1.0, -1.0])  # 反射，det=-1
        with pytest.warns(UserWarning, match="镜像"):
            check_transform_sanity(T)

    def test_nan_warns(self):
        """NaN 应发 warning"""
        T = np.eye(4)
        T[0, 3] = np.nan
        with pytest.warns(UserWarning, match="NaN"):
            check_transform_sanity(T)

    def test_bad_shape_raises(self):
        """非 (4,4) 应抛 ValueError"""
        with pytest.raises(ValueError, match="T 必须为"):
            check_transform_sanity(np.zeros((3, 3)))


class TestAxisMatrices:
    """预设轴变换矩阵"""

    def test_opengl_to_opencv_diag(self):
        """OpenGL→OpenCV: diag(1, -1, -1, 1)"""
        np.testing.assert_array_equal(
            np.diag(AXIS_OPENGL_TO_OPENCV),
            [1, -1, -1, 1]
        )

    def test_opencv_to_opengl_equal(self):
        """OpenCV→OpenGL 与 OpenGL→OpenCV 相同（自逆）"""
        np.testing.assert_array_equal(AXIS_OPENCV_TO_OPENGL, AXIS_OPENGL_TO_OPENCV)

    def test_opengl_inverse_is_self(self):
        """OpenGL→OpenCV 矩阵自逆"""
        np.testing.assert_array_almost_equal(
            AXIS_OPENGL_TO_OPENCV @ AXIS_OPENGL_TO_OPENCV,
            np.eye(4)
        )

    def test_ros_to_opencv_basis(self):
        """ROS REP-103 → OpenCV 轴映射正确"""
        M = AXIS_ROS_REP103_TO_OPENCV_XFYFZU
        # ROS x(前) → OpenCV z(前)
        ros_front = np.array([1, 0, 0, 0])
        result = M @ ros_front
        np.testing.assert_array_almost_equal(result[:3], [0, 0, 1])
        # ROS y(左) → -OpenCV x(右)
        ros_left = np.array([0, 1, 0, 0])
        result = M @ ros_left
        np.testing.assert_array_almost_equal(result[:3], [-1, 0, 0])
        # ROS z(上) → -OpenCV y(下)
        ros_up = np.array([0, 0, 1, 0])
        result = M @ ros_up
        np.testing.assert_array_almost_equal(result[:3], [0, -1, 0])
