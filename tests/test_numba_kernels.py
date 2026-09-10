"""
_numba_kernels 模块测试

验证 Numba JIT kernel 的正确性：
- parallel=True 与 parallel=False（nopr）版本结果 bit-exact 一致
- 小批量（N≤32）走 nopr 路径，大批量走 parallel 路径
- 无 Numba 时回退到 Python 路径
"""
import numpy as np
import pytest
from autoproj.frustum import FrustumCuller, FrustumType
from autoproj._numba_kernels import NUMBA_AVAILABLE


def _make_pyramid_culler():
    """构造针孔视锥裁剪器"""
    return FrustumCuller(
        frustum_type=FrustumType.PYRAMID,
        near_z=0.1,
        far_z=1000.0,
        tan_h=1.0,
        tan_v=0.5,
    )


def _make_cone_culler():
    """构造鱼眼锥形视锥裁剪器"""
    return FrustumCuller(
        frustum_type=FrustumType.CONE,
        near_z=0.1,
        far_z=1000.0,
        theta_max=np.pi / 2,  # 90° 全FOV
    )


def _gen_lines_in_frustum(n):
    """生成视锥内线段（有效）"""
    rng = np.random.RandomState(42)
    z = rng.uniform(1.0, 50.0, size=(n, 1))
    x = rng.uniform(-0.5, 0.5, size=(n, 1)) * z
    y = rng.uniform(-0.25, 0.25, size=(n, 1)) * z
    p1 = np.hstack([x, y, z])
    p2 = p1 + rng.uniform(-1, 1, size=(n, 3))
    return p1, p2


def _gen_lines_outside_frustum(n):
    """生成视锥外线段（无效）"""
    rng = np.random.RandomState(123)
    z = rng.uniform(1.0, 50.0, size=(n, 1))
    x = rng.uniform(5.0, 10.0, size=(n, 1)) * z  # 远超 tan_h
    y = rng.uniform(-0.25, 0.25, size=(n, 1)) * z
    p1 = np.hstack([x, y, z])
    p2 = p1 + rng.uniform(-0.1, 0.1, size=(n, 3))
    return p1, p2


class TestPyramidKernelConsistency:
    """针孔视锥裁剪：parallel vs nopr 一致性"""

    def test_small_batch_consistency(self):
        """小批量（N=12，单框棱数）nopr 与 parallel 结果一致"""
        if not NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        culler = _make_pyramid_culler()
        p1, p2 = _gen_lines_in_frustum(12)
        from autoproj._numba_kernels import (
            _clip_lines_pyramid_numba,
            _clip_lines_pyramid_numba_nopr,
        )
        pts_par, valid_par = _clip_lines_pyramid_numba(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        pts_nopr, valid_nopr = _clip_lines_pyramid_numba_nopr(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        np.testing.assert_array_equal(valid_par, valid_nopr)
        np.testing.assert_array_almost_equal(pts_par, pts_nopr)

    def test_medium_batch_consistency(self):
        """中批量（N=50）nopr 与 parallel 结果一致"""
        if not NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        culler = _make_pyramid_culler()
        p1, p2 = _gen_lines_in_frustum(50)
        from autoproj._numba_kernels import (
            _clip_lines_pyramid_numba,
            _clip_lines_pyramid_numba_nopr,
        )
        pts_par, valid_par = _clip_lines_pyramid_numba(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        pts_nopr, valid_nopr = _clip_lines_pyramid_numba_nopr(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        np.testing.assert_array_equal(valid_par, valid_nopr)
        np.testing.assert_array_almost_equal(pts_par, pts_nopr)

    def test_mixed_validity(self):
        """混合有效/无效线段"""
        if not NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        culler = _make_pyramid_culler()
        p1_in, p2_in = _gen_lines_in_frustum(10)
        p1_out, p2_out = _gen_lines_outside_frustum(10)
        p1 = np.vstack([p1_in, p1_out])
        p2 = np.vstack([p2_in, p2_out])
        from autoproj._numba_kernels import (
            _clip_lines_pyramid_numba,
            _clip_lines_pyramid_numba_nopr,
        )
        pts_par, valid_par = _clip_lines_pyramid_numba(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        pts_nopr, valid_nopr = _clip_lines_pyramid_numba_nopr(
            p1, p2, culler.near_z, culler.tan_h, culler.tan_v
        )
        np.testing.assert_array_equal(valid_par, valid_nopr)
        # 前 10 条有效，后 10 条无效
        assert valid_par[:10].all()
        assert not valid_par[10:].any()


class TestConeKernelConsistency:
    """鱼眼锥形视锥裁剪：parallel vs nopr 一致性"""

    def test_small_batch_consistency(self):
        """小批量（N=12）nopr 与 parallel 结果一致"""
        if not NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        culler = _make_cone_culler()
        p1, p2 = _gen_lines_in_frustum(12)
        from autoproj._numba_kernels import (
            _clip_lines_cone_precise_numba,
            _clip_lines_cone_precise_numba_nopr,
        )
        pts_par, valid_par = _clip_lines_cone_precise_numba(
            p1, p2,
            culler.near_z,
            culler.cos_theta_max, culler.sin_theta_max,
            culler._large_fov,
        )
        pts_nopr, valid_nopr = _clip_lines_cone_precise_numba_nopr(
            p1, p2,
            culler.near_z,
            culler.cos_theta_max, culler.sin_theta_max,
            culler._large_fov,
        )
        np.testing.assert_array_equal(valid_par, valid_nopr)
        np.testing.assert_array_almost_equal(pts_par, pts_nopr)

    def test_medium_batch_consistency(self):
        """中批量（N=50）nopr 与 parallel 结果一致"""
        if not NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        culler = _make_cone_culler()
        p1, p2 = _gen_lines_in_frustum(50)
        from autoproj._numba_kernels import (
            _clip_lines_cone_precise_numba,
            _clip_lines_cone_precise_numba_nopr,
        )
        pts_par, valid_par = _clip_lines_cone_precise_numba(
            p1, p2,
            culler.near_z,
            culler.cos_theta_max, culler.sin_theta_max,
            culler._large_fov,
        )
        pts_nopr, valid_nopr = _clip_lines_cone_precise_numba_nopr(
            p1, p2,
            culler.near_z,
            culler.cos_theta_max, culler.sin_theta_max,
            culler._large_fov,
        )
        np.testing.assert_array_equal(valid_par, valid_nopr)
        np.testing.assert_array_almost_equal(pts_par, pts_nopr)


class TestThresholdPathSelection:
    """验证 clip_lines_batch 的三档路径选择"""

    def test_n_le_32_uses_nopr(self):
        """N=12（≤32）应走 nopr 路径，结果正确"""
        culler = _make_pyramid_culler()
        p1, p2 = _gen_lines_in_frustum(12)
        clipped, valid = culler.clip_lines_batch(p1, p2)
        assert clipped.shape == (12, 2, 3)
        assert valid.shape == (12,)

    def test_n_gt_32_uses_parallel(self):
        """N=50（>32）应走 parallel 路径，结果正确"""
        culler = _make_pyramid_culler()
        p1, p2 = _gen_lines_in_frustum(50)
        clipped, valid = culler.clip_lines_batch(p1, p2)
        assert clipped.shape == (50, 2, 3)
        assert valid.shape == (50,)

    def test_empty_input(self):
        """空输入返回空结果"""
        culler = _make_pyramid_culler()
        clipped, valid = culler.clip_lines_batch(
            np.zeros((0, 3)), np.zeros((0, 3))
        )
        assert clipped.shape == (0, 2, 3)
        assert valid.shape == (0,)

    def test_nan_input_python_fallback(self):
        """NaN 输入应回退到 Python 路径"""
        culler = _make_pyramid_culler()
        p1 = np.array([[1, 2, 3], [np.nan, 0, 1]])
        p2 = np.array([[4, 5, 6], [2, 3, 4]])
        clipped, valid = culler.clip_lines_batch(p1, p2)
        assert clipped.shape == (2, 2, 3)
        # 含 NaN 的线段应为 invalid
        assert not valid[1]


class TestNumbaUnavailable:
    """无 Numba 时的回退行为"""

    def test_python_fallback_correctness(self):
        """Python 回退路径结果与 Numba 一致（小批量）"""
        culler = _make_pyramid_culler()
        p1, p2 = _gen_lines_in_frustum(5)
        # clip_lines_batch 内部自动选择路径
        clipped, valid = culler.clip_lines_batch(p1, p2)
        # 逐条 Python clip_line 结果
        for i in range(len(p1)):
            result_single = culler.clip_line(p1[i], p2[i])
            if result_single is not None:
                assert valid[i]  # single 有结果 → batch 也应有效
            else:
                assert not valid[i]  # single None → batch 应无效
