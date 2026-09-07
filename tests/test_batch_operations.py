"""
批量接口与数据量感知后端选择测试

测试覆盖:
1. project_boxes 与 project_box 单条结果一致性 (3种相机类型)
2. project_polygons 与 project_polygon 单条结果一致性 (3种相机类型)
3. FrustumCuller.clip_lines_batch 与 clip_line 单条结果一致性
4. BackendSelector.select(n_points, operation) 数据量感知选择
5. 退化输入: 空数组、NaN 值、形状错误
"""
import numpy as np
import pytest
from autoproj import CameraFactory
from autoproj.projection import Projector
from autoproj.backends import BackendSelector
from autoproj.frustum import FrustumCuller, FrustumType
from autoproj.geometry import Polygon3D


# ========== 相机 fixtures ==========
@pytest.fixture(params=[
    ('Pinhole', dict(category='pinhole', width=1920, height=1080,
                     fx=1000, fy=1000, cx=960, cy=540,
                     dist_coeffs=[0.05, -0.02, 0.001, -0.001, 0.01, 0.02, -0.005, 0.003])),
    ('KannalaBrandt', dict(category='fisheye', sub_type='kannala', width=1920, height=1080,
                            fx=500, fy=500, cx=960, cy=540,
                            k1=0.1, k2=0.05, k3=0.01, k4=0.001)),
    ('FTheta', dict(category='fisheye', sub_type='ftheta', width=1920, height=1080,
                    fw_poly=[0, 500, 50], cx=960, cy=540)),
])
def projector(request):
    """生成三种相机的 Projector fixture"""
    cam_name, kwargs = request.param
    # 默认用 numpy 后端, 便于交叉验证 (避免 CuPy/NumPy 类型差异)
    backend = BackendSelector.select(backend_name='numpy', auto_fallback=False)
    cam = CameraFactory.create(backend=backend, **kwargs)
    proj = Projector(cam, soft_clip_ratio=0.05)
    return proj, cam_name


# ========== 数据生成 ==========
def _gen_boxes(n: int, seed: int = 42) -> np.ndarray:
    """生成 n 个 (8, 3) 包围盒, 在相机前方"""
    rng = np.random.default_rng(seed)
    base = np.array([
        [0.5, -1.0, -1.0], [0.5,  1.0, -1.0], [0.5,  1.0, 1.0], [0.5, -1.0, 1.0],
        [2.5, -1.0, -1.0], [2.5,  1.0, -1.0], [2.5,  1.0, 1.0], [2.5, -1.0, 1.0],
    ], dtype=np.float64)
    offsets = rng.uniform(low=[-5, -5, 5], high=[5, 5, 30], size=(n, 3))
    return (base[None] + offsets[:, None]).astype(np.float64)  # (n, 8, 3)


def _gen_polygons(n: int, seed: int = 42):
    """生成 n 个多边形对象, 顶点数 3-7"""
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        m = rng.integers(3, 8)
        verts = rng.uniform(low=[-5, -5, 5], high=[5, 5, 30], size=(m, 3)).astype(np.float64)
        polys.append(Polygon3D(verts, is_closed=True))
    return polys


# ========== TestClipLinesBatch ==========
class TestClipLinesBatch:
    """测试 FrustumCuller.clip_lines_batch 与 clip_line 单条结果一致"""

    @pytest.fixture
    def pyramid_culler(self):
        return FrustumCuller(
            frustum_type=FrustumType.PYRAMID,
            near_z=0.1, far_z=1000.0,
            tan_h=0.5, tan_v=0.4,
            expansion_factor=1.0,
        )

    def test_empty_input(self, pyramid_culler):
        """空数组输入应返回空结果"""
        clipped, valid = pyramid_culler.clip_lines_batch(
            np.empty((0, 3)), np.empty((0, 3))
        )
        assert clipped.shape == (0, 2, 3)
        assert len(valid) == 0

    def test_shape_mismatch_raises(self, pyramid_culler):
        """p1s 和 p2s 形状不一致应抛出 ValueError"""
        with pytest.raises(ValueError, match="must have same shape"):
            pyramid_culler.clip_lines_batch(
                np.zeros((5, 3)), np.zeros((6, 3))
            )

    def test_wrong_ndim_raises(self, pyramid_culler):
        """p1s.ndim != 2 应抛出 ValueError"""
        with pytest.raises(ValueError, match="must have shape"):
            pyramid_culler.clip_lines_batch(
                np.zeros((5, 4, 3)), np.zeros((5, 4, 3))
            )

    def test_consistency_with_single_clip_line(self, pyramid_culler):
        """200 条线段与单条 clip_line 结果应完全一致"""
        np.random.seed(42)
        N = 200
        p1s = np.random.uniform(-5, 15, (N, 3))
        p2s = np.random.uniform(-5, 15, (N, 3))
        # 制造部分跨过 near plane 的线段
        p1s[:50, 2] = -1
        p2s[50:100, 2] = -1

        # 单条循环
        single_valid = []
        single_clipped = []
        for i in range(N):
            r = pyramid_culler.clip_line(p1s[i], p2s[i])
            if r is None:
                single_valid.append(False)
                single_clipped.append(None)
            else:
                p1, p2 = r
                near_sq = 0.1 ** 2
                r1 = p1[0]**2 + p1[1]**2 + p1[2]**2
                r2 = p2[0]**2 + p2[1]**2 + p2[2]**2
                if r1 >= near_sq and r2 >= near_sq:
                    single_valid.append(True)
                    single_clipped.append((p1, p2))
                else:
                    single_valid.append(False)
                    single_clipped.append(None)

        # 批量
        batch_clipped, batch_valid = pyramid_culler.clip_lines_batch(p1s, p2s)

        # 比对
        assert batch_clipped.shape == (N, 2, 3)
        assert len(batch_valid) == N
        mismatch = 0
        for i in range(N):
            if single_valid[i] != batch_valid[i]:
                mismatch += 1
                continue
            if single_valid[i]:
                d1 = np.abs(single_clipped[i][0] - batch_clipped[i, 0]).max()
                d2 = np.abs(single_clipped[i][1] - batch_clipped[i, 1]).max()
                if d1 > 1e-9 or d2 > 1e-9:
                    mismatch += 1
        assert mismatch == 0

    def test_nan_input_filtered(self, pyramid_culler):
        """含 NaN 的线段应被标记为无效 (NaN 端点对应位置保持 NaN)"""
        p1s = np.array([
            [1.0, 0.0, 10.0],   # 正常
            [np.nan, 0, 10],    # 含 NaN
            [0.0, 1.0, 10.0],   # 正常
        ])
        p2s = np.array([
            [2.0, 0.0, 10.0],
            [2.0, 0.0, 10.0],
            [0.0, 2.0, 10.0],
        ])
        clipped, valid = pyramid_culler.clip_lines_batch(p1s, p2s)
        # 第 1 条正常
        assert valid[0]
        # 第 2 条含 NaN, 无效
        assert not valid[1]
        # 第 3 条正常
        assert valid[2]


# ========== TestProjectBoxes ==========
class TestProjectBoxes:
    """测试 project_boxes 与 project_box 单条结果一致"""

    def test_empty_input(self, projector):
        proj, _ = projector
        results = proj.project_boxes([])
        assert results == []

    def test_single_box_as_ndarray(self, projector):
        """(8, 3) ndarray 应自动包装为单元素列表"""
        proj, _ = projector
        boxes = _gen_boxes(1)
        results = proj.project_boxes(boxes[0])  # (8, 3) ndarray
        assert len(results) == 1

    def test_consistency_3_cameras(self, projector):
        """200 个框, project_boxes 与 project_box 单条结果一致"""
        proj, cam_name = projector
        N = 200
        boxes = _gen_boxes(N)

        single_results = [proj.project_box(b, pts_in_cam=True) for b in boxes]
        batch_results = proj.project_boxes(boxes, pts_in_cam=True)

        assert len(batch_results) == N
        mismatch = 0
        total_edges = 0
        for i in range(N):
            s_edges = single_results[i]['edges']
            b_edges = batch_results[i]['edges']
            if len(s_edges) != len(b_edges):
                mismatch += 1
                continue
            total_edges += len(s_edges)
            for se, be in zip(s_edges, b_edges):
                for key in ['pt1', 'pt2', 'draw_pt1', 'draw_pt2']:
                    d = np.abs(se[key] - be[key]).max()
                    if d > 1e-6:
                        mismatch += 1
                        break
                if se['pt1_is_corner'] != be['pt1_is_corner']:
                    mismatch += 1
                    break
        assert mismatch == 0, f'{cam_name}: mismatch={mismatch}, total_edges={total_edges}'

    def test_nan_in_box_raises(self, projector):
        proj, _ = projector
        boxes = _gen_boxes(10)
        boxes[3, 0, 0] = np.nan
        with pytest.raises(ValueError, match="contain NaN"):
            proj.project_boxes(boxes, pts_in_cam=True)

    def test_wrong_shape_raises(self, projector):
        proj, _ = projector
        bad_boxes = np.zeros((10, 7, 3))  # 不是 (8, 3)
        with pytest.raises(ValueError, match="must be"):
            proj.project_boxes(bad_boxes, pts_in_cam=True)


# ========== TestProjectPolygons ==========
class TestProjectPolygons:
    """测试 project_polygons 与 project_polygon 单条结果一致"""

    def test_empty_input(self, projector):
        proj, _ = projector
        assert proj.project_polygons([]) == []

    def test_consistency_3_cameras(self, projector):
        proj, cam_name = projector
        N = 100
        polys = _gen_polygons(N)

        single_results = [proj.project_polygon(p, pts_in_cam=True) for p in polys]
        batch_results = proj.project_polygons(polys, pts_in_cam=True)

        assert len(batch_results) == N
        mismatch = 0
        total_edges = 0
        for i in range(N):
            s_edges = single_results[i]['edges']
            b_edges = batch_results[i]['edges']
            if len(s_edges) != len(b_edges):
                mismatch += 1
                continue
            total_edges += len(s_edges)
            for se, be in zip(s_edges, b_edges):
                for k in range(2):
                    d = np.abs(se[k] - be[k]).max()
                    if d > 1e-6:
                        mismatch += 1
                        break
        assert mismatch == 0, f'{cam_name}: mismatch={mismatch}, total_edges={total_edges}'

    def test_list_of_ndarray_input(self, projector):
        """List[ndarray] 输入应正确处理"""
        proj, _ = projector
        N = 50
        rng = np.random.default_rng(42)
        vert_list = [rng.uniform(low=[-5, -5, 5], high=[5, 5, 30],
                                 size=(np.random.randint(3, 7), 3)).astype(np.float64)
                     for _ in range(N)]
        results = proj.project_polygons(vert_list, pts_in_cam=True)
        assert len(results) == N
        # 每个 polygon 的 vertices 应有对应长度
        for i, r in enumerate(results):
            assert len(r['vertices']) == len(vert_list[i])


# ========== TestBackendSelectorDataAware ==========
class TestBackendSelectorDataAware:
    """测试数据量感知后端选择

    当前实现: 仅 numpy 可用, 所有 op 始终返回 numpy
    未来扩展: 重新引入 cuda/cpp 后, 测试需更新为多后端候选判断
    """

    def setup_method(self):
        BackendSelector.reset()

    def test_small_points_returns_numpy(self):
        """小规模点云 (10 点) 当前返回 numpy"""
        b = BackendSelector.select(n_points=10, operation='project_points')
        assert b.name() == 'numpy'

    def test_large_points_returns_numpy(self):
        """1M 点云当前也返回 numpy (未注册 cuda/cpp)"""
        b = BackendSelector.select(n_points=1_000_000, operation='project_points')
        assert b.name() == 'numpy'

    def test_box_operation_returns_numpy(self):
        """project_boxes 任意规模都返回 numpy"""
        for n in [10, 100, 10000, 1_000_000]:
            BackendSelector.reset()
            b = BackendSelector.select(n_points=n, operation='project_boxes')
            assert b.name() == 'numpy', f'n={n} expected numpy, got {b.name()}'

    def test_explicit_backend_name_overrides_n_points(self):
        """显式 backend_name 优先于 n_points 数据量感知"""
        # 显式指定 numpy 应直接返回 numpy (即使 n_points/operation 也传入)
        b = BackendSelector.select(backend_name='numpy', n_points=10, operation='project_boxes')
        assert b.name() == 'numpy'

    def test_unknown_operation_uses_default(self):
        """未知 operation 应使用 default 阈值 (当前仍返回 numpy)"""
        b = BackendSelector.select(n_points=100, operation='unknown_op')
        assert b.name() == 'numpy'
        BackendSelector.reset()
        b = BackendSelector.select(n_points=100000, operation='unknown_op')
        assert b.name() == 'numpy'

    def test_selection_cache_hit(self):
        """重复 (n_points, operation) 应命中缓存"""
        b1 = BackendSelector.select(n_points=100, operation='project_points')
        b2 = BackendSelector.select(n_points=100, operation='project_points')
        assert b1 is b2  # 同一实例 (因 _backends 懒加载)

    def test_reset_clears_cache(self):
        """reset 应清空缓存和后端实例"""
        BackendSelector.select(n_points=100, operation='project_points')
        assert len(BackendSelector._selection_cache) > 0
        assert BackendSelector._backends['numpy'] is not None
        BackendSelector.reset()
        assert len(BackendSelector._selection_cache) == 0
        assert BackendSelector._backends['numpy'] is None
