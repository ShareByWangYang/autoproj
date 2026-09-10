"""
Numba JIT 加速核心 kernel

提供 Liang-Barsky 线段裁剪的并行批量化实现, 用于 FrustumCuller.clip_lines_batch。

设计要点:
1. JIT 函数为顶层函数 (不能用 self), 所有参数显式传入
2. PYRAMID 类型用 Liang-Barsky 标量算法, prange 并行处理 N 条线
3. CONE (鱼眼) 类型暂保留 Python 实现, 因二次方程 + 折叠检测向量化复杂
4. 首次调用触发 JIT 编译 (~1s), 后续调用极速
5. 若 numba 不可用, FrustumCuller 自动回退到纯 Python for 循环

接口契约 (与 FrustumCuller.clip_line 一致):
- 输入: (N, 3) 起点数组 + (N, 3) 终点数组
- 输出: (N, 2, 3) 裁剪后端点 + (N,) bool 有效掩码
- 无效线段 (整段在视锥外) 对应 mask=False, 端点为 NaN
"""
import numpy as np

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# 阈值: N > 此值时启用 Numba parallel=True, 否则用 parallel=False
# (基准: 12 棱时 parallel=True 线程调度 ~22µs > parallel=False ~1.8µs)
# 小批量走 nopr 版本, 既消除 Python 循环开销又避免线程池调度开销
BATCH_PARALLEL_THRESHOLD = 32


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _clip_lines_pyramid_numba(
        p1s, p2s,
        near_z, tan_h, tan_v
    ):
        """
        Liang-Barsky 批量并行裁剪 (PYRAMID 视锥)

        视锥定义 (5 平面, 无远裁剪面):
        - Near:   z >= near_z
        - Left:   x >= -z * tan_h
        - Right:  x <=  z * tan_h
        - Bottom: y >= -z * tan_v
        - Top:    y <=  z * tan_v

        Args:
            p1s: (N, 3) float64 起点数组
            p2s: (N, 3) float64 终点数组
            near_z, tan_h, tan_v: 视锥参数 (Python float)

        Returns:
            clipped: (N, 2, 3) 裁剪后端点 [p1_clipped, p2_clipped]
            valid: (N,) bool 有效掩码 (False = 整段在视锥外)
        """
        N = p1s.shape[0]
        clipped = np.empty((N, 2, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=np.bool_)

        for i in prange(N):
            x1 = p1s[i, 0]; y1 = p1s[i, 1]; z1 = p1s[i, 2]
            x2 = p2s[i, 0]; y2 = p2s[i, 1]; z2 = p2s[i, 2]

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            t_enter = 0.0
            t_exit = 1.0
            ok = True  # False 表示整段在视锥外

            # Near plane: z = near_z,  inside: z >= near_z
            # t 满足 z1 + t*dz >= near_z  =>  num = near_z - z1, denom = dz
            num = near_z - z1
            if dz > 0.0:
                t = num / dz
                if t > t_enter:
                    t_enter = t
            elif dz < 0.0:
                t = num / dz
                if t < t_exit:
                    t_exit = t
            else:
                # dz == 0, 线段平行于 near plane
                if num > 0.0:
                    # z1 < near_z 且 dz=0, 整段在 near plane 后
                    ok = False

            if ok:
                # Left plane: x + z*tan_h >= 0  =>  num = -(x1 + z1*tan_h), denom = dx + dz*tan_h
                num = -(x1 + z1 * tan_h)
                denom = dx + dz * tan_h
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                # Right plane: x - z*tan_h <= 0  =>  num = x1 - z1*tan_h, denom = dz*tan_h - dx
                num = x1 - z1 * tan_h
                denom = dz * tan_h - dx
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                # Bottom plane: y + z*tan_v >= 0  =>  num = -(y1 + z1*tan_v), denom = dy + dz*tan_v
                num = -(y1 + z1 * tan_v)
                denom = dy + dz * tan_v
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                # Top plane: y - z*tan_v <= 0  =>  num = y1 - z1*tan_v, denom = dz*tan_v - dy
                num = y1 - z1 * tan_v
                denom = dz * tan_v - dy
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                if t_enter > t_exit or t_exit < 0.0 or t_enter > 1.0:
                    ok = False
                else:
                    if t_enter < 0.0:
                        t_enter = 0.0
                    if t_exit > 1.0:
                        t_exit = 1.0
                    # 计算 p_enter, p_exit
                    clipped[i, 0, 0] = x1 + t_enter * dx
                    clipped[i, 0, 1] = y1 + t_enter * dy
                    clipped[i, 0, 2] = z1 + t_enter * dz
                    clipped[i, 1, 0] = x1 + t_exit * dx
                    clipped[i, 1, 1] = y1 + t_exit * dy
                    clipped[i, 1, 2] = z1 + t_exit * dz
                    valid[i] = True

            if not ok:
                # 无效线段填充 NaN (保持形状一致)
                clipped[i, 0, 0] = np.nan
                clipped[i, 0, 1] = np.nan
                clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan
                clipped[i, 1, 1] = np.nan
                clipped[i, 1, 2] = np.nan
                valid[i] = False

        return clipped, valid

    # parallel=False 版本: 小批量 (N≤BATCH_PARALLEL_THRESHOLD) 专用
    # 数学逻辑与 parallel=True 版本完全一致, 仅 prange→range + 无 parallel=True
    # 避免 Numba 线程池调度开销 (~20µs), 在 12 棱量级比 parallel 快 ~12x
    @njit(fastmath=True, cache=True)
    def _clip_lines_pyramid_numba_nopr(
        p1s, p2s,
        near_z, tan_h, tan_v
    ):
        """Liang-Barsky 批量裁剪 (PYRAMID, 单线程 JIT)

        与 _clip_lines_pyramid_numba 数学完全一致, 仅取消 parallel/prange,
        供 N≤BATCH_PARALLEL_THRESHOLD 的小批量路径调用 (如单框 project_box)。
        """
        N = p1s.shape[0]
        clipped = np.empty((N, 2, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=np.bool_)

        for i in range(N):
            x1 = p1s[i, 0]; y1 = p1s[i, 1]; z1 = p1s[i, 2]
            x2 = p2s[i, 0]; y2 = p2s[i, 1]; z2 = p2s[i, 2]

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            t_enter = 0.0
            t_exit = 1.0
            ok = True

            num = near_z - z1
            if dz > 0.0:
                t = num / dz
                if t > t_enter:
                    t_enter = t
            elif dz < 0.0:
                t = num / dz
                if t < t_exit:
                    t_exit = t
            else:
                if num > 0.0:
                    ok = False

            if ok:
                num = -(x1 + z1 * tan_h)
                denom = dx + dz * tan_h
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                num = x1 - z1 * tan_h
                denom = dz * tan_h - dx
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                num = -(y1 + z1 * tan_v)
                denom = dy + dz * tan_v
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                num = y1 - z1 * tan_v
                denom = dz * tan_v - dy
                if denom > 0.0:
                    t = num / denom
                    if t > t_enter:
                        t_enter = t
                elif denom < 0.0:
                    t = num / denom
                    if t < t_exit:
                        t_exit = t
                else:
                    if num > 0.0:
                        ok = False

            if ok:
                if t_enter > t_exit or t_exit < 0.0 or t_enter > 1.0:
                    ok = False
                else:
                    if t_enter < 0.0:
                        t_enter = 0.0
                    if t_exit > 1.0:
                        t_exit = 1.0
                    clipped[i, 0, 0] = x1 + t_enter * dx
                    clipped[i, 0, 1] = y1 + t_enter * dy
                    clipped[i, 0, 2] = z1 + t_enter * dz
                    clipped[i, 1, 0] = x1 + t_exit * dx
                    clipped[i, 1, 1] = y1 + t_exit * dy
                    clipped[i, 1, 2] = z1 + t_exit * dz
                    valid[i] = True

            if not ok:
                clipped[i, 0, 0] = np.nan
                clipped[i, 0, 1] = np.nan
                clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan
                clipped[i, 1, 1] = np.nan
                clipped[i, 1, 2] = np.nan
                valid[i] = False

        return clipped, valid

    @njit(parallel=True, fastmath=True, cache=True)
    def _clip_lines_cone_numba(
        p1s, p2s,
        near_z, cos_theta_max, sin_theta_max, large_fov
    ):
        """
        锥形视锥 (鱼眼) 批量并行裁剪

        注意: 此实现仅做近裁剪面 + 简单角度过滤的近似版本,
        对 θ_max ≤ 90° 使用代数式 (x²+y²)*cos² - z²*sin² <= 0,
        对 θ_max > 90° (large_fov=True) 仅做近裁剪面过滤,
        不处理精确的边界交点计算 (那需要解二次方程并做折叠检测,
        向量化复杂度高, 当前版本留给上层 Python 路径处理).

        若调用方需要精确的鱼眼裁剪结果, 应使用 FrustumCuller.clip_line
        单条循环版本, 而非此批量近似版本.
        """
        N = p1s.shape[0]
        clipped = np.empty((N, 2, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=np.bool_)

        cos2 = cos_theta_max * cos_theta_max
        sin2 = sin_theta_max * sin_theta_max

        for i in prange(N):
            x1 = p1s[i, 0]; y1 = p1s[i, 1]; z1 = p1s[i, 2]
            x2 = p2s[i, 0]; y2 = p2s[i, 1]; z2 = p2s[i, 2]

            # 仅做近裁剪面 + 端点角度过滤的近似裁剪
            # 精确边界交点求解留给上层 Python 路径处理
            r1_sq = x1 * x1 + y1 * y1 + z1 * z1
            r2_sq = x2 * x2 + y2 * y2 + z2 * z2
            near_sq = near_z * near_z

            # 端点是否在视锥内 (近似判断)
            if r1_sq >= near_sq and r2_sq >= near_sq:
                if large_fov:
                    # θ > 90°, 仅检查 r >= near_z (代数式失效)
                    in1 = True
                    in2 = True
                else:
                    # θ ≤ 90°, 使用代数式
                    # 内: (x²+y²)*cos² - z²*sin² <= 0
                    in1 = (x1 * x1 + y1 * y1) * cos2 - z1 * z1 * sin2 <= 0.0
                    in2 = (x2 * x2 + y2 * y2) * cos2 - z2 * z2 * sin2 <= 0.0

                if in1 and in2:
                    # 两端点均在视锥内, 直接返回原线段
                    clipped[i, 0, 0] = x1; clipped[i, 0, 1] = y1; clipped[i, 0, 2] = z1
                    clipped[i, 1, 0] = x2; clipped[i, 1, 1] = y2; clipped[i, 1, 2] = z2
                    valid[i] = True
                else:
                    # 至少一端在视锥外, 标记无效交由上层精确处理
                    clipped[i, 0, 0] = np.nan
                    clipped[i, 0, 1] = np.nan
                    clipped[i, 0, 2] = np.nan
                    clipped[i, 1, 0] = np.nan
                    clipped[i, 1, 1] = np.nan
                    clipped[i, 1, 2] = np.nan
                    valid[i] = False
            else:
                clipped[i, 0, 0] = np.nan
                clipped[i, 0, 1] = np.nan
                clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan
                clipped[i, 1, 1] = np.nan
                clipped[i, 1, 2] = np.nan
                valid[i] = False

        return clipped, valid

    @njit(fastmath=True, cache=True)
    def _cone_is_inside(px, py, pz, near_sq, cos_theta_max, cos_sq, sin_sq, large_fov):
        """锥形视锥内外判断 (标量, 供 _clip_lines_cone_precise_numba 调用).

        - large_fov=True  (θ_max > 90°): 几何式 z/r >= cos_theta_max
        - large_fov=False (θ_max ≤ 90°): 代数式 (x²+y²)*cos² - z²*sin² <= 0

        近裁剪面通过 r >= near_z (即 r² >= near_sq) 判断.
        光线必须从相机前方进入 (z > 0), 否则视为视锥外.
        """
        if pz <= 0.0:
            return False
        r_sq = px * px + py * py + pz * pz
        if r_sq < near_sq:
            return False
        if large_fov:
            r = np.sqrt(r_sq)
            return pz / r >= cos_theta_max
        else:
            return (px * px + py * py) * cos_sq - pz * pz * sin_sq <= 0.0

    @njit(parallel=True, fastmath=True, cache=True)
    def _clip_lines_cone_precise_numba(
        p1s, p2s,
        near_z, cos_theta_max, sin_theta_max, large_fov
    ):
        """
        锥形视锥精确批量裁剪 (替代近似版 _clip_lines_cone_numba)

        完整实现 _clip_line_cone / _clip_line_large_fov 的标量逻辑:
        1. 内外判断 (near_z 距离 + cone 角度)
        2. 候选点收集 (近球面二次方程 + 锥面二次方程 + 端点)
        3. 排序去重, 找中点在视锥内的最长子区间

        替代近似版的 NaN 标记策略, 让所有线段在 Numba 内一次性精确处理,
        消除上层 needs_precise 路径的 Python for 循环 (FTheta 负优化根因).
        """
        N = p1s.shape[0]
        clipped = np.empty((N, 2, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=np.bool_)

        cos_sq = cos_theta_max * cos_theta_max
        sin_sq = sin_theta_max * sin_theta_max
        near_sq = near_z * near_z

        # 候选点缓冲 (最多 7 个: 2 端点 + 2 near + 2 cone + 1 z=0平面)
        for i in prange(N):
            x1 = p1s[i, 0]; y1 = p1s[i, 1]; z1 = p1s[i, 2]
            x2 = p2s[i, 0]; y2 = p2s[i, 1]; z2 = p2s[i, 2]

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            inside1 = _cone_is_inside(x1, y1, z1, near_sq, cos_theta_max,
                                       cos_sq, sin_sq, large_fov)
            inside2 = _cone_is_inside(x2, y2, z2, near_sq, cos_theta_max,
                                       cos_sq, sin_sq, large_fov)

            # 两端都在视锥内: 直接返回原线段
            if inside1 and inside2:
                clipped[i, 0, 0] = x1; clipped[i, 0, 1] = y1; clipped[i, 0, 2] = z1
                clipped[i, 1, 0] = x2; clipped[i, 1, 1] = y2; clipped[i, 1, 2] = z2
                valid[i] = True
                continue

            # 候选点收集 (定长数组)
            cand_t = np.empty(7, dtype=np.float64)
            cand_x = np.empty(7, dtype=np.float64)
            cand_y = np.empty(7, dtype=np.float64)
            cand_z = np.empty(7, dtype=np.float64)
            n_cand = 0

            # 1) 端点 (如果在内)
            if inside1:
                cand_t[n_cand] = 0.0
                cand_x[n_cand] = x1; cand_y[n_cand] = y1; cand_z[n_cand] = z1
                n_cand += 1
            if inside2:
                cand_t[n_cand] = 1.0
                cand_x[n_cand] = x2; cand_y[n_cand] = y2; cand_z[n_cand] = z2
                n_cand += 1

            # 2) 近球面交点: |p1 + t*d|² = near_z²  =>  a_r*t² + b_r*t + c_r = 0
            a_r = dx * dx + dy * dy + dz * dz
            if a_r > 1e-12:
                b_r = 2.0 * (x1 * dx + y1 * dy + z1 * dz)
                c_r = x1 * x1 + y1 * y1 + z1 * z1 - near_sq
                disc_r = b_r * b_r - 4.0 * a_r * c_r
                if disc_r >= 0.0:
                    sqrt_disc_r = np.sqrt(disc_r)
                    t1_r = (-b_r - sqrt_disc_r) / (2.0 * a_r)
                    t2_r = (-b_r + sqrt_disc_r) / (2.0 * a_r)
                    # t1_r
                    if 0.0 <= t1_r <= 1.0:
                        px = x1 + t1_r * dx
                        py = y1 + t1_r * dy
                        pz = z1 + t1_r * dz
                        if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                            cos_sq, sin_sq, large_fov):
                            cand_t[n_cand] = t1_r
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1
                    # t2_r
                    if 0.0 <= t2_r <= 1.0:
                        px = x1 + t2_r * dx
                        py = y1 + t2_r * dy
                        pz = z1 + t2_r * dz
                        if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                            cos_sq, sin_sq, large_fov):
                            cand_t[n_cand] = t2_r
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1

            # 3) 锥面交点: (x²+y²)*cos² - z²*sin² = 0  =>  a*t² + b*t + c = 0
            a = (dx * dx + dy * dy) * cos_sq - dz * dz * sin_sq
            b = 2.0 * (x1 * dx + y1 * dy) * cos_sq - 2.0 * z1 * dz * sin_sq
            c = (x1 * x1 + y1 * y1) * cos_sq - z1 * z1 * sin_sq

            if abs(a) > 1e-12:
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_disc = np.sqrt(disc)
                    t1_c = (-b - sqrt_disc) / (2.0 * a)
                    t2_c = (-b + sqrt_disc) / (2.0 * a)
                    # t1_c
                    if 0.0 <= t1_c <= 1.0:
                        px = x1 + t1_c * dx
                        py = y1 + t1_c * dy
                        pz = z1 + t1_c * dz
                        if px * px + py * py + pz * pz >= near_sq:
                            # 只保留真实视锥边界交点（z 与 cos 同号）
                            if pz * cos_theta_max >= 0.0:
                                cand_t[n_cand] = t1_c
                                cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                                n_cand += 1
                    # t2_c
                    if 0.0 <= t2_c <= 1.0:
                        px = x1 + t2_c * dx
                        py = y1 + t2_c * dy
                        pz = z1 + t2_c * dz
                        if px * px + py * py + pz * pz >= near_sq:
                            # 只保留真实视锥边界交点（z 与 cos 同号）
                            if pz * cos_theta_max >= 0.0:
                                cand_t[n_cand] = t2_c
                                cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                                n_cand += 1
            elif abs(b) > 1e-12:
                # 退化为一元一次方程
                t_c = -c / b
                if 0.0 <= t_c <= 1.0:
                    px = x1 + t_c * dx
                    py = y1 + t_c * dy
                    pz = z1 + t_c * dz
                    if px * px + py * py + pz * pz >= near_sq:
                        # 只保留真实视锥边界交点（z 与 cos 同号）
                        if pz * cos_theta_max >= 0.0:
                            cand_t[n_cand] = t_c
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1

            # 4) z=0 平面交点 (光线必须从相机前方进入)
            #    z1 + t*dz = 0  =>  t = -z1 / dz
            if abs(dz) > 1e-12:
                t_z0 = -z1 / dz
                if 0.0 <= t_z0 <= 1.0:
                    px = x1 + t_z0 * dx
                    py = y1 + t_z0 * dy
                    pz = z1 + t_z0 * dz
                    if px * px + py * py + pz * pz >= near_sq:
                        cand_t[n_cand] = t_z0
                        cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                        n_cand += 1

            if n_cand < 2:
                # 候选点不足, 整段在视锥外
                clipped[i, 0, 0] = np.nan; clipped[i, 0, 1] = np.nan; clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan; clipped[i, 1, 1] = np.nan; clipped[i, 1, 2] = np.nan
                valid[i] = False
                continue

            # 简单冒泡排序 by t (n_cand <= 6, 简单排序足够)
            for ii in range(n_cand):
                for jj in range(ii + 1, n_cand):
                    if cand_t[jj] < cand_t[ii]:
                        tmp = cand_t[ii]; cand_t[ii] = cand_t[jj]; cand_t[jj] = tmp
                        tmp = cand_x[ii]; cand_x[ii] = cand_x[jj]; cand_x[jj] = tmp
                        tmp = cand_y[ii]; cand_y[ii] = cand_y[jj]; cand_y[jj] = tmp
                        tmp = cand_z[ii]; cand_z[ii] = cand_z[jj]; cand_z[jj] = tmp

            # 去重 (相邻 t 差距 < 1e-9 视为同一点)
            best_length = -1.0
            best_sx = np.nan; best_sy = np.nan; best_sz = np.nan
            best_ex = np.nan; best_ey = np.nan; best_ez = np.nan
            prev_t = cand_t[0]
            found_valid = False

            for ii in range(1, n_cand):
                t_start = prev_t
                t_end = cand_t[ii]
                # 跳过重复点
                if abs(t_end - t_start) <= 1e-9:
                    prev_t = t_end
                    continue

                t_mid = 0.5 * (t_start + t_end)
                px = x1 + t_mid * dx
                py = y1 + t_mid * dy
                pz = z1 + t_mid * dz
                if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                    cos_sq, sin_sq, large_fov):
                    length = t_end - t_start
                    if length > best_length:
                        best_length = length
                        # 找 t_start 对应的候选点 (prev_t 之前的最后一个)
                        best_sx = cand_x[ii - 1]
                        best_sy = cand_y[ii - 1]
                        best_sz = cand_z[ii - 1]
                        best_ex = cand_x[ii]
                        best_ey = cand_y[ii]
                        best_ez = cand_z[ii]
                        found_valid = True
                prev_t = t_end

            if found_valid:
                clipped[i, 0, 0] = best_sx; clipped[i, 0, 1] = best_sy; clipped[i, 0, 2] = best_sz
                clipped[i, 1, 0] = best_ex; clipped[i, 1, 1] = best_ey; clipped[i, 1, 2] = best_ez
                valid[i] = True
            else:
                clipped[i, 0, 0] = np.nan; clipped[i, 0, 1] = np.nan; clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan; clipped[i, 1, 1] = np.nan; clipped[i, 1, 2] = np.nan
                valid[i] = False

        return clipped, valid

    # parallel=False 版本: 小批量 (N≤BATCH_PARALLEL_THRESHOLD) 专用
    # 与 _clip_lines_cone_precise_numba 数学完全一致, 仅 prange→range + 无 parallel
    @njit(fastmath=True, cache=True)
    def _clip_lines_cone_precise_numba_nopr(
        p1s, p2s,
        near_z, cos_theta_max, sin_theta_max, large_fov
    ):
        """锥形视锥精确批量裁剪 (单线程 JIT)

        与 _clip_lines_cone_precise_numba 数学完全一致, 仅取消 parallel/prange,
        供 N≤BATCH_PARALLEL_THRESHOLD 的小批量路径调用。
        """
        N = p1s.shape[0]
        clipped = np.empty((N, 2, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=np.bool_)

        cos_sq = cos_theta_max * cos_theta_max
        sin_sq = sin_theta_max * sin_theta_max
        near_sq = near_z * near_z

        for i in range(N):
            x1 = p1s[i, 0]; y1 = p1s[i, 1]; z1 = p1s[i, 2]
            x2 = p2s[i, 0]; y2 = p2s[i, 1]; z2 = p2s[i, 2]

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            inside1 = _cone_is_inside(x1, y1, z1, near_sq, cos_theta_max,
                                       cos_sq, sin_sq, large_fov)
            inside2 = _cone_is_inside(x2, y2, z2, near_sq, cos_theta_max,
                                       cos_sq, sin_sq, large_fov)

            if inside1 and inside2:
                clipped[i, 0, 0] = x1; clipped[i, 0, 1] = y1; clipped[i, 0, 2] = z1
                clipped[i, 1, 0] = x2; clipped[i, 1, 1] = y2; clipped[i, 1, 2] = z2
                valid[i] = True
                continue

            cand_t = np.empty(7, dtype=np.float64)
            cand_x = np.empty(7, dtype=np.float64)
            cand_y = np.empty(7, dtype=np.float64)
            cand_z = np.empty(7, dtype=np.float64)
            n_cand = 0

            if inside1:
                cand_t[n_cand] = 0.0
                cand_x[n_cand] = x1; cand_y[n_cand] = y1; cand_z[n_cand] = z1
                n_cand += 1
            if inside2:
                cand_t[n_cand] = 1.0
                cand_x[n_cand] = x2; cand_y[n_cand] = y2; cand_z[n_cand] = z2
                n_cand += 1

            a_r = dx * dx + dy * dy + dz * dz
            if a_r > 1e-12:
                b_r = 2.0 * (x1 * dx + y1 * dy + z1 * dz)
                c_r = x1 * x1 + y1 * y1 + z1 * z1 - near_sq
                disc_r = b_r * b_r - 4.0 * a_r * c_r
                if disc_r >= 0.0:
                    sqrt_disc_r = np.sqrt(disc_r)
                    t1_r = (-b_r - sqrt_disc_r) / (2.0 * a_r)
                    t2_r = (-b_r + sqrt_disc_r) / (2.0 * a_r)
                    if 0.0 <= t1_r <= 1.0:
                        px = x1 + t1_r * dx
                        py = y1 + t1_r * dy
                        pz = z1 + t1_r * dz
                        if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                            cos_sq, sin_sq, large_fov):
                            cand_t[n_cand] = t1_r
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1
                    if 0.0 <= t2_r <= 1.0:
                        px = x1 + t2_r * dx
                        py = y1 + t2_r * dy
                        pz = z1 + t2_r * dz
                        if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                            cos_sq, sin_sq, large_fov):
                            cand_t[n_cand] = t2_r
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1

            a = (dx * dx + dy * dy) * cos_sq - dz * dz * sin_sq
            b = 2.0 * (x1 * dx + y1 * dy) * cos_sq - 2.0 * z1 * dz * sin_sq
            c = (x1 * x1 + y1 * y1) * cos_sq - z1 * z1 * sin_sq

            if abs(a) > 1e-12:
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_disc = np.sqrt(disc)
                    t1_c = (-b - sqrt_disc) / (2.0 * a)
                    t2_c = (-b + sqrt_disc) / (2.0 * a)
                    if 0.0 <= t1_c <= 1.0:
                        px = x1 + t1_c * dx
                        py = y1 + t1_c * dy
                        pz = z1 + t1_c * dz
                        if px * px + py * py + pz * pz >= near_sq:
                            if pz * cos_theta_max >= 0.0:
                                cand_t[n_cand] = t1_c
                                cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                                n_cand += 1
                    if 0.0 <= t2_c <= 1.0:
                        px = x1 + t2_c * dx
                        py = y1 + t2_c * dy
                        pz = z1 + t2_c * dz
                        if px * px + py * py + pz * pz >= near_sq:
                            if pz * cos_theta_max >= 0.0:
                                cand_t[n_cand] = t2_c
                                cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                                n_cand += 1
            elif abs(b) > 1e-12:
                t_c = -c / b
                if 0.0 <= t_c <= 1.0:
                    px = x1 + t_c * dx
                    py = y1 + t_c * dy
                    pz = z1 + t_c * dz
                    if px * px + py * py + pz * pz >= near_sq:
                        if pz * cos_theta_max >= 0.0:
                            cand_t[n_cand] = t_c
                            cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                            n_cand += 1

            if abs(dz) > 1e-12:
                t_z0 = -z1 / dz
                if 0.0 <= t_z0 <= 1.0:
                    px = x1 + t_z0 * dx
                    py = y1 + t_z0 * dy
                    pz = z1 + t_z0 * dz
                    if px * px + py * py + pz * pz >= near_sq:
                        cand_t[n_cand] = t_z0
                        cand_x[n_cand] = px; cand_y[n_cand] = py; cand_z[n_cand] = pz
                        n_cand += 1

            if n_cand < 2:
                clipped[i, 0, 0] = np.nan; clipped[i, 0, 1] = np.nan; clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan; clipped[i, 1, 1] = np.nan; clipped[i, 1, 2] = np.nan
                valid[i] = False
                continue

            for ii in range(n_cand):
                for jj in range(ii + 1, n_cand):
                    if cand_t[jj] < cand_t[ii]:
                        tmp = cand_t[ii]; cand_t[ii] = cand_t[jj]; cand_t[jj] = tmp
                        tmp = cand_x[ii]; cand_x[ii] = cand_x[jj]; cand_x[jj] = tmp
                        tmp = cand_y[ii]; cand_y[ii] = cand_y[jj]; cand_y[jj] = tmp
                        tmp = cand_z[ii]; cand_z[ii] = cand_z[jj]; cand_z[jj] = tmp

            best_length = -1.0
            best_sx = np.nan; best_sy = np.nan; best_sz = np.nan
            best_ex = np.nan; best_ey = np.nan; best_ez = np.nan
            prev_t = cand_t[0]
            found_valid = False

            for ii in range(1, n_cand):
                t_start = prev_t
                t_end = cand_t[ii]
                if abs(t_end - t_start) <= 1e-9:
                    prev_t = t_end
                    continue

                t_mid = 0.5 * (t_start + t_end)
                px = x1 + t_mid * dx
                py = y1 + t_mid * dy
                pz = z1 + t_mid * dz
                if _cone_is_inside(px, py, pz, near_sq, cos_theta_max,
                                    cos_sq, sin_sq, large_fov):
                    length = t_end - t_start
                    if length > best_length:
                        best_length = length
                        best_sx = cand_x[ii - 1]
                        best_sy = cand_y[ii - 1]
                        best_sz = cand_z[ii - 1]
                        best_ex = cand_x[ii]
                        best_ey = cand_y[ii]
                        best_ez = cand_z[ii]
                        found_valid = True
                prev_t = t_end

            if found_valid:
                clipped[i, 0, 0] = best_sx; clipped[i, 0, 1] = best_sy; clipped[i, 0, 2] = best_sz
                clipped[i, 1, 0] = best_ex; clipped[i, 1, 1] = best_ey; clipped[i, 1, 2] = best_ez
                valid[i] = True
            else:
                clipped[i, 0, 0] = np.nan; clipped[i, 0, 1] = np.nan; clipped[i, 0, 2] = np.nan
                clipped[i, 1, 0] = np.nan; clipped[i, 1, 1] = np.nan; clipped[i, 1, 2] = np.nan
                valid[i] = False

        return clipped, valid

else:
    # Numba 不可用时的占位 (FrustumCuller 检测后回退到 Python)
    _clip_lines_pyramid_numba = None
    _clip_lines_pyramid_numba_nopr = None
    _clip_lines_cone_numba = None
    _cone_is_inside = None
    _clip_lines_cone_precise_numba = None
    _clip_lines_cone_precise_numba_nopr = None
