# Changelog

本项目所有重要变更均会记录在此文件中。
格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [2.1.2] - 2026-09-10

### 变更 (Changed)

- **删除 `backends/` 死代码骨架**：v0.x 时代遗留的可插拔后端架构
  （`Backend`/`NumPyBackend`/`BackendSelector`）完全未参与实际计算路径，
  `Projector` 与 `Camera` 已直接使用 numpy/numba。删除该目录及全部导出，
  清理 `__init__.py`/`camera.py`/`tests/`/`README.md` 中的相关引用
  - `Camera.__init__` 的 `backend` 参数已移除（此前仅赋值给 `self.backend`
    但从未被读取，是死属性）
  - 删除 `tests/test_backends.py`（16 项测试），清理
    `tests/test_batch_operations.py` 中的 `TestBackendSelectorDataAware` 类
- **`pyproject.toml` classifiers 补全 Python 3.12**：CI 矩阵已测试 3.12，
  classifiers 此前遗漏

### 性能优化

- **`clip_lines_batch` 三档路径选择**：小批量 (N≤32) 从 Python 逐条循环改为
  Numba `parallel=False` JIT，消除 Python 循环开销
  - 新增 `_clip_lines_pyramid_numba_nopr` 和 `_clip_lines_cone_precise_numba_nopr`
    两个单线程 JIT kernel（数学与 `parallel=True` 版本完全一致，仅取消
    `prange`/`parallel`，避免线程池调度开销）
  - 路径选择：N>32 走 `parallel=True`（多线程并行），0<N≤32 走 `parallel=False`
    （单线程 JIT），有 NaN 或 Numba 不可用走 Python 回退
  - 效果：单框 `project_box` 从 ~433µs 降至 ~135µs（**3.3x 提速**），
    64 框单元素循环 vs 批量差距从 4.3x 降至 **1.08x**（基本消除）
  - 12 棱裁剪：Python 161µs → Numba(nopr) 1.8µs（**89x 提速**）
- **`project_lines` 输入快速路径**：ndarray 输入走 `np.asarray` 零拷贝路径，
  list-of-tuples 输入才走 `np.asarray` 转换，120K 线段加速 2.5x
- **`Camera.project` 检查合并**：1M 点规模下，将 `_check_depth_range` +
  `_check_fov` 内联合并为单次 `np.logical_and.reduce()`，省 3 个临时 bool 数组
- **减少 `astype` 调用**：像素坐标转换 `np.stack([u, v]).astype(int32)` 替代
  两次单独 `astype`，`preserve_extra` 路径用 `np.trunc()` 替代 `astype(int32)`

### 测试

- 117 项 pytest 全通过（删除 16 项 backends 测试，原 133 项）
- bit-exact 一致性验证：3 相机 × 100 框 × 12 棱 = 3600 边，0 不一致

## [2.1.1] - 2026-09-09

### 变更 (Changed)

- **单元素/批量投影路径统一为真实向量化（架构对齐）**
  - 此前 `project_box`/`project_polygon` 单元素路径使用 Python for 循环逐条
    `clip_line` + 逐条投影 + set 角点匹配，而 `project_boxes`/`project_polygons`
    仅在 N 超过阈值（64/32）时才走向量化路径，导致阈值悬崖：
    N=64→65 框投影耗时从 38.6ms 暴跌至 1.9ms（20 倍反直觉跳变）
  - 现重构为"单元素是批量核心 N=1 的薄包装"：
    - 新增 `_project_boxes_core` 向量化核心（N≥1 通用），`project_box` 改为
      调用核心的 N=1 路径
    - `project_polygon` 改为调用 `project_polygons` 的 N=1 路径
    - 删除 `_BATCH_BOX_THRESHOLD`/`_BATCH_POLYGON_THRESHOLD` 阈值与小批量回退分支
    - 删除单元素路径的 Python 循环裁剪与 `_match_corner_pixel` 静态方法
  - 单元素与批量共享完全相同的批量变换/裁剪/投影/角点匹配/软裁剪数学，
    性能随棱数连续缩放（clip_lines_batch 内部 ≤32 棱走精确逐边裁剪、
    >32 棱自动切换 Numba 并行），无阈值悬崖
- **`project_polygon` 近裁剪面检查修正**：旧单元素路径用 `z <= 1e-6` 判定，
  会错误丢弃鱼眼相机 θ>90° 的 z<0 有效边缘点；现统一为径向距离检查
  （`r >= near_z`），与批量路径及鱼眼投影模型一致
- **`project_lines` 结果填充向量化**：消除外参 `hstack` 残留（改用
  `R @ p.T + t` 广播），结果列表预分配 + 按索引回填，无裁剪分支用
  向量化掩码判断线段有效性
- **`project_polygons` 边构造向量化**：边端点索引改用 `np.arange`/fancy-indexing
  构造，消除逐边 Python 追加循环；外参变换消除 `hstack`

### 修复 (Fixed)

- **恢复 `Projector._match_corner_pixel` 静态方法**：向量化重构曾移除该方法，
  导致直接调用它的外部可视化脚本报 `AttributeError`。现恢复为公共兼容入口
  （内部已改用 NumPy 广播匹配，此方法仅供外部调用方复用）。

### 性能优化

| 场景 | 重构前 | 重构后 | 提速 |
|---|---|---|---|
| 64 框投影（阈值悬崖点） | 38.6 ms | 1.94 ms | ~20x |
| 单框 project_box | 690 µs | 475 µs | 1.45x |
| 100 框投影 | 2.80 ms（已向量化） | 2.94 ms | 持平 |
| 1000 框投影 | 30.3 ms | 31.7 ms | 持平 |
| 120K 线段投影 | 314 ms | 280 ms | 1.12x |

### 测试

- 133 项 pytest 全通过
- 新增单-批一致性验证：16,445 条边（3 相机 × 裁剪开/关 × 框/多边形）
  单元素与批量结果 bit-exact（像素差 < 1e-9，角点标志 0 不一致）
- 阈值悬崖消除验证：N=64 与 N=65 耗时连续（1.94ms vs 2.00ms）

## [2.1.0] - 2026-09-07

### 新增 (Added)

- **`autoproj.conventions` 模块**：入口适配层，提供坐标系/外参/内参约定转换工具函数
  - `build_transform(R, t, quat, quat_order, direction)`：统一外参构造，支持 wxyz/xyzw 四元数顺序、显式声明 w2c/c2w 方向
  - `invert_transform(T)`：c2w ↔ w2c 互转（刚体变换精确逆）
  - `scale_intrinsics(fx, fy, cx, cy, src_size, dst_size)`：分辨率变更时缩放内参（像素中心对齐）
  - `check_transform_sanity(T, direction)`：外参合理性检查（R 正交性/det/有限性），发 warning 不抛异常
  - 3 个预设轴变换矩阵：`AXIS_OPENGL_TO_OPENCV`、`AXIS_ROS_REP103_TO_OPENCV_XFYFZU`、`AXIS_OPENCV_TO_ROS_REP103_XFYFZU`
- **`Projector(health_check=True)`**：投影健康检查开关
  - 诊断三类静默错误：z<0 比例异常（外参方向反）、valid 率 <5%（轴向/单位/模型族错配）、det(R)<0（轴镜像）
  - 仅发 warning 不自动修正，避免"打补丁式翻轴"连锁错误
- **README "Coordinate System Conventions" 章节**：显式钉死契约（OpenCV 约定/外参方向/单位/畸变不可混用）+ 常见数据源对齐表
- **GitHub Actions CI**（`.github/workflows/ci.yml`）：让 README 的 build badge 名副其实

### 变更 (Changed)

- **`pyproject.toml`**：版本 → 2.1.0；Development Status 升级为 `5 - Production/Stable`；
  project.urls 修正为实际仓库地址 `ShareByWangYang/autoproj`（此前误写为 `autoproj/autoproj`）
- **`autoproj/__init__.py`**：`__version__` → 2.1.0，导出 `conventions` 模块
- **`frustum.py`**：修正 docstring 中相机系 y 轴方向笔误（"y: up" → "y: down"），补全 OpenCV 坐标系契约说明
- **`camera.py`**：Camera 基类 docstring 新增坐标系契约块
- **`projection.py`**：Projector docstring 同步契约并提示 c2w → w2c 的转换路径
- **`README.md`**：补全 Projector API 表（此前缺失 `project_boxes`/`project_polygon`/`project_polygons`/`extend_edges_to_boundary`/`set_cull_frustum` 5 个方法）+ 新增 Conventions API 表

### 性能优化

- **`project_boxes` 向量化重写**（10K 框 3107ms → 410ms，**7.6x 提速**）：
  - 棱端点构造：双重 Python 循环 → NumPy fancy-indexing
  - 角点匹配：626k 次 `round()` + Python set 查找 → NumPy 广播 Chebyshev(L∞) 距离向量化
  - 软裁剪：逐边 Python 分支 → mask + `np.where`
  - 函数调用从 1.17M 次降至 50k 次
- **消除 `hstack` 三处**（`project_points` / `project_boxes` / `Camera._transform_to_camera`）：
  - 改为 `R @ p.T + t[:, None]` 广播，避免 (N,4) 齐次数组分配/拷贝
  - 1M 点外参开销 40ms → 10ms（4x）
- **`KannalaBrandtCamera` 清理**：
  - 删除主路径死代码 `r_3d = np.sqrt(...)`（赋值后从未被计算引用），1M 点省 4.2ms
  - 统一 `_project_raw_pixels` 的 KB 分支：`arccos + clip + r_3d` → `arctan2`，与主路径数学一致

### 修复 (Fixed)

- **`_project_raw_pixels` F-Theta 分支**：修复多项式属性名错误（`coeffs` 不存在 → `fw_poly`）+ 入射角公式错误（`arctan(r_xy)` → `arctan2(√(x²+y²), z)`），与 `FThetaCamera.project()` 主路径完全对齐
- **`projection.py` 注释**：清理 v1.x CuPy 残留说明（CuPy 在 2.0.0 已移除）

### 测试

- 133 项 pytest 全通过，无回归
- 34,722 条边逐边 bit-exact 对比（3 相机 × 软裁剪开/关）：像素差 0.0、corner 标志 0 不一致
- `project_points` 1M 点输出 `array_equal` 完全一致

## [2.0.0] - 2026-08-24

### 重大变更 (Major Breaking Changes)

- **移除 CUDA / C++ 后端**：基于完整基准测试 (`benchmark_results.csv`) 评估
  - 自动驾驶 10Hz 雷达逐帧可视化场景下, CUDA 仅 1.46x 加速但启动开销 1.5ms, 净收益 < 3% 帧预算
  - C++ 后端在 project_box/project_lines 上优势 < 1.5ms/帧, Numba JIT 已接管线段/框批量裁剪加速
  - 净减少 ~2500 行代码 (cuda_backend.py / cpp_backend.py / _projection_cpp.cpp / build_utils.py)
  - 消除 cupy/CuPy 类型混合 bug 风险 (self.np / _to_numpy / _align_to_np 兼容层)
  - 消除首次使用 C++ 自动编译可能失败的问题
  - 按 SemVer 规范, 破坏性 API 变更 (移除已导出的 CUDABackend/CPythonBackend 类) 触发主版本号升级 1.x → 2.0.0
- **保留后端架构骨架**：为未来重新引入 cuda/cpp 后端预留扩展点
  - `BackendSelector` 类结构保留, `_backends` 字典仅注册 'numpy'
  - `_select_by_data_size` 接口保留, 当前始终返回 'numpy'
  - `_BACKEND_THRESHOLDS` 表保留结构, 仅含 numpy 项
  - `Backend` 抽象基类 (`base.py`) 完整保留
  - 未来扩展只需: (1) 加入 cpp_backend.py/cuda_backend.py (2) 在 _get_backend 实现懒加载 (3) 更新 _BACKEND_THRESHOLDS

### 变更 (Changed)

- **`autoproj/__init__.py`**：移除 `CUDABackend`, `CPythonBackend` 导出, `__version__` 更新为 `2.0.0`
- **`camera.py`**：移除 `self.np` / `_to_numpy` / `_align_to_np` / `hasattr(self.backend, 'project_xxx')` 分支
  - 三个相机类 (Pinhole/KannalaBrandt/FTheta) 的 `project()` 仅保留 NumPy 向量化路径作为唯一实现
- **`projection.py`**：移除 `np = self.np if hasattr(self, 'np') else __import__('numpy')` 兼容层
- **`setup.py`**：移除 pybind11 / BuildExt / C++ 扩展配置, 简化为纯 Python 包
- **`pyproject.toml`**：
  - 版本统一为 `2.0.0` (代码库 `__version__` / `pyproject.toml` / `examples/benchmark.py` 全部对齐)
  - 移除 `cuda` 可选依赖 (cupy)
  - 移除 `dev` 中 pybind11
  - 新增 `jit` 可选依赖 (numba)
  - 移除 classifiers 中 "Programming Language :: C++"

### 测试更新

- **`test_backends.py`**：删除 cuda/cpp 相关测试, 新增 `test_select_unknown_raises` / `test_data_aware_returns_numpy` / `test_data_aware_cache`
- **`test_batch_operations.py::TestBackendSelectorDataAware`**：所有断言改为 `assert b.name() == 'numpy'`
- **删除**：`test_cpp_backend.py` / `test_auto_build.py`

### 性能基准 (保留)

- project_points 100K 点: numpy 6.0ms (10Hz 雷达 100ms 帧预算内)
- project_boxes 10K 框 (Pinhole): numpy + Numba 批量接口 14x 加速 vs 单条循环
- project_lines 1200 线段 (Pinhole): numpy + Numba 3.95x 加速 vs Python for 循环
- project_boxes FTheta 10K 框: numpy + Numba 精确版 2.24x 加速 (原近似版 0.63x 负优化已修复)

## [1.0.0] - 2026-08-24

### 新增 (Added)

- **`max_fov_deg` 参数**：替代 `max_fov_half_angle`，以度数设置鱼眼相机 FOV 上限
  - `CameraFactory.create_fisheye()` 和 `create_pinhole()` 支持 `max_fov_deg` 参数
  - `KannalaBrandtCamera` 和 `FThetaCamera` 在 `_compute_theta_max()` 中使用该参数限制搜索范围
  - 默认 `None` 表示无上限（最大可用 FOV）
  - 可视化代码支持从 `camera_config_dict` 读取配置
- **`frustum_scale` 参数**：合并 `fov_tolerance` 和 `frustum_expansion` 为单一参数
  - `None` = 自动计算（调用 `camera.compute_expansion_factor()`）
  - `float` = 手动缩放因子（1.0 = 无缩放）
  - `Projector(camera, frustum_scale=None)` 替代 `Projector(camera, frustum_expansion='auto')`
- **`draw_pt1`/`draw_pt2` 返回值**：在 `project_box` 中直接返回实际绘制端点，简化可视化代码绘制逻辑
- **`set_frustum_scale()` 方法**：替代 `set_frustum_expansion()`，支持动态更新视锥缩放因子
- **`test_fov_limits.py` 测试文件**：覆盖大FOV场景
  - `max_fov_deg` 参数传递测试
  - 大FOV（θ > 90°）渲染测试
  - FrustumCuller 大FOV裁剪测试
  - FOV一致性测试

### 变更 (Changed)

- **参数体系重构**：
  - 移除 `margin`，统一使用 `boundary_ratio`（边界余量比例）
  - `cull_frustum` 由 `Union[bool, float]` 改为纯 `bool`
  - `Projector` 构造函数参数调整：`frustum_scale=None` 替代 `frustum_expansion='auto'`
- **`_check_bounds` 方法签名简化**：移除冗余的 `margin` 参数，方法内直接调用 `_get_dynamic_margin()` 计算边界余量
- **向后兼容**：`frustum_expansion` 旧参数名通过 `**kwargs` 接收并发出 `DeprecationWarning`，`'auto'` 映射为 `None`
- **可视化代码整合**：移除重复实现，统一使用 autoproj 库
  - 删除 `_extend_edge_to_boundary()` 静态方法（重复实现）
  - 删除 `_match_corner()` 静态方法，改用 `Projector._match_corner_pixel()`
  - `clip_line_to_frustum()` 和 `_project_box_legacy()` 标记为 `[DEPRECATED]`

### 修复 (Fixed)

- **版本号不一致**：`__init__.py` 中 `__version__` 更新为 `1.0.0`
- **`set_frustum_expansion` 属性名错误**：错误设置 `self.frustum_expansion`（不存在的属性），修复为正确设置 `self._frustum_scale`，并新增 `set_frustum_scale()` 作为推荐方法
- **FrustumCuller NoneType错误**：修复 `frustum_scale` 为 `None` 时的 `max()` 调用错误
- **可视化代码硬编码容差**：将两处硬编码的 `TOLERANCE = 3.0` 替换为使用 `projector._corner_match_tolerance`
- **退化输入处理**：增加空数组、NaN 值、near_z=0 等边界情况的显式检查
  - `Camera.__init__`：验证 `near_z>0`、`far_z>0`、`near_z<far_z`、`width>0`、`height>0`
  - `FrustumCuller.__init__`：验证 `near_z>0`、`expansion_factor>0`
  - `cull_points`：空数组返回空结果，NaN 点标记为无效
  - `clip_line`：端点形状验证 `(3,)`，NaN 端点返回 `None`
  - `project_points`：空数组返回空结果，NaN 点标记为无效
  - `project_box`：角点形状验证 `(8,3)`，NaN 角点抛出 `ValueError`
  - `project_polygon`：空多边形返回空结果
  - `_project_raw_pixels`：空数组返回 `(0,2)` 空数组
- **CUDA 后端类型混合错误**：修复 CuPy 数组与 NumPy 0维标量混合运算时报错 `Unsupported type <class 'numpy.ndarray'>`
  - `FrustumCuller`：新增 `_to_python_scalar` / `_to_numpy` 静态方法，所有内部参数（`theta_max`、`cos_theta_max`、`tan_h`、`tan_v` 等）规范化为 Python float，所有公开方法（`cull_points`、`clip_line`、`is_point_inside`）输入强制转为 NumPy
  - `Projector._project_raw_pixels`：将畸变系数（`k1`/`k2`/`p1`/`p2`/`fw_poly` 等）提取为 Python float 后参与运算，避免 NumPy 标量污染 CuPy 数组运算
  - `PinholeCamera`：`dist_coeffs` 强制存储为 NumPy ndarray（与 backend 无关），消除 CuPy 数组传入 `np.testing.assert_array_almost_equal` 时的隐式转换错误
  - `PinholeCamera._check_fov` / `KannalaBrandtCamera._check_fov` / `FThetaCamera._check_fov_theta`：`_real_tan_h`/`_real_tan_v`/`_theta_max`/`eff_scale` 等几何参数提取为 Python float 后参与比较
  - `PinholeCamera._apply_distortion` / `KannalaBrandtCamera.project` / `FThetaCamera.project`：畸变系数 `k1`-`k6`/`p1`/`p2`/`fw_poly` 在参与投影计算前提取为 Python float/list
  - `KannalaBrandtCamera._compute_theta_max` / `FThetaCamera._compute_theta_max`：几何参数预计算强制使用 NumPy（局部 `import numpy as np` 覆盖 `self.np`），返回值统一为 Python float
  - `FThetaCamera.__init__`：`fw_poly` 强制存储为 NumPy ndarray（与 backend 无关）
  - `Camera._transform_to_camera`：将 `backend.matmul` 返回值（CUDA 后端为 NumPy）转换为 `self.np` 类型后再返回，避免后续 `np.maximum` 等 ufunc 报错

### 测试覆盖度

- 全部 **107 个测试用例通过**（multi_traj conda 环境，CUDA 后端默认启用）
- 新增 33 个退化输入测试（`test_degenerate_inputs.py`）
- 测试覆盖：相机工厂、投影、视锥裁剪、配置加载、C++后端、CUDA后端、FOV限制、软裁剪、退化输入等
- 验证 CUDA/C++/NumPy 三种后端在 Pinhole/KannalaBrandt/FTheta 三种相机上投影结果一致

## [0.9.0] - 2026-08-19

### 新增 (Added)

- **`max_fov_deg` 参数**（原 `max_fov_half_angle`，1.0.0 已重命名）：允许用户自定义鱼眼相机 FOV 上限
- **`Projector.corner_match_tolerance` 参数**：将硬编码的 `TOLERANCE=3.0` 提取为可配置参数
  - 默认值 3.0 像素
  - 用于判断裁剪端点是否为原始角点
- **`test_fov_limits.py` 测试文件**：新增测试用例覆盖大FOV场景
- **`FrustumCuller.from_camera()` None检查修复**：当 `frustum_scale` 为 `None` 时安全回退到默认值

### 变更 (Changed)

- **可视化代码整合**：移除重复实现，统一使用 autoproj 库

### 修复 (Fixed)

- **版本号不一致**：`__init__.py` 中 `__version__` 从 `0.8.0` 更新为 `0.9.0`
- **FrustumCuller NoneType错误**：修复 `frustum_scale` 为 `None` 时的 `max()` 调用错误
- **可视化代码硬编码容差**：将两处硬编码的 `TOLERANCE = 3.0` 替换为使用 `projector._corner_match_tolerance`

### 测试覆盖度

- 全部 **63 个测试用例通过**

## [0.8.0] - 2026-08-17

### 新增 (Added)

- **Camera.compute_real_fov()**：计算考虑畸变的真实 FOV。通过反解畸变方程获得图像边界对应的真实入射角，而非使用标称 FOV。
  - `PinholeCamera`：使用二分法反解 OpenCV 8 参数 rational polynomial 畸变方程（径向部分），切向畸变用不动点迭代精炼
  - `KannalaBrandtCamera`：使用二分法反解 θ 多项式（θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹）
  - `FThetaCamera`：使用已有的 `_theta_max`（通过二分法求解多项式方程获得）
- **Camera.compute_expansion_factor()**：基于真实 FOV 和标称 FOV 的比值自动计算视锥扩展因子。
  - 桶形畸变（k1<0）：真实 FOV > 标称 FOV → expansion > 1.0
  - 枕形畸变（k1>0）：真实 FOV < 标称 FOV → expansion < 1.0
- **PinholeCamera._undistort_normalized()**：反解畸变方程的内部方法，从畸变后归一化坐标反推畸变前归一化坐标。
- **KannalaBrandtCamera._undistort_theta()**：反解 θ 多项式的内部方法。
- **KannalaBrandtCamera.fov_h / fov_v**：新增标称 FOV 属性（基于针孔模型近似）。
- **Projector._project_raw_pixels()**：原始投影方法，仅执行 3D→2D 光学投影，不做边界检查、clamp 或置 -1。用于视锥裁剪后的棱端点投影和 culler 有效但相机无效的点重新投影。
- **FrustumCuller**：视锥裁剪器，从 projection3d 移植。支持 Liang-Barsky（金字塔视锥/针孔）、二次方程（圆锥视锥/鱼眼）、Sutherland-Hodgman（多边形裁剪）三种算法。
- **Projector 视锥裁剪集成**：`project_box`/`project_lines`/`project_polygon` 支持棱线级视锥裁剪，对 12 条棱逐条裁剪保留可见部分。

### 变更 (Changed)

- **Projector frustum_expansion='auto' 精确计算**：`'auto'` 模式不再固定返回 1.05，而是调用 `camera.compute_expansion_factor()` 基于相机内参和畸变系数精确计算真实 FOV 扩展范围。使过滤后的点云最接近真实相机可视视野范围。
- **FOV 扩展因子修复**：`project_points()` 现在正确使用 `FrustumCuller` 的扩展 FOV（含 `expansion_factor`）进行点云验证。此前，`camera.project()` 内部的固定 5% 容差 `_check_fov` 会覆盖扩展后的 FOV 结果，导致边界点被错误过滤。修复后：
  - culler 判定有效的点即使被相机内部检查过滤，也会通过 `_project_raw_pixels()` 重新计算像素坐标
  - 最终有效性完全基于 culler 的扩展 FOV 判断，不再受相机固定容差影响

### 修复 (Fixed)

- **点云投影 FOV 扩展因子未生效**：此前 `project_points` 中虽然使用了 `FrustumCuller.cull_points` 获取扩展 FOV 验证结果，但后续 `camera.project()` 内部的 `_check_fov(tolerance=0.05)` 会将扩展边界处的点过滤掉（置为 `-1`），最终 `valid & cam_valid` 使扩展因子失效。现已修复：对 culler 判定有效但相机无效的点重新计算像素坐标，并用 culler 结果作为最终有效性依据。
- **`_project_raw_pixels` FThetaCamera 路径错误**：此前 `Projector._project_raw_pixels()` 中 FThetaCamera 分支错误地访问不存在的 `cam.coeffs` 属性，导致回退到线性近似而非使用多项式 `fw_poly`。这使得视锥裁剪后的 F-Theta 棱线端点像素坐标计算错误。现已修复为直接使用 `cam.fw_poly` 多项式和 `arctan2` 入射角公式，与 `FThetaCamera.project()` 的 NumPy 路径保持一致。

## [0.6.0] - 2026-08-10

### 新增 (Added)

- **CUDABackend 完整投影实现**：为 CUDA 后端新增 `project_pinhole`、`project_kannala_brandt`、`project_ftheta` 三个投影方法，使用 CuPy 向量化运算，使 GPU 加速能真正用于投影核心逻辑（此前 CUDA 后端仅实现基础算子，投影仍走纯 Python 路径）。
- **Projector.project_point_cloud()**：新增 `PointCloud` 对象投影方法，返回 `(result, valid, filtered_cloud)`，自动按有效掩码过滤点云。
- **Projector.set_frustum_expansion()**：支持运行时动态更新视锥体扩展系数。
- **CameraFactory.get_default_subtype()**：查询指定类别的默认子类型。
- **C++ OpenMP 并行化**：三个投影函数均添加 `#pragma omp parallel for schedule(static) if(n_points > 1000000)`，在超百万点的大规模点云场景下启用多核并行处理，避免小数据集的线程创建开销。
- **Camera.margin 属性**：在 Camera 基类暴露边界检查余量参数，支持通过构造函数配置。

### 变更 (Changed)

- **Projector.frustum_expansion 生效**：此前该参数仅存储不生效。现在会乘以相机基础 margin（默认 20px）作为有效边界余量，`frustum_expansion=2.0` 会使余量变为 40px。
- **Camera._transform_to_camera 使用 backend.matmul**：齐次坐标变换改用后端统一的 `matmul` 接口，使 C++/CUDA 后端可加速矩阵运算。
- **CameraFactory.create_from_dict 优先使用 from_dict**：工厂现在优先调用相机类的 `from_dict` 类方法（含参数验证与默认值填充），而非直接 `**kwargs` 构造，消除了 `from_dict` 死代码。
- **CameraFactory.list_subtypes 健壮性**：类别不存在时返回空列表而非抛异常。
- **ConfigLoader 子类型保存还原**：`save_camera_config` 现在读取 `camera.sub_type` 还原原始子类型字符串，避免 `pinhole:wide_angle` 被错误保存为 `pinhole:standard`。
- **CUDABackend.is_available() 验证增强**：除 GPU 分配外，增加 `sqrt`/`arctan2` 计算验证与 `Stream.null.synchronize()`，确保 CUDA context 完全可用。
- **CUDABackend 输入显式 cp.asarray**：所有算子方法通过 `_to_gpu()` 显式将 NumPy 数组转为 CuPy 数组，避免隐式拷贝开销。

### 修复 (Fixed)

- **CUDABackend.astype 缺失 .get()**：此前 `astype` 返回 CuPy 数组而非 NumPy 数组，导致后续 NumPy 操作报错。现已修复。
- **恢复向后兼容的 clamp 和 int32 行为**：此前修改误移除了 numpy/C++/CUDA 三条投影路径中对有效点 UV 的 clamp（`[0, w-1]`）和 int32 类型转换，导致下游消费者（如用户脚本的二次边界过滤）行为异常。现已全部恢复，与 0.5.0 版本行为完全一致。

### 文档 (Documentation)

- README 新增 `project_point_cloud` 和 `set_frustum_expansion` 的 API 说明与使用示例。
- README Features 部分更新，反映 CUDA 完整投影、OpenMP 并行化、PointCloud 集成等新特性。
- Backend、NumPyBackend、Box3D 等类添加设计说明注释，解释接口保留原因。

## [0.5.0] - 2024

### 初始版本

- 支持针孔相机（OpenCV 8 参数畸变）、Kannala-Brandt 鱼眼、F-Theta 鱼眼三种相机模型
- 提供 NumPy、C++（pybind11）、CUDA（CuPy）三种计算后端，按优先级自动选择并降级
- 支持 YAML/JSON 配置文件加载与保存
- C++ 后端自动检测编译器、安装 pybind11、按需编译
- 分层命名相机工厂 API（`pinhole` / `fisheye:kannala` / `fisheye:ftheta`）
- 投影结果保持输入形状（`N×3+`），透传额外通道（intensity/gpstime）
