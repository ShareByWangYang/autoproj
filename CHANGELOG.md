# Changelog

本项目所有重要变更均会记录在此文件中。
格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

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
