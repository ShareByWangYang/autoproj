# AutoProj

> High-precision 3D-to-2D projection engine for autonomous driving and embodied intelligence

[![PyPI Version](https://img.shields.io/pypi/v/autoproj)](https://pypi.org/project/autoproj/)
[![License](https://img.shields.io/github/license/ShareByWangYang/autoproj)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ShareByWangYang/autoproj/ci.yml)](https://github.com/ShareByWangYang/autoproj/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/ShareByWangYang/autoproj)](https://codecov.io/gh/ShareByWangYang/autoproj)

## Features

- **Multi-camera support**:
  - ✅ Pinhole camera with OpenCV 8-parameter distortion
  - ✅ Kannala-Brandt fisheye camera
  - ✅ F-Theta fisheye camera
- **High-precision projection**: Sub-pixel accuracy for critical perception tasks
- **Hierarchical naming API**: Intuitive camera category organization (`pinhole`, `fisheye:kannala`, `fisheye:ftheta`)
- **Numba JIT acceleration**: Batch operations (`project_boxes` / `project_lines` / `project_polygons`) use Numba `@njit(parallel=True)` for 4-15x speedup over Python loops
- **Flexible configuration**: Load camera parameters from YAML/JSON files with round-trip type preservation
- **Frustum scaling**: Adjustable boundary margin via `Projector(frustum_scale=...)` for customizable view frustum culling
  - `frustum_scale=None`: Automatically computes expansion factor from distortion coefficients via `camera.compute_expansion_factor()`
  - `frustum_scale=1.5`: Manual scaling factor (1.0 = no scaling)
- **FOV upper limit**: `max_fov_deg` parameter (degrees) to cap fisheye camera FOV, preventing extreme-angle projection artifacts (`None` = no limit)
- **Soft clipping**: `soft_clip_ratio` parameter retains slightly out-of-bounds corners when their 2D projection is close to the clipped point (pinhole cameras only)
- **Frustum culling**: `FrustumCuller` performs geometric FOV filtering before distortion, preventing OpenCV distortion "fold-back" artifacts
  - Liang-Barsky algorithm for pyramid frustum (pinhole cameras)
  - Quadratic equation solver for cone frustum (fisheye cameras)
  - Sutherland-Hodgman algorithm for polygon clipping
  - Edge-level clipping for 3D bounding boxes (12 edges clipped individually)
- **PointCloud integration**: `Projector.project_point_cloud()` directly accepts `PointCloud` objects with automatic filtering
- **Python-only implementation**: Pure Python + NumPy + Numba, no compiler toolchain required

## Coordinate System Conventions

AutoProj 内部统一采用 **OpenCV 约定**，不做任何坐标系自动猜测/翻轴。
调用方若数据源约定不同，须在入口处通过 [`conventions`](#conventions) 模块完成显式变换后再传入。

| 项目 | 约定 | 备注 |
|------|------|------|
| 相机系 | **OpenCV**: x 右 / y 下 / z 前，原点为光心 | 前方点 z>0；针孔视锥为棱锥，鱼眼为圆锥 |
| 像素原点 | **左上角**，u 向右增长，v 向下增长 | 与 OpenCV `cv2.projectPoints` 一致 |
| 外参 `T_to_cam` | **world→camera** 的 4×4 齐次矩阵，`p_cam = T @ p_world` | 若手头是 c2w（SLAM/渲染域常见），先 `conventions.invert_transform(T_c2w)` |
| 单位 | **米**（`near_z`/`far_z` 默认 0.1/1000 米） | 毫米数据须先乘 1e-3，否则全部超 `far_z` 被判 invalid |
| 畸变系数 | **不可跨模型混用** | 针孔 k1-k6/p1-p2 与鱼眼 k1-k4 物理含义完全不同；`cv2.fisheye` 的 k1-k4 可原样填入 `fisheye:kannala` |

### 常见数据源的对齐方式

| 数据源约定 | 与 OpenCV 的差异 | 对齐方式 |
|------------|------------------|----------|
| OpenGL / Blender | y 上、z 后 | 轴变换 `diag(1, -1, -1)`；或 `conventions.AXIS_OPENGL_TO_OPENCV` |
| ROS REP-103 (LiDAR/车体) | x 前、y 左、z 上 | 需按实际传感器 extrinsic 构造，调用方负责 |
| 四元数 `xyzw` (ROS/eigen) | 本库 `build_transform` 默认 `wxyz` | 传 `quat_order='xyzw'` |

> **静默错误警告**：外参方向弄反、轴向错配、四元数顺序错、单位毫米、模型族错配
> **都不会报错**，只会产生旋转/镜像/全 invalid 等症状。建议开启投影健康检查
> (`Projector(..., health_check=True)`) 辅助定位。

## Conventions

`autoproj.conventions` 提供入口适配工具（纯函数，不改动核心数学）：

```python
from autoproj import conventions

# 1. 由四元数 + 平移构造 w2c 外参
T_w2c = conventions.build_transform(
    quat=[0.99, 0, 0, 0.14],   # wxyz 默认；ROS 用 quat_order='xyzw'
    t=[1.0, 0.0, 1.5],          # 相机在源系的位置（w2c 下即 t_w2c）
    direction='w2c'             # 显式声明方向
)

# 2. c2w ↔ w2c 互转
T_c2w = conventions.invert_transform(T_w2c)
T_w2c = conventions.invert_transform(T_c2w)  # SLAM 给的 c2w 直接转

# 3. 常见轴变换矩阵（右乘到外参 R）
T_gl = conventions.AXIS_OPENGL_TO_OPENCV   # diag(1,-1,-1) 的 4x4
# T_w2c_opencv = T_gl @ T_w2c_opengl_axis

# 4. 标定分辨率与使用分辨率不一致时缩放内参
fx2, fy2, cx2, cy2 = conventions.scale_intrinsics(
    fx=1000, fy=1000, cx=960, cy=540,
    src_size=(1920, 1080), dst_size=(960, 540)
)
```

## Installation

```bash
# Basic installation
pip install autoproj

# With Numba JIT acceleration (recommended for batch operations)
pip install autoproj[jit]

# Development installation
pip install autoproj[dev]
```

## Quick Start

### Basic Projection

```python
from autoproj import CameraFactory, Projector
import numpy as np

# Create a pinhole camera
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000, fy=1000, cx=960, cy=540,
    dist_coeffs=[0.1, -0.05, 0.01, -0.01]
)

# Create a projector with frustum scaling (optional)
# frustum_scale=None: auto-compute expansion from distortion coefficients
# frustum_scale=1.0: no scaling (uses boundary_ratio for margin)
projector = Projector(camera, frustum_scale=1.0)

# Generate test points (in camera coordinate system)
points_3d = np.array([[1, 0, 10], [0, 0, 10], [-1, 0, 10]])

# Project points (returns: result array of shape (N, 3+), valid mask of shape (N,))
result, valid = projector.project_points(points_3d, pts_in_cam=True)
print(f"Pixels (u, v, depth):\n{result}")
print(f"Valid points: {int(valid.sum())}/{len(valid)}")
```

### Project a PointCloud Object

```python
from autoproj import CameraFactory, Projector, PointCloud
import numpy as np

camera = CameraFactory.create_pinhole(
    width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
)
projector = Projector(camera)

# Create a PointCloud with colors and intensity
points = np.random.uniform(-5, 5, (1000, 3))
points[:, 2] = np.random.uniform(5, 50, 1000)  # z: 5-50m
colors = np.random.randint(0, 255, (1000, 3))
intensity = np.random.uniform(0, 1, 1000)
cloud = PointCloud(points, colors=colors, intensity=intensity)

# Project and get filtered cloud in one call
result, valid, filtered_cloud = projector.project_point_cloud(cloud, pts_in_cam=True)
print(f"Valid: {valid.sum()}/{len(valid)}, Filtered cloud: {filtered_cloud.num_points} points")
```

### Load from Configuration File

```python
from autoproj import load_camera, load_camera_from_dict

# Load camera from YAML file
camera = load_camera('camera_config.yaml')

# Or from JSON file
camera = load_camera('camera_config.json')

# Or from dictionary
config = {
    'type': 'pinhole',
    'width': 1920,
    'height': 1080,
    'fx': 1000,
    'fy': 1000,
    'cx': 960,
    'cy': 540
}
camera = load_camera_from_dict(config)
```

## Camera Types

### Pinhole Camera

```python
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000, fy=1000, cx=960, cy=540,
    dist_coeffs=[k1, k2, p1, p2, k3, k4, k5, k6],  # 8 parameters
    near_z=0.1,
    far_z=1000.0,
    boundary_ratio=0.02  # boundary margin as ratio of max(width, height)
)
```

### Fisheye Camera (Kannala-Brandt)

```python
camera = CameraFactory.create_fisheye(
    sub_type='kannala',  # default
    width=1920, height=1080,
    fx=500, fy=500, cx=960, cy=540,
    k1=0.1, k2=0.05, k3=0.01, k4=0.005,
    max_fov_deg=180  # cap FOV at 180° to avoid extreme-angle artifacts (None = no limit)
)
```

### Fisheye Camera (F-Theta)

```python
camera = CameraFactory.create_fisheye(
    sub_type='ftheta',
    width=1920, height=1080,
    fw_poly=[0, 500, 50],  # Focal length polynomial: fw(theta) = a0 + a1*theta + a2*theta^2 + ...
    cx=960, cy=540,
    max_fov_deg=170  # cap FOV at 170° (None = no limit)
)
```

## Examples

Run the example scripts to see AutoProj in action:

```bash
# Basic projection example
python examples/basic_projection.py

# Fisheye camera projection
python examples/fisheye_projection.py

# Configuration file loader
python examples/config_loader.py
```

## Configuration File Format

### YAML Example (`camera_config.yaml`)

```yaml
type: pinhole
width: 1920
height: 1080
fx: 1000.0
fy: 1000.0
cx: 960.0
cy: 540.0
dist_coeffs: [0.1, -0.05, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0]
near_z: 0.1
far_z: 1000.0
```

### JSON Example (`camera_config.json`)

```json
{
    "type": "fisheye:kannala",
    "width": 1920,
    "height": 1080,
    "fx": 500.0,
    "fy": 500.0,
    "cx": 960.0,
    "cy": 540.0,
    "k1": 0.1,
    "k2": 0.05
}
```

## API Reference

### CameraFactory

| Method | Description |
|--------|-------------|
| `create(category, sub_type=None, **kwargs)` | Create a camera by category and optional subtype |
| `create_pinhole(sub_type=None, **kwargs)` | Create a pinhole camera |
| `create_fisheye(sub_type=None, **kwargs)` | Create a fisheye camera |
| `create_from_dict(config)` | Create camera from dictionary with `type` field |
| `list_categories()` | List available camera categories |
| `list_subtypes(category)` | List available subtypes for a category (returns empty list if category not found) |
| `get_default_subtype(category)` | Query the default subtype for a category (returns None if not found) |
| `register_camera_type(category, sub_type, camera_class, is_default=False)` | Register a custom camera type |

### Camera Types

| Class | Description |
|-------|-------------|
| `PinholeCamera` | Pinhole camera with OpenCV 8-parameter distortion |
| `KannalaBrandtCamera` | Kannala-Brandt fisheye camera (θ-polynomial distortion) |
| `FThetaCamera` | F-Theta equidistant fisheye camera (polynomial focal length) |

### Projector

| Method | Description |
|--------|-------------|
| `project_points(points_3d, T_to_cam=None, pts_in_cam=False, preserve_extra=False)` | Project 3D points to 2D image plane. Returns `(result, valid)`. Default (`preserve_extra=False`): `result` is `(N, 2)` int32 (columns: `u`, `v`, invalid points set to `-1`, valid clamped to `[0, w-1]`). With `preserve_extra=True`: `result` is `(N, 3+)` float64 (`u`, `v`, `depth`, plus extra input columns). `valid` is a boolean mask of shape `(N,)`. |
| `project_box(box_input, T_to_cam=None, pts_in_cam=False, cull_frustum=None, extend_to_boundary=False)` | Project a single 3D bounding box (8 corners). Returns a dict with corner projections, visible edges, and clipping info. |
| `project_boxes(boxes, T_to_cam=None, pts_in_cam=False, cull_frustum=None, extend_to_boundary=False)` | Batch project 3D bounding boxes. Returns `List[Dict]` (same format as `project_box`). Uses Numba JIT for batches > 64. |
| `project_lines(lines_input, T_to_cam=None, pts_in_cam=False, cull_frustum=None)` | Project 3D line segments. Returns `List[Optional[(p1, p2)]]` of visible segments (None if fully clipped). Supports `(N,2,3)` ndarray, `LineSet`, or list of tuples. |
| `project_polygon(polygon, T_to_cam=None, pts_in_cam=False, cull_frustum=None)` | Project a single 3D polygon. Returns a dict with vertex projections and clipped polygon vertices. |
| `project_polygons(polygons, T_to_cam=None, pts_in_cam=False, cull_frustum=None)` | Batch project 3D polygons. Returns `List[Dict]`. Uses Numba JIT for batches > 32. |
| `project_point_cloud(cloud, T_to_cam=None, pts_in_cam=False)` | Project a `PointCloud` object. Returns `(result, valid, filtered_cloud)` where `filtered_cloud` is the input cloud filtered by the valid mask. |
| `extend_edges_to_boundary(...)` | Extend clipped box edges to image boundary for 2D visual optimization. |
| `set_frustum_scale(scale)` | Dynamically update the frustum scaling factor. `None` = auto-compute, `float` = manual scale. |
| `set_frustum_expansion(expansion)` | [DEPRECATED] Use `set_frustum_scale` instead. Accepts `'auto'` or float for backward compatibility. |
| `set_cull_frustum(enabled)` | Dynamically enable/disable frustum culling at runtime. |

**Constructor parameters**: `Projector(camera, cull_frustum=True, frustum_scale=None, corner_match_tolerance=3.0, soft_clip_ratio=None, health_check=False)`. Set `health_check=True` to enable projection sanity diagnostics (warns on low valid ratio, z<0 majority, det(R)<0 mirror, out-of-bounds majority) — see [Coordinate System Conventions](#coordinate-system-conventions).

### FrustumCuller

| Method | Description |
|--------|-------------|
| `FrustumCuller(frustum_type, near_z, expansion_factor, **kwargs)` | Create a frustum culler. `frustum_type` can be `FrustumType.PYRAMID` (pinhole) or `FrustumType.CONE` (fisheye). |
| `FrustumCuller.from_camera(camera, expansion_factor=1.0)` | Create a culler from a Camera object, automatically selecting the frustum type. |
| `cull_points(points)` | Test if points are inside the frustum. Returns `(points, valid_mask)`. |
| `clip_line(p1, p2)` | Clip a 3D line segment against the frustum. Returns list of visible segments. |
| `clip_polygon(polygon)` | Clip a 3D polygon against the frustum using Sutherland-Hodgman. Returns clipped polygon vertices. |

### Geometry

| Class | Description |
|-------|-------------|
| `PointCloud` | Point cloud with optional colors and intensity. Supports `transform()` and `filter_by_mask()`. |
| `Box3D` | 3D bounding box with yaw rotation. Provides `corners` property and serialization methods. |

### ConfigLoader

| Function | Description |
|----------|-------------|
| `load_camera(file_path)` | Load camera from YAML/JSON file |
| `load_camera_from_dict(config)` | Load camera from dictionary |
| `save_camera_config(camera, file_path)` | Save camera to YAML/JSON file |

### Conventions

`autoproj.conventions` — Entry-point adaptation utilities (pure functions, no core math changes). See [Coordinate System Conventions](#coordinate-system-conventions) for usage.

| Function / Constant | Description |
|---------------------|-------------|
| `build_transform(R=None, t=None, quat=None, quat_order='wxyz', direction='w2c')` | Build a 4×4 homogeneous transform from rotation + translation. Supports `wxyz`/`xyzw` quaternion order; `direction` declares `w2c` or `c2w` semantics (no implicit flipping). |
| `invert_transform(T)` | Invert a rigid-body transform (c2w ↔ w2c). Uses `[R^T \| -R^T·t]` form, more stable than generic 4×4 inverse. |
| `scale_intrinsics(fx, fy, cx, cy, src_size, dst_size)` | Scale intrinsics when calibration resolution ≠ usage resolution. Uses `+0.5/-0.5` pixel-center alignment. |
| `check_transform_sanity(T, direction='w2c')` | Sanity-check a transform matrix (R orthogonality, det(R) sign, finite values). Warns on issues, does not raise. |
| `AXIS_OPENGL_TO_OPENCV` | 4×4 axis transform `diag(1, -1, -1)` for OpenGL/Blender → OpenCV. |
| `AXIS_OPENCV_TO_OPENGL` | Inverse of above (same matrix, self-inverse). |
| `AXIS_ROS_REP103_TO_OPENCV_XFYFZU` | 4×4 axis remap for ROS REP-103 (x-front/y-left/z-up) → OpenCV camera axes. |

## Development

### Setup Development Environment

```bash
# Clone repository
git clone git@github.com:ShareByWangYang/autoproj.git
cd autoproj

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ -v --cov=autoproj --cov-report=html

# Run specific test file
pytest tests/test_camera.py -v
```

### Branching Strategy

- `main`: Stable releases
- `develop`: Development integration
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches

### Commit Message Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `refactor:` Code refactoring
- `test:` Test updates
- `chore:` Build/CI updates

## License

MIT License - see [LICENSE](LICENSE) for details.

Third-party licenses are listed in [NOTICE](NOTICE).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use AutoProj in your research, please cite:

```bibtex
@software{autoproj,
  author = {FeiChangShuai},
  title = {AutoProj: High-precision 3D-to-2D projection engine},
  year = {2026},
  url = {https://github.com/ShareByWangYang/autoproj}
}
```

## Disclaimer

This project is not affiliated with any other project of the same name.
The author of this project is FeiChangShuai, not the repository owner
`ShareByWangYang` (which is only the hosting location).

**AI-Assisted Development (Vibe Coding)**: This project was developed with
the assistance of AI coding tools (vibe coding). All code has been reviewed
and tested by the author, but the AI-assisted nature of development is
disclosed for transparency. Users are encouraged to review the code and
run the test suite before production use.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.
