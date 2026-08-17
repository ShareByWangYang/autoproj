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
- **Multi-backend support**: NumPy (CPU), C++ (CPU, via pybind11 + OpenMP), and CUDA (GPU, via CuPy) with automatic fallback
- **Full GPU acceleration**: CUDA backend now implements all three projection methods (pinhole/Kannala-Brandt/F-Theta) for true GPU speedup
- **Flexible configuration**: Load camera parameters from YAML/JSON files with round-trip type preservation
- **Frustum expansion**: Adjustable boundary margin via `Projector(frustum_expansion=...)` for customizable view frustum culling
  - `frustum_expansion='auto'`: Automatically computes expansion factor from distortion coefficients via `camera.compute_expansion_factor()`
  - `frustum_expansion=1.5`: Manual expansion factor (1.0 = no expansion)
- **Frustum culling**: `FrustumCuller` performs geometric FOV filtering before distortion, preventing OpenCV distortion "fold-back" artifacts
  - Liang-Barsky algorithm for pyramid frustum (pinhole cameras)
  - Quadratic equation solver for cone frustum (fisheye cameras)
  - Sutherland-Hodgman algorithm for polygon clipping
  - Edge-level clipping for 3D bounding boxes (12 edges clipped individually)
- **PointCloud integration**: `Projector.project_point_cloud()` directly accepts `PointCloud` objects with automatic filtering
- **Automatic C++ backend**: Auto-detect compiler, install pybind11, compile on-demand with graceful fallback
- **OpenMP parallelism**: C++ extension uses `#pragma omp parallel for` for multi-core projection of large point clouds (enabled when n_points > 1M)
- **Python-only implementation**: No external dependencies required for core functionality

## Installation

```bash
# Basic installation
pip install autoproj

# With CUDA support (optional, requires CuPy)
pip install autoproj[cuda]

# Development installation
pip install autoproj[cuda,dev]
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

# Create a projector with frustum expansion (optional)
# frustum_expansion=2.0 doubles the boundary margin from 20px to 40px
projector = Projector(camera, frustum_expansion=1.0)

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
    far_z=1000.0
)
```

### Fisheye Camera (Kannala-Brandt)

```python
camera = CameraFactory.create_fisheye(
    sub_type='kannala',  # default
    width=1920, height=1080,
    fx=500, fy=500, cx=960, cy=540,
    k1=0.1, k2=0.05, k3=0.01, k4=0.005
)
```

### Fisheye Camera (F-Theta)

```python
camera = CameraFactory.create_fisheye(
    sub_type='ftheta',
    width=1920, height=1080,
    fw_poly=[0, 500, 50],  # Focal length polynomial: fw(theta) = a0 + a1*theta + a2*theta^2 + ...
    cx=960, cy=540
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

# C++ backend auto-build demonstration
python examples/auto_build_example.py
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
| `project_points(points_3d, T_to_cam=None, pts_in_cam=False)` | Project 3D points to 2D image plane. Returns `(result, valid)` tuple where `result` has shape `(N, 3+)` (columns: `u`, `v`, `depth`, plus any extra input columns preserved) and `valid` is a boolean mask of shape `(N,)`. |
| `project_box(box_3d, T_to_cam=None)` | Project a single 3D bounding box (8 corners). Returns `(result, valid)`. |
| `project_lines(lines, T_to_cam=None)` | Project 3D line segments. Returns `(result, valid)` where a line is valid only if both endpoints are valid. |
| `project_point_cloud(cloud, T_to_cam=None, pts_in_cam=False)` | Project a `PointCloud` object. Returns `(result, valid, filtered_cloud)` where `filtered_cloud` is the input cloud filtered by the valid mask. |
| `set_frustum_expansion(expansion)` | Dynamically update the frustum expansion coefficient. The effective boundary margin is `20 * expansion` pixels. |

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

### Backend

| Class | Description |
|-------|-------------|
| `Backend` | Abstract base class for all backends |
| `NumPyBackend` | CPU backend based on NumPy (default fallback, uses pure Python projection path) |
| `CPythonBackend` | High-performance C++ backend via pybind11 with OpenMP (auto-built, 10-15x faster) |
| `CUDABackend` | GPU-accelerated backend via CuPy. Implements all three projection methods for full GPU acceleration. |
| `BackendSelector` | Automatic backend selection with fallback. Priority: CUDA > C++ > NumPy |

### ConfigLoader

| Function | Description |
|----------|-------------|
| `load_camera(file_path)` | Load camera from YAML/JSON file |
| `load_camera_from_dict(config)` | Load camera from dictionary |
| `save_camera_config(camera, file_path)` | Save camera to YAML/JSON file |

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

## Performance

AutoProj supports multiple backends for different performance requirements:

| Backend | Description | Requirements |
|---------|-------------|--------------|
| NumPy | CPU-based (default fallback) | numpy |
| C++ | High-performance CPU via pybind11 (10-15x faster than NumPy) | g++/clang++ + pybind11 (auto-installed) |
| CUDA | GPU-accelerated | cupy + CUDA toolkit |

```python
from autoproj import BackendSelector

# Auto-select best available backend (CUDA > C++ > NumPy)
backend = BackendSelector.select()
print(f"Using backend: {backend.name()}")

# Force specific backend
backend = BackendSelector.select('numpy')
backend = BackendSelector.select('cpp')    # Auto-builds if not available
backend = BackendSelector.select('cuda')   # Falls back to C++/NumPy if CUDA not available

# List available backends
available = BackendSelector.available_backends()
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use AutoProj in your research, please cite:

```bibtex
@software{autoproj,
  author = {Wang Yang},
  title = {AutoProj: High-precision 3D-to-2D projection engine},
  year = {2024},
  url = {https://github.com/ShareByWangYang/autoproj}
}
```
