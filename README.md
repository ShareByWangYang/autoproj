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
- **Multi-backend support**: NumPy (CPU) and CUDA (GPU) with automatic fallback
- **Flexible configuration**: Load camera parameters from YAML/JSON files
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

# Create a projector
projector = Projector(camera)

# Generate test points (in camera coordinate system)
points_3d = np.array([[1, 0, 10], [0, 0, 10], [-1, 0, 10]])

# Project points
result = projector.project_points(points_3d, pts_in_cam=True)
print(f"Pixels: {result['pixels']}")
print(f"Valid points: {result['num_valid']}/{result['num_points']}")
```

### Load from Configuration File

```python
from autoproj import load_camera

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
| `create_pinhole(**kwargs)` | Create a pinhole camera |
| `create_fisheye(sub_type='kannala', **kwargs)` | Create a fisheye camera |
| `create_from_dict(config)` | Create camera from dictionary |
| `list_categories()` | List available camera categories |
| `list_subtypes(category)` | List available subtypes for a category |

### Projector

| Method | Description |
|--------|-------------|
| `project_points(points_3d, pts_in_cam=False)` | Project 3D points to 2D image plane |
| `project_boxes(boxes_3d, pts_in_cam=False)` | Project 3D bounding boxes |
| `project_lines(lines_3d, pts_in_cam=False)` | Project 3D lines |

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
| NumPy | CPU-based (default) | numpy |
| CUDA | GPU-accelerated | cupy |

```python
from autoproj import BackendSelector

# Auto-select best available backend
backend = BackendSelector.select()
print(f"Using backend: {backend.name()}")

# Force specific backend
backend = BackendSelector.select('numpy')
backend = BackendSelector.select('cuda')  # Falls back to numpy if CUDA not available
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
