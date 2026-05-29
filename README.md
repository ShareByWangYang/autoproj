# AutoProj

> High-precision 3D-to-2D projection engine for autonomous driving and robotics

## Features

- ✅ **Multi-camera support**: Pinhole, Kannala-Brandt fisheye, F-Theta fisheye
- ✅ **High-precision distortion correction**: OpenCV 8-parameter rational function model
- ✅ **Hierarchical naming API**: Intuitive camera category organization
- ✅ **Flexible projection**: Point clouds, 3D bounding boxes, and lines
- ✅ **Python-only implementation**: No external dependencies required

## Installation

```bash
# Basic installation
pip install autoproj

# With CUDA support (optional)
pip install autoproj[cuda]

# Development installation
pip install autoproj[cuda,dev]
```

## Quick Start

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

## Camera Types

### Pinhole Camera
```python
camera = CameraFactory.create_pinhole(
    width=1920, height=1080,
    fx=1000, fy=1000, cx=960, cy=540,
    dist_coeffs=[k1, k2, p1, p2, k3, k4, k5, k6]
)
```

### Fisheye Camera (Kannala-Brandt)
```python
camera = CameraFactory.create_fisheye(
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
    fw_poly=[0, 500, 50]  # Focal length polynomial coefficients
)
```

## Documentation

- [API Reference](docs/api.md)
- [Usage Guide](docs/usage.md)
- [Camera Models](docs/camera_models.md)

## Development

```bash
# Clone repository
git clone git@github.com:ShareByWangYang/autoproj.git
cd autoproj

# Install dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=autoproj
```

## License

MIT License

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.