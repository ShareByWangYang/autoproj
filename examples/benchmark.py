#!/usr/bin/env python3
"""
AutoProj Performance Benchmark Tool
====================================
Comprehensive benchmark for comparing different backends and camera models.
"""

import sys
import time
import numpy as np
import argparse
from typing import Dict, List, Tuple

def import_backends() -> Dict:
    """Import and return available backends."""
    backends = {}
    
    # Try NumPy backend
    try:
        from autoproj.backends import NumPyBackend
        backends['NumPy'] = NumPyBackend()
        print("✓ NumPy backend available")
    except Exception as e:
        print(f"✗ NumPy backend: {e}")
    
    # Try C++ backend
    try:
        from autoproj.backends import CPythonBackend
        backends['C++'] = CPythonBackend()
        print("✓ C++ backend available")
    except Exception as e:
        print(f"✗ C++ backend: {e}")
    
    # Try CUDA backend
    try:
        from autoproj.backends import CUDABackend
        backends['CUDA'] = CUDABackend()
        print("✓ CUDA backend available")
    except Exception as e:
        print(f"✗ CUDA backend: {e}")
    
    return backends

def generate_test_points(n_points: int, width: int, height: int) -> np.ndarray:
    """Generate random 3D points for testing."""
    # Generate points in camera coordinates
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(-10, 10, n_points)
    z = np.random.uniform(1, 50, n_points)
    
    return np.column_stack((x, y, z))

def benchmark_backend(
    backend,
    camera_type: str,
    points_3d: np.ndarray,
    n_iterations: int = 10
) -> Tuple[float, float, float]:
    """
    Benchmark a single backend with a camera type.
    
    Returns: (min_time, avg_time, max_time) in seconds
    """
    from autoproj import CameraFactory
    
    # Create camera
    if camera_type == 'pinhole':
        camera = CameraFactory.create_pinhole(
            width=1920, height=1080,
            fx=1000.0, fy=1000.0,
            cx=960.0, cy=540.0,
            dist_coeffs=np.array([0.1, -0.05, 0.001, -0.001, 0.01, 0.02, 0.03, 0.04]),
            backend=backend
        )
    elif camera_type == 'kannala':
        camera = CameraFactory.create_fisheye(
            sub_type='kannala',
            width=1920, height=1080,
            fx=500.0, fy=500.0,
            cx=960.0, cy=540.0,
            k1=0.01, k2=0.001, k3=0.0001, k4=0.00001,
            backend=backend
        )
    elif camera_type == 'ftheta':
        camera = CameraFactory.create_fisheye(
            sub_type='ftheta',
            width=1920, height=1080,
            fw_poly=np.array([0.0, 500.0, 50.0, 10.0]),
            backend=backend
        )
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")
    
    # Warmup
    for _ in range(3):
        camera.project(points_3d, pts_in_cam=True)
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        pixels, valid = camera.project(points_3d, pts_in_cam=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times_array = np.array(times)
    return np.min(times_array), np.mean(times_array), np.max(times_array)

def run_benchmark(
    backends: Dict,
    n_points: int = 1000000,
    n_iterations: int = 10,
    output_format: str = 'text'
):
    """Run complete benchmark suite."""
    print(f"\n{'='*80}")
    print(f"AutoProj Performance Benchmark v0.5.0")
    print(f"{'='*80}")
    print(f"Test points: {n_points:,}")
    print(f"Iterations: {n_iterations}")
    print(f"Backends: {list(backends.keys())}")
    print()
    
    camera_types = ['pinhole', 'kannala', 'ftheta']
    results = {}
    
    for camera_type in camera_types:
        print(f"\nTesting {camera_type.upper()} camera:")
        print(f"{'-'*60}")
        
        results[camera_type] = {}
        points_3d = generate_test_points(n_points, 1920, 1080)
        
        for name, backend in backends.items():
            try:
                min_t, avg_t, max_t = benchmark_backend(
                    backend, camera_type, points_3d, n_iterations
                )
                
                throughput = n_points / avg_t
                results[camera_type][name] = {
                    'min': min_t,
                    'avg': avg_t,
                    'max': max_t,
                    'throughput': throughput
                }
                
                print(f"  {name:10s}: {avg_t*1000:6.2f}ms avg | {throughput/1e6:6.2f}M pts/s")
            except Exception as e:
                print(f"  {name:10s}: FAILED - {e}")
                results[camera_type][name] = None
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if output_format == 'text':
        print_text_summary(results, n_points)
    elif output_format == 'csv':
        print_csv_summary(results)
    elif output_format == 'json':
        print_json_summary(results)

def print_text_summary(results: Dict, n_points: int):
    """Print summary in text format."""
    for camera_type in results:
        print(f"\n{camera_type.upper()} CAMERA:")
        print(f"{'Backend':12s} | {'Time (ms)':12s} | {'Throughput':12s} | {'Speedup':10s}")
        print(f"{'-'*54}")
        
        # Find baseline (NumPy if available)
        baseline = None
        if 'NumPy' in results[camera_type] and results[camera_type]['NumPy'] is not None:
            baseline = results[camera_type]['NumPy']['avg']
        
        for name in results[camera_type]:
            r = results[camera_type][name]
            if r is None:
                continue
            
            speedup = baseline / r['avg'] if baseline else 1.0
            print(f"{name:12s} | {r['avg']*1000:10.2f}ms  | {r['throughput']/1e6:10.2f}M  | {speedup:8.2f}x")

def print_csv_summary(results: Dict):
    """Print summary in CSV format."""
    print("camera_type,backend,min_ms,avg_ms,max_ms,throughput_M_pts_s")
    for camera_type in results:
        for name in results[camera_type]:
            r = results[camera_type][name]
            if r is None:
                continue
            print(f"{camera_type},{name},{r['min']*1000:.3f},{r['avg']*1000:.3f},"
                  f"{r['max']*1000:.3f},{r['throughput']/1e6:.3f}")

def print_json_summary(results: Dict):
    """Print summary in JSON format."""
    import json
    print(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description='AutoProj Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # Default benchmark (1M points, 10 iterations)
  python benchmark.py --points 5000000        # 5 million points
  python benchmark.py --iterations 50         # More iterations for stable results
  python benchmark.py --format csv            # Output in CSV format
        """
    )
    
    parser.add_argument(
        '--points', '-p',
        type=int,
        default=1000000,
        help='Number of test points (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of iterations per test (default: 10)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['text', 'csv', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    args = parser.parse_args()
    
    backends = import_backends()
    
    if not backends:
        print("Error: No backends available!")
        return 1
    
    run_benchmark(backends, args.points, args.iterations, args.format)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
