from .base import Backend
from typing import Optional
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CUDABackend(Backend):
    """
    CUDA后端实现

    基于CuPy的GPU加速实现，需要安装cupy包。
    提供与 CPythonBackend 对等的三个投影方法，使 GPU 加速能真正用于投影核心逻辑。

    注意：所有方法在返回前都会通过 .get() 将 CuPy 数组转回 NumPy 数组，
    以保证与 NumPy 后端的接口契约一致。
    """
    
    def __init__(self):
        if CUPY_AVAILABLE:
            self.np = cp
        else:
            self.np = np
    
    def name(self) -> str:
        return 'cuda'
    
    def is_available(self) -> bool:
        """
        检查 CUDA 后端是否可用。

        验证步骤：
        1. CuPy 已安装
        2. 能够在 GPU 上分配数组
        3. 能够执行核心数学运算（sqrt/arctan2），确保 CUDA context 正常
        """
        if not CUPY_AVAILABLE:
            return False
        try:
            # 验证 GPU 分配与核心计算均可用
            x = cp.array([1.0, 2.0, 3.0])
            _ = cp.sqrt(x)
            _ = cp.arctan2(x, x)
            cp.cuda.Stream.null.synchronize()
            return True
        except Exception:
            return False
    
    def _to_gpu(self, x):
        """将输入显式转为 CuPy 数组，避免隐式拷贝开销"""
        if isinstance(x, cp.ndarray):
            return x
        return cp.asarray(x)
    
    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_g = self._to_gpu(a)
        b_g = self._to_gpu(b)
        return cp.dot(a_g, b_g).get()
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_g = self._to_gpu(a)
        b_g = self._to_gpu(b)
        return cp.matmul(a_g, b_g).get()
    
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return cp.sqrt(self._to_gpu(x)).get()
    
    def arctan(self, x: np.ndarray) -> np.ndarray:
        return cp.arctan(self._to_gpu(x)).get()
    
    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return cp.arctan2(self._to_gpu(y), self._to_gpu(x)).get()
    
    def cos(self, x: np.ndarray) -> np.ndarray:
        return cp.cos(self._to_gpu(x)).get()
    
    def sin(self, x: np.ndarray) -> np.ndarray:
        return cp.sin(self._to_gpu(x)).get()
    
    def maximum(self, x: np.ndarray, y: float) -> np.ndarray:
        return cp.maximum(self._to_gpu(x), y).get()
    
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return cp.clip(self._to_gpu(x), min_val, max_val).get()
    
    def zeros_like(self, x: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return cp.zeros_like(self._to_gpu(x)).get()
        return cp.zeros_like(self._to_gpu(x), dtype=dtype).get()
    
    def ones(self, shape: tuple, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return cp.ones(shape).get()
        return cp.ones(shape, dtype=dtype).get()
    
    def hstack(self, arrays: list) -> np.ndarray:
        gpu_arrays = [self._to_gpu(a) for a in arrays]
        return cp.hstack(gpu_arrays).get()
    
    def astype(self, x: np.ndarray, dtype: np.dtype) -> np.ndarray:
        # 修复：之前遗漏 .get()，会返回 CuPy 数组导致后续 NumPy 操作报错
        return self._to_gpu(x).astype(dtype).get()
    
    def asarray(self, x, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return cp.asarray(x, dtype=dtype).get()
    
    # ===== 投影专用方法（与 CPythonBackend 接口对齐）=====
    
    def project_pinhole(
        self,
        points_3d: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        dist_coeffs: np.ndarray,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """
        GPU 加速的针孔相机投影。

        使用 CuPy 向量化运算，与 CPythonBackend.project_pinhole 行为一致：
        - 边界内不做 clamp（与纯 Python 路径保持一致）
        - 无效点 UV 置 -1
        """
        pts = self._to_gpu(np.asarray(points_3d, dtype=np.float64))
        dist = self._to_gpu(np.asarray(dist_coeffs, dtype=np.float64))

        x_c = pts[:, 0]
        y_c = pts[:, 1]
        z_c = pts[:, 2]

        # 深度范围检查
        valid = (z_c > near_z) & (z_c < far_z)

        # 安全除法
        safe_z = cp.where(valid, z_c, 1.0)
        x_norm = x_c / safe_z
        y_norm = y_c / safe_z

        # 畸变系数
        k1 = float(dist[0]) if dist.shape[0] > 0 else 0.0
        k2 = float(dist[1]) if dist.shape[0] > 1 else 0.0
        p1 = float(dist[2]) if dist.shape[0] > 2 else 0.0
        p2 = float(dist[3]) if dist.shape[0] > 3 else 0.0
        k3 = float(dist[4]) if dist.shape[0] > 4 else 0.0
        k4 = float(dist[5]) if dist.shape[0] > 5 else 0.0
        k5 = float(dist[6]) if dist.shape[0] > 6 else 0.0
        k6 = float(dist[7]) if dist.shape[0] > 7 else 0.0

        r2 = x_norm ** 2 + y_norm ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3

        numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
        denom = 1 + k4 * r2 + k5 * r4 + k6 * r6
        denom = cp.maximum(denom, 1e-10)
        radial = numerator / denom

        x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
        y_dist = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        # 边界检查（与 Python 路径一致：margin=20，有效点 clamp 到 [0, w-1]）
        in_bounds = (u >= -20) & (u < width + 20) & (v >= -20) & (v < height + 20)
        valid = valid & in_bounds

        # 恢复 clamp 行为，与 Python/C++ 路径保持一致
        u = cp.clip(u, 0, width - 1)
        v = cp.clip(v, 0, height - 1)
        u = cp.where(valid, u, -1.0)
        v = cp.where(valid, v, -1.0)

        pixels = cp.stack([u, v], axis=1).get()
        return pixels, valid.get()
    
    def project_kannala_brandt(
        self,
        points_3d: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        k1: float, k2: float, k3: float, k4: float,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """GPU 加速的 Kannala-Brandt 鱼眼相机投影"""
        pts = self._to_gpu(np.asarray(points_3d, dtype=np.float64))

        x_c = pts[:, 0]
        y_c = pts[:, 1]
        z_c = pts[:, 2]

        valid = (z_c > near_z) & (z_c < far_z)

        safe_z = cp.where(valid, z_c, 1.0)
        x_norm = x_c / safe_z
        y_norm = y_c / safe_z

        r = cp.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = cp.arctan(r)

        theta_d = theta + k1 * theta ** 3 + k2 * theta ** 5 + \
                   k3 * theta ** 7 + k4 * theta ** 9

        safe_r = cp.maximum(r, 1e-10)
        scale = theta_d / safe_r

        x_dist = x_norm * scale
        y_dist = y_norm * scale

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        in_bounds = (u >= -20) & (u < width + 20) & (v >= -20) & (v < height + 20)
        valid = valid & in_bounds

        # 恢复 clamp 行为，与 Python/C++ 路径保持一致
        u = cp.clip(u, 0, width - 1)
        v = cp.clip(v, 0, height - 1)
        u = cp.where(valid, u, -1.0)
        v = cp.where(valid, v, -1.0)

        pixels = cp.stack([u, v], axis=1).get()
        return pixels, valid.get()
    
    def project_ftheta(
        self,
        points_3d: np.ndarray,
        fw_poly: np.ndarray,
        cx: float, cy: float,
        width: int, height: int,
        near_z: float, far_z: float
    ) -> tuple:
        """GPU 加速的 F-Theta 鱼眼相机投影"""
        pts = self._to_gpu(np.asarray(points_3d, dtype=np.float64))
        poly = self._to_gpu(np.asarray(fw_poly, dtype=np.float64))

        x_c = pts[:, 0]
        y_c = pts[:, 1]
        z_c = pts[:, 2]

        valid = (z_c > near_z) & (z_c < far_z)

        theta = cp.arctan2(cp.sqrt(x_c ** 2 + y_c ** 2), z_c)

        # 多项式求值：r = sum(a_i * theta^i)
        r = cp.zeros_like(theta)
        theta_pow = cp.ones_like(theta)
        for i in range(poly.shape[0]):
            r = r + poly[i] * theta_pow
            theta_pow = theta_pow * theta

        phi = cp.arctan2(y_c, x_c)
        u = r * cp.cos(phi) + cx
        v = r * cp.sin(phi) + cy

        in_bounds = (u >= -20) & (u < width + 20) & (v >= -20) & (v < height + 20)
        valid = valid & in_bounds

        # 恢复 clamp 行为，与 Python/C++ 路径保持一致
        u = cp.clip(u, 0, width - 1)
        v = cp.clip(v, 0, height - 1)
        u = cp.where(valid, u, -1.0)
        v = cp.where(valid, v, -1.0)

        pixels = cp.stack([u, v], axis=1).get()
        return pixels, valid.get()
