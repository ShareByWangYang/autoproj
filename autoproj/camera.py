import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple


class Camera(ABC):
    """
    相机基类，定义通用投影接口

    坐标系契约（详见 README "Coordinate System Conventions"）：
        - 相机系：OpenCV 约定（x 右 / y 下 / z 前），原点为光心
        - 像素原点：图像左上角，u 向右增长，v 向下增长
        - 外参 T_to_cam：world→camera 的 4×4 齐次矩阵（p_cam = T @ p_world）
        - 单位：米（near_z/far_z 默认 0.1/1000 米）
        - 畸变系数不可跨模型混用：针孔 k1-k6/p1-p2 与鱼眼 k1-k4 含义完全不同

    Attributes:
        width: 图像宽度（像素）
        height: 图像高度（像素）
        cx: 主点x坐标
        cy: 主点y坐标
        near_z: 近裁剪面深度
        far_z: 远裁剪面深度
        boundary_ratio: 边界 margin 占图像最大尺寸的比例（默认0.02，即2%）
        max_fov_deg: FOV 上限（度），None 表示无上限。
            仅对鱼眼相机（KannalaBrandtCamera, FThetaCamera）生效。
            针孔相机传入此参数会被忽略并给出 warning。
    """

    # _frustum_scale 的默认值：当未手动指定且自动计算未完成时使用
    _DEFAULT_FRUSTUM_SCALE = 1.05

    def __init__(
        self,
        width: int,
        height: int,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        boundary_ratio: float = 0.02,
        max_fov_deg: Optional[float] = None,
        **kwargs
    ):
        # --- 退化输入验证 ---
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive integer, got {width}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive integer, got {height}")
        if not (isinstance(near_z, (int, float)) and near_z > 0):
            raise ValueError(f"near_z must be a positive number, got {near_z}")
        if not (isinstance(far_z, (int, float)) and far_z > 0):
            raise ValueError(f"far_z must be a positive number, got {far_z}")
        if near_z >= far_z:
            raise ValueError(f"near_z ({near_z}) must be less than far_z ({far_z})")

        self.width = width
        self.height = height
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.near_z = near_z
        self.far_z = far_z
        # 用户可配置的边界 margin 比例（占图像最大尺寸的比例）
        # 默认 0.02（2%），用于 _check_bounds 的动态 margin 计算
        self.boundary_ratio = boundary_ratio
        # FOV 上限（度数），None 表示无上限
        # 仅对鱼眼相机生效，用于限制 _theta_max 的搜索范围
        self._max_fov_rad: Optional[float] = np.radians(max_fov_deg) if max_fov_deg is not None else None
        # 子类型标识，由 CameraFactory 创建时设置，用于配置保存时还原类型信息
        self.sub_type: Optional[str] = None
        # 动态视锥缩放因子（由 compute_expansion_factor() 自动计算）
        # None = 尚未计算，Projector 初始化时自动调用 compute_expansion_factor()
        # 用户可通过 Projector(frustum_scale=X) 覆盖此值
        self._frustum_scale: Optional[float] = None
        # 动态扩展因子缓存（由 compute_expansion_factor() 计算，向后兼容）
        self._expansion_factor: Optional[float] = None
        # 真实 FOV 边界（由 compute_expansion_factor() 反解计算）
        # 直接用于 _check_fov，比线性近似更准确
        self._real_tan_h: Optional[float] = None
        self._real_tan_v: Optional[float] = None
        # 是否已经初始化过动态参数
        self._params_initialized: bool = False

        # --- 向后兼容 ---
        if 'fov_tolerance' in kwargs:
            warnings.warn(
                "'fov_tolerance' is deprecated and ignored. "
                "Use Projector(frustum_scale=X) to manually set frustum scale. "
                "Pass frustum_scale=None (default) for auto-calculation.",
                DeprecationWarning, stacklevel=2
            )
        if 'margin' in kwargs:
            old_margin = kwargs.pop('margin')
            if boundary_ratio == 0.02:  # 默认值，未被用户显式设置
                self.boundary_ratio = old_margin / max(width, height)
            warnings.warn(
                f"'margin' is deprecated, converted to boundary_ratio={self.boundary_ratio:.4f}. "
                "Use 'boundary_ratio' directly.",
                DeprecationWarning, stacklevel=2
            )
        if 'max_fov_half_angle' in kwargs:
            old_rad = kwargs.pop('max_fov_half_angle')
            if max_fov_deg is None:
                self._max_fov_rad = old_rad
            warnings.warn(
                "'max_fov_half_angle' (radians) is deprecated, use 'max_fov_deg' (degrees).",
                DeprecationWarning, stacklevel=2
            )
    
    @abstractmethod
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影3D点到图像平面
        
        Args:
            points_3d: 3D点云，形状为(N, 3)
            T_to_cam: 外参变换矩阵（4x4），将点云变换到相机坐标系
            pts_in_cam: 点云是否已在相机坐标系中
        
        Returns:
            (像素坐标, 有效掩码)
        """
        pass

    def _transform_to_camera(self, points_3d: np.ndarray, T_to_cam: np.ndarray) -> np.ndarray:
        """将点云变换到相机坐标系

        使用 R @ p.T + t 的广播形式，避免 hstack 构造 (N,4) 齐次数组带来的
        额外内存分配与拷贝（1M 点规模可省 ~19ms）。
        """
        R = T_to_cam[:3, :3]
        t = T_to_cam[:3, 3]
        return (R @ points_3d.T + t[:, None]).T

    def _check_depth_range(self, points_cam: np.ndarray) -> np.ndarray:
        """检查深度范围"""
        z = points_cam[:, 2]
        return (z > self.near_z) & (z < self.far_z)
    
    def _check_fov(self, x_norm: np.ndarray, y_norm: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于未畸变的归一化坐标检查是否在 FOV 内

        在畸变之前检查归一化坐标的几何边界，防止畸变将 FOV 外的
        大角度点映射回图像内。这是真正的几何 FOV 检查。

        优先使用 compute_expansion_factor() 计算的真实 FOV 边界
        （_real_tan_h, _real_tan_v），并应用 eff_tolerance 与
        compute_frustum_tan_bounds 保持一致的容差计算。

        Args:
            x_norm: 未畸变的 x 归一化坐标 (x/z)
            y_norm: 未畸变的 y 归一化坐标 (y/z)
            tolerance: 容差系数（相对），默认0.05。
                        仅在无真实边界时使用。

        Returns:
            (N,) bool 数组，True 表示在 FOV 内
        """
        # np 已在模块顶部 import, 无需重新绑定

        # 计算有效缩放因子（与 compute_frustum_tan_bounds 保持一致）
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE

        # 优先使用 compute_expansion_factor() 计算的真实 FOV 边界
        if hasattr(self, '_real_tan_h') and self._real_tan_h is not None:
            # 提取为 Python float，避免 NumPy 标量与 CuPy 数组比较时类型错误
            x_max = float(self._real_tan_h) * float(eff_scale)
            y_max = float(self._real_tan_v) * float(eff_scale)
        else:
            # 退化到标称边界 + 默认缩放
            x_max = (self.width - self.cx) / self.fx * float(eff_scale)
            y_max = (self.height - self.cy) / self.fy * float(eff_scale)

        return (np.abs(x_norm) <= x_max) & (np.abs(y_norm) <= y_max)
    
    def _check_bounds(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        检查像素边界

        边界余量由 _get_dynamic_margin() 自动计算
        （基于 boundary_ratio * 图像尺寸 * frustum_scale）。

        Args:
            u: 像素 x 坐标数组
            v: 像素 y 坐标数组
        """
        # np 已在模块顶部 import, 无需重新绑定
        margin = self._get_dynamic_margin()
        return (u >= -margin) & (u < self.width + margin) & \
               (v >= -margin) & (v < self.height + margin)

    def _get_dynamic_margin(self) -> float:
        """
        计算动态边界 margin，供 C++ 后端和 _check_bounds 共用。
        取 max(boundary_ratio * 图像最大尺寸 * 缩放因子, boundary_ratio * 图像最大尺寸)
        """
        base_margin = float(self.boundary_ratio) * max(self.width, self.height)
        scale = self._frustum_scale or 1.0
        return max(base_margin, base_margin * scale)

    def _ensure_params_initialized(self) -> None:
        """
        确保动态参数（frustum_scale, expansion_factor）已初始化。

        在首次调用时自动调用 compute_expansion_factor()，
        使 _check_fov 和 _check_bounds 使用动态缩放和 margin。
        Projector 初始化时也会调用此方法（通过 _compute_effective_expansion）。
        """
        if not self._params_initialized:
            self.compute_expansion_factor()
            self._params_initialized = True

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        通过反解畸变方程，计算图像边界对应的真实入射角。
        子类应覆盖此方法以提供精确的畸变感知 FOV 计算。

        畸变对 FOV 的影响：
        - 桶形畸变（k1<0）：径向压缩 → 真实 FOV > 标称 FOV
        - 枕形畸变（k1>0）：径向拉伸 → 真实 FOV < 标称 FOV

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # 默认：无畸变，使用标称 FOV
        if hasattr(self, 'fov_h'):
            return self.fov_h, self.fov_v
        # np 已在模块顶部 import, 无需重新绑定
        fov_h = 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
        fov_v = 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
        return fov_h, fov_v

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        通过比较考虑畸变的真实 FOV 和标称 FOV，
        计算使视锥边界匹配真实可视范围的扩展因子。

        expansion_factor = tan(θ_real_max) / tan(θ_nominal_max)

        桶形畸变（k1<0）: 真实 FOV > 标称 FOV → expansion > 1.0
        枕形畸变（k1>0）: 真实 FOV < 标称 FOV → expansion < 1.0

        同时自动计算 _frustum_scale 和 _expansion_factor 缓存，
        供 _check_fov 和 _check_bounds 使用。

        Returns:
            扩展因子（1.0 表示无扩展/无畸变）
        """
        # 默认：无畸变，返回 1.0
        self._expansion_factor = 1.0
        if self._frustum_scale is None:
            self._frustum_scale = self._DEFAULT_FRUSTUM_SCALE
        return 1.0


class PinholeCamera(Camera):
    """
    针孔相机模型（支持OpenCV 8参数畸变模型）

    Args:
        width: 图像宽度
        height: 图像高度
        fx: x方向焦距
        fy: y方向焦距
        cx: 主点x坐标（默认图像中心）
        cy: 主点y坐标（默认图像中心）
        dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, k4, k5, k6]
        near_z: 近裁剪面
        far_z: 远裁剪面
        boundary_ratio: 边界 margin 占图像最大尺寸的比例（默认0.02）
        max_fov_deg: FOV 上限（度），None 表示无上限。针孔相机忽略此参数。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dist_coeffs: Optional[Union[list, np.ndarray]] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        boundary_ratio: float = 0.02,
        max_fov_deg: Optional[float] = None,
        **kwargs
    ):
        if max_fov_deg is not None:
            warnings.warn(
                "'max_fov_deg' is ignored for PinholeCamera (only effective for fisheye).",
                UserWarning, stacklevel=2
            )
        super().__init__(
            width=width, height=height, cx=cx, cy=cy,
            near_z=near_z, far_z=far_z,
            boundary_ratio=boundary_ratio,
            max_fov_deg=max_fov_deg,
            **kwargs
        )
        self.fx = fx
        self.fy = fy

        # dist_coeffs 是相机标定参数（数据描述）。
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(8, dtype=np.float64)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
            if len(self.dist_coeffs) < 8:
                padding = np.zeros(8 - len(self.dist_coeffs), dtype=np.float64)
                self.dist_coeffs = np.concatenate([self.dist_coeffs, padding])

        self.k1, self.k2, self.p1, self.p2 = self.dist_coeffs[:4]
        self.k3, self.k4, self.k5, self.k6 = self.dist_coeffs[4:]
        self.has_distortion = bool(np.any(self.dist_coeffs != 0))
    
    @classmethod
    def from_dict(cls, config: dict) -> 'PinholeCamera':
        """
        从字典创建针孔相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fx: x方向焦距
                - fy: y方向焦距
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - dist_coeffs: 畸变系数（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
                - max_fov_deg: FOV上限 度（可选，None表示无上限）
        
        Returns:
            PinholeCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fx=config['fx'],
            fy=config['fy'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            dist_coeffs=config.get('dist_coeffs'),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0),
            boundary_ratio=config.get('boundary_ratio', 0.02),
            max_fov_deg=config.get('max_fov_deg')
        )
    
    @property
    def fov_h(self) -> float:
        """水平视场角（度）"""
        # np 已在模块顶部 import, 无需重新绑定
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
    
    @property
    def fov_v(self) -> float:
        """垂直视场角（度）"""
        # np 已在模块顶部 import, 无需重新绑定
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
    
    def _apply_distortion(self, x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用畸变校正"""
        # np 已在模块顶部 import, 无需重新绑定
        if not self.has_distortion:
            return x_norm, y_norm

        # 提取畸变系数为 Python float，避免 NumPy 标量与 CuPy 数组混合运算
        k1 = float(self.k1); k2 = float(self.k2); k3 = float(self.k3)
        k4 = float(self.k4); k5 = float(self.k5); k6 = float(self.k6)
        p1 = float(self.p1); p2 = float(self.p2)

        r2 = x_norm ** 2 + y_norm ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3

        numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
        denom = 1 + k4 * r2 + k5 * r4 + k6 * r6
        denom = np.maximum(denom, 1e-10)
        radial = numerator / denom

        x_dist = x_norm * radial
        y_dist = y_norm * radial

        x_dist += 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
        y_dist += p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

        return x_dist, y_dist

    def _radial_factor(self, r2: float) -> float:
        """计算径向畸变因子（给定 r²）"""
        r4 = r2 * r2
        r6 = r4 * r2
        numerator = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        denom = 1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
        denom = max(denom, 1e-10)
        return numerator / denom

    def _bisect_radial_1d(self, target: float, y_fixed: float = 0.0,
                          max_iter: int = 100, tol: float = 1e-12) -> float:
        """
        求解一维径向畸变方程：x · radial(x, y_fixed) = target

        关键：当分母 k4,k5,k6 较大时，f(x) = x · radial(x) 先增后减，
        存在物理最大值 f_max。若 target > f_max，则返回 f_max 对应的
        x（物理极限，FOV 饱和）。

        扫描范围自适应：从 target 的倍数开始，逐步扩展到找到物理最大值。

        Args:
            target: 畸变后的归一化坐标值
            y_fixed: 固定的 y 坐标（水平 FOV 计算时为 0）
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            畸变前的归一化坐标 x
        """
        if abs(target) < 1e-12:
            return 0.0

        target_signed = target
        target_abs = abs(target_signed)
        sign = 1.0 if target_signed >= 0 else -1.0

        def f(x):
            r2 = x * x + y_fixed * y_fixed
            return x * self._radial_factor(r2)

        # 自适应扫描：从 target 的倍数开始，逐步扩展
        # 仅扫描物理正区间（radial_factor > 0, f(x) > 0）
        scan_max = max(10.0, 5.0 * target_abs)
        max_x, max_f = 0.0, 0.0
        for _ in range(5):
            for x in np.arange(0.001, scan_max, scan_max / 200.0):
                r2 = x * x + y_fixed * y_fixed
                rf = self._radial_factor(r2)
                fx = x * rf
                if rf > 0 and fx > max_f:
                    max_x, max_f = x, fx
            if max_f >= target_abs or scan_max >= 100.0 * max(1.0, target_abs):
                break
            scan_max *= 2.0

        # 如果 target 超过物理最大值，返回 max_x 作为上限估计
        if target_abs > max_f:
            return sign * max_x

        # 在 [0, max_x] 内用二分法精确求解
        lo, hi = 0.001, max_x
        result, best_err = 0.0, float('inf')

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            f_mid = abs(f(mid))
            err = abs(f_mid - target_abs)
            if err < best_err:
                best_err = err
                result = mid

            if f_mid > target_abs:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break

        # 如果二分法收敛到的点不理想，尝试在 max_x 附近找更好的解
        if best_err > tol:
            for dx in np.arange(-0.5, 0.5, 0.01):
                x_test = max_x + dx
                if x_test > 0:
                    f_test = abs(f(x_test))
                    err = abs(f_test - target_abs)
                    if err < best_err:
                        best_err = err
                        result = x_test

        return sign * abs(result)

    def _undistort_normalized(
        self,
        x_dist: float,
        y_dist: float,
        max_iter: int = 50,
        tol: float = 1e-12
    ) -> Tuple[float, float]:
        """
        反解畸变方程：给定畸变后归一化坐标，求畸变前归一化坐标

        策略：先用二分法求解纯径向畸变（稳定收敛），
        再用固定点迭代加入切向畸变修正。

        对于 FOV 计算场景（y_dist=0 或 x_dist=0），
        二分法直接给出精确解，无需后续迭代。

        Args:
            x_dist: 畸变后 x 归一化坐标
            y_dist: 畸变后 y 归一化坐标
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            (x_norm, y_norm): 畸变前归一化坐标
        """
        if not self.has_distortion:
            return x_dist, y_dist

        # 用二分法求解径向部分（作为初值）
        x = self._bisect_radial_1d(x_dist, y_fixed=0.0)
        y = self._bisect_radial_1d(y_dist, y_fixed=0.0)

        # 检查是否为折叠点解（target 超过物理最大值时 _bisect_radial_1d
        # 返回折叠点）。此时多项式模型已无法表示更大角度，直接返回折叠点。
        # 避免后续定点迭代发散（径向因子变负时 x_dist/radial 会震荡）。
        def _check_folding(x_val, target_val):
            """检查 x_val 是否为折叠点：f(x_val) != target_val"""
            r2 = x_val * x_val
            f_val = abs(x_val * self._radial_factor(r2))
            return abs(f_val - abs(target_val)) > 0.01 * max(1.0, abs(target_val))

        x_is_folding = _check_folding(x, x_dist)
        y_is_folding = _check_folding(y, y_dist)

        if x_is_folding or y_is_folding:
            # 折叠点：多项式模型物理极限，直接返回，不做迭代
            return x, y

        # 如果切向畸变系数为零，二分法已经给出精确解
        if abs(self.p1) < 1e-15 and abs(self.p2) < 1e-15:
            # 验证：检查二分法是否同时满足 x 和 y 方程
            # 对 x 方程：x_dist = x * radial(x, y)
            r2_check = x * x + y * y
            radial_check = self._radial_factor(r2_check)
            x_residual = x * radial_check - x_dist
            y_residual = y * radial_check - y_dist

            # 如果残差较大，说明需要联立求解
            if abs(x_residual) > 1e-10 or abs(y_residual) > 1e-10:
                # 用固定点迭代精炼（对小残差有效）
                for _ in range(max_iter):
                    r2 = x * x + y * y
                    radial = self._radial_factor(r2)
                    x_new = x_dist / radial
                    y_new = y_dist / radial
                    if abs(x_new - x) < tol and abs(y_new - y) < tol:
                        x, y = x_new, y_new
                        break
                    x, y = x_new, y_new
            return x, y

        # 有切向畸变：用固定点迭代精炼（对强畸变可能不收敛，
        # 但二分法已给出较好初值，通常能在几步内收敛）
        for _ in range(max_iter):
            r2 = x * x + y * y
            radial = self._radial_factor(r2)

            x_new = (x_dist - 2 * self.p1 * x * y - self.p2 * (r2 + 2 * x * x)) / radial
            y_new = (y_dist - self.p1 * (r2 + 2 * y * y) - 2 * self.p2 * x * y) / radial

            if abs(x_new - x) < tol and abs(y_new - y) < tol:
                x, y = x_new, y_new
                break

            x, y = x_new, y_new

        return x, y

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        通过反解畸变方程，计算图像边界对应的真实入射角。

        图像边界对应的畸变后归一化坐标：
            x_dist_right = (width - 1 - cx) / fx
            x_dist_left  = -cx / fx
            y_dist_top   = (height - 1 - cy) / fy
            y_dist_bottom= -cy / fy

        反解得到畸变前归一化坐标后，计算入射角：
            θ = arctan(x_norm)

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        if not self.has_distortion:
            return self.fov_h, self.fov_v

        # 图像边界对应的畸变后归一化坐标
        x_dist_right = (self.width - 1 - self.cx) / self.fx
        x_dist_left = -self.cx / self.fx
        y_dist_top = (self.height - 1 - self.cy) / self.fy
        y_dist_bottom = -self.cy / self.fy

        # 反解得到畸变前归一化坐标
        # 水平 FOV：在 y=0 平面计算
        x_norm_right, _ = self._undistort_normalized(x_dist_right, 0.0)
        x_norm_left, _ = self._undistort_normalized(x_dist_left, 0.0)

        # 垂直 FOV：在 x=0 平面计算
        _, y_norm_top = self._undistort_normalized(0.0, y_dist_top)
        _, y_norm_bottom = self._undistort_normalized(0.0, y_dist_bottom)

        # 真实 FOV（度）= arctan(right) + arctan(|left|)
        fov_h_real = np.degrees(np.arctan(x_norm_right) + np.arctan(abs(x_norm_left)))
        fov_v_real = np.degrees(np.arctan(y_norm_top) + np.arctan(abs(y_norm_bottom)))

        return fov_h_real, fov_v_real

    def compute_frustum_tan_bounds(self, margin_px: Optional[float] = None) -> Tuple[float, float]:
        """
        计算视锥裁剪的有效 tan 边界（水平和垂直方向）。

        核心原则：视锥裁剪的 tan 边界必须与点云投影的 FOV 检查一致。
        点云使用 _check_fov(x_norm, y_norm) 判断有效性，其边界为
        _real_tan_h * (1 + tol)。视锥必须使用相同的边界，才能保证
        裁剪后的 box 棱线端点与点云具有一致的可见性。

        因此：frustum_tan_h = _real_tan_h * (1 + tol)
              frustum_tan_v = _real_tan_v * (1 + tol)

        这确保了：
        1. 被点云 FOV 检查接受的点，也会被视锥接受
        2. 视锥裁剪后的端点，其投影像素在图像范围内
        3. 对于桶形畸变（k1<0），tan > _real_tan，正确扩展视锥
        4. 对于枕形畸变（k1>0），tan < _real_tan，正确收缩视锥

        Args:
            margin_px: 边界余量（像素），保留接口一致性，实际 FOV 边界基于 tol

        Returns:
            (frustum_tan_h, frustum_tan_v): 用于视锥裁剪的 tan 边界
        """
        # 使用 _real_tan_h/v（已由 compute_expansion_factor 计算）
        if self._real_tan_h is None or self._real_tan_v is None:
            # 回退：基于标称 FOV
            tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
            tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
            return max(tan_h_nominal, 1e-10), max(tan_v_nominal, 1e-10)

        # FOV 缩放因子（与 _check_fov 中的 eff_scale 一致）
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE

        # frustum_tan = real_tan * eff_scale
        # 这与 _check_fov 的边界完全一致
        frustum_tan_h = self._real_tan_h * eff_scale
        frustum_tan_v = self._real_tan_v * eff_scale

        return max(frustum_tan_h, 1e-10), max(frustum_tan_v, 1e-10)

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        比较考虑畸变的真实 FOV 和标称 FOV，
        同时计算对角线方向的扩展因子（对角 FOV 通常比 H/V 更大）。

        桶形畸变（k1<0）: expansion > 1.0
        枕形畸变（k1>0）: expansion < 1.0

        同时自动计算 _frustum_scale 和 _expansion_factor 缓存，
        供 _check_fov 和 _check_bounds 使用。

        Returns:
            扩展因子
        """
        if not self.has_distortion:
            self._expansion_factor = 1.0
            if self._frustum_scale is None:
                self._frustum_scale = self._DEFAULT_FRUSTUM_SCALE
            
            # 即使无畸变，也计算 _real_tan 和 _frustum_tan
            # 确保 _check_fov 和 FrustumCuller 使用一致的边界
            tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
            tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
            
            self._real_tan_h = tan_h_nominal
            self._real_tan_v = tan_v_nominal
            self._frustum_tan_h, self._frustum_tan_v = self.compute_frustum_tan_bounds()
            
            return 1.0

        # 标称 tan 值（取较大的半边）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy

        # 图像边界对应的畸变后归一化坐标
        x_dist_right = (self.width - 1 - self.cx) / self.fx
        x_dist_left = -self.cx / self.fx
        y_dist_top = (self.height - 1 - self.cy) / self.fy
        y_dist_bottom = -self.cy / self.fy

        # 反解得到畸变前归一化坐标
        x_norm_right, _ = self._undistort_normalized(x_dist_right, 0.0)
        x_norm_left, _ = self._undistort_normalized(x_dist_left, 0.0)
        _, y_norm_top = self._undistort_normalized(0.0, y_dist_top)
        _, y_norm_bottom = self._undistort_normalized(0.0, y_dist_bottom)

        # 真实最大 |tan| 值
        tan_h_real = max(abs(x_norm_right), abs(x_norm_left))
        tan_v_real = max(abs(y_norm_top), abs(y_norm_bottom))

        # 保存真实 FOV 边界供 _check_fov 使用
        self._real_tan_h = tan_h_real
        self._real_tan_v = tan_v_real

        # 扩展因子：取水平和垂直中较大的（基于真实 tan）
        expansion_h = tan_h_real / tan_h_nominal if tan_h_nominal > 1e-10 else 1.0
        expansion_v = tan_v_real / tan_v_nominal if tan_v_nominal > 1e-10 else 1.0

        # 对角线扩展因子（图像角落点的 FOV 通常比 H/V 更大）
        tan_diag_nominal = np.sqrt(tan_h_nominal**2 + tan_v_nominal**2)
        x_corner = (self.width - 1 - self.cx) / self.fx
        y_corner = (self.height - 1 - self.cy) / self.fy
        x_norm_corner, y_norm_corner = self._undistort_normalized(x_corner, y_corner)
        tan_diag_real = np.sqrt(x_norm_corner**2 + y_norm_corner**2)
        expansion_diag = tan_diag_real / tan_diag_nominal if tan_diag_nominal > 1e-10 else 1.0

        expansion = max(expansion_h, expansion_v, expansion_diag)

        # 先缓存扩展因子和动态缩放因子（在 compute_frustum_tan_bounds 之前）
        self._expansion_factor = expansion
        if self._frustum_scale is None:
            # 自动计算缩放因子：clamp 到 [_DEFAULT_FRUSTUM_SCALE, 1.15]
            self._frustum_scale = max(self._DEFAULT_FRUSTUM_SCALE, min(expansion, 1.15))

        # 然后计算视锥裁剪的有效 tan 边界（此时 _frustum_scale 已更新）
        self._frustum_tan_h, self._frustum_tan_v = self.compute_frustum_tan_bounds()

        return expansion
    
    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        # 确保动态参数（expansion_factor, frustum_scale）已初始化
        self._ensure_params_initialized()

        # np 已在模块顶部 import, 无需重新绑定
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        # 提取XYZ进行计算
        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        # NumPy 向量化路径 (唯一实现)
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        # 使用 safe_z 除法替代 masked assignment（更高效，invalid 点 z≤0
        # 会被 depth check 过滤，x_norm≈0 不影响结果）
        safe_z = np.maximum(z_c, 1e-10)
        # 除法已产生 float64，无需额外 astype
        x_norm = x_c / safe_z
        y_norm = y_c / safe_z

        # 合并深度+FOV检查为单次 logical_and.reduce，减少中间布尔数组分配
        # （1M 点规模下省 ~3 个 (N,) bool 临时数组 ≈ 3MB）
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE
        if hasattr(self, '_real_tan_h') and self._real_tan_h is not None:
            x_max = float(self._real_tan_h) * float(eff_scale)
            y_max = float(self._real_tan_v) * float(eff_scale)
        else:
            x_max = (self.width - self.cx) / self.fx * float(eff_scale)
            y_max = (self.height - self.cy) / self.fy * float(eff_scale)
        valid = np.logical_and.reduce([
            z_c > self.near_z,
            z_c < self.far_z,
            np.abs(x_norm) <= x_max,
            np.abs(y_norm) <= y_max
        ])

        x_dist, y_dist = self._apply_distortion(x_norm, y_norm)

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        # 对有效点 UV 做 clamp 到 [0, w-1]，无效点置 -1
        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            # 向后兼容：返回 (N, 2) int32
            # 先 stack 再 astype，减少 1 次 astype 调用（2→1）
            result = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
            return result, valid

        # preserve_extra=True: 返回 (N, 3+) float64
        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        # np.trunc 替代 astype(int32) 避免 int32 中间临时数组
        result[:, 0] = np.trunc(u_clip)
        result[:, 1] = np.trunc(v_clip)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid


class KannalaBrandtCamera(Camera):
    """
    Kannala-Brandt鱼眼相机模型（θ多项式畸变）

    Args:
        width: 图像宽度
        height: 图像高度
        fx: x方向焦距
        fy: y方向焦距
        cx: 主点x坐标
        cy: 主点y坐标
        k1, k2, k3, k4: 畸变系数（θ^3, θ^5, θ^7, θ^9系数）
        near_z: 近裁剪面
        far_z: 远裁剪面
        boundary_ratio: 边界 margin 占图像最大尺寸的比例（默认0.02）
        max_fov_deg: FOV 上限（度），None 表示无上限。
            例如 180 = 限制全FOV为180°，超出范围的目标被舍弃。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        boundary_ratio: float = 0.02,
        max_fov_deg: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            width=width, height=height, cx=cx, cy=cy,
            near_z=near_z, far_z=far_z,
            boundary_ratio=boundary_ratio,
            max_fov_deg=max_fov_deg,
            **kwargs
        )
        self.fx = fx
        self.fy = fy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        # 预计算最大物理入射角 θ_max (rad)，用于 FOV 几何检查
        #
        # 原理：Kannala-Brandt 模型中，图像半径 r = f · θ_d(θ)。
        # 最大半径 r_max 是从主点到图像最远角落/边缘的距离。
        # 真实 θ_max 通过反解 θ_d(θ) = r_max/f 获得，
        # 比针孔近似 (arctan(r_max/f)) 更准确，
        # 能够正确反映鱼眼相机的实际可视范围（通常 80°~100°+ 半角）。
        self._theta_max = self._compute_theta_max()

    def _compute_theta_max(self) -> float:
        """
        数值求解 Kannala-Brandt 模型的最大物理入射角 θ_max (rad)

        反解：θ_d(θ_max) = r_max / fx (或 fy，取较大者以覆盖更广 FOV)
        其中 θ_d(θ) = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹

        使用二分法求解，搜索上限由 max_fov_deg 决定：
        - None: 无上限（搜索 [0, π]，支持 >180° FOV）
        - 设为值: 搜索 [0, _max_fov_rad]
        当目标值超过 θ_d 多项式物理最大值时，返回物理极限对应的 θ。

        注意：二分法仅在 θ_d 单调递增区间 [0, max_theta] 内有效，
        因此搜索上界约束为 min(theta_limit, max_theta)。

        此方法为几何参数预计算，保证返回值为 Python float。
        """
        r_max = max(
            self.cx, self.width - 1 - self.cx,
            self.cy, self.height - 1 - self.cy,
        )
        theta_d_target_h = r_max / self.fx
        theta_d_target_v = r_max / self.fy
        theta_d_target = max(theta_d_target_h, theta_d_target_v)

        # 确定搜索上限
        theta_limit = self._max_fov_rad if self._max_fov_rad is not None else np.pi

        # 将畸变系数提取为 Python float，避免 NumPy 标量在迭代时引入额外开销
        k1 = float(self.k1); k2 = float(self.k2)
        k3 = float(self.k3); k4 = float(self.k4)

        def eval_theta_d(theta: float) -> float:
            return (theta
                    + k1 * theta**3
                    + k2 * theta**5
                    + k3 * theta**7
                    + k4 * theta**9)

        # 无畸变：θ = arctan(r_max/f)，而非 r_max/f 本身
        # r_max/f 是 tan(θ)，需要转成角度 θ = arctan(r_max/f)
        if abs(k1) < 1e-15 and abs(k2) < 1e-15 \
                and abs(k3) < 1e-15 and abs(k4) < 1e-15:
            return float(min(np.arctan(theta_d_target), theta_limit))

        # 先扫描找 θ_d 物理最大值及对应 θ（仅取正值区间，上限 theta_limit）
        theta_arr = np.arange(0.001, theta_limit, 0.001)
        theta_d_arr = np.array([eval_theta_d(t) for t in theta_arr])
        mask = theta_d_arr > 0
        if mask.any():
            idx_max = np.argmax(theta_d_arr[mask])
            max_theta_d = float(theta_d_arr[mask][idx_max])
            max_theta = float(theta_arr[mask][idx_max])
        else:
            idx_max = np.argmax(np.abs(theta_d_arr))
            max_theta_d = abs(float(theta_d_arr[idx_max]))
            max_theta = float(theta_arr[idx_max])

        if theta_d_target > max_theta_d:
            return max_theta

        # 二分法搜索区间 [0, min(theta_limit, max_theta)]
        # 必须限制在 θ_d 单调递增区间内，避免折叠回折导致二分法收敛到错误根
        search_limit = min(theta_limit, max_theta)
        lo, hi = 0.0, search_limit
        for _ in range(80):
            mid = (lo + hi) / 2
            if eval_theta_d(mid) < theta_d_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14:
                break
        return (lo + hi) / 2

    def _check_fov(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        tolerance: float = 0.05
    ) -> np.ndarray:
        """
        鱼眼相机 FOV 几何检查（覆盖父类方法）

        使用入射角 θ（而非针孔归一化坐标）判断是否在视场内：
          θ = arctan(sqrt(x_norm² + y_norm²))
          θ ≤ θ_max · (1 + tolerance)

        这对任意 θ_max（包括 > 90° 的鱼眼）均稳定，
        避免了针孔近似 tan(θ_max) = max|x_norm| 的严重低估问题。
        """
        # np 已在模块顶部 import, 无需重新绑定
        r_norm = np.sqrt(x_norm ** 2 + y_norm ** 2)
        # θ = arctan(r)，对大角度数值稳定
        theta = np.arctan(r_norm)
        # 使用动态/用户缩放因子（若已设置），否则使用默认值
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE
        # 提取为 Python float，避免 NumPy 标量与 CuPy 数组比较错误
        theta_limit = float(self._theta_max) * float(eff_scale)
        return theta <= theta_limit

    def _check_depth_range(self, points_cam: np.ndarray) -> np.ndarray:
        """
        覆盖基类方法：使用径向距离检查深度范围

        鱼眼相机的有效点可能 z < 0（θ > 90° 的边缘点），
        因此不能仅用 z > near_z 判断，必须使用径向距离
        r = sqrt(x² + y² + z²) >= near_z。
        """
        # np 已在模块顶部 import, 无需重新绑定
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        return (r >= self.near_z) & (r < self.far_z)

    @classmethod
    def from_dict(cls, config: dict) -> 'KannalaBrandtCamera':
        """
        从字典创建Kannala-Brandt鱼眼相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fx: x方向焦距
                - fy: y方向焦距
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - k1, k2, k3, k4: 畸变系数（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
                - max_fov_deg: FOV上限 度（可选，None表示无上限）
        
        Returns:
            KannalaBrandtCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fx=config['fx'],
            fy=config['fy'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            k1=config.get('k1', 0.0),
            k2=config.get('k2', 0.0),
            k3=config.get('k3', 0.0),
            k4=config.get('k4', 0.0),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0),
            boundary_ratio=config.get('boundary_ratio', 0.02),
            max_fov_deg=config.get('max_fov_deg')
        )

    @property
    def fov_h(self) -> float:
        """水平视场角（度）— 标称值，基于针孔模型近似"""
        # np 已在模块顶部 import, 无需重新绑定
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))

    @property
    def fov_v(self) -> float:
        """垂直视场角（度）— 标称值，基于针孔模型近似"""
        # np 已在模块顶部 import, 无需重新绑定
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))

    def _undistort_theta(self, theta_d: float, max_iter: int = 100, tol: float = 1e-12) -> float:
        """
        反解 Kannala-Brandt θ 多项式：给定畸变后角度 θ_d，求真实入射角 θ

        畸变模型：θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹

        使用二分法反解（θ 单调递增映射到 θ_d 的正区间）。
        当 theta_d 超过物理最大值时，返回物理极限对应的 θ。
        搜索上限由 max_fov_deg 决定（若设置）。

        Args:
            theta_d: 畸变后角度（弧度）
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            θ: 真实入射角（弧度）
        """
        # 无畸变时直接返回
        if self.k1 == 0 and self.k2 == 0 and self.k3 == 0 and self.k4 == 0:
            return theta_d

        # 确定搜索上限（与 _compute_theta_max 保持一致）
        theta_limit = self._max_fov_rad if self._max_fov_rad is not None else np.pi

        def eval_theta_d(theta):
            return (theta + self.k1 * theta**3 + self.k2 * theta**5
                    + self.k3 * theta**7 + self.k4 * theta**9)

        # 先扫描找 θ_d 物理最大值及对应 θ（仅取正值区间，上限 theta_limit）
        theta_arr = np.arange(0.001, theta_limit, 0.001)
        theta_d_arr = np.array([eval_theta_d(t) for t in theta_arr])
        mask = theta_d_arr > 0
        if mask.any():
            idx_max = np.argmax(theta_d_arr[mask])
            max_theta_d = theta_d_arr[mask][idx_max]
            max_theta = theta_arr[mask][idx_max]
        else:
            idx_max = np.argmax(np.abs(theta_d_arr))
            max_theta_d = abs(theta_d_arr[idx_max])
            max_theta = theta_arr[idx_max]

        # 如果 theta_d 超过物理最大值，返回模型物理极限对应的 θ
        # 确保返回值语义正确（物理入射角），而非畸变后角度
        if theta_d > max_theta_d:
            return max_theta

        # 二分法：θ ∈ [0, min(theta_limit, max_theta)]
        # 必须限制在 θ_d 单调递增区间内，避免折叠回折导致二分法收敛到错误根
        search_limit = min(theta_limit, max_theta)
        lo, hi = 0.0, search_limit
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            theta_d_pred = eval_theta_d(mid)
            if theta_d_pred < theta_d:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        鱼眼畸变模型：θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
        图像半径：r = fx · θ_d

        图像边界对应的最大半径 r_max，反解得到真实入射角 θ_max。

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # 图像边界对应的最大半径
        r_max_h = max(self.cx, self.width - 1 - self.cx)
        r_max_v = max(self.cy, self.height - 1 - self.cy)

        # 畸变后角度
        theta_d_h = r_max_h / self.fx
        theta_d_v = r_max_v / self.fy

        # 反解得到真实入射角
        theta_real_h = self._undistort_theta(theta_d_h)
        theta_real_v = self._undistort_theta(theta_d_v)

        # 转为度数（FOV = 2 × θ_max）
        fov_h_real = 2 * np.degrees(theta_real_h)
        fov_v_real = 2 * np.degrees(theta_real_v)

        return fov_h_real, fov_v_real

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        对于圆锥视锥（鱼眼），expansion_factor = θ_real / θ_nominal

        其中：
            θ_nominal = arctan(max(cx, w-1-cx) / fx)  （针孔模型近似）
            θ_real 通过反解 θ 多项式获得

        同时自动计算 _frustum_scale 和 _expansion_factor 缓存，
        供 _check_fov 和 _check_bounds 使用。

        Returns:
            扩展因子
        """
        # 标称半角（针孔模型近似）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
        theta_nominal_h = np.arctan(tan_h_nominal)
        theta_nominal_v = np.arctan(tan_v_nominal)

        # 真实半角
        fov_h_real, fov_v_real = self.compute_real_fov()
        theta_real_h = np.radians(fov_h_real / 2)
        theta_real_v = np.radians(fov_v_real / 2)

        # 扩展因子（角度比）
        expansion_h = theta_real_h / theta_nominal_h if theta_nominal_h > 1e-10 else 1.0
        expansion_v = theta_real_v / theta_nominal_v if theta_nominal_v > 1e-10 else 1.0

        expansion = max(expansion_h, expansion_v)

        # 缓存扩展因子和动态缩放因子
        self._expansion_factor = expansion
        # 2D延长已补偿视觉连续性，使用固定小缩放即可
        if self._frustum_scale is None:
            self._frustum_scale = self._DEFAULT_FRUSTUM_SCALE

        return expansion

    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        # 确保动态参数（expansion_factor, frustum_scale）已初始化
        self._ensure_params_initialized()

        # np 已在模块顶部 import, 无需重新绑定
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        # NumPy 向量化路径 (唯一实现)
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        # 使用原始相机坐标计算入射角 θ ∈ [0, π]
        # 支持 θ > 90° 的大鱼眼（z < 0 的边缘点）
        # 复用 r_xy² 避免重复计算 x²+y²
        r_xy_sq = x_c**2 + y_c**2
        # arctan2(r_xy, z) 等价于 arccos(z/r_3d)，但少算一步（无需 clip，
        # 也无需 r_3d 开方）；深度范围检查已在下方合并完成
        theta = np.arctan2(np.sqrt(r_xy_sq), z_c)

        # 合并深度+FOV检查为单次 logical_and.reduce，减少中间布尔数组分配
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE
        theta_limit = float(self._theta_max) * float(eff_scale)
        valid = np.logical_and.reduce([
            z_c > self.near_z,
            z_c < self.far_z,
            theta <= theta_limit
        ])

        # 畸变投影
        k1 = float(self.k1); k2 = float(self.k2)
        k3 = float(self.k3); k4 = float(self.k4)
        theta_d = theta + k1 * theta**3 + k2 * theta**5 + \
                  k3 * theta**7 + k4 * theta**9

        r_xy = np.sqrt(r_xy_sq)
        safe_r_xy = np.maximum(r_xy, 1e-10)
        scale = theta_d / safe_r_xy

        x_dist = x_c * scale
        y_dist = y_c * scale

        u = float(self.fx) * x_dist + float(self.cx)
        v = float(self.fy) * y_dist + float(self.cy)

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            # 先 stack 再 astype，减少 1 次 astype 调用（2→1）
            result = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
            return result, valid

        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        # np.trunc 替代 astype(int32) 避免 int32 中间临时数组
        result[:, 0] = np.trunc(u_clip)
        result[:, 1] = np.trunc(v_clip)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid


class FThetaCamera(Camera):
    """
    F-Theta等距鱼眼相机模型（多项式映射）

    Args:
        width: 图像宽度
        height: 图像高度
        fw_poly: 焦距多项式系数 [a0, a1, a2, ...]
        cx: 主点x坐标
        cy: 主点y坐标
        near_z: 近裁剪面
        far_z: 远裁剪面
        boundary_ratio: 边界 margin 占图像最大尺寸的比例（默认0.02）
        max_fov_deg: FOV 上限（度），None 表示无上限。
            例如 180 = 限制全FOV为180°，超出范围的目标被舍弃。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fw_poly: Union[list, np.ndarray],
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        near_z: float = 0.1,
        far_z: float = 1000.0,
        boundary_ratio: float = 0.02,
        max_fov_deg: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            width=width, height=height, cx=cx, cy=cy,
            near_z=near_z, far_z=far_z,
            boundary_ratio=boundary_ratio,
            max_fov_deg=max_fov_deg,
            **kwargs
        )
        # fw_poly 是相机标定参数（数据描述）。
        self.fw_poly = np.array(fw_poly, dtype=np.float64)
        self._theta_max = self._compute_theta_max()
    
    def _compute_theta_max(self) -> float:
        """
        数值求解 FOV 最大角度 θ_max，使得 fw_poly(θ_max) = r_max

        使用二分法搜索，上限由 max_fov_deg 决定：
        - None: 无上限（搜索 [0, π]，支持 >180° FOV）
        - 设为值: 搜索 [0, _max_fov_rad]
        当目标超过多项式物理最大值时，返回物理极限对应的 θ。
        当多项式接近线性（fw_poly[0]=0, fw_poly[1]=fx）时，
        θ_max = r_max/fx，但需用 arctan 约束物理合理性。

        此方法为几何参数预计算，保证返回值为 Python float。
        """
        r_max = max(self.cx, self.width - self.cx, self.cy, self.height - self.cy)

        # 确定搜索上限
        theta_limit = self._max_fov_rad if self._max_fov_rad is not None else np.pi

        # 多项式系数提取为 Python list
        fw_poly_list = [float(c) for c in self.fw_poly]

        def eval_poly(theta):
            result = 0.0
            for i, coeff in enumerate(fw_poly_list):
                result += coeff * (theta ** i)
            return result

        # 检测是否为近似线性模型（fw_poly[0]=0, fw_poly[1]>0, 其余≈0）
        is_linear = (len(fw_poly_list) >= 2
                     and abs(fw_poly_list[0]) < 1e-10
                     and fw_poly_list[1] > 0
                     and all(abs(c) < 1e-10 for c in fw_poly_list[2:]))
        if is_linear:
            # 线性模型: fw_poly(θ) = fx * θ → θ_max = r_max/fx
            # 但物理上 θ 不能超过 π，且应使用 arctan 约束
            fx = fw_poly_list[1]
            return float(min(np.arctan(r_max / fx), theta_limit))

        # 先扫描找多项式物理最大值及对应 θ（仅取正值区间，上限 theta_limit）
        theta_arr = np.arange(0.001, theta_limit, 0.001)
        r_arr = np.array([eval_poly(t) for t in theta_arr])
        mask = r_arr > 0
        if mask.any():
            idx_max = np.argmax(r_arr[mask])
            max_r = float(r_arr[mask][idx_max])
            max_theta = float(theta_arr[mask][idx_max])
        else:
            idx_max = np.argmax(np.abs(r_arr))
            max_r = abs(float(r_arr[idx_max]))
            max_theta = float(theta_arr[idx_max])

        if r_max > max_r:
            return max_theta

        # 二分法搜索区间 [0, min(theta_limit, max_theta)]
        # 必须限制在多项式单调递增区间内，避免折叠回折导致二分法收敛到错误根
        search_limit = min(theta_limit, max_theta)
        lo, hi = 0.0, search_limit
        for _ in range(80):
            mid = (lo + hi) / 2
            if eval_poly(mid) < r_max:
                lo = mid
            else:
                hi = mid
        return lo
    
    def _check_fov_theta(self, theta: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于角度 θ 检查是否在 FOV 内（F-Theta 模型）

        Args:
            theta: 入射角数组 (rad)
            tolerance: 容差系数（相对），默认5%

        Returns:
            (N,) bool 数组
        """
        # np 已在模块顶部 import, 无需重新绑定
        # 使用动态/用户缩放因子（若已设置），否则使用默认值
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE
        # 提取为 Python float，避免 NumPy 标量与 CuPy 数组比较错误
        theta_max = float(self._theta_max) * float(eff_scale)
        return theta <= theta_max
    
    def _check_fov_ftheta(self, x_norm: np.ndarray, y_norm: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
        """
        基于未畸变归一化坐标的 FOV 检查（F-Theta 模型）
        
        将归一化坐标转换为角度后与 θ_max 比较。
        
        Args:
            x_norm: 未畸变 x 归一化坐标
            y_norm: 未畸变 y 归一化坐标
            tolerance: 容差系数
        """
        # np 已在模块顶部 import, 无需重新绑定
        r_norm = np.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = np.arctan(r_norm)
        return self._check_fov_theta(theta, tolerance)
    
    def _check_depth_range(self, points_cam: np.ndarray) -> np.ndarray:
        """
        覆盖基类方法：使用径向距离检查深度范围

        F-Theta 鱼眼相机的有效点可能 z < 0（θ > 90° 的边缘点），
        因此使用径向距离 r = sqrt(x² + y² + z²) >= near_z。
        """
        # np 已在模块顶部 import, 无需重新绑定
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        return (r >= self.near_z) & (r < self.far_z)

    @classmethod
    def from_dict(cls, config: dict) -> 'FThetaCamera':
        """
        从字典创建F-Theta鱼眼相机
        
        Args:
            config: 相机配置字典，包含以下键：
                - width: 图像宽度
                - height: 图像高度
                - fw_poly: 焦距多项式系数
                - cx: 主点x坐标（可选）
                - cy: 主点y坐标（可选）
                - near_z: 近裁剪面（可选）
                - far_z: 远裁剪面（可选）
                - max_fov_deg: FOV上限 度（可选，None表示无上限）
        
        Returns:
            FThetaCamera实例
        """
        return cls(
            width=config['width'],
            height=config['height'],
            fw_poly=config['fw_poly'],
            cx=config.get('cx'),
            cy=config.get('cy'),
            near_z=config.get('near_z', 0.1),
            far_z=config.get('far_z', 1000.0),
            boundary_ratio=config.get('boundary_ratio', 0.02),
            max_fov_deg=config.get('max_fov_deg')
        )

    @property
    def fx(self) -> float:
        """等效 x 焦距（取多项式一阶系数）"""
        return self.fw_poly[1] if len(self.fw_poly) > 1 else 1.0

    @property
    def fy(self) -> float:
        """等效 y 焦距（同 fx）"""
        return self.fx

    def compute_real_fov(self) -> Tuple[float, float]:
        """
        计算考虑畸变的真实 FOV（度）

        F-Theta 模型：r = Σ coeff_i · θ^i
        _theta_max 已通过二分法求解 r(θ) = r_max 获得，即为真实半 FOV。

        Returns:
            (fov_h_real, fov_v_real): 水平和垂直真实 FOV（度）
        """
        # _theta_max 已是真实入射角（通过二分法求解多项式方程获得）
        fov_h_real = 2 * np.degrees(self._theta_max)
        fov_v_real = fov_h_real  # F-Theta 是旋转对称模型

        return fov_h_real, fov_v_real

    def compute_expansion_factor(self) -> float:
        """
        基于真实 FOV 计算视锥扩展因子

        对于圆锥视锥，expansion_factor = θ_real / θ_nominal

        其中：
            θ_nominal = arctan(max(cx, w-1-cx) / fx)  （针孔模型近似）
            θ_real = _theta_max（通过多项式反解获得）

        同时自动计算 _frustum_scale 和 _expansion_factor 缓存，
        供 _check_fov 和 _check_bounds 使用。

        Returns:
            扩展因子
        """
        # 标称半角（针孔模型近似，使用等效焦距）
        tan_h_nominal = max(self.cx, self.width - 1 - self.cx) / self.fx
        tan_v_nominal = max(self.cy, self.height - 1 - self.cy) / self.fy
        theta_nominal = max(np.arctan(tan_h_nominal), np.arctan(tan_v_nominal))

        if theta_nominal < 1e-10:
            self._expansion_factor = 1.0
            if self._frustum_scale is None:
                self._frustum_scale = self._DEFAULT_FRUSTUM_SCALE
            return 1.0

        # 真实半角
        theta_real = self._theta_max
        expansion = theta_real / theta_nominal

        # 缓存扩展因子和动态缩放因子
        self._expansion_factor = expansion
        # 2D延长已补偿视觉连续性，使用固定小缩放即可
        if self._frustum_scale is None:
            self._frustum_scale = self._DEFAULT_FRUSTUM_SCALE

        return expansion

    def project(
        self,
        points_3d: np.ndarray,
        T_to_cam: Optional[np.ndarray] = None,
        pts_in_cam: bool = False,
        preserve_extra: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影点云到图像平面

        Args:
            points_3d: (N, 3+) 3D点云，支持额外维度（intensity, gpstime等）
            T_to_cam: 4x4外参矩阵
            pts_in_cam: 点是否已在相机坐标系
            preserve_extra: 是否保留额外列（depth + 原始额外列）。
                False（默认）：返回 (N, 2) int32，仅 UV，与原始版本完全兼容
                True：返回 (N, 3+) float64，包含 UV + depth + 原始额外列

        Returns:
            preserve_extra=False:
                result: (N, 2) int32，第0列=u, 第1列=v，有效点UV在[0,w-1]，无效点为-1
            preserve_extra=True:
                result: (N, 3+) float64，第0列=u, 第1列=v(int32), 第2列=depth, 第3列+=原始额外列
            valid: (N,) 有效掩码
        """
        # 确保动态参数（expansion_factor, frustum_scale）已初始化
        self._ensure_params_initialized()

        # np 已在模块顶部 import, 无需重新绑定
        points_3d = np.asarray(points_3d)
        original_shape = points_3d.shape
        n_points = original_shape[0]

        xyz = points_3d[:, :3]

        if not pts_in_cam:
            if T_to_cam is None:
                raise ValueError("T_to_cam must be provided when pts_in_cam is False")
            points_cam = self._transform_to_camera(xyz, T_to_cam)
        else:
            points_cam = xyz

        # NumPy 向量化路径 (唯一实现)
        x_c, y_c, z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        theta = np.arctan2(np.sqrt(x_c**2 + y_c**2), z_c)

        # 合并深度+FOV检查为单次 logical_and.reduce，减少中间布尔数组分配
        eff_scale = self._frustum_scale if self._frustum_scale is not None else self._DEFAULT_FRUSTUM_SCALE
        theta_max = float(self._theta_max) * float(eff_scale)
        valid = np.logical_and.reduce([
            z_c > self.near_z,
            z_c < self.far_z,
            theta <= theta_max
        ])

        # 使用 Horner 法则计算多项式，比逐项循环快
        fw_poly_list = [float(c) for c in self.fw_poly]
        r = np.polynomial.polynomial.polyval(theta, fw_poly_list)

        phi = np.arctan2(y_c, x_c)
        u = r * np.cos(phi) + float(self.cx)
        v = r * np.sin(phi) + float(self.cy)

        in_bounds = self._check_bounds(u, v)
        valid = valid & in_bounds

        u_clip = np.clip(u, 0, self.width - 1)
        v_clip = np.clip(v, 0, self.height - 1)
        u_clip[~valid] = -1
        v_clip[~valid] = -1

        if not preserve_extra:
            # 先 stack 再 astype，减少 1 次 astype 调用（2→1）
            result = np.stack([u_clip, v_clip], axis=1).astype(np.int32)
            return result, valid

        result = np.zeros((n_points, max(3, original_shape[1])), dtype=np.float64)
        # np.trunc 替代 astype(int32) 避免 int32 中间临时数组
        result[:, 0] = np.trunc(u_clip)
        result[:, 1] = np.trunc(v_clip)
        result[:, 2] = z_c
        if original_shape[1] > 3:
            result[:, 3:] = points_3d[:, 3:]

        return result, valid
