"""
Conventions - 入口适配工具模块

提供坐标系、外参方向、四元数顺序、内参分辨率缩放等显式适配函数。
本模块为纯函数工具，不改动核心投影数学，所有函数均在调用方入口完成对齐，
核心库内部统一采用 OpenCV 约定（x 右 / y 下 / z 前），不做任何自动猜测/翻轴。

坐标系契约（详见 README "Coordinate System Conventions"）：
    - 相机系：OpenCV 约定（x 右 / y 下 / z 前），原点为光心
    - 像素原点：图像左上角，u 向右增长，v 向下增长
    - 外参 T_to_cam：world→camera 的 4×4 齐次矩阵（p_cam = T @ p_world）
    - 单位：米
"""

import numpy as np
from typing import Optional, Union, Tuple, Sequence

__all__ = [
    'build_transform',
    'invert_transform',
    'scale_intrinsics',
    'check_transform_sanity',
    'AXIS_OPENGL_TO_OPENCV',
    'AXIS_OPENCV_TO_OPENGL',
    'AXIS_ROS_REP103_TO_OPENCV_XFYFZU',
]


# ---------------------------------------------------------------------------
# 预设轴变换矩阵（4×4 齐次，仅含对角元 + 1）
# 用法：T_aligned = AXIS_xxx @ T_original  （左乘，对 R 的列/行约定无关）
# ---------------------------------------------------------------------------

def _diag34(a: float, b: float, c: float) -> np.ndarray:
    """构造 diag(a, b, c, 1) 的 4×4 矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = a
    T[1, 1] = b
    T[2, 2] = c
    return T


# OpenGL/Blender: x 右 / y 上 / z 后 → OpenCV: x 右 / y 下 / z 前
# 对 y、z 取反，即 diag(1, -1, -1, 1)
AXIS_OPENGL_TO_OPENCV: np.ndarray = _diag34(1.0, -1.0, -1.0)
# 反向变换与正向相同（diag(1,-1,-1) 的逆等于自身）
AXIS_OPENCV_TO_OPENGL: np.ndarray = AXIS_OPENGL_TO_OPENCV

# ROS REP-103 车体/LiDAR 系：x 前 / y 左 / z 上
# 该矩阵将其映射到 OpenCV 相机系（x 右 / y 下 / z 前）的常用约定：
#   x_ros(前) → z_cam(前)；y_ros(左) → -x_cam(右)；z_ros(上) → -y_cam(下)
# 注意：此为 ROS 车体→OpenCV 相机的轴重排，不是简单翻转。
# 若 LiDAR extrinsic 已经是 w2c 形式，通常不需要此矩阵，应直接用 build_transform。
# 仅在需要把"以 ROS 车体系表达的点"重映射到"OpenCV 相机轴向"时使用。
AXIS_ROS_REP103_TO_OPENCV_XFYFZU: np.ndarray = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 外参构造与互转
# ---------------------------------------------------------------------------

def build_transform(
    R: Optional[np.ndarray] = None,
    t: Optional[Sequence[float]] = None,
    quat: Optional[Sequence[float]] = None,
    quat_order: str = 'wxyz',
    direction: str = 'w2c',
) -> np.ndarray:
    """
    由旋转 + 平移构造 4×4 齐次变换矩阵。

    Args:
        R: 3×3 旋转矩阵。若提供则忽略 quat。
        t: 长度 3 的平移向量。默认 [0, 0, 0]。
        quat: 长度 4 的四元数。仅当 R=None 时使用。
            quat_order='wxyz'（默认）: [w, x, y, z]
            quat_order='xyzw'（ROS/eigen 常用）: [x, y, z, w]
        direction: 变换方向的显式声明
            'w2c': world→camera，p_cam = T @ p_world（本库默认契约）
            'c2w': camera→world，p_world = T @ p_cam（SLAM/NeRF 常见输出）
            无论传入哪个方向，返回的总是所声明方向对应的 4×4 矩阵；
            调用方负责用 invert_transform 在需要时转换。

    Returns:
        4×4 齐次变换矩阵（float64）

    Raises:
        ValueError: R 和 quat 同时为 None，或维度不合法、四元数阶错误等

    Note:
        本函数不做 w2c↔c2w 的自动翻转，direction 仅作为"语义声明 + 校验"。
        若你拿到的是 c2w 而核心需要 w2c，请调用：
            T_w2c = invert_transform(build_transform(..., direction='c2w'))
    """
    if R is None and quat is None:
        raise ValueError("必须提供 R 或 quat 之一")

    # 构造旋转部分
    if R is not None:
        R_arr = np.asarray(R, dtype=np.float64)
        if R_arr.shape != (3, 3):
            raise ValueError(f"R 必须为 (3,3)，得到 {R_arr.shape}")
    else:
        q = np.asarray(quat, dtype=np.float64).ravel()
        if q.size != 4:
            raise ValueError(f"quat 必须长度为 4，得到 {q.size}")
        quat_order = quat_order.lower()
        if quat_order == 'wxyz':
            w, x, y, z = q
        elif quat_order == 'xyzw':
            x, y, z, w = q
        else:
            raise ValueError(f"quat_order 必须为 'wxyz' 或 'xyzw'，得到 {quat_order}")
        R_arr = _quat_to_rotmat(w, x, y, z)

    # 构造平移部分
    if t is None:
        t_arr = np.zeros(3, dtype=np.float64)
    else:
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        if t_arr.size != 3:
            raise ValueError(f"t 必须长度为 3，得到 {t_arr.size}")

    # 组装 4×4
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_arr
    T[:3, 3] = t_arr

    # direction 仅做语义校验，不做隐式翻转
    direction = direction.lower()
    if direction not in ('w2c', 'c2w'):
        raise ValueError(f"direction 必须为 'w2c' 或 'c2w'，得到 {direction}")

    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    求齐次变换矩阵的逆（c2w ↔ w2c 互转）。

    对刚体变换 T = [R | t; 0 1]，其逆为 [R^T | -R^T·t; 0 1]，
    比通用 4×4 求逆更稳定、更快。

    Args:
        T: 4×4 齐次变换矩阵

    Returns:
        4×4 逆变换矩阵

    Raises:
        ValueError: T 不是 4×4
    """
    T_arr = np.asarray(T, dtype=np.float64)
    if T_arr.shape != (4, 4):
        raise ValueError(f"T 必须为 (4,4)，得到 {T_arr.shape}")
    R = T_arr[:3, :3]
    t = T_arr[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


# ---------------------------------------------------------------------------
# 内参分辨率缩放
# ---------------------------------------------------------------------------

def scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    当标定分辨率与实际使用分辨率不一致时，按比例缩放内参。

    所有内参都是像素量纲，与分辨率线性相关：
        scale_x = dst_w / src_w,  scale_y = dst_h / src_h
        fx' = fx * scale_x,  cx' = (cx + 0.5) * scale_x - 0.5
        fy' = fy * scale_y,  cy' = (cy + 0.5) * scale_y - 0.5

    采用 "+0.5 / -0.5" 像素中心对齐约定，避免主点偏移半像素误差。

    Args:
        fx, fy, cx, cy: 源分辨率下的内参
        src_size: (width, height) 源分辨率
        dst_size: (width, height) 目标分辨率

    Returns:
        (fx', fy', cx', cy') 缩放后的内参

    Raises:
        ValueError: 分辨率非正或维度不匹配
    """
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"分辨率必须为正，src={src_size}, dst={dst_size}")
    sx = dst_w / src_w
    sy = dst_h / src_h
    fx2 = float(fx) * sx
    fy2 = float(fy) * sy
    cx2 = (float(cx) + 0.5) * sx - 0.5
    cy2 = (float(cy) + 0.5) * sy - 0.5
    return fx2, fy2, cx2, cy2


# ---------------------------------------------------------------------------
# 外参合理性检查（静默错误诊断）
# ---------------------------------------------------------------------------

def check_transform_sanity(
    T: np.ndarray,
    direction: str = 'w2c',
    tol_det: float = 1e-6,
) -> None:
    """
    对变换矩阵做基础合理性检查，发现问题发 warning（不抛异常）。

    检查项：
        - R 是否正交（R @ R.T ≈ I）
        - det(R) 是否接近 1（接近 -1 提示存在轴镜像，常见于手写轴变换错误）
        - t 是否为有限值

    Args:
        T: 4×4 齐次变换矩阵
        direction: 'w2c' 或 'c2w'，仅用于 warning 文本提示
        tol_det: det(R) 偏离 1 的容差
    """
    import warnings
    T_arr = np.asarray(T, dtype=np.float64)
    if T_arr.shape != (4, 4):
        raise ValueError(f"T 必须为 (4,4)，得到 {T_arr.shape}")
    R = T_arr[:3, :3]
    t = T_arr[:3, 3]
    issues = []

    # 正交性
    orth_err = np.linalg.norm(R @ R.T - np.eye(3))
    if orth_err > 1e-3:
        issues.append(f"R 非正交（||R@R^T - I||={orth_err:.2e}），可能旋转矩阵构造错误")

    # 行列式
    det = np.linalg.det(R)
    if abs(abs(det) - 1.0) > tol_det:
        issues.append(f"det(R)={det:.6f}，偏离 1；若接近 -1 提示存在轴镜像")
    if det < 0:
        issues.append(f"det(R)={det:.6f} < 0，旋转含镜像反射，常见于手写轴变换错误")

    # 有限性
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        issues.append("T 含 NaN/Inf，数据异常")

    if issues:
        msg = f"[{direction}] 变换矩阵可能存在问题: " + "; ".join(issues)
        warnings.warn(msg, stacklevel=2)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """
    单位四元数 → 3×3 旋转矩阵（右手系，被动旋转/坐标系旋转约定）。

    与 OpenCV、ROS、eigen 的 quat→R 主流约定一致。
    不强制归一化输入，但会内部归一以避免数值漂移。
    """
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("四元数模长为 0，无法转换为旋转矩阵")
    w, x, y, z = w / n, x / n, y / n, z / n
    # 标准四元数→旋转矩阵公式
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
