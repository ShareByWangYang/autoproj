"""
Geometry - 3D 几何体数据结构模块

提供点云、3D 包围盒、线段集合、多边形等几何体的数据结构定义。
所有几何体均以 NumPy 数组存储，便于与投影/裁剪模块进行向量化运算。

坐标系契约：OpenCV 约定（x 右 / y 下 / z 前），单位为米。
"""

import numpy as np
from typing import Optional, Union, List, Tuple


class PointCloud:
    """
    点云类

    Args:
        points: 点云坐标，形状为(N, 3)
        colors: 点云颜色，形状为(N, 3)（可选）
        intensity: 反射强度，形状为(N,)（可选）
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ):
        self.points = np.asarray(points)
        self.colors = np.asarray(colors) if colors is not None else None
        self.intensity = np.asarray(intensity) if intensity is not None else None

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        if self.colors is not None and self.colors.shape[0] != self.points.shape[0]:
            raise ValueError("colors must have the same number of points")

        if self.intensity is not None and self.intensity.shape[0] != self.points.shape[0]:
            raise ValueError("intensity must have the same number of points")

    @property
    def num_points(self) -> int:
        """点云数量"""
        return self.points.shape[0]

    def transform(self, T: np.ndarray) -> 'PointCloud':
        """
        对点云应用变换

        Args:
            T: 4x4变换矩阵

        Returns:
            变换后的点云
        """
        points_h = np.hstack([self.points, np.ones((self.num_points, 1))])
        transformed = (T @ points_h.T).T[:, :3]

        return PointCloud(transformed, self.colors, self.intensity)

    def filter_by_mask(self, mask: np.ndarray) -> 'PointCloud':
        """
        根据掩码过滤点云

        Args:
            mask: 布尔掩码，形状为(N,)

        Returns:
            过滤后的点云
        """
        mask = np.asarray(mask)
        new_points = self.points[mask]
        new_colors = self.colors[mask] if self.colors is not None else None
        new_intensity = self.intensity[mask] if self.intensity is not None else None

        return PointCloud(new_points, new_colors, new_intensity)


class LineSet:
    """
    3D 线段集合

    用于表示一组线段，每条线段由两个端点定义。
    常用于表示 3D 包围盒的棱线、多边形的边等。

    Attributes:
        BOX_EDGES: 3D包围盒的12条棱的顶点索引对，可直接用于
                   Box3D.get_lines() 或自定义box的棱线定义。
    """

    BOX_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    def __init__(
        self,
        points: np.ndarray,
        lines: Optional[List[Tuple[int, int]]] = None,
        colors: Optional[np.ndarray] = None
    ):
        """
        初始化线段集合

        Args:
            points: (N, 3) 点坐标数组
            lines: 线段索引列表，每项为 (start_idx, end_idx)
            colors: (M, 3) 线段颜色数组
        """
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim == 1:
            self.points = self.points.reshape(1, -1)

        self.lines = lines if lines is not None else []
        self.colors = np.asarray(colors) if colors is not None else None

    @classmethod
    def from_segments(
        cls,
        segments: List[Tuple[np.ndarray, np.ndarray]]
    ) -> 'LineSet':
        """
        从线段列表创建 LineSet

        Args:
            segments: [(point1, point2), ...] 每个线段的两个端点

        Returns:
            LineSet 对象
        """
        points = []
        lines = []
        idx = 0
        for p1, p2 in segments:
            points.append(np.asarray(p1, dtype=np.float64))
            points.append(np.asarray(p2, dtype=np.float64))
            lines.append((idx, idx + 1))
            idx += 2
        return cls(np.array(points), lines)

    def transform(self, T: np.ndarray) -> 'LineSet':
        """对所有端点施加4x4变换矩阵"""
        points_h = np.hstack([self.points, np.ones((len(self.points), 1))])
        points_transformed = (T @ points_h.T).T[:, :3]

        colors = self.colors.copy() if self.colors is not None else None
        return LineSet(points_transformed, self.lines.copy(), colors)

    def get_segment(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定索引的线段两个端点"""
        i, j = self.lines[idx]
        return self.points[i].copy(), self.points[j].copy()

    @property
    def num_segments(self) -> int:
        """线段数量"""
        return len(self.lines)

    def to_segments(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """转换为线段端点列表"""
        return [self.get_segment(i) for i in range(self.num_segments)]


class Box3D:
    """
    3D包围盒类

    Args:
        center: 中心点坐标 (x, y, z)
        size: 尺寸 (length, width, height)
        yaw: 偏航角（弧度）
        rotation: 旋转矩阵 (3,3) 或欧拉角序列，优先级高于yaw
        label: 标签
        score: 置信度
    """

    def __init__(
        self,
        center: Union[list, np.ndarray],
        size: Union[list, np.ndarray],
        yaw: float = 0.0,
        rotation: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        score: Optional[float] = None
    ):
        self.center = np.array(center, dtype=np.float64)
        self.size = np.array(size, dtype=np.float64)
        self.yaw = yaw
        self.label = label
        self.score = score

        if self.center.shape[0] != 3:
            raise ValueError("center must have 3 elements")
        if self.size.shape[0] != 3:
            raise ValueError("size must have 3 elements")

        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=np.float64)
        else:
            c = np.cos(yaw)
            s = np.sin(yaw)
            self.rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    @property
    def corners(self) -> np.ndarray:
        """获取包围盒的8个角点"""
        l, w, h = self.size / 2

        local_corners = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
        ])

        world_corners = (self.rotation @ local_corners.T).T + self.center
        return world_corners

    def get_corners(self) -> np.ndarray:
        """获取8个角点（兼容方法名）"""
        return self.corners

    def get_lines(self) -> LineSet:
        """获取包围盒的12条棱作为LineSet"""
        corners = self.corners
        return LineSet(corners, LineSet.BOX_EDGES)

    @classmethod
    def from_pose(
        cls,
        x: float, y: float, z: float,
        length: float, width: float, height: float,
        yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
        **kwargs
    ) -> 'Box3D':
        """
        从位姿参数创建Box3D

        Args:
            x, y, z: 中心位置
            length, width, height: 盒尺寸
            yaw, pitch, roll: 欧拉角（弧度）
        """
        center = np.array([x, y, z], dtype=np.float64)
        size = np.array([length, width, height], dtype=np.float64)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        rotation = Rz @ Ry @ Rx
        return cls(center, size, yaw=yaw, rotation=rotation, **kwargs)

    def transform(self, T: np.ndarray) -> 'Box3D':
        """施加4x4变换矩阵"""
        R = T[:3, :3]
        t = T[:3, 3]
        center_new = R @ self.center + t
        rotation_new = R @ self.rotation
        return Box3D(
            center_new, self.size.copy(),
            rotation=rotation_new,
            label=self.label, score=self.score
        )

    def to_numpy(self) -> np.ndarray:
        """
        转换为numpy数组（中心点+尺寸+角度）

        返回 7 元组 [x, y, z, l, w, h, yaw]，可用于批量序列化或与
        其他 3D 检测框架（如 OpenPCDet, MMDetection3D）的数据交换。
        """
        return np.concatenate([self.center, self.size, [self.yaw]])

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> 'Box3D':
        """
        从numpy数组创建包围盒

        Args:
            data: 7 元素数组 [x, y, z, l, w, h, yaw]
        """
        if len(data) != 7:
            raise ValueError("data must have 7 elements: [x, y, z, l, w, h, yaw]")
        return cls(data[:3], data[3:6], data[6])


class Polygon3D:
    """
    3D多边形几何

    用于表示任意3D多边形，支持闭合和非闭合两种模式。
    常用于3D可视化中的复杂形状投影。

    Args:
        vertices: (N, 3) 顶点数组
        is_closed: 是否闭合（首尾相连）
        label: 标签
    """

    def __init__(
        self,
        vertices: np.ndarray,
        is_closed: bool = True,
        label: Optional[str] = None
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.ndim == 1:
            self.vertices = self.vertices.reshape(1, -1)

        self.is_closed = is_closed
        self.label = label

    def get_lines(self) -> LineSet:
        """获取多边形的边作为LineSet"""
        n = len(self.vertices)
        if n < 2:
            return LineSet(self.vertices, [])

        lines = []
        for i in range(n - 1):
            lines.append((i, i + 1))

        if self.is_closed and n > 2:
            lines.append((n - 1, 0))

        return LineSet(self.vertices, lines)

    def transform(self, T: np.ndarray) -> 'Polygon3D':
        """施加4x4变换矩阵"""
        vertices_h = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        vertices_transformed = (T @ vertices_h.T).T[:, :3]
        return Polygon3D(vertices_transformed, self.is_closed, self.label)

    @property
    def num_vertices(self) -> int:
        """顶点数量"""
        return len(self.vertices)