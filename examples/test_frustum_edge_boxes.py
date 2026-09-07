#!/usr/bin/env python3
"""
Frustum Edge Box Test - Visualize clipping of 3D boxes at frustum edges

在相机视锥边缘构建各种类型的3D Box，包含被视锥面各种截断的情况，
使用 OpenCV 绘制 BEV 视图和图像投影，并拼接显示便于对比。

坐标系约定:
- 相机坐标系 (autoproj): x:right, y:up, z:forward
- 用户坐标系: x:前(forward), y:左(left), z:上(up)
- 坐标转换: cam_x = -user_y, cam_y = user_z, cam_z = user_x
- BEV视图: 显示用户坐标系的 xy 平面 (前-左平面)

Usage:
    python test_frustum_edge_boxes.py
"""

import sys
import math
import numpy as np
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoproj import CameraFactory, Projector, FrustumCuller, FrustumType
from autoproj.geometry import Box3D, LineSet


# ==================== 颜色定义 (BGR) ====================
COLOR_BG       = (30, 30, 30)     # 深灰背景
COLOR_FRUSTUM  = (0, 200, 0)     # 绿色 - 视锥
COLOR_FRUSTUM_FILL = (0, 80, 0)  # 深绿 - 视锥填充
COLOR_BOX_ORIG = (120, 120, 120) # 灰色 - 原始Box
COLOR_BOX_CLIP = (0, 0, 220)    # 红色 - 裁剪后Box
COLOR_CAM      = (255, 255, 255)# 白色 - 相机
COLOR_GRID     = (50, 50, 50)   # 网格
COLOR_TEXT     = (255, 255, 255)# 白色文字
COLOR_TEXT_BG  = (0, 0, 0)      # 文字背景
COLOR_CENTER   = (80, 80, 80)   # 中心线
COLOR_CORNER_OK = (0, 0, 220)   # 有效角点
COLOR_CORNER_NO = (120, 120, 120)# 无效角点


# ==================== 坐标转换 ====================

def user_to_camera(x, y, z):
    """用户坐标(x:前,y:左,z:上) -> 相机坐标(x:右,y:上,z:前)"""
    return np.array([-y, z, x])


def create_test_camera():
    """创建测试相机"""
    camera = CameraFactory.create_pinhole(
        width=1280, height=720,
        fx=500.0, fy=500.0,
        cx=640.0, cy=360.0,
        near_z=0.5,
        far_z=80.0
    )
    camera.margin = 0
    return camera


def create_box_corners(center_user, size_user, yaw=0.0):
    """从用户坐标创建8个角点(相机坐标系)"""
    cx, cy, cz = center_user
    l, w, h = size_user
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    corners_user = np.array([
        [cx-l/2, cy-w/2, cz-h/2], [cx+l/2, cy-w/2, cz-h/2],
        [cx+l/2, cy+w/2, cz-h/2], [cx-l/2, cy+w/2, cz-h/2],
        [cx-l/2, cy-w/2, cz+h/2], [cx+l/2, cy-w/2, cz+h/2],
        [cx+l/2, cy+w/2, cz+h/2], [cx-l/2, cy+w/2, cz+h/2],
    ])

    rotated = np.zeros((8, 3))
    for i in range(8):
        x, y, z = corners_user[i]
        rx = cos_y*(x-cx) - sin_y*(y-cy) + cx
        ry = sin_y*(x-cx) + cos_y*(y-cy) + cy
        rotated[i] = [rx, ry, z]

    corners_cam = np.zeros((8, 3))
    for i in range(8):
        corners_cam[i] = user_to_camera(*rotated[i])
    return corners_cam


def get_test_box_configs():
    """获取测试Box配置列表

    相机参数: fx=500, fy=500, cx=640, cy=360
    tan_h = 640/500 = 1.28, tan_v = 360/500 = 0.72
    near_z=0.5m, far_z=80m

    At z=10m: 水平范围 [-12.8, 12.8], 垂直范围 [-7.2, 7.2]
    """
    return [
        ("Fully Inside",     (10, 0, 0),     (3, 3, 3),  0.0),
        ("Clipped Left",     (10, 13, 0),    (4, 4, 4),  0.0),
        ("Clipped Right",    (10, -13, 0),   (4, 4, 4),  0.0),
        ("Clipped Top",      (10, 0, 8),     (4, 4, 4),  0.0),
        ("Clipped Bottom",   (10, 0, -8),    (4, 4, 4),  0.0),
        ("Clipped Near",     (0.5, 0, 0),    (3, 3, 3),  0.0),
        # Far: 视锥无远截止面，远处目标只要在FOV角度内就完整投影
        ("Far (No Clip)",    (79, 0, 0),     (3, 3, 3),  0.0),
        ("Very Far (NoClip)",(200, 0, 0),    (3, 3, 3),  0.0),
        # Clipped L+Near: 跨过near和left两个平面，部分在内
        ("Clipped L+Near",   (2, 4, 0),      (3, 3, 3),  0.0),
        ("Clipped R+Top",    (10, -13, 8),   (4, 4, 4),  0.0),
        # Far+Top: 远处但顶部超出FOV → 只裁剪Top，不裁剪Far
        ("Far+Top Clip",     (79, 0, 8),     (3, 3, 3),  0.0),
        ("Large Multi-Clip", (10, 0, 0),     (25, 25, 25), 0.0),
        ("Rotated Box",      (10, 12, 6),    (4, 3, 3),  np.pi/6),
    ]


# ==================== OpenCV 绘图工具 ====================

def put_text(img, text, org, font_scale=0.5, color=COLOR_TEXT, thickness=1, bg=True):
    """在图像上绘制带背景的文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    if bg:
        cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+baseline+2), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8):
    """绘制虚线"""
    pt1 = np.array(pt1, dtype=np.float64)
    pt2 = np.array(pt2, dtype=np.float64)
    diff = pt2 - pt1
    length = np.sqrt(np.sum(diff**2))
    if length < 1:
        return
    n = int(length / dash_length)
    for i in range(0, n, 2):
        s = i / n
        e = min((i + 1) / n, 1.0)
        p1 = (pt1 + diff * s).astype(int)
        p2 = (pt1 + diff * e).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)


def draw_dashed_polyline(img, pts, color, thickness=1, dash_length=8):
    """绘制虚线多段线"""
    for i in range(len(pts) - 1):
        draw_dashed_line(img, pts[i], pts[i+1], color, thickness, dash_length)


# ==================== BEV 绘图 ====================

class BEVCanvas:
    """BEV画布: 用户坐标系, 原点在相机位置

    支持两种视图模式:
    - 'top_down': 纯俯视图 (forward→右, left→上), 忽略高度 z
    - 'oblique':  斜后上方视角 (默认45°), forward→右下, left→左, up→上
                  可显示3D物体的高度信息

    坐标映射:
    - top_down: px = margin + x_fwd*scale,  py = height/2 - y_left*scale
    - oblique:  px = origin_x + x_fwd*scale*sin(α) - y_left*scale
                py = origin_y + x_fwd*scale*cos(α) - z_up*scale*sin(α)
                其中 α 为俯角 (view_angle)
    """

    def __init__(self, width=800, height=800, scale=8.0, margin=40,
                 view_mode='oblique', view_angle=45.0):
        """
        Args:
            width, height: 画布尺寸(像素)
            scale: 每米对应的像素数
            margin: 画布边缘留白(像素)
            view_mode: 'oblique' (斜后上方视角) 或 'top_down' (纯俯视)
            view_angle: 斜视角的俯角(度), 仅 oblique 模式生效

        Oblique 投影公式 (相机朝前水平):
            px = origin_x + x_fwd * scale - y_left * scale * sin(α)
            py = origin_y - y_left * scale * cos(α) - z_up * scale + x_fwd * scale * sin(α) * tilt

        x_fwd (前方): 主要映射到屏幕右方 (水平), 轻微下倾产生立体感
        y_left (左方): 映射到屏幕左上方 (对角), 产生3D深度
        z_up (上方): 映射到屏幕上方 (垂直)
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.margin = margin
        self.view_mode = view_mode
        self.view_angle = view_angle

        if view_mode == 'oblique':
            alpha = math.radians(view_angle)
            self._cos_a = math.cos(alpha)
            self._sin_a = math.sin(alpha)
            self._tilt = 0.3  # 前方下倾系数 (0=纯水平, 0.5=明显下倾)
            # 原点: 左侧留出 left 空间, 下方留出 forward 空间
            self.origin_x = int(width * 0.30)
            self.origin_y = int(height * 0.55)
        else:
            self._cos_a = 0.0
            self._sin_a = 0.0
            self._tilt = 0.0
            self.origin_x = margin
            self.origin_y = height // 2

        self.img = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    def world_to_pixel(self, x_fwd, y_left, z_up=0.0):
        """世界坐标(米) -> 画布像素坐标

        Args:
            x_fwd: 前方距离(米)
            y_left: 左方距离(米)
            z_up: 上方距离(米), top_down 模式忽略
        """
        if self.view_mode == 'oblique':
            px = int(self.origin_x + x_fwd * self.scale
                     - y_left * self.scale * self._sin_a)
            py = int(self.origin_y - y_left * self.scale * self._cos_a
                     - z_up * self.scale
                     + x_fwd * self.scale * self._sin_a * self._tilt)
        else:
            px = int(self.margin + x_fwd * self.scale)
            py = int(self.height / 2 - y_left * self.scale)
        return (px, py)

    def draw_grid(self, step_m=5):
        """绘制网格 (地面 z=0)"""
        if self.view_mode == 'oblique':
            # 斜视角: forward 方向网格线沿对角线延伸, left 方向近水平线
            max_fwd_px = (self.width - self.origin_x) / self.scale
            max_fwd_py = (self.height - self.origin_y) / (self.scale * self._sin_a * self._tilt)
            max_fwd = min(max_fwd_px, max_fwd_py)

            # 前方网格线 (constant x_fwd): 从 bottom-right 到 top-left 的对角线
            for m in range(0, int(max_fwd) + 1, step_m):
                p1 = self.world_to_pixel(m, -40)
                p2 = self.world_to_pixel(m, 40)
                cv2.line(self.img, p1, p2, COLOR_GRID, 1)
                put_text(self.img, f"{m}m", (p2[0] + 2, p2[1] - 5),
                         font_scale=0.35, color=(100, 100, 100), bg=False)

            # 左方网格线 (constant y_left): 近水平线, 轻微下倾
            for m in range(-40, 41, step_m):
                p1 = self.world_to_pixel(0, m)
                p2 = self.world_to_pixel(int(max_fwd), m)
                cv2.line(self.img, p1, p2, COLOR_GRID, 1)
                if m != 0:
                    put_text(self.img, f"{m}m", (p1[0] - 30, p1[1] - 3),
                             font_scale=0.35, color=(100, 100, 100), bg=False)
        else:
            max_fwd = (self.width - 2 * self.margin) / self.scale
            for m in range(0, int(max_fwd) + 1, step_m):
                px = self.world_to_pixel(m, 0)[0]
                cv2.line(self.img, (px, 0), (px, self.height), COLOR_GRID, 1)
                put_text(self.img, f"{m}m", (px + 2, self.height - 10), font_scale=0.35,
                         color=(100, 100, 100), bg=False)

            half_h = (self.height / 2 - self.margin) / self.scale
            for m in range(-int(half_h), int(half_h) + 1, step_m):
                py = self.world_to_pixel(0, m)[1]
                cv2.line(self.img, (0, py), (self.width, py), COLOR_GRID, 1)
                if m != 0:
                    put_text(self.img, f"{m}m", (2, py - 3), font_scale=0.35,
                             color=(100, 100, 100), bg=False)

    def draw_frustum(self, culler):
        """绘制视锥 (oblique 模式下绘制3D视锥体, top_down 模式绘制地面梯形)"""
        near_z = culler.near_z
        tan_h = culler.tan_h

        if self.view_mode == 'oblique':
            # 3D 视锥: 底面(near)和远截面, 含上下边界
            tan_v = culler.tan_v
            # 计算最远显示距离 (受画布右边界和下边界约束)
            max_fwd_px = (self.width - self.origin_x) / self.scale
            max_fwd_py = (self.height - self.origin_y) / (self.scale * self._sin_a * self._tilt)
            far_z_display = min(max_fwd_px, max_fwd_py)

            # near 面四角 (forward=near_z)
            near_bl = self.world_to_pixel(near_z, -near_z * tan_h, -near_z * tan_v)
            near_br = self.world_to_pixel(near_z,  near_z * tan_h, -near_z * tan_v)
            near_tl = self.world_to_pixel(near_z, -near_z * tan_h,  near_z * tan_v)
            near_tr = self.world_to_pixel(near_z,  near_z * tan_h,  near_z * tan_v)
            # far 面四角 (forward=far_z_display)
            far_bl = self.world_to_pixel(far_z_display, -far_z_display * tan_h, -far_z_display * tan_v)
            far_br = self.world_to_pixel(far_z_display,  far_z_display * tan_h, -far_z_display * tan_v)
            far_tl = self.world_to_pixel(far_z_display, -far_z_display * tan_h,  far_z_display * tan_v)
            far_tr = self.world_to_pixel(far_z_display,  far_z_display * tan_h,  far_z_display * tan_v)

            # 填充视锥体 (用远截面填充)
            overlay = self.img.copy()
            far_pts = np.array([far_bl, far_br, far_tr, far_tl], dtype=np.int32)
            cv2.fillPoly(overlay, [far_pts], COLOR_FRUSTUM_FILL)
            cv2.addWeighted(overlay, 0.2, self.img, 0.8, 0, self.img)

            # 绘制 near 面和 far 面边框
            cv2.polylines(self.img, [np.array([near_bl, near_br, near_tr, near_tl], dtype=np.int32)],
                          True, COLOR_FRUSTUM, 1, cv2.LINE_AA)
            cv2.polylines(self.img, [far_pts], True, COLOR_FRUSTUM, 1, cv2.LINE_AA)

            # 绘制四条棱线 (near 角到 far 角)
            for n_pt, f_pt in [(near_bl, far_bl), (near_br, far_br), (near_tl, far_tl), (near_tr, far_tr)]:
                draw_dashed_line(self.img, n_pt, f_pt, COLOR_FRUSTUM, 1, dash_length=6)

            put_text(self.img, f"near={near_z:.1f}m", near_bl,
                     font_scale=0.35, color=COLOR_FRUSTUM)
            put_text(self.img, f"far~{far_z_display:.0f}m", far_bl,
                     font_scale=0.35, color=COLOR_FRUSTUM)
        else:
            # top_down: 地面梯形
            far_z_display = (self.width - 2 * self.margin) / self.scale
            pts = [
                self.world_to_pixel(near_z, -near_z * tan_h),
                self.world_to_pixel(near_z,  near_z * tan_h),
                self.world_to_pixel(far_z_display,   far_z_display * tan_h),
                self.world_to_pixel(far_z_display,  -far_z_display * tan_h),
            ]
            pts_arr = np.array(pts, dtype=np.int32)
            overlay = self.img.copy()
            cv2.fillPoly(overlay, [pts_arr], COLOR_FRUSTUM_FILL)
            cv2.addWeighted(overlay, 0.3, self.img, 0.7, 0, self.img)
            cv2.polylines(self.img, [pts_arr], True, COLOR_FRUSTUM, 2, cv2.LINE_AA)
            put_text(self.img, f"near={near_z:.1f}m", pts[0],
                     font_scale=0.35, color=COLOR_FRUSTUM)
            put_text(self.img, "(no far plane)", pts[2],
                     font_scale=0.35, color=COLOR_FRUSTUM)

    def draw_camera(self):
        """绘制相机位置标记"""
        pos = self.world_to_pixel(0, 0)
        cv2.drawMarker(self.img, pos, COLOR_CAM, cv2.MARKER_CROSS, 15, 2)
        put_text(self.img, "Cam", (pos[0] + 8, pos[1] - 5), font_scale=0.4,
                 color=COLOR_CAM)

    def draw_box_bev(self, box_corners_cam, culler):
        """在BEV上绘制完整的3D Box 8点框投影

        Box角点编号约定 (与LineSet.BOX_EDGES一致):
          0-3: 底面 (z较低), 1->2->3->0 逆时针
          4-7: 顶面 (z较高), 5->6->7->4 逆时针
          0-4, 1-5, 2-6, 3-7: 垂直棱

        绘制内容:
        - 原始8点框: 底面蓝色虚线, 顶面青色虚线, 垂直棱灰色虚线
        - 裁剪后可见棱: 红色实线
        - 角点: 视锥内红色实心圆, 视锥外灰色空心圆, 标注编号
        - 截断交点: 黄色方块 + 所在平面标注

        Args:
            box_corners_cam: (8, 3) 相机坐标系角点
            culler: 视锥裁剪器
        """
        # 转换角点到用户BEV坐标 (forward, left, up)
        box_bev = np.zeros((8, 3))  # [forward, left, up]
        for i in range(8):
            cam_x, cam_y, cam_z = box_corners_cam[i]
            box_bev[i, 0] = cam_z       # forward
            box_bev[i, 1] = -cam_x     # left
            box_bev[i, 2] = cam_y      # up

        def to_px(bev_pt):
            """BEV世界坐标(含高度) -> 画布像素"""
            return self.world_to_pixel(bev_pt[0], bev_pt[1], bev_pt[2])

        # ---- 1. 绘制原始8点框 (虚线) ----
        # 底面 (0-1-2-3-0): 深橙虚线
        bottom_color = (180, 80, 0)
        for i, j in [(0,1), (1,2), (2,3), (3,0)]:
            p1 = to_px(box_bev[i])
            p2 = to_px(box_bev[j])
            draw_dashed_line(self.img, p1, p2, bottom_color, 1, dash_length=6)

        # 顶面 (4-5-6-7-4): 暗青虚线
        top_color = (140, 140, 0)
        for i, j in [(4,5), (5,6), (6,7), (7,4)]:
            p1 = to_px(box_bev[i])
            p2 = to_px(box_bev[j])
            draw_dashed_line(self.img, p1, p2, top_color, 1, dash_length=6)

        # 垂直棱 (0-4, 1-5, 2-6, 3-7): 灰色虚线
        for i, j in [(0,4), (1,5), (2,6), (3,7)]:
            p1 = to_px(box_bev[i])
            p2 = to_px(box_bev[j])
            draw_dashed_line(self.img, p1, p2, COLOR_BOX_ORIG, 1, dash_length=6)

        # ---- 2. 绘制角点 ----
        for i in range(8):
            p = to_px(box_bev[i])
            inside = culler.is_point_inside(box_corners_cam[i])
            if inside:
                # 视锥内: 红色实心圆
                cv2.circle(self.img, p, 4, (0, 0, 220), -1)
                cv2.circle(self.img, p, 4, (255, 255, 255), 1)
            else:
                # 视锥外: 灰色空心圆
                cv2.circle(self.img, p, 4, (100, 100, 100), 1)
            # 标注角点编号
            label = f"{i}"
            if i < 4:
                # 底面角点标在右下
                offset = (p[0] + 6, p[1] + 12)
            else:
                # 顶面角点标在右上
                offset = (p[0] + 6, p[1] - 6)
            put_text(self.img, label, offset, font_scale=0.35,
                     color=(200, 200, 200))

        # ---- 3. 识别原始角点集合（用于区分"原始端点"与"截断交点"） ----
        orig_cam_corners_set = set()
        for i in range(8):
            c = box_corners_cam[i]
            orig_cam_corners_set.add((round(c[0], 4), round(c[1], 4), round(c[2], 4)))

        def is_original_corner_cam(pt_cam, tol=1e-3):
            """相机坐标系下判断是否是原始8角点之一"""
            key = (round(pt_cam[0], 4), round(pt_cam[1], 4), round(pt_cam[2], 4))
            for (ox, oy, oz) in orig_cam_corners_set:
                if (abs(pt_cam[0]-ox) < max(tol, tol*max(abs(pt_cam[0]), abs(ox))) and
                    abs(pt_cam[1]-oy) < max(tol, tol*max(abs(pt_cam[1]), abs(oy))) and
                    abs(pt_cam[2]-oz) < max(tol, tol*max(abs(pt_cam[2]), abs(oz)))):
                    return True
            return False

        def identify_plane_bev(pt_cam):
            """相机坐标 → 在BEV上识别交点所在平面（Near/Left/Right）"""
            z = pt_cam[2]
            tan_h = culler.tan_h
            near_z = culler.near_z
            xn = pt_cam[0] / z if z > 1e-6 else 0
            planes = []
            if abs(z - near_z) / max(near_z, 1e-6) < 0.02:
                planes.append("Near")
            if abs(xn + tan_h) / tan_h < 0.02:
                planes.append("Left")
            if abs(xn - tan_h) / tan_h < 0.02:
                planes.append("Right")
            return "+".join(planes) if planes else ""

        # ---- 4. 绘制裁剪后可见棱 (红色实线) + 标记截断交点 ----
        COLOR_CLIP_PT = (0, 220, 255)  # 黄
        clip_annotations = []
        seen_clip_px = set()

        for i, j in LineSet.BOX_EDGES:
            p1_cam, p2_cam = box_corners_cam[i], box_corners_cam[j]
            clipped = culler.clip_line(p1_cam, p2_cam)
            if clipped is not None:
                cp1, cp2 = clipped
                pt1 = to_px([cp1[2], -cp1[0], cp1[1]])
                pt2 = to_px([cp2[2], -cp2[0], cp2[1]])
                cv2.line(self.img, pt1, pt2, COLOR_BOX_CLIP, 2, cv2.LINE_AA)

                # 标记端点
                for (pt_px, pt_cam) in [(pt1, cp1), (pt2, cp2)]:
                    if is_original_corner_cam(pt_cam):
                        # 原始角点：上面已经画过了，这里画一个红边
                        cv2.circle(self.img, pt_px, 3, COLOR_BOX_CLIP, -1)
                    else:
                        # 截断交点：黄色方块 + 收集标注
                        s = 4
                        cv2.rectangle(self.img,
                                      (pt_px[0]-s, pt_px[1]-s),
                                      (pt_px[0]+s, pt_px[1]+s),
                                      COLOR_CLIP_PT, -1)
                        cv2.rectangle(self.img,
                                      (pt_px[0]-s, pt_px[1]-s),
                                      (pt_px[0]+s, pt_px[1]+s),
                                      (0, 0, 0), 1)
                        # 去重并收集
                        key = (pt_px[0] // 3, pt_px[1] // 3)
                        if key not in seen_clip_px:
                            seen_clip_px.add(key)
                            plane_name = identify_plane_bev(pt_cam)
                            if plane_name:
                                clip_annotations.append((pt_px, plane_name))

        # 绘制截断交点的平面标注
        for (px, py), plane_name in clip_annotations:
            # 放在点的右上方
            lx, ly = px + 6, py - 6
            if lx > self.width - 70:
                lx = px - 70
            if ly < 15:
                ly = py + 15
            put_text(self.img, plane_name, (lx, ly), font_scale=0.35,
                     color=COLOR_CLIP_PT)

    def draw_title(self, title, subtitle=""):
        """在画布顶部绘制标题"""
        put_text(self.img, title, (10, 22), font_scale=0.6,
                 color=COLOR_TEXT, thickness=1)
        if subtitle:
            put_text(self.img, subtitle, (10, 42), font_scale=0.4,
                     color=(180, 180, 180))

    def draw_info(self, info_lines):
        """在右下角绘制信息"""
        y = self.height - 10 - len(info_lines) * 18
        for line in info_lines:
            put_text(self.img, line, (10, y), font_scale=0.4,
                     color=(200, 200, 200))
            y += 18

    def get_image(self):
        return self.img.copy()


# ==================== 图像投影绘图 ====================

def draw_image_projection(camera, box_corners_cam, projector, culler, title=""):
    """在黑屏图像上绘制完整的3D Box 8点框投影

    绘制内容:
    - 原始8点框: 底面橙色虚线, 顶面青色虚线, 垂直棱灰色虚线
    - 裁剪后可见棱: 红色实线
    - 原始角点: 视锥内红色实心圆, 视锥外灰色空心圆, 标注编号
    - 截断交点(线/视锥面交点): 黄色空心方块 + 所在平面标注

    Args:
        camera: 相机对象
        box_corners_cam: (8, 3) 相机坐标系角点
        projector: 投影器
        culler: 视锥裁剪器（用于识别交点所在平面）
        title: 标题

    Returns:
        img: (H, W, 3) BGR图像
    """
    img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    # 绘制图像边界
    cv2.rectangle(img, (0, 0), (camera.width - 1, camera.height - 1), (60, 60, 60), 1)

    # 绘制 FOV 边框（几何 FOV，用绿色半透明矩形表示理论视锥投影区域）
    # 实际上这就是图像本身的边界，但我们强调它
    cv2.rectangle(img, (0, 0), (camera.width - 1, camera.height - 1), (0, 120, 0), 2)

    # 绘制中心十字线
    cx, cy = int(camera.cx), int(camera.cy)
    cv2.line(img, (cx, 0), (cx, camera.height), (40, 40, 40), 1)
    cv2.line(img, (0, cy), (camera.width, cy), (40, 40, 40), 1)

    # 投影所有8个角点（获取原始像素，便于绘制虚线）
    pixels, valid = camera.project(box_corners_cam, pts_in_cam=True)

    # ---- 1. 绘制原始8点框 (虚线, 仅绘制像素在图像内的端点，允许画到边界外) ----
    bottom_color = (180, 80, 0)   # 深橙 - 底面
    top_color = (140, 140, 0)    # 暗青 - 顶面

    def to_int_pt(p):
        """float像素→int，允许负值/超界用于 OpenCV 自动裁剪"""
        return (int(round(p[0])), int(round(p[1])))

    # 为了绘制原始8点框的虚线（即使一端在FOV外），我们也使用原始投影方法
    # 先获取所有8个角点的原始投影（不经过valid过滤）
    raw_pixels = projector._project_raw_pixels(box_corners_cam)

    # 底面 (0-1-2-3-0)
    for i, j in [(0,1), (1,2), (2,3), (3,0)]:
        if box_corners_cam[i, 2] > 1e-6 and box_corners_cam[j, 2] > 1e-6:
            p1 = to_int_pt(raw_pixels[i])
            p2 = to_int_pt(raw_pixels[j])
            # 区分端点有效性：至少一端在视锥内才画（否则完全在外面意义不大）
            ci = culler.is_point_inside(box_corners_cam[i])
            cj = culler.is_point_inside(box_corners_cam[j])
            if ci or cj:
                draw_dashed_line(img, p1, p2, bottom_color, 1, dash_length=6)

    # 顶面 (4-5-6-7-4)
    for i, j in [(4,5), (5,6), (6,7), (7,4)]:
        if box_corners_cam[i, 2] > 1e-6 and box_corners_cam[j, 2] > 1e-6:
            p1 = to_int_pt(raw_pixels[i])
            p2 = to_int_pt(raw_pixels[j])
            ci = culler.is_point_inside(box_corners_cam[i])
            cj = culler.is_point_inside(box_corners_cam[j])
            if ci or cj:
                draw_dashed_line(img, p1, p2, top_color, 1, dash_length=6)

    # 垂直棱 (0-4, 1-5, 2-6, 3-7)
    for i, j in [(0,4), (1,5), (2,6), (3,7)]:
        if box_corners_cam[i, 2] > 1e-6 and box_corners_cam[j, 2] > 1e-6:
            p1 = to_int_pt(raw_pixels[i])
            p2 = to_int_pt(raw_pixels[j])
            ci = culler.is_point_inside(box_corners_cam[i])
            cj = culler.is_point_inside(box_corners_cam[j])
            if ci or cj:
                draw_dashed_line(img, p1, p2, COLOR_BOX_ORIG, 1, dash_length=6)

    # ---- 2. 绘制裁剪后可见棱 (红色实线) + 收集截断交点 ----
    result = projector.project_box(box_corners_cam, pts_in_cam=True)
    edges = result['edges']  # List[(pixel0_float, pixel1_float)]
    corners_valid = result['valid']

    # 先收集所有原始角点的原始像素（便于识别哪些端点是截断交点）
    corner_raw_px_set = set()
    for i in range(8):
        px_u = round(float(raw_pixels[i, 0]), 2)
        px_v = round(float(raw_pixels[i, 1]), 2)
        corner_raw_px_set.add((px_u, px_v))

    def is_original_corner(pt_float, tol=0.5):
        """判断像素点是否是原始8角点之一（浮点容差比较）"""
        u = round(float(pt_float[0]), 2)
        v = round(float(pt_float[1]), 2)
        for (cu, cv) in corner_raw_px_set:
            if abs(u - cu) < tol and abs(v - cv) < tol:
                return True
        return False

    # 识别截断交点所在的平面
    def identify_intersection_plane(pt_cam):
        """给定相机坐标系的点，判断它位于哪个视锥面（用于标注）"""
        z = pt_cam[2]
        if z <= 1e-6:
            return "?"
        xn = pt_cam[0] / z
        yn = pt_cam[1] / z
        tan_h, tan_v = culler.tan_h, culler.tan_v
        near_z = culler.near_z

        planes = []
        # 检查每个平面（容差 2%，无远平面）
        if abs(z - near_z) / max(near_z, 1e-6) < 0.02:
            planes.append("Near")
        if abs(xn + tan_h) / tan_h < 0.02:
            planes.append("Left")
        if abs(xn - tan_h) / tan_h < 0.02:
            planes.append("Right")
        if abs(yn + tan_v) / tan_v < 0.02:
            planes.append("Bottom")
        if abs(yn - tan_v) / tan_v < 0.02:
            planes.append("Top")
        return "+".join(planes) if planes else "Edge"

    # 需要在相机坐标空间下获取每条裁剪边的端点，用于平面识别
    clipped_edges_with_cam = []
    for i, j in LineSet.BOX_EDGES:
        clipped = culler.clip_line(box_corners_cam[i], box_corners_cam[j])
        if clipped is not None:
            cp1, cp2 = clipped
            if cp1[2] > 1e-6 and cp2[2] > 1e-6:
                clipped_edges_with_cam.append((cp1, cp2, i, j))

    # 绘制棱线（红色实线）
    COLOR_EDGE = (0, 0, 255)       # BGR 红 - 截断棱线
    COLOR_ORIG_ENDPOINT = (0, 80, 255)  # BGR 橙红 - 保留的原始角点
    COLOR_CLIP_POINT = (0, 220, 255)    # BGR 黄 - 视锥面交点
    COLOR_CLIP_POINT_BG = (0, 0, 0)     # 标注背景

    # 先绘制所有边
    for p1_px, p2_px in edges:
        pt1 = to_int_pt(p1_px)
        pt2 = to_int_pt(p2_px)
        # OpenCV 会自动裁剪超出画布的线段，所以即使端点在图像外也能画到边界
        cv2.line(img, pt1, pt2, COLOR_EDGE, 2, cv2.LINE_AA)

    # 再标记端点类型并收集标注
    clip_point_annotations = []  # (pixel_xy, plane_name)
    original_endpoint_count = 0
    clip_point_count = 0

    for idx_edge, (p1_px, p2_px) in enumerate(edges):
        for end_idx, (end_px, end_cam) in enumerate([
            (p1_px, clipped_edges_with_cam[idx_edge][0] if idx_edge < len(clipped_edges_with_cam) else None),
            (p2_px, clipped_edges_with_cam[idx_edge][1] if idx_edge < len(clipped_edges_with_cam) else None)
        ]):
            pt_px = to_int_pt(end_px)
            # 检查是否是原始角点
            if is_original_corner(end_px):
                original_endpoint_count += 1
                # 不重复绘制（原始角点会在下面统一绘制）
            else:
                clip_point_count += 1
                # 截断交点：黄色方块标记
                s = 4
                cv2.rectangle(img, (pt_px[0]-s, pt_px[1]-s),
                                     (pt_px[0]+s, pt_px[1]+s),
                              COLOR_CLIP_POINT, -1)
                cv2.rectangle(img, (pt_px[0]-s, pt_px[1]-s),
                                     (pt_px[0]+s, pt_px[1]+s),
                              (0, 0, 0), 1)
                # 收集标注
                if end_cam is not None:
                    plane_name = identify_intersection_plane(end_cam)
                    clip_point_annotations.append((pt_px, plane_name))

    # 去重的截断交点标注（避免同一点重复标）
    seen_pts = set()
    unique_annotations = []
    for pt, name in clip_point_annotations:
        key = (pt[0] // 3, pt[1] // 3)
        if key not in seen_pts:
            seen_pts.add(key)
            unique_annotations.append((pt, name))

    # 绘制截断交点的平面标注
    for (px, py), plane_name in unique_annotations:
        # 放在点的右上方（避免与线重叠）
        lx, ly = px + 7, py - 7
        # 但如果太靠近右/上边界，就调整方向
        if lx > camera.width - 80:
            lx = px - 80
        if ly < 15:
            ly = py + 15
        put_text(img, plane_name, (lx, ly), font_scale=0.35,
                 color=COLOR_CLIP_POINT, bg=True, thickness=1)

    # ---- 3. 绘制原始8角点 ----
    for i in range(8):
        if box_corners_cam[i, 2] <= 1e-6:
            continue
        pt_raw = to_int_pt(raw_pixels[i])
        # 是否在画布内（至少部分可见才标记）
        in_canvas = (-50 <= pt_raw[0] < camera.width + 50 and
                     -50 <= pt_raw[1] < camera.height + 50)
        if not in_canvas:
            continue
        inside = culler.is_point_inside(box_corners_cam[i])
        if inside:
            # 视锥内: 红色实心圆 + 白边
            cv2.circle(img, pt_raw, 5, COLOR_CORNER_OK, -1)
            cv2.circle(img, pt_raw, 5, (255, 255, 255), 1)
        else:
            # 视锥外: 灰色空心圆
            cv2.circle(img, pt_raw, 4, COLOR_CORNER_NO, 1)
        # 标注角点编号
        put_text(img, f"C{i}", (pt_raw[0] + 8, pt_raw[1] + 5), font_scale=0.35,
                 color=(220, 220, 220) if inside else (140, 140, 140))

    # 标题
    put_text(img, title, (10, 22), font_scale=0.6, color=COLOR_TEXT)

    # 信息
    n_valid = int(sum(corners_valid))
    n_edges = len(edges)
    info = f"CornersInFrustum:{n_valid}/8  VisibleEdges:{n_edges}/12  ClipPoints:{clip_point_count}"
    put_text(img, info, (10, camera.height - 15), font_scale=0.45,
             color=(200, 200, 200))

    # 图例
    legend_y = camera.height - 55
    items = [
        (COLOR_EDGE, "Clipped Edge"),
        ((0, 80, 255), "Original Corner (in)"),
        (COLOR_CLIP_POINT, "Frustum Intersection"),
        (COLOR_CORNER_NO, "Original Corner (out)"),
    ]
    for lx, (col, name) in enumerate(items):
        x = 10 + lx * 230
        cv2.line(img, (x, legend_y), (x + 18, legend_y), col, 2, cv2.LINE_AA)
        put_text(img, name, (x + 24, legend_y + 4), font_scale=0.35,
                 color=(180, 180, 180))

    return img


# ==================== 拼接与布局 ====================

def concat_bev_image(bev_img, img_proj, gap=10, view_mode='oblique'):
    """水平拼接BEV和图像投影, 中间留间隔

    Args:
        bev_img: BEV图像
        img_proj: 图像投影
        gap: 间隔像素
        view_mode: 'oblique' 或 'top_down', 用于标签显示

    Returns:
        拼接后的图像
    """
    h1, w1 = bev_img.shape[:2]
    h2, w2 = img_proj.shape[:2]
    max_h = max(h1, h2)

    # 创建空白画布
    result = np.full((max_h, w1 + gap + w2, 3), COLOR_BG, dtype=np.uint8)

    # 垂直居中放置
    y1 = (max_h - h1) // 2
    y2 = (max_h - h2) // 2

    result[y1:y1+h1, :w1] = bev_img
    result[y2:y2+h2, w1+gap:] = img_proj

    # 绘制标签
    if view_mode == 'oblique':
        view_label = "BEV (Oblique 45°: Camera Forward-Horizontal)"
    else:
        view_label = "BEV (Top-Down: Forward-Left)"
    put_text(result, view_label, (5, max_h - 8),
             font_scale=0.35, color=(150, 150, 150))
    put_text(result, "Image Projection", (w1 + gap + 5, max_h - 8),
             font_scale=0.35, color=(150, 150, 150))

    return result


def concat_vertical(images, gap=10):
    """垂直拼接多张图像"""
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    widths = [img.shape[1] for img in images]
    max_w = max(widths)
    total_h = sum(img.shape[0] for img in images) + gap * (len(images) - 1)

    result = np.full((total_h, max_w, 3), COLOR_BG, dtype=np.uint8)
    y = 0
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        x = (max_w - w) // 2
        result[y:y+h, x:x+w] = img
        y += h + gap

    return result


def draw_section_title(img, text):
    """在拼接图中绘制分隔标题"""
    h = 30
    bar = np.full((h, img.shape[1], 3), (50, 50, 50), dtype=np.uint8)
    put_text(bar, text, (10, 20), font_scale=0.5, color=COLOR_TEXT)
    return np.vstack([bar, img])


# ==================== 主函数 ====================

def main():
    print("=" * 80)
    print("Frustum Edge Box Test - OpenCV Visualization")
    print("=" * 80)

    camera = create_test_camera()
    print(f"\nCamera: {camera.width}x{camera.height}, FOV: {camera.fov_h:.1f}x{camera.fov_v:.1f}")

    culler = FrustumCuller.from_camera(camera, expansion_factor=1.0)
    projector = Projector(camera, cull_frustum=True)

    print(f"Culler: tan_h={culler.tan_h:.4f}, tan_v={culler.tan_v:.4f}")
    print(f"  At z=10m: width={2*10*culler.tan_h:.1f}m, height={2*10*culler.tan_v:.1f}m")

    configs = get_test_box_configs()

    # ==================== 1. BEV + 图像投影拼接图 ====================
    print(f"\nGenerating BEV + Image projection for {len(configs)} test boxes...")

    bev_canvas_w = 800
    bev_canvas_h = 800
    bev_scale = 8.0  # pixels per meter

    all_pairs = []

    for idx, (name, center_user, size_user, yaw) in enumerate(configs):
        corners_cam = create_box_corners(center_user, size_user, yaw)

        # 统计信息
        inside_count = sum(1 for c in corners_cam if culler.is_point_inside(c))
        visible_edges = sum(1 for i, j in LineSet.BOX_EDGES
                           if culler.clip_line(corners_cam[i], corners_cam[j]) is not None)

        center_str = f"Center=({center_user[0]:.1f},{center_user[1]:.1f},{center_user[2]:.1f})"
        size_str = f"Size=({size_user[0]:.1f},{size_user[1]:.1f},{size_user[2]:.1f})"

        print(f"  [{idx+1:2d}] {name:20s} inside={inside_count}/8 edges={visible_edges}/12")

        # --- 绘制 BEV (斜后上方45°视角) ---
        bev = BEVCanvas(width=bev_canvas_w, height=bev_canvas_h, scale=bev_scale,
                        view_mode='oblique', view_angle=45.0)
        bev.draw_grid(step_m=5)
        bev.draw_frustum(culler)
        bev.draw_camera()
        bev.draw_box_bev(corners_cam, culler)
        bev.draw_title(f"BEV: {name}", f"{center_str}  {size_str}")
        bev.draw_info([
            f"Inside corners: {inside_count}/8",
            f"Visible edges: {visible_edges}/12",
        ])
        bev_img = bev.get_image()

        # --- 绘制图像投影 ---
        img_proj = draw_image_projection(
            camera, corners_cam, projector, culler,
            title=f"Image: {name}"
        )

        # 拼接
        pair = concat_bev_image(bev_img, img_proj, gap=20, view_mode='oblique')
        all_pairs.append(pair)

    # 垂直拼接所有测试用例
    print(f"\nConcatenating all {len(all_pairs)} test cases...")
    final_img = concat_vertical(all_pairs, gap=15)

    output_path = project_root / 'examples' / 'frustum_edge_combined.png'
    cv2.imwrite(str(output_path), final_img)
    print(f"Combined image saved: {output_path}")
    print(f"  Size: {final_img.shape[1]}x{final_img.shape[0]}")

    # ==================== 2. 分组拼接 (每4个一组) ====================
    print(f"\nGenerating grouped images (4 per group)...")

    group_size = 4
    n_groups = (len(configs) + group_size - 1) // group_size

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, len(configs))
        group_pairs = all_pairs[start:end]

        group_img = concat_vertical(group_pairs, gap=15)
        group_img = draw_section_title(group_img,
            f"Group {g+1}/{n_groups}: Test cases {start+1}-{end}")

        output_group = project_root / 'examples' / f'frustum_edge_group_{g+1}.png'
        cv2.imwrite(str(output_group), group_img)
        print(f"  Group {g+1} saved: {output_group}")

    # ==================== 3. 验证报告 ====================
    print(f"\n{'='*80}")
    print("Validation Summary")
    print(f"{'='*80}")

    for name, center_user, size_user, yaw in configs:
        corners_cam = create_box_corners(center_user, size_user, yaw)
        result = projector.project_box(corners_cam, pts_in_cam=True)
        n_valid = int(sum(result['valid']))
        n_edges = len(result['edges'])

        # 检查各平面外的角点数（注意：视锥无远裁剪面，不检查far）
        plane_info = []
        for plane_name, check_fn in [
            ('near',   lambda c: c[2] < culler.near_z),
            ('left',   lambda c: c[0] < -c[2] * culler.tan_h),
            ('right',  lambda c: c[0] > c[2] * culler.tan_h),
            ('bottom', lambda c: c[1] < -c[2] * culler.tan_v),
            ('top',    lambda c: c[1] > c[2] * culler.tan_v),
        ]:
            outside = sum(1 for c in corners_cam if check_fn(c))
            if outside > 0:
                plane_info.append(f"{plane_name}:{outside}")

        plane_str = ', '.join(plane_info) if plane_info else 'none'
        status = '[OK]' if n_edges > 0 or n_valid > 0 else '[FAIL]'
        print(f"  {status} {name:20s} Corners: {n_valid}/8, Edges: {n_edges}/12, Outside: {plane_str}")

    print(f"\n{'='*80}")
    print("All visualizations generated successfully!")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
