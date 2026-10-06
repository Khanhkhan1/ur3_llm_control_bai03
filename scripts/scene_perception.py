#!/usr/bin/env python3
"""Overhead-camera perception: finds the colored cubes on the table and decides which zones
are free or occupied.

    /overhead_camera/image --HSV color segmentation--> cube blobs (pixels)
        --camera model (intrinsics + calibrated pose)--> cube centers on the table (m)
        --zone layout--> which cube is in which zone

Serves the latest result on /perception/detect_objects, and publishes an annotated image
(/perception/debug_image) and RViz markers (/perception/markers) of what it sees.
Nothing here knows where a cube "should" be: positions only ever come from the image.
"""
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from ur3_llm_control.srv import DetectObjects
from workspace import load_workspace

# HSV ranges (OpenCV: H 0-179, S/V 0-255), one cube per color. Saturation >= 120 keeps the
# grey table, the white/black zone markers and the robot's pale joint caps out.
COLOR_RANGES = {
    "red_cube": [((0, 120, 50), (8, 255, 255)), ((170, 120, 50), (179, 255, 255))],
    "yellow_cube": [((18, 120, 50), (35, 255, 255))],
    "green_cube": [((45, 120, 50), (85, 255, 255))],
    "blue_cube": [((100, 120, 50), (128, 255, 255))],
    "purple_cube": [((130, 120, 50), (160, 255, 255))],
}
# Drawing / marker colors (B, G, R) in 0-255.
DRAW_BGR = {
    "red_cube": (40, 40, 220),
    "yellow_cube": (0, 210, 230),
    "green_cube": (40, 180, 40),
    "blue_cube": (220, 70, 20),
    "purple_cube": (200, 30, 140),
}
# A 5 cm cube is ~40 px across in this camera; anything far smaller is noise or a sliver
# of a cube hidden by the arm.
MIN_AREA_PX = 400
MAX_AREA_PX = 6000


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


class CameraModel:
    """Pinhole model of a Gazebo camera sensor. The sensor looks along +x of its link with
    +z up in the image, i.e. optical (x right, y down, z forward) = link (-y, -z, x)."""

    def __init__(self, cfg: dict):
        self.pos = np.array([cfg["x"], cfg["y"], cfg["z"]], dtype=float)
        self.rot = rpy_to_matrix(cfg["roll"], cfg["pitch"], cfg["yaw"])
        self.hfov = float(cfg["hfov"])
        self.fx = self.fy = self.cx = self.cy = None

    def set_intrinsics(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def ensure_intrinsics(self, width, height):
        if self.fx is None:
            f = (width / 2.0) / math.tan(self.hfov / 2.0)
            self.set_intrinsics(f, f, width / 2.0, height / 2.0)

    def pixel_to_plane(self, u, v, z):
        """Point where the ray through pixel (u, v) meets the horizontal plane at height z."""
        xo, yo = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        d = self.rot @ np.array([1.0, -xo, -yo])
        t = (z - self.pos[2]) / d[2]
        return self.pos + t * d

    def point_to_pixel(self, x, y, z):
        p = self.rot.T @ (np.array([x, y, z]) - self.pos)
        return (self.fx * (-p[1] / p[0]) + self.cx, self.fy * (-p[2] / p[0]) + self.cy)


class ScenePerception(Node):
    def __init__(self):
        super().__init__("scene_perception")
        self.declare_parameter("workspace_file", "")
        self.ws = load_workspace(self.get_parameter("workspace_file").value)
        self.camera = CameraModel(self.ws["camera"])
        self.table = self.ws["table"]
        self.cube_size = float(self.ws["cube_size"])
        # The camera mostly sees the cubes' top faces.
        self.top_z = float(self.table["top_z"]) + self.cube_size
        self.zones = {n: (float(z["x"]), float(z["y"])) for n, z in self.ws["zones"].items()}
        self.zone_half = float(self.ws["zone_half_size"])

        self._lock = threading.Lock()
        self._latest = None  # (receive time, detections dict, zone occupants dict)

        images = ReentrantCallbackGroup()
        self.create_subscription(
            Image, self.ws["camera"]["image_topic"], self._on_image, 2, callback_group=images
        )
        self.create_subscription(
            CameraInfo, self.ws["camera"]["info_topic"], self._on_info, 2, callback_group=images
        )
        self.debug_pub = self.create_publisher(Image, "perception/debug_image", 2)
        self.marker_pub = self.create_publisher(MarkerArray, "perception/markers", 2)
        self.create_service(
            DetectObjects,
            "perception/detect_objects",
            self._on_detect,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info("scene_perception ready: /perception/detect_objects")

    # ------------------------------------------------------------------ callbacks

    def _on_info(self, msg: CameraInfo):
        if msg.k[0] > 0.0:
            self.camera.set_intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_image(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
        self.camera.ensure_intrinsics(msg.width, msg.height)
        detections = self.detect(bgr)
        occupants = self.zone_occupants(detections)
        with self._lock:
            self._latest = (time.monotonic(), detections, occupants)
        self.publish_debug(bgr, detections, occupants, msg.header)
        self.publish_markers(detections, occupants)

    def _on_detect(self, request, response):
        # Use a frame captured after the request, so a cube that was just moved (or an arm
        # that just left the view) is seen where it is now.
        asked = time.monotonic()
        deadline = asked + 3.0
        latest = None
        while time.monotonic() < deadline:
            with self._lock:
                latest = self._latest
            if latest is not None and latest[0] > asked + 0.05:
                break
            time.sleep(0.05)
        if latest is None or latest[0] <= asked:
            response.success = False
            response.message = "no camera image received"
            return response
        _, detections, occupants = latest
        response.success = True
        response.message = f"{len(detections)} objects detected"
        for name, det in sorted(detections.items()):
            response.names.append(name)
            response.x.append(det["x"])
            response.y.append(det["y"])
            response.yaw.append(det["yaw"])
        for zone in sorted(occupants):
            response.zones.append(zone)
            response.zone_occupants.append(occupants[zone])
        return response

    # ------------------------------------------------------------------ vision

    def on_table(self, x, y):
        t = self.table
        return t["x_min"] <= x <= t["x_max"] and t["y_min"] <= y <= t["y_max"]

    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        detections = {}
        for name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = [c for c in contours if MIN_AREA_PX <= cv2.contourArea(c) <= MAX_AREA_PX]
            if not candidates:
                continue
            blob = max(candidates, key=cv2.contourArea)
            m = cv2.moments(blob)
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y, _ = self.camera.pixel_to_plane(u, v, self.top_z)
            if not self.on_table(x, y):
                continue
            # Cube yaw from the blob's minimum-area rectangle: project one rectangle edge
            # onto the table and take its direction (a cube repeats every 90 deg).
            corners = cv2.boxPoints(cv2.minAreaRect(blob))
            p0 = self.camera.pixel_to_plane(*corners[0], self.top_z)
            p1 = self.camera.pixel_to_plane(*corners[1], self.top_z)
            yaw = math.remainder(math.atan2(p1[1] - p0[1], p1[0] - p0[0]), math.pi / 2)
            detections[name] = {
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
                "pixel": (u, v),
                "contour": blob,
            }
        return detections

    def zone_occupants(self, detections):
        occupants = {}
        for zone, (zx, zy) in self.zones.items():
            inside = [
                n
                for n, d in detections.items()
                if abs(d["x"] - zx) <= self.zone_half and abs(d["y"] - zy) <= self.zone_half
            ]
            occupants[zone] = ",".join(sorted(inside))
        return occupants

    # ------------------------------------------------------------------ output

    def publish_debug(self, bgr, detections, occupants, header):
        if self.debug_pub.get_subscription_count() == 0:
            return
        img = bgr.copy()
        h = self.zone_half + 0.005
        z = float(self.table["top_z"])
        for zone, (zx, zy) in self.zones.items():
            corners = [
                self.camera.point_to_pixel(zx + dx, zy + dy, z)
                for dx, dy in ((h, h), (h, -h), (-h, -h), (-h, h))
            ]
            pts = np.array(corners, np.int32)
            occupied = occupants.get(zone, "")
            color = (40, 40, 220) if occupied else (40, 170, 40)
            cv2.polylines(img, [pts], True, color, 2)
            label = f"{zone}: {occupied if occupied else 'free'}"
            u, v = self.camera.point_to_pixel(zx - h, zy + h, z)
            cv2.putText(img, label, (int(u), int(v) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        for name, det in detections.items():
            color = DRAW_BGR.get(name, (255, 255, 255))
            cv2.drawContours(img, [det["contour"]], -1, (255, 255, 255), 2)
            u, v = det["pixel"]
            cv2.circle(img, (int(u), int(v)), 3, (0, 0, 0), -1)
            text = f"{name.replace('_cube', '')} ({det['x']:.2f}, {det['y']:.2f})"
            cv2.putText(img, text, (int(u) - 45, int(v) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (int(u) - 45, int(v) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"overhead camera: {len(detections)} cubes", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        out = Image()
        out.header = header
        out.height, out.width = img.shape[:2]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = img.tobytes()
        self.debug_pub.publish(out)

    def publish_markers(self, detections, occupants):
        if self.marker_pub.get_subscription_count() == 0:
            return
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        top = float(self.table["top_z"])
        for i, (name, det) in enumerate(sorted(detections.items())):
            b, g, r = DRAW_BGR.get(name, (200, 200, 200))
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = stamp
            m.ns = "detected_cubes"
            m.id = i
            # A translucent shell around the cube: the solid, colored box in RViz is the
            # MoveIt collision object skill_server builds from this same detection.
            m.type = Marker.CUBE
            m.pose.position.x, m.pose.position.y = det["x"], det["y"]
            m.pose.position.z = top + self.cube_size / 2.0
            m.pose.orientation.z = math.sin(det["yaw"] / 2.0)
            m.pose.orientation.w = math.cos(det["yaw"] / 2.0)
            m.scale.x = m.scale.y = m.scale.z = self.cube_size + 0.008
            m.color.r, m.color.g, m.color.b, m.color.a = r / 255.0, g / 255.0, b / 255.0, 0.35
            arr.markers.append(m)
        for i, (zone, (zx, zy)) in enumerate(sorted(self.zones.items())):
            occupied = occupants.get(zone, "")
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = stamp
            m.ns = "zones"
            m.id = i
            m.type = Marker.CUBE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = zx, zy, top + 0.001
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * self.zone_half
            m.scale.z = 0.002
            m.color.r, m.color.g, m.color.b, m.color.a = (0.9, 0.2, 0.2, 0.8) if occupied else (0.2, 0.8, 0.2, 0.8)
            arr.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns = "zone_labels"
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = zx - 0.075, zy, top + self.cube_size + 0.04
            t.pose.orientation.w = 1.0
            t.scale.z = 0.025
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            # Two lines rather than "zone: state": RViz draws spaces very wide here.
            t.text = f"{zone}\n{occupied if occupied else 'free'}"
            arr.markers.append(t)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = ScenePerception()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
