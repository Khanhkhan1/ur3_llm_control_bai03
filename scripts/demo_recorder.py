#!/usr/bin/env python3
"""Records the demo video straight from the simulation: Gazebo side view | annotated
overhead camera (what perception sees), with task_runner's output underneath.

    demo_recorder.py --ros-args -p output:=/path/demo.mp4
Stop with Ctrl-C; the file is finalized on exit.
"""
import signal
import textwrap

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

W, H = 640, 480
LOG_LINES = 14
LOG_H = 20 * LOG_LINES + 16
TITLE_H = 34


def to_bgr(msg: Image):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (W, H)) if (msg.width, msg.height) != (W, H) else img.copy()


class DemoRecorder(Node):
    def __init__(self):
        super().__init__("demo_recorder")
        self.declare_parameter("output", "demo.mp4")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("title", "UR3 + Robotiq 2F-85 | LLM skill planning with camera")
        self.output = self.get_parameter("output").value
        fps = float(self.get_parameter("fps").value)
        self.title = self.get_parameter("title").value
        self.side = np.zeros((H, W, 3), np.uint8)
        self.top = np.zeros((H, W, 3), np.uint8)
        self.lines = []
        self.create_subscription(Image, "video_camera/image", self._on_side, 2)
        self.create_subscription(Image, "perception/debug_image", self._on_top, 2)
        self.create_subscription(String, "task_runner/log", self._on_log, 100)
        size = (2 * W, TITLE_H + H + LOG_H)
        self.writer = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.frames = 0
        self.create_timer(1.0 / fps, self._write_frame)
        self.get_logger().info(f"recording {size[0]}x{size[1]} @ {fps} fps to {self.output}")

    def _on_side(self, msg):
        self.side = to_bgr(msg)

    def _on_top(self, msg):
        self.top = to_bgr(msg)

    def _on_log(self, msg):
        for line in textwrap.wrap(msg.data, 120) or [""]:
            self.lines.append(line)
        self.lines = self.lines[-LOG_LINES:]

    def _write_frame(self):
        frame = np.full((TITLE_H + H + LOG_H, 2 * W, 3), 30, np.uint8)
        cv2.putText(frame, self.title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
        frame[TITLE_H:TITLE_H + H, :W] = self.side
        frame[TITLE_H:TITLE_H + H, W:] = self.top
        cv2.putText(frame, "Gazebo (side view)", (10, TITLE_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y = TITLE_H + H + 22
        for line in self.lines:
            color = (235, 235, 235)
            if "SUCCESS" in line or line.endswith(" OK") or "FREE" in line:
                color = (120, 230, 120)
            if "FAIL" in line or "REJECTED" in line or "OCCUPIED" in line or "NO (" in line:
                color = (110, 140, 255)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y += 20
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        self.get_logger().info(f"saved {self.frames} frames to {self.output}")


def main():
    rclpy.init()
    node = DemoRecorder()
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while rclpy.ok() and not stop["flag"]:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
