#!/usr/bin/env python3
"""The task-level picture of the cell, built from the overhead camera: where each cube is,
which zone holds what, what the gripper is holding, and where a cube can be put aside.
Plain Python (no ROS) so the planner/validator logic can be tested on its own."""
import math

TEMPORARY_POSITION = "temporary_position"


class WorldModel:
    def __init__(self, workspace: dict):
        self.zones = {n: (float(z["x"]), float(z["y"])) for n, z in workspace["zones"].items()}
        self.zone_half = float(workspace["zone_half_size"])
        self.free_space = workspace["free_space"]
        self.known_objects = list(workspace["objects"])
        self.objects = {}  # name -> (x, y, yaw), last seen by the camera (or just placed)
        self.held = None
        self.temporary_position = None  # (x, y) chosen by find_free_position()

    # ------------------------------------------------------------------ camera updates

    def update_from_detection(self, names, xs, ys, yaws):
        """Merge a camera detection. Objects not seen this time keep their last known pose
        (the arm can hide a cube); the held object is never taken from the image."""
        for name, x, y, yaw in zip(names, xs, ys, yaws):
            if name == self.held:
                continue
            self.objects[name] = (float(x), float(y), float(yaw))

    # ------------------------------------------------------------------ queries

    def zone_of(self, x, y):
        for zone, (zx, zy) in self.zones.items():
            if abs(x - zx) <= self.zone_half and abs(y - zy) <= self.zone_half:
                return zone
        return None

    def zone_occupant(self, zone):
        """Name of the object in `zone`, or None when it is free."""
        for name, (x, y, _) in sorted(self.objects.items()):
            if self.zone_of(x, y) == zone:
                return name
        return None

    def location_of(self, name):
        if name == self.held:
            return "gripper"
        if name not in self.objects:
            return None
        x, y, _ = self.objects[name]
        return self.zone_of(x, y) or "table"

    def find_free_position(self):
        """A spot on the table far enough from every cube and every zone center for the
        open fingers to reach around a cube there (and around its neighbours later), and
        far enough from the robot base. Among those, the one closest to the middle of the
        search area."""
        fs = self.free_space
        clearance = float(fs["clearance"])
        obstacles = [(x, y) for (x, y, _) in self.objects.values()] + list(self.zones.values())
        cx = (fs["x_min"] + fs["x_max"]) / 2.0
        cy = (fs["y_min"] + fs["y_max"]) / 2.0
        step = float(fs["step"])
        best, best_cost = None, math.inf
        nx = int(round((fs["x_max"] - fs["x_min"]) / step)) + 1
        ny = int(round((fs["y_max"] - fs["y_min"]) / step)) + 1
        for i in range(nx):
            x = fs["x_min"] + i * step
            for j in range(ny):
                y = fs["y_min"] + j * step
                if math.hypot(x, y) < float(fs["min_radius"]):
                    continue
                if min(math.hypot(x - ox, y - oy) for ox, oy in obstacles) < clearance:
                    continue
                cost = math.hypot(x - cx, y - cy)
                if cost < best_cost - 1e-9:
                    best, best_cost = (round(x, 3), round(y, 3)), cost
        return best

    # ------------------------------------------------------------------ for the LLM

    def summary(self) -> str:
        lines = ["Objects (seen by the overhead camera):"]
        for name in sorted(self.objects):
            loc = self.location_of(name)
            x, y, _ = self.objects[name]
            where = f"in {loc}" if loc not in ("table", None) else "on the table"
            lines.append(f"- {name}: {where} (x={x:.3f}, y={y:.3f})")
        if self.held:
            lines.append(f"- {self.held}: held in the gripper")
        lines.append("Zones:")
        for zone in sorted(self.zones):
            occ = self.zone_occupant(zone)
            lines.append(f"- {zone}: " + (f"occupied by {occ}" if occ else "free"))
        lines.append("Gripper: " + (f"holding {self.held}" if self.held else "empty"))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Symbolic state for the plan validator: object -> location, plus the held object."""
        return {
            "locations": {n: self.location_of(n) for n in self.objects},
            "held": self.held,
            "zones": sorted(self.zones),
        }
