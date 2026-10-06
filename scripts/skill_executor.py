#!/usr/bin/env python3
"""Skill Executor: runs the skills a validated plan is made of.

Perception/reasoning skills run here, on top of the camera (scene_perception) and the
WorldModel built from it:
    detect_objects()      refresh the WorldModel from a new camera image
    check_zone(zone)      camera check of one zone (compared with what the plan expects)
    find_free_position()  choose a free spot on the table -> "temporary_position"
Motion skills are forwarded to skill_server (MoveIt 2), with coordinates looked up in the
WorldModel -- i.e. from the camera, never from the LLM:
    pick(object), place(object, target), home()
"""
import rclpy

from ur3_llm_control.srv import DetectObjects, Home, Pick, Place, UpdateScene
from world_model import TEMPORARY_POSITION


class SkillExecutor:
    def __init__(self, node, world, timeout_sec: float = 240.0):
        self._node = node
        self.world = world
        self._timeout = timeout_sec
        self._clients = {
            "home": node.create_client(Home, "skill/home"),
            "pick": node.create_client(Pick, "skill/pick"),
            "place": node.create_client(Place, "skill/place"),
            "update_scene": node.create_client(UpdateScene, "skill/update_scene"),
            "detect": node.create_client(DetectObjects, "perception/detect_objects"),
        }
        for name, cli in self._clients.items():
            if not cli.wait_for_service(timeout_sec=20.0):
                raise RuntimeError(
                    f"Service {cli.srv_name} not available -- are skill_server and "
                    "scene_perception running?"
                )

    def _call(self, name, request):
        future = self._clients[name].call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout)
        return future.result()

    # ------------------------------------------------------------------ perception skills

    def detect_objects(self):
        res = self._call("detect", DetectObjects.Request())
        if res is None or not res.success:
            return "CAMERA_FAILED", False
        self.world.update_from_detection(res.names, res.x, res.y, res.yaw)
        self.sync_scene()
        return f"SUCCESS ({len(res.names)} objects)", True

    def check_zone(self, zone, expected="unchecked"):
        """Camera check of `zone`. `expected` is the occupant the validated plan assumed
        (None = free); a different answer means the scene changed and the plan is stale."""
        status, ok = self.detect_objects()
        if not ok:
            return status, False
        occupant = self.world.zone_occupant(zone)
        status = f"OCCUPIED ({occupant})" if occupant else "FREE"
        if expected != "unchecked" and occupant != expected:
            return status + f" -- plan assumed {expected or 'free'}", False
        return status, True

    def find_free_position(self):
        pos = self.world.find_free_position()
        if pos is None:
            return "NO_FREE_POSITION", False
        self.world.temporary_position = pos
        return f"SUCCESS ({pos[0]:.2f}, {pos[1]:.2f})", True

    # ------------------------------------------------------------------ motion skills

    def sync_scene(self):
        """Give skill_server the cubes as the camera sees them (collision objects)."""
        req = UpdateScene.Request()
        for name, (x, y, yaw) in sorted(self.world.objects.items()):
            req.names.append(name)
            req.x.append(x)
            req.y.append(y)
            req.yaw.append(yaw)
        self._call("update_scene", req)

    def home(self):
        res = self._call("home", Home.Request())
        status = res.status if res else "FAILED"
        return status, status == "SUCCESS"

    def pick(self, obj):
        if obj not in self.world.objects:
            return "OBJECT_NOT_FOUND", False
        x, y, yaw = self.world.objects[obj]
        self.sync_scene()
        req = Pick.Request()
        req.object, req.x, req.y, req.yaw = obj, x, y, yaw
        res = self._call("pick", req)
        status = res.status if res else "FAILED"
        if status == "SUCCESS":
            self.world.held = obj
            del self.world.objects[obj]
        return status, status == "SUCCESS"

    def place(self, obj, target):
        if self.world.held != obj:
            return "NOT_HOLDING", False
        if target in self.world.zones:
            occupant = self.world.zone_occupant(target)
            if occupant:
                return f"ZONE_OCCUPIED ({occupant})", False
            x, y = self.world.zones[target]
        elif target == TEMPORARY_POSITION:
            if self.world.temporary_position is None:
                return "NO_TEMPORARY_POSITION", False
            x, y = self.world.temporary_position
        else:
            return "INVALID_TARGET", False
        req = Place.Request()
        req.object, req.x, req.y = obj, x, y
        res = self._call("place", req)
        status = res.status if res else "FAILED"
        if status == "SUCCESS":
            self.world.held = None
            self.world.objects[obj] = (x, y, 0.0)
            if target == TEMPORARY_POSITION:
                self.world.temporary_position = None
        return status, status == "SUCCESS"

    def run(self, step, expected_occupant="unchecked"):
        skill = step["skill"]
        if skill == "detect_objects":
            return self.detect_objects()
        if skill == "check_zone":
            return self.check_zone(step["zone"], expected_occupant)
        if skill == "find_free_position":
            return self.find_free_position()
        if skill == "pick":
            return self.pick(step["object"])
        if skill == "place":
            return self.place(step["object"], step["target"])
        if skill == "home":
            return self.home()
        return "UNKNOWN_SKILL", False
