#!/usr/bin/env python3
"""Entry point: one natural-language command, end to end.

    Camera -> scene state -> LLM Planner -> JSON plan -> Plan Validator
           -> Skill Executor (perception skills + skill_server / MoveIt 2) -> UR3 + 2F-85
           -> camera verification of the result

If the camera shows something different from what the validated plan assumed (a
check_zone answer that does not match), the remaining plan is dropped and a new one is
requested from the current scene.

Usage (see run_command.sh):
    task_runner.py --ros-args -p command:="Put the red cube in zone B."
"""
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from llm_planner import LLMPlannerError, Planner
from skill_executor import SkillExecutor
from task_validator import PlanValidationError, format_call, validate_plan
from workspace import load_workspace
from world_model import TEMPORARY_POSITION, WorldModel

MAX_LLM_ATTEMPTS = 3  # first plan + corrections after validator rejections
MAX_PLANNING_ROUNDS = 2  # re-plan once if the camera contradicts the plan mid-way


class TaskRunnerNode(Node):
    def __init__(self):
        super().__init__("task_runner")
        self.declare_parameter("command", "")
        self.declare_parameter("llm_base_url", "http://127.0.0.1:20128/v1")
        self.declare_parameter("llm_model", "oc/muse-spark-1.3-contributor-free")
        self.declare_parameter("llm_api_key", "")
        # Every printed line is also published, so demo_recorder can overlay it on video.
        self.log_pub = self.create_publisher(String, "task_runner/log", 50)

    def log(self, text: str = ""):
        print(text, flush=True)
        for line in text.splitlines() or [""]:
            self.log_pub.publish(String(data=line))


def execution_line(call: str, status: str, width: int = 38) -> str:
    dots = "." * max(1, width - len(call))
    return f"{call} {dots} {status}"


def get_valid_plan(node, planner, command, world):
    """Ask the LLM for a plan and validate it; send rejections back for correction."""
    raw = planner.plan(command, world.summary())
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            steps, expectations = validate_plan(raw, world.snapshot())
            return steps, expectations
        except PlanValidationError as exc:
            node.log("LLM PLAN (rejected by validator):")
            for step in raw.get("plan", []) if isinstance(raw, dict) else []:
                node.log(f"- {format_call(step) if isinstance(step, dict) else step}")
            node.log(f"REJECTED: {exc}")
            node.log()
            if attempt == MAX_LLM_ATTEMPTS:
                raise
            node.log("Asking the LLM to correct the plan...")
            raw = planner.replan(str(exc), world.summary())
    raise PlanValidationError("no valid plan")


def main():
    rclpy.init()
    node = TaskRunnerNode()

    command = node.get_parameter("command").value
    if not command and len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        command = sys.argv[1]
    if not command:
        print('ERROR: no command given. Pass -p command:="..." or as the first argument.')
        rclpy.shutdown()
        sys.exit(1)

    # Prefer the environment variable so the real key never lands in a committed file.
    api_key = os.environ.get("NINEROUTER_API_KEY") or node.get_parameter("llm_api_key").value
    planner = Planner(
        node.get_parameter("llm_base_url").value, api_key, node.get_parameter("llm_model").value
    )

    world = WorldModel(load_workspace())
    executor = SkillExecutor(node, world)

    node.log("USER COMMAND:")
    node.log(command)
    node.log()

    # The arm starts at (or returns to) home, out of the overhead camera's view.
    status, ok = executor.home()
    status, ok = executor.detect_objects() if ok else (status, ok)
    if not ok:
        node.log(f"Could not observe the scene: {status}")
        node.log("TASK FAILED")
        rclpy.shutdown()
        sys.exit(1)

    node.log("CAMERA SCENE:")
    node.log(world.summary())
    node.log()

    task_ok = False
    expectations = {"final": {}}
    for round_no in range(1, MAX_PLANNING_ROUNDS + 1):
        try:
            steps, expectations = get_valid_plan(node, planner, command, world)
        except LLMPlannerError as exc:
            node.log(f"LLM PLANNING FAILED: {exc}")
            break
        except PlanValidationError:
            node.log("No valid plan after corrections.")
            break

        node.log("LLM PLAN:")
        for step in steps:
            node.log(f"- {format_call(step)}")
        node.log()
        node.log(f"PLAN VALIDATION: OK ({len(steps)} steps checked against the camera scene)")
        node.log()

        node.log("EXECUTION:")
        scene_changed = False
        task_ok = True
        for i, step in enumerate(steps):
            expected = expectations["check_zone"].get(i, "unchecked")
            status, ok = executor.run(step, expected)
            node.log(execution_line(format_call(step), status))
            if not ok:
                task_ok = False
                scene_changed = step["skill"] == "check_zone" and "plan assumed" in status
                break
        node.log()
        if task_ok or not scene_changed or round_no == MAX_PLANNING_ROUNDS:
            break
        node.log("The camera contradicts the plan: re-planning from the current scene.")
        node.log("CAMERA SCENE:")
        node.log(world.summary())
        node.log()

    if task_ok:
        # Final check with a fresh image: is every moved object where the plan put it?
        node.log("CAMERA VERIFICATION:")
        status, ok = executor.detect_objects()
        if not ok:
            node.log(f"camera ........ {status}")
            task_ok = False
        for obj, target in expectations["final"].items():
            actual = world.location_of(obj)
            if target == TEMPORARY_POSITION:
                good, claim = actual == "table", f"{obj} set aside on the table"
            else:
                good, claim = actual == target, f"{obj} in {target}"
            task_ok = task_ok and good
            node.log(execution_line(claim, "OK" if good else f"NO (seen: {actual})"))
        node.log()

    node.log("TASK SUCCESS" if task_ok else "TASK FAILED")

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if task_ok else 1)


if __name__ == "__main__":
    main()
