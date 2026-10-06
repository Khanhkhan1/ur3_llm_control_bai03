#!/usr/bin/env python3
"""Calls the 9Router (OpenAI-compatible) chat completion endpoint and turns a
natural-language command, plus the scene the camera currently sees, into a structured
JSON skill plan.

The LLM only chooses skills, their name arguments and their order -- it never sees or
produces coordinates, joint values or trajectories (task_validator.py enforces this and
checks the plan against the scene before anything moves).
"""
import json
import urllib.request

SYSTEM_PROMPT = """You are the task planner for a UR3 robot arm with a two-finger gripper
and an overhead camera, working on a table with colored cubes and three placement zones.

Available skills:
- detect_objects()            refresh the scene from the overhead camera
- check_zone(zone)            ask the camera whether a zone is free or which object is in it
- find_free_position()        choose an empty spot on the table; it becomes "temporary_position"
- pick(object)                grasp an object (the gripper must be empty)
- place(object, target)       put the held object down; target is a zone or "temporary_position"
- home()                      move the arm back to its home pose

Objects: red_cube, yellow_cube, blue_cube, green_cube, purple_cube
Zones: zone_a, zone_b, zone_c

Rules:
1. A zone holds at most one object. Always check_zone(zone) before placing into it.
2. If the target zone is occupied by another object, first move that object out of the
   way: find_free_position(), pick(that object), place(that object, temporary_position).
   Only then pick the requested object and place it into the zone.
3. The gripper holds one object at a time: every pick is followed by a place.
4. Use the current scene below to know where things are; only use objects listed there.
5. Always finish the plan with home().

Return ONLY a JSON object of the form:
{"plan": [{"skill": "check_zone", "zone": "zone_a"},
          {"skill": "pick", "object": "red_cube"},
          {"skill": "place", "object": "red_cube", "target": "zone_a"},
          {"skill": "home"}]}

Each step has "skill" plus only that skill's arguments (zone / object / target).
Do not generate coordinates or robot joint commands. Do not add any explanation,
markdown, or text outside the JSON object."""


class LLMPlannerError(RuntimeError):
    pass


def build_user_message(command: str, scene_summary: str) -> str:
    return f"Current scene:\n{scene_summary}\n\nUser command:\n{command}"


def _chat(messages, base_url, api_key, model, timeout):
    payload = {"model": model, "temperature": 0, "messages": messages}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any transport error uniformly
        raise LLMPlannerError(f"LLM request failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMPlannerError(f"Unexpected LLM response: {body}") from exc


def parse_plan(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    # Some models wrap the object in prose despite the instructions: keep the outer {...}.
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        content = content[start : end + 1]
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMPlannerError(f"LLM did not return valid JSON: {content!r}") from exc
    if not isinstance(plan, dict) or "plan" not in plan:
        raise LLMPlannerError(f"LLM JSON missing 'plan' key: {plan!r}")
    return plan


class Planner:
    """One planning conversation: the first request, then corrections when the validator
    rejects a plan (the rejection reason is sent back so the LLM can fix it)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout
        self.messages = []

    def plan(self, command: str, scene_summary: str) -> dict:
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(command, scene_summary)},
        ]
        return self._ask()

    def replan(self, rejection: str, scene_summary: str) -> dict:
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"The plan validator rejected that plan: {rejection}\n\n"
                    f"Current scene:\n{scene_summary}\n\n"
                    "Return a corrected plan as the same JSON object only."
                ),
            }
        )
        return self._ask()

    def _ask(self) -> dict:
        content = _chat(self.messages, self.base_url, self.api_key, self.model, self.timeout)
        self.messages.append({"role": "assistant", "content": content})
        return parse_plan(content)
