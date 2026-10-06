#!/usr/bin/env python3
"""Plan Validator: checks an LLM plan before anything moves.

1. Form: every step is a whitelisted skill with exactly its named arguments, and only
   names (objects, zones, temporary_position) -- never coordinates or joint values.
2. Feasibility: the plan is simulated step by step on the scene the camera reported
   (which object is where, which zone holds what). A step whose precondition fails
   rejects the whole plan, e.g. placing into a zone that is still occupied, picking with
   a full gripper, or placing without having checked the target zone first.

Returns the steps plus what the simulation expects (zone contents at each check_zone,
final location of each moved object) so the executor can compare it with the camera.
"""
from world_model import TEMPORARY_POSITION

# skill -> its argument names
ALLOWED_SKILLS = {
    "detect_objects": [],
    "check_zone": ["zone"],
    "find_free_position": [],
    "pick": ["object"],
    "place": ["object", "target"],
    "home": [],
}
MAX_STEPS = 20


class PlanValidationError(RuntimeError):
    pass


def format_call(step: dict) -> str:
    args = ", ".join(str(step[a]) for a in ALLOWED_SKILLS.get(step.get("skill"), []) if a in step)
    return f"{step.get('skill')}({args})"


def validate_plan(plan: dict, snapshot: dict):
    """Return (steps, expectations) or raise PlanValidationError.

    snapshot: {"locations": {object: zone | "table"}, "held": object | None, "zones": [...]}
    expectations: {"check_zone": {step_index: occupant or None}, "final": {object: location}}
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("plan"), list):
        raise PlanValidationError("Plan must be a JSON object with a 'plan' array")
    steps = plan["plan"]
    if not steps:
        raise PlanValidationError("Plan is empty")
    if len(steps) > MAX_STEPS:
        raise PlanValidationError(f"Plan has {len(steps)} steps (limit {MAX_STEPS})")

    zones = set(snapshot["zones"])
    locations = dict(snapshot["locations"])
    held = snapshot.get("held")
    checked_zones = set()
    temp_free = False  # a temporary position has been chosen and is still empty
    moved = set()
    expect_check = {}

    def occupant(zone):
        return next((o for o, loc in sorted(locations.items()) if loc == zone), None)

    for i, step in enumerate(steps):
        where = f"Step {i + 1}"
        if not isinstance(step, dict) or "skill" not in step:
            raise PlanValidationError(f"{where} is missing 'skill'")
        skill = step["skill"]
        if skill not in ALLOWED_SKILLS:
            raise PlanValidationError(
                f"{where}: skill '{skill}' is not in the allowed list {sorted(ALLOWED_SKILLS)}"
            )
        expected_args = ALLOWED_SKILLS[skill]
        extra = sorted(set(step) - {"skill"} - set(expected_args))
        if extra:
            raise PlanValidationError(
                f"{where}: {format_call(step)} has unexpected field(s) {extra}; skills take "
                "only names, never coordinates or joint values"
            )
        missing = [a for a in expected_args if a not in step]
        if missing:
            raise PlanValidationError(f"{where}: {skill} is missing argument(s) {missing}")
        call = format_call(step)

        if skill == "check_zone":
            zone = step["zone"]
            if zone not in zones:
                raise PlanValidationError(f"{where}: unknown zone '{zone}' (zones: {sorted(zones)})")
            checked_zones.add(zone)
            expect_check[i] = occupant(zone)

        elif skill == "find_free_position":
            temp_free = True

        elif skill == "pick":
            obj = step["object"]
            if obj not in locations:
                raise PlanValidationError(
                    f"{where}: {call}: '{obj}' is not in the scene the camera sees "
                    f"({sorted(locations)})"
                )
            if held is not None:
                raise PlanValidationError(f"{where}: {call}: the gripper is still holding {held}")
            if locations[obj] == "gripper":
                raise PlanValidationError(f"{where}: {call}: {obj} is already held")
            held = obj
            locations[obj] = "gripper"
            moved.add(obj)

        elif skill == "place":
            obj, target = step["object"], step["target"]
            if held != obj:
                raise PlanValidationError(
                    f"{where}: {call}: the gripper holds {held or 'nothing'}, not {obj}"
                )
            if target in zones:
                if target not in checked_zones:
                    raise PlanValidationError(
                        f"{where}: {call}: {target} must be checked with check_zone({target}) "
                        "before placing into it"
                    )
                other = occupant(target)
                if other is not None:
                    raise PlanValidationError(
                        f"{where}: {call}: {target} is occupied by {other} at this point; move "
                        f"{other} to a temporary position first (find_free_position, pick, place)"
                    )
            elif target == TEMPORARY_POSITION:
                if not temp_free:
                    raise PlanValidationError(
                        f"{where}: {call}: call find_free_position() before placing at "
                        f"{TEMPORARY_POSITION}"
                    )
                temp_free = False
            else:
                raise PlanValidationError(
                    f"{where}: {call}: target must be one of {sorted(zones)} or {TEMPORARY_POSITION}"
                )
            held = None
            locations[obj] = target

    if steps[-1]["skill"] != "home":
        raise PlanValidationError("The plan must end with home()")
    if held is not None:
        raise PlanValidationError(f"The plan ends with {held} still in the gripper")

    final = {obj: locations[obj] for obj in sorted(moved)}
    return steps, {"check_zone": expect_check, "final": final}
