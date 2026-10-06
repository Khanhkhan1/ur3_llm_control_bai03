# ur3_llm_control — Bài 03: LLM skill planning with gripper and camera

ROS 2 (Humble) package: a UR3 with a Robotiq 2F-85 gripper and an overhead camera,
in Gazebo, takes a natural-language command, looks at the table, lets an LLM (via
[9Router](https://github.com/decolua/9router)) choose the skills, validates the plan
against what the camera sees, then picks and places the cubes physically.

```
Natural Language Command -> Camera scene -> LLM Planner -> JSON Plan -> Plan Validator
  -> Skill Executor -> MoveIt 2 -> UR3 + 2F-85 -> camera verification
```

## Cell

- UR3 + Robotiq 2F-85 (real finger contact and friction: object poses are never set).
- Overhead RGB camera, table, `zone_a` / `zone_b` / `zone_c`, five 5 cm cubes
  (`red`, `yellow`, `blue`, `green`, `purple`).
- Start layout: `blue_cube` already in `zone_b`, `green_cube` in `zone_c`, so
  *"Put the red cube in zone B."* cannot be done with a direct pick → place.

## Skills

| Skill | Where | What it does |
|---|---|---|
| `detect_objects()` | `scene_perception.py` | HSV segmentation of the overhead image → cube positions on the table (pinhole model + calibrated camera pose) and zone occupancy |
| `check_zone(zone)` | `skill_executor.py` | fresh camera check: zone free, or which cube occupies it |
| `find_free_position()` | `world_model.py` | free table spot ≥ 11 cm from every cube and zone center → `temporary_position` |
| `pick(object)` | `skill_server` (C++) | approach, Cartesian descent, close until the fingers stop on the cube, attach it in MoveIt, lift |
| `place(object, target)` | `skill_server` | target = zone or `temporary_position`; descent, open, detach, lift |
| `home()` | `skill_server` | back to the `up` pose (out of the camera's view) |

The LLM only picks skills, names and order. Coordinates always come from the camera.

**Plan validator** (`task_validator.py`): whitelist of skills and arguments, no
coordinates or joint values, plus a step-by-step simulation of the plan on the camera
scene: no placing into an occupied zone, `check_zone` before placing into a zone, one
object in the gripper at a time, `find_free_position` before `temporary_position`, end
with `home()` and an empty gripper. A rejected plan is sent back to the LLM with the
reason (up to 2 corrections). If a `check_zone` at run time disagrees with the plan, the
task is re-planned from the current scene.

**Collision checking**: MoveIt plans against the table and every cube the camera sees.
The held cube is attached to the gripper.

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/Khanhkhan1/ur3_llm_control_bai03.git ur3_llm_control
vcs import . < ur3_llm_control/dependencies.repos   # UR description / sim / MoveIt, Robotiq
cd ~/ros2_ws && colcon build --symlink-install
```

## Run

```bash
# 9Router (local LLM gateway), key from its dashboard -> Endpoint & Key
9router --skip-update -n -l -H 127.0.0.1
export NINEROUTER_API_KEY=<your-key>

source ~/ros2_ws/install/setup.bash
ros2 launch ur3_llm_control llm_robot.launch.py ur_type:=ur3

# once "skill_server ready" is printed:
$(ros2 pkg prefix ur3_llm_control)/lib/ur3_llm_control/run_command.sh \
  "Put the red cube in zone B."

# optional: record side view + camera + log to a video
$(ros2 pkg prefix ur3_llm_control)/lib/ur3_llm_control/demo_recorder.py \
  --ros-args -p output:=demo.mp4
```

```
CAMERA SCENE:
- blue_cube: in zone_b ...   zone_b: occupied by blue_cube

LLM PLAN:
- check_zone(zone_b)
- find_free_position()
- pick(blue_cube)
- place(blue_cube, temporary_position)
- pick(red_cube)
- place(red_cube, zone_b)
- home()

EXECUTION:
check_zone(zone_b) .................... OCCUPIED (blue_cube)
find_free_position() .................. SUCCESS (0.33, -0.24)
pick(blue_cube) ....................... SUCCESS
...
CAMERA VERIFICATION:
red_cube in zone_b .................... OK

TASK SUCCESS
```

## Notes

- Gripper commands stay inside (0, 0.8) rad. A Gazebo (DART) joint sitting exactly on its
  limit ignores velocity commands, so a gripper sent to 0.0 never closes again.
- Gazebo cameras use the `ogre` render engine. `ogre2` cannot render headless on macOS.
- If other ROS 2 / Gazebo machines share the network (log spam like `string data is not
  null-terminated` or `Unknown message type [9]`), isolate the run in every terminal:
  `export ROS_DOMAIN_ID=73 IGN_IP=127.0.0.1`.

## Result

![Demo](docs/demo.png)
