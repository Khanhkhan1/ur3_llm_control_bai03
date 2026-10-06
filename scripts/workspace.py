#!/usr/bin/env python3
"""Loads config/workspace.yaml (table, zones, camera calibration, free-space search area),
shared by scene_perception.py and task_runner.py."""
import os

import yaml
from ament_index_python.packages import get_package_share_directory


def load_workspace(path: str = "") -> dict:
    if not path:
        path = os.path.join(
            get_package_share_directory("ur3_llm_control"), "config", "workspace.yaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
