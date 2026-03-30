"""Compatibility helpers for Isaac Sim 4.2 and newer Python namespaces.

The crucial detail is import order: in Isaac 4.2, most ``omni.isaac.core`` modules
only become importable after ``SimulationApp`` has been instantiated. This module
therefore keeps the heavy runtime imports lazy while still exposing a stable surface
to the rest of the app.
"""

from __future__ import annotations

import importlib.util
from typing import Any

ISAAC_API_FLAVOR = "unknown"
ROS2_BRIDGE_EXTENSION_NAMES = ()
_RUNTIME_LOADED = False
_ENABLE_EXTENSION = None
_RUNTIME_EXPORTS = {
    "World",
    "FixedCuboid",
    "VisualCapsule",
    "VisualCuboid",
    "VisualCylinder",
    "VisualSphere",
    "rotations",
    "viewports",
    "get_prim_at_path",
    "add_reference_to_stage",
    "ArticulationAction",
    "SingleManipulator",
    "ParallelGripper",
    "get_assets_root_path",
}

try:
    if importlib.util.find_spec("isaacsim.core.api") is None:
        raise ImportError("isaacsim.core.api is unavailable")

    from isaacsim import SimulationApp

    ISAAC_API_FLAVOR = "isaacsim"
    ROS2_BRIDGE_EXTENSION_NAMES = ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge")
except ImportError:
    from omni.isaac.kit import SimulationApp

    ISAAC_API_FLAVOR = "omni.isaac"
    ROS2_BRIDGE_EXTENSION_NAMES = ("omni.isaac.ros2_bridge", "isaacsim.ros2.bridge")


def _load_runtime_symbols() -> None:
    global _RUNTIME_LOADED, _ENABLE_EXTENSION

    if _RUNTIME_LOADED:
        return

    if ISAAC_API_FLAVOR == "isaacsim":
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import (
            FixedCuboid,
            VisualCapsule,
            VisualCuboid,
            VisualCylinder,
            VisualSphere,
        )
        from isaacsim.core.utils import rotations, viewports
        from isaacsim.core.utils.extensions import enable_extension as runtime_enable_extension
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.storage.native import get_assets_root_path
    else:
        from omni.isaac.core import World
        from omni.isaac.core.objects import (
            FixedCuboid,
            VisualCapsule,
            VisualCuboid,
            VisualCylinder,
            VisualSphere,
        )
        from omni.isaac.core.utils import rotations, viewports
        from omni.isaac.core.utils.extensions import enable_extension as runtime_enable_extension
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.manipulators import SingleManipulator
        from omni.isaac.manipulators.grippers import ParallelGripper
        from omni.isaac.nucleus import get_assets_root_path

    globals().update(
        {
            "World": World,
            "FixedCuboid": FixedCuboid,
            "VisualCapsule": VisualCapsule,
            "VisualCuboid": VisualCuboid,
            "VisualCylinder": VisualCylinder,
            "VisualSphere": VisualSphere,
            "rotations": rotations,
            "viewports": viewports,
            "get_prim_at_path": get_prim_at_path,
            "add_reference_to_stage": add_reference_to_stage,
            "ArticulationAction": ArticulationAction,
            "SingleManipulator": SingleManipulator,
            "ParallelGripper": ParallelGripper,
            "get_assets_root_path": get_assets_root_path,
        }
    )
    _ENABLE_EXTENSION = runtime_enable_extension
    _RUNTIME_LOADED = True


def enable_ros2_bridge() -> None:
    """Enable the Isaac ROS 2 bridge using the active API flavor."""
    _load_runtime_symbols()
    last_error = None
    for extension_name in ROS2_BRIDGE_EXTENSION_NAMES:
        try:
            _ENABLE_EXTENSION(extension_name)
            return
        except Exception as exc:  # pragma: no cover - depends on installed Isaac flavor.
            last_error = exc
    tried = ", ".join(ROS2_BRIDGE_EXTENSION_NAMES) or "<none>"
    raise RuntimeError(f"Unable to enable Isaac ROS 2 bridge. Tried: {tried}") from last_error


def enable_extension(extension_name: str) -> None:
    """Enable an arbitrary Isaac/Kit extension by name."""

    _load_runtime_symbols()
    _ENABLE_EXTENSION(extension_name)


def __getattr__(name: str) -> Any:
    if name in _RUNTIME_EXPORTS:
        _load_runtime_symbols()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
