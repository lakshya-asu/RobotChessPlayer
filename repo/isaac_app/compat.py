"""Compatibility helpers for Isaac Sim 4.2 and newer Python namespaces."""

from __future__ import annotations

ISAAC_API_FLAVOR = "unknown"
ROS2_BRIDGE_EXTENSION_NAMES = ()

try:
    from isaacsim import SimulationApp
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import (
        FixedCuboid,
        VisualCapsule,
        VisualCuboid,
        VisualCylinder,
        VisualSphere,
    )
    from isaacsim.core.utils import rotations, viewports
    from isaacsim.core.utils.extensions import enable_extension as _enable_extension
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot.manipulators import SingleManipulator
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.storage.native import get_assets_root_path

    ISAAC_API_FLAVOR = "isaacsim"
    ROS2_BRIDGE_EXTENSION_NAMES = ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge")
except ImportError:
    from omni.isaac.kit import SimulationApp
    from omni.isaac.core import World
    from omni.isaac.core.objects import (
        FixedCuboid,
        VisualCapsule,
        VisualCuboid,
        VisualCylinder,
        VisualSphere,
    )
    from omni.isaac.core.utils import rotations, viewports
    from omni.isaac.core.utils.extensions import enable_extension as _enable_extension
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.manipulators import SingleManipulator
    from omni.isaac.manipulators.grippers import ParallelGripper
    from omni.isaac.nucleus import get_assets_root_path

    ISAAC_API_FLAVOR = "omni.isaac"
    ROS2_BRIDGE_EXTENSION_NAMES = ("omni.isaac.ros2_bridge", "isaacsim.ros2.bridge")


def enable_ros2_bridge() -> None:
    """Enable the Isaac ROS 2 bridge using the active API flavor."""
    last_error = None
    for extension_name in ROS2_BRIDGE_EXTENSION_NAMES:
        try:
            _enable_extension(extension_name)
            return
        except Exception as exc:  # pragma: no cover - depends on installed Isaac flavor.
            last_error = exc
    tried = ", ".join(ROS2_BRIDGE_EXTENSION_NAMES) or "<none>"
    raise RuntimeError(f"Unable to enable Isaac ROS 2 bridge. Tried: {tried}") from last_error
