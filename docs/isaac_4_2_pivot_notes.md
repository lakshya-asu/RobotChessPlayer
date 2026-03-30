# Isaac 4.2 Pivot Notes

This branch is intentionally following the Ekumen pattern:

- native Isaac Sim app process
- ROS-side MoveIt and game logic
- Panda robot retained for now
- simulator pivot first, robot swap only if necessary

## Concrete decisions

- Robot: Franka Panda
- Simulator: Isaac Sim 4.2.0
- Install style: native local install inside the workspace
- Default install path: `third_party/isaac-sim-4.2.0`
- Container workflow: deferred unless the native install proves unstable

## Immediate next implementation targets

1. Replace the transitional Isaac bridge with a Panda-native Isaac control adapter.
2. Import the full physical chess scene into Isaac instead of the procedural placeholder scene.
3. Keep the ROS perception and MoveIt stack intact while swapping only the simulation backend.
4. Reintroduce dual-robot execution once the single-Panda Isaac path is stable.
