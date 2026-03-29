"""ROS subscriber that mirrors `/chess/board_state` into Isaac piece visuals."""

from __future__ import annotations

from rclpy.node import Node
from std_msgs.msg import String


class IsaacBoardStateNode(Node):
    """Subscribe to FEN updates and refresh the visible board state."""

    def __init__(self, scene, board_topic: str = "/chess/board_state") -> None:
        super().__init__("isaac_board_state")
        self._scene = scene
        self._pending_fen: str | None = None
        self._subscription = self.create_subscription(String, board_topic, self._on_board_state, 10)

    def _on_board_state(self, msg: String) -> None:
        self._pending_fen = msg.data

    def consume_pending_fen(self) -> str | None:
        fen = self._pending_fen
        self._pending_fen = None
        return fen
