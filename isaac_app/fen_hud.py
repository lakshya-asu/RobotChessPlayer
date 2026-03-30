"""Lightweight Isaac GUI HUD for the current chessboard FEN."""

from __future__ import annotations

from typing import Optional

from rclpy.node import Node
from std_msgs.msg import Float32, String

from isaac_app.chessboard import FILES, board_map_from_fen


def render_fen_board_text(fen: str) -> str:
    """Render a compact ASCII board from a FEN string."""

    piece_map = board_map_from_fen(fen)
    lines = ["  a b c d e f g h"]
    for rank in range(8, 0, -1):
        row = []
        for file_ in FILES:
            row.append(piece_map.get(f"{file_}{rank}", "."))
        lines.append(f"{rank} " + " ".join(row))
    return "\n".join(lines)


class IsaacFenHud(Node):
    """Display the latest perceived or accepted game FEN in an Isaac UI window."""

    def __init__(
        self,
        observed_fen_topic: str = "/perception/observed_fen",
        current_fen_topic: str = "/game/current_fen",
        confidence_topic: str = "/perception/fen_confidence",
    ) -> None:
        super().__init__("isaac_fen_hud")
        import omni.ui as ui

        self._ui = ui
        self._observed_fen = ""
        self._current_fen = ""
        self._confidence = 0.0

        self.create_subscription(String, observed_fen_topic, self._on_observed_fen, 10)
        self.create_subscription(String, current_fen_topic, self._on_current_fen, 10)
        self.create_subscription(Float32, confidence_topic, self._on_confidence, 10)

        self._window = ui.Window(
            "Board State",
            width=300,
            height=260,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._source_label = ui.Label(
                    "source: waiting",
                    alignment=ui.Alignment.LEFT_TOP,
                    height=20,
                )
                self._fen_label = ui.Label(
                    "fen: waiting",
                    alignment=ui.Alignment.LEFT_TOP,
                    word_wrap=True,
                    height=40,
                )
                self._board_label = ui.Label(
                    render_fen_board_text(""),
                    alignment=ui.Alignment.LEFT_TOP,
                    height=0,
                    style={"font_size": 16},
                )

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_window"):
                self._window.visible = False
        except Exception:
            pass
        return super().destroy_node()

    def _on_observed_fen(self, msg: String) -> None:
        if msg.data:
            self._observed_fen = msg.data
            self._refresh()

    def _on_current_fen(self, msg: String) -> None:
        if msg.data:
            self._current_fen = msg.data
            self._refresh()

    def _on_confidence(self, msg: Float32) -> None:
        self._confidence = float(msg.data)
        self._refresh()

    def _selected_fen(self) -> tuple[str, str]:
        if self._observed_fen:
            return self._observed_fen, "perception"
        if self._current_fen:
            return self._current_fen, "game"
        return "", "waiting"

    def _refresh(self) -> None:
        fen, source = self._selected_fen()
        try:
            board_text = render_fen_board_text(fen)
        except Exception:
            board_text = "invalid fen"
        self._source_label.text = f"source: {source}  confidence: {self._confidence:.2f}"
        self._fen_label.text = f"fen: {fen or 'waiting'}"
        self._board_label.text = board_text
