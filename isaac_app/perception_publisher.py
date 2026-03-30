"""Simulator-assisted overhead RGB and segmentation publisher for Isaac."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from isaac_app.chessboard import FILES, board_map_from_fen


PIECE_TO_LABEL: Dict[str, int] = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}
ARUCO_MARKER_ID = 23
ARUCO_DICTIONARY = "DICT_4X4_50"


class IsaacPerceptionPublisher(Node):
    """Publish a deterministic overhead RGB image plus mono8 piece labels."""

    def __init__(
        self,
        board_state_provider: Callable[[], Dict[str, str]] | None = None,
        board_topic: str = "/chess/board_state",
        image_topic: str = "/perception/camera/image_raw",
        segmentation_topic: str = "/perception/camera/segmentation",
        camera_info_topic: str = "/perception/camera/camera_info",
        image_size_px: int = 960,
        publish_period_sec: float = 0.2,
    ) -> None:
        super().__init__("isaac_perception_publisher")
        self._reference_fen = ""
        self._board_state_provider = board_state_provider
        self._image_size_px = int(image_size_px)
        self._rgb_pub = self.create_publisher(Image, image_topic, 10)
        self._segmentation_pub = self.create_publisher(Image, segmentation_topic, 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)
        self.create_subscription(String, board_topic, self._on_board_state, 10)
        self._timer = self.create_timer(max(float(publish_period_sec), 0.05), self._publish_images)

    def _on_board_state(self, msg: String) -> None:
        self._reference_fen = msg.data

    def _publish_images(self) -> None:
        piece_map = (
            self._board_state_provider()
            if self._board_state_provider is not None
            else board_map_from_fen(self._reference_fen)
        )
        rgb_image, label_image = self._render_board_images(piece_map)
        stamp = self.get_clock().now().to_msg()

        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "perception_overhead_camera"
        rgb_msg.height = rgb_image.shape[0]
        rgb_msg.width = rgb_image.shape[1]
        rgb_msg.encoding = "bgr8"
        rgb_msg.is_bigendian = False
        rgb_msg.step = rgb_image.shape[1] * 3
        rgb_msg.data = rgb_image.tobytes()
        self._rgb_pub.publish(rgb_msg)

        seg_msg = Image()
        seg_msg.header = rgb_msg.header
        seg_msg.height = label_image.shape[0]
        seg_msg.width = label_image.shape[1]
        seg_msg.encoding = "mono8"
        seg_msg.is_bigendian = False
        seg_msg.step = label_image.shape[1]
        seg_msg.data = label_image.tobytes()
        self._segmentation_pub.publish(seg_msg)

        info_msg = CameraInfo()
        info_msg.header = rgb_msg.header
        info_msg.height = rgb_msg.height
        info_msg.width = rgb_msg.width
        self._camera_info_pub.publish(info_msg)

    def _render_board_images(self, piece_map: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        image_size = self._image_size_px
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[:] = (44, 44, 44)
        labels = np.zeros((image_size, image_size), dtype=np.uint8)

        board_left = int(image_size * 0.225)
        board_top = int(image_size * 0.225)
        board_right = int(image_size * 0.775)
        board_bottom = int(image_size * 0.775)
        board = image[board_top:board_bottom, board_left:board_right]
        label_board = labels[board_top:board_bottom, board_left:board_right]
        square_px = board.shape[0] // 8

        for row in range(8):
            for col in range(8):
                color = (220, 220, 220) if (row + col) % 2 == 0 else (70, 90, 130)
                y0 = row * square_px
                x0 = col * square_px
                board[y0 : y0 + square_px, x0 : x0 + square_px] = color

        self._draw_aruco_anchor(
            image=image,
            board_left=board_left,
            board_top=board_top,
            board_width=board.shape[1],
            square_px=square_px,
        )

        for square_name, piece_symbol in piece_map.items():
            file_index = FILES.index(square_name[0])
            rank_index = 8 - int(square_name[1])
            center_x = file_index * square_px + (square_px // 2)
            center_y = rank_index * square_px + (square_px // 2)
            radius = max(square_px // 3, 6)

            rgb_color = (20, 20, 20) if piece_symbol.islower() else (245, 245, 245)
            rr, cc = np.ogrid[:board.shape[0], :board.shape[1]]
            disk = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= radius ** 2
            board[disk] = rgb_color
            label_board[disk] = PIECE_TO_LABEL[piece_symbol]
            self._draw_piece_glyph(board, piece_symbol, center_x, center_y, radius)

        return image, labels

    def _draw_piece_glyph(
        self,
        board: np.ndarray,
        piece_symbol: str,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> None:
        glyph = piece_symbol.upper()
        text_color = (240, 240, 240) if piece_symbol.islower() else (25, 25, 25)
        font_scale = max(radius / 24.0, 0.35)
        thickness = max(int(radius / 7), 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            glyph,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_origin = (
            int(center_x - text_width / 2),
            int(center_y + text_height / 2),
        )
        cv2.putText(
            board,
            glyph,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_aruco_anchor(
        self,
        image: np.ndarray,
        board_left: int,
        board_top: int,
        board_width: int,
        square_px: int,
    ) -> None:
        marker_size = max(square_px, 36)
        gap_px = max(int(square_px * 0.35), 12)
        marker_left = int(board_left + (board_width / 2) - (marker_size / 2))
        marker_top = max(board_top - gap_px - marker_size, 0)
        marker_bottom = marker_top + marker_size
        marker_right = marker_left + marker_size

        marker = np.full((marker_size, marker_size), 255, dtype=np.uint8)
        dictionary = None
        if hasattr(cv2, "aruco"):
            dict_id = getattr(cv2.aruco, ARUCO_DICTIONARY, None)
            if dict_id is not None:
                dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        if dictionary is not None:
            marker = cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID, marker_size)
        else:
            tile = marker_size // 4
            for row in range(4):
                for col in range(4):
                    color = 0 if (row + col) % 2 == 0 else 255
                    y0 = row * tile
                    x0 = col * tile
                    marker[y0 : y0 + tile, x0 : x0 + tile] = color

        image[marker_top:marker_bottom, marker_left:marker_right] = cv2.cvtColor(
            marker,
            cv2.COLOR_GRAY2BGR,
        )
