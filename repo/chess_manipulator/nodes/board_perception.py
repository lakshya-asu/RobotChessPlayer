#!/usr/bin/env python3
"""Fixed-camera perception scaffold for the ROS GZ chessboard."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import String

from chess_manipulator.perception.board_vision import (
    BoardCrop,
    BoardObservation,
    annotate_board,
    crop_board,
    infer_observed_board,
    occupancy_grid,
)


class BoardPerceptionNode(Node):
    """Observe the overhead board camera and publish perception diagnostics."""

    def __init__(self) -> None:
        super().__init__("board_perception")
        self.declare_parameter("image_topic", "/perception/camera/image_raw")
        self.declare_parameter("reference_fen_topic", "/chess/board_state")
        self.declare_parameter("observed_fen_topic", "/perception/observed_fen")
        self.declare_parameter("fen_confidence_topic", "/perception/fen_confidence")
        self.declare_parameter(
            "initial_fen",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        self.declare_parameter("crop_normalized", [0.225, 0.225, 0.775, 0.775])
        self.declare_parameter("occupancy_threshold", 0.18)
        self.declare_parameter("commit_confidence_threshold", 0.65)
        self.declare_parameter("publish_debug_image", True)

        crop_values = [float(value) for value in self.get_parameter("crop_normalized").value]
        self._crop = BoardCrop(*crop_values)
        self._threshold = float(self.get_parameter("occupancy_threshold").value)
        self._commit_threshold = float(self.get_parameter("commit_confidence_threshold").value)
        self._publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self._bridge = CvBridge()
        self._latest_ascii: Optional[str] = None
        self._latest_observation: Optional[BoardObservation] = None
        self._reference_fen = str(self.get_parameter("initial_fen").value)

        image_topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        reference_fen_topic = str(self.get_parameter("reference_fen_topic").value)
        self.create_subscription(String, reference_fen_topic, self._on_reference_fen, 10)
        observed_fen_topic = str(self.get_parameter("observed_fen_topic").value)
        fen_confidence_topic = str(self.get_parameter("fen_confidence_topic").value)
        self._occupancy_pub = self.create_publisher(String, "/perception/board_occupancy", 10)
        self._status_pub = self.create_publisher(String, observed_fen_topic, 10)
        self._confidence_pub = self.create_publisher(Float32, fen_confidence_topic, 10)
        self._debug_pub = self.create_publisher(Image, "/perception/debug/annotated_image", 10)
        self.get_logger().info(f"board_perception listening on {image_topic}")

    def _on_reference_fen(self, msg: String) -> None:
        if msg.data:
            self._reference_fen = msg.data

    def _on_image(self, msg: Image) -> None:
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            board_bgr = crop_board(image_bgr, self._crop)
            board_bgr = cv2.resize(board_bgr, (800, 800), interpolation=cv2.INTER_AREA)
            observation = infer_observed_board(
                self._reference_fen,
                board_bgr,
                threshold=self._threshold,
            )
            occupancy_scores = occupancy_grid(board_bgr, threshold=self._threshold)
            self._occupancy_pub.publish(String(data=observation.occupancy_ascii))
            self._status_pub.publish(String(data=observation.fen))
            self._confidence_pub.publish(Float32(data=observation.confidence))
            self._latest_ascii = observation.occupancy_ascii
            self._latest_observation = observation
            if observation.confidence >= self._commit_threshold:
                self._reference_fen = observation.fen

            if self._publish_debug_image:
                annotated = annotate_board(board_bgr, occupancy_scores)
                debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
        except Exception as exc:  # pragma: no cover - runtime ROS protection
            self.get_logger().error(f"board_perception failed to process image: {exc}")


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = BoardPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
