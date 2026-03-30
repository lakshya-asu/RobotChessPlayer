#!/usr/bin/env python3
"""Fixed-camera perception node for simulator-backed overhead chess cameras."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import String

from chess_manipulator.perception.board_vision import (
    BoardCrop,
    BoardObservation,
    annotate_board,
    board_origin_from_bounds,
    board_bounds_from_aruco,
    board_pixel_bounds,
    build_piece_templates,
    crop_board,
    crop_with_bounds,
    detect_aruco_marker,
    infer_observed_board,
    infer_observed_board_from_labels,
    infer_observed_board_from_templates,
    resize_board_image,
    threshold_scores,
)


class BoardPerceptionNode(Node):
    """Observe the overhead board camera and publish perception diagnostics."""

    def __init__(self) -> None:
        super().__init__("board_perception")
        self.declare_parameter("image_topic", "/perception/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/perception/camera/camera_info")
        self.declare_parameter("segmentation_topic", "")
        self.declare_parameter("reference_fen_topic", "/chess/board_state")
        self.declare_parameter("observed_fen_topic", "/perception/observed_fen")
        self.declare_parameter("fen_confidence_topic", "/perception/fen_confidence")
        self.declare_parameter("occupancy_topic", "/perception/board_occupancy")
        self.declare_parameter("debug_image_topic", "/perception/debug/annotated_image")
        self.declare_parameter(
            "initial_fen",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        self.declare_parameter("crop_normalized", [0.225, 0.225, 0.775, 0.775])
        self.declare_parameter("resize_px", 800)
        self.declare_parameter("occupancy_threshold", 0.18)
        self.declare_parameter("commit_confidence_threshold", 0.65)
        self.declare_parameter("signal_threshold", 0.02)
        self.declare_parameter("use_segmentation_mask", False)
        self.declare_parameter("segmentation_weight", 0.7)
        self.declare_parameter("segmentation_max_age_sec", 0.25)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("use_aruco_anchor", True)
        self.declare_parameter("aruco_marker_id", 23)
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("board_pose_topic", "/perception/board_pose")
        self.declare_parameter("square_size_m", 0.076)
        self.declare_parameter("board_origin", [0.235, -0.266, 0.31])

        crop_values = [float(value) for value in self.get_parameter("crop_normalized").value]
        self._crop = BoardCrop(*crop_values)
        self._resize_px = int(self.get_parameter("resize_px").value)
        self._threshold = float(self.get_parameter("occupancy_threshold").value)
        self._commit_threshold = float(self.get_parameter("commit_confidence_threshold").value)
        self._signal_threshold = float(self.get_parameter("signal_threshold").value)
        self._use_segmentation_mask = bool(self.get_parameter("use_segmentation_mask").value)
        self._segmentation_weight = float(self.get_parameter("segmentation_weight").value)
        self._segmentation_max_age_sec = float(self.get_parameter("segmentation_max_age_sec").value)
        self._publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self._use_aruco_anchor = bool(self.get_parameter("use_aruco_anchor").value)
        self._aruco_marker_id = int(self.get_parameter("aruco_marker_id").value)
        self._aruco_dictionary = str(self.get_parameter("aruco_dictionary").value)
        self._square_size_m = float(self.get_parameter("square_size_m").value)
        self._nominal_board_origin = tuple(
            float(value) for value in self.get_parameter("board_origin").value
        )
        self._bridge = CvBridge()
        self._latest_ascii: Optional[str] = None
        self._latest_observation: Optional[BoardObservation] = None
        self._reference_fen = str(self.get_parameter("initial_fen").value)
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_segmentation_mask = None
        self._latest_segmentation_stamp: Optional[Time] = None
        self._piece_templates = None

        image_topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, 10)
        segmentation_topic = str(self.get_parameter("segmentation_topic").value)
        if segmentation_topic:
            self.create_subscription(Image, segmentation_topic, self._on_segmentation, 10)
        reference_fen_topic = str(self.get_parameter("reference_fen_topic").value)
        self.create_subscription(String, reference_fen_topic, self._on_reference_fen, 10)
        observed_fen_topic = str(self.get_parameter("observed_fen_topic").value)
        fen_confidence_topic = str(self.get_parameter("fen_confidence_topic").value)
        occupancy_topic = str(self.get_parameter("occupancy_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        board_pose_topic = str(self.get_parameter("board_pose_topic").value)
        self._occupancy_pub = self.create_publisher(String, occupancy_topic, 10)
        self._status_pub = self.create_publisher(String, observed_fen_topic, 10)
        self._confidence_pub = self.create_publisher(Float32, fen_confidence_topic, 10)
        self._debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        self._board_pose_pub = self.create_publisher(PoseStamped, board_pose_topic, 10)
        self.get_logger().info(
            "board_perception listening on "
            f"{image_topic} (camera_info={camera_info_topic or 'disabled'}, "
            f"segmentation={segmentation_topic or 'disabled'})"
        )

    def _on_reference_fen(self, msg: String) -> None:
        if msg.data:
            self._reference_fen = msg.data

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._latest_camera_info = msg

    def _on_segmentation(self, msg: Image) -> None:
        try:
            mask = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self._latest_segmentation_mask = mask
            self._latest_segmentation_stamp = Time.from_msg(msg.header.stamp)
        except Exception as exc:  # pragma: no cover - runtime ROS protection
            self.get_logger().warning(f"board_perception failed to decode segmentation image: {exc}")

    def _segmentation_for_image(self, image_msg: Image):
        if not self._use_segmentation_mask or self._latest_segmentation_mask is None:
            return None
        if self._latest_segmentation_stamp is None:
            return None
        image_stamp = Time.from_msg(image_msg.header.stamp)
        age = abs((image_stamp - self._latest_segmentation_stamp).nanoseconds) / 1e9
        if age > self._segmentation_max_age_sec:
            return None
        return self._latest_segmentation_mask

    def _on_image(self, msg: Image) -> None:
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            nominal_bounds = board_pixel_bounds(image_bgr.shape, self._crop)
            aruco_visible = (
                detect_aruco_marker(
                    image_bgr,
                    marker_id=self._aruco_marker_id,
                    dictionary_name=self._aruco_dictionary,
                )
                is not None
            )
            bounds = (
                board_bounds_from_aruco(
                    image_bgr,
                    self._crop,
                    marker_id=self._aruco_marker_id,
                    dictionary_name=self._aruco_dictionary,
                )
                if self._use_aruco_anchor
                else None
            )
            resolved_bounds = bounds if bounds is not None else nominal_bounds
            board_origin = board_origin_from_bounds(
                resolved_bounds,
                nominal_bounds,
                self._nominal_board_origin,
                self._square_size_m,
            )
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = board_origin[0]
            pose_msg.pose.position.y = board_origin[1]
            pose_msg.pose.position.z = board_origin[2]
            pose_msg.pose.orientation.w = 1.0
            self._board_pose_pub.publish(pose_msg)
            board_bgr = (
                crop_with_bounds(image_bgr, resolved_bounds)
                if resolved_bounds is not None
                else crop_board(image_bgr, self._crop)
            )
            board_bgr = resize_board_image(board_bgr, size_px=self._resize_px)
            segmentation_mask = self._segmentation_for_image(msg)
            if segmentation_mask is not None:
                segmentation_mask = (
                    crop_with_bounds(segmentation_mask, resolved_bounds)
                    if resolved_bounds is not None
                    else crop_board(segmentation_mask, self._crop)
                )
                segmentation_mask = resize_board_image(segmentation_mask, size_px=self._resize_px)

            if segmentation_mask is not None and segmentation_mask.ndim == 2 and self._piece_templates is None:
                self._piece_templates = build_piece_templates(
                    self._reference_fen,
                    board_bgr,
                    segmentation_mask,
                )

            if (
                segmentation_mask is not None
                and segmentation_mask.ndim == 2
                and self._piece_templates
            ):
                observation = infer_observed_board_from_templates(
                    self._reference_fen,
                    board_bgr,
                    segmentation_mask,
                    self._piece_templates,
                    threshold=self._threshold,
                    min_signal_score=self._signal_threshold,
                )
            elif segmentation_mask is not None and segmentation_mask.ndim == 2:
                observation = infer_observed_board_from_labels(self._reference_fen, segmentation_mask)
            else:
                observation = infer_observed_board(
                    self._reference_fen,
                    board_bgr,
                    threshold=self._threshold,
                    segmentation_mask=segmentation_mask,
                    segmentation_weight=self._segmentation_weight,
                    min_signal_score=self._signal_threshold,
                )
            occupancy_scores = threshold_scores(observation.square_scores, threshold=self._threshold)
            self._occupancy_pub.publish(String(data=observation.occupancy_ascii))
            self._status_pub.publish(String(data=observation.fen))
            self._confidence_pub.publish(Float32(data=observation.confidence))
            self._latest_ascii = observation.occupancy_ascii
            self._latest_observation = observation
            if observation.confidence >= self._commit_threshold:
                self._reference_fen = observation.fen

            if self._publish_debug_image:
                annotated = annotate_board(board_bgr, occupancy_scores)
                cv2.putText(
                    annotated,
                    f"fen_conf={observation.confidence:.2f} signal={observation.signal_score:.2f}",
                    (10, annotated.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if segmentation_mask is not None:
                    cv2.putText(
                        annotated,
                        "segmentation-assisted"
                        if not self._piece_templates
                        else "aruco + template-assisted",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    annotated,
                    "aruco-locked" if aruco_visible else "aruco-fallback",
                    (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 220, 0),
                    1,
                    cv2.LINE_AA,
                )
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
