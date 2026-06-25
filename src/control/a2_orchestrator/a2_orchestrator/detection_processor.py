#!/usr/bin/env python3
"""Track YOLO detections and publish investigate/resume points for the mission orchestrator."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped
from object_detection_msgs.msg import ObjectDetectionInfoArray
from rclpy.node import Node
from std_msgs.msg import Bool

WORLD_MATCH_DISTANCE = 1.5
MIN_BBOX_CENTER_SEPARATION_PX = 60.0

CAMERA_WIDTH = 640.0
CAMERA_HEIGHT = 320.0
MIN_BBOX_DIMENSION = 30.0
NO_DEPTH_Z = -1.0


@dataclass
class TrackedObject:
    """A detected object tracked in map frame across multiple observations."""

    class_id: str
    global_x: float
    global_y: float
    global_z: float
    confidence: float
    bbox: tuple
    pending_confirmation: bool = False
    continue_sent: bool = False
    bad: bool = False
    observation_count: int = 1


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return XY distance between two map-frame points."""
    return math.hypot(x1 - x2, y1 - y2)


def bbox_dimensions(bbox: tuple) -> tuple[float, float]:
    """Return width and height of a bounding box tuple (min_x, min_y, max_x, max_y)."""
    min_x, min_y, max_x, max_y = bbox
    return max_x - min_x, max_y - min_y


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """Return pixel centre of a bounding box."""
    min_x, min_y, max_x, max_y = bbox
    return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5


def bbox_center_distance(bbox_a: tuple, bbox_b: tuple) -> float:
    """Return pixel distance between two bounding-box centres."""
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    return math.hypot(ax - bx, ay - by)


def bbox_near_edge(bbox: tuple, margin: float = 5.0) -> bool:
    """Return True when the bbox touches the image border within ``margin`` pixels."""
    min_x, min_y, max_x, max_y = bbox
    return (
        min_x <= margin
        or min_y <= margin
        or max_x >= CAMERA_WIDTH - margin
        or max_y >= CAMERA_HEIGHT - margin
    )


def is_small_bbox(bbox: tuple) -> bool:
    """Return True when either bbox dimension is below ``MIN_BBOX_DIMENSION``."""
    width, height = bbox_dimensions(bbox)
    return width < MIN_BBOX_DIMENSION or height < MIN_BBOX_DIMENSION


def bbox_area(bbox: tuple) -> float:
    """Return the pixel area of a bounding box."""
    width, height = bbox_dimensions(bbox)
    return width * height


def normalize_class_id(class_id: str) -> str:
    """Normalize class labels for consistent matching."""
    return str(class_id).strip().lower()


def is_better_observation(
    candidate_bbox: tuple,
    candidate_confidence: float,
    current_bbox: tuple,
    current_confidence: float,
) -> bool:
    """Return True when the candidate observation is strictly better than the current one."""
    return (
        bbox_area(candidate_bbox) > bbox_area(current_bbox)
        and candidate_confidence > current_confidence
    )


class DetectionProcessor(Node):
    """Subscribe to ``/detection_info``, track objects, and publish ``/investigate_point``.

    Processing is gated by ``/detection/enable``. CSV export is triggered by
    ``/detection/save`` when exploration ends. Tracks are cleared by
    ``/detection/reset`` at the start of a new session.
    """

    def __init__(self) -> None:
        """Declare parameters, create pubs/subs, and start with processing disabled."""
        super().__init__('detection_processor')

        self.declare_parameter('detection_info_topic', '/detection_info')
        self.declare_parameter('investigate_point_topic', '/investigate_point')
        self.declare_parameter('detection_enable_topic', '/detection/enable')
        self.declare_parameter('detection_save_topic', '/detection/save')
        self.declare_parameter('detection_reset_topic', '/detection/reset')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('output_csv', 'detections.csv')
        self.declare_parameter('world_match_distance', WORLD_MATCH_DISTANCE)
        self.declare_parameter('tf_lookup_tolerance_sec', 0.1)

        self._detection_info_topic = self.get_parameter('detection_info_topic').value
        self._investigate_point_topic = self.get_parameter('investigate_point_topic').value
        self._detection_enable_topic = self.get_parameter('detection_enable_topic').value
        self._detection_save_topic = self.get_parameter('detection_save_topic').value
        self._detection_reset_topic = self.get_parameter('detection_reset_topic').value
        self._map_frame = self.get_parameter('map_frame').value
        self._csv_path = self.get_parameter('output_csv').value
        self._world_match_distance = float(
            self.get_parameter('world_match_distance').value
        )
        self._tf_lookup_tolerance_sec = float(
            self.get_parameter('tf_lookup_tolerance_sec').value
        )

        self.objects: list[TrackedObject] = []
        self._processing_enabled = False

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            ObjectDetectionInfoArray,
            self._detection_info_topic,
            self.detection_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self._detection_enable_topic,
            self._enable_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self._detection_save_topic,
            self._save_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self._detection_reset_topic,
            self._reset_callback,
            10,
        )

        self._investigate_point_pub = self.create_publisher(
            PointStamped,
            self._investigate_point_topic,
            10,
        )

        self._csv_headers = [
            'class', 'x', 'y', 'z',
            'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y',
            'bad', 'observation_count',
        ]

        self.get_logger().info(
            f'DetectionProcessor ready: sub={self._detection_info_topic}, '
            f'csv={self._csv_path}, map_frame={self._map_frame}, '
            f'match={self._world_match_distance:.1f}m'
        )

    def _save_callback(self, msg: Bool) -> None:
        """Write tracked detections to CSV when commanded."""
        if not msg.data:
            return
        self.get_logger().info(
            f'Save requested ({len(self.objects)} track(s) in memory)'
        )
        self.write_detections_csv()

    def _reset_callback(self, msg: Bool) -> None:
        """Clear tracked objects when a new exploration session starts."""
        if not msg.data:
            return
        self.reset_tracks()

    def _enable_callback(self, msg: Bool) -> None:
        """Latch enable flag; do not clear tracks here (reset/save handle that)."""
        if msg.data == self._processing_enabled:
            return
        self._processing_enabled = msg.data
        self.get_logger().info(
            f'Detection processing {"enabled" if msg.data else "disabled"}'
        )

    def reset_tracks(self) -> None:
        """Clear all tracked objects."""
        count = len(self.objects)
        self.objects.clear()
        if count:
            self.get_logger().info(f'Cleared {count} tracked object(s) for new session')

    def transform_point_to_map(
        self, x: float, y: float, z: float, source_frame: str, stamp
    ) -> PointStamped | None:
        """Transform a detection position from ``source_frame`` into ``map`` frame."""
        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = stamp
        point.point.x = x
        point.point.y = y
        point.point.z = z

        lookup_times = [
            Time.from_msg(stamp),
            Time.from_msg(stamp) - Duration(seconds=self._tf_lookup_tolerance_sec),
            Time.from_msg(stamp) + Duration(seconds=self._tf_lookup_tolerance_sec),
        ]

        for lookup_time in lookup_times:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    point.header.frame_id,
                    lookup_time,
                )
                return tf2_geometry_msgs.do_transform_point(point, transform)
            except Exception:  # noqa: BLE001
                continue

        self.get_logger().warn(
            f'Could not transform {source_frame} -> {self._map_frame} at detection time; '
            'skipping (latest TF would collapse distinct objects)'
        )
        return None

    def _same_physical_object(
        self,
        candidate: TrackedObject,
        class_id: str,
        wx: float,
        wy: float,
        bbox: tuple,
        *,
        same_frame: bool,
    ) -> bool:
        """Return True if ``candidate`` is the same instance as the new observation."""
        if normalize_class_id(candidate.class_id) != class_id:
            return False

        if distance_xy(wx, wy, candidate.global_x, candidate.global_y) >= (
            self._world_match_distance
        ):
            return False

        if same_frame and bbox_center_distance(
            bbox, candidate.bbox
        ) >= MIN_BBOX_CENTER_SEPARATION_PX:
            return False

        return True

    def find_matching_object(
        self,
        class_id: str,
        wx: float,
        wy: float,
        bbox: tuple,
        extra_tracks: list[TrackedObject] | None = None,
    ) -> TrackedObject | None:
        """Return the best matching track for this observation, if any."""
        same_frame_ids = {id(obj) for obj in (extra_tracks or [])}

        candidates = list(self.objects)
        if extra_tracks:
            candidates.extend(extra_tracks)

        best_match: TrackedObject | None = None
        best_distance = self._world_match_distance

        for obj in candidates:
            if not self._same_physical_object(
                obj,
                class_id,
                wx,
                wy,
                bbox,
                same_frame=id(obj) in same_frame_ids,
            ):
                continue
            dist = distance_xy(wx, wy, obj.global_x, obj.global_y)
            if dist < best_distance:
                best_distance = dist
                best_match = obj

        return best_match

    def publish_investigate_point(self, waypoint: PointStamped) -> None:
        """Publish a non-origin map point so the orchestrator sends FAR to investigate."""
        self._investigate_point_pub.publish(waypoint)
        self.get_logger().info(
            f'Publishing {self._investigate_point_topic}: '
            f'({waypoint.point.x:.3f}, {waypoint.point.y:.3f}, '
            f'{waypoint.point.z:.3f}) frame={waypoint.header.frame_id}'
        )

    def publish_resume_exploration(self) -> None:
        """Publish an empty/origin point so the orchestrator resumes TARE exploration."""
        empty_point = PointStamped()
        self._investigate_point_pub.publish(empty_point)
        self.get_logger().info(
            f'Publishing {self._investigate_point_topic}: origin (resume exploration)'
        )

    def write_detections_csv(self) -> bool:
        """Write one CSV row per tracked object. Returns True on success."""
        tracked = list(self.objects)
        csv_dir = os.path.dirname(os.path.abspath(self._csv_path))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        try:
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self._csv_headers)
                writer.writeheader()
                for obj in tracked:
                    writer.writerow(
                        {
                            'class': obj.class_id,
                            'x': obj.global_x,
                            'y': obj.global_y,
                            'z': obj.global_z,
                            'bbox_min_x': obj.bbox[0],
                            'bbox_min_y': obj.bbox[1],
                            'bbox_max_x': obj.bbox[2],
                            'bbox_max_y': obj.bbox[3],
                            'bad': obj.bad,
                            'observation_count': obj.observation_count,
                        }
                    )
        except OSError as ex:
            self.get_logger().error(f'Failed to write detection CSV: {ex}')
            return False

        self.get_logger().info(
            f'Wrote {len(tracked)} detection(s) to {self._csv_path}'
        )
        for idx, obj in enumerate(tracked, start=1):
            self.get_logger().info(
                f'  [{idx}] {obj.class_id} '
                f'({obj.global_x:.2f}, {obj.global_y:.2f}, {obj.global_z:.2f})'
            )
        return True

    def detection_callback(self, msg: ObjectDetectionInfoArray) -> None:
        """Process YOLO detections when enabled; publish investigate or resume signals."""
        if not self._processing_enabled:
            return

        source_frame = msg.header.frame_id
        stamp = msg.header.stamp
        if len(msg.info) == 0:
            return

        self.get_logger().info(
            f'Received {len(msg.info)} detection(s) from {source_frame}'
        )

        created_this_msg: list[TrackedObject] = []

        for detection in msg.info:
            class_id = normalize_class_id(detection.class_id)
            bbox = (
                detection.bounding_box_min_x,
                detection.bounding_box_min_y,
                detection.bounding_box_max_x,
                detection.bounding_box_max_y,
            )

            if abs(detection.position.z - NO_DEPTH_Z) < 1e-3:
                self.get_logger().info(
                    f'Skipping class={class_id} id={detection.id}: no depth'
                )
                continue

            is_bad = is_small_bbox(bbox) or bbox_near_edge(bbox)

            world_point = self.transform_point_to_map(
                detection.position.x,
                detection.position.y,
                detection.position.z,
                source_frame,
                stamp,
            )
            if world_point is None:
                continue

            wx = world_point.point.x
            wy = world_point.point.y
            wz = world_point.point.z

            matched_object = self.find_matching_object(
                class_id, wx, wy, bbox, extra_tracks=created_this_msg
            )

            if matched_object is None:
                obj = TrackedObject(
                    class_id=class_id,
                    global_x=wx,
                    global_y=wy,
                    global_z=wz,
                    confidence=detection.confidence,
                    bbox=bbox,
                    bad=is_bad,
                )
                self.objects.append(obj)
                created_this_msg.append(obj)
                self.get_logger().info(
                    f'New track ({len(self.objects)} total): class={class_id} '
                    f'pos=({wx:.2f}, {wy:.2f}, {wz:.2f}) bad={is_bad}'
                )
                if is_bad:
                    self.publish_investigate_point(world_point)
                else:
                    self.publish_resume_exploration()
                continue

            matched_object.observation_count += 1
            if is_better_observation(
                bbox,
                detection.confidence,
                matched_object.bbox,
                matched_object.confidence,
            ):
                was_bad = matched_object.bad
                matched_object.global_x = wx
                matched_object.global_y = wy
                matched_object.global_z = wz
                matched_object.confidence = detection.confidence
                matched_object.bbox = bbox
                matched_object.bad = is_bad

                if was_bad and not is_bad:
                    self.publish_resume_exploration()
                elif is_bad:
                    self.publish_investigate_point(world_point)
            elif matched_object.pending_confirmation and not matched_object.continue_sent:
                self.publish_resume_exploration()
                matched_object.pending_confirmation = False
                matched_object.continue_sent = True


def main(args=None) -> None:
    """Run the detection processor node until shutdown."""
    rclpy.init(args=args)
    node = DetectionProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
