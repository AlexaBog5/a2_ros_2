#!/usr/bin/env python3
"""Track YOLO detections and publish investigate/resume points for the mission orchestrator."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import rclpy
from rclpy.time import Time
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped
from object_detection_msgs.msg import ObjectDetectionInfoArray
from rclpy.node import Node
from std_msgs.msg import Bool

WORLD_MATCH_DISTANCE_NOISY = 2.0
WORLD_MATCH_DISTANCE = 1.0
# Only merge CSV rows closer than this (duplicate re-observations of one object).
CSV_DEDUPE_DISTANCE = 0.4

CAMERA_WIDTH = 640.0
CAMERA_HEIGHT = 640.0
MIN_BBOX_DIMENSION = 50.0


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


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return XY distance between two map-frame points."""
    return math.hypot(x1 - x2, y1 - y2)


def bbox_dimensions(bbox: tuple) -> tuple[float, float]:
    """Return width and height of a bounding box tuple (min_x, min_y, max_x, max_y)."""
    min_x, min_y, max_x, max_y = bbox
    return max_x - min_x, max_y - min_y


def bbox_near_edge(bbox: tuple, margin: float = 10.0) -> bool:
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


class DetectionProcessor(Node):
    """Subscribe to ``/detection_info``, track objects, and publish ``/investigate_point``.

    Processing is gated by ``/detection/enable`` (published by ``mission_orchestrator``).
    CSV export is triggered by ``/detection/save`` when exploration ends (before nav home).
    All valid detections (small, edge, or large) are tracked and written to CSV.
    Small/edge detections may also trigger investigation; investigation status does
    not affect whether a detection is saved.
    """

    def __init__(self) -> None:
        """Declare parameters, create pubs/subs, and start with processing disabled."""
        super().__init__('detection_processor')

        self.declare_parameter('detection_info_topic', '/detection_info')
        self.declare_parameter('investigate_point_topic', '/investigate_point')
        self.declare_parameter('detection_enable_topic', '/detection/enable')
        self.declare_parameter('detection_save_topic', '/detection/save')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('output_csv', 'detections.csv')
        self.declare_parameter('world_match_distance', WORLD_MATCH_DISTANCE)

        self._detection_info_topic = self.get_parameter('detection_info_topic').value
        self._investigate_point_topic = self.get_parameter(
            'investigate_point_topic'
        ).value
        self._detection_enable_topic = self.get_parameter(
            'detection_enable_topic'
        ).value
        self._detection_save_topic = self.get_parameter('detection_save_topic').value
        self._map_frame = self.get_parameter('map_frame').value
        self._csv_path = self.get_parameter('output_csv').value

        # Create CSV directory and file with header at start
        csv_dir = os.path.dirname(os.path.abspath(self._csv_path))
        if csv_dir:
            try:
                os.makedirs(csv_dir, exist_ok=True)
            except OSError as ex:
                self.get_logger().error(f'Failed to create directory {csv_dir}: {ex}')
        try:
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['class', 'x', 'y', 'z'])
                writer.writeheader()
            self.get_logger().info(f'Initialized empty CSV file at {self._csv_path}')
        except OSError as ex:
            self.get_logger().error(f'Failed to initialize CSV file: {ex}')

        self.objects: list[TrackedObject] = []
        self._processing_enabled = True

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

        self._investigate_point_pub = self.create_publisher(
            PointStamped,
            self._investigate_point_topic,
            10,
        )

        self._csv_headers = ['class', 'x', 'y', 'z']

        self.get_logger().info(
            f'DetectionProcessor ready: sub={self._detection_info_topic}, '
            f'pub={self._investigate_point_topic}, '
            f'enable={self._detection_enable_topic}, '
            f'save={self._detection_save_topic} (waiting for enable)'
        )

    def _save_callback(self, msg: Bool) -> None:
        """Write tracked detections to CSV when commanded by the mission orchestrator."""
        if not msg.data:
            return
        self.write_detections_csv()

    def _enable_callback(self, msg: Bool) -> None:
        """Latch enable flag from orchestrator; clear tracks when a new session starts."""
        if msg.data == self._processing_enabled:
            return
        if msg.data:
            self.objects.clear()
        self._processing_enabled = msg.data
        self.get_logger().info(
            f'Detection processing {"enabled" if msg.data else "disabled"}'
        )

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

        # Try exact timestamp first, but if unavailable, fall back to the
        # latest/closest available transform so processing isn't blocked by
        # strict timestamp matching.
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                point.header.frame_id,
                stamp,
            )
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except Exception as ex:  # noqa: BLE001
            self.get_logger().warn(
                f'Exact transform to {self._map_frame} failed: {ex} - trying latest available transform'
            )
            try:
                latest = Time()
                transform = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    point.header.frame_id,
                    latest,
                )
                self.get_logger().info('Using latest available transform as fallback')
                return tf2_geometry_msgs.do_transform_point(point, transform)
            except Exception as ex2:  # noqa: BLE001
                self.get_logger().warn(
                    f'Fallback transform to {self._map_frame} failed: {ex2}'
                )
                return None

    def _upsert_tracked_object(
        self,
        class_id: str,
        wx: float,
        wy: float,
        wz: float,
        confidence: float,
        bbox: tuple,
    ) -> tuple[TrackedObject, bool]:
        """Insert or update a tracked object. Returns ``(object, is_new)``."""
        matched_object = self.find_matching_object(class_id, wx, wy)

        if matched_object is None:
            obj = TrackedObject(
                class_id=class_id,
                global_x=wx,
                global_y=wy,
                global_z=wz,
                confidence=confidence,
                bbox=bbox,
            )
            self.objects.append(obj)
            self.get_logger().info(
                f'New track ({len(self.objects)} total): class={class_id} '
                f'pos=({wx:.2f}, {wy:.2f}, {wz:.2f}) conf={confidence:.2f}'
            )
            return obj, True

        self.get_logger().debug(
            f'Updating track class={class_id} '
            f'pos=({wx:.2f}, {wy:.2f}, {wz:.2f}) conf={confidence:.2f}'
        )
        matched_object.global_x = wx
        matched_object.global_y = wy
        matched_object.global_z = wz
        matched_object.confidence = max(matched_object.confidence, confidence)
        matched_object.bbox = bbox
        return matched_object, False

    def find_matching_object(
        self, class_id: str, x: float, y: float
    ) -> TrackedObject | None:
        """Return the closest tracked object of ``class_id`` within match distance."""
        best_match: TrackedObject | None = None
        best_distance = self._world_match_distance

        for obj in self.objects:
            if obj.class_id != class_id:
                continue
            dist = distance_xy(x, y, obj.global_x, obj.global_y)
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

    def destroy_node(self):
        """Shut down without writing CSV (orchestrator triggers save via ``/detection/save``)."""
        return super().destroy_node()

    def aggregate_close_detections(self) -> list[dict]:
        """Prepare one CSV row per tracked object, deduping only exact re-hits."""
        rows: list[dict] = []
        for obj in self.objects:
            rows.append(
                {
                    'class_id': obj.class_id,
                    'global_x': obj.global_x,
                    'global_y': obj.global_y,
                    'global_z': obj.global_z,
                    'confidence': obj.confidence,
                }
            )

        deduped: list[dict] = []
        for row in rows:
            merged = False
            for existing in deduped:
                if row['class_id'] != existing['class_id']:
                    continue
                if distance_xy(
                    row['global_x'],
                    row['global_y'],
                    existing['global_x'],
                    existing['global_y'],
                ) <= CSV_DEDUPE_DISTANCE:
                    total_confidence = existing['confidence'] + row['confidence']
                    if total_confidence > 0.0:
                        existing['global_x'] = (
                            existing['global_x'] * existing['confidence']
                            + row['global_x'] * row['confidence']
                        ) / total_confidence
                        existing['global_y'] = (
                            existing['global_y'] * existing['confidence']
                            + row['global_y'] * row['confidence']
                        ) / total_confidence
                        existing['global_z'] = (
                            existing['global_z'] * existing['confidence']
                            + row['global_z'] * row['confidence']
                        ) / total_confidence
                        existing['confidence'] = total_confidence / 2.0
                    merged = True
                    break
            if not merged:
                deduped.append(row)

        return deduped

    def write_detections_csv(self) -> bool:
        """Write aggregated tracked detections to ``output_csv``. Returns True on success."""
        tracked = list(self.objects)
        rows = self.aggregate_close_detections()
        csv_dir = os.path.dirname(os.path.abspath(self._csv_path))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        try:
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self._csv_headers)
                writer.writeheader()
                for data in rows:
                    writer.writerow(
                        {
                            'class': data['class_id'],
                            'x': data['global_x'],
                            'y': data['global_y'],
                            'z': data['global_z'],
                        }
                    )
        except OSError as ex:
            self.get_logger().error(f'Failed to write detection CSV: {ex}')
            return False

        self.get_logger().info(
            f'Wrote {len(rows)} detection(s) to {self._csv_path} '
            f'from {len(tracked)} track(s)'
        )
        return True

    def detection_callback(self, msg: ObjectDetectionInfoArray) -> None:
        """Process YOLO detections when enabled; publish investigate or resume signals."""
        if not self._processing_enabled:
            return

        source_frame = msg.header.frame_id
        stamp = msg.header.stamp
        self.get_logger().info(
            f'Received {len(msg.info)} detections from {source_frame}'
        )

        for detection in msg.info:
            bbox = (
                detection.bounding_box_min_x,
                detection.bounding_box_min_y,
                detection.bounding_box_max_x,
                detection.bounding_box_max_y,
            )

            if detection.position.z == -1:
                self.get_logger().info(
                    f'Skipping class={detection.class_id} id={detection.id}: no depth'
                )
                continue

            is_small = is_small_bbox(bbox)
            near_edge = bbox_near_edge(bbox)

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

            matched_object, is_new = self._upsert_tracked_object(
                detection.class_id,
                wx,
                wy,
                wz,
                detection.confidence,
                bbox,
            )

            if is_small or near_edge:
                if is_new:
                    self.get_logger().info(
                        f'Small/edge detection: class={detection.class_id}, '
                        'requesting investigation'
                    )
                    self.publish_investigate_point(world_point)
                else:
                    self.get_logger().info(
                        f'Small/edge detection near existing track; skipping publish '
                        f'class={detection.class_id}'
                    )
                continue

            if is_new:
                self.write_detections_csv()
                self.publish_resume_exploration()
                continue

            if detection.confidence > matched_object.confidence:
                self.get_logger().info(
                    f'Updating {matched_object.class_id} confidence '
                    f'{matched_object.confidence:.2f}->{detection.confidence:.2f}'
                )
                matched_object.global_x = wx
                matched_object.global_y = wy
                matched_object.global_z = wz
                matched_object.confidence = detection.confidence
                matched_object.bbox = bbox
                self.write_detections_csv()

            if matched_object.pending_confirmation and not matched_object.continue_sent:
                self.publish_resume_exploration()
                matched_object.pending_confirmation = False
                matched_object.continue_sent = True
                self.write_detections_csv()


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