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
from std_msgs.msg import Bool, String

WORLD_MATCH_DISTANCE_NOISY = 2.0
WORLD_MATCH_DISTANCE = 1.0

CAMERA_WIDTH = 640.0
CAMERA_HEIGHT = 320.0
MIN_BBOX_DIMENSION = 30.0


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


class DetectionProcessor(Node):
    """Subscribe to ``/detection_info``, track objects, and publish ``/investigate_point``.

    Processing is gated by ``/detection/enable`` (published by ``mission_orchestrator``).
    CSV export is triggered by ``/detection/save`` when exploration ends (before nav home).
    While disabled, detections are ignored so objects visible during stand/walk do not
    trigger investigation. When disabled, tracked objects are cleared.
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

        self._investigate_point_pub = self.create_publisher(
            PointStamped,
            self._investigate_point_topic,
            10,
        )
        self._dummy_pub = self.create_publisher(
            String,
            '/dummy',
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
        self._dummy_pub.publidh("SAVING")
        if not msg.data:
            return
        self._dummy_pub.publidh("SAVING")
        self.write_detections_csv()

    def _enable_callback(self, msg: Bool) -> None:
        """Latch enable flag from orchestrator; clear tracks when processing stops."""
        if msg.data == self._processing_enabled:
            return
        self._processing_enabled = msg.data
        if not msg.data:
            self.objects.clear()
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

    def find_matching_object(
        self, class_id: str, x: float, y: float
    ) -> TrackedObject | None:
        """Return a tracked object of ``class_id`` within ``WORLD_MATCH_DISTANCE_NOISY`` of (x, y)."""
        for obj in self.objects:
            if obj.class_id != class_id:
                continue
            if distance_xy(x, y, obj.global_x, obj.global_y) < WORLD_MATCH_DISTANCE_NOISY:
                return obj
        return None

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

    def publish_dummy(self, detection, world_point: PointStamped) -> None:
        """Publish a good detection summary to /dummy for debugging or downstream use."""
        msg = String()
        msg.data = (
            f'class={detection.class_id} id={detection.id} '
            f'confidence={detection.confidence:.2f} '
            f'pos=({world_point.point.x:.3f}, {world_point.point.y:.3f}, {world_point.point.z:.3f})'
        )
        self._dummy_pub.publish(msg)
        self.get_logger().info(f'Publishing /dummy: {msg.data}')

    def destroy_node(self):
        """Shut down without writing CSV (orchestrator triggers save via ``/detection/save``)."""
        return super().destroy_node()

    def aggregate_close_detections(self) -> list[dict]:
        """Cluster nearby tracked objects by class for CSV export."""
        clusters: list[dict] = []

        for obj in self.objects:
            matched_cluster = None
            for cluster in clusters:
                if obj.class_id != cluster['class_id']:
                    continue
                if distance_xy(
                    obj.global_x,
                    obj.global_y,
                    cluster['global_x'],
                    cluster['global_y'],
                ) <= WORLD_MATCH_DISTANCE_NOISY:
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                clusters.append(
                    {
                        'class_id': obj.class_id,
                        'global_x': obj.global_x,
                        'global_y': obj.global_y,
                        'objs': [obj],
                    }
                )
            else:
                matched_cluster['objs'].append(obj)

        aggregated: list[dict] = []
        for cluster in clusters:
            objs = cluster['objs']
            total_confidence = sum(o.confidence for o in objs)
            if total_confidence <= 0.0:
                continue

            aggregated.append(
                {
                    'class_id': cluster['class_id'],
                    'global_x': sum(o.global_x * o.confidence for o in objs)
                    / total_confidence,
                    'global_y': sum(o.global_y * o.confidence for o in objs)
                    / total_confidence,
                    'global_z': sum(o.global_z * o.confidence for o in objs)
                    / total_confidence,
                    'confidence': total_confidence / len(objs),
                    'bbox_min_x': sum(o.bbox[0] * o.confidence for o in objs)
                    / total_confidence,
                    'bbox_min_y': sum(o.bbox[1] * o.confidence for o in objs)
                    / total_confidence,
                    'bbox_max_x': sum(o.bbox[2] * o.confidence for o in objs)
                    / total_confidence,
                    'bbox_max_y': sum(o.bbox[3] * o.confidence for o in objs)
                    / total_confidence,
                    'pending_confirmation': any(
                        o.pending_confirmation for o in objs
                    ),
                    'continue_sent': any(o.continue_sent for o in objs),
                }
            )

        return aggregated

    def write_detections_csv(self) -> bool:
        """Write aggregated tracked detections to ``output_csv``. Returns True on success."""
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
            f'Wrote {len(rows)} detection(s) to {self._csv_path}'
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

            if is_small or near_edge:
                matched_object = self.find_matching_object(
                    detection.class_id, wx, wy
                )
                if matched_object is None:
                    self.get_logger().info(
                        f'Small/edge detection without nearby track: '
                        f'class={detection.class_id}'
                    )
                    self.publish_investigate_point(world_point)
                else:
                    self.get_logger().info(
                        f'Small/edge detection near existing track; skipping publish '
                        f'class={detection.class_id}'
                    )
                continue

            matched_object = self.find_matching_object(detection.class_id, wx, wy)

            if matched_object is None:
                self.get_logger().info(
                    f'New object: class={detection.class_id} '
                    f'pos=({wx:.2f}, {wy:.2f}, {wz:.2f})'
                )
                self.objects.append(
                    TrackedObject(
                        class_id=detection.class_id,
                        global_x=wx,
                        global_y=wy,
                        global_z=wz,
                        confidence=detection.confidence,
                        bbox=bbox,
                    )
                )
                self.publish_dummy(detection, world_point)
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

            if matched_object.pending_confirmation and not matched_object.continue_sent:
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