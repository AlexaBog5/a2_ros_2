#!/usr/bin/env python3
"""Multiplex TARE and FAR planner waypoints onto /way_point for local_planner."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

VALID_SOURCES = frozenset({'tare', 'far'})


class WaypointMux(Node):
    """Forward waypoints from the selected planner to local_planner."""

    def __init__(self) -> None:
        super().__init__('waypoint_mux')

        self.declare_parameter('tare_waypoint_topic', '/tare/way_point')
        self.declare_parameter('far_waypoint_topic', '/far/way_point')
        self.declare_parameter('output_waypoint_topic', '/way_point')
        self.declare_parameter('select_topic', '/planner/select')
        self.declare_parameter('default_source', 'tare')
        self.declare_parameter('goal_point_topic', '/goal_point')
        self.declare_parameter('odom_topic', '/state_estimation')
        self.declare_parameter('goal_frame', 'map')

        tare_topic = self.get_parameter('tare_waypoint_topic').value
        far_topic = self.get_parameter('far_waypoint_topic').value
        output_topic = self.get_parameter('output_waypoint_topic').value
        select_topic = self.get_parameter('select_topic').value
        default_source = self.get_parameter('default_source').value.lower()
        goal_point_topic = self.get_parameter('goal_point_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self._goal_frame = self.get_parameter('goal_frame').value

        if default_source not in VALID_SOURCES:
            self.get_logger().warn(
                f'Invalid default_source "{default_source}", using "tare"'
            )
            default_source = 'tare'

        self._active_source = default_source
        self._last_odom: Odometry | None = None
        self._output_pub = self.create_publisher(PointStamped, output_topic, 10)
        self._goal_point_pub = self.create_publisher(
            PointStamped, goal_point_topic, 10
        )

        self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)

        self.create_subscription(
            PointStamped,
            tare_topic,
            self._make_waypoint_callback('tare'),
            10,
        )
        self.create_subscription(
            PointStamped,
            far_topic,
            self._make_waypoint_callback('far'),
            10,
        )
        self.create_subscription(String, select_topic, self._select_callback, 10)

        self.get_logger().info(
            f'Waypoint mux active source: {self._active_source} '
            f'(tare={tare_topic}, far={far_topic}, out={output_topic}, '
            f'select={select_topic})'
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self._last_odom = msg

    def _current_position_waypoint(self) -> PointStamped | None:
        if self._last_odom is None:
            return None
        waypoint = PointStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = self._goal_frame
        waypoint.point = self._last_odom.pose.pose.position
        return waypoint

    def _stop_at_current_position(self) -> None:
        waypoint = self._current_position_waypoint()
        if waypoint is None:
            self.get_logger().warn(
                'No odometry yet; cannot publish stop goal for FAR switch'
            )
            return
        self._output_pub.publish(waypoint)
        self._goal_point_pub.publish(waypoint)
        self.get_logger().info(
            f'Published stop goal at ({waypoint.point.x:.2f}, '
            f'{waypoint.point.y:.2f}, {waypoint.point.z:.2f})'
        )

    def _make_waypoint_callback(self, source: str):
        def callback(msg: PointStamped) -> None:
            if source != self._active_source:
                return
            self._output_pub.publish(msg)

        return callback

    def _select_callback(self, msg: String) -> None:
        source = msg.data.strip().lower()
        if source not in VALID_SOURCES:
            self.get_logger().warn(
                f'Ignoring invalid planner select "{msg.data}" '
                f'(expected "tare" or "far")'
            )
            return
        if source == self._active_source:
            return
        previous_source = self._active_source
        self._active_source = source
        if previous_source == 'tare' and source == 'far':
            self._stop_at_current_position()
        self.get_logger().info(f'Switched active planner to: {source}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
