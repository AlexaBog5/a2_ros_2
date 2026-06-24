#!/usr/bin/env python3
"""Autonomous survey mission: stand, explore, save map, return home via mega stack + mux."""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import rclpy
from a2_interfaces.msg import OperatingMode
from a2_interfaces.srv import SetOperatingMode
from direct_lidar_inertial_odometry.srv import SavePCD
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, String

from a2_orchestrator.mission_state import MissionState

RESUME_ORIGIN_EPSILON = 1e-3


class MissionOrchestrator(Node):
    """State machine for locomotion, TARE explore, map save, and FAR return home."""

    MODE_STAND = OperatingMode.STAND_UP
    MODE_UNLOCK = OperatingMode.BALANCE_STAND
    MODE_WALK = OperatingMode.VELOCITY_MOVE
    MODE_SIT = OperatingMode.STAND_DOWN

    def __init__(self) -> None:
        """Declare parameters, create pubs/subs, and start the 0.2 s state timer."""
        super().__init__('mission_orchestrator')
        self._declare_parameters()
        self._load_parameters()

        self._state = MissionState.CHECK_PREREQS
        self._node_launched_at = time.monotonic()
        self._state_entered_mono = time.monotonic()
        self._exploring_started_mono: Optional[float] = None
        self._explore_stop_reason = ''

        self._mode_request_pending = False
        self._last_mode_accepted: Optional[bool] = None

        self._odom_seen = False
        self._exploration_finished = False
        self._last_odom: Optional[Odometry] = None
        self._home = Point()
        self._nav_goal_sent = False
        self._map_save_done = False
        self._map_save_pending = False
        self._map_save_succeeded: Optional[bool] = None
        self._done_logged = False
        self._start_explore_done = False
        self._nav_setup_done = False
        self._sit_balance_done = False

        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._goal_pub = self.create_publisher(PointStamped, self._goal_topic, 10)
        self._planner_select_pub = self.create_publisher(
            String, self._planner_select_topic, 10
        )
        self._start_exploration_pub = self.create_publisher(
            Bool, self._start_exploration_topic, 10
        )
        self._detection_enable_pub = self.create_publisher(
            Bool, self._detection_enable_topic, 10
        )
        self._detection_save_pub = self.create_publisher(
            Bool, self._detection_save_topic, 10
        )

        self.create_subscription(
            Odometry, self._odom_topic, self._odom_callback, qos_sensor
        )
        self.create_subscription(
            Bool, self._exploration_finish_topic, self._exploration_finish_callback, 10
        )
        self.create_subscription(
            PointStamped,
            self._investigate_point_topic,
            self._investigate_object_callback,
            10,
        )

        self._mode_client = self.create_client(SetOperatingMode, '/a2/set_mode')
        self._save_pcd_client = self.create_client(
            SavePCD, self._dlio_save_pcd_service
        )
        self.create_timer(0.2, self._tick)
        self._publish_detection_enable(False)
        self._set_status('initialized')

    # ------------------------------------------------------------------ params

    def _declare_parameters(self) -> None:
        """Register ROS parameters (do not declare use_sim_time — launch sets it)."""
        self.declare_parameter('save_dir', './runs/a2_mission')
        self.declare_parameter('stand_wait_sec', 3.0)
        self.declare_parameter('exploration_finish_topic', '/exploration_finish')
        self.declare_parameter('exploration_timeout_sec', 600.0)
        self.declare_parameter('home_arrival_threshold_m', 0.5)
        self.declare_parameter('nav_home_timeout_sec', 600.0)
        self.declare_parameter('skip_home', False)
        self.declare_parameter('map_leaf_size', 0.15)
        self.declare_parameter('prereq_timeout_sec', 60.0)
        self.declare_parameter('odom_topic', '/state_estimation')
        self.declare_parameter('goal_topic', '/goal_point')
        self.declare_parameter('planner_select_topic', '/planner/select')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('status_topic', '/mission/status')
        self.declare_parameter('dlio_save_pcd_service', '/save_pcd')
        self.declare_parameter('start_exploration_topic', '/start_exploration')
        self.declare_parameter('home_goal_x', 0.0)
        self.declare_parameter('home_goal_y', 0.0)
        self.declare_parameter('home_goal_z', 0.0)
        self.declare_parameter('investigate_point_topic', '/investigate_point')
        self.declare_parameter('detection_enable_topic', '/detection/enable')
        self.declare_parameter('detection_save_topic', '/detection/save')

    def _load_parameters(self) -> None:
        """Read declared parameters into instance fields and create save_dir."""
        self._save_dir = self.get_parameter('save_dir').value
        self._stand_wait_sec = self.get_parameter('stand_wait_sec').value
        self._exploration_finish_topic = self.get_parameter(
            'exploration_finish_topic'
        ).value
        self._exploration_timeout_sec = self.get_parameter(
            'exploration_timeout_sec'
        ).value
        self._home_threshold = self.get_parameter('home_arrival_threshold_m').value
        self._nav_home_timeout_sec = self.get_parameter('nav_home_timeout_sec').value
        self._skip_home = self.get_parameter('skip_home').value
        self._map_leaf_size = self.get_parameter('map_leaf_size').value
        self._prereq_timeout_sec = self.get_parameter('prereq_timeout_sec').value
        self._odom_topic = self.get_parameter('odom_topic').value
        self._goal_topic = self.get_parameter('goal_topic').value
        self._planner_select_topic = self.get_parameter(
            'planner_select_topic'
        ).value
        self._map_frame = self.get_parameter('map_frame').value
        self._status_topic = self.get_parameter('status_topic').value
        self._dlio_save_pcd_service = self.get_parameter('dlio_save_pcd_service').value
        self._start_exploration_topic = self.get_parameter(
            'start_exploration_topic'
        ).value
        self._home = Point(
            x=float(self.get_parameter('home_goal_x').value),
            y=float(self.get_parameter('home_goal_y').value),
            z=float(self.get_parameter('home_goal_z').value),
        )
        self._investigate_point_topic = self.get_parameter(
            'investigate_point_topic'
        ).value
        self._detection_enable_topic = self.get_parameter(
            'detection_enable_topic'
        ).value
        self._detection_save_topic = self.get_parameter('detection_save_topic').value

        os.makedirs(self._save_dir, exist_ok=True)

    # ------------------------------------------------------------------ helpers

    def _set_status(self, detail: str) -> None:
        """Publish ``STATE:detail`` on ``/mission/status`` and log it."""
        msg = String()
        msg.data = f'{self._state.name}:{detail}'
        self._status_pub.publish(msg)
        self.get_logger().info(f'[{self._state.name}] {detail}')

    def _transition(self, new_state: MissionState, detail: str = '') -> None:
        """Enter ``new_state``, reset state timer, publish status, and sync detection hooks."""
        self._state = new_state
        self._state_entered_mono = time.monotonic()
        self._set_status(detail or new_state.name.lower())
        if new_state in (MissionState.EXPLORING, MissionState.INVESTIGATING):
            self._publish_detection_enable(True)
        elif new_state == MissionState.SAVE_MAP:
            self._publish_detection_save()
            self._publish_detection_enable(False)

    def _elapsed(self) -> float:
        """Seconds since this node was created (wall clock, not sim time)."""
        return time.monotonic() - self._node_launched_at

    def _state_elapsed(self) -> float:
        """Seconds since the current state was entered (wall clock)."""
        return time.monotonic() - self._state_entered_mono

    def _exploring_elapsed(self) -> float:
        """Seconds since exploration started (wall clock)."""
        if self._exploring_started_mono is None:
            return 0.0
        return time.monotonic() - self._exploring_started_mono

    def _select_planner(self, source: str) -> None:
        """Publish ``tare`` or ``far`` on ``/planner/select`` for waypoint_mux."""
        msg = String()
        msg.data = source
        self._planner_select_pub.publish(msg)
        self.get_logger().info(f'Selected planner: {source}')

    def _publish_start_exploration(self) -> None:
        """Tell TARE to begin when ``kAutoStart`` is false."""
        msg = Bool()
        msg.data = True
        self._start_exploration_pub.publish(msg)

    def _publish_home_goal(self) -> None:
        """Send FAR a ``PointStamped`` goal on ``/goal_point`` (``_home`` from params)."""
        self._publish_goal_point(self._home)

    def _publish_goal_point(self, point: Point) -> None:
        """Publish a map-frame goal on ``/goal_point`` for FAR planner."""
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._map_frame
        goal.point = point
        self._goal_pub.publish(goal)

    def _publish_detection_enable(self, enabled: bool) -> None:
        """Tell ``detection_processor`` whether to track detections and publish investigate points."""
        msg = Bool()
        msg.data = enabled
        self._detection_enable_pub.publish(msg)
        self.get_logger().info(
            f'Detection processing {"enabled" if enabled else "disabled"} '
            f'on {self._detection_enable_topic}'
        )

    def _publish_detection_save(self) -> None:
        """Tell ``detection_processor`` to write tracked detections to CSV before nav home."""
        msg = Bool()
        msg.data = True
        self._detection_save_pub.publish(msg)
        self.get_logger().info(
            f'Detection save requested on {self._detection_save_topic}'
        )

    @staticmethod
    def _is_resume_point(msg: PointStamped) -> bool:
        """Return True when ``msg`` is an empty/origin signal to resume TARE exploration."""
        return (
            abs(msg.point.x) < RESUME_ORIGIN_EPSILON
            and abs(msg.point.y) < RESUME_ORIGIN_EPSILON
            and abs(msg.point.z) < RESUME_ORIGIN_EPSILON
        )

    def _check_exploration_timeout(self) -> bool:
        """Transition to SAVE_MAP on exploration timeout. Returns True if transitioned."""
        if (
            self._exploration_timeout_sec > 0.0
            and self._exploring_elapsed() >= self._exploration_timeout_sec
        ):
            self._explore_stop_reason = 'timeout'
            self._transition(MissionState.SAVE_MAP, 'exploration timeout')
            return True
        return False

    # ------------------------------------------------------------------ callbacks

    def _odom_callback(self, msg: Odometry) -> None:
        """Cache latest odometry for prereqs, origin.txt, and nav-home distance."""
        self._last_odom = msg
        self._odom_seen = True

    def _exploration_finish_callback(self, msg: Bool) -> None:
        """Set finish flag when TARE publishes true on ``/exploration_finish``."""
        if not msg.data:
            return
        if self._state != MissionState.EXPLORING:
            return
        self._exploration_finished = True

    def _investigate_object_callback(self, msg: PointStamped) -> None:
        """Handle investigate/resume signals from ``detection_processor`` on ``/investigate_point``.

        Origin point (0,0,0) resumes TARE exploration. Any other point switches to FAR
        and navigates to the detected object. Only active in EXPLORING or INVESTIGATING.
        """
        if self._state not in (MissionState.EXPLORING, MissionState.INVESTIGATING):
            return

        if self._is_resume_point(msg):
            self._select_planner('tare')
            if self._state != MissionState.EXPLORING:
                self._transition(MissionState.EXPLORING, 'resumed exploration')
            return

        self._select_planner('far')
        time.sleep(1)
        self._publish_goal_point(msg.point)
        detail = (
            f'investigating object ({msg.point.x:.2f}, '
            f'{msg.point.y:.2f}, {msg.point.z:.2f})'
        )
        if self._state != MissionState.INVESTIGATING:
            self._transition(MissionState.INVESTIGATING, detail)
        else:
            self._set_status(detail)

    # ------------------------------------------------------------------ mode FSM

    def _request_mode(self, mode: int) -> bool:
        """Async call to ``/a2/set_mode``; result handled in ``_on_mode_response``."""
        if self._mode_request_pending or self._last_mode_accepted is not None:
            return False
        if not self._mode_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn('Waiting for /a2/set_mode...')
            return False

        req = SetOperatingMode.Request()
        req.mode = mode
        self._mode_request_pending = True
        future = self._mode_client.call_async(req)
        future.add_done_callback(lambda f: self._on_mode_response(f, mode))
        return True

    def _mode_response_ready(self) -> bool:
        """True when an async mode request has completed."""
        return not self._mode_request_pending and self._last_mode_accepted is not None

    def _consume_mode_response(self) -> bool:
        """Return whether the last mode request was accepted and clear the latch."""
        accepted = bool(self._last_mode_accepted)
        self._last_mode_accepted = None
        return accepted

    def _on_mode_response(self, future, requested_mode: int) -> None:
        """Service callback: store accept/reject; transition to FAILED on hard errors."""
        self._mode_request_pending = False
        try:
            response = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f'Mode {requested_mode} call failed: {ex}')
            self._last_mode_accepted = False
            self._transition(MissionState.FAILED, 'mode service error')
            return

        self._last_mode_accepted = response.success
        if response.success:
            self.get_logger().info(
                f'Mode {requested_mode} accepted: {response.message}'
            )
            return

        self.get_logger().error(
            f'Mode {requested_mode} rejected: {response.message}'
        )
        self._transition(MissionState.FAILED, response.message)

    # ------------------------------------------------------------------ home / map

    def _record_home(self) -> bool:
        """Write actual start pose to ``origin.txt`` (nav goal uses ``home_goal_*``)."""
        if self._last_odom is None:
            return False
        start = self._last_odom.pose.pose.position
        origin_path = os.path.join(self._save_dir, 'origin.txt')
        with open(origin_path, 'w', encoding='utf-8') as handle:
            handle.write(f'{start.x:.6f} {start.y:.6f} {start.z:.6f}\n')
        self.get_logger().info(
            f'Start pose recorded to origin.txt ({start.x:.2f}, {start.y:.2f}, '
            f'{start.z:.2f}); nav goal is ({self._home.x:.2f}, '
            f'{self._home.y:.2f}, {self._home.z:.2f})'
        )
        return True

    def _distance_to_home(self) -> Optional[float]:
        """XY distance from current odom to ``_home``; None if odom unavailable."""
        if self._last_odom is None:
            return None
        dx = self._last_odom.pose.pose.position.x - self._home.x
        dy = self._last_odom.pose.pose.position.y - self._home.y
        return math.hypot(dx, dy)

    def _request_save_map(self) -> bool:
        """Async call to DLIO SavePCD; result handled in ``_on_save_map_response``."""
        if self._map_save_pending or self._map_save_succeeded is not None:
            return False
        if not self._save_pcd_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn(
                f'Waiting for {self._dlio_save_pcd_service}...'
            )
            return False

        req = SavePCD.Request()
        req.leaf_size = float(self._map_leaf_size)
        req.save_path = self._save_dir
        self._map_save_pending = True
        future = self._save_pcd_client.call_async(req)
        future.add_done_callback(self._on_save_map_response)
        return True

    def _map_save_response_ready(self) -> bool:
        """True when an async SavePCD request has completed."""
        return not self._map_save_pending and self._map_save_succeeded is not None

    def _consume_map_save_response(self) -> bool:
        """Return whether SavePCD succeeded and clear the latch."""
        succeeded = bool(self._map_save_succeeded)
        self._map_save_succeeded = None
        return succeeded

    def _on_save_map_response(self, future) -> None:
        """Service callback: latch success/failure for ``_tick_save_map``."""
        self._map_save_pending = False
        try:
            response = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f'SavePCD failed: {ex}')
            self._map_save_succeeded = False
            return

        if response.success:
            self.get_logger().info(
                f'Map saved to {self._save_dir}/clean_map.pcd'
            )
            self._map_save_succeeded = True
            return

        self.get_logger().error('SavePCD returned success=False')
        self._map_save_succeeded = False

    # ------------------------------------------------------------------ state machine (one _tick_* handler per MissionState)

    def _tick(self) -> None:
        """Dispatch the active state handler; runs every 0.2 s from the timer."""
        if self._state in (MissionState.FAILED, MissionState.DONE):
            return

        handlers = {
            MissionState.CHECK_PREREQS: self._tick_prereqs,
            MissionState.STAND: self._tick_stand,
            MissionState.WAIT_STAND: self._tick_wait_stand,
            MissionState.UNLOCK: self._tick_unlock,
            MissionState.WALK: self._tick_walk,
            MissionState.RECORD_HOME: self._tick_record_home,
            MissionState.START_EXPLORE: self._tick_start_explore,
            MissionState.EXPLORING: self._tick_exploring,
            MissionState.INVESTIGATING: self._tick_investigating,
            MissionState.SAVE_MAP: self._tick_save_map,
            MissionState.NAV_HOME: self._tick_nav_home,
            MissionState.SIT_DOWN: self._tick_sit_down,
            MissionState.DONE: self._tick_done,
        }
        handlers[self._state]()

    def _tick_prereqs(self) -> None:
        """Wait for ``/a2/set_mode`` and odometry before locomotion. Next: STAND or FAILED."""
        if self._elapsed() > self._prereq_timeout_sec:
            self._transition(
                MissionState.FAILED,
                'prerequisite timeout (start sim + mega stack first)',
            )
            return
        if not self._mode_client.wait_for_service(timeout_sec=0.0):
            self._set_status('waiting for /a2/set_mode')
            return
        if not self._odom_seen:
            self._set_status('waiting for odometry')
            return
        self._transition(MissionState.STAND)

    def _tick_stand(self) -> None:
        """Request STAND_UP (mode 2). Next: WAIT_STAND or FAILED."""
        if not self._mode_request_pending and self._last_mode_accepted is None:
            self._request_mode(self.MODE_STAND)
            return
        if not self._mode_response_ready():
            return
        if not self._consume_mode_response():
            return
        self._transition(MissionState.WAIT_STAND, 'stand accepted')

    def _tick_wait_stand(self) -> None:
        """Pause ``stand_wait_sec`` after stand before unlock. Next: UNLOCK."""
        if self._state_elapsed() < self._stand_wait_sec:
            return
        self._transition(MissionState.UNLOCK)

    def _tick_unlock(self) -> None:
        """Request BALANCE_STAND (mode 3). Next: WALK or FAILED."""
        if not self._mode_request_pending and self._last_mode_accepted is None:
            self._request_mode(self.MODE_UNLOCK)
            return
        if not self._mode_response_ready():
            return
        if not self._consume_mode_response():
            return
        self._transition(MissionState.WALK, 'unlock accepted')

    def _tick_walk(self) -> None:
        """Request VELOCITY_MOVE (mode 4). Next: RECORD_HOME or FAILED."""
        if not self._mode_request_pending and self._last_mode_accepted is None:
            self._request_mode(self.MODE_WALK)
            return
        if not self._mode_response_ready():
            return
        if not self._consume_mode_response():
            return
        self._transition(MissionState.RECORD_HOME, 'walk accepted')

    def _tick_record_home(self) -> None:
        """Write ``origin.txt`` from current odom. Next: START_EXPLORE."""
        if not self._record_home():
            self._set_status('waiting for odometry')
            return
        self._exploration_finished = False
        self._start_explore_done = False
        self._transition(MissionState.START_EXPLORE)

    def _tick_start_explore(self) -> None:
        """Once: ``/planner/select=tare``, ``/start_exploration=true``. Next: EXPLORING."""
        if not self._start_explore_done:
            self._select_planner('tare')
            self._publish_start_exploration()
            self._start_explore_done = True
            self._exploring_started_mono = time.monotonic()
            self._exploration_finished = False
            self._transition(MissionState.EXPLORING, 'exploring')

    def _tick_exploring(self) -> None:
        """Wait for ``/exploration_finish`` or timeout. Next: SAVE_MAP or INVESTIGATING."""
        if self._exploration_finished:
            self._explore_stop_reason = 'complete'
            self._transition(MissionState.SAVE_MAP, 'exploration complete')
            return
        if self._check_exploration_timeout():
            return
        self._set_status(
            f'exploring ({self._exploring_elapsed():.0f}s, '
            f'limit={self._exploration_timeout_sec:.0f}s)'
        )

    def _tick_investigating(self) -> None:
        """Navigate to a detected object via FAR; resume via origin on ``/investigate_point``."""
        if self._check_exploration_timeout():
            return
        self._set_status(
            f'investigating ({self._exploring_elapsed():.0f}s, '
            f'limit={self._exploration_timeout_sec:.0f}s)'
        )

    def _tick_save_map(self) -> None:
        """Request SavePCD once (async). Next: NAV_HOME, SIT_DOWN, or FAILED."""
        if not self._map_save_done:
            if self._map_save_succeeded is None and not self._map_save_pending:
                if not self._request_save_map():
                    if self._state_elapsed() > 5.0:
                        self._transition(MissionState.FAILED, 'map save failed')
                    else:
                        self._set_status('waiting for SavePCD')
                    return
            if not self._map_save_response_ready():
                self._set_status('saving map')
                return
            if not self._consume_map_save_response():
                self._transition(MissionState.FAILED, 'map save failed')
                return
            self._map_save_done = True
        if self._skip_home:
            self._sit_balance_done = False
            self._transition(MissionState.SIT_DOWN, 'map saved, skip_home')
            return
        self._nav_setup_done = False
        self._nav_goal_sent = False
        self._transition(MissionState.NAV_HOME, 'map saved, navigating home')

    def _tick_nav_home(self) -> None:
        """Once: select FAR and publish home goal; then wait for arrival. Next: SIT_DOWN or FAILED."""
        if self._state_elapsed() > self._nav_home_timeout_sec:
            self._transition(MissionState.FAILED, 'nav home timeout')
            return

        if not self._nav_setup_done:
            self._select_planner('far')
            time.sleep(1)
            self._publish_home_goal()
            self._nav_setup_done = True
            self._nav_goal_sent = True
            self._set_status('switched to FAR, home goal published')
            return

        dist = self._distance_to_home()
        if dist is not None and dist <= self._home_threshold:
            self._sit_balance_done = False
            self._transition(
                MissionState.SIT_DOWN,
                f'home reached ({dist:.2f} m)',
            )
            return

        detail = (
            f'navigating home ({dist:.2f} m)' if dist is not None else 'navigating'
        )
        self._set_status(detail)

    def _tick_sit_down(self) -> None:
        """Stop walking then sit: BALANCE_STAND → STAND_DOWN. Next: DONE."""
        if not self._sit_balance_done:
            if not self._mode_request_pending and self._last_mode_accepted is None:
                self._request_mode(self.MODE_UNLOCK)
                return
            if not self._mode_response_ready():
                return
            if not self._consume_mode_response():
                self._transition(MissionState.FAILED, 'sit balance failed')
                return
            self._sit_balance_done = True
            return

        if not self._mode_request_pending and self._last_mode_accepted is None:
            self._request_mode(self.MODE_SIT)
            return
        if not self._mode_response_ready():
            return
        if not self._consume_mode_response():
            self._transition(MissionState.FAILED, 'sit down failed')
            return
        self._transition(MissionState.DONE, 'sitting down')

    def _tick_done(self) -> None:
        """Publish final status once. Terminal state."""
        if self._done_logged:
            return
        self._done_logged = True
        self._set_status(f'finished ({self._explore_stop_reason})')


def main(args=None) -> None:
    """Run the mission orchestrator node until shutdown."""
    rclpy.init(args=args)
    node = MissionOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
