#!/usr/bin/env python3
"""
Multi-Goal Manager for FAR Planner.
This is a standalone node that listens to RViz 'Goalpoint' clicks
on `/goal_point`, queues them up, and publishes them sequentially to `/far_planner/goal_point`
for FAR Planner to execute.

It also supports going home manually (via RViz '2D Pose Estimate' or '/go_home' topic)
or automatically after all waypoints or a specific count are reached.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
import math

class MultiGoalManager(Node):
    def __init__(self):
        super().__init__('multi_goal_manager')
        
        # Declare parameters with defaults
        self.declare_parameter('goal_timeout', 30.0)      # Timeout in seconds per goal
        self.declare_parameter('reach_threshold', 0.8)    # Distance in meters to switch to next goal
        self.declare_parameter('publish_rate', 1.0)       # Rate (Hz) to republish active goal to far_planner
        self.declare_parameter('use_multi_goal', True)     # Enable queueing
        self.declare_parameter('auto_go_home', True)      # Automatically go home when queue completes
        self.declare_parameter('max_goals_before_home', 0) # Max goals before auto-returning home (0 = disabled)
        
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.reach_threshold = self.get_parameter('reach_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_multi_goal = self.get_parameter('use_multi_goal').value
        self.auto_go_home = self.get_parameter('auto_go_home').value
        self.max_goals_before_home = self.get_parameter('max_goals_before_home').value
        
        self.get_logger().info(
            f"\n=======================================================\n"
            f"MultiGoalManager Initialized!\n"
            f"  - Goal Timeout: {self.goal_timeout}s\n"
            f"  - Reach Threshold: {self.reach_threshold}m\n"
            f"  - Auto Go Home: {self.auto_go_home}\n"
            f"  - Max Goals Before Home: {self.max_goals_before_home}\n"
            f"  - Intercepting '2D Pose Estimate' button for Go Home!\n"
            f"======================================================="
        )
        
        # Publishers
        self.far_goal_pub = self.create_publisher(PointStamped, '/far_planner/goal_point', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        
        # Subscribers
        self.rviz_goal_sub = self.create_subscription(
            PointStamped, '/goal_point', self.on_rviz_goal, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/state_estimation', self.on_odom, 10)
        self.clear_sub = self.create_subscription(
            Empty, '/clear_goals', self.on_clear_goals, 10)
        self.reset_graph_sub = self.create_subscription(
            Empty, '/reset_visibility_graph', self.on_clear_goals, 10)
            
        # Go Home manual triggers (Topic & standard RViz '2D Pose Estimate' button)
        self.go_home_sub = self.create_subscription(
            Empty, '/go_home', self.on_go_home_msg, 10)
        self.rviz_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.on_rviz_pose, 10)
            
        # State variables
        self.goal_queue = []
        self.current_goal = None
        self.current_goal_start_time = None
        self.last_pub_time = self.get_clock().now()
        self.latest_odom = None
        
        self.home_pose = None
        self.is_going_home = False
        self.goals_completed_count = 0
        
        # Control Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def on_odom(self, msg):
        self.latest_odom = msg
        # Record initial robot position as the "home" coordinate
        if self.home_pose is None:
            self.home_pose = msg.pose.pose.position
            self.get_logger().info(
                f"Recorded Home Position: ({self.home_pose.x:.2f}, "
                f"{self.home_pose.y:.2f}, {self.home_pose.z:.2f})"
            )

    def on_rviz_goal(self, msg):
        if msg.header.frame_id == "multi_goal_manager_cancel":
            return

        # Normalize incoming 3D Point to be on ground
        if self.latest_odom and abs(msg.point.z) < 0.05:
            msg.point.z = self.latest_odom.pose.pose.position.z

        # Reset states if starting a fresh sequence
        if self.is_going_home or (not self.goal_queue and not self.current_goal):
            self.is_going_home = False
            self.goals_completed_count = 0

        if self.use_multi_goal:
            self.goal_queue.append(msg)
            self.get_logger().info(
                f"Queued goal #{len(self.goal_queue)}: "
                f"({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})"
            )
        else:
            self.goal_queue = [msg]
            self.current_goal = None # Force starting the new goal immediately

    def on_go_home_msg(self, msg):
        self.get_logger().info("Received /go_home command.")
        self.trigger_go_home()

    def on_rviz_pose(self, msg):
        self.get_logger().info("Received '2D Pose Estimate' click from RViz.")
        self.trigger_go_home()

    def trigger_go_home(self):
        if self.home_pose is None:
            self.get_logger().warn("Cannot go home: initial position not recorded yet.")
            return
            
        self.is_going_home = True
        self.goal_queue.clear()
        
        home_goal = PointStamped()
        home_goal.header.frame_id = self.latest_odom.header.frame_id if self.latest_odom else "odom"
        home_goal.header.stamp = self.get_clock().now().to_msg()
        home_goal.point = self.home_pose
        
        self.goal_queue.append(home_goal)
        self.current_goal = None
        self.get_logger().info(
            f"Heading Home! Destination: ({self.home_pose.x:.2f}, "
            f"{self.home_pose.y:.2f}, {self.home_pose.z:.2f})"
        )

    def on_clear_goals(self, msg):
        self.get_logger().info("Clearing all goals in the queue.")
        self.goal_queue.clear()
        self.current_goal = None
        self.is_going_home = False
        self.goals_completed_count = 0
        
        if self.latest_odom:
            cancel_goal = PointStamped()
            cancel_goal.header = self.latest_odom.header
            cancel_goal.header.frame_id = "multi_goal_manager_cancel"
            cancel_goal.point = self.latest_odom.pose.pose.position
            self.far_goal_pub.publish(cancel_goal)

    def start_next_goal(self):
        if not self.goal_queue:
            self.current_goal = None
            return
        self.current_goal = self.goal_queue.pop(0)
        self.current_goal_start_time = self.get_clock().now()
        self.last_pub_time = self.get_clock().now()
        
        goal_name = "Home" if self.is_going_home else f"Goal (index remaining: {len(self.goal_queue)})"
        self.get_logger().info(
            f"Starting {goal_name}: ({self.current_goal.point.x:.2f}, "
            f"{self.current_goal.point.y:.2f}, {self.current_goal.point.z:.2f}) "
            f"in frame '{self.current_goal.header.frame_id}'"
        )
        self.far_goal_pub.publish(self.current_goal)

    def transition_to_next_goal(self):
        self.current_goal = None
        self.goals_completed_count += 1
        
        # Check if we hit the limit of goals before heading home
        if (self.max_goals_before_home > 0 and 
            self.goals_completed_count >= self.max_goals_before_home and 
            not self.is_going_home):
            self.get_logger().info(f"Reached threshold of {self.max_goals_before_home} goals. Heading home.")
            self.trigger_go_home()
            return

        if self.goal_queue:
            self.start_next_goal()
        else:
            self.get_logger().info("Goal queue completed.")
            if self.auto_go_home and not self.is_going_home:
                self.get_logger().info("auto_go_home is enabled. Heading home.")
                self.trigger_go_home()

    def control_loop(self):
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.reach_threshold = self.get_parameter('reach_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_multi_goal = self.get_parameter('use_multi_goal').value
        self.auto_go_home = self.get_parameter('auto_go_home').value
        self.max_goals_before_home = self.get_parameter('max_goals_before_home').value
        
        if self.current_goal:
            if not self.latest_odom:
                return
                
            dx = self.latest_odom.pose.pose.position.x - self.current_goal.point.x
            dy = self.latest_odom.pose.pose.position.y - self.current_goal.point.y
            dz = self.latest_odom.pose.pose.position.z - self.current_goal.point.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            elapsed = (self.get_clock().now() - self.current_goal_start_time).nanoseconds / 1e9
            
            time_since_pub = (self.get_clock().now() - self.last_pub_time).nanoseconds / 1e9
            if time_since_pub >= (1.0 / self.publish_rate):
                self.far_goal_pub.publish(self.current_goal)
                self.last_pub_time = self.get_clock().now()
                
            if dist < self.reach_threshold:
                self.get_logger().info(f"Goal reached! (Remaining: {dist:.2f}m)")
                self.transition_to_next_goal()
            elif elapsed > self.goal_timeout:
                self.get_logger().info(f"Goal timed out! (Elapsed: {elapsed:.1f}s > {self.goal_timeout}s)")
                self.transition_to_next_goal()
        else:
            if self.goal_queue:
                self.start_next_goal()
                
        self.publish_markers()

    def publish_markers(self):
        msg = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.header.frame_id = self.latest_odom.header.frame_id if self.latest_odom else "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "multi_goal"
        delete_marker.action = Marker.DELETEALL
        msg.markers.append(delete_marker)
        
        if not self.latest_odom:
            self.marker_pub.publish(msg)
            return
            
        target_frame = self.latest_odom.header.frame_id
        marker_id = 0
        
        line = Marker()
        line.header.frame_id = target_frame
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "multi_goal"
        line.id = marker_id
        marker_id += 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.color.r = 0.0; line.color.g = 1.0; line.color.b = 0.5; line.color.a = 0.75
        line.pose.orientation.w = 1.0
        
        line.points.append(self.latest_odom.pose.pose.position)
        
        if self.current_goal:
            line.points.append(self.current_goal.point)
            
            sphere = Marker()
            sphere.header.frame_id = target_frame; sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "multi_goal"; sphere.id = marker_id; marker_id += 1
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position = self.current_goal.point; sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.5; sphere.scale.y = 0.5; sphere.scale.z = 0.5
            
            if self.is_going_home:
                # Home active goal: Green
                sphere.color.r = 0.0; sphere.color.g = 1.0; sphere.color.b = 0.0; sphere.color.a = 0.9
            else:
                sphere.color.r = 1.0; sphere.color.g = 0.85; sphere.color.b = 0.0; sphere.color.a = 0.9
            msg.markers.append(sphere)
            
            text = Marker()
            text.header.frame_id = target_frame; text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "multi_goal"; text.id = marker_id; marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position = Point(x=self.current_goal.point.x, y=self.current_goal.point.y, z=self.current_goal.point.z + 0.6)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.35
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            
            elapsed = (self.get_clock().now() - self.current_goal_start_time).nanoseconds / 1e9
            time_left = max(0.0, self.goal_timeout - elapsed)
            
            prefix = "Returning Home" if self.is_going_home else "Active"
            text.text = f"{prefix} (T-min: {time_left:.1f}s)"
            msg.markers.append(text)
            
        for i, goal in enumerate(self.goal_queue):
            line.points.append(goal.point)
            
            sphere = Marker()
            sphere.header.frame_id = target_frame; sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "multi_goal"; sphere.id = marker_id; marker_id += 1
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position = goal.point; sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.4; sphere.scale.y = 0.4; sphere.scale.z = 0.4
            sphere.color.r = 0.0; sphere.color.g = 0.9; sphere.color.b = 0.9; sphere.color.a = 0.75
            msg.markers.append(sphere)
            
            text = Marker()
            text.header.frame_id = target_frame; text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "multi_goal"; text.id = marker_id; marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position = Point(x=goal.point.x, y=goal.point.y, z=goal.point.z + 0.5)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = 0.9; text.color.g = 0.9; text.color.b = 0.9; text.color.a = 0.9
            
            text.text = f"Goal {i+1}"
            msg.markers.append(text)
            
        if len(line.points) > 1:
            msg.markers.append(line)
            
        self.marker_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiGoalManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
