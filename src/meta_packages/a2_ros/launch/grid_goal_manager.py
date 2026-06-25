#!/usr/bin/env python3
"""
Grid-based Coverage and Navigation Manager for FAR Planner.
Generates a grid inside a user-defined bounding box, filters nodes that are occluded
by obstacle boundaries published by FAR Planner, sorts the path, and navigates them 
sequentially with success ranges and active timeouts.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
import math
from tf2_ros import Buffer, TransformListener

class GridGoalManager(Node):
    def __init__(self):
        super().__init__('grid_goal_manager')
        
        # Bounding box & Grid generation parameters
        self.declare_parameter('x_min', 1.0)
        self.declare_parameter('x_max', 9.0)
        self.declare_parameter('y_min', -4.0)
        self.declare_parameter('y_max', 4.0)
        self.declare_parameter('spacing', 2.0)            # Distance between grid points (meters)
        self.declare_parameter('inflation_radius', 0.8)    # Collision clearance radius around obstacles (meters)
        self.declare_parameter('use_local_frame', True)    # Generate grid relative to robot's current pose
        
        # Timing & Threshold parameters
        self.declare_parameter('goal_timeout', 25.0)      # Max duration (seconds) spent per grid point
        self.declare_parameter('reach_threshold', 0.8)    # Distance (meters) to declare success
        self.declare_parameter('publish_rate', 1.0)       # Frequency (Hz) to republish active goal to FAR Planner
        
        # Read parameters
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.spacing = self.get_parameter('spacing').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.reach_threshold = self.get_parameter('reach_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_local_frame = self.get_parameter('use_local_frame').value

        self.get_logger().info(
            f"\n=======================================================\n"
            f"Grid Goal Manager Initialized!\n"
            f"  - Local Frame Generation: {self.use_local_frame}\n"
            f"  - Bounding Box: X [{self.x_min}, {self.x_max}], Y [{self.y_min}, {self.y_max}]\n"
            f"  - Spacing: {self.spacing}m | Inflation: {self.inflation_radius}m\n"
            f"  - Goal Timeout: {self.goal_timeout}s | Success Radius: {self.reach_threshold}m\n"
            f"======================================================="
        )

        # TF Setup for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.far_goal_pub = self.create_publisher(PointStamped, '/far_planner/goal_point', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/state_estimation', self.on_odom, 10)
        self.poly_sub = self.create_subscription(
            MarkerArray, '/viz_poly_topic', self.on_polygons, 10)
        self.clear_sub = self.create_subscription(
            Empty, '/clear_goals', self.on_clear, 10)
        self.reset_graph_sub = self.create_subscription(
            Empty, '/reset_visibility_graph', self.on_clear, 10)
            
        # State variables
        self.latest_odom = None
        self.latest_segments = []
        self.raw_grid_points = []
        self.goal_queue = []
        
        self.current_goal = None
        self.current_goal_start_time = None
        self.last_pub_time = self.get_clock().now()
        
        self.grid_generated = False
        self.grid_started = False
        self.search_completed = False

        # Generate initial lattice points only if not using local frame
        if not self.use_local_frame:
            self.generate_raw_grid()
        
        # Control Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def generate_raw_grid(self):
        self.raw_grid_points = []
        curr_x = self.x_min
        while curr_x <= self.x_max:
            curr_y = self.y_min
            while curr_y <= self.y_max:
                p = Point()
                p.x = float(curr_x)
                p.y = float(curr_y)
                p.z = 0.0 # Set height dynamically from odometry
                self.raw_grid_points.append(p)
                curr_y += self.spacing
            curr_x += self.spacing
        self.get_logger().info(f"Generated raw grid containing {len(self.raw_grid_points)} lattice nodes.")
        self.grid_generated = True

    def on_odom(self, msg):
        self.latest_odom = msg

        # Generate grid relative to robot's current pose on first odom if using local frame
        if not self.grid_generated and self.use_local_frame:
            self.generate_local_grid(msg)

        # Align grid height to robot's altitude on the first message if using global raw grid
        elif self.grid_generated and not self.use_local_frame and len(self.raw_grid_points) > 0 and self.raw_grid_points[0].z == 0.0:
            for p in self.raw_grid_points:
                p.z = msg.pose.pose.position.z

        # Start grid path sequence once we have odometry (no need to wait for obstacles)
        if not self.grid_started and not self.search_completed and self.grid_generated:
            self.build_path_sequence()

    def generate_local_grid(self, msg):
        robot_pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        # Compute yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.raw_grid_points = []
        curr_x = self.x_min
        while curr_x <= self.x_max:
            curr_y = self.y_min
            while curr_y <= self.y_max:
                p = Point()
                # Rotate and translate relative to the robot
                p.x = robot_pos.x + curr_x * math.cos(yaw) - curr_y * math.sin(yaw)
                p.y = robot_pos.y + curr_x * math.sin(yaw) + curr_y * math.cos(yaw)
                p.z = robot_pos.z
                self.raw_grid_points.append(p)
                curr_y += self.spacing
            curr_x += self.spacing
            
        self.get_logger().info(
            f"Generated local grid relative to robot pose containing {len(self.raw_grid_points)} nodes.\n"
            f"  - Robot Position: ({robot_pos.x:.2f}, {robot_pos.y:.2f}, {robot_pos.z:.2f}), Yaw: {yaw:.2f} rad\n"
            f"  - Local Bounds: X [{self.x_min}, {self.x_max}], Y [{self.y_min}, {self.y_max}]"
        )
        self.grid_generated = True

    def on_polygons(self, msg):
        segments = []
        for marker in msg.markers:
            if marker.ns in ["global_contour", "unmatched_contour"]:
                pts = marker.points
                for i in range(0, len(pts) - 1, 2):
                    if i + 1 < len(pts):
                        segments.append((pts[i], pts[i+1]))
        self.latest_segments = segments

    def build_path_sequence(self):
        self.get_logger().info("Filtering grid against obstacle polygons...")
        
        # 1. Filter out points too close to boundary segments
        clean_points = []
        for p in self.raw_grid_points:
            in_collision = False
            for seg in self.latest_segments:
                A = seg[0]
                B = seg[1]
                
                # Math: Minimum distance from point P to line segment AB
                vx = B.x - A.x
                vy = B.y - A.y
                vz = B.z - A.z
                
                wx = p.x - A.x
                wy = p.y - A.y
                wz = p.z - A.z
                
                v_dot_v = vx*vx + vy*vy + vz*vz
                if v_dot_v < 1e-6:
                    # Segment is effectively a point
                    dist = math.sqrt(wx*wx + wy*wy + wz*wz)
                else:
                    t = (wx*vx + wy*vy + wz*vz) / v_dot_v
                    t = max(0.0, min(1.0, t))
                    cx = A.x + t * vx
                    cy = A.y + t * vy
                    cz = A.z + t * vz
                    dist = math.sqrt((p.x - cx)**2 + (p.y - cy)**2 + (p.z - cz)**2)
                    
                if dist < self.inflation_radius:
                    in_collision = True
                    break
            if not in_collision:
                clean_points.append(p)

        self.get_logger().info(f"Filtering complete. Navigable nodes: {len(clean_points)}/{len(self.raw_grid_points)}")
        
        if not clean_points:
            self.get_logger().warn("No navigable points found inside the bounding box. Try expanding boundaries or reducing spacing.")
            return

        # 2. Sort the nodes using a Nearest Neighbor solver starting from the robot
        self.get_logger().info("Sequencing optimal path...")
        sorted_points = []
        curr_pose = self.latest_odom.pose.pose.position
        while clean_points:
            next_pt = min(clean_points, key=lambda p: math.sqrt((curr_pose.x - p.x)**2 + (curr_pose.y - p.y)**2 + (curr_pose.z - p.z)**2))
            sorted_points.append(next_pt)
            clean_points.remove(next_pt)
            curr_pose = next_pt

        # 3. Create PointStamped goal queue in "map" frame so FAR Planner processes it correctly
        self.goal_queue = []
        for pt in sorted_points:
            g = PointStamped()
            g.header.frame_id = "map"
            g.header.stamp = self.get_clock().now().to_msg()
            g.point = pt
            self.goal_queue.append(g)

        self.grid_started = True
        self.start_next_goal()

    def on_clear(self, msg):
        self.get_logger().info("Cancelling grid search sequence.")
        self.goal_queue.clear()
        self.current_goal = None
        self.grid_started = False
        self.grid_generated = False
        self.search_completed = False
        
        # Command FAR Planner to hold position
        if self.latest_odom:
            cancel_goal = PointStamped()
            cancel_goal.header = self.latest_odom.header
            cancel_goal.header.frame_id = "multi_goal_manager_cancel"
            cancel_goal.point = self.latest_odom.pose.pose.position
            self.far_goal_pub.publish(cancel_goal)

    def transform_point(self, point_msg, target_frame):
        if point_msg.header.frame_id == target_frame:
            return point_msg
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                point_msg.header.frame_id,
                rclpy.time.Time()
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
            qx = tf.transform.rotation.x
            qy = tf.transform.rotation.y
            qz = tf.transform.rotation.z
            qw = tf.transform.rotation.w
            
            px = point_msg.point.x
            py = point_msg.point.y
            pz = point_msg.point.z
            
            vx = (1.0 - 2.0*qy*qy - 2.0*qz*qz)*px + 2.0*(qx*qy - qz*qw)*py + 2.0*(qx*qz + qy*qw)*pz
            vy = 2.0*(qx*qy + qz*qw)*px + (1.0 - 2.0*qx*qx - 2.0*qz*qz)*py + 2.0*(qy*qz - qx*qw)*pz
            vz = 2.0*(qx*qz - qy*qw)*px + 2.0*(qy*qz + qx*qw)*py + (1.0 - 2.0*qx*qx - 2.0*qy*qy)*pz
            
            output = PointStamped()
            output.header.frame_id = target_frame
            output.header.stamp = point_msg.header.stamp
            output.point.x = vx + tx
            output.point.y = vy + ty
            output.point.z = vz + tz
            return output
        except Exception as e:
            return point_msg

    def start_next_goal(self):
        if not self.goal_queue:
            self.current_goal = None
            self.grid_started = False
            self.search_completed = True
            self.get_logger().info("Grid search completed successfully!")
            return
            
        self.current_goal = self.goal_queue.pop(0)
        self.current_goal_start_time = self.get_clock().now()
        self.last_pub_time = self.get_clock().now()
        
        self.get_logger().info(
            f"Starting grid node ({len(self.goal_queue)} left): "
            f"({self.current_goal.point.x:.2f}, {self.current_goal.point.y:.2f}, {self.current_goal.point.z:.2f})"
        )
        self.far_goal_pub.publish(self.current_goal)

    def transition_to_next_goal(self):
        self.current_goal = None
        if self.goal_queue:
            self.start_next_goal()
        else:
            self.get_logger().info("Grid search queue finished.")
            self.grid_started = False
            self.search_completed = True

    def is_point_occluded(self, p):
        # Checks if a point has become occluded by active segments
        for seg in self.latest_segments:
            A = seg[0]
            B = seg[1]
            vx, vy, vz = B.x - A.x, B.y - A.y, B.z - A.z
            wx, wy, wz = p.x - A.x, p.y - A.y, p.z - A.z
            v_dot_v = vx*vx + vy*vy + vz*vz
            if v_dot_v < 1e-6:
                dist = math.sqrt(wx*wx + wy*wy + wz*wz)
            else:
                t = max(0.0, min(1.0, (wx*vx + wy*vy + wz*vz) / v_dot_v))
                cx = A.x + t * vx
                cy = A.y + t * vy
                cz = A.z + t * vz
                dist = math.sqrt((p.x - cx)**2 + (p.y - cy)**2 + (p.z - cz)**2)
            if dist < self.inflation_radius:
                return True
        return False

    def check_dynamic_occlusion(self):
        # Prune queued waypoints that have become occluded by newly mapped obstacles
        i = 0
        while i < len(self.goal_queue):
            if self.is_point_occluded(self.goal_queue[i].point):
                self.get_logger().info(
                    f"Pruning occluded grid node: "
                    f"({self.goal_queue[i].point.x:.2f}, {self.goal_queue[i].point.y:.2f})"
                )
                self.goal_queue.pop(i)
            else:
                i += 1

    def control_loop(self):
        # Read parameters dynamically
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.reach_threshold = self.get_parameter('reach_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.inflation_radius = self.get_parameter('inflation_radius').value

        if self.current_goal:
            if not self.latest_odom:
                return
                
            # 1. Check if the active goal was dynamically blocked
            if self.is_point_occluded(self.current_goal.point):
                self.get_logger().warn("Active goal node is now occluded! Skipping immediately.")
                self.transition_to_next_goal()
                return

            # 2. Transform the active goal to the odometry frame for distance checks
            target_frame = self.latest_odom.header.frame_id
            goal_in_odom = self.transform_point(self.current_goal, target_frame)

            # 3. Check distance for success range
            dx = self.latest_odom.pose.pose.position.x - goal_in_odom.point.x
            dy = self.latest_odom.pose.pose.position.y - goal_in_odom.point.y
            dz = self.latest_odom.pose.pose.position.z - goal_in_odom.point.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            elapsed = (self.get_clock().now() - self.current_goal_start_time).nanoseconds / 1e9
            
            # Periodically republish
            time_since_pub = (self.get_clock().now() - self.last_pub_time).nanoseconds / 1e9
            if time_since_pub >= (1.0 / self.publish_rate):
                self.far_goal_pub.publish(self.current_goal)
                self.last_pub_time = self.get_clock().now()
                
            if dist < self.reach_threshold:
                self.get_logger().info(f"Node reached! Success radius: {dist:.2f}m")
                self.transition_to_next_goal()
            elif elapsed > self.goal_timeout:
                self.get_logger().info(f"Node timed out! Time spent: {elapsed:.1f}s")
                self.transition_to_next_goal()
        else:
            if self.grid_started and self.goal_queue:
                self.start_next_goal()

        # Dynamic occlusion pruning for pending queue
        if self.grid_started and self.goal_queue:
            self.check_dynamic_occlusion()
                
        self.publish_markers()

    def publish_markers(self):
        msg = MarkerArray()
        
        # Delete old markers
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
        
        # Path line connecting nodes
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
            # Transform for marker visualization
            goal_in_odom = self.transform_point(self.current_goal, target_frame)
            line.points.append(goal_in_odom.point)
            
            # Active node sphere (Green/Yellow)
            sphere = Marker()
            sphere.header.frame_id = target_frame; sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "multi_goal"; sphere.id = marker_id; marker_id += 1
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position = goal_in_odom.point; sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.5; sphere.scale.y = 0.5; sphere.scale.z = 0.5
            sphere.color.r = 1.0; sphere.color.g = 0.85; sphere.color.b = 0.0; sphere.color.a = 0.9
            msg.markers.append(sphere)
            
            # Active node text
            text = Marker()
            text.header.frame_id = target_frame; text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "multi_goal"; text.id = marker_id; marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position = Point(x=goal_in_odom.point.x, y=goal_in_odom.point.y, z=goal_in_odom.point.z + 0.6)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.35
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            
            elapsed = (self.get_clock().now() - self.current_goal_start_time).nanoseconds / 1e9
            time_left = max(0.0, self.goal_timeout - elapsed)
            text.text = f"Grid Node (T-min: {time_left:.1f}s)"
            msg.markers.append(text)
            
        for i, goal in enumerate(self.goal_queue):
            q_odom = self.transform_point(goal, target_frame)
            line.points.append(q_odom.point)
            
            # Queued spheres (Cyan)
            sphere = Marker()
            sphere.header.frame_id = target_frame; sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "multi_goal"; sphere.id = marker_id; marker_id += 1
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position = q_odom.point; sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.35; sphere.scale.y = 0.35; sphere.scale.z = 0.35
            sphere.color.r = 0.0; sphere.color.g = 0.9; sphere.color.b = 0.9; sphere.color.a = 0.7
            msg.markers.append(sphere)
            
            # Queued text
            text = Marker()
            text.header.frame_id = target_frame; text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "multi_goal"; text.id = marker_id; marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position = Point(x=q_odom.point.x, y=q_odom.point.y, z=q_odom.point.z + 0.5)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = 0.9; text.color.g = 0.9; text.color.b = 0.9; text.color.a = 0.9
            text.text = f"P{i+1}"
            msg.markers.append(text)
            
        if len(line.points) > 1:
            msg.markers.append(line)
            
        self.marker_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GridGoalManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
