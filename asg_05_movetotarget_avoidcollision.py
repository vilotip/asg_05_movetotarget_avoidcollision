#!/usr/bin/env python

import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
import math


TARGET_X = 6.0
TARGET_Y = 0.0

CLEARANCE_MARGIN = 0.8   
SCAN_SPEED       = 0.15  

# state machine
STATE_SCANNING            = 0
STATE_NAVIGATING_WAYPOINT = 1
STATE_NAVIGATING_GOAL     = 2

current_state = STATE_SCANNING
waypoint_x, waypoint_y = None, None
latest_range = float('inf')

def range_cb(msg):
    global latest_range
    latest_range = msg.range

def robot_pose_fn(listener):
    try:
        (trans, rot) = listener.lookupTransform("/odom", "/base_footprint", rospy.Time(0))
        return trans[0], trans[1], euler_from_quaternion(rot)[2]
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return None, None, None

def align_robot_yaw(target_yaw, pub, listener, cmd_msg):
    print(f"Aligning heading to {target_yaw:.2f} rad...")
    
    while not rospy.is_shutdown():
        rx, ry, ryaw = robot_pose_fn(listener)
        if ryaw is None:
            rospy.sleep(0.05)
            continue

        yaw_err = math.atan2(math.sin(target_yaw - ryaw), math.cos(target_yaw - ryaw))

        if abs(yaw_err) <= 0.05:
            break

        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.2 if yaw_err > 0 else -0.2
        pub.publish(cmd_msg)
        rospy.sleep(0.05)

    cmd_msg.angular.z = 0.0
    pub.publish(cmd_msg)
    rospy.sleep(0.2)
    
def edge_detection(scan_steps, scan_speed, pub, listener, cmd_msg, is_left_pass):
    global latest_range
    cmd_msg.angular.z = scan_speed
    
    # check if facing wall or open space at time=0s
    is_initial_inf = math.isinf(latest_range) or latest_range <= 0.1
    INF_THRESHOLD_STEPS = 5 
    inf_counter = INF_THRESHOLD_STEPS if is_initial_inf else 0
    was_open_space = is_initial_inf

    prev_obs_x, prev_obs_y = None, None
    detected_edge_x, detected_edge_y = None, None
    edge_found = False
    edge_type = None  # "leading" vs "trailing" edge

    for _ in range(scan_steps):
        pub.publish(cmd_msg)
        rospy.sleep(0.1)
        rx, ry, ryaw = robot_pose_fn(listener)
        if rx is None: 
            continue

        is_current_inf = math.isinf(latest_range) or latest_range <= 0.1

        if is_current_inf: # open facing
            inf_counter += 1
            if inf_counter >= INF_THRESHOLD_STEPS: # ignore hole in the wall
                # TRAILING EDGE: obstacle -> open space
                if not was_open_space and prev_obs_x is not None and not edge_found:
                    detected_edge_x, detected_edge_y = prev_obs_x, prev_obs_y
                    edge_found = True
                    edge_type = "trailing"
                    print(f"Detected Trailing Edge (not_inf -> inf): ({detected_edge_x:.2f}, {detected_edge_y:.2f})")
                
                was_open_space = True
        else: # obstacle facing
            current_obs_x = rx + latest_range * math.cos(ryaw)
            current_obs_y = ry + latest_range * math.sin(ryaw)

            # LEADING EDGE: was open space -> obstacle
            if was_open_space and not edge_found:
                detected_edge_x, detected_edge_y = current_obs_x, current_obs_y
                edge_found = True
                edge_type = "leading"
                print(f"Detected Leading Edge (inf -> not_inf): ({detected_edge_x:.2f}, {detected_edge_y:.2f})")

            prev_obs_x, prev_obs_y = current_obs_x, current_obs_y
            was_open_space = False
            inf_counter = 0

    # tangent angle calculation (always project into open area)
    if edge_found and detected_edge_x is not None:
        rx, ry, _ = robot_pose_fn(listener)
        edge_angle = math.atan2(detected_edge_y - ry, detected_edge_x - rx)
        
        if edge_type == "trailing":
            # Shift toward open space (+pi/2 for Left sweep, -pi/2 for Right sweep)
            tangent_angle = edge_angle + (math.pi / 2.0 if is_left_pass else -math.pi / 2.0)
        else:
            # LEADING EDGE: Invert angle to project back into open space (-pi/2 for Left sweep, +pi/2 for Right sweep)
            tangent_angle = edge_angle + (-math.pi / 2.0 if is_left_pass else math.pi / 2.0)
        
        # waypoint calculation based on tangent angle
        wp_x = detected_edge_x + CLEARANCE_MARGIN * math.cos(tangent_angle)
        wp_y = detected_edge_y + CLEARANCE_MARGIN * math.sin(tangent_angle)
        return wp_x, wp_y, detected_edge_x, detected_edge_y

    return None, None, None, None
    
def run_scan(pub, listener, cmd_msg):
    # 1. SCAN LEFT
    left_wp_x, left_wp_y, left_e_x, left_e_y = edge_detection(
        scan_steps=120, scan_speed=SCAN_SPEED, pub=pub, listener=listener, cmd_msg=cmd_msg, is_left_pass=True
    )
    
    # 2. RETURN TO CENTER 
    cmd_msg.angular.z = -SCAN_SPEED
    for _ in range(120):
        pub.publish(cmd_msg)
        rospy.sleep(0.1)
        
    # 3. SCAN RIGHT
    right_wp_x, right_wp_y, right_e_x, right_e_y = edge_detection(
        scan_steps=120, scan_speed=-SCAN_SPEED, pub=pub, listener=listener, cmd_msg=cmd_msg, is_left_pass=False
    )

    # 4. RETURN TO CENTER 
    cmd_msg.angular.z = SCAN_SPEED
    for _ in range(120):
        pub.publish(cmd_msg)
        rospy.sleep(0.1)

    cmd_msg.angular.z = 0.0
    pub.publish(cmd_msg)

    # select optimal direction - closest edge waypoint
    rx, ry, _ = robot_pose_fn(listener)
    left_dist = math.sqrt((left_e_x - rx)**2 + (left_e_y - ry)**2) if left_e_x else -1.0
    right_dist = math.sqrt((right_e_x - rx)**2 + (right_e_y - ry)**2) if right_e_x else -1.0

    if left_dist > right_dist and left_wp_x is not None:
        print(f"Selected Left Waypoint: ({left_wp_x:.2f}, {left_wp_y:.2f})")
        return left_wp_x, left_wp_y
    elif right_wp_x is not None:
        print(f"Selected Right Waypoint: ({right_wp_x:.2f}, {right_wp_y:.2f})")
        return right_wp_x, right_wp_y
    else:
        return None, None

# --- Main ROS Setup ---
rospy.init_node("movetotarget_avoidcollision")
listener = tf.TransformListener()
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
rospy.Subscriber("/range", Range, range_cb)
msg = Twist()
rospy.sleep(1.0)

# Main Navigation Loop
while not rospy.is_shutdown():
    rx, ry, robot_yaw = robot_pose_fn(listener)
    if rx is None:
        rospy.sleep(0.05)
        continue

    # =========================================================================
    # STATE 1: SCANNING LOOP
    # =========================================================================
    if current_state == STATE_SCANNING:
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        pub.publish(msg)

        align_robot_yaw(0.0, pub, listener, msg)
        
        found_wp_x, found_wp_y = run_scan(pub, listener, msg)

        if found_wp_x is not None and found_wp_y is not None:
            waypoint_x, waypoint_y = found_wp_x, found_wp_y
            print(f"Waypoint registered: ({waypoint_x:.2f}, {waypoint_y:.2f})")
            current_state = STATE_NAVIGATING_WAYPOINT
        else:
            print("No blocking edges found. Driving to Goal.")
            waypoint_x, waypoint_y = None, None
            current_state = STATE_NAVIGATING_GOAL

    # =========================================================================
    # STATE 2 & 3: DRIVING
    # =========================================================================
    else:
        if current_state == STATE_NAVIGATING_WAYPOINT:
            curr_tx, curr_ty = waypoint_x, waypoint_y
        else:
            curr_tx, curr_ty = TARGET_X, TARGET_Y

        dx = curr_tx - rx
        dy = curr_ty - ry
        distance = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(angle_rad - robot_yaw), math.cos(angle_rad - robot_yaw))

        if distance > 0.15:
            if abs(yaw_error) > 0.1:
                msg.linear.x = 0.0
                msg.angular.z = 0.2 if yaw_error > 0 else -0.2
            else:
                msg.linear.x = 0.2
                msg.angular.z = 0.0
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            pub.publish(msg)

            if current_state == STATE_NAVIGATING_WAYPOINT:
                print("Reached waypoint. Re-aligning and scanning again...")
                current_state = STATE_SCANNING
            elif current_state == STATE_NAVIGATING_GOAL:
                print("SUCCESS: Target reached!")
                break

        pub.publish(msg)
        rospy.sleep(0.1)
