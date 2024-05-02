# Global dictionary to hold environment data
import math
import random
import numpy as np
import gym
from gym import spaces
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Quaternion, Pose, PoseArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gym.envs.registration import register
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel, SetModelState, GetModelState, GetModelStateRequest


class RobotGymEnv(gym.Env):
    def __init__(self, namespace='robot1', center_pos=(0,0)):

        self.center = center_pos
        self.namespace = namespace
        self.max_steps = 500
        self.current_step = 0
        self.boxes = []
        
        self.observation_space = spaces.Box(
            low=np.array([0.0]*30 + [-1.0, 0.0]),
            high=np.array([1.0]*30 + [1.0, 1.0]),
            dtype=np.float32
        )


        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),  # Min angle, min velocity
            high=np.array([1.0, 1.5]),  # Max angle, max velocity
            dtype=np.float32
        )

        
        self.latest_scan = None
        self.latest_odom = None
        self.goal_distance = 0.0
        self.goal_angle = 0.0
        #self.spawn_goal_marker((0,-1))
        self.goal_position = self.generate_goal_position()

        # ROS Publishers and Subscribers with namespace
        
        self.cmd_vel_publisher = rospy.Publisher(f'/{self.namespace}/{self.namespace}/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber(f'/{self.namespace}/{self.namespace}/scan', LaserScan, self.scan_callback)
        rospy.Subscriber(f'/{self.namespace}/{self.namespace}/odom', Odometry, self.odom_callback)
        

        self.reset()
        
        
    def box_positions(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        temp = []
        for i in range(10):  # Assuming you have 10 boxes as mentioned
            model_name = f'box{i}_{self.namespace}'
            try:
                # Prepare the request
                model_state_request = GetModelStateRequest(model_name, 'world')
                
                # Call the service and get the response
                model_state_response = get_model_state_service(model_state_request)
                
                if model_state_response.success:
                    # Store pose information in the dictionary
                    temp.append(model_state_response.pose)
                else:
                    rospy.logwarn(f"Failed to get model state for {model_name}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        self.boxes = temp
            
        
    def generate_goal_position(self):
        self.box_positions()
        
        max_attempts = 1000
        for _ in range(max_attempts):
            #print('Generating a goal position...')
            # Generate a random angle and radius within 4.5 meters
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, 2.0)
            goal_x = self.center[0] + radius * math.cos(angle)
            goal_y = self.center[1] + radius * math.sin(angle)
            goal_position = (goal_x, goal_y)

            # Check that the goal position is not within 0.2 meters of any box
            if all(self.distance(goal_position, (box.position.x, box.position.y)) >= 0.2 for box in self.boxes):
                print("Valid goal position found: ", goal_position)
                return goal_position

        rospy.logwarn("Failed to find a suitable goal position after %d attempts", max_attempts)
        return None  # Return None if no valid position is found after max_attempts


    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def scan_callback(self, data):
        # Preprocess laser scan data to match the paper's setup
        scan_ranges = np.array(data.ranges)
        num_bins = 30
        bin_size = len(scan_ranges) // num_bins
        binned_ranges = [min(scan_ranges[i:i + bin_size]) for i in range(0, len(scan_ranges), bin_size)]
        self.latest_scan = np.array(binned_ranges) / data.range_max  # Normalize distances
    

    def odom_callback(self, msg):
        # Update robot's position and compute goal distance and angle
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.latest_odom = np.array([position.x, position.y, yaw])
        
        goal_x, goal_y = self.goal_position
        robot_x, robot_y = position.x, position.y
        
        self.goal_distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2) / 20.0  # Normalize distance
        goal_angle = np.arctan2(goal_y - robot_y, goal_x - robot_x)
        self.goal_angle = (goal_angle - yaw) / np.pi  # Normalize angle

    def step(self, action):
        # Apply action
        cmd = Twist()
        cmd.linear.x = action[1]
        cmd.angular.z = action[0]
        self.cmd_vel_publisher.publish(cmd)

        # Assume a short delay for action execution
        rospy.sleep(0.1)

        # Increment step counter and check for max steps
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Assemble state: laser scan + goal distance and angle
        
        full_state = np.concatenate((self.latest_scan, [self.goal_distance, self.goal_angle]))
        

        # Compute reward and check if goal is reached
        reward, done = self.compute_reward_and_done(done)

        return full_state, reward, done, {}

    def compute_reward_and_done(self, done):
        # Check for collision (laser scan values close to zero)
        
        
        if np.any(self.latest_scan < 0.015):  # Adjust threshold as needed
            return -50.0, True  # Collision penalty and end episode

        # Check if goal is reached
        if self.goal_distance < 0.03:  # Adjust threshold as needed
            return 100.0, True  # Reward and end episode

        # No reward or penalty for other cases
        return 0.0, done

    def reset(self):
        # Reset step counter
        self.current_step = 0
        # print("In Robot env: " + str(self.environments))
        # Reset goal position
        # Reset simulation and robot state
        # self.reset_simulation()  # Resets the simulation
        self.reset_environment(self.namespace) # Resets the world to its initial state
        rospy.sleep(0.1)
        # Get initial observation
        self.goal_position = self.generate_goal_position()
       #self.move_goal_marker(self.goal_position)
        initial_observation = self.get_initial_observation()
        
        return initial_observation

    def get_initial_observation(self):
        # Wait for fresh sensor data
        
        while self.latest_scan is None or self.latest_odom is None:
            rospy.sleep(0.1)  # Wait for callbacks to receive data


        # Combine laser scan data and odometry for the initial observation
        initial_observation = np.concatenate([self.latest_scan, [self.goal_distance, self.goal_angle]])

        # Reset placeholders for the next reset cycle
        self.latest_scan = None
        self.latest_odom = None

        return initial_observation
    
    def get_namespace(self):
        return self.namespace
    
    # def spawn_goal_marker(self, position):
    #     # Ensure the service is available
    #     rospy.wait_for_service('/gazebo/spawn_sdf_model')
    #     try:
    #         spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #         with open("/home/easz/catkin_ws/src/testing_pkg/urdf/marker.sdf", 'r') as file:
    #             goal_sdf = file.read()

    #         pose = Pose()
    #         pose.position.x = position[0]
    #         pose.position.y = position[1]
    #         pose.position.z = 0.01  # slightly above the ground to ensure visibility

    #         # Spawn the model
    #         spawn_model(f"goal_marker_{self.namespace}", goal_sdf, "", pose, "world")
    #     except rospy.ServiceException as e:
    #         rospy.logerr("Model spawn service call failed: %s", e)

    # def move_goal_marker(self, position):
    #     # Move the existing marker to a new position
    #     rospy.wait_for_service('/gazebo/set_model_state')
    #     try:
    #         set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    #         state = ModelState()
    #         state.model_name = f"goal_marker_{self.namespace}"
    #         state.pose.position.x = position[0]
    #         state.pose.position.y = position[1]
    #         state.pose.position.z = 0.01
    #         set_state(state)
    #     except rospy.ServiceException as e:
    #         rospy.logerr("Model set state service call failed: %s", e)

    # def delete_goal_marker(self):
    #     # Delete the marker if needed
    #     rospy.wait_for_service('/gazebo/delete_model')
    #     try:
    #         delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    #         delete_model(f"goal_marker_{self.namespace}")
    #     except rospy.ServiceException as e:
    #         rospy.logerr("Model delete service call failed: %s", e)

    def angle_to_quaternion(self,yaw):
        """Convert a yaw angle (in radians) to a quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(x=0, y=0, z=sy, w=cy)

    def reset_environment(self,namespace):
        
        center_x, center_y = self.center
        
        # Prepare new random positions for the robot and boxes relative to the center
        new_positions = []

        # Generate new position for the robot
        robot_new_x = random.uniform(center_x - 2, center_x + 2)
        robot_new_y = random.uniform(center_y - 2, center_y + 2)
        new_positions.append((robot_new_x, robot_new_y))

        # Generate new positions for the boxes
        for _ in range(0,10):
            box_new_x = random.uniform(center_x - 2, center_x + 2)
            box_new_y = random.uniform(center_y - 2, center_y + 2)
            new_positions.append((box_new_x, box_new_y))

        # Check for overlaps
        if not self.positions_are_valid(new_positions):
            return self.reset_environment(namespace)  # Recursively retry until a valid configuration is found


        # If no overlaps, proceed to update positions
        robot_state = ModelState()
        robot_state.model_name = namespace
        robot_state.pose.position.x = new_positions[0][0]
        robot_state.pose.position.y = new_positions[0][1]
        robot_state.pose.position.z = 0.06  # Adjust height if necessary

        # Generate a random yaw angle (in radians) and convert it to a quaternion
        yaw_angle = random.uniform(0, 2 * math.pi)  # Random angle from 0 to 360 degrees
        robot_state.pose.orientation = self.angle_to_quaternion(yaw_angle)
        
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Reset robot position
        robot_resp = set_state(robot_state)
        if not robot_resp.success:
            rospy.logerr("Failed to reset robot position for %s", namespace)
            return False
        
        # Reset positions of boxes
        for idx in range(1,11):
            box_state = ModelState()
            box_state.model_name = f'box{idx-1}_{namespace}'
            box_state.pose.position.x = new_positions[idx][0]
            box_state.pose.position.y = new_positions[idx][1]
            box_state.pose.position.z = 0  # Ground level
            box_resp = set_state(box_state)
            if not box_resp.success:
                rospy.logerr("Failed to reset position for box %s", f'box{idx-1}_{namespace}')
                return False
        rospy.loginfo("Successfully reset environment for %s", namespace)

        # goal_state = ModelState()
        # goal_state.model_name = f"goal_marker_{self.namespace}"
        # goal_state.pose.position.x = self.goal_position[0]
        # goal_state.pose.position.y = self.goal_position[1]
        # goal_state.pose.position.z = 0.01
        # goal_rsp = set_state(goal_state)
        # if not box_resp.success:
        #         rospy.logerr("Failed to reset position for goal object %s", goal_state.model_name)
        return True
        

    def positions_are_valid(self, positions):
        """Check if there are any overlapping positions."""
        # You can define a threshold distance below which two objects are considered to be overlapping
        threshold = 0.5  # meters
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    if ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5 < threshold:
                        return False
        return True
    
register(
    id='RobotGym-v1',
    entry_point='multithread_env:RobotGymEnv',
)