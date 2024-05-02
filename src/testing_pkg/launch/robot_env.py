import random
import numpy as np
import gym
from gym import spaces
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gym.envs.registration import register
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel, SetModelState
from geometry_msgs.msg import Pose

from python_launch import reset_environment
from globals import environments

class RobotGymEnv(gym.Env):
    def __init__(self, namespace='robot1', center_pos=(0,0)):
        rospy.init_node(f'{namespace}_gym_node', anonymous=True)
        
        self.namespace = namespace
        self.max_steps = 500
        self.current_step = 0
        self.center = center_pos
        
        self.observation_space = spaces.Box(
            low=np.array([0.0]*30 + [-1.0, 0.0]),
            high=np.array([1.0]*30 + [1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(49)  # 7 * 7 possible actions

        self.goal_position = self.generate_goal_position()
        self.latest_scan = None
        self.latest_odom = None

        # ROS Publishers and Subscribers with namespace
        self.cmd_vel_publisher = rospy.Publisher(f'/{self.namespace}/{self.namespace}/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber(f'/{self.namespace}/{self.namespace}/scan', LaserScan, self.scan_callback)
        rospy.Subscriber(f'/{self.namespace}/{self.namespace}/odom', Odometry, self.odom_callback)

        self.reset()
        self.spawn_goal_marker(self.get_environment_center())

        

    def generate_goal_position(self):
        namespace = self.namespace  # Make sure the namespace for this gym instance is set correctly
        if namespace in environments:
            env = environments[namespace]
            center_x, center_y = env['center']
            
            while True:
                # Generate a random position around the center
                goal_x = random.uniform(center_x - 2, center_x + 2)
                goal_y = random.uniform(center_y - 2, center_y + 2)

                # Check distance from robot
                robot_x, robot_y = env['robot_position'][:2]
                if np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2) < 0.5:
                    continue  # Too close to the robot, retry

                # Check distance from each box
                too_close_to_box = False
                for box in env['boxes']:
                    box_x = box['pose'].position.x
                    box_y = box['pose'].position.y
                    if np.sqrt((goal_x - box_x)**2 + (goal_y - box_y)**2) < 0.5:
                        too_close_to_box = True
                        break

                if too_close_to_box:
                    continue  # Too close to a box, retry

                # If the goal position is not too close to any object
                self.move_goal_marker((goal_x, goal_y))
                return (goal_x, goal_y)
        else:
            # Default goal position if namespace is not found
            return (1.3478442430496216, -0.004217624664306641)
    
    def get_environment_center(self):
        # Retrieve center for the current namespace environment
        center = environments.get(self.namespace, {}).get('center', (0, 0))
        return center

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
        # Map discrete action to linear and angular velocities
        linear_vels = np.linspace(0.5, 1.5, 7)  # 7 linear velocities from -0.5 to 0.5
        angular_vels = np.linspace(-1.0, 1.0, 7)  # 7 angular velocities from -1.0 to 1.0
        linear_vel = linear_vels[action // 7]  # Integer division to get linear velocity index
        angular_vel = angular_vels[action % 7]  # Modulo to get angular velocity index

        # Apply action
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
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
        
        if np.any(self.latest_scan < 0.02):  # Adjust threshold as needed
            return -50.0, True  # Collision penalty and end episode

        # Check if goal is reached
        if self.goal_distance < 0.01:  # Adjust threshold as needed
            return 100.0, True  # Reward and end episode

        # No reward or penalty for other cases
        return 0.0, done

    def reset(self):
        # Reset step counter
        self.current_step = 0
        print("In Robot env: " + str(environments))
        # Reset goal position
        self.goal_position = self.generate_goal_position()

        # Reset simulation and robot state
        # self.reset_simulation()  # Resets the simulation
        reset_environment(self.namespace) # Resets the world to its initial state

        # Get initial observation
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
    
    def spawn_goal_marker(self, position):
        # Ensure the service is available
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            with open("/home/easz/catkin_ws/src/testing_pkg/urdf/marker.sdf", 'r') as file:
                goal_sdf = file.read()

            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = 0.01  # slightly above the ground to ensure visibility

            # Spawn the model
            spawn_model("goal_marker", goal_sdf, "", pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr("Model spawn service call failed: %s", e)

    def move_goal_marker(self, position):
        # Move the existing marker to a new position
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state = ModelState()
            state.model_name = "goal_marker"
            state.pose.position.x = position[0]
            state.pose.position.y = position[1]
            state.pose.position.z = 0.01

            set_state(state)
        except rospy.ServiceException as e:
            rospy.logerr("Model set state service call failed: %s", e)

    def delete_goal_marker(self):
        # Delete the marker if needed
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model("goal_marker")
        except rospy.ServiceException as e:
            rospy.logerr("Model delete service call failed: %s", e)

# Register the environment with Gym
register(
    id='RobotGym-v0',
    entry_point='robot_env:RobotGymEnv',
)
