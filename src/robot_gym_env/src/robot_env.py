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
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from gym.envs.registration import register


class RobotGymEnv(gym.Env):
    def __init__(self):
        rospy.init_node('robot_gym_node', anonymous=True)
        
        self.goal_distance = 0.0  # Initialize goal distance
        self.goal_angle = 0.0     # Initialize goal angle
        
        # Observation space: laser scan ranges plus distance and angle to the goal
        num_laser_bins = 720
        self.observation_space = spaces.Box(
            low=np.array([0.0]*num_laser_bins + [0.0, -np.pi]),
            high=np.array([np.inf]*num_laser_bins + [np.inf, np.pi]),
            dtype=np.float32
        )

        # Action space: linear and angular velocities
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([0.5, 1.0]),
            dtype=np.float32
        )

        self.goal_position = self.generate_goal_position()
        self.latest_scan = None  # Placeholder for the latest laser scan data
        self.latest_odom = None  # Placeholder for the latest odometry data

        # ROS Publishers and Subscribers
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # Wait for the service to reset the world
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.state = None


    def generate_goal_position(self):
        # Implement logic to generate goal position
        # This is just an example
        goal_x = np.random.uniform(-10, 10)
        goal_y = np.random.uniform(-10, 10)
        return (goal_x, goal_y)

    def scan_callback(self, data):
        # Preprocess laser scan data
        self.latest_scan = np.array(data.ranges) / data.range_max  # Normalize distances

    def odom_callback(self, msg):
        # Update robot's position and compute goal distance and angle
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.latest_odom = np.array([position.x, position.y, yaw])
        
        goal_x, goal_y = self.goal_position
        robot_x, robot_y = position.x, position.y
        
        self.goal_distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        goal_angle = np.arctan2(goal_y - robot_y, goal_x - robot_x)
        self.goal_angle = goal_angle - yaw

    def step(self, action):
        # Apply action
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.cmd_vel_publisher.publish(cmd)
        
        # Assume a short delay for action execution
        rospy.sleep(0.1)
        
        # Assemble state: laser scan + goal distance and angle
        full_state = np.append(self.state, [self.goal_distance, self.goal_angle])
        
        # Compute reward and check if goal is reached
        reward, done = self.compute_reward_and_done()
        
        return full_state, reward, done, {}


    def get_initial_observation(self):
        # Wait for fresh sensor data
        while self.latest_scan is None or self.latest_odom is None:
            rospy.sleep(0.1)  # Wait for callbacks to receive data

        # Combine laser scan data and odometry for the initial observation
        initial_observation = np.concatenate([self.latest_scan, self.latest_odom])

        # Reset placeholders for the next reset cycle
        self.latest_scan = None
        self.latest_odom = None

        return initial_observation


    def reset(self):
        # Define the desired initial state
        model_state = ModelState()
        model_state.model_name = 'robot'  # Replace with your robot's model name
        model_state.pose.position.x = 0.0
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.0  
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0  # Neutral orientation

        # Call the service to set the robot's state
        try:
            resp = self.set_model_state(model_state)
            if not resp.success:
                rospy.logerr("Failed to set model state in Gazebo: " + resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        try:
            self.reset_simulation()  # Resets the simulation
            # self.reset_world()  # Optionally, reset the world to its initial state without affecting the simulation time
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        
        # Return the initial observation
        initial_observation = self.get_initial_observation()
        return initial_observation



    def compute_reward_and_done(self):
        # Implement your reward function based on the goal distance, goal angle, and laser data
        # For simplicity, we'll give a high reward for being close to the goal and penalize based on distance
        
        if self.goal_distance < 0.2:  # If within 20 cm of the goal
            return 100.0, True  # Reward, Done
        
        # Penalize based on distance to goal and angle
        reward = -self.goal_distance - abs(self.goal_angle)
        return reward, False


    

