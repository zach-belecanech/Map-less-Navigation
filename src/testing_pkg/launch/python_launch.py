#!/usr/bin/env python3

import math
import rospy
import json
import xacro
import roslaunch
import random
import os
import copy
from geometry_msgs.msg import Quaternion, Pose, PoseArray
from gazebo_msgs.srv import SpawnModel
from std_msgs.msg import String

# Global dictionary to hold environment data
envs = {}

def pose_to_dict(pose):
    """Convert a Pose to a dictionary."""
    return {
        'position': {
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z
        },
        'orientation': {
            'x': pose.orientation.x,
            'y': pose.orientation.y,
            'z': pose.orientation.z,
            'w': pose.orientation.w
        }
    }

def setup_publishers():
    """Setup publishers for each environment."""
    for ns in envs:
        pub = rospy.Publisher(f'/{ns}/box_positions', PoseArray, queue_size=10)
        pub2 = rospy.Publisher(f'/{ns}/env_data', String, queue_size=10)
        publish_positions(pub, pub2)

def publish_positions(pub, pub2):
    """Publish positions of boxes for each environment."""
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        for ns, data in envs.items():
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "world"

            for box in data['boxes']:
                pose_array.poses.append(box['pose'])
            pub.publish(pose_array)

            tdata = copy.deepcopy(data)
            tdata["boxes"] = []
            pub2.publish(json.dumps(tdata))
        rate.sleep()

def spawn_model(model_name, model_path, x, y, z, namespace):
    """Function to spawn SDF model in the Gazebo simulation under the specified namespace."""
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        with open(model_path, 'r') as model_file:
            model_xml = model_file.read()

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = Quaternion(x=0, y=0, z=0, w=1)  # Default orientation if not specified

        resp = spawn_service(model_name, model_xml, namespace, pose, "world")
        
        if resp.success:
            # Add the box to the environment dictionary under the appropriate namespace
            if namespace in envs:
                envs[namespace]['boxes'].append({'name': model_name})
            else:
                envs[namespace] = {'boxes': {'name': model_name}}
        else:
            print(f"Failed to spawn model {model_name} in {namespace}.")
        return resp.success
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return False


def spawn_urdf_model(model_name, urdf_path, x, y, z, namespace=''):
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    
    try:
        spawn_service = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        
        # Load and process the URDF file using xacro
        # Make sure this path is correct
        robot_description = xacro.process_file(urdf_path, mappings={'ns': namespace})
        
        model_xml = robot_description.toxml()
        
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        print('got the pose')
        # Spawn the robot in the appropriate namespace
        resp = spawn_service(model_name, model_xml, namespace, pose, "world")
        print('spawned in that shit vro')
        if resp.success:
            # Set the robot_description parameter in the namespace and launch the node
            
            launch = launch_robot_nodes(namespace, model_xml)
            return launch, resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def launch_robot_nodes(namespace, robot_description):
    """Launch robot_state_publisher and joint_state_publisher nodes."""
    
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    rospy.set_param(f'{namespace}/robot_description', robot_description)
    rsp_node = roslaunch.core.Node('robot_state_publisher', 'robot_state_publisher', namespace=namespace, output='screen')
    jsp_node = roslaunch.core.Node('joint_state_publisher', 'joint_state_publisher', namespace=namespace, output='screen')
    launch.launch(rsp_node)
    launch.launch(jsp_node)

def main():
    rospy.init_node('gazebo_model_spawner', anonymous=True)

    num_rooms = 16
    room_distance = 6
    robot_positions = []
    grid_size = int(math.ceil(math.sqrt(num_rooms)))  # Calculate the grid size
    for i in range(num_rooms):
        x = (i % grid_size) * room_distance
        y = (i // grid_size) * room_distance
        robot_positions.append((x,y))
    # Define paths and constants
    robot_description_path = '/home/belecanechzm/Map-less-Navigation/src/testing_pkg/urdf/newRobot.urdf.xacro'
    box_model_path = '/home/belecanechzm/Map-less-Navigation/src/testing_pkg/worlds/box.sdf'

    # Spawning robots
    #robot_positions = [(0, 0), (0, 6), (6, 0), (6, 6), (12, 0), (12, 6), (18, 0), (18, 6), (24, 0), (24, 6)]
    for i, (x, y) in enumerate(robot_positions, start=1):
        namespace = f'robot{i}'
        # Initialize the environment entry with center and empty box list
        envs[namespace] = {'boxes': []}
        spawn_urdf_model(f'robot{i}', robot_description_path, x, y, 0.06, namespace)

        # Spawn boxes around the center
        for j in range(10):
            box_x = x + 1  # Adjusted for a little offset
            box_y = y - 1.5 + 0.3 * j  # Calculate the y position for each box
            spawn_model(f'box{j}_robot{i}', box_model_path, box_x, box_y, 0, namespace)

    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node due to keyboard interrupt.")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt exception caught.")
    except Exception as e:
        rospy.logerr(f"An unexpected exception occurred: {e}")
    finally:
        rospy.loginfo("Node cleanup and shutdown.")
        


if __name__ == '__main__':
    main()
