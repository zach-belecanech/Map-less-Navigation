#!/usr/bin/env python3
import math

def generate_wall(x, y, length, height, name_suffix, orientation="0 0 0"):
    return f"""
    <model name='wall_{name_suffix}'>
        <static>true</static>
        <pose>{x} {y} 0 {orientation}</pose>
        <link name='link'>
            <collision name='collision'>
                <geometry>
                    <box><size>{length} 0.1 {height}</size></box>
                </geometry>
            </collision>
            <visual name='visual'>
                <geometry>
                    <box><size>{length} 0.1 {height}</size></box>
                </geometry>
                <material>
                    <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Grey</name>
                    </script>
                </material>
            </visual>
        </link>
    </model>
    """

def generate_room(x_origin, y_origin, room_id):
    room_width = 5
    room_height = 5
    walls = []
    walls.append(generate_wall(x_origin, y_origin + room_height/2, room_width, 1, f"{room_id}_north"))
    walls.append(generate_wall(x_origin, y_origin - room_height/2, room_width, 1, f"{room_id}_south"))
    walls.append(generate_wall(x_origin + room_width/2, y_origin, room_height, 1, f"{room_id}_east", "0 0 1.5707"))
    walls.append(generate_wall(x_origin - room_width/2, y_origin, room_height, 1, f"{room_id}_west", "0 0 1.5707"))
    return "\n".join(walls)


def generate_box(x, y, box_num, robot_name):
    return f"""
    <model name='box{box_num}_{robot_name}'>
        <static>true</static>
        <pose>{x} {y} 0 0 0 0</pose>
        <link name='link'>
            <collision name='collision'>
                <geometry>
                    <box><size>0.5 0.5 0.5</size></box>
                </geometry>
            </collision>
            <visual name='visual'>
                <geometry>
                    <box><size>0.5 0.5 0.5</size></box>
                </geometry>
                <material>
                    <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Red</name>
                    </script>
                </material>
            </visual>
        </link>
    </model>
    """


def main():
    num_rooms = 200  # Specify the number of rooms you want to generate
    room_distance = 6  # Distance between the centers of two rooms
    robot_urdf_path = "/home/belecanechzm/Map-less-Navigation/src/testing_pkg/urdf/newRobot.urdf.xacro"

    sdf_header = f"""
    <sdf version='1.7'>
        <world name='default'>
            <include>
                <uri>model://sun</uri>
            </include>
            <model name='ground_plane'>
                <static>true</static>
                <link name='link'>
                    <collision name='collision'>
                        <geometry>
                            <plane>
                                <normal>0 0 1</normal>
                                <size>100 100</size>
                            </plane>
                        </geometry>
                    </collision>
                    <visual name='visual'>
                        <geometry>
                            <plane>
                                <normal>0 0 1</normal>
                                <size>100 100</size>
                            </plane>
                        </geometry>
                        <material>
                            <script>
                                <uri>file://media/materials/scripts/gazebo.material</uri>
                                <name>Gazebo/Grey</name>
                            </script>
                        </material>
                    </visual>
                </link>
            </model>
    """
    sdf_footer = """
        </world>
    </sdf>
    """
    rooms = []
    boxes = []
    grid_size = int(math.ceil(math.sqrt(num_rooms)))  # Calculate the grid size
    for i in range(num_rooms):
        x = (i % grid_size) * room_distance
        y = (i // grid_size) * room_distance
        rooms.append(generate_room(x, y, i))

    

    
    sdf_content = f"{sdf_header}{''.join(rooms)}{''.join(boxes)}{sdf_footer}"
    
    with open("/home/belecanechzm/Map-less-Navigation/src/testing_pkg/worlds/custom_rooms_world.world", "w") as f:
        f.write(sdf_content)

if __name__ == "__main__":
    main()
