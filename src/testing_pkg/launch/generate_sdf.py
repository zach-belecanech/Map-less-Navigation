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

def main():
    num_rooms = 16  # Specify the number of rooms you want to generate
    room_distance = 6  # Distance between the centers of two rooms
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
    grid_size = int(math.ceil(math.sqrt(num_rooms)))  # Calculate the grid size
    for i in range(num_rooms):
        x = (i % grid_size) * room_distance
        y = (i // grid_size) * room_distance
        rooms.append(generate_room(x, y, i))
    
    sdf_content = f"{sdf_header}{''.join(rooms)}{sdf_footer}"
    
    with open("/home/easz/catkin_ws/src/testing_pkg/worlds/custom_rooms_world.world", "w") as f:
        f.write(sdf_content)

if __name__ == "__main__":
    main()
