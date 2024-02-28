#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

class LaserScanProcessor:
    def __init__(self):
        rospy.init_node('laser_scan_processor')
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.processed_scan_publisher = rospy.Publisher('/processed_scan', LaserScan, queue_size=10)

    def scan_callback(self, msg):
        # Normalize range readings to [0, 1]
        normalized_ranges = np.array(msg.ranges) / msg.range_max

        # Downsample the readings (e.g., take every 10th reading)
        downsampled_ranges = normalized_ranges[::10]

        # Create a new LaserScan message with processed data
        processed_scan = LaserScan()
        processed_scan.header = msg.header
        processed_scan.angle_min = msg.angle_min
        processed_scan.angle_max = msg.angle_max
        processed_scan.angle_increment = msg.angle_increment * 10  # Adjust for downsampling
        processed_scan.time_increment = msg.time_increment
        processed_scan.scan_time = msg.scan_time
        processed_scan.range_min = 0
        processed_scan.range_max = 1
        processed_scan.ranges = downsampled_ranges.tolist()

        # Publish the processed scan
        self.processed_scan_publisher.publish(processed_scan)

if __name__ == '__main__':
    processor = LaserScanProcessor()
    rospy.spin()
