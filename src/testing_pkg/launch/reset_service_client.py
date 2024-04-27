#!/usr/bin/env python3
import sys
import rospy
from testing_pkg.srv import ResetEnvironment

def reset_environment_client(namespace):
    rospy.wait_for_service('reset_environment')
    try:
        reset_environment = rospy.ServiceProxy('reset_environment', ResetEnvironment)
        resp = reset_environment(namespace)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    rospy.init_node('reset_environment_client')
    if len(sys.argv) == 2:
        namespace = sys.argv[1]
        print("Requesting reset for namespace %s" % namespace)
        print("Reset successful: %s" % reset_environment_client(namespace))
    else:
        print("Usage: reset_service_client.py <namespace>")
