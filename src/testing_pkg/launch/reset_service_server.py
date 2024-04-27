#!/usr/bin/env python3
import rospy
from testing_pkg.srv import ResetEnvironment, ResetEnvironmentResponse

def handle_reset_environment(req):
    print("Resetting environment for namespace: %s" % req.namespace)
    return ResetEnvironmentResponse(success=True)

def reset_environment_server():
    rospy.init_node('reset_environment_server')
    s = rospy.Service('reset_environment', ResetEnvironment, handle_reset_environment)
    print("Ready to reset environment.")
    rospy.spin()

if __name__ == "__main__":
    reset_environment_server()
