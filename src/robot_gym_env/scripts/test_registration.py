#!/usr/bin/env python3
import gym

def test_gym_environment():
    print("Attempting to create RobotGym-v0 environment...")
    env = gym.make('RobotGym-v0')
    print("Environment created successfully!")

if __name__ == "__main__":
    test_gym_environment()
