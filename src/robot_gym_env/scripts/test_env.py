#!/usr/bin/env python3

import gym
import numpy as np
import time
import robot_gym_env
# Make sure to import your gym environment package correctly


def main():
    # Initialize the gym environment
    env = gym.make('RobotGym-v0')
    
    # Reset the environment to get the initial observation
    observation = env.reset()
    print("Initial Observation:", observation)

    # Number of steps to simulate
    num_steps = 50

    for step in range(num_steps):
        # Generate a random action: replace this with any specific action you want to test
        action = env.action_space.sample()

        # Take the action in the environment
        observation, reward, done, _ = env.step(action)
        print(f"\nStep: {step + 1}")
        print("Action:", action)
        print("Observation:", observation)
        print("Reward:", reward)
        print("Done:", done)

        # Break the loop if the episode is completed
        if done:
            print("Episode finished after {} timesteps".format(step + 1))
            break

        # Sleep to slow down the simulation for observation
        time.sleep(0.1)

    # Reset the environment to test the reset functionality
    print("\nResetting Environment...")
    observation = env.reset()
    print("Post-reset Observation:", observation)

    # Close the environment properly
    env.close()

if __name__ == '__main__':
    main()
