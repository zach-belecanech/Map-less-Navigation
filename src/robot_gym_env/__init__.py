from gym.envs.registration import register

register(
    id='RobotGym-v0',
    entry_point='robot_gym_env.src.robot_env:RobotGymEnv',
)
