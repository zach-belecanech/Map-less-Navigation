import math
import time
import tensorflow as tf
import numpy as np
import gym
from multithread_env import RobotGymEnv
from globals import environments
import rospy
from geometry_msgs.msg import PoseArray, Pose
import threading
from std_msgs.msg import String
from threading import Thread, Lock
from tensorflow.keras.callbacks import TensorBoard
import random
import os
import datetime

log_dir = "/home/belecanechzm/tensor_logs/log_10"
summary_writer = tf.summary.create_file_writer(log_dir)

class ExperienceBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()

    def add(self, experience):
        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append(experience)

    def get_dataset(self, batch_size):
        with self.lock:
            # Convert list of tuples to a tuple of lists (columns)
            states, actions, rewards, next_states, dones = zip(*self.buffer)

            # Convert each list to a tensor
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # Ensure your actions are compatible with tf.int32
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.bool)

            # Create a dataset from tensors
            dataset = tf.data.Dataset.from_tensor_slices((states, actions, rewards, next_states, dones))
            dataset = dataset.shuffle(buffer_size=len(self.buffer))
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return dataset


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, num_turning_angles, num_velocities):
        super(ActorCriticNetwork, self).__init__()
        self.num_turning_angles = num_turning_angles
        self.num_velocities = num_velocities

        # Common layers
        self.common_layers = [
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu')
        ]

        # Actor network
        self.lstm = tf.keras.layers.LSTM(512)
        self.actor_turning = tf.keras.layers.Dense(num_turning_angles, activation='softmax')
        self.actor_velocity = tf.keras.layers.Dense(num_velocities, activation='softmax')

        # Critic network
        self.critic_layers = [
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)  # Predicts state value
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.common_layers:
            x = layer(x)

        # Actor output
        lstm_output = self.lstm(tf.expand_dims(x, axis=1))  # LSTM expects time sequence data
        turning_probs = self.actor_turning(lstm_output)
        velocity_probs = self.actor_velocity(lstm_output)

        # Critic output
        critic_value = x
        for layer in self.critic_layers:
            critic_value = layer(critic_value)

        return turning_probs, velocity_probs, critic_value

    def compute_loss(self,turning_probs, velocity_probs, values, actions_turning, actions_velocity, rewards, dones, next_values, gamma=0.99):
        # Convert actions to one-hot encoding
        action_turning_one_hot = tf.one_hot(actions_turning, depth=turning_probs.shape[-1], dtype=tf.float32)
        action_velocity_one_hot = tf.one_hot(actions_velocity, depth=velocity_probs.shape[-1], dtype=tf.float32)

        # Calculate log probabilities
        log_probs_turning = tf.math.log(tf.reduce_sum(turning_probs * action_turning_one_hot, axis=-1))
        log_probs_velocity = tf.math.log(tf.reduce_sum(velocity_probs * action_velocity_one_hot, axis=-1))
        log_probs = log_probs_turning + log_probs_velocity

        # Compute the discounted rewards and advantages
        returns = rewards + gamma * next_values * (1 - dones)
        advantages = returns - values

        # Actor loss
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

        # Critic loss
        critic_loss = tf.reduce_mean(tf.square(advantages))

        # Entropy loss (to encourage exploration)
        entropy_turning = -tf.reduce_mean(tf.reduce_sum(turning_probs * tf.math.log(turning_probs), axis=1))
        entropy_velocity = -tf.reduce_mean(tf.reduce_sum(velocity_probs * tf.math.log(velocity_probs), axis=1))
        entropy_loss = entropy_turning + entropy_velocity

        # Total loss
        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss  # You can adjust the weight of the entropy term

        return total_loss


def setup_tensorflow_gpu_memory(limit=1024):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
        except RuntimeError as e:
            print(e)

def train_thread(model, namespace, center, num_episodes, experience_buffer, writer, thread_id, lock):
    #print("training mf \n")
    env = RobotGymEnv(namespace=namespace, center_pos=center)
    for episode in range(num_episodes):
        try:
            state = env.reset()
            state = np.reshape(state, [1, -1])
            done = False
            total_reward = 0
            steps = 0

            while not done:
                #print(f'Locked and getting probs on thread: {thread_id}')
                with lock:
                    turning_probs, velocity_probs, value = model(tf.convert_to_tensor(state, dtype=tf.float32))
                #print(f'Successfully got the probs on thread: {thread_id}')
                # Assuming you need to select one turning angle and one velocity independently
                turning_action = np.random.choice(range(turning_probs.shape[-1]), p=turning_probs.numpy().flatten())
                velocity_action = np.random.choice(range(velocity_probs.shape[-1]), p=velocity_probs.numpy().flatten())
                action = (turning_action, velocity_action)

                # Assuming your environment can handle a tuple for action
                #print(f'Taking a step on thread: {thread_id}')
                next_state, reward, done, _ = env.step(action)
                #print(f'Took the step on thread: {thread_id}')
                next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
                #print(f'Getting the lock for experience buffer on thread: {thread_id}')
                with lock:
                   # print(f'Got the lock yurrrrr (experience buffer lock) thread: {thread_id}')
                    experience_buffer.add((state, action, reward, next_state, done))
                #print(f'done putting the stuff into experience buffer on thread: {thread_id}')
                state = next_state
                total_reward += reward
                steps += 1

            average_reward = total_reward / steps
            with writer.as_default():
                tf.summary.scalar("Total Reward", total_reward, step=episode + thread_id * num_episodes)
                tf.summary.scalar("Average Reward", average_reward, step=episode + thread_id * num_episodes)
                writer.flush()
        except ValueError as e:
            print(f"Error during training at episode {episode} in thread {thread_id}: {e}")

def learn_from_buffer(model, experience_buffer, batch_size, optimizer, lock, total_steps, writer):
    print("entered the buffer bitch\n")
    current_step = 0
    while current_step < total_steps:
        print(f"Current buffer size: {len(experience_buffer.buffer)}")
        if len(experience_buffer.buffer) >= batch_size:
            print('We getting some data in this mf on the learning thread')
            dataset = experience_buffer.get_dataset(batch_size)
            print('We have got the dataset on the learning thread')
            for (states, actions, rewards, next_states, dones) in dataset:
                actions_turning = np.array([action[0] for action in actions])
                actions_velocity = np.array([action[1] for action in actions])
                print('Aquring lock on learning thread')
                with lock:
                    print('Got the lock on learning thread')
                    with tf.GradientTape() as tape:
                        turning_probs, velocity_probs, values = model(states)
                        next_turning_probs, next_velocity_probs, next_values = model(next_states)
                        loss = model.compute_loss(turning_probs, velocity_probs, values,
                                                actions_turning, actions_velocity, rewards,
                                                dones, next_values, gamma=0.99)
                        entropy_turning = -tf.reduce_mean(tf.reduce_sum(turning_probs * tf.math.log(turning_probs + 1e-10), axis=1))
                        entropy_velocity = -tf.reduce_mean(tf.reduce_sum(velocity_probs * tf.math.log(velocity_probs + 1e-10), axis=1))
                        entropy = entropy_turning + entropy_velocity

                    grads = tape.gradient(loss, model.trainable_variables)
                    if None in grads:
                        print("None gradients found")
                        for grad, var in zip(grads, model.trainable_variables):
                            if grad is None:
                                print(f"No gradient for variable: {var.name}")

                    valid_grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                    optimizer.apply_gradients(valid_grads_vars)
                print('done with lock on learning thread')

                with writer.as_default():
                    tf.summary.scalar("Loss", loss, step=current_step)
                    tf.summary.scalar("Learning Rate", optimizer.learning_rate.numpy(), step=current_step)
                    tf.summary.scalar("Entropy", entropy, step=current_step)
                    for grad, var in zip(grads, model.trainable_variables):
                        tf.summary.histogram(f"Gradients/{var.name}", grad, step=current_step)
                    writer.flush()

                if current_step % 10000 == 0:
                    model_save_path = f'/home/belecanechzm/models/checkpoints/model_step_{current_step}'
                    model.save(model_save_path, save_format='tf')
                    with writer.as_default():
                        tf.summary.scalar("Saved", 100, step=current_step)

                current_step += 1

        time.sleep(0.1)

initial_learning_rate = 0.005
final_learning_rate = 0.001
decay_steps = 8000000  # Total number of steps over which the learning rate will decay

def get_learning_rate(current_step):
    if current_step < decay_steps:
        return initial_learning_rate - (current_step * (initial_learning_rate - final_learning_rate) / decay_steps)
    return final_learning_rate

def main():
    print('Started\n')
    rospy.init_node('robot_gym_node', anonymous=True)
    num_episodes = 500
    num_rooms = 16
    room_distance = 6
    robots = []
    center_pos = []
    grid_size = int(math.ceil(math.sqrt(num_rooms)))
    total_steps = num_episodes * num_rooms * 150  # This is an example calculation; adjust based on actual expected steps

    for i in range(num_rooms):
        x = (i % grid_size) * room_distance
        y = (i // grid_size) * room_distance
        center_pos.append((x, y))
        robots.append(f'robot{i+1}')

    setup_tensorflow_gpu_memory()

    shared_model = ActorCriticNetwork(7,7)
    print('Started model')
    experience_buffer = ExperienceBuffer(100000)
    lock = Lock()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005, decay=0.99, epsilon=0.00005)
    

    collection_threads = [
        Thread(target=train_thread, args=(shared_model, namespace, center, num_episodes, experience_buffer, summary_writer, i, lock))
        for i, (namespace, center) in enumerate(zip(robots, center_pos))
    ]

    learning_thread = Thread(target=learn_from_buffer, args=(shared_model, experience_buffer, 128, optimizer, lock, total_steps, summary_writer))

    learning_thread.start()
    for thread in collection_threads:
        thread.start()
    

    for thread in collection_threads:
        thread.join()
    learning_thread.join()

    shared_model.save(f'/home/belecanechzm/models/checkpoints/model_final_0', save_format='tf')

if __name__ == '__main__':
    main()
