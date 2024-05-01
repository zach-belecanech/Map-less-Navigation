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

log_dir = "/home/easz/tensor_logs/log_9"
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

    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            return random.sample(self.buffer, batch_size)



class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.action_size = action_size

        self.common_layers = [
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu')
        ]

        # Actor network
        self.actor_layers = [
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')  # Probability distribution over actions
        ]

        # Critic network
        self.critic_layers = [
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)  # Value function
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.common_layers:
            x = layer(x)

        # Actor output
        actor_output = x
        for layer in self.actor_layers:
            actor_output = layer(actor_output)


        # Critic output
        critic_output = x
        for layer in self.critic_layers:
            critic_output = layer(critic_output)

        if actor_output is None or critic_output is None:
            print("Output of the network is None.")

        if len(actor_output.shape) == 3:
            actor_output = tf.squeeze(actor_output, axis=1)

        return actor_output, critic_output

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        # Forward pass
        policy_logits, values = self(states)
        if policy_logits.shape[1] == 1:
            policy_logits = tf.squeeze(policy_logits, axis=1)  # This removes the middle dimension if it's size 1
        _, next_values = self(next_states)

        # Convert actions to one-hot encoding
        action_one_hot = tf.one_hot(actions, self.action_size, dtype=tf.float32)

        # Compute advantages
        advantages = rewards + gamma * next_values * (1 - dones) - values

        # Actor loss (policy gradient)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=action_one_hot, logits=policy_logits)
        actor_loss = tf.reduce_mean(policy_loss * tf.stop_gradient(advantages))

        # Critic loss (value function)
        critic_loss = tf.reduce_mean(tf.square(advantages))

        # Entropy loss (to encourage exploration)
        #entropy_loss = -tf.reduce_mean(policy_logits * tf.nn.softmax(policy_logits))

        # Total loss
        #total_loss = actor_loss + critic_loss + 0.0001 * entropy_loss  # Adjust the entropy coefficient as needed
        total_loss = actor_loss + critic_loss
        return total_loss

def setup_tensorflow_gpu_memory(limit=512):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
        except RuntimeError as e:
            print(e)

def train_thread(model, namespace, center, num_episodes, experience_buffer, writer, thread_id):
    env = RobotGymEnv(namespace=namespace, center_pos=center)
    for episode in range(num_episodes):
        try:
            state = env.reset()
            state = np.reshape(state, [1, -1])
            done = False
            total_reward = 0

            while not done:
                policy_logits, value = model(tf.convert_to_tensor(state))
                action_probs = tf.nn.softmax(policy_logits).numpy().flatten()
                action = np.random.choice(env.action_space.n, p=action_probs)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

                experience_buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

            with writer.as_default():
                tf.summary.scalar("Total Reward", total_reward, step=episode + thread_id * num_episodes)
                writer.flush()
        except ValueError as e:
            print(f"Error during training at episode {episode} in thread {thread_id}: {e}")

def learn_from_buffer(model, experience_buffer, batch_size, optimizer, lock, total_steps, writer):
    current_step = 0

    while True:
        if len(experience_buffer.buffer) >= batch_size:
            experiences = experience_buffer.sample(batch_size)
            try:
                if experiences:
                    lr = get_learning_rate(current_step)
                    tf.keras.backend.set_value(optimizer.learning_rate, lr)

                    states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))
                    with lock, tf.GradientTape() as tape:
                        loss = model.compute_loss(states, actions, rewards, next_states, dones)

                    grads = tape.gradient(loss, model.trainable_variables)
                    if None in grads:
                        print("None gradients found")
                        for grad, var in zip(grads, model.trainable_variables):
                            if grad is None:
                                print(f"No gradient for variable: {var.name}")

                    # Apply gradient clipping or other operations only on valid (non-None) gradients
                    valid_grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                    clipped_grads = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in valid_grads_vars]
                    optimizer.apply_gradients(clipped_grads)

                    
                    
                    with writer.as_default():
                        tf.summary.scalar("Loss", loss, step=current_step)
                        tf.summary.scalar("Learning Rate", lr, step=current_step)
                        tf.summary.scalar("Saved Checkpoints", 100, step=current_step)
                        writer.flush()

                    if current_step % 10000:
                        try:
                            model_save_path = f'/home/easz/models/checkpoints/model_step_{current_step}'
                            model.save(model_save_path, save_format='tf')
                            with writer.as_default():
                                tf.summary.scalar("Saved", 100, step=current_step)
                        except Exception as e:
                            print('could not save: ', e)

                    current_step += 1  # Increment the step count only when an update is performed

                    # Save the model every 10,000 steps
                    if current_step >= total_steps:  # Check if the total number of desired updates has been reached
                        break


            except ValueError as e:
                print(f"Error during learning at step {current_step}: {e}")
                # Handle or log the error as needed

            time.sleep(0.1)  # Avoid tight loop if buffer is frequently empty



initial_learning_rate = 0.005
final_learning_rate = 0.001
decay_steps = 8000000  # Total number of steps over which the learning rate will decay

def get_learning_rate(current_step):
    if current_step < decay_steps:
        return initial_learning_rate - (current_step * (initial_learning_rate - final_learning_rate) / decay_steps)
    return final_learning_rate

def main():
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

    shared_model = ActorCriticNetwork(49)
    experience_buffer = ExperienceBuffer(100000)
    lock = Lock()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005, decay=0.99, epsilon=0.00005)
    

    collection_threads = [
        Thread(target=train_thread, args=(shared_model, namespace, center, num_episodes, experience_buffer, summary_writer, i))
        for i, (namespace, center) in enumerate(zip(robots, center_pos))
    ]

    learning_thread = Thread(target=learn_from_buffer, args=(shared_model, experience_buffer, 128, optimizer, lock, total_steps, summary_writer))

    learning_thread.start()
    for thread in collection_threads:
        thread.start()
    

    for thread in collection_threads:
        thread.join()
    learning_thread.join()

    shared_model.save(f'/home/easz/models/checkpoints/model_final_128_homePC.h5')

if __name__ == '__main__':
    main()
