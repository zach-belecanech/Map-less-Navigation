import tensorflow as tf
import numpy as np
import gym
from robot_env import RobotGymEnv

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

        return actor_output, critic_output

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        # Forward pass
        policy_logits, values = self(states)
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
        entropy_loss = -tf.reduce_mean(policy_logits * tf.nn.softmax(policy_logits))

        # Total loss
        total_loss = actor_loss + critic_loss + 0.01 * entropy_loss  # Adjust the entropy coefficient as needed

        return total_loss

def train(env, num_episodes, learning_rate=0.0007):
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    model = ActorCriticNetwork(action_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, -1])
        done = False
        total_reward = 0

        while not done:
            # Get action probabilities and value from the model
            policy_logits, value = model(tf.convert_to_tensor(state))
            if tf.reduce_any(tf.math.is_nan(policy_logits)):
                print("NaN detected in policy logits")

            action_probs = tf.nn.softmax(policy_logits).numpy().flatten()
            if np.isnan(action_probs).any():
                print("NaN detected in action probabilities")

            action = np.random.choice(action_size, p=action_probs)

            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward

            # Train the model
            with tf.GradientTape() as tape:
                loss = model.compute_loss(state, action, reward, next_state, done)
                if tf.reduce_any(tf.math.is_nan(loss)):
                    print("NaN detected in loss")

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state

        print(f'Episode: {episode + 1}, Total Reward: {total_reward}')

    return model

def main():
    env = RobotGymEnv()
    num_episodes = 500

    action_size = env.action_space.n
    model = ActorCriticNetwork(action_size)
    trained_model = train(env, num_episodes)

if __name__ == '__main__':
    main()
