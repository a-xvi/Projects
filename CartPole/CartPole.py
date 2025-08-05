import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
import matplotlib.pyplot as plt
from collections import deque

# Initialize the environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate
        self.model = self.build_model()  # Build the Q-network

    def build_model(self):
        """Builds a simple neural network for Q-learning."""
        model = models.Sequential([
            layers.Dense(24, activation="relu", input_shape=(self.state_size,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in memory for experience replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Corrected typo
            return random.randrange(self.action_size)  # Random action
        # Ensure state is reshaped properly
        state = np.array(state).reshape(1, self.state_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Action with the highest Q-value

    def train_from_memory(self, batch_size=32):
        """Trains the model using experience replay."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Ensure next_state is reshaped properly
                next_state = np.array(next_state).reshape(1, self.state_size)
                target += self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])

            # Ensure state is reshaped properly
            state = np.array(state).reshape(1, self.state_size)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon

# Training parameters
episodes = 500
batch_size = 32
agent = DQNAgent(state_size, action_size)

for episode in range(episodes):
    # Reset the environment
    state = env.reset()[0]  # For gym >= 0.26, reset() returns (state, info)
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(200):  # Limit each episode to 200 steps
        # Choose an action
        action = agent.act(state)
        # Take the action in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state).reshape(1, state_size)
        # Store the experience in memory
        agent.remember(state, action, reward, next_state, done)
        # Update the current state
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

    # Train the agent using experience replay
    agent.train_from_memory(batch_size)

env.close()