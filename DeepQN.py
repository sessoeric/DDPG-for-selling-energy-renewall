'''
Q-Network: The neural network that approximates the Q-values.
Target Network: A separate neural network that provides stable target Q-values during training.
Experience Replay Buffer: Stores the agent's experiences to sample from them during training.
Training Loop: Implements the episode and time-step loop for training the agent
'''

import numpy as np
import random
import gym  # OpenAI Gym for environments
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        
        # Q-Network
        self.q_network = self.build_model()
        
        # Target Q-Network
        self.target_network = self.build_model()
        self.update_target_network()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of experiences from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Get the max Q-value for the next state from the target network
                target += self.gamma * np.amax(self.target_network.predict(next_state)[0])

            # Get the current Q-values and update the target for the action taken
            target_f = self.q_network.predict(state)
            target_f[0][action] = target

            # Train the Q-network with the updated target
            self.q_network.fit(state, target_f, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(episodes):
    env = gym.make('CartPole-v1')  # Choose an environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for time in range(10):  # Limit for time steps in each episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e + 1}/{episodes}, Score: {time}")
                break

        agent.replay()

        # Update target network every C episodes
        if e % 10 == 0:
            agent.update_target_network()

    env.close()

# Train the DQN agent
train_dqn_agent(10)
