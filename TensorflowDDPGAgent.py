###%%
from tensorflow import GradientTape, reduce_mean  # , square
from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization , LeakyReLU, ELU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from collections import deque
from keras import Model
import tensorflow as tf 
from EnergyMarketEnv import *
import matplotlib.pyplot as plt


""" ** Implement a DDPG using Tensorflow models for maximizing the profit of selling electricity 
       over a finite time horizon ** """


# Create a DDPG agent class
class DDPGAgent:
    def __init__(self, state_shape, action_shape, action_bound):
        #set the value of hyperparemeters
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.actor_lr = 1e-5
        self.critic_lr = 1e-4
        self.tau = 1e-4
        self.gamma = 0.95

        # build actor and critic models
        self.actor = self.build_actor_model()
        self.critic = self.build_critic_model()

        # build target-actor and target-critic models as a copy of the actor and the critic, respectively.
        self.target_actor = self.build_actor_model()
        self.target_critic = self.build_critic_model()

        #set target-actor and target-critic weights as a copy of the actor and the critic weights, respectively.
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        #
        self.critic_loss = MeanSquaredError()
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
        self.buffer_size = 1000000
        self.memory = deque(maxlen=self.buffer_size)

    # Beispiel mit Leaky ReLU
    def build_actor_model(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(128)(inputs)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(16)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(128)(x)
        output = Dense(self.action_shape[0], activation='tanh')(x)
        output = output * self.action_bound
        model = Model(inputs, output)
        return model

    # Beispiel mit ELU
    def build_critic_modelss(self):
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=self.action_shape)
        x = Concatenate()([state_input, action_input])
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(128)(x)
        output = Dense(1)(x)
        model = Model([state_input, action_input], output)
        return model

        # Build Actor Model
    def build_actor_modelss(self):
        inputs = Input(shape=self.state_shape)

        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        output = Dense(self.action_shape[0], activation='tanh')(x)
        output = output * self.action_bound
        
        model = Model(inputs, output)
        return model

    # Build Critic Model
    def build_critic_model(self):
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=self.action_shape)
        
        x = Concatenate()([state_input, action_input]) 
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.1)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.1)(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
       # x = Dropout(0.1)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.1)(x)
        
        x = Dense(128, activation='relu')(x)
        output = Dense(1)(x)
        
        model = Model([state_input, action_input], output)
        return model

    #build actor model : convolutional neural network
    def build_actor_models(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(512, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.action_shape[0], activation='sigmoid')(x)
        #map the actions produced by the actor network to the action space of our environment
        output = output*self.action_bound
        #output = (output + 1.0)*0.5*self.action_bound
        model = Model(inputs, output)
        return model

    #build critic model : convolutional neural network
    def build_critic_models(self):
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=self.action_shape)
        x = Concatenate()([state_input, action_input])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        #Produce a single value for each action dimension
        output = Dense(1)(x)
        model = Model([state_input, action_input], output)
        return model

    # Update Target Network weights with soft update approach
    def update_target(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight*self.tau + target_weight*(1 - self.tau))

    #predict action
    def get_action(self, state):
            #Use the actor network to predict the action
            state = np.expand_dims(state, axis=0)
            #add a batch dimension to the state
            action = self.actor.predict(state)[0]
            return action

    #buffer memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Select randomly stored experiences in replay buffer, where the total number of experiences is equal to the batch size.
    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    #Train the Networks
    def train(self, states, actions, next_states, rewards, dones):
        states =  tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Learning process for the actor and critic network
        with GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_Q_values = self.target_critic([next_states, target_actions])
            target_Q_values = rewards + self.gamma*target_Q_values*(1 - dones)
            Q_values = self.critic([states, actions])
            critic_loss = self.critic_loss(target_Q_values, Q_values)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Learning process for the actor network
        with GradientTape() as tape:
            predicted_actions = self.actor(states)
            #minimizing the negative mean of the Q-values encourages the actor to take actions that lead to higher expected Q-values
            actor_loss = -reduce_mean(self.critic([states, predicted_actions]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks
        self.update_target(self.target_actor.variables, self.actor.variables)
        self.update_target(self.target_critic.variables, self.critic.variables)
   
    def train_batch(self, batch_size):
        # Check if enough experiences are available in the replay buffer
        if len(self.memory) < batch_size:
                   return
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones =   self.sample_batch(batch_size)
        self.train(states, actions, next_states, rewards, dones)

""" ** Initialize the environment ** """

env = EnergyMarketEnv()

state_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_bound = env.action_space.high[0]

agent = DDPGAgent(state_shape, action_shape, action_bound)

episodes = 100
max_steps = 20
batch_size = 100
total_score = 0
rewards_per_episode = []
for episode in range(episodes):
    state = env.reset()
    score = 0
  
    for step in range(max_steps):
        # Get the next action using the updated exploration rate
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        score += reward
        state = next_state

        if done:
            break
    agent.train_batch(batch_size)
    rewards_per_episode.append(score)
    print(f"Episode {episode + 1}, Reward: {score}")
    total_score += score

env.close()
print(f"Mean reward: {total_score/episodes}")
# %%
plt.figure(figsize=(10, 6))
plt.scatter(range(1, episodes + 1), rewards_per_episode, color='blue', alpha=0.6)
plt.title('Belohnungsentwicklung über Episoden')
plt.xlabel('Episoden')
plt.ylabel('Belohnung')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')  # Horizontale Linie bei y=0
plt.show()


