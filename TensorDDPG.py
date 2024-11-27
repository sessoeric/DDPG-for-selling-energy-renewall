import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Input, Concatenate, LSTM
from keras.optimizers import Adam
from keras.models import Model
from EnergyMarketEnv import *  # Passen Sie diesen Import an Ihre Umgebung an.

class DDPGAgent:
    def __init__(self, state_shape, action_shape, action_bound):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.actor_lr = 1e-5
        self.critic_lr = 1e-4
        self.tau = 1e-3
        self.gamma = 0.95

        # Initialisieren der Modelle
        self.actor = self.build_actor_model()
        self.critic = self.build_critic_model()
        self.target_actor = self.build_actor_model()
        self.target_critic = self.build_critic_model()

        # Kopieren der Gewichte
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Optimierer
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

        # Erfahrungsspeicher initialisieren
        self.memory = deque(maxlen=1000000)

    def build_actor_model(self):
        inputs = Input(shape=(None, *self.state_shape))  # Zustandseingaben mit sequenzieller Form
        x = LSTM(128)(inputs)  # LSTM für Zustände
        x = Dense(64, activation='relu')(x)
        output = Dense(self.action_shape[0], activation='tanh')(x)
        output = output * self.action_bound  # Skalierung der Aktion
        model = Model(inputs, output)
        return model

    def build_critic_model(self):
        state_input = Input(shape=(None, *self.state_shape))  # Zustandseingaben mit sequenzieller Form
        action_input = Input(shape=self.action_shape)
        x = LSTM(128)(state_input)  # LSTM für Zustände
        x = Concatenate()([x, action_input])  # Kombinieren von Zustand und Aktion
        x = Dense(64, activation='relu')(x)
        output = Dense(1)(x)  # Q-Wert Ausgabe
        model = Model([state_input, action_input], output)
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)  # Hinzufügen der Batch-Dimension
        action = self.actor.predict(state)[0]
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        # Zufällige Stichprobe aus dem Erfahrungsspeicher
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Vorhersage für die nächsten Aktionen und Q-Werte
        target_actions = self.get_action(next_states)  # Aktionsvorhersage
        target_Q_values = self.get_Q_value(next_states, target_actions)
        target_Q_values = rewards + self.gamma * target_Q_values * (1 - dones)

        # Q-Werte für die aktuellen Zustände und Aktionen
        Q_values = self.get_Q_value(states, actions)

        # Critic Verlust berechnen
        critic_loss = np.mean((target_Q_values - Q_values) ** 2)

        # Manuelle Gradientenberechnung für Critic
        critic_grads = -2 * (target_Q_values - Q_values) / batch_size  # Einfacher Gradient
        self.critic_weights -= 0.01 * critic_grads  # Lernrate anpassen

        # Actor Verlust berechnen
        predicted_actions = self.get_action(states)
        actor_loss = -np.mean(self.get_Q_value(states, predicted_actions))

        # Manuelle Gradientenberechnung für Actor
        actor_grads = -predicted_actions / batch_size  # Einfacher Gradient
        self.actor_weights -= 0.01 * actor_grads  # Lernrate anpassen

    def update_target(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1 - self.tau))

# Beispielverwendung
if __name__ == "__main__":
    # Initialisieren Sie die Umgebung
    env = EnergyMarketEnv()  # Ihre Umgebung hier anpassen

    state_shape = env.observation_space.shape  # Zustandform anpassen
    action_shape = env.action_space.shape  # Aktionsform anpassen
    action_bound = env.action_space.high[0]  # Aktionsgrenze anpassen

    agent = DDPGAgent(state_shape, action_shape, action_bound)

    episodes = 100
    max_steps = 20
    batch_size = 64
    total_score = 0

    for episode in range(episodes):
        state = env.reset()
        score = 0
        state_sequence = []

        for step in range(max_steps):
            # Speichern Sie den aktuellen Zustand in der Sequenz
            state_sequence.append(state)
            if len(state_sequence) > 50:  # Maximale Sequenzlänge von 50
                state_sequence.pop(0)  # Ältesten Zustand entfernen

            # Wandeln Sie die Sequenz in das richtige Format um
            current_state = np.array(state_sequence).reshape((1, len(state_sequence), state_shape[0]))

            action = agent.get_action(current_state)

            next_state, reward, done, _ = env.step(action)
            agent.remember(current_state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                break

        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        print(f"Episode {episode + 1}, Reward: {score}")
        total_score += score

    env.close()
    print(f"Mean reward: {total_score / episodes}")
