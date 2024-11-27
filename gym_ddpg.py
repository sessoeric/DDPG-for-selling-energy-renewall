import gym
from stable_baselines3 import PPO

# Erstelle die Gym-Umgebung
env = gym.make('CartPole-v1')

# Erstelle das RL-Modell
model = PPO('MlpPolicy', env, verbose=1)

# Training des Modells
num_episodes = 100  # Gesamtanzahl der Episoden
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)  # Vorhersage des nächsten Schrittes
        obs, reward, done, info = env.step(action)  # Ausführen des Schrittes
