import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from EnergyMarketEnv import *

# Environment
env = EnergyMarketEnv()
n_actions = env.action_space.shape[-1]

# Action Noise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), 
                                            sigma=0.2 * np.ones(n_actions))

# Model
model = DDPG("MlpPolicy", env, learning_rate=0.0001, action_noise=action_noise, 
              verbose=1, policy_kwargs=dict(net_arch=[256, 256]))

# Training
model.learn(total_timesteps=18000, log_interval=10)

# Evaluation
episodes = 100
total_score = 0
for episode in range(1, episodes + 1):
    obs = env.reset()
    score = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        score += reward
    print(f"Episode {episode}, Reward: {score}")
    total_score += score

print(f"Mean reward: {total_score / episodes}")
