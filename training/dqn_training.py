# -*- coding: utf-8 -*-
"""dqn_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LB1YZDw39OrGHQmfe8HL3sdRpNCl2vAE
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# # %cd /content/drive/MyDrive/Prince_Ndanyuzwe_rl_summative-main.zip

# # Unzip the project directory
# !unzip /content/drive/MyDrive/Prince_Ndanyuzwe_rl_summative-main.zip -d /content/drive/MyDrive/

# Commented out IPython magic to ensure Python compatibility.
# Change to the unzipped directory
# %cd /content/drive/MyDrive/Prince_Ndanyuzwe_rl_summative-main

# !pip install gym
# !pip install stable-baselines3
# !pip install torch torchvision torchaudio
# !pip install opencv-python-headless

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Import custom environment
from environment.custom_env import PatientMonitoringEnv

# Create logs and model directories
log_dir = "logs/dqn"
model_dir = "models/dqn"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Initialize environment
env = PatientMonitoringEnv()
env = Monitor(env, log_dir)

# Define model hyperparameters
model = DQN(
    "MlpPolicy", env,
    learning_rate=0.0005,
    gamma=0.99,
    batch_size=64,
    buffer_size=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    tensorboard_log=log_dir,
    verbose=1
)

# Train the model
TIMESTEPS = 100000
model.learn(total_timesteps=TIMESTEPS, log_interval=10)

# Save trained model
model.save(f"{model_dir}/dqn_patient_monitoring")
print("DQN training complete and model saved.")

# Evaluate trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward} \u00b1 {std_reward}")

# Close environment
env.close()

from google.colab import files
import shutil

# Move model to Google Drive

shutil.move("models/dqn/dqn_patient_monitoring", "/content/drive/MyDrive/training")

# OR Download directly
files.download("dqn_training.zip")