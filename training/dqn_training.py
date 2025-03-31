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
