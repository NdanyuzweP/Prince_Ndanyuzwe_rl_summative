import numpy as np
from custom_env import PatientMonitoringEnv

# Create environment
env = PatientMonitoringEnv()

# Reset environment
obs, _ = env.reset()
print(f"Initial State: {obs}")

# Run test episode
done = False
step_count = 0

while not done and step_count < 10:  # Run for 10 steps or until done
    action = np.random.choice(4)  # Randomly pick an action (A1 - A4)
    obs, reward, done, _, _ = env.step(action)
    
    print(f"Step {step_count + 1}: Action {action}, Reward {reward}, New State {obs}")
    env.render()
    
    step_count += 1

# Close environment
env.close()
print("Test complete.")
