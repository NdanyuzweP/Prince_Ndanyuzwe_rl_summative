import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatientMonitoringEnv(gym.Env):
    """
    Custom Gym environment for proactive patient monitoring.
    """
    def __init__(self):
        super(PatientMonitoringEnv, self).__init__()
        
        # Define discrete state space (HR, BP, SpO2, Temp)
        self.state_space = {
            'HR': ['Normal', 'Elevated', 'Critical'],
            'BP': ['Normal', 'High', 'Very High'],
            'SpO2': ['Normal', 'Low', 'Very Low'],
            'Temp': ['Normal', 'Fever', 'High Fever']
        }
        self.state_values = {key: len(values) for key, values in self.state_space.items()}
        
        # Define action space (Discrete: 4 actions)
        self.action_space = spaces.Discrete(4)  # A1 - A4
        
        # Define observation space (MultiDiscrete for categorical states)
        self.observation_space = spaces.MultiDiscrete(
            [self.state_values[key] for key in self.state_space]
        )
        
        self.state = None
        self.time_step = 0
        self.max_time_steps = 50  # Limit per episode

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Start in a random state (not always stable)
        self.state = np.random.randint(
            [values for values in self.state_values.values()]
        )
        self.time_step = 0
        return self.state, {}

    def step(self, action):
        """Execute agent action and transition to new state."""
        self.time_step += 1
        reward = 0
        done = False

        # Extract current state values
        hr, bp, spo2, temp = self.state

        # Define action responses
        if action == 0:  # A1: Continue Monitoring
            reward = 2  # Small reward for appropriate monitoring
        elif action == 1:  # A2: Send Mild Alert
            if hr == 1 or bp == 1 or spo2 == 1 or temp == 1:
                reward = 5  # Correct alert
            else:
                reward = -5  # False alarm penalty
        elif action == 2:  # A3: Request Medical Evaluation
            if hr == 2 or bp == 2 or spo2 == 2 or temp == 2:
                reward = 10  # Critical alert correct
            else:
                reward = -5  # Unnecessary escalation
        elif action == 3:  # A4: Activate Emergency Protocol
            if hr == 2 and bp == 2 and spo2 == 2 and temp == 2:
                reward = 10  # Emergency intervention needed
            else:
                reward = -10  # Overreacting penalty
        
        # Delayed rewards for proactive interventions
        if action in [1, 2] and any([hr == 1, bp == 1, spo2 == 1, temp == 1]):
            reward += 3  # Reward for early intervention

        # Transition probabilities (simulate vitals changing randomly)
        self.state = np.clip(self.state + np.random.choice([-1, 0, 1], size=4), 0, 2)
        
        # Terminal conditions
        if all(val == 0 for val in self.state):  # Patient stabilizes
            done = True
            reward += 5
        elif self.time_step >= self.max_time_steps:  # Episode ends
            done = True
        
        return self.state, reward, done, False, {}

    def render(self):
        """Render the environment state."""
        state_labels = {key: self.state_space[key][self.state[i]] for i, key in enumerate(self.state_space)}
        print(f"Time Step: {self.time_step}, State: {state_labels}")

    def close(self):
        """Close environment."""
        pass
