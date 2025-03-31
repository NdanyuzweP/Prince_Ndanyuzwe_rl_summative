from fastapi import FastAPI
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model_path = "models/final_dqn_patient_monitoring.zip"
model = DQN.load(model_path)

# Create the environment
env = gym.make("environment/PatientMonitoringEnv-v0")  # Ensure this matches your environment name

@app.get("/")
def home():
    return {"message": "Patient Monitoring API is running!"}

@app.post("/predict")
def predict(state: list):
    """
    Accepts a patient's state as input and returns the model's action.
    Example input: [0, 1, 2, 0] (HR, BP, SpO2, Temp)
    """
    state_array = np.array(state).reshape(1, -1)
    action, _ = model.predict(state_array, deterministic=True)
    return {"action": int(action)}

# Run the server (if running locally)
# uvicorn main:app --host 0.0.0.0 --port 8000