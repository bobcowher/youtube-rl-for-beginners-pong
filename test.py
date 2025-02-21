from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation 
import ale_py

total_steps = 0
step_repeat = 4

learning_rate = 0.0001
gamma = 0.99

hidden_layer = 128

# print(observation.shape)

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = gym.make("ALE/Pong-v5", render_mode="human")


env = ResizeObservation(env, (64, 64))

env = GrayscaleObservation(env, keep_dim=True)

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)


# Training Phase 1

agent.test() 

