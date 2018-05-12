import gym
import numpy as np
from agent import DQNAgent

EPISODES = 10

if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size, init_exploration_rate=0.0)
  agent.load("model.h5")

  for test_episode_idx in range(EPISODES):
    env.reset()
    # take a random step in the beginning to not always start the same way
    action = env.action_space.sample()
    state, _, _, _ = env.step(action)
    state = np.reshape(state, [1, state_size]).astype(np.float32)
    score = 0
    game_over = False
    while not game_over:
      env.render()
      action = agent.act(state)
      state, reward, game_over, _ = env.step(action)
      state = np.reshape(state, [1, state_size]).astype(np.float32)
      score += reward
      if game_over:
        print("Finished with score: {}".format(score))

  env.close()
