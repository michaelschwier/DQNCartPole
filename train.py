import gym
import numpy as np
from agent import DQNAgent

EPISODES = 1000

if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  game_over = False
  batch_size = 32
  best_score = 0

  for episodeIdx in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size]).astype(np.float32)
    for time in range(500):
      action = agent.act(state)
      next_state, reward, game_over, _ = env.step(action)
      reward = reward if not game_over else -10
      next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)
      agent.remember(state, action, reward, next_state, game_over)
      state = next_state
      if game_over:
        print("episode: {}/{}, score: {}, e: {:.2}".format(episodeIdx, EPISODES, time, agent.exploration_rate))
        break
    if (episodeIdx > 100) and (time >= best_score):
      print("Saving current model")
      agent.save("model.h5")
      best_score = time
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
