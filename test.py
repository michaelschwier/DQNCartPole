import gym
import numpy as np
from agent import DQNAgent

EPISODES = 10


# create an environment that runs longer than the default
def getLongRunEnv():
  gym.envs.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    reward_threshold=4750.0,
  )
  return  gym.make('CartPole-v2')


if __name__ == "__main__":
  env = getLongRunEnv()
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size, init_exploration_rate=0.0)
  agent.load("model.h5")

  for test_episode_idx in range(EPISODES):
    state = env.reset()
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
