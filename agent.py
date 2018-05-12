import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:

  def __init__(self, state_size, action_size, init_exploration_rate = 1.0):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95    # discount rate
    self.exploration_rate = init_exploration_rate
    self.exploration_rate_min = 0.01
    self.exploration_rate_decay = 0.995
    self.learning_rate = 0.001
    self.model = self._build_model()


  def _build_model(self):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model


  def remember(self, state, action, reward, next_state, game_over):
    self.memory.append((state, action, reward, next_state, game_over))


  def act(self, state):
    if np.random.rand() < self.exploration_rate:
      return random.randrange(self.action_size)
    else:
      act_values = self.model.predict(state)
      return np.argmax(act_values[0])


  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, game_over in minibatch:
      target = reward
      if not game_over:
        target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
      target_f = self.model.predict(state)
      target_f[0][action] = target
      self.model.fit(state, target_f, epochs=1, verbose=0)
    if self.exploration_rate > self.exploration_rate_min:
      self.exploration_rate *= self.exploration_rate_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)

