import random
import gym
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Multiply
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


  # Neural Net for Deep-Q learning Model
  def _build_model(self):
    state_inputs = Input(shape=(self.state_size,))
    action_inputs = Input(shape=(self.action_size,))
    x = Dense(24, activation='relu')(state_inputs)
    x = Dense(24, activation='relu')(x)
    predictions = Dense(self.action_size, activation='linear')(x)
    masked_predictions = Multiply()([predictions, action_inputs])
    model = Model(inputs=[state_inputs, action_inputs], outputs=predictions)
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model


  def remember(self, state, action, reward, next_state, game_over):
    action_one_hot = np.zeros((1, self.action_size), dtype=np.float32)
    np.put(action_one_hot, action, 1)
    self.memory.append((state, action_one_hot, reward, next_state, game_over))


  def act(self, state):
    if np.random.rand() < self.exploration_rate:
      return random.randrange(self.action_size)
    else:
      act_values = self.model.predict([state, np.ones((1, self.action_size))])
      return np.argmax(act_values[0])


  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action_one_hot, reward, next_state, game_over in minibatch:
      target_values = np.empty(action_one_hot.shape, dtype=np.float32)
      target_values.fill(reward)
      if not game_over:
        est_values = self.model.predict([next_state, np.ones(action_one_hot.shape)])
        target_values = target_values + self.gamma * np.amax(est_values, axis=1)
      self.model.fit([state, action_one_hot], action_one_hot * target_values, 
                     epochs=1, batch_size=len(state), verbose=0)
    if self.exploration_rate > self.exploration_rate_min:
      self.exploration_rate *= self.exploration_rate_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)

