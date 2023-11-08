import numpy as np
import random
from collections import deque

class DQN:
    def __init__(self, state_dim, action_dim, hidden_sizes, gamma, epsilon, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.lr = lr  # Learning rate
        self.replay_buffer = deque(maxlen=10000)
        self.model = self.build_model(state_dim, action_dim, hidden_sizes)
        self.target_model = self.build_model(state_dim, action_dim, hidden_sizes)
        self.update_target_model()

    def build_model(self, state_dim, action_dim, hidden_sizes):
        # Simplified model, in practice use a neural network framework
        # For example: [64, 64] means two hidden layers with 64 neurons each
        layers = [state_dim] + hidden_sizes + [action_dim]
        model = {'weights': [], 'biases': []}
        for i in range(len(layers) - 1):
            model['weights'].append(np.random.randn(layers[i], layers[i+1]))
            model['biases'].append(np.zeros(layers[i+1]))
        return model

    def update_target_model(self):
        self.target_model = self.model.copy()

    def predict(self, model, state):
        # Simplified forward pass
        activation = state
        for w, b in zip(model['weights'], model['biases']):
            activation = np.dot(activation, w) + b
            activation = np.maximum(activation, 0)  # ReLU activation
        return activation

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.predict(self.model, state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.predict(self.target_model, next_state))
            target_f = self.predict(self.model, state)
            target_f[0][action] = target
            self.fit(self.model, state, target_f)

    def fit(self, model, state, target_f):
        # Simplified backpropagation
        # In practice, use a neural network framework with optimizer and loss function
        pass

# Example usage:
# state_dim = 4  # Example for CartPole
# action_dim = 2  # Example for CartPole
# hidden_sizes = [64, 64]
# gamma = 0.95
# epsilon = 1.0
# lr = 0.001
# dqn = DQN(state_dim, action_dim, hidden_sizes, gamma, epsilon, lr)

# Train the DQN with experiences
# for e in range(n_episodes):
#     state = env.reset()
#     state = np.reshape(state, [1, state_dim])
#     for time in range(500):
#         action = dqn.act(state)
#         next_state, reward, done, _ = env.step(action)
#         reward = reward if not done else -10
#         next_state = np.reshape(next_state, [1, state_dim])
#         dqn.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print("episode: {}/{}, score: {}".format(e, n_episodes, time))
#             break
#     if len(dqn.replay_buffer) > batch_size:
#         dqn.replay(batch_size)
