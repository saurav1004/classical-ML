import numpy as np
import random

class QLearning:
    def __init__(self, alpha, gamma, epsilon, actions):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions  # List of actions
        self.Q = {}             # Q-table

    def get_Q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def set_Q_value(self, state, action, value):
        self.Q[(state, action)] = value

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore action space
        else:
            Q_values = [self.get_Q_value(state, a) for a in self.actions]
            max_Q = max(Q_values)
            # If multiple actions have the same max Q value, randomly choose one of them
            count = Q_values.count(max_Q)
            if count > 1:
                best = [i for i in range(len(self.actions)) if Q_values[i] == max_Q]
                i = random.choice(best)
            else:
                i = Q_values.index(max_Q)
            return self.actions[i]  # Exploit learned values

    def learn(self, state, action, reward, next_state):
        old_value = self.get_Q_value(state, action)
        next_max = max([self.get_Q_value(next_state, a) for a in self.actions])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.set_Q_value(state, action, new_value)

    def train(self, initial_state, transition_function, n_episodes):
        for episode in range(n_episodes):
            state = initial_state
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = transition_function(state, action)
                self.learn(state, action, reward, next_state)
                state = next_state

# Example usage:
# Define a transition function that gives the next state and reward given the current state and action
# This is environment specific and would need to be defined based on the problem you're solving
def transition_function(state, action):
    # Implement your transition rule here
    next_state = None
    reward = None
    done = False
    return next_state, reward, done

# Define the set of actions available in the environment
actions = ['up', 'down', 'left', 'right']

# Initialize Q-Learning algorithm
q_learning = QLearning(alpha=0.1, gamma=0.9, epsilon=0.1, actions=actions)

# Train the model
initial_state = 'YourInitialStateHere'
q_learning.train(initial_state, transition_function, n_episodes=1000)

# Use q_learning.Q to access the Q-values
