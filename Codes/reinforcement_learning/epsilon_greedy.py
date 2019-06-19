import numpy as np

def epsilon_greedy(Q_s, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions) 
    else:
        return np.argmax(Q_s)

class EpsilonGreedy:
    def __init__(self, epsilon_min, epsilon_decay, n_actions):
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.epsilon_decay = epsilon_decay
        
    def get_action(self, Q_s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) 
        else:
            return np.argmax(Q_s) 
        self.epsilon *= self.epsilon_decay
 
        
        




    
    




