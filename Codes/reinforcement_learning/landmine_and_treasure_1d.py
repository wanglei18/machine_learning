import numpy as np
from PythonCodes.reinforcement_learning.environment import Environment

class LandmineAndTreasure(Environment):
    def __init__(self, m, d):
        n_states = m
        n_actions = 2
        S = range(n_states)
        S_end = {0, d, n_states - 1}
        A = range(n_actions)
        R = np.zeros((n_states, n_actions))
        R[n_states - 2][1] = 1000
        R[1][0] = 1000
        R[d + 1][0] = -1000
        R[d - 1][1] = -1000
        T = np.zeros((n_states, n_actions))
        for s in S:
            T[s][0] = s - 1
            T[s][1] = s + 1
            if s in S_end:
                T[s][0] = s
                T[s][1] = s
        self.S, self.A, self.T, self.R, self.S_end = S, A, T, R, S_end
           
        




    
    




