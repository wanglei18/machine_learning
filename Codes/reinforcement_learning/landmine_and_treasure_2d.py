import numpy as np

class LandmineAndTreasure2d:
    def __init__(self, m, n):
        n_states = m * n
        n_actions = 4
        S = range(n_states)
        S_end = range(n)
        A = range(n_actions)
        # 0: up; 1: right; 2: down; 3:left 
        R = np.zeros((n_states, n_actions))
        for j in range(n-1):
            R[n + j][0] = -1000
        R[2 * n - 1][0] = 100
        T = np.zeros((n_states, n_actions))
        for i in range(m):
            for j in range(n):
                s = n * i + j
                T[s][0] = s - n if i > 0 else s 
                T[s][1] = s + 1 if j < n - 1 else s 
                T[s][2] = s + n if i < m - 1 else s 
                T[s][3] = s - 1 if j > 0 else s 
        for s in S_end:
            for a in A:
                T[s][a] = s                    
        self.S, self.A, self.T, self.R, self.S_end = S, A, T, R, S_end
           
    def reset(self):
        self.s_start = np.random.randint(0, len(self.S))    
        




    
    




