import numpy as np

def get_environment(m, d):
    n_states = m
    n_actions = 2
    S = range(n_states)
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
        if s == 0 or s == n_states - 1 or s == d:
            T[s][0] = s
            T[s][1] = s
    return S,A,T,R    
         
        




    
    




