import numpy as np
import PythonCodes.reinforcement_learning.epsilon_greedy as eg
   
def sarsa(env, n_iter, gamma, epsilon, eta):
    S, A, T, R = env.S, env.A, env.T, env.R
    n_states, n_actions = len(S), len(A)
    Q = np.zeros((n_states, n_actions))
    for iter in range(n_iter):
        env.reset()
        s = env.s_start
        while s not in env.S_end:
            a = eg.epsilon_greedy(Q[s], n_actions, epsilon)
            s_next = int(T[s][a])
            a_next = eg.epsilon_greedy(Q[s_next], n_actions, epsilon)
            Q[s][a] = (1 - eta) * Q[s][a] + eta * (R[s][a] + gamma * Q[s_next][a_next])
            s = s_next
    return np.argmax(Q, axis = 1)
        
#env =lat1d.LandmineAndTreasure(30, 10)
#print(sarsa(env, 500, 0.95, 0.2, 0.99, 0.8))




    
    




