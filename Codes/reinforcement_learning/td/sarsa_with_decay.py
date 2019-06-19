import numpy as np
import PythonCodes.reinforcement_learning.epsilon_greedy as eg
   
def sarsa(env, n_iter, gamma, epsilon_min, epsilon_decay, eta):
    S, A, T, R = env.S, env.A, env.T, env.R
    n_states, n_actions = len(S), len(A)
    Q = np.zeros((n_states, n_actions))
    epsilon_greedy = eg.EpsilonGreedy(epsilon_min, epsilon_decay, n_actions)
    for iter in range(n_iter):
        env.reset()
        s = env.s_start
        a = epsilon_greedy.get_action(Q[s])
        while s not in env.S_end:
            s_next = int(T[s][a])
            a_next = epsilon_greedy.get_action(Q[s_next])
            Q[s][a] = (1 - eta) * Q[s][a] + eta * (R[s][a] + gamma * Q[s_next][a_next])
            s = s_next
            a = a_next
    return np.argmax(Q, axis = 1)
        
#env =lat1d.LandmineAndTreasure(30, 10)
#print(sarsa(env, 500, 0.95, 0.2, 0.99, 0.8))




    
    




