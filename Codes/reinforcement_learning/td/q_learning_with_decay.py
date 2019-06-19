import numpy as np
import PythonCodes.reinforcement_learning.epsilon_greedy as eg
      
def q_learning(env, n_iter, gamma, epsilon_min, epsilon_decay, eta):
    S, A, T, R = env.S, env.A, env.T, env.R
    n_states, n_actions = len(S), len(A)
    Q = np.zeros((n_states, n_actions))
    epsilon_greedy = eg.EpsilonGreedy(epsilon_min, epsilon_decay, n_actions)
    for iter in range(n_iter):
        env.reset()
        s = env.s_start
        while s not in env.S_end:
            a = epsilon_greedy.get_action(Q[s])
            s_next = int(T[s][a])
            a_next = np.argmax(Q[s])
            Q[s][a] = (1 - eta) * Q[s][a] + eta * (R[s][a] + gamma * Q[s_next][a_next])
            s = s_next
    pi = np.argmax(Q, axis = 1)
    return pi





    
    




