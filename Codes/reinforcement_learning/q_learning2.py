import numpy as np

n_states = 10
S = range(n_states)
A = [0, 1]
n_actions = len(A)
R = np.zeros((n_states, n_actions))
R[n_states-2][1] = 1
R[1][0] = -1
T = np.zeros((n_states, n_actions))
for s in range(n_states):
    T[s][0] = max(0, s-1)
    T[s][1] = min(n_states - 1, s + 1)
    
def epsilon_greedy(Q, s, epsilon):
    r = np.random.rand()
    if r < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(Q[s])
             
def q_learning(n_iter, gamma, epsilon, eta):
    Q = np.zeros((n_states, n_actions))
    for iter in range(n_iter):
        s = int(n_states / 2)
        counts = 0
        while s < n_states - 1 and s > 0:
            a = epsilon_greedy(Q, s, epsilon)
            s_next = int(T[s][a])
            a_next = epsilon_greedy(Q, s_next, epsilon)
            Q[s][a] = (1 - eta) * Q[s][a] + eta * (R[s][a] + gamma * Q[s_next][a_next])
            s = s_next
            counts += 1
        print("iteration ", iter, " takes ", counts, " steps")
    pi = np.zeros((n_states, n_actions))
    for s in S:
        a = np.argmax(Q[s])
        pi[s][a] = 1
    return pi
        
print(q_learning(500, 0.8, 0.5, 0.2))




    
    




