import numpy as np
    
def compute_policy_Q_values(S, A, T, R, gamma, pi):
    V = np.zeros(len(S))
    Q = np.zeros((len(S), len(A)))
    while True:
        for s in S:
            for a in A:
                Q[s][a] = R[s][a] + gamma * V[int(T[s][a])]
        converge = True
        for s in S:
            if Q[s][pi[s]] != V[s]:
                converge = False
            V[s] = Q[s][pi[s]]
        if converge:
            break
    return Q
 
def policy_iteration(env, gamma):
    S,A,T,R = env.S, env.A, env.T, env.R
    pi = np.argmax(R, axis = 1) 
    n = 0
    while True:
        Q_pi = compute_policy_Q_values(S, A, T, R, gamma, pi)
        converge = True
        for s in S:
            if np.argmax(Q_pi[s]) != pi[s]:
                converge = False
            pi[s] = np.argmax(Q_pi[s])
        if converge:
            print(n)
            break
        n += 1
    return pi
 
        
        




    
    




