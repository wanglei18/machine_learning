import numpy as np
        
def compute_environment_Q_values(S, A, T, R, gamma):
    V = np.zeros(len(S))
    Q = np.zeros((len(S), len(A)))
    n = 0
    while True:
        n += 1
        for s in S:
            for a in A:
                Q[s][a] = R[s][a] + gamma * V[int(T[s][a])]
        converge = True    
        for s in S:    
            if np.max(Q[s]) != V[s]:
                converge = False
            V[s] = np.max(Q[s])
        if converge:
            print(n)
            break
    return Q

def value_iteration(env, gamma):
    S,A,T,R = env.S, env.A, env.T, env.R
    Q = compute_environment_Q_values(S,A,T,R, gamma)
    pi = np.argmax(Q, axis = 1) 
    return pi
 
         
        




    
    




