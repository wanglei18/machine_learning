import numpy as np
import gym
import matplotlib.pyplot as plt

def softmax(scores):
    e = np.exp(scores)
    s = e.sum()
    return e / s

def REINFORCE(env, state_size, n_actions, n_iter, gamma, eta):
    W = np.random.rand(state_size, n_actions) 
    global_steps = 0
    I = []
    S = []
    for iter in range(n_iter):
        state = env.reset()
        done = False
        discount = 1
        while not done:
            global_steps += 1
            s = state.reshape(1, state_size)
            probs = softmax(s.dot(W))
            a = np.random.choice(n_actions, p = probs.reshape(-1))
            state, reward, done, info = env.step(a)
            y = np.zeros(n_actions)
            y[a] = 1
            y = y.reshape(1, n_actions)
            gradient = s.T.dot(probs - y)
            if done:
                reward = -100
            W = W - eta * discount * reward * gradient
            discount *= gamma
        if (iter + 1) % 10 == 0:
            I.append(iter)
            S.append(global_steps / 10)
            global_steps = 0
    
    plt.figure(0)
    plt.plot(I,S)
    plt.show()        
                
env = gym.make("CartPole-v0")
REINFORCE(env, 4, 2, 1000, 0.95, 0.1)     
                
                
                
                
                
                
                
            
            
        

    


                
 



















    
    




