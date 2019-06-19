import numpy as np
import gym

def softmax(scores):
    e = np.exp(scores)
    s = e.sum()
    return e / s

def REINFORCE(env, state_size, n_actions, n_iter, gamma, eta):
    W = np.random.rand(state_size, n_actions) 
    for iter in range(n_iter):
        state = env.reset()
        done = False
        discount = 1
        steps = 0
        while not done:
            steps += 1
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
                print("iteration {} lasts for {} steps".format(iter, steps))
            W = W - eta * discount * reward * gradient
            discount *= gamma
                
env = gym.make("CartPole-v0")
REINFORCE(env, 4, 2, 3000, 0.95, 0.1)     
                
                
                
                
                
                
                
            
            
        

    


                
 



















    
    




