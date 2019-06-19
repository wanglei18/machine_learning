import numpy as np
import gym
import matplotlib.pyplot as plt

def softmax(scores):
    e = np.exp(scores)
    s = e.sum()
    return e / s

def actor_critic(env, state_size, n_actions, n_iter, gamma, eta_u, eta_W):
    W = np.random.rand(state_size, n_actions)
    u = np.random.rand(state_size, 1)
    global_steps = 0
    I = []
    S = []
    for iter in range(n_iter):
        state = env.reset()
        done = False
        discount = 1
        while not done:
            global_steps += 1
            s_cur = state.reshape(1, state_size)
            probs = softmax(s_cur.dot(W))
            a_cur = np.random.choice(n_actions, p = probs.reshape(-1))
            state, reward, done, info = env.step(a_cur)
            if done:
                reward = -100
                delta = reward - s_cur.dot(u)
            else:
                s_next = state.reshape(1, state_size)
                delta = reward + gamma * s_next.dot(u) - s_cur.dot(u)
            y = np.zeros(n_actions)
            y[a_cur] = 1
            y = y.reshape(1, n_actions)
            gradient_W = s_cur.T.dot(probs - y)
            W = W - eta_W * discount * delta * gradient_W
            gradient_u = s_cur.T
            u = u + eta_u * discount * delta * gradient_u
            discount *= gamma
        if (iter + 1) % 10 == 0:
            I.append(iter)
            S.append(global_steps / 10)
            global_steps = 0                     
    plt.figure(0)
    plt.plot(I,S)
    plt.show() 

env = gym.make("CartPole-v0")
actor_critic(env, 4, 2, 1000, 0.95, 0.1, 0.1)     
                
                
                
                
                
                
                
            
            
        

    


                
 



















    
    




