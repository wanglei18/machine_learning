import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import PythonCodes.reinforcement_learning.epsilon_greedy as eg

state_size = 4
n_hidden1 = 24
n_hidden2 = 24
n_actions = 2
State = tf.placeholder(tf.float32, shape=[None, state_size])
hidden1 = tf.layers.dense(State, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
Q_values = tf.layers.dense(hidden2, n_actions)

Target = tf.placeholder(tf.float32)
Action = tf.placeholder(tf.int32)
Q_value = tf.reduce_sum(Q_values * tf.one_hot(Action, n_actions))
loss = tf.reduce_mean(tf.square(Target - Q_value))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

def epsilon_greedy(q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions) 
    else:
        return np.argmax(q) 

env = gym.make("CartPole-v1")
n_iterations = 1000
gamma = 0.95
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
stop_penalty = -100
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    epsilon = epsilon_max
    global_steps = 0
    I = []
    S = []
    for iteration in range(n_iterations):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            global_steps += 1
            s_cur = state.reshape(1, state_size)
            Q_s_cur = Q_values.eval(feed_dict={State: s_cur})
            a_cur = eg.epsilon_greedy(Q_s_cur, n_actions, epsilon)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            state, reward, done, info = env.step(a_cur)
            s_next = state.reshape(1, state_size)
            if done:
                target = stop_penalty
                sess.run(training_op, feed_dict={State: s_cur, Action: a_cur, Target: target})
                #print(steps)
                break
            else:
                Q_s_next = Q_values.eval(feed_dict={State: s_next})
                target = reward + gamma * np.max(Q_s_next, axis=1)
                sess.run(training_op, feed_dict={State: s_cur, Action: a_cur, Target: target})
        if (iteration + 1) % 100 == 0:
            print(1.0 * global_steps / 100)
            I.append(iteration + 1)
            S.append(1.0 * global_steps / 100)
            global_steps = 0
    plt.figure(0)
    plt.plot(I,S)
    plt.show()
env.close()
            
            
        
        





    
    




