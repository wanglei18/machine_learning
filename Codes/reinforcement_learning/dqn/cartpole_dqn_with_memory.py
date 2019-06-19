import numpy as np
import gym
import tensorflow as tf

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
Q_value = tf.reduce_sum(Q_values * tf.one_hot(Action, n_actions), axis=1)
loss = tf.reduce_mean(tf.square(Target - Q_value))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

def epsilon_greedy(q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions) 
    else:
        return np.argmax(q) 

env = gym.make("CartPole-v0")
n_iterations = 1000
gamma = 0.95
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
min_memory = 100
stop_penalty = -100
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    memory = []
    epsilon = epsilon_max
    for iteration in range(n_iterations):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            state = obs.reshape(1, state_size)
            Q_state = Q_values.eval(feed_dict={State: state})
            action = epsilon_greedy(Q_state, epsilon)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            obs, reward, done, info = env.step(action)
            next_state = obs.reshape(1, state_size)
            memory.append((state, action, reward, next_state, done))
            if done:
                print(step)
                break
            if len(memory) > min_memory:
                idx = np.random.randint(0, len(memory))
                (state, action, reward, next_state, done) = memory[idx]
                if done:
                    target = stop_penalty
                else:
                    Q_next_state = Q_values.eval(feed_dict={State: next_state})
                    target = reward + gamma * np.max(Q_next_state, axis=1)
                sess.run(training_op, feed_dict={State: state, Action: action, Target: target})
env.close()           
            
        
        





    
    




