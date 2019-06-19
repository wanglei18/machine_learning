import numpy as np
import gym
import tensorflow as tf

n_inputs = 4
n_hidden1 = 4
n_hidden2 = 4
n_outputs = 2
X_state = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.layers.dense(X_state, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
q_values = tf.layers.dense(hidden2, n_outputs)
 
y = tf.placeholder(tf.float32, shape=[None, 1])
X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)
loss = tf.reduce_mean(tf.square(y - q_value))
optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.95, use_nesterov=True)
training_op = optimizer.minimize(loss)

def epsilon_greedy(q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) 
    else:
        return np.argmax(q) 

env = gym.make("CartPole-v0")
n_games = 10
n_steps = 1000
n_iterations = 250
gamma = 0.95
epsilon = 0.01
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./cartpole_dqn_mode/cartpole_dqn.ckpt")
    obs = env.reset()
    for step in range(n_steps):
        print("step = ", step)
        s = obs.reshape(1, n_inputs)
        Q_s = q_values.eval(feed_dict={X_state: s})
        a = epsilon_greedy(Q_s, epsilon)
        obs, reward, done, info = env.step(a)
        if done:
            break
env.close()
    
    
    






    
    




