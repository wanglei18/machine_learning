import numpy as np
import gym
import tensorflow as tf

n_inputs = 4
n_hidden1 = 10
n_outputs = 2
X_state = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X_state, n_hidden1, activation=tf.nn.relu)
q_values = tf.layers.dense(hidden, n_outputs)
 
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
n_games = 100
n_steps = 100
n_iterations = 1000
gamma = 0.95
epsilon = 0.5
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        for game in range(n_games):
            obs = env.reset()
            for step in range(n_steps):
                s = obs.reshape(1, n_inputs)
                Q_s = q_values.eval(feed_dict={X_state: s})
                a = epsilon_greedy(Q_s, epsilon)
                obs, reward, done, info = env.step(a)
                s_next = obs.reshape(1, n_inputs)
                Q_s_next = q_values.eval(feed_dict={X_state: s_next})
                y_val = reward + gamma * np.max(Q_s_next, axis=1, keepdims=True)
                sess.run(training_op, feed_dict={X_state: s, X_action: [a], y: y_val})
                if done:
                    break
        
    obs = env.reset()
    for step in range(n_steps):
        print("step = ", step)
        s = obs.reshape(1, n_inputs)
        Q_s = q_values.eval(feed_dict={X_state: s})
        a = epsilon_greedy(Q_s, epsilon)
        obs, reward, done, info = env.step(a)
        if done:
            break
    saver.save(sess, "./cartpole_dqn_mode/cartpole_dqn.ckpt")






    
    




