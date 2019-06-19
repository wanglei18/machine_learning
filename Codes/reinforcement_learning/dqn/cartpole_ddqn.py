import numpy as np
import gym
import tensorflow as tf
import PythonCodes.reinforcement_learning.dqn.dqn_util as util

n_inputs = 4
n_hidden1 = 10
n_outputs = 2

def q_network(X_state, name):
    with tf.variable_scope(name) as scope:
        hidden = tf.layers.dense(X_state, n_hidden1, activation=tf.nn.relu)
        q_values = tf.layers.dense(hidden, n_outputs)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return q_values, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, n_inputs])
online_q_values, online_vars = q_network(X_state, name="online")
target_q_values, target_vars = q_network(X_state, name="target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)
 
y = tf.placeholder(tf.float32, shape=[None, 1])
X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)
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
n_steps = 1000000
iterations_per_copy = 100
n_iterations = 10000
gamma = 0.95
epsilon = 0.5
replay_memory_size = 500000
batch_size = 50

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    replay_memory = util.ReplayMemory(replay_memory_size)
    for iteration in range(n_iterations):
        print("Iteration: ", iteration)
        for game in range(n_games):
            obs = env.reset()
            for step in range(n_steps):
                s = obs.reshape(1, n_inputs)
                Q_s = online_q_values.eval(feed_dict={X_state: s})
                action = epsilon_greedy(Q_s, epsilon)
                obs, reward, done, info = env.step(action)
                s_next = obs.reshape(1, n_inputs)
                replay_memory.append((s, action, reward, s_next, 1.0 - done))
                
                s, action, rewards, s_next, continues = replay_memory.sample(batch_size)[0]
                Q_s_next = target_q_values.eval(feed_dict={X_state: s_next})
                y_val = reward + gamma * np.max(Q_s_next, axis=1, keepdims=True)
                sess.run(training_op, feed_dict={X_state: s, X_action: [action], y: y_val})
                if done:
                    break
        if iteration % iterations_per_copy == 0:
            copy_online_to_target.run()
        obs = env.reset()
        for step in range(n_steps):
            s = obs.reshape(1, n_inputs)
            Q_s = online_q_values.eval(feed_dict={X_state: s})
            epsilon2 = 0.01
            action = epsilon_greedy(Q_s, epsilon2)
            obs, reward, done, info = env.step(action)
            if done:
                print(step)
                break
    saver.save(sess, "./cartpole_dqn_mode/cartpole_dqn.ckpt")






    
    




