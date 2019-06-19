import numpy as np
import gym
import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 1
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  
p = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p), num_samples=1)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
grads = [grad for grad, var in grads_and_vars]

grad_placeholders = []
grads_and_vars_new = []
for grad, var in grads_and_vars:
    grad_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    grad_placeholders.append(grad_placeholder)
    grads_and_vars_new.append((grad_placeholder, var))
training_op = optimizer.apply_gradients(grads_and_vars_new)

def discount_rewards(rewards, gamma):
    discount_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * gamma
        discount_rewards[step] = cumulative_rewards
    return discount_rewards

env = gym.make("CartPole-v0")
n_games = 10
n_steps = 1000
n_iterations = 250
gamma = 0.95
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        game_rewards = []
        game_grads = []
        for game in range(n_games):
            step_rewards = []
            step_grads = []
            obs = env.reset()
            for step in range(n_steps):
                action_val, grads_val = sess.run(
                    [action, grads], 
                    feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                step_rewards.append(reward)
                step_grads.append(grads_val)
                if done:
                    break
            game_rewards.append(discount_rewards(step_rewards, gamma))
            game_grads.append(step_grads)        
        feed_dict = {}
        for var_idx, grad_placeholder in enumerate(grad_placeholders):
            grads_mean = np.mean(
                [reward * game_grads[game_idx][step][var_idx]
                    for game_idx, step_rewards in enumerate(game_rewards)
                        for step, reward in enumerate(step_rewards)], axis=0)
            feed_dict[grad_placeholder] = grads_mean
        sess.run(training_op, feed_dict=feed_dict)
        obs = env.reset()
        for step in range(n_steps):
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                print("Iteration ", iteration, ": ", step)
                break
 



















    
    




