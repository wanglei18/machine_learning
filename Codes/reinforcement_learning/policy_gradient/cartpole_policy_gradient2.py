import numpy as np
import gym
import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 1
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
Hidden = tf.layers.dense(X, n_hidden, activation = tf.nn.relu)
Logits = tf.layers.dense(Hidden, n_outputs)
Outputs = tf.nn.sigmoid(Logits)  
P = tf.concat(axis = 1, values = [Outputs, 1 - Outputs])
Action = tf.multinomial(tf.log(P), num_samples = 1)

Y = 1.0 - tf.to_float(Action)
Cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = Logits)
Optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
Grads_and_vars = Optimizer.compute_gradients(Cross_entropy)
Grads = [Grad for Grad, Var in Grads_and_vars]

Grad_placeholders = []
Grads_and_vars_new = []
for Grad, Var in Grads_and_vars:
    Grad_placeholder = tf.placeholder(tf.float32, shape=Grad.get_shape())
    Grad_placeholders.append(Grad_placeholder)
    Grads_and_vars_new.append((Grad_placeholder, Var))
Training_op = Optimizer.apply_gradients(Grads_and_vars_new)

def discount_rewards(rewards, gamma):
    discount_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * gamma
        discount_rewards[step] = cumulative_rewards
    return discount_rewards

env = gym.make("CartPole-v0")
n_games = 50
n_iterations = 100
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
            done = False
            while not done:
                action, grads = sess.run(
                    [Action, Grads], 
                    feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action[0][0])
                step_rewards.append(reward)
                step_grads.append(grads)
            game_rewards.append(discount_rewards(step_rewards, gamma))
            game_grads.append(step_grads)        
        feed_dict = {}
        for var_idx, Grad_placeholder in enumerate(Grad_placeholders):
            grads_mean = np.mean(
                [reward * game_grads[game_idx][step][var_idx]
                    for game_idx, step_rewards in enumerate(game_rewards)
                        for step, reward in enumerate(step_rewards)], axis=0)
            feed_dict[Grad_placeholder] = grads_mean
        sess.run(Training_op, feed_dict=feed_dict)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            action = Action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action[0][0])
            if done:
                print("Iteration ", iteration, ": ", steps)
                
 



















    
    




