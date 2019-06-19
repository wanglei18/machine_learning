import gym
import matplotlib.pyplot as plt

env = gym.make("MsPacman-v0")
print(env.action_space)
obs = env.reset()
plt.imshow(obs)
plt.show()









    
    




