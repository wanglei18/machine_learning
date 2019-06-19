import matplotlib.pyplot as plt
import gym

env = gym.make("CartPole-v0")
state = env.reset()
print(state)

step = 0
action = 1
while True:
    step += 1
    state, reward, done, info = env.step(action) 
    plt.figure(step)
    plt.imshow(env.render(mode = "rgb_array"))
    if done:
        break
plt.show()








    
    




