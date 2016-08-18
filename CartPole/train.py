import gym
import numpy as np

num_epochs = 20
theta = np.zeros(4)
sigma = np.zeros(4)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    for epoch in range(num_epochs):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
