import gym
import numpy as np

num_epochs = 20
mu = np.random.rand(4)
sigma = np.random.rand(4)
theta = np.zeros(4)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    for epoch in range(num_epochs):
        observation = env.reset()
        theta = np.random.normal(mu, sigma, 4)
        for t in range(100):
            env.render()
            action = np.dot(theta, observation)
            if action < 0:
                action = 0
            else:
                action = 1
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
