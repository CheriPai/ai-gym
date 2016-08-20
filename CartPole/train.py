import gym
import numpy as np

num_samples = 40
p = 0.2
mu = np.random.rand(4)
sigma = np.random.rand(4)
theta = np.zeros((100, 4))
scores = np.zeros(100)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    for i in range(num_samples):
        observation = env.reset()
        theta[i] = np.random.normal(mu, sigma, 4)
        for t in range(100):
            env.render()
            action = np.dot(theta[i], observation)
            if action < 0:
                action = 0
            else:
                action = 1
            observation, reward, done, info = env.step(action)
            if done:
                scores[i] = t+1
                print("Episode finished after {} timesteps".format(scores[i]))
                break
