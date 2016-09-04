import gym
import numpy as np

num_samples = 40
iterations = 100
p = 0.2
mu = np.random.rand(4)
sigma = np.random.rand(4)
theta = np.zeros((iterations, 4))
scores = np.zeros(iterations)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    for i in range(num_samples):
        observation = env.reset()
        theta[i] = np.random.normal(mu, sigma, 4)
        for t in range(iterations):
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
        top_p = num_samples * p
        top = scores.argsort()[-top_p:][::-1]
        print(top)
