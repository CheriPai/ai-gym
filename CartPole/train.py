import gym
import numpy as np
import sys

# num_samples = 18
num_samples = 10
iterations = 100
p = 0.2
mu = np.random.rand(4)
sigma = np.random.rand(4)
theta = np.zeros((iterations, 4))
scores = np.zeros(num_samples)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    while True:
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
        top_p = int(num_samples * p)
        top = scores.argsort()[-top_p:][::-1]
        mu = np.mean(theta[top], axis=0)
        scores = sorted(scores, reverse=True)
        scores = np.sort(scores)[::-1]
        if scores[0] >= 95:
            print("Solution", theta[top[0]])
            sys.exit(0)


