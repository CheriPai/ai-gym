import gym
import json
import numpy as np
from gym import wrappers
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i] = state_t

            # There should be no target values for actions not taken.
            predictions = model.predict(np.concatenate((state_t, state_tp1), axis=0))
            targets[i] = predictions[0][0]
            Q_sa = np.max(predictions[1][0])

            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":

    env = gym.make("LunarLander-v2")
    wrappers.Monitor(env, "LunarLander-experiment-1")

    # parameters
    epsilon = 0.1
    num_actions = env.action_space.n
    epoch = 999
    max_memory = 100000
    hidden_size = 100
    batch_size = 128
    grid_size = env.observation_space.shape[0]
    max_steps = 600

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size, ), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(lr=.002), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Define environment/game

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    for e in range(epoch):
        # Use linearly decreasing function for random action hyperparameter
        loss = 0.
        # get initial input
        input_t = np.array(env.reset()).reshape(1, -1)
        game_over = False
        total_reward = 0

        for step in range(max_steps):
            env.render()
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1).squeeze()
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over, info = env.step(action)
            input_t = np.array(input_t).reshape(1, -1)
            total_reward += reward

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

            if game_over:
                break

        print("Epoch {:03d}/{} | Loss {:.4f} | Reward {}".format(e, epoch, loss, total_reward))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
