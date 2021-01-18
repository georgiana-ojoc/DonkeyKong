from collections import deque
from datetime import datetime
from positions import get_all_positions

import copy
import cv2
import imageio
import keyboard
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import retro
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

positions = get_all_positions()


class Action(object):
    def __init__(self, ):
        self.A = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.LEFT = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.DOWN = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.UP = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.B = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_action(action_number):
        A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        RIGHT = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
        LEFT = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        DOWN = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
        UP = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        B = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

        if action_number == 0:
            return A
        elif action_number == 1:
            return RIGHT
        elif action_number == 2:
            return LEFT
        elif action_number == 3:
            return DOWN
        elif action_number == 4:
            return UP
        elif action_number == 5:
            return B
        elif action_number == 6:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    def get_action_number(actions):
        actions_number = []
        for action in actions:
            action = action.detach().cpu().numpy()[::-1]
            index = np.argmax(action)
            if action.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0]:
                index = 6
            elif action.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
                index = 5
            actions_number += [index]
        return np.array(actions_number)


class ReplayBuffer:
    def __init__(self, device, maximum_size=1_000_000):
        self.device = device
        self.buffer = deque(maxlen=maximum_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size=256):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        # batch = random.sample(self.buffer, batch_size)
        batch = list(self.buffer)[-batch_size:]

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return {"state": torch.FloatTensor(state_batch).to(self.device),
                "action": torch.FloatTensor(action_batch).to(self.device),
                "reward": torch.FloatTensor(reward_batch).to(self.device),
                "next_state": torch.FloatTensor(next_state_batch).to(self.device),
                "done": torch.FloatTensor(done_batch).to(self.device)}

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        std = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv1.weight, mean=0.0, std=std)
        self.conv1.bias.data.fill_(0.0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        std = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv2.weight, mean=0.0, std=std)
        self.conv2.bias.data.fill_(0.0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv3.weight, mean=0.0, std=std)
        self.conv3.bias.data.fill_(0.0)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        self.fc1.bias.data.fill_(0.0)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, num_actions)

    def forward(self, states):
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten imathut.
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class QModule(nn.Module):
    def __init__(self):
        super(QModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(64 * 32, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, 7)

    def forward(self, state):
        output = F.relu(self.conv1(state))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.fc1(output.view(output.size(), -1)))
        V = self.V(output)
        A = self.A(output)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class DQNAgent(object):
    def __init__(self):
        self.env = None

        # Actor and target Actor using Adam optimizer
        self.model = QModule().to(device)
        self.target_model = copy.deepcopy(self.model)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # Replay memory for training
        self.BATCH_SIZE = 10
        self.REPLAY_MEMORY_SIZE = 1_000_000
        self.replay_memory = ReplayBuffer(device)

        # constants
        self.EPS_MIN = 0.01
        self.EPS_EP = 100
        self.GAMMA = 0.99
        self.TAU = .005

    def set(self, env):
        self.env = env

    def memorize(self, current_state, action, reward, next_state, done):
        self.replay_memory.push(current_state, action, reward, next_state, done)

    def act(self, state, episode):
        # exploration = max((self.EPS_MIN - 1) / 10 * episode + 1, self.EPS_MIN)
        # if random.random() < exploration:
        # return Action.get_action(random.randint(0, 6))

        state = np.array([state])
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            output = self.model(state).detach().cpu().numpy()
            raw_action = np.argmax(output)

        action = Action.get_action(raw_action)
        return action

    def update_target(self, ):
        with torch.no_grad():
            for model_kernel, model_target_kernel in zip(self.model.parameters(), self.target_model.parameters()):
                model_target_kernel.copy_((1 - self.TAU) * model_target_kernel + self.TAU * model_kernel)

    def train(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        samples = self.replay_memory.sample(self.BATCH_SIZE)
        self.train_with_sample(samples)

    def train_with_sample(self, samples):

        current_state, action, reward, next_state, done = \
            samples["state"], samples["action"], samples["reward"], samples["next_state"], samples["done"]

        with torch.no_grad():
            next_action = self.model(next_state).argmax(dim=-1, keepdim=True)
            next_q = self.target_model(next_state).gather(1, next_action).squeeze()
            Y = reward + (1.0 - done) * self.GAMMA * next_q

        action = torch.LongTensor(Action.get_action_number(action)).to(device)
        current_q = self.model(current_state).gather(1, action.unsqueeze(dim=-1)).squeeze()
        loss = F.mse_loss(current_q, Y.detach())

        # Update critic model using the previous computed loss
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        self.update_target()


def downscale(state, x, y):
    gray_image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    max_h, max_w = gray_image.shape
    h = 50
    w = 75
    y = y if y > h else h
    y = y if y + h < max_h else max_h - h
    x = x if x > w else w
    x = x if x + w < max_w else max_w - w
    frame = gray_image[y - h:y + h, x - w:x + w]
    height = int(frame.shape[0] / 4)
    width = int(frame.shape[1] / 4)
    frame = cv2.resize(frame, (width, height))
    return np.array([cv2.resize(frame, (width, height))])


def add_movies(agent):
    path = os.path.join(os.getcwd(), "movies")
    for _, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] != ".bk2":
                continue
            print(file)
            movie = retro.Movie(os.path.join(path, file))
            movie.step()
            environment = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL,
                                     players=movie.players)
            environment.initial_state = movie.get_state()
            current_frame = environment.reset()
            current_frame = downscale(current_frame, 0, 0)
            steps = 0
            prev = None
            while movie.step():
                steps += 1
                action = []
                for player in range(movie.players):
                    for index in range(environment.num_buttons):
                        action.append(movie.get_key(index, player))
                next_frame, reward, done, info = environment.step(action)
                # environment.render()
                if done:
                    break
                next_frame = downscale(next_frame, info['y'], info['x'])
                info["reward"] = reward
                info["action"] = action
                added_reward = compute_added_reward(info, prev, movie=True)
                reward += 2 * added_reward
                agent.memorize(current_frame, action, reward, next_frame, done)
                if steps % 10 == 0:
                    agent.train()
                prev = info
                current_frame = next_frame
            environment.close()


def compute_added_reward(info, prev, movie=False):
    if movie:
        left = 14
        right = 213
    else:
        left = 30
        right = 202
    reward = 0
    if info['x'] + info['y'] == 0 or prev is None:
        return 0
    if info["status"] != 255:
        if info['x'] == prev['x'] and info['y'] == prev['y']:
            for ladder in positions["broken_ladders"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward -= 50
                    break
        if info["y"] != prev['y']:
            if info["button"] == 8:  # Urca scara
                if info['y'] < prev['y']:
                    reward += 100
            reward += 50
        if info['x'] != prev['x']:
            reward += 10
            finish = positions["finish"][0]
            if finish['x'] - 1 <= info['x'] <= finish['x'] + 1 and finish['y'] - 1 <= info['y'] <= finish['y'] + 1:
                reward += 500
                return reward
            for ladder in positions["ladders"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward += 50
                    break
            for ladder in positions["hammers"]:
                if ladder['x'] - 1 <= info['x'] <= ladder['x'] + 1 and ladder['y'] - 1 <= info['y'] <= ladder['y'] + 1:
                    reward += 50
                    break
        elif (info['y'] == positions["start"][0]['y'] and info['x'] <
              positions["start"][0]['x']) or (info['x'] < left or info['x'] > right):
            reward -= 500
    elif prev["status"] != info["status"]:
        reward -= 50
    return reward


def main():
    path = os.path.join(os.getcwd(), "GIFs")
    if not os.path.exists(path):
        os.mkdir(path)

    # TRAIN PHASE
    agent = DQNAgent()
    add_movies(agent)
    env = retro.make(game="DonkeyKong-Nes")
    agent.set(env)
    rewards_per_episode = []
    show_render = False
    for episode in range(1_000):
        start_time = time.time()
        print("EPISODE: ", episode)

        images = []
        best_x = positions["start"][0]['x']
        best_y = positions["start"][0]['y']

        current_state = env.reset()
        current_state = downscale(current_state, 0, 0)

        steps = 0
        done = False
        prev = None
        rewards = []
        actions = []

        while not done:
            if keyboard.is_pressed('f12'):
                while keyboard.is_pressed('f12'):
                    pass
                show_render = ~ show_render
            if show_render:
                env.render()

            action = agent.act(current_state, episode)
            # action = Action.get_action(1)
            next_state, reward, done, info = env.step(action)
            next_state = downscale(next_state, info['y'], info['x'])

            images += [env.render(mode="rgb_array")]
            if info['status'] != 4 and 0 < info['y'] <= best_y:
                best_x = info['x']
                best_y = info['y']

            info['reward'] = reward
            info['action'] = action
            reward += compute_added_reward(info, prev)
            actions.append(action)
            rewards.append(reward)
            # print(reward, env.get_action_meaning(action))

            agent.memorize(current_state, action, reward, next_state, done)
            if steps % 10:
                agent.train()

            steps += 1
            current_state = next_state
            prev = info

        rewards_per_episode += [np.array(rewards).sum()]
        print("TOTAL EPISODE REWARD: ", np.array(rewards).sum())
        print("BEST EPISODE REWARD: ", np.array(rewards).max())
        print("AVERAGE EPISODE REWARD: ", np.array(rewards).mean())
        print("TIME:", time.time() - start_time)
        print("STEPS: ", steps)
        print()

        if best_y <= 205:
            imageio.mimsave(os.path.join(path, str(best_y) + ' ' + str(best_x) + ' ' +
                                         str(int(rewards_per_episode[-1])) + ' ' +
                                         datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".gif"),
                            [np.array(image) for image in images], fps=30)

    # plot_reward(rewards_per_episode, range(1000))
    # TEST PHASE
    while True:
        input("PRESS ANY KEY FOR DEMONSTRATION...")
        current_state = env.reset()
        steps = 0
        done = False
        rewards = []

        while not done:
            env.render()
            action = agent.act(current_state, 1_000_000)
            next_state, reward, done, info = env.step(action)
            rewards += [reward]
            current_state = next_state
            steps += 1

        print("TOTAL EPISODE REWARD:", np.array(rewards).sum())
        print("BEST EPISODE REWARD:", np.array(rewards).max())
        print("AVERAGE EPISODE REWARD:", np.array(rewards).mean())
        print("TIME:", time.time() - start_time)
        print("STEPS:", steps)
        print()


def plot_reward(rewards, episodes):
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward evolution")
    plt.show()


if __name__ == '__main__':
    main()
