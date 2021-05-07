# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: MsPacman                                                 ***
# ***                                               RELINE: DQN model                                                ***
# ***                                       Training for 1000 + 1000 episodes                                        ***
# **********************************************************************************************************************
# **********************************************************************************************************************


from lib import dqn_model
from lib import wrappers
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import collections
import cv2
import datetime
import gym
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random


DEFAULT_ENV_NAME = "MsPacmanNoFrameskip-v4"
MEAN_REWARD_BOUND = 400
MAX_ITERATIONS = 2000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# experience unit : state, action -> new_state, reward, done or not
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


# Experience "container" with a fixed capacity (i.e. max experiences)
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # add experience
    def append(self, experience):
        self.buffer.append(experience)

    # provide a random batch of the experience
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.bug_flags = [False, False, False, False]

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.bug_flags = [False, False, False, False]
        # wait the start of the game
        for i in range(65):
            state, reward, done, _ = env.step(0)

    @staticmethod
    def check_bug1():
        folder_bug = 'bug_left/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 0
        top = 90
        right = 15
        bottom = 120
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug3():
        folder_bug = 'bug_left/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 0
        top = 42
        right = 15
        bottom = 72
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug2():
        folder_bug = 'bug_right/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 305
        top = 90
        right = 320
        bottom = 120
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug4():
        folder_bug = 'bug_right/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 305
        top = 42
        right = 320
        bottom = 72
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @torch.no_grad()  # disable gradient calculation. It will reduce memory consumption.
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            # random action (eps-Greedy)
            action = random.randint(0, 4)
            self.count_random_moves += 1
            self.count_total_moves += 1
        else:
            # net action
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)  # move the tensor on the device
            q_vals_v = net(state_v)  # obtain the q_value from the net
            _, act_v = torch.max(q_vals_v, dim=1)  # pick the best action possible
            action = int(act_v.item())
            self.count_total_moves += 1

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        # check bug
        env.env.ale.saveScreenPNG('current_screen.png')
        if not self.bug_flags[0] and self.check_bug1():
            self.bug_flags[0] = True
            # reward += 50
        if not self.bug_flags[1] and self.check_bug2():
            self.bug_flags[1] = True
            # reward += 50
        if not self.bug_flags[2] and self.check_bug3():
            self.bug_flags[2] = True
            # reward += 50
        if not self.bug_flags[3] and self.check_bug4():
            self.bug_flags[3] = True
            # reward += 50

        self.total_reward += reward

        # save the experience
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state  # update the state
        # episode is over
        if is_done:
            f = open('bug_log_RL-baseline.txt', 'a+')
            if self.bug_flags[0]:
                f.write('BUG1 ')
            if self.bug_flags[1]:
                f.write('BUG2 ')
            if self.bug_flags[2]:
                f.write('BUG3 ')
            if self.bug_flags[3]:
                f.write('BUG4 ')
            f.write('\n')
            f.close()
            done_reward = self.total_reward
            print('tot random moves: %d / %d (%.2f %s) with epsilon: %.2f' % (
                self.count_random_moves, self.count_total_moves,
                (self.count_random_moves * 100 / self.count_total_moves), '%', epsilon))
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0  # no discounted reward for done states
    next_state_values = next_state_values.detach()  # return the tensor without connection to its calculation history

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# **********************************************************************************************************************
# *                                                   TRAINING START                                                   *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n****************************************************************')
    print("* RL-baseline model's training on MsPacman game is starting... *")
    print('****************************************************************\n')
    # set the device -> cuda or cpu
    device = "cpu"
    # create the wrapped environment
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    num_actions = 5  # exclude actions: 5 6 7 8
    # 0 -> none
    # 1 -> up
    # 2 -> right
    # 3 -> left
    # 4 -> down
    net = dqn_model.DQN(env.observation_space.shape, num_actions).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, num_actions).to(device)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    f = open('bug_log_RL-baseline.txt', 'w')
    f.close()
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = float(np.mean(total_rewards[-100:]))

            e = datetime.datetime.now()
            print("frames: %d, episodes: %d , mean reward: %.3f, eps: %.2f, speed: %.2f f/s, time: %s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed, e.strftime("%Y-%m-%d %H:%M:%S")))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_RL-baseline.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if len(total_rewards) == MAX_ITERATIONS:
                print("training ends")
                break

        # not enough experience for the training
        if len(buffer) < REPLAY_START_SIZE:
            continue
        # update target net
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last_RL-baseline.dat")
            print('Target net update at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_SIZE:
            print('Experience replay buffer full at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx % REPLAY_SIZE == 0 and frame_idx > REPLAY_SIZE:
            print('Experience replay buffer refilled with new experiences at frame: %d , games: %d'
                  % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_START_SIZE:
            print('Training starts at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == EPSILON_DECAY_LAST_FRAME:
            print('Epsilon reaches the minimum value at frame: %d , games: %d' % (frame_idx, len(total_rewards)))
            tgt_net.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last_RL-baseline.dat")

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        # print('loss: %.3f , frame: %d , games: %d' % (loss_t, frame_idx, len(total_rewards)))
        loss_t.backward()
        optimizer.step()

    env.close()

    lines = [line for line in open('bug_log_RL-baseline.txt', 'r')]
    lines_1k = lines[-1000:]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0
    count_3bug = 0
    count_4bug = 0

    for line in lines_1k:
        if line.strip() == '':
            count_0bug += 1
        elif len(line.strip().split()) == 1:
            count_1bug += 1
        elif len(line.strip().split()) == 2:
            count_2bug += 1
        elif len(line.strip().split()) == 3:
            count_3bug += 1
        elif len(line.strip().split()) == 4:
            count_4bug += 1

    print('\nReport injected bugs spotted during last 1000 episodes:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print('3 injected bugs spotted in %d episodes' % count_3bug)
    print('4 injected bugs spotted in %d episodes' % count_4bug)
    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
