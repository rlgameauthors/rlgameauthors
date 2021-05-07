# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                               Game: SuperTuxKart                                               ***
# ***                                        RELINE: DQN + info injected bugs                                        ***
# ***                                        Training for 2300 + 1000 episodes                                       ***
# **********************************************************************************************************************
# **********************************************************************************************************************

from datetime import datetime
from gym.envs.external_games.supertuxkart import SuperTuxKart
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import collections
import cv2
import dqn_model
import pandas as pd
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim


FRAME_BUFFER_SIZE = 4
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 150
GAME_LEVEL = 'scotland'
GAME_MODE = 'time'
LAPS = 1
GAME_SPEED = 1
WINDOW_ACTIONS = 2
OVERLAP_ACTIONS = 1
GAME_LENGTH = 90  # time in seconds
OBS_PERFORMANCE = True
DEFAULT_ENV_NAME = "SuperTuxKart"
MEAN_REWARD_BOUND = 50000
MAX_ITER = 3300
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1
EPSILON_FINAL = 0.01
THRESHOLD = 19.35
new_threshold = THRESHOLD  # for training without pregames
NUM_PREGAMES = 100

ACTIONS = ['none',  # 0
           'throttle',  # 1
           'use_powerup',  # 2
           'nitro',  # 3
           'left',  # 4
           'right'  # 5
           ]

POSSIBLE_ACTIONS = [
    [ACTIONS[0]],  # NONE
    [ACTIONS[1]],  # UP
    [ACTIONS[4]],  # LEFT
    [ACTIONS[5]],  # RIGHT
    [ACTIONS[1], ACTIONS[4]],  # UP + LEFT
    [ACTIONS[1], ACTIONS[5]],  # UP + RIGHT
    [ACTIONS[1], ACTIONS[2]],  # UP + POWER
    [ACTIONS[1], ACTIONS[3]]   # UP + NITRO
]

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
        self.state = None
        self.total_reward = 0.0
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.path_done = 0
        self.old_centering = 0
        self.starting_time = 0
        self.count_completed = 0
        self.time_consumption = []
        self.performed_actions = []
        self.reward_consumption = 0
        self.count_delta_path_small = 0
        self._reset()

    def _reset(self):
        env.reset()
        ob, r, is_done, info = self.env.step(ACTIONS[0])
        while info['game_time'] < 0:
            ob, r, is_done, info = self.env.step(ACTIONS[0])
        self.state = self._preprocess_observation(ob[1])
        self.total_reward = 0.0
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.old_centering = info['centering']
        self.starting_time = info['game_time']
        self.time_consumption = []
        self.performed_actions = []
        self.reward_consumption = 0
        self.count_delta_path_small = 0

    @staticmethod
    def _preprocess_observation(observation):
        preprocessed_ob = []
        try:
            for elem in observation:
                screen = np.array(elem.pixels).reshape([SCREEN_HEIGHT, SCREEN_WIDTH, 3]).astype(np.float32)
                gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                preprocessed_ob.append(gray_screen.astype(np.uint8))
            return preprocessed_ob
        except:
            return None

    def _reward1(self, path_done_a, position=0):
        if abs(position) > 5:
            return -1
        if path_done_a - self.path_done >= 10:
            return 10
        if 0.01 < path_done_a - self.path_done < 10:
            return path_done_a - self.path_done
        return 0
    
    def _reward2(self, path_done_a, consumption, threshold, position=0):
        # bonus consumption
        bonus_consumption = 0
        if consumption > threshold:
            bonus_consumption = 10

        if abs(position) > 20:
            self.count_delta_path_small = 0
            bonus_consumption = 0
            self.reward_consumption += bonus_consumption
            return -1
        if path_done_a - self.path_done >= 10:
            self.count_delta_path_small = 0
            self.reward_consumption += bonus_consumption
            return 10 + bonus_consumption
        if 0.25 < path_done_a - self.path_done < 10:
            self.count_delta_path_small = 0
            self.reward_consumption += bonus_consumption
            return (path_done_a - self.path_done) + bonus_consumption

        # check not first delta_path_done too small
        self.count_delta_path_small += 1
        if self.count_delta_path_small > 1:
            bonus_consumption = 0
            self.reward_consumption += bonus_consumption
            return 0
        self.reward_consumption += bonus_consumption
        return bonus_consumption

    @torch.no_grad()  # disable gradient calculation. It will reduce memory consumption.
    def play_step(self, net, threshold, flag_consumption=False, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            # random action (eps-Greedy)
            action = random.randint(0, len(POSSIBLE_ACTIONS) - 1)
            self.count_random_moves += 1
            self.count_total_moves += 1
        else:
            # net action
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device, dtype=torch.float)  # move the tensor on the device
            q_vals_v = net(state_v)  # obtain the q_value from the net
            _, act_v = torch.max(q_vals_v, dim=1)  # pick the best action possible
            action = int(act_v.item())
            self.count_total_moves += 1

        # do step in the environment
        ob, r, is_done, info = self.env.step(POSSIBLE_ACTIONS[action])
        self.performed_actions.append(action)

        current_time_consumption = round(ob[0][4],2)
        self.time_consumption.append(current_time_consumption)
        new_state = self._preprocess_observation(ob[1])
        path_done_new = info['path_done']
        new_centering = info['centering']
        
        if flag_consumption:
            current_reward = self._reward2(path_done_new, current_time_consumption, threshold, new_centering)
            # current_reward = self._reward3(path_done_new, new_centering)
        else:
            current_reward = self._reward1(path_done_new, new_centering)
        elapsed_time = info['game_time'] - self.starting_time
        delta_path_done = round(abs(path_done_new - self.path_done), 2)

        self.total_reward += current_reward
        self.path_done = path_done_new
        
        if is_done:
            self.count_completed += 1
            if not flag_consumption:
                current_reward += 1000
            else:
                current_reward += 10 * self.reward_consumption

        # save the experience
        exp = Experience(self.state, action, current_reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state  # update the state
        # episode is over
        if is_done or elapsed_time > GAME_LENGTH:
            done_reward = self.total_reward
            print('score %2f, tot_bonus %2f, last path_done %2f,  game time: %2f, game completed %d ... time %s' % (
                done_reward, self.reward_consumption, self.path_done, info['game_time'], self.count_completed, datetime.now()))
            print('tot random moves: %d / %d (%.2f %s) with epsilon: %.2f' % (
                self.count_random_moves, self.count_total_moves,
                (self.count_random_moves * 100 / self.count_total_moves), '%', epsilon))

            saved_time_consumption = self.time_consumption
            saved_performed_actions = self.performed_actions
            self._reset()
            return done_reward, saved_time_consumption, saved_performed_actions
        return done_reward, 0, 0


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device, dtype=torch.float)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device, dtype=torch.float)
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
# *                                                   TRAINING STARTS                                                  *
# **********************************************************************************************************************


if __name__ == "__main__":
    # set the device -> GPU or CPU
    device = "cuda"
    # create the wrapped environment
    env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAME_BUFFER_SIZE, width=SCREEN_WIDTH, height=SCREEN_HEIGHT,
                       level=GAME_LEVEL, mode=GAME_MODE, speedup=GAME_SPEED, observe_performance=OBS_PERFORMANCE,
                       performance_window_size=WINDOW_ACTIONS, performance_window_overlap=OVERLAP_ACTIONS, laps=LAPS)

# **********************************************************************************************************************
# *                                                  DEFINE THRESHOLD                                                  *
# **********************************************************************************************************************

    if NUM_PREGAMES > 0:
        print('******************************* STARTING PRE GAMES TO SET THE THRESHOLD *******************************')
        # create the net and the target net
        net = dqn_model.DQN((FRAME_BUFFER_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT), len(POSSIBLE_ACTIONS)).to(device)
        tgt_net = dqn_model.DQN((FRAME_BUFFER_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT), len(POSSIBLE_ACTIONS)).to(device)
        net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best_RL-baseline.dat"))
        tgt_net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best_RL-baseline.dat"))
        print(net)

        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)
        epsilon = EPSILON_START

        threshold = THRESHOLD

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        time_consumption_games = []
        saved_game_actions = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_mean_reward = None
        iteration = 0

        flag_pregames = True
        flag_consumption_training = False
        flag_base_training = False
        count_base_games = 0
        info_for_thrasholds = []
        base_training_starts_game = 0

        while flag_pregames:
            frame_idx += 1
            epsilon = 0.0
            reward_game, time_consumption_, game_actions = agent.play_step(
                net, threshold, flag_consumption_training, epsilon, device=device)
            if reward_game is not None:
                iteration += 1
                total_rewards.append(reward_game)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = float(np.mean(total_rewards[-100:]))
                time_consumption_games.append(time_consumption_)
                saved_game_actions.append(game_actions)
                print("frames %d, games %d, mean reward %.3f, eps %.2f, speed %.2f f/s, time %s" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon, speed, datetime.now()))

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_pregames.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, pre-model saved"
                              % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if iteration >= MAX_ITER:
                    print("Max iteration!")
                    env.close()
                    break

                # check and save time consumption to build the threshold
                if flag_base_training and iteration > base_training_starts_game + 1:
                    count_base_games += 1
                    for elem in time_consumption_:
                        info_for_thrasholds.append(elem)

                    if count_base_games == NUM_PREGAMES:
                        print('100 games after the base training')
                        first_quartile = round(np.percentile(info_for_thrasholds, 25), 2)
                        third_quartile = round(np.percentile(info_for_thrasholds, 75), 2)
                        delta = max(first_quartile - 12.39, third_quartile - 12.86)
                        new_threshold = threshold + delta
                        print('1st quartile: %.2f , 3rd quartile: %.2f , delta: %d'
                              % (first_quartile, third_quartile, delta))
                        print('NEW THRESHOLD: ', new_threshold)
                        flag_pregames = False

            # not enough experience for the training
            if len(buffer) < REPLAY_START_SIZE:
                continue
            # update target net
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
                print('Target net update at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

            if frame_idx == REPLAY_SIZE:
                print('Experience replay buffer full at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

            if frame_idx % REPLAY_SIZE == 0 and frame_idx > REPLAY_SIZE:
                print('Experience replay buffer refilled with new experiences at frame: %d , games: %d'
                      % (frame_idx, len(total_rewards)))

            if frame_idx == REPLAY_START_SIZE:
                print('Training starts at frame: %d , games: %d' % (frame_idx, len(total_rewards)))
                flag_base_training = True
                base_training_starts_game = len(total_rewards)  # start counting from the next game

            if frame_idx == EPSILON_DECAY_LAST_FRAME:
                print('Epsilon reaches the minimum value at frame: %d , games: %d' % (frame_idx, len(total_rewards)))
                tgt_net.load_state_dict(net.state_dict())
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last_pregames.dat")

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            # print('loss: %.3f , frame: %d , games: %d, time %s' % (loss_t, frame_idx, len(total_rewards), datetime.now()))
            loss_t.backward()
            optimizer.step()

        # save actions
        f_actions = open('pregames_actions.txt', 'w')
        for game_action in saved_game_actions:
            for elem in game_action:
                f_actions.write(str(elem) + ' ')
            f_actions.write('\n')
        f_actions.close()
        # save consumption info
        len_max = max([len(elem) for elem in time_consumption_games])
        for elem in time_consumption_games:
            while len(elem) < len_max:
                elem.append(-1)

        time_consumption_games_df = pd.DataFrame({'game_1': time_consumption_games[0]})
        for i in range(len(time_consumption_games) - 1):
            column_name = 'game_' + str(i + 2)
            time_consumption_games_df[column_name] = time_consumption_games[i + 1]

        time_consumption_games_df.to_csv('fps_info_pregames.csv')

# **********************************************************************************************************************
# *                                             REAL TRAINING IS STARTING                                              *
# **********************************************************************************************************************

    print('**************************************** REAL TRAINING IS STARTING ****************************************')
    # real traning with consumption
    net = dqn_model.DQN((FRAME_BUFFER_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT), len(POSSIBLE_ACTIONS)).to(device)
    tgt_net = dqn_model.DQN((FRAME_BUFFER_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT), len(POSSIBLE_ACTIONS)).to(device)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    threshold = new_threshold

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    time_consumption_games = []
    saved_game_actions = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    iteration = 0

    flag_consumption_training = True
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward_game, time_consumption_, game_actions = agent.play_step(net, threshold, flag_consumption_training,
                                                                       epsilon, device=device)
        if reward_game is not None:
            iteration += 1
            total_rewards.append(reward_game)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = float(np.mean(total_rewards[-100:]))
            time_consumption_games.append(time_consumption_)
            saved_game_actions.append(game_actions)
            print("frames %d, games %d, mean reward %.3f, eps %.2f, speed %.2f f/s, time %s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed, datetime.now()))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_RELINE.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if iteration >= MAX_ITER:
                print("Max iterations reached. Training ends\n")
                print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
                env.close()
                break

            if iteration % 100 == 0:
                # save actions
                f_actions = open('games_actions_RELINE.txt', 'w')
                for game_action in saved_game_actions:
                    for elem in game_action:
                        f_actions.write(str(elem) + ' ')
                    f_actions.write('\n')
                f_actions.close()
                # save consumption info
                len_max = max([len(elem) for elem in time_consumption_games])
                for elem in time_consumption_games:
                    while len(elem) < len_max:
                        elem.append(-1)

                time_consumption_games_df = pd.DataFrame({'game_1': time_consumption_games[0]})
                for i in range(len(time_consumption_games) - 1):
                    column_name = 'game_' + str(i + 2)
                    time_consumption_games_df[column_name] = time_consumption_games[i + 1]

                time_consumption_games_df.to_csv('fps_info_RELINE.csv')

        # not enough experience for the training
        if len(buffer) < REPLAY_START_SIZE:
            continue
        # update target net
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last_RELINE.dat")
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
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last_RELINE.dat")

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        # print('loss: %.3f , frame: %d , games: %d, time %s' % (loss_t, frame_idx, len(total_rewards), datetime.now()))
        loss_t.backward()
        optimizer.step()

    env.close()

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
