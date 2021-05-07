# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: CartPole                                                 ***
# ***                                RELINE: Cross Entropy Method + info injected bugs                               ***
# ***                                           Training for 200 iterations                                          ***
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple


HIDDEN_SIZE = 128  # neural network size
BATCH_SIZE = 16    # num episodes
PERCENTILE = 70    # elite episodes
MAX_ITER = 200     # training iterations

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # OBSERVATION:
    # - x coordinate of the stick's center of mass
    # - speed
    # - angle to the platform
    # - angular speed
    sm = nn.Softmax(dim=1)
    flag_injected_bug_spotted = [False, False]
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        if -0.5 < next_obs[0] < -0.45 and not flag_injected_bug_spotted[0]:
            reward += 50
            flag_injected_bug_spotted[0] = True
        if 0.45 < next_obs[0] < 0.5 and not flag_injected_bug_spotted[1]:
            reward += 50
            flag_injected_bug_spotted[1] = True
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            flag_injected_bug_spotted = [False, False]
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

# **********************************************************************************************************************
# *                                                   TRAINING START                                                   *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n****************************************************************')
    print("* RL-baseline model's training on CartPole game is starting... *")
    print('****************************************************************\n')
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 1000  # episode length

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        if iter_no == MAX_ITER:
            print("Training ends")
            print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
            torch.save(net.state_dict(), './model_RELINE')
            break

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
