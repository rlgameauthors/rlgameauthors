# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: CartPole                                                 ***
# ***                                             Random: random actions                                             ***
# ***                               Play 1000 episodes and save injected bugs spotted                                ***
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************

import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 1000  # episode length

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # OBSERVATION:
    # - x coordinate of the stick's center of mass
    # - speed
    # - angle to the platform
    # - angular speed

    filename = 'injected_bugs_spotted_random.txt'
    file = open(filename, 'w+')

    for i in range(1000):
        flag_injected_bug_spotted = [False, False]
        episode_reward = 0.0
        episode_steps = []
        obs = env.reset()
        while True:
            action = np.random.choice(n_actions)
            next_obs, reward, is_done, _ = env.step(action)
            if -0.5 < next_obs[0] < -0.45 and not flag_injected_bug_spotted[0]:
                file.write('BUG1 ')
                flag_injected_bug_spotted[0] = True
            if 0.45 < next_obs[0] < 0.5 and not flag_injected_bug_spotted[1]:
                file.write('BUG2 ')
                flag_injected_bug_spotted[1] = True
            episode_reward += reward
            if is_done:
                print('game %d, reward %d' % (i, episode_reward))
                file.write('\n')
                break
            obs = next_obs

    env.close()
    file.close()

    lines = [line for line in open(filename, 'r')]
    lines_1k = lines[:1000]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0

    for line in lines_1k:
        if line.strip() == '':
            count_0bug += 1
        elif len(line.strip().split()) == 1:
            count_1bug += 1
        elif len(line.strip().split()) == 2:
            count_2bug += 1
    print('Report injected bugs spotted:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|

