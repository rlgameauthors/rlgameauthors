import argparse

import gym
from gym.envs.external_games.supertuxkart import SuperTuxKart
from gym import wrappers, logger

class SuperTuxKartAgent(object):
    TOLERANCE = 0.1
    CENTERING_TOLERANCE = 6
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.last_centering = 0
        self.count_no_reward = 0

    def act(self, observation, reward, done, scores):
        if reward <= 0.05:
            self.count_no_reward += 1
            if self.count_no_reward > 10:
                self.count_no_reward = 0
                return 'rescue'
        
        if scores is None:
            return 'none'
        centering = scores['centering']
        delta_centering = centering - self.last_centering
        self.last_centering = centering
        
        if abs(delta_centering) > self.TOLERANCE:
            if delta_centering > 0:
                return ['throttle', 'left']
            else:
                return ['throttle', 'right']
        else:
            if centering > self.CENTERING_TOLERANCE:
                return ['throttle', 'left']
            elif centering < -self.CENTERING_TOLERANCE:
                return ['throttle', 'right']
            else:
                return 'throttle'
        


if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.DEBUG)

    env = SuperTuxKart(spf=0.000887, framebuffer_size=4, width=800, height=600, level='scotland', mode='time', speedup=1, observe_performance=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    agent = SuperTuxKartAgent(env.action_space)

    episode_count = 5
    reward = 0
    scores = None

    for i in range(episode_count):
        done = False
        ob = None
        j = 0
        while not done:
            action = agent.act(ob, reward, done, scores)
            ob, reward, done, scores = env.step(action)
            ob[1][0].save(f"screenshots/EP-{i}:STEP-{j}.ppm", overwrite=True)
            j += 1
        ob = env.reset()
    env.close()
