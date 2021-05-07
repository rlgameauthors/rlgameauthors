from gym import logger
from gym.envs.external_games.supertuxkart import SuperTuxKart

FRAMEBUFFER_SIZE = 10
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time')
env.reset()

logger.set_level(logger.DEBUG)

last_gt   = 0
last_st = 0

for _ in range(10):
    for i in range(10):
        ob, reward, done, scores = env.step('throttle')
        gt = scores['game_time']
        st = scores['system_time']
        
        delta_gt = gt - last_gt
        delta_st = st - last_st
        
        print(f"Game:   {delta_gt}")
        print(f"System: {delta_st}")
        
        last_gt = gt
        last_st = st

    env.reset()
    
env.close() 
