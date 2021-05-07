import time
from gym import logger
from gym.envs.external_games.supertuxkart import SuperTuxKart

FRAMEBUFFER_SIZE = 4
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time')

logger.set_level(logger.DEBUG)

last_gt = None
last_st = None

delta_gts = []
delta_sts = []
step_times = []
tot_game_fps = []
tot_gym_fps = []
for i in range(100):
    before = time.time()
    ob, reward, done, scores = env.step('throttle')
    after = time.time()
    gt = scores['game_time']
    st = scores['system_time']
    
    if last_st is not None and last_gt is not None:
        delta_gt = gt - last_gt
        delta_st = st - last_st
        
        gym_fps = float("inf")
        if delta_st > 0:
            gym_fps = FRAMEBUFFER_SIZE / delta_st
        
        print("---------------------------------")
        print(f"Game:    {delta_gt}")
        print(f"System:  {delta_st}")
        print(f"Step:    {after - before}")
        print(f"GameFPS: {env.fps}")
        print(f"GymFPS:  {gym_fps}")
        
        delta_gts.append(delta_gt)
        delta_sts.append(delta_st)
        step_times.append(after - before)
        tot_game_fps.append(env.fps)
        if gym_fps != float("inf"):
            tot_gym_fps.append(gym_fps)
    
    last_gt = gt
    last_st = st

print("------------ OVERALL ------------")
print(f"Game:    {sum(delta_gts) / len(delta_gts)}")
print(f"System:  {sum(delta_sts) / len(delta_sts)}")
print(f"Step:    {sum(step_times) / len(step_times)}")
print(f"GameFPS: {sum(tot_game_fps) / len(tot_game_fps)}")
print(f"GymFPS:  {sum(tot_gym_fps) / len(tot_gym_fps)}")

env.close()
