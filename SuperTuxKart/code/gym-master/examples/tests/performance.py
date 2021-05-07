from gym.envs.external_games.supertuxkart import SuperTuxKart
import sys
import random

SCREENSHOTS = (len(sys.argv) > 1 and sys.argv[1] == '--screenshots')

FRAMEBUFFER_SIZE = 4
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time', speedup=1, observe_performance=True, performance_window_size=15, performance_window_overlap=5)

ACTIONS_WORDS = ['none', 'throttle', 'brake', 'use_powerup', 'drift', 'nitro', 'left', 'right']
ACTIONS_NUMBERS = list(range(len(ACTIONS_WORDS)))

screenshot_id = 0
actions = 0
def make_action(action):
    global screenshot_id
    global actions
    
    print(action)
    ob, reward, done, scores = env.step(action)
    print(env.fps)
    print(ob[0])
    screenshot_id = screenshot_id + 1
    actions += 1
    if ob[0][4] is not None:
        print(f"{ob[0][4]},{ob[0][5]}", file=sys.stderr)
    
for _ in range(10):
    for action in ACTIONS_WORDS:
        make_action(action)

    for a1 in ACTIONS_WORDS:
        others = ACTIONS_WORDS.copy()
        others.remove(a1)
        for a2 in others:
            action = [a1, a2]
            make_action(action)
    env.reset()
    print("--- RUN DONE ---")
    actions = 0
