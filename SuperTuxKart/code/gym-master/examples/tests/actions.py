from gym.envs.external_games.supertuxkart import SuperTuxKart
import sys
import random

SCREENSHOTS = (len(sys.argv) > 1 and sys.argv[1] == '--screenshots')

FRAMEBUFFER_SIZE = 10
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time', speedup=1, observe_performance=False)

ACTIONS_WORDS = ['none', 'throttle', 'brake', 'use_powerup', 'drift', 'nitro', 'left', 'right']
ACTIONS_NUMBERS = list(range(len(ACTIONS_WORDS)))

screenshot_id = 0
def make_action(action):
    global screenshot_id
    print(action)
    ob, reward, done, scores = env.step(action)
    print(scores)
    print(env.fps)
    if SCREENSHOTS:
        print(f"Screenshot ID: #{screenshot_id}")
        for j in range(FRAMEBUFFER_SIZE):
            ob[1][j].save(f'./screenshots/screenshot-{screenshot_id}-{j}.ppm', overwrite=True)
    screenshot_id = screenshot_id + 1

for _ in range(10):
    for action in ACTIONS_WORDS:
        make_action(action)

    for action in ACTIONS_NUMBERS:
        make_action(action)

    for a1 in ACTIONS_WORDS:
        others = ACTIONS_WORDS.copy()
        others.remove(a1)
        for a2 in others:
            action = [a1, a2]
            make_action(action)

    for a1 in ACTIONS_NUMBERS:
        others = ACTIONS_NUMBERS.copy()
        others.remove(a1)
        for a2 in others:
            action = [a1, a2]
            make_action(action)
