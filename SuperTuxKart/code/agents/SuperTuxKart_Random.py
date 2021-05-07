# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                               Game: SuperTuxKart                                               ***
# ***                                             Random: random actions                                             ***
# ***                                                 1000 episodes                                                  ***
# **********************************************************************************************************************
# **********************************************************************************************************************

from gym.envs.external_games.supertuxkart import SuperTuxKart
import random

DEFAULT_ENV_NAME = "SuperTuxKart"
FRAME_RATE = 20
FRAME_BUFFER_SIZE = 4
SCREEN_WIDTH = 200  # 1024
SCREEN_HEIGHT = 150  # 768
GAME_LEVEL = 'scotland'
GAME_MODE = 'time'
GAME_SPEED = 1
GAME_LENGTH = 90  # time in seconds

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
    [ACTIONS[1], ACTIONS[3]]  # UP + NITRO
]

# set the device -> GPU or CPU
device = "cuda"
# create the wrapped environment
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAME_BUFFER_SIZE, width=SCREEN_WIDTH, height=SCREEN_HEIGHT,
                   level=GAME_LEVEL, mode=GAME_MODE, speedup=GAME_SPEED, observe_performance=True,
                   performance_window_size=2, performance_window_overlap=1, laps=1)

all_games_positions = []
all_consumptions = []

for game in range(1000):
    env.reset()
    ob, r, is_done, info = env.step(ACTIONS[0])
    while info['game_time'] < 0:
        ob, r, is_done, info = env.step(ACTIONS[0])
    starting_time = info['game_time']
    old_position_x = round(info['centering'], 2)
    old_position_y = round(info['path_done'], 2)

    current_game_positions = []
    current_consumptions = []
    flag = True
    while flag:
        action = random.randint(0, len(POSSIBLE_ACTIONS) - 1)
        # do step in the environment
        ob, r, is_done, info = env.step(POSSIBLE_ACTIONS[action])
        position_x = round(info['centering'], 2)
        position_y = round(info['path_done'], 2)

        current_game_positions.append([position_x, position_y])
        current_consumptions.append(round(ob[0][4], 2))

        elapsed_time = info['game_time'] - starting_time

        old_position_x = position_x
        old_position_y = position_y

        if is_done or elapsed_time > GAME_LENGTH:
            flag = False
            all_games_positions.append(current_game_positions)
            all_consumptions.append(current_consumptions)

f = open('Random_games_positions.txt', 'w')
for game_pos in all_games_positions:
    for elem in game_pos:
        f.write(str(elem[0]) + ' ' + str(elem[1]) + ' , ')
    f.write('\n')
f.close()

f = open('Random_games_consumptions.txt', 'w')
for game_cons in all_consumptions:
    for elem in game_cons:
        f.write(str(elem) + ' , ')
    f.write('\n')
f.close()

env.close()

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
