# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: MsPacman                                                 ***
# ***                                             Random: random actions                                             ***
# ***                                                 1000 episodes                                                  ***
# **********************************************************************************************************************
# **********************************************************************************************************************


from lib import wrappers
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import cv2
import os
import random

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

# **********************************************************************************************************************
# *                                                   1000 episodes                                                    *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n*******************************************************')
    print("* Random model's playing 1000 episodes of MsPacman... *")
    print('*******************************************************\n')

    DEFAULT_ENV_NAME = "MsPacmanNoFrameskip-v4"
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    f = open('bug_log_Random.txt', 'w+')
    f.close()

    for game in range(10):

        state = env.reset()
        total_reward = 0.0
        # wait the game starts
        for i in range(65):
            state, reward, is_done, _ = env.step(0)

        bug_flags = [False, False, False, False]
        count_actions = 0
        while True:

            # random action
            action = random.randint(0, 4)
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
            count_actions += 1
            # check injected bugs
            env.env.ale.saveScreenPNG('current_screen.png')
            if not bug_flags[0] and check_bug1():
                bug_flags[0] = True
            if not bug_flags[1] and check_bug2():
                bug_flags[1] = True
            if not bug_flags[2] and check_bug3():
                bug_flags[2] = True
            if not bug_flags[3] and check_bug4():
                bug_flags[3] = True

            if is_done:
                f = open('bug_log_Random.txt', 'a+')
                str_bug = ''
                if bug_flags[0]:
                    f.write('BUG1 ')
                if bug_flags[1]:
                    f.write('BUG2 ')
                if bug_flags[2]:
                    f.write('BUG3 ')
                if bug_flags[3]:
                    f.write('BUG4 ')
                f.write('\n')
                f.close()
                done_reward = total_reward
                print('episode: %d, tot moves: %d , total score: %d' % (game, count_actions, total_reward))
                print(str_bug)
                break
    env.close()

    lines = [line for line in open('bug_log_Random.txt', 'r')]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0
    count_3bug = 0
    count_4bug = 0

    for line in lines:
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
