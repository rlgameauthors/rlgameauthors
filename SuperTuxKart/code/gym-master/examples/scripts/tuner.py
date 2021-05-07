import sys
import random
from gym.envs.external_games.supertuxkart import SuperTuxKart

TARGET_FPS = int(sys.argv[1])
TOLERANCE  = int(sys.argv[2])
FRAMEBUFFER_SIZE = 10

print(f"Trying to achieve {TARGET_FPS} FPS...")
spf = 1.0 / TARGET_FPS
confidence = 0

def decide_bounded(lower_bound, upper_bound, real_fps, spf, confidence):
    if real_fps < (TARGET_FPS - TOLERANCE):
        confidence = 0
        upper_bound = spf
        
        spf = lower_bound + random.random() * (upper_bound - lower_bound)
        print(f"Bounded between {int(1 / upper_bound)} and {int(1 / lower_bound)}: using {spf}.")
    elif real_fps > (TARGET_FPS + TOLERANCE):
        confidence = 0
        lower_bound = spf
        
        spf = lower_bound + random.random() * (upper_bound - lower_bound)
        print(f"Bounded between {int(1 / upper_bound)} and {int(1 / lower_bound)}: using {spf}.")
    else:
        confidence += 1
        print(f"Increased confidence to {confidence}")
    
    return (lower_bound, upper_bound, spf, confidence)

upper_bound = None
lower_bound = None
while confidence < 3:
    env = SuperTuxKart(spf=spf, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time')

    last_gt   = 0
    last_st = 0
    deltas = []
    for i in range(20):
        ob, reward, done, scores = env.step('throttle')
        gt = scores['game_time']
        
        delta_gt = gt - last_gt
        last_gt = gt
        deltas.append(delta_gt)
    
    average_delta = (sum(deltas) / len(deltas)) / FRAMEBUFFER_SIZE
    real_fps = int(1.0 / average_delta)
    print(f"Achieved {real_fps} FPS with {int(1 / spf)} target FPS.")
    
    if upper_bound is not None and lower_bound is not None:
        lower_bound, upper_bound, spf, confidence = decide_bounded(lower_bound, upper_bound, real_fps, spf, confidence)
    else:
        if real_fps < TARGET_FPS:
            if upper_bound is None or spf < upper_bound:
                upper_bound = spf
            
            if lower_bound is not None and upper_bound is not None:
                _, _, spf, confidence = decide_bounded(lower_bound, upper_bound, real_fps, spf, confidence)
            else:
                confidence = 0
                spf /= 2
                print(f"Increasing target FPS at {1 / spf}.")
        elif real_fps >= TARGET_FPS:
            if lower_bound is None or spf > lower_bound:
                lower_bound = spf
            
            if lower_bound is not None and upper_bound is not None:
                _, _, spf, confidence = decide_bounded(lower_bound, upper_bound, real_fps, spf, confidence)
            else:
                confidence = 0
                spf *= 2
                print(f"Decreasing target FPS at {1 / spf}.")

    env.close()
    
print(f"Use {spf} as SFP parameter")
