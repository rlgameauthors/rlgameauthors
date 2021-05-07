from gym.envs.external_games.supertuxkart import SuperTuxKart

FRAMEBUFFER_SIZE = 10
env = SuperTuxKart(spf=0.000887, framebuffer_size=FRAMEBUFFER_SIZE, width=800, height=600, level='scotland', mode='time', speedup=10, observe_performance=True)

for i in range(20):
    ob, reward, done, scores = env.step('throttle')
    
    for j in range(10):
        print(ob[0])
        ob[1][j].save(f'./screenshots/screenshot-{i}-{j}.ppm', overwrite=True)

env.close() 
