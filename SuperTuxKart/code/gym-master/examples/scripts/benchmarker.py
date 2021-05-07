from gym.envs.external_games.utils import Action, GameRegion, ProcessWrapper, GameHandler
import sys
import time

PID = int(sys.argv[1])
SLEEP = float(sys.argv[2])

gh = GameHandler(ProcessWrapper(PID), window="none")

print(f"time,cpu,gpu,tram,dram,vram")
while True:
    time.sleep(SLEEP)
    cpu = gh.used_cpu()
    gpu = gh.used_gpu()
    tram = gh.used_total_ram()
    dram = gh.used_data_ram()
    vram = gh.used_vram()
    print(f"{time.time()},{cpu},{gpu},{tram},{dram},{vram}")
