# rlgameauthors

`rlgameauthors` is the replication package of the work *“Using Reinforcement Learning for Load Testing of Video Games”*.

The purpose of this repository is to provide code and data to replicate our results.

## Requirements

- `requirements_study1.txt` provides the list of libraries needed for study1.
- `requirements_study2.txt` provides the list of libraries needed for study2.

In order to install the requirements run the following command in your virtual environment:

```
pip install -r requirements_study<x>.txt 
```

## Content

### CartPole

The `CartPole` folder contains everything needed to replicate study 1 on CartPole game.

- `CartPole_RELINE_train.py` is the script to train the agent using the RELINE approach for 200 steps. It provides as output `model_RELINE` (*i.e.,* the model trained)
- `CartPole_RL-baseline_train.py` is the script to train the agent using the RL-baseline for 200 steps. It provides as output `model_RL-baseline`.
- `CartPole_RELINE_1k_episodes.py` is the script to play additional 1000 episodes (still training) with `model_RELINE`. It provide as output information about the injected bugs spotted.
- `CartPole_RL-baseline_1k_episodes.py` is the script to play additional 1000 episodes (still training) with `model_RL-baseline`. It provide as output information about the injected bugs spotted.
- `CartPole_Random_1k_episodes.py` is the script to play 1000 episodes with a random agent. It provides as output information about the injected bugs spotted.
- `results.xlsx` contains our results obtained with the 3 different approaches.
- `RL-baseline_agent_playing.mp4` is a video of the RL-baseline agent playing CartPole.

### MsPacman

The `MsPacman` folder contains everything needed to replicate study 1 on MsPacman game.

- ‘msPacman_RELINE.py’ is the script to train the agent using the RELINE approach for 1000 steps and to play additional 1000 episodes (still training), used to create a report of the injected bugs spotted. It provides as output the best (`MsPacmanNoFrameskip-v4-best_RELINE.dat`) and the last (`MsPacmanNoFrameskip-v4-last_RELINE.dat`)  model trained and information about the injected bugs spotted.
- `msPacman_RL-baseline.py` is the script to train the agent using the RL-baseline for 1000 steps and to play additional 1000 episodes (still training), used to create a report of the injected bugs spotted. It provides as output the best (`MsPacmanNoFrameskip-v4-best_RL-baseline.dat`) and the last (`MsPacmanNoFrameskip-v4-last_RL-baseline.dat`)  model trained and information about the injected bugs spotted.
- `msPacman_Random.py` is the script to play 1000 episodes with a random agent. It provides as output information about the injected bugs spotted.
- `bug_left` and `bug_right` folders contain images used to check if the agent is in one of the injected bug locations.
- `lib` folder contains files used by python scripts to define the DQN model used by the RELINE approach and the RL-baseline and the game environment.
- `results.xlsx` contains our results obtained with the 3 different approaches.
- `RL-baseline_agent_playing.mp4` is a video of the RL-baseline agent playing MsPacman.


### SuperTuxKart

The `SuperTuxKart` folder contains everything needed to replicate study 2 on SuperTuxKart game.

- `300episodes_RL-baseline_agent.csv`, `300episodes_same_actions.csv` files containing information about the rendering time of frames collected during 300 episodes played by the RL-baseline best agent and 300 episodes played performing always the same actions respectively. We used this information to verify the stability of the metric used and to evaluate the threshold needed for the RELINE approach’s reward function.
- `300episodes_RL-baseline_agent.png`, `300episodes_same_actions.png` images graphically showing the information about the rendering time of frames contained in the 300episodes_RL-baseline_agent.csv` and `300episodes_same_actions.csv` files.
- `FPS_info_RELINE.xlsx`, `FPS_info_RL-baseline.xlsx`, `FPS_info_Random.xlsx` files containing information about the rendering time of frames collected during episodes played by the different agents trained with the 3 different approaches (RELINE, RL-baseline, Random).
- `RELINE_training.png`, `RL-baseline_training.png` images showing the mean reward (of last 100 games) during the training of the agents using the RELINE approach and the RL-baseline.
- `RL-baseline_agent_playing.avi` is a video of the RL-baseline agent playing SuperTuxKart.
- `code`:
    - `agents`:
    	- `SuperTuxKart_RELINE.py` is the script to train the agent using the RELINE approach for 2300 steps and to play additional 1000 episodes (still training), used to create the report of the low-FPS positions spotted. It provides as output the best (`SuperTuxKart-best_RELINE.dat`) and the last (`SuperTuxKart-last_RELINE.dat`)  model trained and information about the rendering time of frames collected during the episodes.
        - `SuperTuxKart_RL-baseline.py` is the script to train the agent using the RL-baseline for 2300 steps and to play additional 1000 episodes (still training), used to create the report of the low-FPS positions spotted. It provides as output the best (`SuperTuxKart-best_RL-baseline.dat`) and the last (`SuperTuxKart-last_RL-baseline.dat`)  model trained and information about the rendering time of frames collected during the episodes.
        - `SuperTuxKart_Random.py` is the script to play 1000 episodes with a random agent. It provides as output information about the rendering time of frames collected during the episodes.
        - `dqn_model.py` is the python file defining the DQN model used by the RELINE approach and the RL-baseline.
    - `gym-master` is the repository containing the wrapper of SuperTuxKart game. Follow the instructions contained in the README file to install it.

	Once the wrapper has been installed, move the `agent` folder in `gym-mastar/gym/`.

	To obtain reliable data (rendering time of frames) run the scripts using the `chrt --rr 1` option, that in Linux maximizes the priority of the process. Also, make sure that no other processes (besides the ones run by the OS) are running on the machine.
