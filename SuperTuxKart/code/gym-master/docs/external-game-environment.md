# External Games

This module provides a framework and many utilities for easily adapting external games to the Gym framework. This works only for games running on Linux with a Xorg instance (usually, Xvfb): Wayland or others are not supported.

## Using an External Game environment
This should be exactly the same as using any other environment. If you want to run an agent on Linux (X11), you can use the following commands:

```bash
pip install . # Installs gym and its dependencies on the system
bin/local_test [--debug-video] python path/to/agent {agent arguments} # Runs a Xvfb instance on display 9
```
The `--debug-video` option allows to start a video stream of the display (file `display.flv`). It requires `ffmpeg`.

Example:

```bash
bin/local_test --debug-video python examples/agents/random_agent.py "colorgame-v0"
```

## Adapting a new game
To adapt a new game, you can create a new class extending the `BaseExternal` class. Classes that adapt external games must implement the following methods:

- `_run_game() -> GameHandler`: Runs the game process and returns a GameHandler object.
- `_close_game() -> None`: Closes the game and kills the process, if any.
- `_start_game() -> None`: Performs the action to skip the title screen and start a new game
- `_setup_controller() -> list`: Returns a list of all the possible commands. A command is a method or a lambda expression.
- `_reset_game() -> None`: Restarts the game
- `_get_scores() -> dict`: Returns any kind of score for the current frame as a dictionary that links the name of the score to its value.
- `_compute_reward() -> float`: Returns the reward at the current state
- `_is_gameover() -> bool`: returns `True` if the current state is game over, `False` otherwise

Optional methods:

- `_get_game_ram() -> float`: Returns the RAM used by the game.
- `_setup_game() -> None:`: Configures the game. If no configuration is required, just skip this.

## Examples
We provide two examples.

### Bastet
Tetris clone with 'bastard' block-choosing AI. Install the `bastet` package to run this game.
Commands:

- Left/right arrow: move block to the left/right
- Up arrow/space: rotate the bloc
- Down arrow: speed-up the fall of the block
- Enter: position the block
- "P": Pause the game

### ColorGame
A very simple Java game that allows to test the external game environment.
You start with a black background and you can use the keys to modify it. You win if you get a background with brightness 1, you loose if the percentage of red is higher than the percentage of blue or green.
The game is already provided as a jar file in the repository. The source code is available as well.
Commands:

- "R": Increase red by 10
- "G": Increase green by 10
- "B": Increase blue by 10
- "A": Increase red, green, and blue by 10
- "Q": Restart the game
- "ESC": Close the game

For each move, the game writes on the standard output the score. If a game over occurs, the game automatically restarts and the "Game over" message is written on the standard output.
