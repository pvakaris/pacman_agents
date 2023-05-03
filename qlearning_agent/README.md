# Pac-Man: Q-learning

This is an implementation of an [AI agent](./mlLearningAgents.py) that is able to learn howto play Pacman uaing Q-learning and then successfully win the games.

Original Pac-Man project developed at [UC Berkley](http://ai.berkeley.edu).

## Usage

This project must be run using Python 3 installed via the anaconda environment.

Tran and run the agent:

```
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

- `-x` specifies how many times to train the agent (without GUI).
- `-n` specifies how many times to run the agent in total. In this case, the agent will be trained on 2000 games and then it will play 10 games with the GUI.

Note, that the map that the agent plays in can be specified by modifying the -l argument.
