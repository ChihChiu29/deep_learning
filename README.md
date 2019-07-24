# DEEP LEARNING TOOLING

Defines easy-to-use DL interfaces and provides common implementation.

The result is an easy-to-extend component libraries for performing DL research.

Several trivial environments, runners, policies, etc. are implemented as well to help testing new components.

## Concept
For the concept and interfaces see definitions in `engine/q_base.py`.

## Usage example
As an e2e example see: `examples/solve_cartpole.py`.

To run it, you need to clone https://github.com/ChihChiu29/qpylib from the parent directory of this repository, then go to the *parent directory* then runs:
```shell
python3 -m deep_learning.example.solve_cartpole
```
It will print out convergence info (it converges in about 300 episodes, which takes about 5 min on an i5 CPU).

Note that by simply changing which gym environment to use, you can already use it to train agent for other environments! However there are many parameters, including models, that can be turned. It's more fun to tweak them by your understanding/intuition. 

## Run environment
If you don't want to install all required dependencies (there isn't too many), you can use a pre-built Docker image by cloning:
https://github.com/ChihChiu29/paio-docker

Then use the scripts in it to download/update Docker image, run image with Jupyter Notebook, etc.
Note that the Docker image has PyCharm installed if you use it with a valid X server on host (XQuartz on Mac OS).




