# DEEP LEARNING TOOLING

Defines easy-to-use interface and extend DL interfaces and provides DQN implementation.

The result is a component-driven library for performing DL research.

Several trivial environments, runners, policies, etc. are implemented as well to help testing new components.


## Usage example
See interfaces in `engine/q_base.py`.

As an e2e example see: `examples/solve_cartpole.py`.

To run it, you need to clone https://github.com/ChihChiu29/qpylib from the parent directory of this repository, then go to the *parent directory* then runs:
```shell
python -m deep_learning.example.solve_cartpole
```
It will print out convergence info.

Note that by simply changing which gym environment to use, you can already use it to train agent for other environments!

## Run environment
If you don't want to install all required dependencies, you can use a pre-built Docker image by cloning:
https://github.com/ChihChiu29/paio-docker

Then use the scripts in it to download/update Docker image, run image with Jupyter Notebook, etc.
Note that the Docker image has PyCharm installed if you use it with a valid X server on host (XQuartz on Mac OS).




