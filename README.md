# Deep Learning (DL) Beginner Toolkit

This project aims for DL researchers, especially beginners.

It defines easy-to-use DL interfaces and provides common implementation, so that the CL concept is clear to beginners, and it's easy for users to jump to experimentation.

Extra attention was paid to build a clean component libraries with type annotations instead of scripts that "just work" but hard to be modified over time.

In addition to providing implementations of famous algorithms and wrappers for Gym environments, several textbook implementations for environments, runners, policies, etc. are implemented as well to help testing new components.

Gym environment wrapper is implemented so that it's easy to show animations, record videos, etc.

## Concept
For the concept and interfaces see definitions in `engine/q_base.py`.

## Usage example
As an e2e example see: `examples/solve_cartpole.py`.

To run it, clone this repository and https://github.com/ChihChiu29/qpylib so that they are sibling directories under the same "parent" directory, then run the following from the parent directory:
```shell
python3 -m deep_learning.example.solve_cartpole
```
It will print out convergence info (it converges in about 300 episodes, which takes about 5 min on an i5 CPU).

Note that by simply changing which gym environment to use, you can already use it to train agent for other environments! However there are many parameters, including models, that can be turned. It's more fun to tweak them by your understanding/intuition. 

## Run environment
If you don't want to install all required dependencies (there isn't too many), you can use a pre-built Docker image by cloning:
https://github.com/ChihChiu29/paio-docker

Then use the scripts in it to download/update Docker image, run image with Jupyter Notebook, etc.
Note that the Docker image has many more tools installed, like PyCharm, OCR, WebDriver, etc. To use a tool with a UI the host that runs Docker needs to have a valid X server -- Linux works, and with Mac OS you can install and run XQuartz first.




