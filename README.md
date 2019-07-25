# Deep Reinforcement Learning (RL) Beginner Toolkit

This project aims for Deep RL researchers, especially beginners.

It defines efficient and easy-to-use DL interfaces and provides common implementation, so that the CL concept is clear to beginners, and it's easy for users to jump to experimentation.

Extra attention was paid to build a clean component libraries with type annotations instead of scripts that "just work" but hard to be modified over time.

In addition to providing implementations of famous algorithms and wrappers for Gym environments, several textbook implementations for environments, runners, policies, etc. are implemented as well to help testing new components.

Gym environment wrapper is implemented so that it's easy to show animations, record videos, etc.


## Concept

There are 4 major types of interfaces: `Environment`, `QFunction`, `Policy`, and `Runner`:

* `Environment`: the main functionality is to take an action, then provide a `Transition` feedback, which is essentially a (s, a, r, s') tuple. The most important implementation is the one wrapping around Gym's environment. There are other simpler environment provided for academic and testing purposes. APIs for showing animations and record videos are also provided.

* `QFunction`: represents a Q(s, a) function, which is more or less "the agent" in this settings. It has APIs to update internal values according to transitions, and to read internal values. Two major implementations are provided:
  - `MemoizationQFunction`: this is the table-lookup based Q-value function used by classical RL.
  - `DQN`: this is an implementation of the Deep-Q-Network (DQN) based Q-value function.
  
* `Policy`: the policy makes the decision. It uses environment and Q-function as references. Implementations for random policy and greedy policies etc. are provided.

* `Runner`: a runners runs multiple episodes and taking care of the interaction between the environment and the Q-function. Other than a `SimpleRunner` that updates Q-function at each step using the most recent transition feedback, a `ExperienceReplayRunner` is also implemented to provide experience replay.

The interfaces are defined in `engine/q_base.py`, and the implementations for these types are defined in `engine/<type>_impl.py` modules. The `engine/q_base.py` module also contains explanation for the basic data structures, like value, action, etc. (They are all numpy arrays for efficiency; their shape are explained.)   


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




