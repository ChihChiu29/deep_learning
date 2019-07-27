# Deep Reinforcement Learning (RL) Beginner Toolkit

This project aims for building a clean, efficient, and easy-to-use Deep RL frameworks for researchers, especially new researches.

* It has a clean design. The DL running loop is broken into a few key interfaces, each aiming for only one functionality. The structure of data they use to exchange is very well documented.
 
* It's easy to use. Implementations are provided for all interfaces for commonly used tools. For example Gym environment is wrapped into an `Environment` interface, which has APIs to easily show animation and record videos. Some implementations for academic examples are provided as well. 

* It's efficient. Extra attention was given to avoid extra copying in Python, and numpy is used intensively. The interface also forces the usage of numpy data structure to ensure high performance. As a result, training a CartPole environment for 500 episodes on a i5 CPU only takes 2 min.

* It's well tested. Unit and integration tests are provided to ensure the integrity of the implementations. Pytype is used everywhere.


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

As an e2e example see: `examples/solve_cartpole_concise.py`.

To run it, clone this repository and https://github.com/ChihChiu29/qpylib so that they are sibling directories under the same "parent" directory, then run the following from the parent directory:
```shell
python3 -m deep_learning.example.solve_cartpole_concise
```
It will print out convergence info (it converges in about 300 episodes, which takes about 5 min on an i5 CPU).

Note that by simply changing which gym environment to use, you can already use it to train agent for other environments!

Once you start tweaking parameters and models, the "full" version of this example is more clear demonstrating different pieces in a run, with more parameters that can be tweaked (it's still short): `examples/solve_cartpole_concise.py`


## Run environment

If you don't want to install all required dependencies (there isn't too many), you can use a pre-built Docker image by cloning:
https://github.com/ChihChiu29/paio-docker

Then use the scripts in it to download/update Docker image, run image with Jupyter Notebook, etc.
Note that the Docker image has many more tools installed, like PyCharm, OCR, WebDriver, etc. To use a tool with a UI the host that runs Docker needs to have a valid X server -- Linux works, and with Mac OS you can install and run XQuartz first.




