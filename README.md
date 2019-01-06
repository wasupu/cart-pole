# CartPole

Gym ai cart pole environment implemented using deep reinforcement learning with keras

https://gym.openai.com/envs/CartPole-v0/

## The Cross-Entropy Method

So, our cross-entropy method is model-free, policy-based, and on-policy, which means the following:

* It doesn't build any model of the environment; it just says to the agent what to do at every step
* It approximates the policy of the agent
* It requires fresh data obtained from the environment

### Algorithm

* Play N number of episodes using our current model and environment.
* Calculate the total reward for every episode and decide on a reward boundary. Usually, we use some percentile of all rewards, such as 50th or 70th.
* Throw away all episodes with a reward below the boundary.
* Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
* Repeat from step 1 until we become satisfied with the result.

### Limitations

* For training, our episodes have to be finite and, preferably, short
* The total reward for the episodes should have enough variability to separate good episodes from bad ones
* There is no intermediate indication about whether the agent has succeeded or failed

## Run

* Clone the project
* Install the packages.

```
pipenv install
```

* Execute

```
pipenv run python card-pole-control.py
```
