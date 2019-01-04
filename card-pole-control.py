import gym

from policy import Policy
import episode
import numpy
import batches

BATCH_SIZE = 16
PERCENTILE = 70

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    observations_size = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    policy = Policy(observations_size, number_of_actions)

    for iter_no, batch in enumerate(batches.iterate(env, policy, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = batches.fulfills(batch, PERCENTILE)

        policy.train(numpy.asarray(obs_v), numpy.asarray(acts_v))

        if reward_m > 199:
            print("Solved!")
            break

        print(f"{iter_no}:  reward_mean={reward_m}, reward_bound={reward_b}")

    episode.run(env, policy, render=True)

    env.close()
