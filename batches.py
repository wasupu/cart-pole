import episode
import numpy


def fulfills(batch, percentile):
    rewards = list(map(lambda step: step.reward, batch))
    reward_bound = numpy.percentile(rewards, percentile)
    reward_mean = float(numpy.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue

        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    return train_obs, train_act, reward_bound, reward_mean


def iterate(env, policy, batch_size):
    batch = []

    while True:
        episode_result = episode.run(env, policy)
        batch.append(episode_result)

        if len(batch) == batch_size:
            yield batch
            batch = []
