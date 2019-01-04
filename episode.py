from collections import namedtuple

EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
Episode = namedtuple('Episode', field_names=['reward', 'steps'])


def run(env, policy, render=False):
    episode_reward = 0.0
    episode_steps = []

    observations = env.reset()

    finished = False

    while not finished:
        if render:
            env.render()

        action = policy.select_action(observations)
        next_observations, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=observations, action=action))

        finished = is_done
        observations = next_observations

    return Episode(reward=episode_reward, steps=episode_steps)
