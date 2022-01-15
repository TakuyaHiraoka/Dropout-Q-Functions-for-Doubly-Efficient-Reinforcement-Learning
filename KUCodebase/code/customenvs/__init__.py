import gym

MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'customenvs.ant:AntTruncatedObsEnv'),
        # TH
        'max_episode_steps': 1000,
        'reward_threshold': 6000.0,
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'customenvs.humanoid:HumanoidTruncatedObsEnv'),

        # TH
        'max_episode_steps': 1000,
    },
)

def register_mbpo_environments():
    for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
        gym.register(**mbpo_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MBPO_ENVIRONMENT_SPECS)

    return gym_ids