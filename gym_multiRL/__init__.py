from gym.envs.registration import register

for reward_type in ['sparse', 'dense','multi']:
    if reward_type =='dense':
        suffix = 'Dense'
    elif reward_type =='multi':
        suffix = 'Multi'
    else:
        suffix = ''
    kwargs = {
        'reward_type': reward_type,
    }
 
    # Fetch
    register(
        id='MultiRLFetchSlide{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiRLFetchPickAndPlace{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiRLFetchReach{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiRLFetchPush{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )