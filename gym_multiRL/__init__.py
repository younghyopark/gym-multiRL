from gym.envs.registration import registry, register, make, spec

def _merge(a, b):
    a.update(b)
    return a

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

        # Hand
    register(
        id='MultiRLHandReach{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiRLHandManipulateBlockRotateZ{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateZTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateZTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateParallel{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateParallelTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateParallelTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateXYZ{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateXYZTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockRotateXYZTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockFull{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='MultiRLHandManipulateBlock{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateBlockTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggRotate{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggRotateTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggRotateTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggFull{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='MultiRLHandManipulateEgg{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulateEggTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenRotate{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenRotateTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenRotateTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenFull{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='MultiRLHandManipulatePen{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenTouchSensors{}-v0'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='MultiRLHandManipulatePenTouchSensors{}-v1'.format(suffix),
        entry_point='gym_multiRL.envs:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )