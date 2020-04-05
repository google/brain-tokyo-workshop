from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='CarRacingColor-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'color',
    },
)

register(
    id='CarRacingColor3-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'color3',
    },
)

register(
    id='CarRacingBar-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'bar',
    },
)

register(
    id='CarRacingBlob-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'blob',
    },
)

register(
    id='CarRacingNoise-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'noise',
    },
)

register(
    id='CarRacingVideo-v0',
    entry_point='car_racing_variants.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900.0,
    kwargs={
        'modification': 'video',
    },
)
