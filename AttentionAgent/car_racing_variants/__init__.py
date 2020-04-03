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

