from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise', 'experimental_mode'])

games = {}


apple_world_simple = Game(env_name='AppleWorldSimple',
        input_size=50,
        output_size=5,
        time_factor=0,
        layers=[100,10],
        activation='tanh',
        noise_bias=0.0,
        output_noise=[False,False,False],
        experimental_mode=True,
)
games['apple_world_simple'] = apple_world_simple

cartpole_swingup = Game(env_name='CartPoleSwingUp',
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  experimental_mode=False,
)
games['cartpole_swingup'] = cartpole_swingup

cartpole_swingup_harder = Game(env_name='CartPoleSwingUpHarder',
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  experimental_mode=False,
)
games['cartpole_swingup_harder'] = cartpole_swingup_harder

learn_cartpole = Game(env_name='CartPoleSwingUpHarder',
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[30, 20],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  experimental_mode=True,
)
games['learn_cartpole'] = learn_cartpole

dream_cartpole_swingup = Game(env_name='DreamCartPoleSwingUp',
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  experimental_mode=False,
)
games['dream_cartpole_swingup'] = dream_cartpole_swingup
