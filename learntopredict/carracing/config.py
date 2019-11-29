from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise', 'rnn_mode', 'experimental_mode'])

games = {}

cartpole_swingup = Game(env_name='CartPoleSwingUp',
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['cartpole_swingup'] = cartpole_swingup

bullet_pendulum = Game(env_name='InvertedPendulumSwingupBulletEnv-v0',
  input_size=5,
  output_size=1,
  time_factor=1000,
  layers=[25, 5],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_pendulum'] = bullet_pendulum

bullet_double_pendulum = Game(env_name='InvertedDoublePendulumBulletEnv-v0',
  input_size=9,
  output_size=1,
  time_factor=0,
  layers=[45, 5],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_double_pendulum'] = bullet_double_pendulum

bullet_minitaur_duck = Game(env_name='MinitaurDuckBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_duck'] = bullet_minitaur_duck

bullet_minitaur_duck = Game(env_name='MinitaurDuckBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_duck'] = bullet_minitaur_duck

bullet_kuka_grasping = Game(env_name='KukaBulletEnv-v0',
  input_size=9,
  output_size=3,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_kuka_grasping'] = bullet_kuka_grasping

bullet_kuka_grasping_stoc = Game(env_name='KukaBulletEnv-v0',
  input_size=9,
  output_size=3,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_kuka_grasping_stoc'] = bullet_kuka_grasping_stoc

bullet_minitaur_duck_stoc = Game(env_name='MinitaurDuckBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=-1.0,
  output_noise=[True, True, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_duck_stoc'] = bullet_minitaur_duck_stoc

bullet_minitaur_ball = Game(env_name='MinitaurBallBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_ball'] = bullet_minitaur_ball

bullet_minitaur_ball_stoc = Game(env_name='MinitaurBallBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=-1.0,
  output_noise=[True, True, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_ball_stoc'] = bullet_minitaur_ball_stoc

bullet_half_cheetah = Game(env_name='HalfCheetahBulletEnv-v0',
  input_size=26,
  output_size=6,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_half_cheetah'] = bullet_half_cheetah


bullet_humanoid = Game(env_name='HumanoidBulletEnv-v0',
  input_size=44,
  output_size=17,
  layers=[220, 85],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_humanoid'] = bullet_humanoid

bullet_ant = Game(env_name='AntBulletEnv-v0',
  input_size=28,
  output_size=8,
  layers=[64, 32],
  time_factor=1000,
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_ant'] = bullet_ant

bullet_walker = Game(env_name='Walker2DBulletEnv-v0',
  input_size=22,
  output_size=6,
  time_factor=1000,
  layers=[110, 30],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_walker'] = bullet_walker

bullet_hopper = Game(env_name='HopperBulletEnv-v0',
  input_size=15,
  output_size=3,
  layers=[75, 15],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_hopper'] = bullet_hopper

bullet_racecar = Game(env_name='RacecarBulletEnv-v0',
  input_size=2,
  output_size=2,
  time_factor=1000,
  layers=[20, 20],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_racecar'] = bullet_racecar

bullet_minitaur = Game(env_name='MinitaurBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur'] = bullet_minitaur

bullet_minitaur_stoc = Game(env_name='MinitaurBulletEnv-v0',
  input_size=28,
  output_size=8,
  time_factor=0,
  layers=[64, 32],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[True, True, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bullet_minitaur_stoc'] = bullet_minitaur_stoc

bipedhard_stoc = Game(env_name='BipedalWalkerHardcore-v2',
  input_size=24,
  output_size=4,
  time_factor=1000,
  layers=[120, 20],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[True, True, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['bipedhard_stoc'] = bipedhard_stoc

bipedhard = Game(env_name='BipedalWalkerHardcore-v2',
  input_size=24,
  output_size=4,
  time_factor=0,
  layers=[40, 40],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['bipedhard'] = bipedhard

biped = Game(env_name='BipedalWalker-v2',
  input_size=24,
  output_size=4,
  time_factor=0,
  layers=[40, 40],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['biped'] = biped

rocketlander = Game(env_name='RocketLander-v0',
  input_size=8,
  output_size=3,
  time_factor=0,
  layers=[32, 16],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['rocketlander'] = rocketlander

vae_racing = Game(env_name='VAERacing-v0',
  input_size=16,
  output_size=3,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing'] = vae_racing

vae_racing_pure_world_linear = Game(env_name='VAERacingPureWorld-v0',
  input_size=10,
  output_size=3,
  time_factor=0,
  layers=[0, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing_pure_world_linear'] = vae_racing_pure_world_linear

vae_racing_pure_world = Game(env_name='VAERacingPureWorld-v0',
  input_size=10,
  output_size=3,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing_pure_world'] = vae_racing_pure_world

vae_racing_world_linear = Game(env_name='VAERacingWorld-v0',
  input_size=26,
  output_size=3,
  time_factor=0,
  layers=[0, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing_world_linear'] = vae_racing_world_linear

vae_racing_world = Game(env_name='VAERacingWorld-v0',
  input_size=26,
  output_size=3,
  time_factor=0,
  layers=[10, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing_world'] = vae_racing_world

learn_vae_racing = Game(env_name='VAERacing-v0',
  input_size=16,
  output_size=3,
  time_factor=0,
  layers=[10, 10],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=True,
)
games['learn_vae_racing'] = learn_vae_racing

vae_racing_stack = Game(env_name='VAERacingStack-v0',
  input_size=32,
  output_size=3,
  time_factor=0,
  layers=[20, 0],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
  experimental_mode=False,
)
games['vae_racing_stack'] = vae_racing_stack

vae_racing_rnn = Game(env_name='VAERacing-v0',
  input_size=16,
  output_size=3,
  time_factor=0,
  layers=[16, 5],
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=True,
  experimental_mode=False,
)
games['vae_racing_rnn'] = vae_racing_rnn

osimrun = Game(env_name='osimrun',
  input_size=41,
  output_size=18,
  time_factor=1000,
  layers=[205, 90],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['osimrun'] = osimrun

robo_reacher = Game(env_name='RoboschoolReacher-v1',
  input_size=9,
  output_size=2,
  layers=[45, 10],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_reacher'] = robo_reacher

robo_flagrun = Game(env_name='RoboschoolHumanoidFlagrunHarder-v1',
  input_size=44,
  output_size=17,
  layers=[220, 85],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_flagrun'] = robo_flagrun

robo_pendulum = Game(env_name='RoboschoolInvertedPendulumSwingup-v1',
  input_size=5,
  output_size=1,
  time_factor=1000,
  layers=[25, 5],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_pendulum'] = robo_pendulum

robo_double_pendulum = Game(env_name='RoboschoolInvertedDoublePendulum-v1',
  input_size=9,
  output_size=1,
  time_factor=0,
  layers=[45, 5],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_double_pendulum'] = robo_double_pendulum

robo_humanoid = Game(env_name='RoboschoolHumanoid-v1',
  input_size=44,
  output_size=17,
  layers=[220, 85],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_humanoid'] = robo_humanoid

robo_ant = Game(env_name='RoboschoolAnt-v1',
  input_size=28,
  output_size=8,
  layers=[140, 40],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_ant'] = robo_ant

robo_walker= Game(env_name='RoboschoolWalker2d-v1',
  input_size=22,
  output_size=6,
  time_factor=1000,
  layers=[110, 30],
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_walker'] = robo_walker

robo_hopper = Game(env_name='RoboschoolHopper-v1',
  input_size=15,
  output_size=3,
  layers=[75, 15],
  time_factor=1000,
  activation='passthru',
  noise_bias=0.0,
  output_noise=[False, False, True],
  rnn_mode=False,
  experimental_mode=False,
)
games['robo_hopper'] = robo_hopper
