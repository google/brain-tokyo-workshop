# dream env using lstm

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

from model import SimpleWorldModel
from config import games

class DreamCartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.t = 0 # timestep
        self.t_limit = 1000
        self.x_threshold = 2.4
        self.dt = 0.01

        self.real_obs_size = 5
        self.hidden_size = games['learn_cartpole'].layers[0]

        self.world_model = SimpleWorldModel(obs_size=self.real_obs_size, hidden_size=self.hidden_size)
        self.param_count = self.world_model.param_count

        # random weights for world model
        #world_params = self.world_model.get_random_model_params(stdev=1.5)
        #self.world_model.set_model_params(world_params)

        # load a trained world model:
        self.world_model.load_model("./log/learn_cartpole.pepg.16.384.best.json") # ignores agent stuff near the end

        high = np.array([np.finfo(np.float32).max]*(self.real_obs_size))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        self.t += 1

        done = False
        if self.t >= self.t_limit:
          done = True

        [x, x_dot, theta, theta_dot] = self.state
        self.obs = self.world_model.state2obs(self.state)
        x_dot_dot, theta_dot_dot = self.world_model.predict_dynamics(self.obs, action)

        x += x_dot * self.dt
        theta += theta_dot * self.dt

        x_dot += x_dot_dot * self.dt
        theta_dot += theta_dot_dot * self.dt

        self.state = [x, x_dot, theta, theta_dot]
        self.obs = self.world_model.state2obs(self.state)
        reward = self.world_model.predict_reward(self.obs)

        x = self.obs[0]
        if  x < -self.x_threshold or x > self.x_threshold:
          done = True

        return self.obs, reward, done, {}

    def _reset(self):
        def init_state():
          x, x_dot, theta, theta_dot = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
          state = np.array([x,x_dot,theta,theta_dot])
          #[rand_x, rand_x_dot, rand_theta, rand_theta_dot] = np.multiply(np.random.rand(4)*2-1, [self.x_threshold, 10., np.pi/2., 10.])
          #state = np.array([rand_x, rand_x_dot, np.pi+rand_theta, rand_theta_dot])
          return state
        self.t = 0 # timestep
        self.state = init_state()
        self.obs = self.world_model.state2obs(self.state)
        return self.obs

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600 # before was 400

        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = screen_height/2 # TOP OF CART
        polewidth = 6.0
        self.l = 0.6 # pole's length
        polelen = scale*self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
  
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
  
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 1)
            self.viewer.add_geom(cart)
  
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0, 0.5, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
  
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 0.5, 1)
            self.viewer.add_geom(self.axle)
  
            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth/2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight/4)
            self.wheel_r = rendering.make_circle(cartheight/4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth/2, -cartheight/2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth/2, -cartheight/2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line((screen_width/2 - self.x_threshold*scale,carty - cartheight/2 - cartheight/4),
              (screen_width/2 + self.x_threshold*scale,carty - cartheight/2 - cartheight/4))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.obs is None: return None

        x, x_dot, c, s, theta_dot = self.obs
        cartx = x*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        theta = np.arctan2(s, c)
        self.poletrans.set_rotation(theta)
        #self.pole_bob_trans.set_translation(-self.l*s, self.l*c)
        self.pole_bob_trans.set_translation(-self.l*np.sin(theta), self.l*np.cos(theta))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
