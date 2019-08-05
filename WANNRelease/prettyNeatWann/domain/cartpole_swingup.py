"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py

Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version

More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l_base = 0.6 # base pole length
        self.l = self.l_base # simulated pole length (see setEnv below)
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        #self.dt = 0.05  # slower reaction (hard mode)
        self.dt = 0.01  # faster reaction (easy mode)
        self.b = 0.1  # friction coefficient

        self.t = 0 # timestep
        #self.t_limit = 200
        self.t_limit = 1000

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-high, high, dtype=float)

        self.seed()
        self.viewer = None
        self.state = None

        self.noise = 0

    def setEnv(self, envChange):
        '''
        Changes the environment, envChange is the percent change of parameter
        '''
        self.l = self.l_base*envChange

    def setNoise(self, noiseVariance):
        '''
        Changes the leven of input noise
        '''
        self.noise = noiseVariance

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stateUpdate(self,action,state, noise=0):
        x, x_dot, theta, theta_dot = state
        x     += np.random.randn() * noise
        theta += np.random.randn() * noise

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)

        thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)

        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt

        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt  

        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.force_mag

        noise_obs = self.stateUpdate(action, self.state, noise=self.noise)
        self.state = self.stateUpdate(action, self.state)

        x,x_dot,theta,theta_dot = self.state


        done = False
        if  x < -self.x_threshold or x > self.x_threshold:
          done = True

        self.t += 1
        if self.t >= self.t_limit:
          done = True

        # Reward staying in the middle
        reward_theta = (np.cos(theta)+1.0)/2.0
        reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))

        reward = reward_theta*reward_x
        #reward = (np.cos(theta)+1.0)/2.0

        x,x_dot,theta,theta_dot = noise_obs
        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])

        return obs, reward, done, {}

    def reset(self):
        #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        self.t = 0 # timestep
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])
        return obs

    def render(self, mode='human', close=False):
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
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)
  
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
  
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
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

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l*np.sin(x[2]), self.l*np.cos(x[2]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
