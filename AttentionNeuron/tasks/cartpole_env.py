"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py
Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
# logger = logging.getLogger(__name__)


class CartPoleSwingUpHarderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, simple=False, redundant_obs=False):
        self.simple = simple
        self.redundant_obs = redundant_obs
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient

        self.t = 0  # timestep
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

        if self.redundant_obs:
            high = np.concatenate([high] * 2, axis=0)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.np_random = None
        self.seed()
        self.viewer = None
        self.state = None
        self.prev_state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.force_mag

        state = self.state
        x, x_dot, theta, theta_dot = state

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (
            (-2 * self.m_p_l * (theta_dot ** 2) * s +
             3 * self.m_p * self.g * s * c +
             4 * action - 4 * self.b * x_dot) /
            (4 * self.total_m - 3 * self.m_p * c ** 2)
        )
        thetadot_update = (
            (-3 * self.m_p_l * (theta_dot ** 2) * s * c +
             6 * self.total_m * self.g * s +
             6 * (action - self.b * x_dot) * c) /
            (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)
        )

        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt

        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt

        self.state = (x, x_dot, theta, theta_dot)

        done = False
        if x < -self.x_threshold or x > self.x_threshold:
            done = True

        self.t += 1

        if self.t >= self.t_limit:
            done = True

        reward_theta = (np.cos(theta) + 1.0) / 2.0
        reward_x = np.cos((x / self.x_threshold) * (np.pi / 2.0))

        reward = reward_theta * reward_x

        prev_x, prev_x_dot, prev_theta, prev_theta_dot = self.prev_state

        c = np.cos(theta)
        s = np.sin(theta)

        # prev_c = np.cos(prev_theta)
        # prev_s = np.sin(prev_theta)

        # print("debug", theta-prev_theta, theta, prev_theta)

        # obs = np.array([x, (x-prev_x)/self.dt, c, s, (theta-prev_theta)/self.dt])
        obs = np.array([x, x_dot, c, s, theta_dot])
        # obs = np.array([x, x_dot, theta, theta_dot])
        if self.redundant_obs:
            obs = np.concatenate([obs] * 2, axis=0)

        self.prev_state = self.state

        return obs, reward, done, {}

    def reset(self):
        if self.simple:
            self.state = self.np_random.normal(
                loc=np.array([0.0, 0.0, np.pi, 0.0]),
                scale=np.array([0.2, 0.2, 0.2, 0.2]),
            )
        else:
            [rand_x, rand_x_dot, rand_theta, rand_theta_dot] = np.multiply(
                self.np_random.rand(4) * 2 - 1,
                [self.x_threshold, 10., np.pi / 2., 10.])
            self.state = np.array(
                [rand_x, rand_x_dot, np.pi + rand_theta, rand_theta_dot])
        self.prev_state = self.state
        self.t = 0  # timestep
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta),
                        theta_dot])  # set zero for init differences
        # obs = np.array([x, x_dot, theta, theta_dot])  # set zero for init
        # differences
        if self.redundant_obs:
            obs = np.concatenate([obs] * 2, axis=0)
        return obs

    def render(self, mode='human', close=False, override_state=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.state is None: return None

        screen_width = 600
        screen_height = 600  # before was 400

        world_width = 5  # max visible position of cart
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 6.0
        polelen = scale * self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        extra_color = 0.0
        if override_state != None:
            extra_color = 0.75

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # real cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            self.cart = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            self.cart.add_attr(self.carttrans)
            self.cart.set_color(1.0, extra_color, extra_color)
            self.viewer.add_geom(self.cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            self.pole = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.pole.set_color(extra_color, extra_color, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            self.pole.add_attr(self.poletrans)
            self.pole.add_attr(self.carttrans)
            self.viewer.add_geom(self.pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            # dream cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            dream_cart = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)],
                                            True)
            self.dream_carttrans = rendering.Transform()
            dream_cart.add_attr(self.dream_carttrans)
            dream_cart.set_color(0.25, 0.25, 0.25)
            self.viewer.add_geom(dream_cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            dream_pole = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)],
                                            True)
            dream_pole.set_color(0.25, 0.25, 0.25)
            self.dream_poletrans = rendering.Transform(translation=(0, 0))
            dream_pole.add_attr(self.dream_poletrans)
            dream_pole.add_attr(self.dream_carttrans)
            self.viewer.add_geom(dream_pole)

            self.dream_axle = rendering.make_circle(polewidth / 2, filled=False)
            self.dream_axle.add_attr(self.dream_poletrans)
            self.dream_axle.add_attr(self.dream_carttrans)
            self.dream_axle.set_color(0.1, .25, .25)
            self.viewer.add_geom(self.dream_axle)

            # Make another circle on the top of the pole
            self.dream_pole_bob = rendering.make_circle(polewidth / 2,
                                                        filled=False)
            self.dream_pole_bob_trans = rendering.Transform()
            self.dream_pole_bob.add_attr(self.dream_pole_bob_trans)
            self.dream_pole_bob.add_attr(self.dream_poletrans)
            self.dream_pole_bob.add_attr(self.dream_carttrans)
            self.dream_pole_bob.set_color(0.25, 0.25, 0.25)
            self.viewer.add_geom(self.dream_pole_bob)

            self.dream_wheel_l = rendering.make_circle(
                cartheight / 4, filled=False)
            self.dream_wheel_r = rendering.make_circle(
                cartheight / 4, filled=False)
            self.dream_wheeltrans_l = rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2))
            self.dream_wheeltrans_r = rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2))
            self.dream_wheel_l.add_attr(self.dream_wheeltrans_l)
            self.dream_wheel_l.add_attr(self.dream_carttrans)
            self.dream_wheel_r.add_attr(self.dream_wheeltrans_r)
            self.dream_wheel_r.add_attr(self.dream_carttrans)
            self.dream_wheel_l.set_color(0.25, 0.25, 0.25)
            self.dream_wheel_r.set_color(0.25, 0.25, 0.25)
            self.viewer.add_geom(self.dream_wheel_l)
            self.viewer.add_geom(self.dream_wheel_r)

            # others:

            self.track = rendering.Line(
                (screen_width / 2 - self.x_threshold * scale,
                 carty - cartheight / 2 - cartheight / 4),
                (screen_width / 2 + self.x_threshold * scale,
                 carty - cartheight / 2 - cartheight / 4)
            )
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        x = self.state
        dream_x = self.state
        if override_state != None:
            dream_x = override_state

        # flash when we peek
        self.cart.set_color(1.0, extra_color, extra_color)
        self.pole.set_color(extra_color, extra_color, 1)

        # real cart
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[2]),
                                            self.l * np.cos(x[2]))

        # dream cart
        dream_cartx = dream_x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.dream_carttrans.set_translation(dream_cartx, carty)
        self.dream_poletrans.set_rotation(dream_x[2])
        self.dream_pole_bob_trans.set_translation(-self.l * np.sin(dream_x[2]),
                                                  self.l * np.cos(dream_x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
