import random
from config import (
    DT,
    MAX_DISTANCE,
    MAX_TIME,
    MAX_TURN_ANGLE,
    MISSILE_RADIUS,
    MISSILE_SPEED,
    TARGET_RADIUS,
    WORLD_LENGTH,
    WORLD_WIDTH,
    TARGET_VX,
    TARGET_VY,
)
from entities import Missile, Target, World
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MissileEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=np.array([0.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        self.world = World(WORLD_LENGTH, WORLD_WIDTH)
        self.target = Target(0, 0, TARGET_VX, TARGET_VY, TARGET_RADIUS)
        self.missile = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="rl")

        self.time = 0.0
        self.los_angle = 0.0

    def _get_obs(self):
        dist = np.sqrt((self.missile.x - self.target.x) ** 2 + (self.missile.y - self.target.y) ** 2)

        los_angle_now = np.arctan2(self.target.y - self.missile.y, self.target.x - self.missile.x)

        rel_angle = np.arctan2(np.sin(los_angle_now - self.missile.heading),np.cos(los_angle_now - self.missile.heading))

        angular_rate = (np.arctan2(np.sin(los_angle_now - self.los_angle),np.cos(los_angle_now - self.los_angle)))/DT

        norm_dist = dist / MAX_DISTANCE
        norm_rel_angle = rel_angle / np.pi
        norm_angular_rate = np.clip(angular_rate / (2 * np.pi), -1.0, 1.0)

        obs = np.array([norm_dist, norm_rel_angle, norm_angular_rate, np.sin(rel_angle)], dtype=np.float32)

        return obs, dist, los_angle_now

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.time = 0.0

        self.missile.x, self.missile.y = 500.0, 500.0
        self.missile.heading = random.uniform(0, 2 * np.pi)

        self.target.x = random.uniform(100, 900)
        self.target.y = random.uniform(100, 900)

        self.los_angle = np.arctan2(self.target.y - self.missile.y, self.target.x - self.missile.x)

        obs, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        self.missile.heading += action[0] * MAX_TURN_ANGLE
        self.missile.heading = (self.missile.heading + np.pi) % (2 * np.pi) - np.pi

        dist_before = np.sqrt((self.missile.x - self.target.x) ** 2 + (self.missile.y - self.target.y) ** 2)

        self.missile.update(DT)
        self.target.update(self.world, DT)

        obs, dist_after, los_angle_now = self._get_obs()
        self.los_angle = los_angle_now

        progress = dist_before - dist_after
        reward = progress * 2.0

        hit = self.target.is_hit(self.missile.x, self.missile.y)
        out_of_bounds = not (0 <= self.missile.x <= WORLD_LENGTH and 0 <= self.missile.y <= WORLD_WIDTH)

        terminated = False
        truncated = False

        if hit:
            reward += 500.0
            terminated = True
        elif out_of_bounds:
            reward -= 200.0
            terminated = True
        elif self.time >= MAX_TIME:
            reward -= 100.0
            truncated = True

        reward -= 0.1
        self.time += DT

        return obs, reward, terminated, truncated, {}