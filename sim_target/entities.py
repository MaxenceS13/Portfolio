from collections import deque
import numpy as np
from config import DT, WORLD_LENGTH, WORLD_WIDTH, MAX_TIME, MAX_DISTANCE, MAX_TURN_ANGLE


class World:
    def __init__(self, length=WORLD_LENGTH, width=WORLD_WIDTH):
        self.length = length
        self.width = width


class Target:
    def __init__(self, x, y, vx, vy, hit_radius, trail_length=50):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.hit_radius = hit_radius
        self.trail = deque(maxlen=trail_length)

    def update(self, world, dt=DT):
        if self.x <= 0 or self.x >= world.length:
            self.vx = -self.vx
        if self.y <= 0 or self.y >= world.width:
            self.vy = -self.vy
        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_hit(self, x_missile, y_missile):
        distance = np.sqrt((x_missile - self.x) ** 2 + (y_missile - self.y) ** 2)
        return distance <= self.hit_radius

    def evade(self, list_missiles, range_evasion=150):
        active_missiles = [m for m in list_missiles if m.status_flight == "Flight"]
        if not active_missiles:
            return
        closest = min(active_missiles,key=lambda m: np.sqrt((m.x - self.x) ** 2 + (m.y - self.y) ** 2))
        distance = np.sqrt((closest.x - self.x) ** 2 + (closest.y - self.y) ** 2)
        if distance <= range_evasion:
            new_heading = (np.arctan2(self.y - closest.y, self.x - closest.x) + np.pi / 2)
            speed = np.sqrt(self.vx**2 + self.vy**2)
            self.vx = np.cos(new_heading) * speed
            self.vy = np.sin(new_heading) * speed


class Missile:
    def __init__(self, x, y, heading, speed, hit_radius, trail_length=50, guidance="pure", color="#00f3ff"):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
        self.hit_radius = hit_radius
        self.trail = deque(maxlen=trail_length)
        self.los_angle = 0.0 # Line of Sight angle
        self.proportional_constant = 0.25 # Usually between 3 and 5 but too high for our simulation : high oscillations (because dt is quite large)
        self.guidance = guidance
        self.color = color
        self.guidance_methods = {
            "pure": self.guide,
            "proportional": self.guide_advanced,
            "rl" : self.guide_RL,
        }
        self.distance = 0.0
        self.impact_time = None # Because 0 is for the target hit at t=0
        self.min_distance = np.inf # So that the following distance is inferior to this one
        self.status_flight = "Flight" # Miss, Flight, Hit, OutOfBounds

    def update(self, dt=DT):
        self.x += self.speed * np.cos(self.heading) * dt
        self.y += self.speed * np.sin(self.heading) * dt

    def guide(self, x_target, y_target, dt=DT):
        self.heading = np.arctan2(y_target - self.y, x_target - self.x)

    def guide_advanced(self, x_target, y_target, dt=DT):
        los_angle_now = np.arctan2(y_target - self.y, x_target - self.x)
        angular_rate = np.arctan2(np.sin(los_angle_now - self.los_angle),np.cos(los_angle_now - self.los_angle)) / dt
        self.los_angle = los_angle_now
        lateral_acc = self.proportional_constant * angular_rate * self.speed
        self.heading = (self.heading + lateral_acc * dt) % (2 * np.pi)

    def guide_RL(self, target_x, target_y, dt=DT):
        # Calcul des observations
        dist = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)
        los_angle_now = np.arctan2(target_y - self.y, target_x - self.x)
        rel_angle = np.arctan2(np.sin(los_angle_now - self.heading), np.cos(los_angle_now - self.heading))
        angular_rate = np.arctan2(np.sin(los_angle_now - self.los_angle), np.cos(los_angle_now - self.los_angle)) / dt
        rl_obs = np.array([dist / MAX_DISTANCE,rel_angle / np.pi,np.clip(angular_rate / (2 * np.pi), -1.0, 1.0),np.sin(rel_angle)], dtype=np.float32)
    
        # Prédiction et modification du heading
        action, _ = self.model.predict(rl_obs, deterministic=True)
        self.heading += action[0] * MAX_TURN_ANGLE
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # Mise à jour de l'angle LOS pour la prochaine frame
        self.los_angle = los_angle_now

class Simulation:        
    def __init__(self, world, target, missiles):
        self.world = world
        self.target = target
        self.missiles = missiles
        self.time = 0.0

    def is_any_hit(self):
        return any(self.target.is_hit(m.x, m.y) for m in self.missiles)

    def is_finished(self):
        return all(missile.status_flight != "Flight" for missile in self.missiles)

    def step(self, dt=DT):
        for missile in self.missiles:
            if missile.status_flight == "Flight":
                if missile.guidance in missile.guidance_methods:
                    missile.guidance_methods[missile.guidance](self.target.x, self.target.y, dt)
                missile.update(dt)
                missile.trail.append((missile.x, missile.y))
                missile.distance = np.sqrt((missile.x - self.target.x) ** 2 + (missile.y - self.target.y) ** 2)

                if missile.distance < missile.min_distance:
                    missile.min_distance = missile.distance

                out_of_bounds = not (0 <= missile.x <= WORLD_LENGTH and 0 <= missile.y <= WORLD_WIDTH)

                if self.target.is_hit(missile.x, missile.y) and missile.impact_time is None:
                    missile.impact_time = self.time
                    missile.status_flight = "Hit"
                elif out_of_bounds:
                    missile.status_flight = "OutOfBounds"
                elif self.time >= MAX_TIME:
                    missile.status_flight = "Miss"

        self.target.update(self.world, dt)
        self.target.trail.append((self.target.x, self.target.y))
        self.time += dt