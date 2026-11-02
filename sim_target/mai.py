import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO 

class World:
    def __init__(self, length, width):
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

    def update(self, world, dt=0.1):
        if self.x <= 0 or self.x >= world.length:
            self.vx = -self.vx
        if self.y <= 0 or self.y >= world.width:
            self.vy = -self.vy
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt

    def is_hit(self, x_missile, y_missile):
        distance = np.sqrt((x_missile - self.x)**2 + (y_missile - self.y)**2)
        return distance <= self.hit_radius

    def evade(self, list_missiles, range_evasion = 100):
        closest = min(list_missiles, key=lambda m: np.sqrt((m.x - self.x)**2 + (m.y - self.y)**2))
        distance = np.sqrt((closest.x - self.x)**2 + (closest.y - self.y)**2)
        if distance <= range_evasion:
            new_heading = np.arctan2(self.y - closest.y, self.x - closest.x) + np.pi/2
            self.speed = np.sqrt(self.vx**2 + self.vy**2)
            self.vx = np.cos(new_heading) * self.speed
            self.vy = np.sin(new_heading) * self.speed

class Missile:
    def __init__(self, x, y, heading, speed, hit_radius,trail_length=50, guidance="pure"):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
        self.hit_radius = hit_radius
        self.trail = deque(maxlen=trail_length)
        self.los_angle = 0  # Line of Sight angle précédent
        self.proportional_constant = 1 # Usually between 3 and 5 but too high for our simulation : high oscillations (because dt is quite large)
        self.guidance = guidance
        self.guidance_methods = {
            "pure": self.guide,
            "proportional": self.guide_advanced
        }
        self.color_methods = {
            "pure": 'bo',
            "proportional": 'gs'
        }
        self.distance = 0

    def update(self, dt=0.1):
        self.x = self.x + self.speed * np.cos(self.heading) * dt
        self.y = self.y + self.speed * np.sin(self.heading) * dt

    def guide(self, x_target, y_target, dt=0.1):
        self.heading = np.arctan2(y_target - self.y, x_target - self.x)
            
    def guide_advanced(self, x_target, y_target, dt=0.1):
        los_angle_now = np.arctan2(y_target - self.y, x_target - self.x)
        angular_rate = np.arctan2(np.sin(los_angle_now - self.los_angle) , np.cos(los_angle_now - self.los_angle)) / dt # Normalizing angles between -pi and pi rather than 0 and 2pi for less oscillations
        self.los_angle = los_angle_now
        lateral_acc = self.proportional_constant * angular_rate * self.speed
        self.heading = (self.heading + lateral_acc * dt) % (2 * np.pi)

class Simulation:
    def __init__(self, world, target, missiles):
        self.world = world
        self.target = target
        self.missiles = missiles
        self.time = 0
        self.distance=0

class MissileEnv(gym.Env): # Creates an environment for the simulation : The class MissileEnv heritates from the class gym.Env which is the parent class for all the environments.

    # The Gymnasium library is a library for training agents. It uses the concept of an environment to define the simulation. The agent will interact with the environment to learn a policy.
    # The environment is composed of 4 parts : 
    # 1. The action space : What the agent can do
    # 2. The observation space : What the agent sees
    # 3. The reset method : Resets the simulation to initial conditions
    # 4. The step method : Goes one step and returns the new observation, the reward, if it is terminated, if it is truncated, and some info
    # 5. The render method : Displays the simulation (optional for the training)
    

    def __init__(self):
        
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -np.pi]), # The first value is the distance between the missile and the target, the second value is the angle between the missile and the target, and the third value is the angle between the missile and the target. The angle is calculated using the arctan2 function.
            high=np.array([1414, np.pi, np.pi]), # The maximum distance is the diagonal of the world (sqrt(1000^2+1000^2)) (maximum possible distance in a square world of side 1000), the maximum angle is pi.
            dtype=np.float32
        )# There are 3 observation space : distance, angle, angular rate.

        self.action_space = spaces.Box( # It controls the line of sight of the missile : -1 : go -45° (left) / +1 : go +45° (right) per dt
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.world = World(1000,1000)
        self.target = Target(random.uniform(0,1000), random.uniform(0,1000), 0, 0, 10)
        self.missile = Missile(500,500,0,50,5)

        self.time=0
        self.max_time=300
        self.los_angle=0
    
    def reset(self, **kwargs):
        self.time = 0
        self.missile.x = 500
        self.missile.y = 500
        self.missile.heading = 0
        self.missile.los_angle = 0
        self.target.x = random.uniform(0,1000)
        self.target.y = random.uniform(0,1000)
        distance = np.sqrt((self.missile.x-self.target.x)**2 + (self.missile.y-self.target.y)**2)
        missile_heading = np.arctan2(self.target.y-self.missile.y, self.target.x-self.missile.x)
        target_heading = 0
        observation=np.array([distance, missile_heading, target_heading], dtype=np.float32) # dtype coherent with the observation space
        info = {}
        return (observation, info)

    def step(self, action, dt = 0.1):
        max_turn = np.pi / 4 # max deviation angle for each "dt" : it is 45 degrees (pi/4)
        self.missile.heading += action[0] * max_turn #action[0] can either be -1 , 0 or 1 : left, straight or right
        distance_before = np.sqrt((self.missile.x-self.target.x)**2 + (self.missile.y-self.target.y)**2)
        self.missile.update(dt)
        self.target.update(self.world, dt)
        distance_after = np.sqrt((self.missile.x-self.target.x)**2 + (self.missile.y-self.target.y)**2)
        bonus = distance_before - distance_after
        reward = 0
        if self.target.is_hit(self.missile.x, self.missile.y): #If the missile hits the target
            reward = 1000
        elif self.time >= self.max_time: #If the missile does not hit the target
            reward = -100
        reward += bonus * 0
        reward += -0.05 # Penalize for each frame
        self.time += dt
        terminated = self.target.is_hit(self.missile.x, self.missile.y) # If the missile hits the target, the episode is terminated
        truncated = self.time >= self.max_time # If the time is greater than the max time, the episode is truncated
        info = {}
        los_angle_now = np.arctan2(self.target.y - self.missile.y, self.target.x - self.missile.x)
        angular_rate = np.arctan2(np.sin(los_angle_now - self.los_angle), np.cos(los_angle_now - self.los_angle)) / dt
        self.los_angle = los_angle_now
        observation=np.array([distance_after, los_angle_now, angular_rate], dtype=np.float32) # dtype coherent with the observation space
        return (observation, reward, terminated, truncated, info)
        
    def render(self):
        pass

    def old_step(self, world, dt=0.1):
        for missile in self.missiles:
            missile.guidance_methods[missile.guidance](self.target.x, self.target.y, dt)
            missile.update(dt)
            missile.trail.append((missile.x, missile.y))
            missile.distance = np.sqrt((missile.x - self.target.x)**2 + (missile.y - self.target.y)**2)
            print(f"Temps : {self.time:.1f} | Missile : ({missile.x:.1f}, {missile.y:.1f}) | Distance : {missile.distance:.2f}")
            if self.target.is_hit(missile.x, missile.y):
                print(f"Le missile a touché la cible en {self.time:.1f} secondes")
                break
        self.target.trail.append((self.target.x, self.target.y))
        self.target.evade(self.missiles, range_evasion = 100)
        self.target.update(world, dt)
        self.time += dt

    def run(self, dt=0.1, max_time=1000):
        while not self.target.is_hit(self.missile.x, self.missile.y) and self.time < max_time:
            self.step(dt)
        if self.time >= max_time:
            print(f"Le missile n'a pas touché la cible en {max_time:.1f} secondes")


# Instanciation
model = PPO.load("missile_ppo") # Loading the model
env = MissileEnv()
obs, info = env.reset()

for _ in range(300):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Obs: dist={obs[0]:.1f} | Reward: {reward:.1f}")
    if terminated or truncated:
        print("Terminé!")
        break

world = World(1000, 1000)

missile_pure = Missile(500, 500, 0, 50, 5, guidance="pure")
missile_proportional = Missile(500, 500, 0, 50, 5, guidance="proportional")
missile_rl = Missile(500, 500, 0, 50, 5, guidance="rl")
missile_rl.heading = 0  # Réinitialise le cap
missile_rl.los_angle = 0
list_missiles = [missile_pure, missile_proportional, missile_rl]

target = Target(random.uniform(0, 1000), random.uniform(0, 1000), 0, 0, 10)
sim = Simulation(world, target, list_missiles)

# Reinforcement Learning State
rl_obs = None
rl_los_angle = 0

def animate(frame, dt=0.1):
    global rl_obs, rl_los_angle
    
    ax.cla()
    ax.set_xlim(0, world.length)
    ax.set_ylim(0, world.width)
    
    any_hit = any(sim.target.is_hit(m.x, m.y) for m in sim.missiles)
    
    if sim.time == 0:
        rl_los_angle = 0

    if not any_hit and sim.time < 300:
        # Missiles purs et proportionnel
        for m in [missile_pure, missile_proportional]:
            m.guidance_methods[m.guidance](target.x, target.y, dt)
            m.update(dt)
            m.trail.append((m.x, m.y))
            m.distance = np.sqrt((m.x - target.x)**2 + (m.y - target.y)**2)
        
        # Missile RL
        distance = np.sqrt((missile_rl.x - target.x)**2 + (missile_rl.y - target.y)**2)
        los_angle_now = np.arctan2(target.y - missile_rl.y, target.x - missile_rl.x)
        angular_rate = np.arctan2(np.sin(los_angle_now - rl_los_angle), np.cos(los_angle_now - rl_los_angle)) / dt
        rl_los_angle = los_angle_now
        
        rl_obs = np.array([distance, los_angle_now, angular_rate], dtype=np.float32)
        action, _ = model.predict(rl_obs)
        #print(f"Obs: dist={distance:.1f} | los_angle={los_angle_now:.3f} | angular_rate={angular_rate:.3f}")
        #print(f"Action: {action[0]:.3f}")
        
        max_turn = np.pi / 4
        missile_rl.heading += action[0] * max_turn
        missile_rl.update(dt)
        missile_rl.trail.append((missile_rl.x, missile_rl.y))
        missile_rl.distance = distance
        
        target.update(world, dt)
        sim.time += dt
    
    # Dessiner les traînées
    for missile in sim.missiles:
        color = missile.color_methods[missile.guidance] if missile.guidance != "rl" else 'mo'
        for i, (x, y) in enumerate(list(missile.trail)[::2]):
            alpha = (i + 1) / len(missile.trail)
            ax.plot(x, y, color, markersize=4, alpha=alpha)
    
    # Dessiner les missiles
    for missile in sim.missiles:
        color = missile.color_methods[missile.guidance] if missile.guidance != "rl" else 'mo'
        ax.plot(missile.x, missile.y, color, markersize=8)
    
    # Dessiner la cible
    ax.plot(target.x, target.y, 'rx', markersize=8)
    
    # Affichage des distances
    ax.text(10, 950, f"Temps : {sim.time:.1f}s", fontsize=10)
    ax.text(10, 900, f"Pure : {missile_pure.distance:.1f}m", fontsize=10, color='blue')
    ax.text(10, 850, f"Prop : {missile_proportional.distance:.1f}m", fontsize=10, color='green')
    ax.text(10, 800, f"RL : {missile_rl.distance:.1f}m", fontsize=10, color='magenta')

start_sim = True

if start_sim: 
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, world.length)
    ax.set_ylim(0, world.width)
    ani = FuncAnimation(fig, animate, interval=50)
    plt.show()
else:
    # Training the missile with the PPO algorithm
    env = MissileEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=2048)
    model.learn(total_timesteps=3_000_000)
    model.save("missile_ppo")
    print("Entraînement terminé et modèle sauvegardé !")