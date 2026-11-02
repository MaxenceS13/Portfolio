import os
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from stable_baselines3 import PPO
import time
import json
from stable_baselines3.common.callbacks import BaseCallback
from plot_training import plot_training_curves

from config import (
    DT,
    MAX_DISTANCE,
    MAX_TIME,
    MAX_TURN_ANGLE,
    MISSILE_RADIUS,
    MISSILE_SPEED,
    TARGET_RADIUS,
    TARGET_VX,
    TARGET_VY,
)
from entities import Missile, Simulation, Target, World
from environment import MissileEnv
from generate_gif import generate_comparison_gif

class LogCallback(BaseCallback):
    """Callback pour sauvegarder les logs d'entraînement"""
    def __init__(self, log_file="training_logs.json"):
        super().__init__()
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumuler les rewards
        self.current_episode_reward += self.locals["rewards"][0]
        
        # À la fin d'un épisode
        if self.locals["dones"][0]:
            self.episode_rewards.append({
                "timestep": self.num_timesteps,
                "reward": self.current_episode_reward
            })
            self.current_episode_reward = 0.0
            
            # Sauvegarder tous les 100 épisodes
            if len(self.episode_rewards) % 100 == 0:
                with open(self.log_file, 'w') as f:
                    json.dump(self.episode_rewards, f)
                print(f"  Logs sauvegardés ({len(self.episode_rewards)} épisodes)")
        
        return True

# Obtenir le chemin absolu du dossier courant
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "missile_ppo")

# --- BASCULE ENTRAÎNEMENT / MODE ---
TRAIN_MODE = False
COMPARISON_MODE = True  # False = animation, True = N_RUNS

plot_training_curves(log_file="training_logs.json", output_dir="./outputs")
sys.exit()

if TRAIN_MODE:
    print("Lancement de l'entraînement RL...")
    start_time = time.time()
    env = MissileEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    log_callback = LogCallback("training_logs.json")
    model.learn(total_timesteps=3_000_000, callback=log_callback)

    elapsed_time = time.time() - start_time
    print(f"✓ Entraînement terminé en {elapsed_time/3600:.1f} heures")

    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé sous '{MODEL_PATH}.zip'.")

    plot_training_curves(log_file="training_logs.json", output_dir="./outputs")

else:
    model = PPO.load(MODEL_PATH)
    
    if COMPARISON_MODE:
        num_runs = 2000
        print(f"Starting {num_runs} comparison runs...")
        
        results = {
            "pure": {"success": [], "impact_time": [], "min_distance": []},
            "proportional": {"success": [], "impact_time": [], "min_distance": []},
            "rl": {"success": [], "impact_time": [], "min_distance": []}
        }
        
        for run in range(num_runs):
            # Réinitialisation pour chaque run
            world = World()
            
            missile_pure = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="pure")
            missile_prop = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="proportional")
            missile_rl = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="rl")
            missile_rl.model = model  # Assigner le modèle au missile RL
            
            missiles = [missile_pure, missile_prop, missile_rl]
            target = Target(
                random.uniform(100, 900), 
                random.uniform(100, 900),
                TARGET_VX,
                TARGET_VY,
                TARGET_RADIUS
            )
            
            sim = Simulation(world, target, missiles)
            
            # Boucle de simulation jusqu'à terminaison
            while not sim.is_finished():
                target.evade(missiles)  # Évasion active de la cible
                sim.step(DT)
            
            # Collecte des métriques pour chaque missile
            guidance_names = ["pure", "proportional", "rl"]
            for missile, guidance in zip(missiles, guidance_names):
                is_hit = missile.status_flight == "Hit"
                results[guidance]["success"].append(1 if is_hit else 0)
                
                if is_hit:
                    results[guidance]["impact_time"].append(missile.impact_time)
                # Toujours collecter min_distance
                results[guidance]["min_distance"].append(missile.min_distance)
            
            if (run + 1) % (num_runs // 10) == 0:  # Affiche tous les 10%
                print(f"  Run {run + 1}/{num_runs} completed")
        
        print("\n" + "="*50)
        print("RÉSULTATS DE 100 RUNS")
        print("="*50)
        
        for guidance in guidance_names:
            success_rate = np.mean(results[guidance]["success"]) * 100
            avg_min_distance = np.mean(results[guidance]["min_distance"])
            
            print(f"\n{guidance.upper()}:")
            print(f"  Taux de réussite : {success_rate:.1f}%")
            print(f"  Distance minimale moyenne : {avg_min_distance:.2f}m")
            
            if results[guidance]["impact_time"]:
                avg_impact_time = np.mean(results[guidance]["impact_time"])
                print(f"  Temps d'impact moyen : {avg_impact_time:.2f}s")

        # --- COMPARISON BOXPLOT ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Comparison of 3 Guidance Methods ({num_runs} runs)", fontsize=14, fontweight='bold')
        
        guidance_names = ["pure", "proportional", "rl"]
        metrics = [
            ("success", "Success (0=Fail, 1=Success)", "Success Rate"),
            ("min_distance", "Distance (m)", "Minimum Distance Reached"),
            ("impact_time", "Time (s)", "Impact Time (successes only)")
        ]

        for i, (metric_key, ylabel, title) in enumerate(metrics):
            ax = axes[i]
            data = [results[g][metric_key] for g in guidance_names]
            
            bp = ax.boxplot(data, patch_artist=True)
            ax.set_xticklabels(guidance_names)
            
            # 1. Boîtes transparentes (sans couleur de fond)
            for patch in bp['boxes']:
                patch.set_facecolor('none')
                
            # 2. Styliser les médianes en orange
            for median in bp['medians']:
                median.set_color('orange')
                median.set_linewidth(2)
                
            # 3. Calcul et affichage de la moyenne
            if metric_key == "impact_time":
                means = [np.mean(results[g][metric_key]) if results[g][metric_key] else 0 for g in guidance_names]
            else:
                means = [np.mean(results[g][metric_key]) for g in guidance_names]
                
            mean_plot = ax.plot(range(1, len(guidance_names) + 1), means, 'ro', markersize=8)[0]
            
            # 4. Légende : plus bas (ex: 'lower right' ou 'center right') pour le Success Rate, 'upper right' pour les autres
            median_line = plt.Line2D([0], [0], color='orange', lw=2)
            
            if i == 0:  # Uniquement pour le Success Rate
                legend_loc = 'center right'  # ou 'lower right' selon ta préférence
            else:
                legend_loc = 'upper right'
                
            ax.legend([median_line, mean_plot], ['Median', 'Mean'], fontsize=9, loc=legend_loc)
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig("./outputs/boxplot_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        generate_comparison_gif(model, num_runs=10, skip_frames=3)
    else:
        # --- ANIMATION SIMPLE ---
        world = World()

        missile_pure = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="pure", color="#00f3ff")
        missile_prop = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="proportional", color="#00ff66")
        missile_rl = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="rl", color="#ff007f")

        missile_rl.model = model

        missiles = [missile_pure, missile_prop, missile_rl]
        target = Target(random.uniform(100, 900), random.uniform(100, 900), TARGET_VX, TARGET_VY, TARGET_RADIUS)

        sim = Simulation(world, target, missiles)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("black")

        def animate(frame):
            ax.cla()
            ax.set_facecolor("#0a0a0c")
            ax.set_xlim(0, sim.world.length)
            ax.set_ylim(0, sim.world.width)
            ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)

            if not sim.is_finished():
                target.evade(missiles)  # Évasion active
                sim.step(DT)           
        ani = FuncAnimation(fig, animate, interval=50)
        plt.show()