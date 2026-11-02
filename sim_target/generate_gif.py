# Imports pour generate_comparison_gif
import os
import random
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import time
import json
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    DT,
    MISSILE_SPEED,
    MISSILE_RADIUS,
    TARGET_RADIUS,
    TARGET_VX,
    TARGET_VY,
    WORLD_LENGTH,
    WORLD_WIDTH,
)
from entities import Missile, Simulation, Target, World

# --- FONCTION POUR CAPTURER ET GÉNÉRER GIF ---
def generate_comparison_gif(model, num_runs=2, skip_frames=3):
    """Lance num_runs simulations et génère un GIF animé"""
    print(f"\n Génération d'un GIF avec {num_runs} runs...")
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comparison.gif')
    frames = []
    
    for run_num in range(num_runs):
        print(f"  Run {run_num + 1}/{num_runs} — Capture en cours...")
        
        # Réinitialisation
        world = World()
        missile_pure = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="pure", color="#00f3ff")
        missile_prop = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="proportional", color="#00ff66")
        missile_rl = Missile(500, 500, 0, MISSILE_SPEED, MISSILE_RADIUS, guidance="rl", color="#ff007f")
        missile_rl.model = model
        
        missiles = [missile_pure, missile_prop, missile_rl]
        target = Target(
            random.uniform(100, 900),
            random.uniform(100, 900),
            TARGET_VX,
            TARGET_VY,
            TARGET_RADIUS
        )
        
        sim = Simulation(world, target, missiles)
        frame_count = 0
        
        # Simulation + capture
        while not sim.is_finished():
            target.evade(missiles)
            sim.step(DT)
            
            # Capturer chaque skip_frames-ème frame
            if frame_count % skip_frames == 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_facecolor("#0a0a0c")
                ax.set_xlim(0, sim.world.length)
                ax.set_ylim(0, sim.world.width)
                ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)
                
                # Traînées
                for i, (x, y) in enumerate(list(sim.target.trail)[::2]):
                    alpha = (i + 1) / len(sim.target.trail) if sim.target.trail else 0
                    ax.plot(x, y, ".", color="#ff3344", markersize=2, alpha=alpha)
                
                for m in sim.missiles:
                    for i, (x, y) in enumerate(list(m.trail)[::2]):
                        alpha = (i + 1) / len(m.trail) if m.trail else 0
                        ax.plot(x, y, ".", color=m.color, markersize=2, alpha=0.7)
                
                # Missiles
                for m in sim.missiles:
                    ax.plot(m.x, m.y, "o", color=m.color, markersize=10, alpha=0.3)
                    ax.plot(m.x, m.y, "o", color="#ffffff", markersize=3)
                
                # Cible
                ax.plot(sim.target.x, sim.target.y, "x", color="#ff0055", markersize=8, markeredgewidth=2)
                ax.plot(sim.target.x, sim.target.y, "o", color="#ff0055", markersize=12, alpha=0.2)
                
                # HUD
                ax.text(10, 960, f"Run {run_num + 1} — Time: {sim.time:.1f}s",
                        fontsize=12, color="#ffffff", family="monospace")
                ax.text(10, 920, f"Pure: {missile_pure.distance:.1f}m | Status: {missile_pure.status_flight}",
                        fontsize=11, color=missile_pure.color, family="monospace")
                ax.text(10, 890, f"Prop: {missile_prop.distance:.1f}m | Status: {missile_prop.status_flight}",
                        fontsize=11, color=missile_prop.color, family="monospace")
                ax.text(10, 860, f"RL: {missile_rl.distance:.1f}m | Status: {missile_rl.status_flight}",
                        fontsize=11, color=missile_rl.color, family="monospace")
                
                # Convertir en image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', dpi=80)
                buf.seek(0)
                frames.append(Image.open(buf))
                plt.close(fig)
            
            frame_count += 1
    
    # Créer le GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms par frame = ~10 fps
            loop=0  # Boucle infinie
        )
        print(f"GIF créé : {output_path}")
    else:
        print("Aucune frame capturée")