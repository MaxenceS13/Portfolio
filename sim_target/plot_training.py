import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os


def load_training_logs(log_file="training_logs.json"):
    """Charge les logs d'entraînement depuis un fichier JSON"""
    if not os.path.exists(log_file):
        print(f" Fichier {log_file} non trouvé")
        return None
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    timesteps = np.array([log["timestep"] for log in logs])
    rewards = np.array([log["reward"] for log in logs])
    
    return timesteps, rewards


def smooth_curve(y, window_size=10):
    """Lisse une courbe avec une moyenne glissante"""
    return uniform_filter1d(y, size=window_size, mode='nearest')


def plot_training_curves(log_file="training_logs.json", output_dir="./outputs"):
    """Génère 3 graphiques : classique, lissé, et analyse"""
    
    data = load_training_logs(log_file)
    if data is None:
        return
    
    timesteps, rewards = data
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n Génération des courbes d'entraînement...")
    
    # ========== GRAPHIQUE 1 : CLASSIQUE ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, rewards, linewidth=1.5, color='#0088ff', alpha=0.8, label="Reward by episode")
    ax.set_xlabel("Timesteps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative reward by episode", fontsize=12, fontweight='bold')
    ax.set_title("PPO Training Curve — Raw Data", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Ajouter des stats
    avg_reward = np.mean(rewards[-100:])
    max_reward = np.max(rewards)
    ax.axhline(avg_reward, color='red', linestyle='--', alpha=0.6, label=f"Average last 100 : {avg_reward:.1f}")
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    path1 = os.path.join(output_dir, "training_raw.png")
    fig.savefig(path1, dpi=150, facecolor='white')
    print(f"  Ok : {path1}")
    plt.close(fig)
    
    # ========== GRAPHIQUE 2 : LISSÉ ==========
    smoothed = smooth_curve(rewards, window_size=20)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, rewards, linewidth=0.5, color='#cccccc', alpha=0.4, label="Raw")
    ax.plot(timesteps, smoothed, linewidth=2.5, color='#00dd44', label="Smooth (window=20)")
    ax.set_xlabel("Timesteps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative reward by episode", fontsize=12, fontweight='bold')
    ax.set_title("PPO Training Curve — Smooth Data", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    path2 = os.path.join(output_dir, "training_smoothed.png")
    fig.savefig(path2, dpi=150, facecolor='white')
    print(f"  Ok : {path2}")
    plt.close(fig)
    
    # ========== GRAPHIQUE 3 : ANALYSE PÉDAGOGIQUE ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sous-graphique 1 : Convergence
    ax1.plot(timesteps, smoothed, linewidth=2.5, color='#0088ff')
    ax1.fill_between(timesteps, smoothed, alpha=0.3, color='#0088ff')
    ax1.set_xlabel("Timesteps", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Reward (smoothed)", fontsize=11, fontweight='bold')
    ax1.set_title("Agent Convergence", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Ajouter des phases
    total_steps = timesteps[-1]
    phases = [
        (0, total_steps * 0.25, "Exploration", '#ff6b6b'),
        (total_steps * 0.25, total_steps * 0.75, "Learning", '#4ecdc4'),
        (total_steps * 0.75, total_steps, "Convergence", '#45b7d1'),
    ]
    
    for start, end, label, color in phases:
        ax1.axvspan(start, end, alpha=0.1, color=color)
    
    # Sous-graphique 2 : Progression par grands blocs (ex: 6 blocs de 500k)
    num_bins = 6
    bin_size = total_steps / num_bins
    bin_labels = []
    bin_rewards = []
    
    for i in range(num_bins):
        start_step = i * bin_size
        end_step = (i + 1) * bin_size
        
        mask = (timesteps >= start_step) & (timesteps < end_step)
        if np.any(mask):
            bin_rewards.append(np.mean(rewards[mask]))
        else:
            bin_rewards.append(0)
            
        label = f"{start_step/1e6:.1f}-{end_step/1e6:.1f}M"
        bin_labels.append(label)

    ax2.bar(bin_labels, bin_rewards, color='#00dd44', alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xlabel("Training intervals (Timesteps)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Average reward", fontsize=11, fontweight='bold')
    ax2.set_title("Reward Progress by Epochs", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
    
    fig.suptitle("PPO Training Analysis", fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    path3 = os.path.join(output_dir, "training_analysis.png")
    fig.savefig(path3, dpi=150, facecolor='white')
    print(f"  ✓ {path3}")
    plt.close(fig)
    
    # ========== STATS FINALES ==========
    print("\n" + "="*50)
    print("PPO Training Stats")
    print("="*50)
    print(f"Total timesteps : {int(timesteps[-1]):,}")
    print(f"Total episodes : {len(rewards)}")
    print(f"Average reward (all) : {np.mean(rewards):.2f}")
    print(f"Average reward (last 100 episodes) : {np.mean(rewards[-100:]):.2f}")
    print(f"Max reward : {np.max(rewards):.2f}")
    print(f"Min reward : {np.min(rewards):.2f}")
    print("="*50)


if __name__ == "__main__":
    plot_training_curves()