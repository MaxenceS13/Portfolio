# 2D Pursuit-Evasion Guidance: Reinforcement Learning (PPO) vs. Classical Guidance Laws

A 2D simulation comparing a Proximal Policy Optimization (PPO) agent against classical guidance laws (**Pure Navigation** and **Proportional Navigation**) facing a reactive, evasive target.

![Pursuit-Evasion Simulation](outputs/simulation.gif) *(Optional: place your generated GIF here)*

---

## Overview

This personal project explores the intersection of **Reinforcement Learning** and **aerospace guidance fundamentals**. The primary goal was to get hands-on experience with RL environments and compare a learned guidance policy with standard analytical approaches in a non-passive target scenario.

### Key Scenario Features
* **Reactive Target:** Triggers an emergency $90^\circ$ perpendicular evasive maneuver when any pursuer approaches within **150 meters**.
* **Impact Criterion:** A pursuer must close the distance to under **8 meters** to achieve a successful capture.
* **Large-Scale Evaluation:** 2,000 independent Monte Carlo test runs across all three guidance laws.

---

## Performance & Statistical Results

Over **2,000 runs**, the PPO agent demonstrated dynamics and success rates matching or exceeding classical guidance methods:

| Metric | RL (PPO) | Pure Navigation | Proportional Navigation |
| :--- | :---: | :---: | :---: |
| **Success Rate** | **99.5%** | 98.5% | 97.9% |
| **Mean Impact Time** | **6.86 s** | 6.87 s | 7.96 s |
| **Min Distance** | **6.00 m** | 6.10 m | 6.31 m |

### Observations
1. **Robustness:** All three approaches show high success against the evasive maneuver.
2. **Trajectory Dynamics:** Both RL and Pure Navigation yield tighter, faster trajectories (around ~6.8s).
3. **Variance:** Proportional Navigation exhibits higher dispersion, with outlier failures reaching up to 90m away at timeout.

---

## RL Training & Convergence

* **Timesteps:** ~3 million steps across 36,600 episodes.
* **Convergence Analysis:** Aggregating evaluation metrics in **500k-step blocks** revealed rapid convergence within the first 500k timesteps, showing a stable mean reward plateau around 1,100 with negligible marginal gain beyond that point.
* **Environment Correction:** Fixed an issue where the target reacted to "ghost" missiles (already recorded impacts), removing behavioral biases and stabilizing learning.

---

## Repository Structure

```text
.
├── config.py             # Global parameters (simulation speed, thresholds, rewards)
├── entities.py           # Classes for Missile/Pursuer, Target, and Physics models
├── environment.py        # Gymnasium-compliant RL environment logic
├── main.py               # Main execution script for evaluation/testing
├── generate_gif.py       # Script to render 2D trajectory animation to GIF
├── plot_training.py      # Plotting tool for convergence and 2000-run boxplot analysis
├── training_logs.json    # Full training metrics and log history
└── outputs/              # Saved figures (boxplots, convergence curves, performance charts)