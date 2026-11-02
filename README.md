# Portfolio

## Overview

Welcome to my portfolio! This repository showcases engineering projects in **aerospace propulsion, flight guidance, numerical simulation, and machine learning**. 

It delivers end-to-end technical implementations—combining physical modeling, algorithm design, exact verification, and interactive data visualization.

---

## Featured Projects

### 1. [Surrogate Model for NACA 4-digit Airfoils](https://github.com/username/repo-name)
* **Description:** A neural network written from scratch in NumPy that predicts lift and drag coefficients (Cl, Cd) of NACA 4-digit airfoils compared to XFOIL, achieving a 4,000x speed-up. Includes automated data generation, robust per-airfoil splitting/cleaning, custom MLP implementation with backpropagation checked via finite differences, and real-time visualization.
* **Tech Stack:** Python, NumPy, XFOIL, Matplotlib
* **Key Metrics:** Cl R² = 0.9962, Cd MAPE = 3.25% on test set.

### 2. [2D Pursuit-Evasion Guidance: PPO vs. Classical Laws](https://github.com/username/repo-name)
* **Description:** A 2D simulation comparing a Proximal Policy Optimization (PPO) reinforcement learning agent against classical guidance laws (Pure Navigation and Proportional Navigation) against a reactive, evasive target across 2,000 Monte Carlo test runs.
* **Tech Stack:** Python, Gymnasium, Stable-Baselines3 (PPO), NumPy, Matplotlib
* **Key Metrics:** RL success rate of 99.5% with a mean impact time of 6.86 s.

### 3. [Ramjet 0D Cycle & 1D Flow (ramjet-1d)](https://github.com/username/repo-name) *(Figures translation to English in progress)*
* **Description:** A dual model of a ramjet engine comparing a 0D station-by-station Brayton cycle with a quasi-1D conservation equations solver. Features strict mathematical verification against exact closed forms (isentropic, Rayleigh, Fanno), thermal choking limit analysis, variable properties evaluation, and an automated testing suite.
* **Tech Stack:** Python, NumPy, SciPy, Matplotlib, Unit Testing (468 tests)
* **Key Insights:** Demonstrates thermal choking singularities, fixed-point mass flow iteration, and exact oracle validation.

---

## Repository Structure

Each project is contained in its own dedicated directory with dedicated documentation:
* `README.md`: In-depth project overview, theoretical background, and instructions.
* `/src` or root scripts: Core source code and simulation modules.
* `/outputs` or `/assets`: Generated figures, GIFs, and benchmark results.

---

## Contact

Feel free to reach out via GitHub if you want to discuss any of these projects, exchange ideas, or collaborate!
