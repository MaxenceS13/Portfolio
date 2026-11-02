"""
Chronomètre : XFOIL contre le MLP, pour calculer la même polaire (Cl et Cd en fonction de alpha).

Prérequis dans le même dossier : naca.py, mlp_numpy.npz
Lancement (venv où XFOIL est installé) : python bench_time.py
Résultat : affiché dans le terminal + enregistré dans bench_result.json
"""
import os
import sys
import json
import time
import contextlib
import numpy as np

from naca import naca4
from xfoil import XFoil
from xfoil.model import Airfoil

# ---------------------------------------------------------------- Réglages
WEIGHTS_FILE = "mlp_numpy.npz"
RE = 5e5
ALPHAS = np.arange(-4, 13, 1.0)          # 17 angles, comme le dataset
PROFILES = [                              # (m, p, t), dans les bornes d'entraînement
    (0.00, 0.40, 0.12),
    (0.02, 0.40, 0.12),
    (0.04, 0.40, 0.12),
    (0.04, 0.40, 0.18),
    (0.02, 0.30, 0.08),
]
N_REPEAT_MLP = 1000                       # répétitions pour mesurer le MLP (il est très rapide)
N_BATCH = 100_000                         # taille du gros lot (bonus : le MLP traite tout d'un coup)

# ---------------------------------------------------------------- Modèle
data = np.load(WEIGHTS_FILE)
W1, b1, W2, b2, W3, b3 = (data[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"])
X_MEAN, X_STD, Y_MEAN, Y_STD = data["X_mean"], data["X_std"], data["Y_mean"], data["Y_std"]


def mlp_predict(X):
  """X : (N, 4) = [m, p, t, alpha] -> (Cl, Cd), comme dans animate.py."""
  a1 = np.tanh(((X - X_MEAN) / X_STD) @ W1 + b1)
  a2 = np.tanh(a1 @ W2 + b2)
  Y = (a2 @ W3 + b3) * Y_STD + Y_MEAN
  return Y[:, 0], 10 ** Y[:, 1]


def mlp_polar(m, p, t):
  X = np.column_stack([np.full_like(ALPHAS, m), np.full_like(ALPHAS, p),
                       np.full_like(ALPHAS, t), ALPHAS])
  return mlp_predict(X)


# ---------------------------------------------------------------- XFOIL
@contextlib.contextmanager
def silence_c_stdout():
  """XFOIL écrit directement sur la sortie C : on la coupe au niveau du système
  pour que l'affichage dans le terminal ne fausse pas le chrono."""
  sys.stdout.flush()
  saved = os.dup(1)
  devnull = os.open(os.devnull, os.O_WRONLY)
  os.dup2(devnull, 1)
  try:
    yield
  finally:
    sys.stdout.flush()
    os.dup2(saved, 1)
    os.close(devnull)
    os.close(saved)


def xfoil_polar(m, p, t):
  """Polaire complète : création du profil + un calcul par angle."""
  x, y = naca4(m, p, t)
  xf = XFoil()
  xf.airfoil = Airfoil(x, y)
  xf.Re = RE
  xf.max_iter = 100
  xf.print = False
  n_fail = 0
  for a in ALPHAS:
    try:
      cl, cd, _, _ = xf.a(a)
      if not (np.isfinite(cl) and np.isfinite(cd)):
        n_fail += 1
    except Exception:
      n_fail += 1
  return n_fail


# ---------------------------------------------------------------- Mesures
def main():
  # --- XFOIL : une mesure par profil (c'est lent, pas besoin de répéter) ---
  t_xfoil, fails = [], 0
  with silence_c_stdout():
    for prof in PROFILES:
      t0 = time.perf_counter()
      fails += xfoil_polar(*prof)
      t_xfoil.append(time.perf_counter() - t0)
  xfoil_mean = float(np.mean(t_xfoil))

  # --- MLP : échauffement, puis N répétitions sur les mêmes profils ---
  for prof in PROFILES:
    mlp_polar(*prof)
  t0 = time.perf_counter()
  for _ in range(N_REPEAT_MLP):
    for prof in PROFILES:
      mlp_polar(*prof)
  mlp_mean = (time.perf_counter() - t0) / (N_REPEAT_MLP * len(PROFILES))

  # --- MLP : un gros lot de points d'un coup ---
  rng = np.random.default_rng(0)
  Xb = np.column_stack([rng.uniform(0.0, 0.06, N_BATCH), rng.uniform(0.2, 0.6, N_BATCH),
                        rng.uniform(0.06, 0.24, N_BATCH), rng.uniform(-4, 12, N_BATCH)])
  t0 = time.perf_counter()
  mlp_predict(Xb)
  t_batch = time.perf_counter() - t0

  speedup = xfoil_mean / mlp_mean
  result = {
      "n_angles": int(len(ALPHAS)),
      "n_profiles": len(PROFILES),
      "xfoil_ms_per_polar": xfoil_mean * 1e3,
      "mlp_ms_per_polar": mlp_mean * 1e3,
      "speedup": speedup,
      "mlp_batch_points": N_BATCH,
      "mlp_batch_s": t_batch,
      "xfoil_failed_points": fails,
  }
  with open("bench_result.json", "w") as f:
    json.dump(result, f, indent=2)

  print(f"Polaire de {len(ALPHAS)} angles, Re = {RE:.0e}, moyenne sur {len(PROFILES)} profils")
  print(f"  XFOIL : {xfoil_mean * 1e3:10.1f} ms par polaire   (détail : "
        + ", ".join(f"{v * 1e3:.0f}" for v in t_xfoil) + " ms)")
  print(f"  MLP   : {mlp_mean * 1e3:10.3f} ms par polaire")
  print(f"  Rapport : x{speedup:,.0f}")
  print(f"  MLP en un seul lot : {N_BATCH:,} points en {t_batch * 1e3:.0f} ms")
  if fails:
    print(f"  (XFOIL n'a pas convergé sur {fails} points : ils sont inclus dans son temps)")
  print("Résultats enregistrés dans bench_result.json")


if __name__ == "__main__":
  main()