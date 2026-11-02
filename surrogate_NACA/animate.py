"""
Animation GIF : un profil NACA 4 chiffres se déforme (m, p, t varient)
et le MLP prédit en direct les polaires Cl(alpha) et Cd(alpha).

Prérequis dans le même dossier :
  - mlp_numpy.npz  (poids + stats de standardisation, créé par le script d'entraînement)
  - naca.py           (fonction naca4(m, p, t) -> X, Y)
  - xfoil_overlay.py  (calcule les points XFOIL, puis les garde dans xfoil_cache.npz)

Lancement : python animate.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from naca import naca4
from xfoil_overlay import load_or_compute, key_of, XFOIL_ALPHAS

# ---------------------------------------------------------------- Réglages
WEIGHTS_FILE = "mlp_numpy.npz"
OUTPUT_GIF = "naca_morph.gif"
SHOW_XFOIL = True         # points noirs = vrais calculs XFOIL (cache : xfoil_cache.npz)
import json
try:
  with open("bench_result.json") as f:
    BENCH = json.load(f)
except FileNotFoundError:
  BENCH = None          # pas de mesure : l'encadré n'est simplement pas affiché
FPS = 25
FRAMES_PER_SEGMENT = 30   # images pour passer d'un profil au suivant
HOLD_FRAMES = 8           # images de pause sur chaque profil clé
ALPHAS = np.linspace(-4, 12, 80)   # plage d'angles tracée (celle du dataset)

# Profils clés (m, p, t) : à garder DANS les bornes d'entraînement
# m : 0-0.06 | p : 0.2-0.6 | t : 0.06-0.24
KEYFRAMES = [
    (0.00, 0.40, 0.12),   # NACA 0012
    (0.02, 0.40, 0.12),   # NACA 2412
    (0.04, 0.40, 0.12),   # NACA 4412
    (0.04, 0.40, 0.18),   # NACA 4418
    (0.02, 0.30, 0.08),   # NACA 2308
    (0.00, 0.40, 0.12),   # retour au début -> boucle propre
]

# ---------------------------------------------------------------- Modèle
data = np.load(WEIGHTS_FILE)
PARAMS = {k: data[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
X_MEAN, X_STD = data["X_mean"], data["X_std"]
Y_MEAN, Y_STD = data["Y_mean"], data["Y_std"]


def forward(X):
  a1 = np.tanh(X @ PARAMS["W1"] + PARAMS["b1"])
  a2 = np.tanh(a1 @ PARAMS["W2"] + PARAMS["b2"])
  return a2 @ PARAMS["W3"] + PARAMS["b3"]


def predict_polar(m, p, t, alphas):
  """Renvoie Cl(alpha) et Cd(alpha) prédits par le MLP pour un profil."""
  X = np.column_stack([np.full_like(alphas, m), np.full_like(alphas, p),
                       np.full_like(alphas, t), alphas])
  Y = forward((X - X_MEAN) / X_STD) * Y_STD + Y_MEAN
  return Y[:, 0], 10 ** Y[:, 1]          # Cl, Cd (on repasse du log10)


# ---------------------------------------------------------------- Géométrie
def camber_line(m, p, n=100):
  """Ligne moyenne (cambrure) d'un NACA 4 chiffres."""
  x = np.linspace(0, 1, n)
  if m == 0:
    return x, np.zeros_like(x)
  yc = np.where(x < p,
                m / p**2 * (2 * p * x - x**2),
                m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
  return x, yc


# ---------------------------------------------------------------- Trajectoire
def smoothstep(u):
  return 3 * u**2 - 2 * u**3            # démarre et finit en douceur


def build_path():
  """Liste de (m, p, t) image par image, avec pauses sur les profils clés."""
  path = []
  for k in range(len(KEYFRAMES) - 1):
    a, b = np.array(KEYFRAMES[k]), np.array(KEYFRAMES[k + 1])
    path += [tuple(a)] * HOLD_FRAMES
    for i in range(FRAMES_PER_SEGMENT):
      s = smoothstep(i / FRAMES_PER_SEGMENT)
      path.append(tuple(a + s * (b - a)))
  path += [tuple(KEYFRAMES[-1])] * HOLD_FRAMES
  return path


def naca_name(m, p, t):
  # Sans cambrure, le profil est symétrique : NACA 00XX (p n'a pas de sens)
  p_digit = round(p * 10) if m >= 0.0025 else 0
  return f"NACA {round(m * 100)}{p_digit}{round(t * 100):02d}"


def physical_text(m, p, t):
  """Signification physique de m, p, t (en anglais, en % de la corde)."""
  # Sans cambrure, la position du maximum de cambrure n'a pas de sens
  pos = f"{p * 100:.0f} % of chord" if m >= 0.0025 else "-"
  return (f"Max camber: {m * 100:.1f} % of chord\n"
          f"Max camber position: {pos}\n"
          f"Max thickness: {t * 100:.1f} % of chord")


# ---------------------------------------------------------------- Figure
path = build_path()

# Polaires XFOIL de tous les profils de l'animation (long la 1re fois, ensuite en cache)
XFOIL = load_or_compute(path) if SHOW_XFOIL else {}

fig = plt.figure(figsize=(11, 5.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
ax_foil = fig.add_subplot(gs[:, 0])
ax_cl = fig.add_subplot(gs[0, 1])
ax_cd = fig.add_subplot(gs[1, 1])

# Profil
ax_foil.set_xlim(-0.05, 1.05)
ax_foil.set_ylim(-0.40, 0.22)
ax_foil.set_aspect("equal")
ax_foil.axis("off")
foil_patch = Polygon(np.zeros((3, 2)), closed=True, facecolor="#4C72B0", edgecolor="black")
ax_foil.add_patch(foil_patch)
(line_camber,) = ax_foil.plot([], [], color="#E8A33D", lw=2, ls="--", label="Mean camber line")
ax_foil.legend(loc="upper right", frameon=False, fontsize=9)
title = ax_foil.set_title("", fontsize=16, fontweight="bold")
params_text = ax_foil.text(0.5, -0.20, "", ha="center", va="top", fontsize=11, linespacing=1.6)
if BENCH:
  bench_text = (f"XFOIL {BENCH['xfoil_ms_per_polar']:,.0f} ms   vs   MLP {BENCH['mlp_ms_per_polar']:.2g} ms\n"
                f"~x{BENCH['speedup']:,.0f} faster ({BENCH['n_angles']}-angle polar)")
  ax_foil.text(0.5, -0.44, bench_text, ha="center", va="top", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F4F4", edgecolor="#999999"))

# Cl(alpha)
(line_cl,) = ax_cl.plot([], [], color="#C44E52", lw=2, label="MLP")
(dots_cl,) = ax_cl.plot([], [], "o", color="black", ms=4, zorder=3, label="XFOIL")
ax_cl.legend(loc="upper left", frameon=False, fontsize=9)
ax_cl.set_xlim(ALPHAS[0], ALPHAS[-1])
ax_cl.set_ylim(-0.6, 1.9)
ax_cl.set_ylabel("Lift coefficient Cl")
ax_cl.grid(True, linestyle=":", alpha=0.7)
ax_cl.set_title("MLP prediction", fontsize=11)

# Cd(alpha)
(line_cd,) = ax_cd.plot([], [], color="#55A868", lw=2, label="MLP")
(dots_cd,) = ax_cd.plot([], [], "o", color="black", ms=4, zorder=3, label="XFOIL")
ax_cd.legend(loc="upper left", frameon=False, fontsize=9)
ax_cd.set_xlim(ALPHAS[0], ALPHAS[-1])
ax_cd.set_ylim(0.003, 0.06)
ax_cd.set_yscale("log")
ax_cd.set_xlabel("Angle of attack α (°)")
ax_cd.set_ylabel("Drag coefficient Cd")
ax_cd.grid(True, which="both", linestyle=":", alpha=0.7)

fig.tight_layout()


def update(frame):
  m, p, t = path[frame]
  x, y = naca4(m, p, t)
  foil_patch.set_xy(np.column_stack([x, y]))
  xc, yc = camber_line(m, p)
  line_camber.set_data(xc, yc)
  title.set_text(naca_name(m, p, t))
  params_text.set_text(physical_text(m, p, t))

  cl, cd = predict_polar(m, p, t, ALPHAS)
  line_cl.set_data(ALPHAS, cl)
  line_cd.set_data(ALPHAS, cd)

  # Points XFOIL (les NaN, c'est-à-dire les calculs échoués ou écartés, ne sont pas tracés)
  if SHOW_XFOIL:
    cl_x, cd_x = XFOIL[key_of(m, p, t)]
    dots_cl.set_data(XFOIL_ALPHAS, cl_x)
    dots_cd.set_data(XFOIL_ALPHAS, cd_x)
  return foil_patch, line_camber, line_cl, dots_cl, line_cd, dots_cd, title, params_text


if __name__ == "__main__":
  anim = FuncAnimation(fig, update, frames=len(path), interval=1000 / FPS, blit=False)
  print(f"Génération de {len(path)} images -> {OUTPUT_GIF} ...")
  anim.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS), dpi=100)
  print("Terminé.")