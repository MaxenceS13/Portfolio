"""
Calcule (une seule fois) les polaires XFOIL des profils traversés par l'animation,
et les garde dans un fichier cache pour ne pas refaire les calculs.

Utilisé par animate.py. Peut aussi se lancer seul : python xfoil_overlay.py
"""
import os
import numpy as np

from naca import naca4

RE = 5e5                                   # le même Reynolds que le dataset
XFOIL_ALPHAS = np.arange(-4, 13, 1.0)      # les mêmes angles que le dataset
CACHE_FILE = "xfoil_cache.npz"

# Mêmes bornes physiques que dans clean_v2.py : on n'affiche pas les points
# que le nettoyage du dataset aurait écartés (XFOIL non convergé, Cd absurde).
CD_MIN, CD_MAX, CL_ABS_MAX = 0.002, 0.05, 2.0


def key_of(m, p, t):
  """Clé d'un profil (arrondie pour que le lookup soit fiable)."""
  return tuple(round(float(v), 5) for v in (m, p, t))


def xfoil_polar(m, p, t, alphas, re=RE):
  """Cl(alpha), Cd(alpha) calculés par XFOIL. NaN là où XFOIL échoue ou est écarté."""
  from xfoil import XFoil                 # import ici : inutile si tout est en cache
  from xfoil.model import Airfoil

  x, y = naca4(m, p, t)
  xf = XFoil()
  xf.airfoil = Airfoil(x, y)
  xf.Re = re
  xf.max_iter = 100
  xf.print = False

  cl = np.full(len(alphas), np.nan)
  cd = np.full(len(alphas), np.nan)
  for k, a in enumerate(alphas):
    try:
      c_l, c_d, _, _ = xf.a(a)
    except Exception:
      continue
    cl[k], cd[k] = c_l, c_d

  bad = (~np.isfinite(cl) | ~np.isfinite(cd) | (cd < CD_MIN) | (cd > CD_MAX)
         | (np.abs(cl) > CL_ABS_MAX))
  cl[bad] = np.nan
  cd[bad] = np.nan
  return cl, cd


def _save(store, alphas, cache_file):
  keys = list(store)
  np.savez(cache_file,
           keys=np.array(keys),
           cl=np.array([store[k][0] for k in keys]),
           cd=np.array([store[k][1] for k in keys]),
           alphas=alphas, re=RE)


def load_or_compute(path, alphas=XFOIL_ALPHAS, cache_file=CACHE_FILE):
  """
  path : liste de (m, p, t), un par image de l'animation (avec doublons).
  Renvoie un dict {clé_du_profil: (Cl, Cd)}, calculé par XFOIL ou lu dans le cache.
  """
  keys = list(dict.fromkeys(key_of(*pt) for pt in path))   # profils uniques

  store = {}
  if os.path.exists(cache_file):
    c = np.load(cache_file)
    if np.array_equal(c["alphas"], alphas) and float(c["re"]) == RE:
      for k, cl, cd in zip(c["keys"], c["cl"], c["cd"]):
        store[tuple(float(v) for v in k)] = (cl, cd)

  missing = [k for k in keys if k not in store]
  if missing:
    print(f"XFOIL : {len(missing)} profils à calculer "
          f"({len(keys) - len(missing)} déjà en cache)")
    for n, k in enumerate(missing, 1):
      store[k] = xfoil_polar(*k, alphas)
      if n % 10 == 0 or n == len(missing):
        print(f"  {n}/{len(missing)}")
        _save(store, alphas, cache_file)           # sauvegarde régulière
  return store


if __name__ == "__main__":
  # Test rapide : un seul profil (NACA 2412)
  cl, cd = xfoil_polar(0.02, 0.4, 0.12, XFOIL_ALPHAS)
  for a, c_l, c_d in zip(XFOIL_ALPHAS, cl, cd):
    print(f"alpha={a:>4.0f}  Cl={c_l:.4f}  Cd={c_d:.5f}")