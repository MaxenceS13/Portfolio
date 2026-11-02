import numpy as np
import pandas as pd
from scipy.stats import qmc
from xfoil import XFoil
from xfoil.model import Airfoil
from naca import naca4

N_PROFILES = 300
ALPHAS = np.arange(-4, 13, 1.0)      # 17 angles
RE = 5e5

# Échantillonnage LHS des profils : m, p, t
sampler = qmc.LatinHypercube(d=3, seed=42)
u = sampler.random(N_PROFILES)
lo = [0.0, 0.2, 0.06]
hi = [0.06, 0.6, 0.24]
params = qmc.scale(u, lo, hi)

rows = []
for pid, (m, p, t) in enumerate(params):
    try:
        x, y = naca4(m, p, t)
        xf = XFoil()
        xf.airfoil = Airfoil(x, y)
        xf.Re = RE
        xf.max_iter = 100
        xf.print = False
        for alpha in ALPHAS:
            cl, cd, cm, cp = xf.a(alpha)
            rows.append((pid, m, p, t, alpha, RE, cl, cd))
    except Exception as e:
        print(f"profil {pid} ignoré: {e}")
    if pid % 20 == 0:
        print(f"{pid}/{N_PROFILES}")
        pd.DataFrame(rows, columns=["profile_id","m","p","t","alpha","Re","Cl","Cd"]) \
          .to_csv("data_raw.csv", index=False)

df = pd.DataFrame(rows, columns=["profile_id","m","p","t","alpha","Re","Cl","Cd"])
df.to_csv("data_raw.csv", index=False)

# Nettoyage
n0 = len(df)
df = df.dropna()
df = df[(df.Cd > 1e-4) & (df.Cd < 0.05) & (df.Cl.abs() < 2.0)]
df.to_csv("data_clean.csv", index=False)
print(f"{n0} points bruts -> {len(df)} points propres")