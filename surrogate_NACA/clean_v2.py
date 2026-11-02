import numpy as np
import pandas as pd

# ---------- Paramètres (à ajuster en regardant vos données) ----------
CD_MIN = 0.002          # Cd plancher physique à Re = 5e5
CD_MAX = 0.05
CL_ABS_MAX = 2.0
ALPHA_PRESTALL = 6.0    # zone où Cl doit croître avec alpha
CL_TOL = -0.02          # tolérance sur la décroissance de Cl
CD_JUMP_MAX = 4.0       # rapport max de Cd entre deux angles consécutifs

df = pd.read_csv("data_clean.csv")
n_profils_avant = df.profile_id.nunique()

def profil_ok(g):
    """Renvoie (True, '') si le profil passe tous les tests, sinon (False, raison)."""
    g = g.sort_values("alpha")

    # Test 1 : bornes physiques sur Cd et Cl
    if g.Cd.min() < CD_MIN or g.Cd.max() > CD_MAX:
        return False, "cd_hors_bornes"
    if g.Cl.abs().max() > CL_ABS_MAX:
        return False, "cl_hors_bornes"

    # Test 2 : Cl monotone croissant en pré-décrochage
    pre = g[g.alpha <= ALPHA_PRESTALL]
    if (np.diff(pre.Cl.values) < CL_TOL).any():
        return False, "cl_non_monotone"

    # Test 3 : pas de saut brutal de Cd entre angles consécutifs
    cd = g.Cd.values
    ratios = np.maximum(cd[1:] / cd[:-1], cd[:-1] / cd[1:])
    if (ratios > CD_JUMP_MAX).any():
        return False, "saut_cd"

    return True, ""

# ---------- Application ----------
resultats = {pid: profil_ok(g) for pid, g in df.groupby("profile_id")}
mauvais = [pid for pid, (ok, _) in resultats.items() if not ok]
raisons = pd.Series([r for ok, r in resultats.values() if not ok]).value_counts()

df_clean = df[~df.profile_id.isin(mauvais)]
df_clean.to_csv("data_clean_v2.csv", index=False)

# ---------- Bilan ----------
n_retires = len(mauvais)
print(f"Profils avant   : {n_profils_avant}")
print(f"Profils retirés : {n_retires} ({100 * n_retires / n_profils_avant:.1f} %)")
print(f"Profils gardés  : {n_profils_avant - n_retires}")
print(f"Points gardés   : {len(df_clean)} / {len(df)}")
print("\nRaisons (un profil est compté pour sa première raison) :")
print(raisons.to_string())

pd.Series(mauvais, name="profile_id").to_csv("profils_retires.csv", index=False)
print("\nIDs retirés sauvegardés dans profils_retires.csv")
print("Le profil 117 est-il retiré ?", 117 in mauvais)