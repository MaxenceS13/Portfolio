"""L'ecoulement de Rayleigh en forme fermee : l'oracle, et la limite.

DEUX ROLES, ET LE SECOND EST LE PLUS IMPORTANT. Ces relations servent d'oracle
au solveur 1D — c'est ce que l'etape 6 a mesure, a 1e-11. Mais elles servent
surtout a savoir, AVANT d'integrer quoi que ce soit, jusqu'ou on a le droit de
chauffer. La chaleur maximale n'est pas une limite numerique qu'on decouvre en
integrant : c'est une limite physique qui s'ecrit.

    T0/T0* = (g+1) M^2 (2 + (g-1) M^2) / (1 + g M^2)^2

Cette fonction vaut 1 en M = 1 et decroit DES DEUX COTES : elle monte de 0 a 1
quand M va de 0 a 1, et redescend de 1 vers « 1 - 1/g^2 = 0,4898 » quand M va de
1 a l'infini. Autrement dit, chauffer pousse le Mach vers 1 par les deux bords,
et « T0* » est la temperature totale que l'on ne peut pas depasser sans que
l'ecoulement se bloque. Au-dela, il n'existe plus de solution stationnaire : le
debit doit diminuer, et pour un statoreacteur cela veut dire que l'entree d'air
se desamorce.

DEUX BRANCHES, DONC DEUX REPONSES A L'INVERSION. Comme pour A/A* a l'etape 2,
c'est a l'appelant de dire de quel cote il se trouve — un rapport de temperature
seul ne le dit pas. Contrairement a A/A*, la fonction n'est PAS plate en M = 1 :
sa derivee y vaut zero aussi, mais l'inversion se fait ici loin du point
critique et le conditionnement n'a pas pose de probleme.

PAS DE SCIPY ICI. Ce module appartient au noyau ; l'inversion se fait par
bissection ecrite a la main, comme celle de « thermo.py », et le test la compare
a « brentq ».
"""

from __future__ import annotations

import numpy as np

from .config import CaseError

BISECTION_STEPS = 80

# Au-dela, l'inversion supersonique n'a plus de sens physique pour ce projet.
MACH_MAX = 50.0


def total_temperature_ratio(mach, gamma: float = 1.4):
    """T0/T0*. Vaut 1 en M = 1 et decroit des deux cotes."""
    mach = np.asarray(mach, dtype=float)
    if np.any(mach <= 0.0):
        raise CaseError(f"le Mach doit etre positif "
                        f"(minimum recu {float(np.min(mach)):g})")
    return ((gamma + 1.0) * mach ** 2 * (2.0 + (gamma - 1.0) * mach ** 2)
            / (1.0 + gamma * mach ** 2) ** 2)


def total_pressure_ratio(mach, gamma: float = 1.4):
    """p0/p0*. C'est elle qui donne la perte de pression d'une chambre."""
    mach = np.asarray(mach, dtype=float)
    return ((1.0 + gamma) / (1.0 + gamma * mach ** 2)
            * ((2.0 + (gamma - 1.0) * mach ** 2) / (gamma + 1.0))
            ** (gamma / (gamma - 1.0)))


def temperature_ratio(mach, gamma: float = 1.4):
    """T/T*."""
    mach = np.asarray(mach, dtype=float)
    return mach ** 2 * ((1.0 + gamma) / (1.0 + gamma * mach ** 2)) ** 2


def pressure_ratio(mach, gamma: float = 1.4):
    """p/p*."""
    mach = np.asarray(mach, dtype=float)
    return (1.0 + gamma) / (1.0 + gamma * mach ** 2)


def supersonic_limit(gamma: float = 1.4) -> float:
    """La valeur de T0/T0* quand M tend vers l'infini : « 1 - 1/gamma^2 ».

    Vaut 0,4898 pour gamma = 1,4. Un ecoulement supersonique ne peut donc pas
    etre chauffe indefiniment non plus — il se bloque aussi, par le haut.
    """
    return 1.0 - 1.0 / gamma ** 2


# -- la limite, qui s'ecrit avant d'integrer --------------------------------

def max_total_temperature_ratio(mach, gamma: float = 1.4):
    """Le plus grand « T0_sortie/T0_entree » admissible, a section constante.

    C'est l'inverse de T0/T0* : chauffer amene T0 vers T0*, et l'atteindre
    bloque l'ecoulement. Mesure a gamma = 1,4 : 21,38 a M = 0,10 ; 5,76 a
    M = 0,20 ; 2,88 a M = 0,30 ; 1,89 a M = 0,40.
    """
    return 1.0 / total_temperature_ratio(mach, gamma)


def max_heat(mach, T0_in: float, cp: float = 1004.5, gamma: float = 1.4):
    """La chaleur maximale par kilogramme, en J/kg. Analytique.

    « q_max = cp T0_in (T0max/T0in - 1) ». Mesure a M = 0,3 et T0 = 300 K :
    T0* = 864,9 K, donc q_max = 567,4 kJ/kg.
    """
    if T0_in <= 0.0:
        raise CaseError(f"T0 d'entree non positive : {T0_in:g} K")
    return cp * T0_in * (max_total_temperature_ratio(mach, gamma) - 1.0)


def critical_total_temperature(mach, T0_in: float, gamma: float = 1.4):
    """T0*, la temperature totale de blocage. Mesure : 864,9 K a M = 0,3."""
    return T0_in * max_total_temperature_ratio(mach, gamma)


# -- l'inversion ------------------------------------------------------------

def mach_from_total_temperature_ratio(ratio, branch: str = "subsonic",
                                      gamma: float = 1.4):
    """Le Mach qui donne ce T0/T0*, sur la branche demandee.

    Bissection vectorisee, comme dans « thermo.py » : SciPy est reserve au
    solveur, et le test compare cette inversion a « brentq ».
    """
    if branch not in ("subsonic", "supersonic"):
        raise CaseError(
            f"branche inconnue : {branch!r} (au choix : subsonic, supersonic)"
        )
    ratio = np.asarray(ratio, dtype=float)
    if np.any(ratio > 1.0):
        raise CaseError(
            f"T0/T0* = {float(np.max(ratio)):g} depasse 1 : c'est exactement "
            f"la definition du blocage, et il n'y a pas de Mach au-dela"
        )
    if np.any(ratio <= 0.0):
        raise CaseError("T0/T0* doit etre strictement positif")

    if branch == "supersonic":
        plancher = supersonic_limit(gamma)
        if np.any(ratio < plancher):
            raise CaseError(
                f"T0/T0* = {float(np.min(ratio)):g} est sous la limite "
                f"supersonique {plancher:g} : aucun Mach n'y correspond"
            )
        lo = np.full(ratio.shape, 1.0)
        hi = np.full(ratio.shape, MACH_MAX)
        sens = -1.0            # la fonction DECROIT sur cette branche
    else:
        lo = np.full(ratio.shape, 1e-12)
        hi = np.full(ratio.shape, 1.0)
        sens = 1.0             # elle croit sur celle-la

    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        trop_petit = sens * (total_temperature_ratio(mid, gamma) - ratio) < 0.0
        lo = np.where(trop_petit, mid, lo)
        hi = np.where(trop_petit, hi, mid)
    return 0.5 * (lo + hi)


def heat_to_mach(mach_in: float, ratio: float, gamma: float = 1.4) -> float:
    """Le Mach atteint apres avoir chauffe d'un facteur « ratio » sur T0.

    Refuse si le rapport demande depasse le maximum : c'est le blocage, et le
    dire AVANT d'integrer est tout l'interet de ce module.
    """
    maximum = float(max_total_temperature_ratio(mach_in, gamma))
    if ratio > maximum:
        raise CaseError(
            f"rapport de temperature {ratio:g} au-dela du blocage : a "
            f"M = {mach_in:g}, la chambre ne peut pas depasser {maximum:.4f} "
            f"(soit {ratio / maximum:.2f} fois trop)"
        )
    branche = "subsonic" if mach_in < 1.0 else "supersonic"
    cible = ratio * float(total_temperature_ratio(mach_in, gamma))
    return float(mach_from_total_temperature_ratio(cible, branche, gamma))
