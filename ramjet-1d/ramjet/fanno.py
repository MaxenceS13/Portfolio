"""L'ecoulement de Fanno en forme fermee : le troisieme oracle.

ETAPE 12. Rayleigh decrit la chaleur a section constante, l'isentropique decrit
la section sans chaleur ; Fanno decrit le FROTTEMENT a section constante et sans
chaleur. Les trois se referment sur les memes variables, et ensemble ils
couvrent chacun des trois termes de l'equation quasi-1D separement. Chaque terme
du solveur a donc son oracle exact, ce qui est la seule facon de savoir lequel
est faux quand un profil deraille.

    4 C_f L*/D = (1 - M^2)/(gamma M^2)
                 + (gamma+1)/(2 gamma) ln[ (gamma+1) M^2 / (2 + (gamma-1) M^2) ]

« L* » est la longueur qu'il reste a parcourir pour atteindre M = 1. Elle
DECROIT vers zero des deux cotes : le frottement, comme la chaleur, pousse le
Mach vers 1 par les deux bords. Un conduit plus long que « L* » n'a pas de
solution stationnaire — c'est le blocage par frottement, et c'est exactement la
meme nature de limite que le blocage thermique de l'etape 7.

CE QUI SURPREND, ET QUI EST VRAI : LE FROTTEMENT ACCELERE UN ECOULEMENT
SUBSONIQUE. On s'attend a ce qu'il freine. Il freine la vitesse debitante en
faisant chuter la pression, mais la temperature chute aussi, donc la vitesse du
son, et le rapport des deux monte. En supersonique, il decelere. Les deux
comportements sont le meme fait : le Mach va vers 1.

UNE IDENTITE REMARQUABLE, ET ELLE SERT DE TEST. Le rapport « p0/p0* » de Fanno
est LITTERALEMENT la meme fonction que « A/A* » de l'isentropique. Ce n'est pas
une coincidence — les deux expriment que le debit critique passe par une section
critique donnee — et cela donne un controle croise gratuit entre deux modules
ecrits separement.

PAS DE SCIPY ICI : module du noyau, bissection ecrite a la main.
"""

from __future__ import annotations

import numpy as np

from .config import CaseError

BISECTION_STEPS = 80
MACH_MAX = 50.0


def friction_length(mach, gamma: float = 1.4):
    """« 4 C_f L*/D » : la longueur reduite qui reste avant M = 1.

    Vaut 0 en M = 1 et croit des deux cotes. Mesure a gamma = 1,4 : 66,9216 a
    M = 0,10 ; 1,0691 a M = 0,50 ; 0,0723 a M = 0,80 ; 0,3050 a M = 2 ; et elle
    plafonne a 0,8215 quand M tend vers l'infini.

    L'ASYMETRIE DES DEUX BRANCHES EST FRAPPANTE, et elle a un sens physique. En
    subsonique la longueur reduite n'est bornee par rien — a M = 0,10 il reste
    66,92, soit 900 fois ce que M = 2 laisse. En supersonique elle plafonne a
    0,8215. Autrement dit : un ecoulement supersonique ne peut jamais parcourir
    qu'une longueur bornee avant de se bloquer, si vite qu'il aille, tandis
    qu'un ecoulement lent peut aller presque indefiniment loin.
    """
    mach = np.asarray(mach, dtype=float)
    if np.any(mach <= 0.0):
        raise CaseError(f"le Mach doit etre positif "
                        f"(minimum recu {float(np.min(mach)):g})")
    carre = mach ** 2
    return ((1.0 - carre) / (gamma * carre)
            + (gamma + 1.0) / (2.0 * gamma)
            * np.log((gamma + 1.0) * carre / (2.0 + (gamma - 1.0) * carre)))


def supersonic_limit(gamma: float = 1.4) -> float:
    """La valeur de « 4 C_f L*/D » quand M tend vers l'infini.

    « -1/gamma + (gamma+1)/(2 gamma) ln((gamma+1)/(gamma-1)) », soit 0,8215 a
    gamma = 1,4. Un ecoulement supersonique ne peut donc jamais parcourir plus
    que cette longueur reduite, si vite qu'il aille : c'est le pendant exact de
    la limite « 1 - 1/gamma^2 » de Rayleigh.
    """
    return (-1.0 / gamma + (gamma + 1.0) / (2.0 * gamma)
            * float(np.log((gamma + 1.0) / (gamma - 1.0))))


def temperature_ratio(mach, gamma: float = 1.4):
    """T/T* = (gamma+1) / (2 + (gamma-1) M^2)."""
    mach = np.asarray(mach, dtype=float)
    return (gamma + 1.0) / (2.0 + (gamma - 1.0) * mach ** 2)


def pressure_ratio(mach, gamma: float = 1.4):
    """p/p* = (1/M) sqrt(T/T*)."""
    mach = np.asarray(mach, dtype=float)
    return np.sqrt(temperature_ratio(mach, gamma)) / mach


def total_pressure_ratio(mach, gamma: float = 1.4):
    """p0/p0*, ET C'EST EXACTEMENT « A/A* » DE L'ISENTROPIQUE.

    « (1/M) [(2 + (gamma-1) M^2)/(gamma+1)]^((gamma+1)/(2(gamma-1))) ». Le test
    compare terme a terme avec « thermo.area_ratio » : deux modules ecrits
    separement doivent rendre le meme nombre a la derniere decimale.
    """
    mach = np.asarray(mach, dtype=float)
    exposant = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return ((1.0 / mach)
            * ((2.0 + (gamma - 1.0) * mach ** 2) / (gamma + 1.0)) ** exposant)


def velocity_ratio(mach, gamma: float = 1.4):
    """u/u* = M sqrt(T/T*)."""
    mach = np.asarray(mach, dtype=float)
    return mach * np.sqrt(temperature_ratio(mach, gamma))


# -- la limite, qui s'ecrit avant d'integrer --------------------------------

def max_length(mach, diameter: float, friction: float,
               gamma: float = 1.4) -> float:
    """La longueur physique maximale, en metres, avant blocage par frottement.

    « L_max = D/(4 C_f) . (4 C_f L*/D) ». Mesure pour D = 0,185 m et
    C_f = 0,003 : **224,0 m** a M = 0,20 et 16,5 m a M = 0,50. Une chambre de
    0,60 m est donc a trois ordres de grandeur du blocage par frottement seul —
    ce qui ne veut pas dire que le frottement ne fait rien, seulement qu'il ne
    bloque pas a lui seul. Ce qu'il fait, c'est RAPPROCHER du blocage
    thermique, et c'est cela qu'il faut mesurer.
    """
    if diameter <= 0.0:
        raise CaseError(f"diametre non positif : {diameter:g} m")
    if friction <= 0.0:
        raise CaseError(f"coefficient de frottement non positif : {friction:g}")
    return float(friction_length(mach, gamma)) * diameter / (4.0 * friction)


def mach_from_friction_length(value, branch: str = "subsonic",
                              gamma: float = 1.4):
    """Le Mach qui correspond a cette longueur reduite, sur la branche demandee.

    Deux branches, comme partout ailleurs : c'est a l'appelant de dire de quel
    cote de M = 1 il se trouve.
    """
    if branch not in ("subsonic", "supersonic"):
        raise CaseError(
            f"branche inconnue : {branch!r} (au choix : subsonic, supersonic)"
        )
    value = np.asarray(value, dtype=float)
    if np.any(value < 0.0):
        raise CaseError(
            f"longueur reduite negative ({float(np.min(value)):g}) : "
            f"« 4 C_f L*/D » vaut zero en M = 1 et croit des deux cotes"
        )
    if branch == "supersonic":
        plafond = supersonic_limit(gamma)
        if np.any(value > plafond):
            raise CaseError(
                f"longueur reduite {float(np.max(value)):g} au-dela de la "
                f"limite supersonique {plafond:g} : aucun Mach n'y correspond, "
                f"si grand soit-il"
            )
        lo = np.full(value.shape, 1.0)
        hi = np.full(value.shape, MACH_MAX)
        sens = 1.0
    else:
        lo = np.full(value.shape, 1e-6)
        hi = np.full(value.shape, 1.0)
        sens = -1.0

    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        trop_petit = sens * (friction_length(mid, gamma) - value) < 0.0
        lo = np.where(trop_petit, mid, lo)
        hi = np.where(trop_petit, hi, mid)
    return 0.5 * (lo + hi)


def mach_after(mach_in: float, length: float, diameter: float,
               friction: float, gamma: float = 1.4) -> float:
    """Le Mach apres une longueur donnee de conduit a section constante.

    On avance sur l'axe « 4 C_f L*/D » : parcourir « L » consomme
    « 4 C_f L/D » de la longueur reduite restante. Refuse si le conduit est
    plus long que ce qui reste, parce que c'est le blocage.
    """
    reste = float(friction_length(mach_in, gamma))
    consomme = 4.0 * friction * length / diameter
    if consomme > reste:
        raise CaseError(
            f"blocage par frottement : un conduit de {length:g} m consomme "
            f"{consomme:.4f} de longueur reduite la ou M = {mach_in:g} n'en "
            f"laisse que {reste:.4f} (soit {consomme / reste:.2f} fois trop). "
            f"Longueur maximale : "
            f"{max_length(mach_in, diameter, friction, gamma):.4f} m"
        )
    branche = "subsonic" if mach_in < 1.0 else "supersonic"
    return float(mach_from_friction_length(reste - consomme, branche, gamma))
