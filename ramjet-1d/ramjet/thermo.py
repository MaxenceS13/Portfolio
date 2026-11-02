"""Gaz parfait : relations isentropiques, rapport de section, choc normal.

CE MODULE EST LA REFERENCE DE TOUT LE RESTE. Le cycle 0D s'en sert pour passer
d'une station a l'autre, le conduit pour dimensionner ses sections, le solveur
1D pour reconstruire les grandeurs statiques. Chacune de ses fonctions a une
forme fermee, donc chacune est verifiable contre une table publiee — et c'est
pour ca qu'il est ecrit en premier.

L'INVERSION DE A/A* EST LE SEUL ENDROIT DIFFICILE, ET SA DIFFICULTE EST
PHYSIQUE. La fonction A/A*(M) a un MINIMUM en M = 1, ou elle vaut 1 : elle y est
donc plate, et son developpement s'ecrit « A/A* ~ 1 + 2/(gamma+1) (M-1)^2 » —
coefficient MESURE a l'etape 8 sur trois gamma, apres avoir d'abord ecrit
« (gamma+1)/4 » de memoire, ce qui predisait 1,05e-3 la ou l'on mesure 8,92e-4.
Une erreur eps sur A/A* donne une erreur en sqrt(eps) sur M au voisinage du col,
et aucun algorithme n'y peut rien : l'information n'est pas dans le nombre.
Mesure de l'aller-retour « M -> A/A* -> M » a l'etape 2 :

    M            1,0001    1,001     1,01      1,05      2,0
    erreur sur M 1,1e-12   2,0e-13   3,3e-14   2,9e-15   < 1e-15

C'est pourquoi les tests exigent 1e-12 sur [1,05 ; 6] et [0,05 ; 0,95], et
mesurent la degradation pres du col au lieu de l'ignorer. Un dimensionnement de
col ne doit jamais passer par cette inversion : c'est la section qui se deduit
du Mach, et pas l'inverse.

PAS DE SCIPY ICI, ET C'EST UNE CONTRAINTE D'ARCHITECTURE. L'inversion se fait
par bissection ecrite a la main, vectorisee. « scipy.optimize.brentq » ferait la
meme chose en une ligne, mais SciPy est reserve a « solver1d.py » ; en echange,
le test compare la bissection a « brentq » et mesure l'ecart entre les deux, ce
qui vaut mieux qu'une ligne de moins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bornes de recherche de l'inversion. A/A* vaut 4,6e7 a M = 100 : aucun
# divergent reel n'en demande autant, et refuser au-dela vaut mieux que de
# chercher dans un intervalle ou la fonction deborde.
MACH_MAX = 100.0

# Nombre d'iterations de bissection. Sur [1, 100], soixante iterations amenent
# l'intervalle a 1e-16 ; quatre-vingts laissent de la marge et ne coutent rien,
# la bissection etant vectorisee sur tout le tableau a la fois.
BISECTION_STEPS = 80


class ThermoError(ValueError):
    """Etat thermodynamique impossible, ou hors du domaine d'une relation."""


# -- relations isentropiques ------------------------------------------------

def total_temperature_ratio(mach, gamma: float = 1.4):
    """T0/T. La relation dont toutes les autres se deduisent."""
    mach = np.asarray(mach, dtype=float)
    return 1.0 + 0.5 * (gamma - 1.0) * mach ** 2


def total_pressure_ratio(mach, gamma: float = 1.4):
    """p0/p, par l'isentropique."""
    return total_temperature_ratio(mach, gamma) ** (gamma / (gamma - 1.0))


def total_density_ratio(mach, gamma: float = 1.4):
    """rho0/rho, par l'isentropique."""
    return total_temperature_ratio(mach, gamma) ** (1.0 / (gamma - 1.0))


def mach_from_total_pressure_ratio(ratio, gamma: float = 1.4):
    """Le Mach qui correspond a un rapport p0/p donne.

    Inversion analytique, sans iteration : c'est celle dont la tuyere de
    l'etape 3 aura besoin pour trouver le Mach de sortie a partir du rapport de
    detente, et elle n'a aucun des problemes de conditionnement de A/A*.
    """
    ratio = np.asarray(ratio, dtype=float)
    if np.any(ratio < 1.0):
        raise ThermoError(
            f"p0/p doit valoir au moins 1 (minimum recu {float(np.min(ratio)):g}) : "
            f"la pression totale ne peut pas etre inferieure a la statique"
        )
    return np.sqrt(2.0 / (gamma - 1.0) * (ratio ** ((gamma - 1.0) / gamma) - 1.0))


def area_ratio(mach, gamma: float = 1.4):
    """A/A*, le rapport de la section a la section critique.

    Vaut 1 en M = 1 et croit des deux cotes : c'est une fonction a MINIMUM, donc
    a deux branches, et c'est toute la difficulte de son inversion.
    """
    mach = np.asarray(mach, dtype=float)
    if np.any(mach <= 0.0):
        raise ThermoError(
            f"le Mach doit etre positif (minimum recu {float(np.min(mach)):g})"
        )
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / mach) * ((2.0 / (gamma + 1.0))
                           * total_temperature_ratio(mach, gamma)) ** exponent


def area_ratio_derivative(mach, gamma: float = 1.4):
    """d(A/A*)/dM = (A/A*) (M^2 - 1) / (M (1 + (gamma-1)/2 M^2)).

    Elle s'annule en M = 1, ce qui est la forme analytique du probleme de
    conditionnement decrit en tete de module.
    """
    mach = np.asarray(mach, dtype=float)
    return (area_ratio(mach, gamma) * (mach ** 2 - 1.0)
            / (mach * total_temperature_ratio(mach, gamma)))


def mach_from_area_ratio(ratio, branch: str = "subsonic", gamma: float = 1.4):
    """Le Mach qui donne ce rapport de section, sur la branche demandee.

    Bissection vectorisee. La branche doit etre choisie par l'appelant parce
    qu'il y a DEUX reponses : c'est la geometrie du conduit et la pression aval
    qui disent laquelle, jamais le rapport de section seul.
    """
    if branch not in ("subsonic", "supersonic"):
        raise ThermoError(
            f"branche inconnue : {branch!r} (au choix : subsonic, supersonic)"
        )
    ratio = np.asarray(ratio, dtype=float)
    if np.any(ratio < 1.0):
        raise ThermoError(
            f"A/A* doit valoir au moins 1 (minimum recu {float(np.min(ratio)):g}) : "
            f"aucune section ne peut etre plus petite que la section critique"
        )
    if branch == "supersonic":
        plafond = area_ratio(MACH_MAX, gamma)
        if np.any(ratio > plafond):
            raise ThermoError(
                f"A/A* = {float(np.max(ratio)):g} depasse ce que M = {MACH_MAX:g} "
                f"donne ({float(plafond):g}) : aucun divergent reel ne va si loin"
            )
        lo = np.full(ratio.shape, 1.0)
        hi = np.full(ratio.shape, MACH_MAX)
    else:
        # A/A* -> +inf quand M -> 0 : la borne basse est un Mach minuscule et
        # non zero, ou la fonction n'est pas definie.
        lo = np.full(ratio.shape, 1e-12)
        hi = np.full(ratio.shape, 1.0)

    # La fonction est decroissante en subsonique, croissante en supersonique :
    # un seul signe suffit a traiter les deux cas.
    sens = 1.0 if branch == "supersonic" else -1.0
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        trop_petit = sens * (area_ratio(mid, gamma) - ratio) < 0.0
        lo = np.where(trop_petit, mid, lo)
        hi = np.where(trop_petit, hi, mid)
    return 0.5 * (lo + hi)


# -- vitesses ---------------------------------------------------------------

def speed_of_sound(temperature, gamma: float = 1.4, R: float = 287.05):
    """a = sqrt(gamma R T)."""
    temperature = np.asarray(temperature, dtype=float)
    if np.any(temperature <= 0.0):
        raise ThermoError(
            f"temperature non positive : {float(np.min(temperature)):g} K"
        )
    return np.sqrt(gamma * R * temperature)


def velocity(mach, temperature, gamma: float = 1.4, R: float = 287.05):
    """u = M a, avec « a » pris a la temperature STATIQUE.

    Le rappel n'est pas superflu : prendre la temperature totale est l'erreur
    la plus frequente de ce calcul. La vitesse allant en sqrt(T), elle
    surestime de sqrt(T0/T), soit sqrt(2,8) = 1,673 a M = 3 — 67 % de trop,
    sans rien casser d'autre et sans se voir sur aucune figure.
    """
    return np.asarray(mach, dtype=float) * speed_of_sound(temperature, gamma, R)


# -- choc normal ------------------------------------------------------------

@dataclass(frozen=True)
class NormalShock:
    """L'etat de part et d'autre d'un choc droit, en rapports."""

    mach_upstream: float
    mach_downstream: float
    pressure_ratio: float          # p2/p1
    temperature_ratio: float       # T2/T1
    density_ratio: float           # rho2/rho1
    total_pressure_ratio: float    # p02/p01 — la seule qui mesure la perte

    def describe(self) -> str:
        return (f"choc droit  M {self.mach_upstream:.3f} -> "
                f"{self.mach_downstream:.4f}   p2/p1 = {self.pressure_ratio:.3f}   "
                f"p02/p01 = {self.total_pressure_ratio:.4f}")


def normal_shock(mach: float, gamma: float = 1.4) -> NormalShock:
    """Relations de Rankine-Hugoniot pour un choc droit.

    REFUSE un Mach amont inferieur a 1. Les formules donneraient un resultat —
    elles sont algebriquement valides — mais ce resultat aurait « p02/p01 > 1 »,
    c'est-a-dire une DETENTE de choc, qui viole le second principe. Un choc de
    detente n'existe pas, et rendre un nombre plutot qu'une erreur laisserait
    croire le contraire.
    """
    mach = float(mach)
    if mach < 1.0:
        raise ThermoError(
            f"choc droit demande a M = {mach:g} : un choc de detente violerait "
            f"le second principe (il faut M >= 1)"
        )
    m2 = mach ** 2
    downstream_sq = ((1.0 + 0.5 * (gamma - 1.0) * m2)
                     / (gamma * m2 - 0.5 * (gamma - 1.0)))
    pressure = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)
    density = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)
    temperature = pressure / density
    total_pressure = (density ** (gamma / (gamma - 1.0))
                      * (1.0 / pressure) ** (1.0 / (gamma - 1.0)))
    return NormalShock(mach_upstream=mach,
                       mach_downstream=float(np.sqrt(downstream_sq)),
                       pressure_ratio=float(pressure),
                       temperature_ratio=float(temperature),
                       density_ratio=float(density),
                       total_pressure_ratio=float(total_pressure))


def strong_shock_limit(gamma: float = 1.4) -> float:
    """Mach aval quand le Mach amont tend vers l'infini : sqrt((gamma-1)/2gamma).

    Vaut 0,37796 pour gamma = 1,4. Un choc, aussi fort soit-il, ne ralentit
    jamais l'ecoulement en dessous de cette valeur — et c'est un test de limite
    que les relations doivent verifier sans qu'on la leur ait donnee.
    """
    return float(np.sqrt((gamma - 1.0) / (2.0 * gamma)))


# -- chocs obliques : l'entree d'air que le 0D ne fait que resumer -----------
#
# ETAPE 13. Le cycle 0D remplace toute l'entree d'air par un nombre, « pi_d »,
# tire de la correlation MIL-E-5008B. Ce qui suit calcule ce nombre a partir
# d'une geometrie de rampes, ce qui permet enfin de repondre a la question que
# l'etape 8 avait laissee ouverte : la correlation decrit une entree « bien
# dessinee », mais bien dessinee COMMENT ?

# Nombres d'iterations, choisis apres mesure. L'optimisation a trois rampes
# coute 788 ms a (80, 90) et 191 ms a (40, 60), pour un ecart de 3,5e-10 sur les
# angles et 2,3e-15 sur la recuperation : le surcout ne payait rien.
BETA_STEPS = 60            # bissection sur l'angle de choc
GOLDEN_STEPS = 40          # section doree sur la force des chocs
# Nombre d'or, pour la recherche du maximum de recuperation.
INVERSE_GOLDEN = (5.0 ** 0.5 - 1.0) / 2.0


@dataclass(frozen=True)
class ObliqueShock:
    """Un choc oblique attache, solution FAIBLE."""

    mach_upstream: float
    deflection: float          # theta, radians
    wave_angle: float          # beta, radians
    normal_mach: float         # M1 sin(beta) : la « force » du choc
    mach_downstream: float
    total_pressure_ratio: float
    pressure_ratio: float
    temperature_ratio: float

    def describe(self) -> str:
        return (f"  rampe {np.degrees(self.deflection):5.2f} deg  ->  "
                f"beta {np.degrees(self.wave_angle):5.2f} deg   "
                f"M {self.mach_upstream:.4f} -> {self.mach_downstream:.4f}   "
                f"pi = {self.total_pressure_ratio:.4f}")


def _deflection(mach: float, beta: float, gamma: float) -> float:
    """La relation theta-beta-M, ecrite dans le sens direct.

    « tan(theta) = 2 cot(beta) (M^2 sin^2(beta) - 1)
                   / (M^2 (gamma + cos 2 beta) + 2) »
    """
    numerateur = 2.0 / np.tan(beta) * (mach ** 2 * np.sin(beta) ** 2 - 1.0)
    denominateur = mach ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0
    return float(np.arctan(numerateur / denominateur))


def max_deflection(mach: float, gamma: float = 1.4) -> float:
    """La deviation maximale pour laquelle le choc reste ATTACHE, en radians.

    Au-dela, le choc se detache et se courbe devant la rampe : il n'existe plus
    de solution oblique, et la recuperation s'effondre. C'est une limite
    physique, pas numerique — la meme famille que le blocage thermique de
    l'etape 7, et elle se refuse de la meme facon.

    Mesure a gamma = 1,4 : 12,11 deg a M = 1,5 ; 22,97 a M = 2 ; 34,07 a M = 3 ;
    38,55 a M = 4 ; et 45,58 deg a la limite M -> infini.
    """
    if mach <= 1.0:
        raise ThermoError(
            f"pas de choc oblique a M = {mach:g} : il faut M > 1"
        )
    # La fonction theta(beta) est unimodale entre l'angle de Mach et 90 deg.
    lo = float(np.arcsin(1.0 / mach))
    hi = np.pi / 2.0
    for _ in range(GOLDEN_STEPS):
        a = hi - INVERSE_GOLDEN * (hi - lo)
        b = lo + INVERSE_GOLDEN * (hi - lo)
        if _deflection(mach, a, gamma) < _deflection(mach, b, gamma):
            lo = a
        else:
            hi = b
    return _deflection(mach, 0.5 * (lo + hi), gamma)


def shock_angle(mach: float, deflection: float, gamma: float = 1.4) -> float:
    """L'angle « beta » du choc oblique faible, en radians.

    DEUX SOLUTIONS, ET ON PREND TOUJOURS LA FAIBLE. Pour une deviation donnee
    il existe un choc faible et un choc fort ; c'est le faible qui s'etablit
    dans une entree d'air, et le fort demande une contre-pression aval qu'une
    rampe libre n'impose pas. La bissection est bornee entre l'angle de Mach et
    l'angle de deviation maximale, ce qui exclut la branche forte par
    construction.
    """
    if deflection <= 0.0:
        raise ThermoError(
            f"deviation nulle ou negative : {np.degrees(deflection):g} deg"
        )
    maximum = max_deflection(mach, gamma)
    if deflection > maximum:
        raise ThermoError(
            f"choc detache : une rampe de {np.degrees(deflection):.2f} deg "
            f"depasse la deviation maximale de {np.degrees(maximum):.2f} deg "
            f"a M = {mach:.4f}. Le choc se courbe devant la rampe et la "
            f"recuperation s'effondre. Reduisez l'angle, ou repartissez-le sur "
            f"plusieurs rampes"
        )
    lo = float(np.arcsin(1.0 / mach))
    hi = _angle_of_max_deflection(mach, gamma)
    for _ in range(BETA_STEPS):
        mid = 0.5 * (lo + hi)
        if _deflection(mach, mid, gamma) < deflection:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _angle_of_max_deflection(mach: float, gamma: float) -> float:
    """Le « beta » qui maximise theta : la frontiere faible/fort."""
    lo = float(np.arcsin(1.0 / mach))
    hi = np.pi / 2.0
    for _ in range(GOLDEN_STEPS):
        a = hi - INVERSE_GOLDEN * (hi - lo)
        b = lo + INVERSE_GOLDEN * (hi - lo)
        if _deflection(mach, a, gamma) < _deflection(mach, b, gamma):
            lo = a
        else:
            hi = b
    return 0.5 * (lo + hi)


def oblique_shock(mach: float, deflection: float,
                  gamma: float = 1.4) -> ObliqueShock:
    """Le choc oblique complet, par la composante NORMALE.

    Tout choc oblique est un choc droit vu de biais : sa composante normale
    « M sin(beta) » subit exactement les relations de Rankine-Hugoniot, et la
    composante tangentielle est conservee. On reutilise donc « normal_shock »
    sans rien reecrire, ce qui garantit que les deux familles ne peuvent pas
    diverger.
    """
    beta = shock_angle(mach, deflection, gamma)
    normal = mach * np.sin(beta)
    droit = normal_shock(float(normal), gamma)
    return ObliqueShock(
        mach_upstream=float(mach), deflection=float(deflection),
        wave_angle=float(beta), normal_mach=float(normal),
        mach_downstream=float(droit.mach_downstream
                              / np.sin(beta - deflection)),
        total_pressure_ratio=float(droit.total_pressure_ratio),
        pressure_ratio=float(droit.pressure_ratio),
        temperature_ratio=float(droit.temperature_ratio),
    )


@dataclass(frozen=True)
class RampInlet:
    """Une entree a rampes : N chocs obliques, puis le choc droit terminal."""

    mach_upstream: float
    shocks: tuple[ObliqueShock, ...]
    terminal: NormalShock
    total_pressure_ratio: float     # le « pi_d » que le 0D reclame

    @property
    def deflections(self) -> tuple[float, ...]:
        return tuple(s.deflection for s in self.shocks)

    @property
    def total_turn(self) -> float:
        """La deviation totale imposee a l'ecoulement, en radians."""
        return float(sum(self.deflections))

    def describe(self) -> str:
        lignes = [f"  entree a {len(self.shocks)} rampe(s), "
                  f"M0 = {self.mach_upstream:.4f}"]
        lignes += [s.describe() for s in self.shocks]
        lignes.append(f"  choc droit terminal a "
                      f"M = {self.terminal.mach_upstream:.4f}   "
                      f"pi = {self.terminal.total_pressure_ratio:.4f}")
        lignes.append(f"  pi_d = {self.total_pressure_ratio:.4f}")
        return "\n".join(lignes)


def ramp_inlet(mach: float, deflections, gamma: float = 1.4) -> RampInlet:
    """La chaine de chocs, puis le choc droit qui ramene en subsonique.

    LE CHOC TERMINAL N'EST PAS UN DETAIL, C'EST LA MOITIE DE LA PERTE. Les
    rampes ne peuvent pas ralentir jusqu'au subsonique : il faut toujours finir
    par un choc droit, et son cout depend du Mach qu'on lui presente. Tout
    l'art d'une entree a rampes est de le lui presenter aussi bas que possible.
    """
    courant = float(mach)
    if courant <= 1.0:
        raise ThermoError(
            f"entree supersonique demandee a M = {courant:g} : il faut M > 1"
        )
    chocs = []
    recuperation = 1.0
    for index, angle in enumerate(deflections):
        try:
            choc = oblique_shock(courant, float(angle), gamma)
        except ThermoError as erreur:
            raise ThermoError(
                f"rampe {index + 1} sur {len(tuple(deflections))} : {erreur}"
            ) from erreur
        chocs.append(choc)
        recuperation *= choc.total_pressure_ratio
        courant = choc.mach_downstream
        if courant <= 1.0:
            raise ThermoError(
                f"la rampe {index + 1} ramene deja l'ecoulement a "
                f"M = {courant:.4f} : les rampes suivantes n'ont plus de choc "
                f"oblique a produire"
            )
    terminal = normal_shock(courant, gamma)
    return RampInlet(
        mach_upstream=float(mach), shocks=tuple(chocs), terminal=terminal,
        total_pressure_ratio=float(recuperation
                                   * terminal.total_pressure_ratio),
    )


def optimal_ramp_angles(mach: float, count: int,
                        gamma: float = 1.4) -> tuple[float, ...]:
    """Les angles qui maximisent la recuperation, par le critere d'Oswatitsch.

    LE CRITERE : a l'optimum, tous les chocs OBLIQUES ont la meme force,
    c'est-a-dire la meme composante normale de Mach. Ce n'est pas une hypothese
    imposee ici, c'est un resultat qu'une optimisation libre a Nelder-Mead
    retrouve a six chiffres — et le present code s'en sert pour ramener une
    optimisation a N parametres a UNE seule inconnue : cette force commune.

    Le choc terminal, lui, est PLUS FAIBLE que les obliques, et c'est mesure :
    1,4252 pour les trois obliques contre 1,3483 pour le droit, a M0 = 3 et
    trois rampes. L'enonce courant « tous les chocs de meme force » vaut pour
    les obliques seuls.
    """
    if count < 1:
        raise ThermoError(f"il faut au moins une rampe (recu {count})")
    if mach <= 1.0:
        raise ThermoError(f"entree supersonique : il faut M > 1 (recu {mach:g})")

    def angles_pour(force: float):
        """Les rampes qui donnent a chaque choc oblique cette force normale."""
        courant, sortie = float(mach), []
        for _ in range(count):
            if force >= courant:
                return None
            beta = float(np.arcsin(force / courant))
            theta = _deflection(courant, beta, gamma)
            if theta <= 0.0:
                return None
            choc = oblique_shock(courant, theta, gamma)
            sortie.append(theta)
            courant = choc.mach_downstream
            if courant <= 1.0:
                return None
        return tuple(sortie)

    def recuperation(force: float) -> float:
        angles = angles_pour(force)
        if angles is None:
            return -1.0
        return ramp_inlet(mach, angles, gamma).total_pressure_ratio

    # La force commune vit entre 1 (chocs infiniment faibles, mais alors le
    # choc terminal encaisse tout) et le Mach amont lui-meme.
    lo, hi = 1.0 + 1e-9, float(mach)
    for _ in range(GOLDEN_STEPS):
        a = hi - INVERSE_GOLDEN * (hi - lo)
        b = lo + INVERSE_GOLDEN * (hi - lo)
        if recuperation(a) < recuperation(b):
            lo = a
        else:
            hi = b
    angles = angles_pour(0.5 * (lo + hi))
    if angles is None:
        raise ThermoError(
            f"aucune configuration a {count} rampes ne tient a M = {mach:g}"
        )
    return angles
