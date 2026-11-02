"""Le statoreacteur complet : assembler le 0D, la geometrie et l'EDO.

CE QUE CE MODULE APPREND, ET QUE NI LE 0D NI LE 1D SEUL NE POUVAIT DIRE. Le 0D
choisit un debit et en deduit des sections. Le 1D integre un conduit a debit
donne. Les mettre bout a bout revele que l'un des deux se trompait de question :

    A GEOMETRIE FIXEE, LE DEBIT N'EST PAS UN PARAMETRE. C'est le col qui le
    fixe.

Un statoreacteur avale ce que son col laisse passer, ni plus ni moins. Le debit
est donc l'inconnue d'une equation — « A*(fin de chambre) = A_col » — et non
une donnee. Mesure sur le cas de reference : la geometrie dessinee pour
10,000 kg/s en avale 10,094. Le 0,94 % d'ecart n'est pas une imprecision, c'est
la somme de deux erreurs de comptage du 0D qui ne se compensent pas tout a
fait : il ajoutait la masse de carburant au col (+3,4 %) et supposait une perte
de chambre de 5 % la ou le 1D en subit 7,4 %.

TROIS MORCEAUX, ET UN SEUL EST INTEGRE. Le profil complet se lit de gauche a
droite, mais il n'est pas calcule de la meme facon partout :

  - le DIFFUSEUR est un saut. Le 0D le resume par « pi_d », et le 1D n'a rien a
    dire sur un systeme de chocs : c'est de la geometrie externe, hors du
    conduit. On applique le saut en x = 0, puis on trace l'ecoulement
    isentropique subsonique que la section impose.
  - la CHAMBRE est integree. C'est la seule zone ou l'EDO fait un travail que
    personne d'autre ne peut faire, parce que la chaleur y est distribuee.
  - la TUYERE est algebrique. Sans chaleur, la solution de l'EDO EST la relation
    de section — l'etape 6 l'a verifie a 1e-12 — et la relation de section, elle,
    n'a pas de singularite en M = 1. On l'utilise donc directement, ce qui
    traverse le col sans jamais approcher le 0/0.

CE DERNIER POINT EST LA REPONSE AU PASSAGE TRANSSONIQUE, et elle merite d'etre
dite franchement : on ne leve pas le 0/0, ON NE LE RENCONTRE PAS. Integrer
jusqu'au col demanderait une regle de L'Hopital et un depart supersonique
amorce a la main ; la relation « A/A* » donne le meme resultat en forme fermee,
sur les deux branches, et l'etape 6 a prouve que c'est le meme resultat. La
seule precaution est le conditionnement pres du col, deja mesure a l'etape 2 :
l'inversion y perd la moitie des chiffres, ce qui laisse M connu a 1e-8.

LE RACCORD SE VERIFIE, ET IL EST EXACT PAR CONSTRUCTION. Le debit solution est
precisement celui pour lequel « A*(fin de chambre) = A_col ». Le Mach que l'EDO
rend en fin de chambre et celui que « A(x)/A_col » donne au meme endroit sont
donc le meme nombre : mesure a 5,6e-17. Le raccord n'est pas ajuste, il tombe
juste.

SCIPY EST AUTORISE ICI. Ce module appartient a la couche 1D, comme
« solver1d ».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .atmosphere import state as atmosphere_state
from .config import CaseError, EngineCase, FlightCase
from .cycle0d import Cycle, run_cycle
from .duct import Duct, choked_area, from_cycle
from .solver1d import (NO_FRICTION, NO_MASS_ADDITION, ChokedFlow,
                       Friction, HeatRelease, MassAddition, Profile,
                       integrate_duct, no_heat, reconstruct)
from .thermo import mach_from_area_ratio

# DEUX PHASES POUR CHERCHER LE DEBIT, ET LA PREMIERE EXISTE A CAUSE D'UNE
# EXCEPTION. Au-dessus d'un certain debit la chambre bloque thermiquement et
# l'integration leve : la fonction dont on cherche le zero n'est pas definie
# partout, et « brentq » ne sait pas lire une exception. On bissecte donc a la
# main tant qu'une borne est bloquee — une exception n'a pas de signe, mais elle
# a un sens, « trop de debit » — puis on passe la main a « brentq » des que les
# deux bornes sont integrables. Mesure : 60 bissections coutaient 1,7 s pour
# 1e-12 ; la version en deux phases coute 0,6 s pour 1e-14.
SAFE_STEPS = 40            # plafond de la phase prudente
BRACKET_WIDTH = 0.05       # on passe a brentq sous cette largeur relative
COARSE_POINTS = 40         # sans effet sur le cout : solve_ivp domine (27 ms)

# En dessous de cette fraction du debit d'essai, on renonce a chercher.
FLOOR_FRACTION = 1e-3


@dataclass(frozen=True, eq=False)
class Engine:
    """Un point de fonctionnement complet, geometrie comprise."""

    cycle: Cycle
    duct: Duct
    mass_flow: float               # celui que le col impose
    profile: Profile               # entree du diffuseur -> sortie de tuyere
    chamber: Profile               # la seule partie integree, gardee a part
    thrust: float                  # N
    specific_thrust: float         # N.s/kg, masse de carburant comprise
    exit_pressure_ratio: float     # p9/p_ambiant
    ambient_pressure: float        # Pa

    @property
    def combustor_pressure_ratio(self) -> float:
        """Le « pi_b » reellement subi. C'est lui que l'etape 9 reinjecte."""
        return self.chamber.total_pressure_ratio

    @property
    def throat_mach(self) -> float:
        """Le Mach au col. Doit valoir 1, ou le col n'en est pas un."""
        i = int(np.argmin(np.abs(self.profile.x - self.duct.throat_x)))
        return float(self.profile.mach[i])

    def describe(self) -> str:
        lignes = [
            f"  debit impose par le col   {self.mass_flow:8.3f} kg/s",
            f"  Mach au col               {self.throat_mach:8.4f}",
            f"  Mach de sortie            {float(self.profile.mach[-1]):8.4f}",
            f"  pi_b subi                 {self.combustor_pressure_ratio:8.4f}"
            f"   (le 0D supposait {self.cycle.engine.pi_b:.4f})",
            f"  p9 / p_ambiant            {self.exit_pressure_ratio:8.4f}",
            f"  poussee                   {self.thrust:8.1f} N",
            f"  poussee specifique        {self.specific_thrust:8.1f} N.s/kg"
            f"   (le 0D annoncait {self.cycle.specific_thrust:.1f})",
        ]
        return "\n".join(lignes)


# -- le debit, qui est une inconnue -----------------------------------------

def _chamber_inlet_mach(mass_flow: float, duct: Duct, T02: float, p02: float,
                        gamma: float, R: float) -> float:
    """Le Mach a l'entree de chambre qu'impose la section, a debit donne."""
    critique = choked_area(mass_flow, T02, p02, gamma, R)
    zone = duct.zone("chamber")
    return float(mach_from_area_ratio(zone.area_start / critique, "subsonic",
                                      gamma))


def _integrate_chamber(mass_flow: float, duct: Duct, T02: float, p02: float,
                       T04: float, gamma: float, R: float, n_points: int,
                       friction: Friction = NO_FRICTION,
                       fuel_air_ratio: float = 0.0) -> Profile:
    """La chambre seule, a debit d'ENTREE donne. Peut lever « ChokedFlow ».

    Le carburant est injecte sur la meme zone que la chaleur, et le debit de
    sortie vaut donc « mdot (1+f) ». Ce n'est plus une correction posee apres
    coup : c'est le quatrieme terme source de l'EDO.
    """
    zone = duct.zone("chamber")
    mach_in = _chamber_inlet_mach(mass_flow, duct, T02, p02, gamma, R)
    heat = HeatRelease(T02, T04, zone.x_start, zone.x_end)
    mass = (MassAddition(fuel_air_ratio, zone.x_start, zone.x_end)
            if fuel_air_ratio > 0.0 else NO_MASS_ADDITION)
    return integrate_duct(duct, heat, mach_in, mass_flow, gamma, R,
                          n_points=n_points, x_start=zone.x_start,
                          x_end=zone.x_end, friction=friction, mass=mass)


def choking_mass_flow(duct: Duct, T02: float, p02: float, T04: float,
                      gamma: float = 1.4, R: float = 287.05,
                      steps: int = SAFE_STEPS,
                      friction: Friction = NO_FRICTION,
                      fuel_air_ratio: float = 0.0) -> float:
    """Le debit pour lequel le col est exactement sonique.

    RESIDU : « A*(fin de chambre) - A_col ». Il est negatif quand le col est
    trop grand — l'ecoulement passe sans bloquer, et la tuyere reste subsonique,
    donc sans poussee — et positif quand le col est trop petit, ce qui n'a pas
    de solution stationnaire. Entre les deux, un seul debit.

    LA FONCTION N'EST PAS DEFINIE PARTOUT. Au-dela d'un certain debit, le Mach
    d'entree de chambre monte assez pour que la chaleur bloque l'ecoulement et
    « integrate_duct » leve. Ce cas est traite comme un residu positif : trop de
    debit, il faut descendre. C'est pour cela que la bissection est ecrite ici
    plutot que deleguee a « brentq », qui ne sait pas lire une exception.
    """
    plafond = duct.throat_area / choked_area(1.0 + fuel_air_ratio, T04, p02,
                                             gamma, R)
    # Ce plafond suppose « pi_b = 1 » : il majore donc toujours la solution.

    def residu(debit: float) -> float:
        profil = _integrate_chamber(debit, duct, T02, p02, T04, gamma, R,
                                    COARSE_POINTS, friction, fuel_air_ratio)
        # LE DEBIT QUI PASSE LE COL EST CELUI DE SORTIE DE CHAMBRE, carburant
        # compris. C'est exactement ce que le « (1+f) » pose a la main de
        # l'etape 8 essayait d'exprimer, et qui est maintenant dans l'EDO.
        return (choked_area(profil.exit_mass_flow, T04,
                            float(profil.total_pressure[-1]), gamma, R)
                - duct.throat_area)

    lo = plafond * 0.5
    while True:
        if lo < plafond * FLOOR_FRACTION:
            raise CaseError(
                f"aucun debit ne bloque ce col : meme a {lo:g} kg/s "
                f"l'ecoulement bloque thermiquement avant la fin de chambre. "
                f"Le col est trop petit pour cette temperature de fin de "
                f"combustion ({T04:g} K)"
            )
        try:
            if residu(lo) < 0.0:
                break
        except ChokedFlow:
            pass
        lo *= 0.5

    # Phase 1 : la seule qui sache lire un blocage.
    hi, hi_bloque = plafond, True
    for _ in range(steps):
        if not hi_bloque and (hi - lo) < BRACKET_WIDTH * hi:
            break
        mid = 0.5 * (lo + hi)
        try:
            trop_grand, bloque = residu(mid) > 0.0, False
        except ChokedFlow:
            trop_grand, bloque = True, True
        if trop_grand:
            hi, hi_bloque = mid, bloque
        else:
            lo = mid
    if hi_bloque:
        raise CaseError(
            f"la recherche du debit n'a pas trouve de borne haute integrable "
            f"en {steps} pas : la chambre bloque des {hi:g} kg/s alors que le "
            f"col en demanderait davantage. Baissez T04 ({T04:g} K) ou "
            f"agrandissez la chambre"
        )

    # Phase 2 : les deux bornes sont integrables, brentq peut finir.
    return float(brentq(residu, lo, hi, xtol=1e-12, rtol=1e-14))


# -- les morceaux algebriques ------------------------------------------------

def isentropic_profile(duct: Duct, x, T0: float, critical_area: float,
                       branch: str, mass_flow: float, gamma: float,
                       R: float) -> Profile:
    """Un troncon sans chaleur, resolu par « A/A* » plutot qu'integre.

    Le meme « reconstruct » que le solveur : seul le chemin vers M change.
    """
    x = np.asarray(x, dtype=float)
    ratio = np.asarray(duct.area(x), dtype=float) / critical_area
    if np.any(ratio < 1.0 - 1e-12):
        raise CaseError(
            f"section plus petite que la section critique "
            f"(A/A* = {float(np.min(ratio)):.6f} < 1) : le debit ne passe pas"
        )
    mach = mach_from_area_ratio(np.maximum(ratio, 1.0), branch, gamma)
    return reconstruct(x, np.asarray(mach, dtype=float) ** 2, duct,
                       no_heat(T0, duct.length), mass_flow, gamma, R)


def _concatenate(profiles: tuple[Profile, ...]) -> Profile:
    """Bout a bout, sans repeter les points de jonction."""
    morceaux: list[Profile] = []
    for i, p in enumerate(profiles):
        decoupe = slice(None) if i == 0 else slice(1, None)
        morceaux.append((p, decoupe))
    # « mass_flow » EST DANS LA LISTE, ET IL A FALLU L'Y METTRE. Tant que le
    # debit etait un scalaire constant, le recopier depuis le premier troncon
    # etait correct. Depuis qu'il varie, l'oublier donnait un tableau de la
    # longueur du diffuseur pour un profil de 596 points — et une poussee de
    # 683 N.s/kg au lieu de 738, c'est-a-dire exactement la valeur SANS le
    # carburant. Le bogue reproduisait le defaut qu'on venait de corriger.
    champs = ("x", "mach", "temperature", "pressure", "velocity", "density",
              "total_temperature", "total_pressure", "area", "mass_flow")
    valeurs = {nom: np.concatenate([getattr(p, nom)[d] for p, d in morceaux])
               for nom in champs}
    return Profile(gamma=profiles[0].gamma, **valeurs)


# -- le moteur complet -------------------------------------------------------

def run_engine(flight: FlightCase | None = None,
               engine: EngineCase | None = None,
               duct: Duct | None = None, n_points: int = 200,
               friction: Friction = NO_FRICTION) -> Engine:
    """Le profil d'un bout a l'autre, et la poussee qu'il donne.

    Le debit n'est pas un argument : il est calcule. Si l'on veut imposer une
    taille, on la met dans la geometrie — c'est elle qui commande.
    """
    flight = FlightCase() if flight is None else flight
    engine = EngineCase() if engine is None else engine
    cycle = run_cycle(flight, engine)
    if duct is None:
        duct = from_cycle(cycle, mass_flow=10.0)

    gamma, R = flight.gamma, flight.R
    T02, p02 = cycle.station("2").T0, cycle.station("2").p0
    T04 = cycle.station("4").T0

    f = cycle.fuel_air_ratio
    debit = choking_mass_flow(duct, T02, p02, T04, gamma, R,
                              friction=friction, fuel_air_ratio=f)
    chambre = _integrate_chamber(debit, duct, T02, p02, T04, gamma, R,
                                 n_points, friction, f)
    debit_sortie = chambre.exit_mass_flow

    zone = duct.zone("chamber")
    critique_amont = choked_area(debit, T02, p02, gamma, R)
    diffuseur = isentropic_profile(
        duct, np.linspace(0.0, zone.x_start, n_points // 2), T02,
        critique_amont, "subsonic", debit, gamma, R)

    # LE COL EST LE SEUL POINT COMMUN AUX DEUX BRANCHES. On le met dans le
    # convergent et on demarre le divergent juste apres, sinon « A/A* = 1 »
    # serait evalue deux fois et rendrait M = 1 deux fois de suite, ce qui se
    # verrait comme un palier sur le trace.
    col = duct.throat_x
    convergent = isentropic_profile(
        duct, np.linspace(zone.x_end, col, n_points // 2), T04,
        duct.throat_area, "subsonic", debit_sortie, gamma, R)
    divergent = isentropic_profile(
        duct, np.linspace(col, duct.length, n_points)[1:], T04,
        duct.throat_area, "supersonic", debit_sortie, gamma, R)

    profil = _concatenate((diffuseur, chambre, convergent, divergent))

    ambiante = atmosphere_state(flight.altitude)["p"]
    A9 = float(profil.area[-1])
    u9 = float(profil.velocity[-1])
    p9 = float(profil.pressure[-1])
    u0 = cycle.velocity_flight
    # LE « (1+f) » N'EST PLUS POSE A LA MAIN. Le debit de sortie sort de l'EDO
    # elle-meme, qui porte desormais un quatrieme terme source. Les termes de
    # quantite de mouvement et de pression parlent donc du meme debit, ce qui
    # n'etait pas le cas et se voyait a l'etape 15 : la perte d'adaptation y
    # devenait negative, ce qui est impossible.
    poussee = (float(profil.mass_flow[-1]) * u9 - debit * u0
               + (p9 - ambiante) * A9)

    return Engine(
        cycle=cycle, duct=duct, mass_flow=debit, profile=profil,
        chamber=chambre, thrust=poussee, specific_thrust=poussee / debit,
        exit_pressure_ratio=p9 / ambiante, ambient_pressure=ambiante,
    )


# -- la boucle 0D <-> 1D -----------------------------------------------------

# LE POINT FIXE EST LINEAIRE, ET C'EST CE QUI DECIDE DE LA METHODE. L'iteration
# nue « pi <- g(pi) » converge geometriquement de rapport 0,172369 — mesure sur
# huit tours, constant a 1e-6 pres des le troisieme. Il faut alors SEPT tours
# pour descendre a 1e-6, la ou le plan en annoncait moins de cinq. La secante
# sur « h(pi) = g(pi) - pi » atteint le point fixe EXACTEMENT en six appels, et
# depuis quatre departs differents (0,80 a 0,99) elle rend les douze memes
# chiffres. C'est elle qu'on garde.
FIXED_POINT_TOL = 1e-12
MAX_ITERATIONS = 20

# Bornes du pas de secante. Un point fixe physique vit entre 0,3 et 1 ; sortir
# de la ne veut rien dire, et la secante peut proposer n'importe quoi si les
# deux derniers residus se ressemblent trop.
PI_FLOOR, PI_CEILING = 0.30, 1.0


class DesignCollapse(CaseError):
    """Le point de fonctionnement s'effondre QUAND ON REINJECTE le pi_b reel.

    A distinguer du refus immediat : ici le premier dessin passe, et c'est la
    contre-reaction qui le detruit. Le 0D seul aurait rendu un moteur d'allure
    normale.
    """

    def __init__(self, message: str, iteration: int, pressure_ratio: float,
                 cause: str) -> None:
        super().__init__(message)
        self.iteration = iteration
        self.pressure_ratio = pressure_ratio
        self.cause = cause


@dataclass(frozen=True, eq=False)
class Convergence:
    """Le point fixe, et le chemin qui y mene."""

    engine: Engine
    history: tuple[float, ...]     # les pi_b successifs, depart compris
    calls: int                     # nombre d'evaluations du 1D
    residual: float                # |g(pi) - pi| au point d'arrivee

    @property
    def combustor_pressure_ratio(self) -> float:
        return self.history[-1]

    @property
    def assumed_pressure_ratio(self) -> float:
        return self.history[0]

    @property
    def correction(self) -> float:
        """L'ecart relatif entre le pi_b suppose et le pi_b subi."""
        return self.combustor_pressure_ratio / self.assumed_pressure_ratio - 1.0

    def choking_margin(self) -> float:
        """Combien de kelvins separent le point converge du blocage.

        Mesure sur le cas de reference : 1 275 K de marge. Le Mach REEL
        d'entree de chambre est ce qui compte, et il n'est jamais celui vise —
        0,2716 quand on demande 0,30, ce qui deplace la limite de 1 749 K a
        2 055 K.
        """
        from . import rayleigh
        T02 = self.engine.cycle.station("2").T0
        mach = _chamber_inlet_mach(self.engine.mass_flow, self.engine.duct,
                                   T02, self.engine.cycle.station("2").p0,
                                   self.engine.cycle.flight.gamma,
                                   self.engine.cycle.flight.R)
        limite = float(rayleigh.critical_total_temperature(mach, T02))
        return limite - self.engine.cycle.station("4").T0

    def describe(self) -> str:
        return "\n".join([
            f"  pi_b suppose au depart    {self.assumed_pressure_ratio:8.4f}",
            f"  pi_b au point fixe        "
            f"{self.combustor_pressure_ratio:8.4f}   "
            f"({100 * self.correction:+.2f} %)",
            f"  appels au modele 1D       {self.calls:8d}",
            f"  residu |g(pi) - pi|       {self.residual:8.1e}",
            f"  marge avant blocage       {self.choking_margin():8.0f} K",
        ])


def converge(flight: FlightCase | None = None,
             engine: EngineCase | None = None, duct: Duct | None = None,
             tol: float = FIXED_POINT_TOL,
             max_iterations: int = MAX_ITERATIONS,
             friction: Friction = NO_FRICTION) -> Convergence:
    """Alterne 0D et 1D jusqu'a ce que « pi_b » ne bouge plus.

    LE 0D SUPPOSE UNE PERTE DE CHAMBRE, LE 1D LA CALCULE. On reinjecte la
    calculee dans le suppose et on recommence. Le point fixe est le moteur dont
    les hypotheses de dimensionnement coincident avec son propre comportement.

    LA GEOMETRIE EST REDESSINEE A CHAQUE TOUR, et c'est le seul chemin par
    lequel la boucle boucle. A geometrie FIGEE, « pi_b » sort du 1D sans
    dependre du « pi_b » d'entree — les donnees de la chambre sont « T02 »,
    « p02 » et « T04 », qu'aucun des deux ne touche — et la « boucle » se ferme
    en un tour, exactement. Passer un « duct » explicite donne donc ce cas
    degenere, ce qui est parfois ce qu'on veut : il repond a « que fait CE
    conduit », quand la boucle repond a « quel conduit est coherent ».
    """
    flight = FlightCase() if flight is None else flight
    engine = EngineCase() if engine is None else engine

    compteur = [0]

    def g(pi: float) -> Engine:
        compteur[0] += 1
        return run_engine(flight, engine.replace(pi_b=float(pi)), duct=duct,
                          friction=friction)

    def garde(pi: float, tour: int) -> Engine:
        try:
            return g(pi)
        except ChokedFlow as erreur:
            raise DesignCollapse(
                f"point de fonctionnement bloque au tour {tour} : en "
                f"reinjectant pi_b = {pi:.4f}, la chambre se bloque "
                f"thermiquement a x = {erreur.x:.4f} m. {erreur}",
                iteration=tour, pressure_ratio=pi, cause=str(erreur)) from erreur
        except CaseError as erreur:
            raise DesignCollapse(
                f"le dessin s'effondre au tour {tour} : le premier passage "
                f"avait accepte ce point, mais en reinjectant "
                f"pi_b = {pi:.4f} au lieu de "
                f"{engine.pi_b:.4f} il n'existe plus de moteur. "
                f"Le 0D seul aurait rendu un resultat d'allure normale. "
                f"Cause : {erreur}",
                iteration=tour, pressure_ratio=pi, cause=str(erreur)) from erreur

    # Amorcage : un tour d'iteration nue fournit le second point de la secante.
    p0 = float(engine.pi_b)
    moteur = garde(p0, 0)
    p1 = moteur.combustor_pressure_ratio
    historique = [p0, p1]
    if abs(p1 - p0) <= tol * abs(p0):
        return Convergence(moteur, tuple(historique), compteur[0],
                           abs(p1 - p0))

    moteur = garde(p1, 1)
    h0, h1 = p1 - p0, moteur.combustor_pressure_ratio - p1
    a, b = p0, p1
    historique.append(moteur.combustor_pressure_ratio)

    for tour in range(2, max_iterations):
        if abs(h1) <= tol * abs(b):
            break
        if abs(h1 - h0) < 1e-16:
            break
        pas = b - h1 * (b - a) / (h1 - h0)
        suivant = min(max(pas, PI_FLOOR), PI_CEILING)
        moteur = garde(suivant, tour)
        h_suivant = moteur.combustor_pressure_ratio - suivant
        historique.append(moteur.combustor_pressure_ratio)
        a, b, h0, h1 = b, suivant, h1, h_suivant
    else:
        raise CaseError(
            f"la boucle 0D <-> 1D n'a pas converge en {max_iterations} tours "
            f"(residu {abs(h1):.2e}, seuil {tol:g}). Dernier pi_b : {b:.6f}"
        )

    # Le moteur rendu est celui du DERNIER pi_b evalue, donc coherent avec lui.
    return Convergence(moteur, tuple(historique), compteur[0], abs(h1))


# -- l'adaptation de la tuyere, hors du point de dessin ----------------------
#
# ETAPE 15. Un statoreacteur vole sur un DOMAINE, pas en un point. Sa geometrie
# est figee : le rapport « A9/A_col » est coule dans le metal, et c'est lui qui
# fixe le Mach de sortie. La pression de sortie, elle, ne s'ajuste pas a
# l'ambiante — d'ou un terme « (p9 - p_amb) A9 » dans la poussee, positif si la
# tuyere est SOUS-detendue et negatif si elle est SUR-detendue.

# LE CRITERE DE DECOLLEMENT, ET IL MARQUE LA FIN DU DOMAINE DE VALIDITE. En
# dessous de ce rapport, la couche limite se decolle des parois du divergent et
# l'ecoulement cesse d'etre celui que le modele calcule. Le decollement est un
# effet visqueux 2D, explicitement hors perimetre : le modele ne peut pas le
# calculer, mais il peut dire ou il cesse d'etre valable, et c'est le minimum.
# La valeur 0,4 est le critere de Summerfield, grossier mais universellement
# cite ; les correlations plus fines (Schmucker, Kalt-Bendall) dependent du
# rapport de section et ne changeraient pas la conclusion.
SEPARATION_RATIO = 0.4


@dataclass(frozen=True)
class Adaptation:
    """Ce que coute une tuyere qui n'est pas adaptee au point de vol."""

    engine: Engine
    exit_pressure_ratio: float     # p9 / p_ambiant
    pressure_term: float           # (p9 - p_amb) A9, en N
    adapted_thrust: float          # F/mdot si A9 etait le bon
    area_ratio_to_adapted: float   # A9 reel / A9 adapte
    exit_impulse: float            # u9 + (p9 - p_amb) A9/mdot, en m/s
    adapted_impulse: float         # la meilleure possible, en m/s

    @property
    def loss(self) -> float:
        """La perte relative de POUSSEE SPECIFIQUE, par rapport a l'adapte.

        ELLE EST POSITIVE PARTOUT DEPUIS QUE L'EDO PORTE LE CARBURANT. Tant que
        le « (1+f) » etait pose a la main, le terme de pression etait calcule
        avec le debit d'AIR et celui de quantite de mouvement avec le debit de
        SORTIE : les deux differaient de 3,5 %, ce qui suffisait a faire
        paraitre une tuyere non adaptee meilleure qu'une adaptee — de 0,019 %.
        Le quatrieme terme source a ferme cet ecart, et un test verifie que le
        minimum est maintenant nul et non negatif.
        """
        return 1.0 - self.engine.specific_thrust / self.adapted_thrust

    @property
    def impulse_loss(self) -> float:
        """LA MEME PERTE, SUR L'IMPULSION DE SORTIE PAR KILOGRAMME DE GAZ.

        « u9 + (p9 - p_amb) A9/mdot_sortie », rapportee a la meilleure possible.
        C'est la quantite que l'optimalite d'une tuyere adaptee gouverne
        directement, et elle est donc l'autre facon de poser la meme question —
        celle qui ne fait pas intervenir la vitesse de vol.
        """
        return 1.0 - self.exit_impulse / self.adapted_impulse

    @property
    def under_expanded(self) -> bool:
        """p9 > p_amb : la detente n'est pas finie a la sortie."""
        return self.exit_pressure_ratio > 1.0

    @property
    def separated(self) -> bool:
        """Au-dela de cette sur-detente, l'ecoulement decolle et le profil
        calcule ne decrit plus rien."""
        return self.exit_pressure_ratio < SEPARATION_RATIO

    def describe(self) -> str:
        etat = ("sous-detendue" if self.under_expanded else "sur-detendue")
        lignes = [
            f"  p9 / p ambiant            "
            f"{self.exit_pressure_ratio:8.4f}   ({etat})",
            f"  terme de pression         {self.pressure_term:8.1f} N   "
            f"({100 * self.pressure_term / self.engine.thrust:+.2f} % de F)",
            f"  poussee reelle            "
            f"{self.engine.specific_thrust:8.1f} N.s/kg",
            f"  poussee si adaptee        {self.adapted_thrust:8.1f} N.s/kg",
            f"  perte d'adaptation        {100 * self.loss:8.2f} %",
            f"  sur l'impulsion de sortie {100 * self.impulse_loss:8.4f} %"
            f"   (la seule garantie positive)",
        ]
        if self.separated:
            lignes.append(
                f"  ATTENTION : p9/p_amb = {self.exit_pressure_ratio:.4f} est "
                f"sous le seuil de decollement ({SEPARATION_RATIO:g}). "
                f"L'ecoulement se decolle du divergent, ce que ce modele ne "
                f"sait pas calculer — les nombres ci-dessus decrivent un "
                f"ecoulement qui n'existe pas")
        return "\n".join(lignes)


def adaptation(engine: Engine) -> Adaptation:
    """Compare la tuyere reelle a celle qui serait adaptee au meme point.

    La tuyere ADAPTEE est celle dont le divergent detendrait exactement jusqu'a
    l'ambiante, a col inchange. Son Mach de sortie se lit sur le rapport de
    pression totale, sa section sur « A/A* », et sa poussee n'a plus de terme
    de pression du tout. La difference est la PERTE D'ADAPTATION, et elle est
    nulle par construction au point de dessin — ce qu'un test verifie.
    """
    from .thermo import area_ratio, mach_from_total_pressure_ratio

    cycle, conduit, profil = engine.cycle, engine.duct, engine.profile
    gamma, R = cycle.flight.gamma, cycle.flight.R
    ambiante = engine.ambient_pressure
    section = float(profil.area[-1])
    terme = (float(profil.pressure[-1]) - ambiante) * section

    p09 = float(profil.total_pressure[-1])
    mach_adapte = float(mach_from_total_pressure_ratio(p09 / ambiante, gamma))
    section_adaptee = conduit.throat_area * float(area_ratio(mach_adapte,
                                                             gamma))
    T04 = cycle.station("4").T0
    T9 = T04 / (1.0 + 0.5 * (gamma - 1.0) * mach_adapte ** 2)
    vitesse = mach_adapte * float(np.sqrt(gamma * R * T9))
    poussee_adaptee = ((1.0 + cycle.fuel_air_ratio) * vitesse
                       - cycle.velocity_flight)

    return Adaptation(
        engine=engine, exit_pressure_ratio=engine.exit_pressure_ratio,
        pressure_term=terme, adapted_thrust=poussee_adaptee,
        area_ratio_to_adapted=section / section_adaptee,
        exit_impulse=(float(profil.velocity[-1])
                      + terme / float(profil.mass_flow[-1])),
        adapted_impulse=vitesse,
    )


def flight_envelope(duct: Duct, machs=None, altitudes=None,
                    engine: EngineCase | None = None):
    """Balaye le domaine de vol a GEOMETRIE FIGEE, et rend les adaptations.

    Rend deux listes : l'une a altitude fixee en balayant le Mach, l'autre a
    Mach fixe en balayant l'altitude. Les points refuses — geometrie
    impossible, blocage — sont ecartes et signales, comme dans le balayage 0D
    de l'etape 4 : savoir OU le domaine s'arrete est une information.
    """
    engine = EngineCase() if engine is None else engine
    machs = (2.0, 2.5, 3.0, 3.5, 4.0) if machs is None else machs
    altitudes = ((0.0, 5000.0, 11000.0, 15000.0, 20000.0, 25000.0)
                 if altitudes is None else altitudes)

    par_mach, par_altitude, refuses = [], [], []
    for mach in machs:
        try:
            moteur = run_engine(FlightCase(mach=float(mach), altitude=11000.0),
                                engine, duct=duct)
            par_mach.append((float(mach), adaptation(moteur)))
        except CaseError as erreur:
            refuses.append((f"M0 = {float(mach):g}", str(erreur)))
    for altitude in altitudes:
        try:
            moteur = run_engine(
                FlightCase(mach=3.0, altitude=float(altitude)), engine,
                duct=duct)
            par_altitude.append((float(altitude), adaptation(moteur)))
        except CaseError as erreur:
            refuses.append((f"z = {float(altitude):g} m", str(erreur)))
    return par_mach, par_altitude, tuple(refuses)
