"""La geometrie du conduit : A(x), dA/dx, et le dimensionnement depuis le 0D.

POURQUOI DES RACCORDS LISSES ET PAS DES SEGMENTS DROITS. « dA/dx » entre
DIRECTEMENT dans l'equation du solveur 1D, multipliee par un facteur qui
explose pres de M = 1. Une section decrite par segments aurait une derivee en
escalier, donc des sauts de « dM^2/dx » a chaque jonction, que l'integrateur a
pas adaptatif paierait en pas rejetes — et qui n'ont aucune raison physique.

Chaque zone suit donc un « smoothstep » cubique, « h(s) = 3s^2 - 2s^3 », qui
vaut 0 en s = 0 et 1 en s = 1 avec une derivee NULLE aux deux bouts. La
consequence est plus forte que la continuite demandee : « dA/dx » s'annule a
toutes les jonctions, donc le raccord est C1 partout sans qu'on ait a imposer
quoi que ce soit, et la chambre a section constante s'y raccorde sans rupture.

LE COL EST UN MINIMUM, DONC « dA/dx » Y EST NUL — ET C'EST LA SECONDE
SINGULARITE DU PROJET. L'equation du 1D s'ecrit avec un « 1 - M^2 » au
denominateur ; au col, le numerateur porte « dA/A » qui s'annule aussi. Le
passage transsonique est donc un 0/0, dont la levee est le vrai travail de
l'etape 6. Ce module ne le resout pas : il fournit la geometrie et signale ou
elle pose le probleme.

LE DIMENSIONNEMENT NE PASSE JAMAIS PAR L'INVERSION DE A/A*, ET C'EST UNE LECON
DE L'ETAPE 2. Pres du col, « A/A* » est plate — une erreur eps sur le rapport
donne sqrt(eps) sur le Mach. On dimensionne donc dans le sens ou l'information
circule : le col se deduit du DEBIT BLOQUE, qui est analytique, et la section de
sortie du Mach de sortie, lui aussi connu. L'inversion ne sert qu'a lire le Mach
de chambre, ou A/A* vaut 2 a 3 et ou elle est parfaitement conditionnee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseError
from .thermo import (area_ratio, mach_from_area_ratio, total_pressure_ratio,
                     total_temperature_ratio)

# Longueurs par defaut, en metres. Proportions d'un statoreacteur de croisiere :
# un diffuseur long, une chambre plus longue encore — c'est elle qui doit
# laisser le temps de bruler — et une tuyere courte.
DEFAULT_LENGTHS = dict(diffuser=0.40, chamber=0.60, converging=0.15,
                       diverging=0.35)


def mass_flow_parameter(gamma: float, R: float) -> float:
    """Le facteur du debit bloque : « mdot sqrt(T0) / (A* p0) ».

    Analytique, et c'est ce qui permet de dimensionner le col sans jamais
    inverser A/A* :

        Gamma = sqrt(gamma/R) ((gamma+1)/2)^(-(gamma+1)/(2(gamma-1)))
    """
    exponent = -(gamma + 1.0) / (2.0 * (gamma - 1.0))
    return float(np.sqrt(gamma / R) * ((gamma + 1.0) / 2.0) ** exponent)


def choked_area(mass_flow: float, T0: float, p0: float, gamma: float,
                R: float) -> float:
    """La section qui bloque exactement ce debit, pour ces conditions totales."""
    if mass_flow <= 0.0:
        raise CaseError(f"debit non positif : {mass_flow:g} kg/s")
    return mass_flow * np.sqrt(T0) / (p0 * mass_flow_parameter(gamma, R))


def area_for_mach(mass_flow: float, mach: float, T0: float, p0: float,
                  gamma: float, R: float) -> float:
    """La section qui, pour ce debit et ces conditions totales, donne ce Mach.

    Sens du calcul : du Mach VERS la section. L'autre sens existe aussi, mais
    il passe par l'inversion de A/A*, mal conditionnee pres du col.
    """
    if not 0.0 < mach:
        raise CaseError(f"le Mach doit etre positif (recu {mach:g})")
    return (choked_area(mass_flow, T0, p0, gamma, R)
            * float(area_ratio(mach, gamma)))


@dataclass(frozen=True)
class Zone:
    """Un troncon a section monotone, raccorde par un cubique."""

    name: str
    x_start: float
    x_end: float
    area_start: float
    area_end: float

    @property
    def length(self) -> float:
        return self.x_end - self.x_start

    def contains(self, x: float) -> bool:
        return self.x_start <= x <= self.x_end


@dataclass(frozen=True, eq=False)
class Duct:
    """Le conduit complet : quatre zones, et une section C1 partout."""

    zones: tuple[Zone, ...]

    def __post_init__(self) -> None:
        if not self.zones:
            raise CaseError("un conduit a besoin d'au moins une zone")
        for amont, aval in zip(self.zones, self.zones[1:]):
            if abs(amont.x_end - aval.x_start) > 1e-12:
                raise CaseError(
                    f"les zones {amont.name!r} et {aval.name!r} ne se touchent "
                    f"pas ({amont.x_end:g} contre {aval.x_start:g})"
                )
            if abs(amont.area_end - aval.area_start) > 1e-12:
                raise CaseError(
                    f"saut de section entre {amont.name!r} et {aval.name!r} : "
                    f"{amont.area_end:g} contre {aval.area_start:g}"
                )
        for zone in self.zones:
            if zone.length <= 0.0:
                raise CaseError(f"zone {zone.name!r} de longueur nulle")
            if min(zone.area_start, zone.area_end) <= 0.0:
                raise CaseError(f"zone {zone.name!r} a une section non positive")

    # -- geometrie ----------------------------------------------------------

    @property
    def length(self) -> float:
        return self.zones[-1].x_end

    @property
    def stations(self) -> tuple[float, ...]:
        """Les abscisses des jonctions, extremites comprises."""
        return (self.zones[0].x_start,) + tuple(z.x_end for z in self.zones)

    def zone(self, name: str) -> Zone:
        """La zone portant ce nom. Sert a borner une integration sur la chambre."""
        for zone in self.zones:
            if zone.name == name:
                return zone
        connues = ", ".join(z.name for z in self.zones)
        raise CaseError(f"zone inconnue : {name!r} (au choix : {connues})")

    def zone_of(self, x: float) -> Zone:
        for zone in self.zones:
            if zone.contains(x):
                return zone
        raise CaseError(
            f"abscisse {x:g} hors du conduit [0, {self.length:g}]"
        )

    def _local(self, x):
        """Pour chaque x : (indice de zone, coordonnee locale s dans [0, 1])."""
        x = np.asarray(x, dtype=float)
        bornes = np.array([zone.x_end for zone in self.zones])
        indices = np.clip(np.searchsorted(bornes, x, side="left"), 0,
                          len(self.zones) - 1)
        debuts = np.array([zone.x_start for zone in self.zones])[indices]
        longueurs = np.array([zone.length for zone in self.zones])[indices]
        return indices, (x - debuts) / longueurs

    def area(self, x):
        """A(x), par raccord cubique « 3s^2 - 2s^3 » dans chaque zone."""
        self._check_range(x)
        indices, s = self._local(x)
        depart = np.array([zone.area_start for zone in self.zones])[indices]
        arrivee = np.array([zone.area_end for zone in self.zones])[indices]
        return depart + (arrivee - depart) * (3.0 * s ** 2 - 2.0 * s ** 3)

    def area_derivative(self, x):
        """dA/dx, analytique. Nulle a TOUTES les jonctions, col compris."""
        self._check_range(x)
        indices, s = self._local(x)
        depart = np.array([zone.area_start for zone in self.zones])[indices]
        arrivee = np.array([zone.area_end for zone in self.zones])[indices]
        longueurs = np.array([zone.length for zone in self.zones])[indices]
        return (arrivee - depart) * (6.0 * s - 6.0 * s ** 2) / longueurs

    def _check_range(self, x) -> None:
        x = np.asarray(x, dtype=float)
        if np.any(x < -1e-12) or np.any(x > self.length + 1e-12):
            raise CaseError(
                f"abscisse hors du conduit [0, {self.length:g}] "
                f"(recu de {float(np.min(x)):g} a {float(np.max(x)):g})"
            )

    # -- points remarquables ------------------------------------------------

    @property
    def throat_x(self) -> float:
        """L'abscisse de section minimale. Le col, s'il y en a un."""
        bornes = np.array(self.stations)
        sections = self.area(bornes)
        return float(bornes[int(np.argmin(sections))])

    @property
    def throat_area(self) -> float:
        return float(self.area(self.throat_x))

    def describe(self) -> str:
        lignes = [f"  conduit de {self.length:.3f} m, "
                  f"{len(self.zones)} zones, col a x = {self.throat_x:.3f} m"]
        for zone in self.zones:
            lignes.append(f"    {zone.name:<12} x [{zone.x_start:.3f} ; "
                          f"{zone.x_end:.3f}]   A [{zone.area_start * 1e4:.1f} ; "
                          f"{zone.area_end * 1e4:.1f}] cm2")
        return "\n".join(lignes)


# -- le pont entre les deux modeles -----------------------------------------

def from_cycle(cycle, mass_flow: float = 10.0, lengths: dict | None = None,
               mach_combustor: float | None = None) -> Duct:
    """Dimensionne le conduit depuis un resultat 0D.

    Quatre sections, et chacune se deduit d'une grandeur que le 0D connait :

      - la CHAMBRE, du Mach qu'on veut y voir (« M_combustor » du cas moteur) et
        des conditions totales de la station 2 ;
      - le COL, du debit bloque aux conditions totales de la station 4 —
        analytique, donc sans inversion ;
      - la SORTIE, du Mach de sortie que le cycle a calcule ;
      - l'ENTREE, du debit capte dans l'ecoulement amont non perturbe.

    Le debit est un parametre : le 0D raisonne par unite de debit, la geometrie
    a besoin d'une echelle, et c'est le seul endroit ou elle entre.
    """
    flight, engine = cycle.flight, cycle.engine
    gamma, R = flight.gamma, flight.R
    mach_chamber = engine.M_combustor if mach_combustor is None else mach_combustor
    if not 0.0 < mach_chamber < 1.0:
        raise CaseError(
            f"Mach de chambre {mach_chamber:g} hors de ]0, 1[ : une chambre "
            f"supersonique est un autre moteur"
        )
    longueurs = dict(DEFAULT_LENGTHS if lengths is None else lengths)
    manquantes = set(DEFAULT_LENGTHS) - set(longueurs)
    if manquantes:
        raise CaseError(f"longueurs manquantes : {', '.join(sorted(manquantes))}")

    station2, station4 = cycle.station("2"), cycle.station("4")
    debit_chaud = mass_flow * (1.0 + cycle.fuel_air_ratio)

    area_chamber = area_for_mach(mass_flow, mach_chamber, station2.T0,
                                 station2.p0, gamma, R)
    area_throat = choked_area(debit_chaud, station4.T0, station4.p0, gamma, R)
    area_exit = area_throat * float(area_ratio(cycle.mach_exit, gamma))
    area_capture = area_for_mach(mass_flow, flight.mach,
                                 cycle.station("0").T0, cycle.station("0").p0,
                                 gamma, R)

    if area_throat >= area_chamber:
        raise CaseError(
            f"le col ({area_throat * 1e4:.1f} cm2) n'est pas plus etroit que la "
            f"chambre ({area_chamber * 1e4:.1f} cm2) : il n'y a pas de "
            f"convergent, donc pas de blocage. Baissez le Mach de chambre"
        )

    x = 0.0
    zones = []
    for nom, section_fin in (("diffuser", area_chamber),
                             ("chamber", area_chamber),
                             ("converging", area_throat),
                             ("diverging", area_exit)):
        depart = area_capture if not zones else zones[-1].area_end
        zones.append(Zone(nom, x, x + longueurs[nom], depart, section_fin))
        x += longueurs[nom]
    return Duct(tuple(zones))


def chamber_exit_mach(duct: Duct, cycle, mass_flow: float = 10.0,
                      include_fuel_mass: bool = True) -> float:
    """Le Mach en fin de chambre, par conservation de la masse.

    C'est ici, et seulement ici, que l'inversion de A/A* sert : le rapport
    « A_chambre / A_col » vaut 1,5, tres loin du col, donc parfaitement
    conditionne.

    L'APPORT DE MASSE DU CARBURANT COMPTE PLUS QU'ON NE CROIT, ET IL FAUDRA
    CHOISIR. La relation de Rayleigh, qui servira d'oracle au solveur 1D,
    suppose le debit CONSTANT : elle ne connait qu'un apport de chaleur. Le
    moteur, lui, ajoute 3,5 % de masse. Mesure sur le cas de reference, chambre
    dimensionnee a M = 0,20 :

        sans l'apport de masse : A/A* = 1,5146   M4 = 0,4250
        avec l'apport de masse : A/A* = 1,4633   M4 = 0,4442
        Rayleigh (masse constante) : A/A* = 1,5148   M4 = 0,4249

    Sans l'apport, l'accord avec Rayleigh est a 1e-4 — la geometrie est donc
    juste. Avec, l'ecart atteint 4,5 % sur le Mach de sortie de chambre, soit
    PLUS que l'effet de pi_b que l'etape 9 cherchera a mesurer. Le drapeau
    existe pour que la comparaison a Rayleigh se fasse a debit constant, et
    l'etape 6 devra dire lequel des deux elle integre.
    """
    flight = cycle.flight
    facteur = (1.0 + cycle.fuel_air_ratio) if include_fuel_mass else 1.0
    station4 = cycle.station("4")
    critique = choked_area(mass_flow * facteur, station4.T0, station4.p0,
                           flight.gamma, flight.R)
    zone = next(z for z in duct.zones if z.name == "chamber")
    return float(mach_from_area_ratio(zone.area_end / critique, "subsonic",
                                      flight.gamma))


def static_state(cycle, mach: float, station: str) -> dict[str, float]:
    """Les grandeurs statiques a un Mach donne, depuis une station totale.

    Utilitaire de dimensionnement et de test : c'est ce qui permet de verifier
    « mdot = rho u A » sans repasser par le solveur.
    """
    flight = cycle.flight
    etat = cycle.station(station)
    T = etat.T0 / float(total_temperature_ratio(mach, flight.gamma))
    p = etat.p0 / float(total_pressure_ratio(mach, flight.gamma))
    rho = p / (flight.R * T)
    u = mach * float(np.sqrt(flight.gamma * flight.R * T))
    return dict(T=T, p=p, rho=rho, u=u, mach=mach)
