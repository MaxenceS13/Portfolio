"""Atmosphere standard ISA, par couches, jusqu'a 32 km.

POURQUOI PAR COUCHES ET PAS PAR UNE FORMULE. L'atmosphere standard n'est pas
une exponentielle : c'est une suite de couches a gradient de temperature
CONSTANT, et la pression s'y integre differemment selon que le gradient est nul
ou non. Recopier une exponentielle unique donnerait 5 % d'erreur a 20 km, ce
qui se propagerait directement dans la pression totale et donc dans la poussee.

LES DEUX INTEGRATIONS, ET POURQUOI ELLES DIFFERENT. L'equilibre hydrostatique
« dp/dz = -rho g » avec « p = rho R T » donne « dp/p = -g dz/(R T) ». Si T varie
lineairement avec z, l'integrale se fait en puissance ; si T est constante, en
exponentielle. Les deux formes sont exactes, et la seconde n'est pas la limite
numerique de la premiere — elle est sa limite MATHEMATIQUE quand le gradient
tend vers zero, et l'evaluer par la premiere donnerait 0/0.

CE QUE CE MODULE NE FAIT PAS. Pas d'humidite, pas de vent, pas de variation
saisonniere ni geographique. « Standard » veut dire « conventionnel » : c'est
une reference de comparaison, pas une meteo. Les performances calculees par ce
projet sont donc des performances EN ATMOSPHERE STANDARD, et le README le dit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Constantes ISA 1976. « R » vaut ici 287,05287 et non les 287,05 du cas de vol :
# c'est la valeur qui definit l'atmosphere standard, et l'utiliser ailleurs
# ferait bouger les tables de reference dans la sixieme decimale.
G0 = 9.80665              # acceleration de la pesanteur, m/s^2
R_ISA = 287.05287         # constante specifique de l'air, J/(kg.K)
P_SEA_LEVEL = 101_325.0   # Pa
T_SEA_LEVEL = 288.15      # K


@dataclass(frozen=True)
class Layer:
    """Une couche a gradient de temperature constant."""

    base_altitude: float      # m
    base_temperature: float   # K
    lapse_rate: float         # K/m — nul pour une couche isotherme
    name: str


# Les trois premieres couches de l'atmosphere standard. Le modele s'arrete a
# 32 km, ce que « config.MAX_ALTITUDE » fait respecter : au-dela, il faudrait
# decrire la mesosphere, et extrapoler donnerait des nombres sans support.
LAYERS = (
    Layer(0.0, 288.15, -0.0065, "troposphere"),
    Layer(11_000.0, 216.65, 0.0, "tropopause (isotherme)"),
    Layer(20_000.0, 216.65, 0.001, "stratosphere inferieure"),
)
TOP_ALTITUDE = 32_000.0


class AtmosphereError(ValueError):
    """Altitude hors du domaine decrit par le modele."""


def _layer_of(altitude: float) -> Layer:
    for layer in reversed(LAYERS):
        if altitude >= layer.base_altitude:
            return layer
    raise AtmosphereError(f"altitude negative : {altitude:g} m")


def _base_pressure(layer: Layer) -> float:
    """Pression a la base d'une couche, integree depuis le niveau de la mer.

    Calculee de proche en proche plutot que tabulee : une table serait une
    seconde source de verite, et c'est exactement la que se logent les
    incoherences de la sixieme decimale entre deux couches.
    """
    pressure = P_SEA_LEVEL
    for lower, upper in zip(LAYERS, LAYERS[1:]):
        if layer.base_altitude <= lower.base_altitude:
            break
        pressure = _pressure_within(lower, pressure, upper.base_altitude)
    return pressure


def _pressure_within(layer: Layer, base_pressure: float, altitude: float) -> float:
    """Pression dans une couche, depuis sa base. Deux formes, deux cas."""
    height = altitude - layer.base_altitude
    if layer.lapse_rate == 0.0:
        # Couche isotherme : exponentielle. Ce n'est pas la limite numerique de
        # l'autre forme, c'est sa limite mathematique — l'autre donnerait 0/0.
        return base_pressure * math.exp(
            -G0 * height / (R_ISA * layer.base_temperature))
    temperature = layer.base_temperature + layer.lapse_rate * height
    exponent = -G0 / (layer.lapse_rate * R_ISA)
    return base_pressure * (temperature / layer.base_temperature) ** exponent


def temperature(altitude: float) -> float:
    """Temperature statique de l'atmosphere standard, K."""
    _check(altitude)
    layer = _layer_of(altitude)
    return layer.base_temperature + layer.lapse_rate * (altitude - layer.base_altitude)


def pressure(altitude: float) -> float:
    """Pression statique de l'atmosphere standard, Pa."""
    _check(altitude)
    layer = _layer_of(altitude)
    return _pressure_within(layer, _base_pressure(layer), altitude)


def density(altitude: float) -> float:
    """Masse volumique, par la loi des gaz parfaits."""
    return pressure(altitude) / (R_ISA * temperature(altitude))


def speed_of_sound(altitude: float, gamma: float = 1.4) -> float:
    """Vitesse du son locale, m/s."""
    return math.sqrt(gamma * R_ISA * temperature(altitude))


def state(altitude: float, gamma: float = 1.4) -> dict[str, float]:
    """Tout l'etat atmospherique en un point, pour les rapports."""
    return dict(altitude=altitude, T=temperature(altitude), p=pressure(altitude),
                rho=density(altitude), a=speed_of_sound(altitude, gamma),
                layer=_layer_of(altitude).name)


def _check(altitude: float) -> None:
    if not 0.0 <= altitude <= TOP_ALTITUDE:
        raise AtmosphereError(
            f"altitude {altitude:g} m hors du modele : il decrit [0, "
            f"{TOP_ALTITUDE:g}] m, soit les trois premieres couches ISA"
        )
