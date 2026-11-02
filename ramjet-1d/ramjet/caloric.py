"""cp(T) et gamma(T) par les polynomes NASA a sept coefficients.

ETAPE 14. Tout le reste du projet suppose « cp = 1004,5 J/(kg.K) » et
« gamma = 1,4 », constants. C'est le premier poste d'erreur du 0D, et le plan
demandait de le MESURER plutot que de le supposer. Ce module ne remplace pas le
modele a cp constant : il le CHIFFRE.

LES COEFFICIENTS SONT DES DONNEES, ET LES DONNEES SE VERIFIENT. Cinquante
nombres recopies sont cinquante occasions de se tromper, et ce projet a deja
attrape deux valeurs citees de memoire qui etaient fausses. Trois controles
independants les protegent, et le second est decisif :

  - « cp » a 298,15 K contre les valeurs de reference : accord a 0,4 % ;
  - LE RACCORD A 1 000 K entre les deux polynomes de chaque espece. Les deux
    intervalles sont ajustes separement mais construits pour se rejoindre : une
    seule faute de frappe dans l'un des dix jeux casse la continuite. Mesure :
    **0,000 % sur les cinq especes** ;
  - la masse molaire de l'air deduite de la composition donne
    « R = 287,048 J/(kg.K) », la ou le reste du projet code 287,05.

PAS DE CHIMIE POUR AUTANT. La composition des produits est FIGEE, deduite de la
seule reaction globale complete « C_c H_h + (c + h/4) O2 -> c CO2 + (h/2) H2O ».
Pas d'equilibre, pas de dissociation, pas de cinetique : le perimetre du projet
les exclut, et c'est justement pourquoi « T04 » reste une donnee imposee. Ce
module dit ce que cette donnee coute, il ne la calcule pas a partir d'une
chimie.

PAS DE SCIPY : module du noyau, bissection ecrite a la main.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseError
from .fuels import Fuel

# Constante des gaz parfaits, CODATA 2018, J/(mol.K).
RU = 8.31446261815324

# Temperature de reference des enthalpies de formation et des PCI.
T_REFERENCE = 298.15

# Bornes des deux intervalles polynomiaux, et bissection de l'inversion.
T_JUNCTION = 1000.0
T_MIN, T_MAX = 200.0, 5000.0
BISECTION_STEPS = 80

# Masses molaires, g/mol.
MOLAR_MASS = {
    "N2": 28.014, "O2": 31.998, "Ar": 39.948, "CO2": 44.009, "H2O": 18.015,
}

# Air sec, fractions MOLAIRES. La somme fait 1 a 1e-6 pres, et un test le
# verifie : une composition qui ne se ferme pas fausse toutes les masses.
AIR_MOLE_FRACTIONS = {
    "N2": 0.780840, "O2": 0.209476, "Ar": 0.009365, "CO2": 0.000319,
}

# NASA-7, format CHEMKIN : (a1..a7) sur [200 ; 1000] puis sur [1000 ; 5000].
# « cp/Ru = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4 »
# « h/(Ru T) = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T »
NASA = {
    "N2": ((3.298677, 1.4082404e-3, -3.963222e-6, 5.641515e-9, -2.444854e-12,
            -1020.8999, 3.950372),
           (2.92664, 1.4879768e-3, -5.68476e-7, 1.0097038e-10, -6.753351e-15,
            -922.7977, 5.980528)),
    "O2": ((3.212936, 1.1274864e-3, -5.75615e-7, 1.3138773e-9, -8.768554e-13,
            -1005.249, 6.034738),
           (3.697578, 6.135197e-4, -1.258842e-7, 1.775281e-11, -1.1364354e-15,
            -1233.9301, 3.189165)),
    "Ar": ((2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366),
           (2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366)),
    "CO2": ((2.275724, 9.922072e-3, -1.0409113e-5, 6.866686e-9, -2.117280e-12,
             -48373.14, 10.188488),
            (4.453623, 3.140168e-3, -1.278410e-6, 2.393996e-10, -1.6690333e-14,
             -48966.96, -0.9553959)),
    "H2O": ((3.386842, 3.474982e-3, -6.354696e-6, 6.968581e-9, -2.506588e-12,
             -30208.11, 2.590233),
            (2.672146, 3.056293e-3, -8.73026e-7, 1.200996e-10, -6.391618e-15,
             -29899.21, 6.862817)),
}


def _coefficients(species: str, temperature):
    """Le jeu qui s'applique, par temperature. Vectorise sur T."""
    if species not in NASA:
        raise CaseError(
            f"espece inconnue : {species!r} "
            f"(au choix : {', '.join(sorted(NASA))})"
        )
    temperature = np.asarray(temperature, dtype=float)
    if np.any(temperature < T_MIN) or np.any(temperature > T_MAX):
        raise CaseError(
            f"temperature hors du domaine des polynomes "
            f"[{T_MIN:g} ; {T_MAX:g}] K "
            f"(recu de {float(np.min(temperature)):g} a "
            f"{float(np.max(temperature)):g})"
        )
    bas, haut = NASA[species]
    return [np.where(temperature < T_JUNCTION, bas[i], haut[i])
            for i in range(7)]


def cp_molar(species: str, temperature):
    """cp molaire, J/(mol.K)."""
    a = _coefficients(species, temperature)
    T = np.asarray(temperature, dtype=float)
    return RU * (a[0] + a[1] * T + a[2] * T ** 2 + a[3] * T ** 3
                 + a[4] * T ** 4)


def enthalpy_molar(species: str, temperature):
    """Enthalpie molaire, J/mol, DE FORMATION comprise (convention NASA)."""
    a = _coefficients(species, temperature)
    T = np.asarray(temperature, dtype=float)
    return RU * T * (a[0] + a[1] * T / 2.0 + a[2] * T ** 2 / 3.0
                     + a[3] * T ** 3 / 4.0 + a[4] * T ** 4 / 5.0 + a[5] / T)


def entropy_molar(species: str, temperature):
    """Entropie molaire a la pression de reference, J/(mol.K).

    « s0/Ru = a1 lnT + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a7 ». C'est elle
    qui permet une detente ISENTROPIQUE a cp variable : sans elle, on serait
    force de reprendre « (p2/p1)^((g-1)/g) » avec un gamma moyen choisi a la
    main, ce qui remplacerait une approximation par une autre.
    """
    a = _coefficients(species, temperature)
    T = np.asarray(temperature, dtype=float)
    return RU * (a[0] * np.log(T) + a[1] * T + a[2] * T ** 2 / 2.0
                 + a[3] * T ** 3 / 3.0 + a[4] * T ** 4 / 4.0 + a[6])


@dataclass(frozen=True)
class Mixture:
    """Un melange de composition FIGEE, decrit par ses fractions molaires."""

    name: str
    mole_fractions: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        total = sum(x for _, x in self.mole_fractions)
        if abs(total - 1.0) > 1e-6:
            raise CaseError(
                f"{self.name} : les fractions molaires font {total:.6f} et "
                f"non 1 — une composition qui ne se ferme pas fausse toutes "
                f"les masses"
            )
        if any(x < 0.0 for _, x in self.mole_fractions):
            raise CaseError(f"{self.name} : fraction molaire negative")

    @property
    def molar_mass(self) -> float:
        """kg/mol."""
        return sum(x * MOLAR_MASS[e] for e, x in self.mole_fractions) / 1000.0

    @property
    def R(self) -> float:
        """Constante massique du melange, J/(kg.K)."""
        return RU / self.molar_mass

    def cp(self, temperature):
        """cp MASSIQUE, J/(kg.K)."""
        molaire = sum(x * cp_molar(e, temperature)
                      for e, x in self.mole_fractions)
        return molaire / self.molar_mass

    def enthalpy(self, temperature):
        """Enthalpie massique, J/kg, formation comprise."""
        molaire = sum(x * enthalpy_molar(e, temperature)
                      for e, x in self.mole_fractions)
        return molaire / self.molar_mass

    def entropy(self, temperature):
        """Entropie massique a la pression de reference, J/(kg.K).

        Les termes de melange « -R ln(x_i) » sont omis : la composition est
        FIGEE, donc ils sont constants et disparaissent de toute difference.
        """
        molaire = sum(x * entropy_molar(e, temperature)
                      for e, x in self.mole_fractions)
        return molaire / self.molar_mass

    def isentropic_temperature(self, T_in: float, pressure_ratio: float
                               ) -> float:
        """La temperature apres une detente isentropique de rapport p2/p1.

        On resout « s0(T2) - s0(T1) = R ln(p2/p1) » par bissection. A cp
        constant cela redonne exactement « T2/T1 = (p2/p1)^((g-1)/g) », et un
        test le verifie sur un gaz monoatomique, ou gamma NE varie pas.
        """
        if pressure_ratio <= 0.0:
            raise CaseError(f"rapport de pression non positif : "
                            f"{pressure_ratio:g}")
        cible = float(self.entropy(T_in)) + self.R * float(
            np.log(pressure_ratio))
        lo, hi = T_MIN, T_MAX
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            if float(self.entropy(mid)) < cible:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def gamma(self, temperature):
        """gamma = cp/(cp - R). Vaut 1,400 a 300 K pour l'air."""
        chaleur = self.cp(temperature)
        return chaleur / (chaleur - self.R)

    def mean_cp(self, T1: float, T2: float) -> float:
        """LE cp QU'IL FAUT A UN CYCLE : « (h2 - h1)/(T2 - T1) ».

        Ce n'est pas cp(T_moyenne), et l'ecart n'est pas negligeable quand cp
        varie vite. C'est celui-la qui conserve l'energie.
        """
        if abs(T2 - T1) < 1e-9:
            return float(self.cp(T1))
        return float((self.enthalpy(T2) - self.enthalpy(T1)) / (T2 - T1))

    def temperature_from_enthalpy(self, target: float) -> float:
        """L'inverse de « enthalpy », par bissection. h croit strictement."""
        lo, hi = T_MIN, T_MAX
        if not (self.enthalpy(lo) <= target <= self.enthalpy(hi)):
            raise CaseError(
                f"{self.name} : enthalpie {target:.4g} J/kg hors du domaine "
                f"[{float(self.enthalpy(lo)):.4g} ; "
                f"{float(self.enthalpy(hi)):.4g}]"
            )
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            if float(self.enthalpy(mid)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def air() -> Mixture:
    """L'air sec standard."""
    return Mixture("air", tuple(AIR_MOLE_FRACTIONS.items()))


# Composition atomique des carburants du registre, par unite de formule.
# Verifiee contre le « f_stoich » stocke : accord a 0,4 %, ce qui dit que les
# deux jeux de donnees sont coherents entre eux.
ATOMS = {"kerosene": (1.0, 1.92), "methane": (1.0, 4.0),
         "hydrogen": (0.0, 2.0)}


def stoichiometric_ratio(fuel: Fuel) -> float:
    """« f » stoechiometrique deduit de la seule composition atomique."""
    if fuel.name not in ATOMS:
        raise CaseError(
            f"composition atomique inconnue pour {fuel.name!r} : ajoutez-la "
            f"a ATOMS pour utiliser le modele a cp variable"
        )
    carbone, hydrogene = ATOMS[fuel.name]
    masse = (carbone * 12.011 + hydrogene * 1.008) / 1000.0
    oxygene = carbone + hydrogene / 4.0
    moles_air = oxygene / AIR_MOLE_FRACTIONS["O2"]
    return masse / (moles_air * air().molar_mass)


def combustion_products(fuel: Fuel, fuel_air_ratio: float) -> Mixture:
    """Les produits FIGES de la reaction globale complete, a « f » donne.

    On brule une mole de carburant dans la quantite d'air que « f » impose, et
    l'on compte ce qui sort : CO2, H2O, l'azote et l'argon inertes, et l'oxygene
    EN EXCES. Aucune dissociation : a 2 000 K elle commence a compter, et c'est
    l'une des limites annoncees du projet.
    """
    if fuel_air_ratio <= 0.0:
        raise CaseError(f"richesse nulle ou negative : {fuel_air_ratio:g}")
    carbone, hydrogene = ATOMS.get(fuel.name, (None, None))
    if carbone is None:
        raise CaseError(f"composition atomique inconnue pour {fuel.name!r}")

    masse_fuel = (carbone * 12.011 + hydrogene * 1.008) / 1000.0
    besoin_o2 = carbone + hydrogene / 4.0
    # Moles d'air par mole de carburant, imposees par « f ».
    moles_air = masse_fuel / (fuel_air_ratio * air().molar_mass)
    if moles_air * AIR_MOLE_FRACTIONS["O2"] < besoin_o2:
        raise CaseError(
            f"melange trop riche : f = {fuel_air_ratio:g} demande plus "
            f"d'oxygene que l'air n'en apporte "
            f"(stoechiometrie a {stoichiometric_ratio(fuel):.4f}). "
            f"Le modele a reaction complete n'a rien a dire au-dela"
        )

    moles = {
        "CO2": moles_air * AIR_MOLE_FRACTIONS["CO2"] + carbone,
        "H2O": hydrogene / 2.0,
        "O2": moles_air * AIR_MOLE_FRACTIONS["O2"] - besoin_o2,
        "N2": moles_air * AIR_MOLE_FRACTIONS["N2"],
        "Ar": moles_air * AIR_MOLE_FRACTIONS["Ar"],
    }
    total = sum(moles.values())
    return Mixture(f"produits ({fuel.name}, f = {fuel_air_ratio:g})",
                   tuple((e, n / total) for e, n in moles.items()))


def combustion_temperature(fuel: Fuel, fuel_air_ratio: float, T_in: float,
                           efficiency: float = 1.0) -> float:
    """T04 REELLE pour un « f » donne, par bilan d'enthalpie a cp variable.

    Le bilan, par kilogramme d'air, avec les enthalpies SENSIBLES rapportees a
    298,15 K — la temperature a laquelle le PCI est defini :

        f eta_b PCI = (1+f) [h_p(T04) - h_p(Tref)] - [h_a(T02) - h_a(Tref)]

    A comparer avec ce que le modele a cp constant en deduit,
    « f eta_b PCI = (1+f) cp T04 - cp T02 » : c'est l'ecart entre les deux qui
    est le resultat de l'etape.
    """
    if not 0.0 < efficiency <= 1.0:
        raise CaseError(f"rendement de combustion hors de ]0, 1] : "
                        f"{efficiency:g}")
    melange_air = air()
    produits = combustion_products(fuel, fuel_air_ratio)

    apport = fuel_air_ratio * efficiency * fuel.lhv
    sensible_air = float(melange_air.enthalpy(T_in)
                         - melange_air.enthalpy(T_REFERENCE))
    cible = ((apport + sensible_air) / (1.0 + fuel_air_ratio)
             + float(produits.enthalpy(T_REFERENCE)))
    return produits.temperature_from_enthalpy(cible)
