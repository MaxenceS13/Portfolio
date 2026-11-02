"""Le rapport texte, et l'ordre dans lequel il dit les choses.

LES ECARTS AUX IDENTITES VIENNENT EN PREMIER, AVANT LE MOINDRE RESULTAT. Ce
n'est pas une coquetterie de presentation : un lecteur presse lit les cinq
premieres lignes et sait s'il peut croire les cinquante suivantes. Un rapport
qui annonce 738 N.s/kg en tete et relegue en annexe le fait que le controle de
quantite de mouvement est a 1e-3 ment par ordonnancement, sans ecrire un seul
nombre faux.

CINQ IDENTITES, ET AUCUNE N'EST UNE COMPARAISON A LA LITTERATURE. Chacune est
une egalite qui doit tomber exactement, a l'arithmetique flottante pres :

    cycle ideal contre forme fermee   le 0D station par station contre l'algebre
    quantite de mouvement             l'impulsion contre l'integrale de p dA
    conservation de la masse          rho u A constant le long du profil
    raccord chambre -> tuyere         l'EDO contre la relation de section
    point fixe de la boucle           g(pi) - pi

La validation contre les fourchettes publiees, elle, arrive plus bas et porte
un autre nom : c'est un accord d'ordre de grandeur, pas une identite.

LES SEUILS SONT CEUX MESURES AUX ETAPES CORRESPONDANTES, avec une marge d'un
facteur cent. Un seuil qu'on n'atteint jamais ne protege de rien ; un seuil
serre au dernier chiffre casse au premier changement de version de SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .solver1d import momentum_residual

LARGEUR = 76


@dataclass(frozen=True)
class Identity:
    """Une egalite qui doit tomber, et le seuil sous lequel on la croit."""

    name: str
    value: float
    threshold: float
    source: str            # l'etape qui l'a mesuree

    @property
    def ok(self) -> bool:
        return self.value < self.threshold

    def line(self) -> str:
        marque = "ok  " if self.ok else "ECHEC"
        return (f"  {marque} {self.name:<38} {self.value:9.2e}"
                f"   seuil {self.threshold:.0e}   {self.source}")


def identities(convergence) -> tuple[Identity, ...]:
    """Les cinq ecarts, calcules sur le point de fonctionnement converge."""
    from .cycle0d import ideal_cycle, run_cycle
    from .config import EngineCase
    from .thermo import mach_from_area_ratio

    moteur = convergence.engine
    cycle, conduit, profil = moteur.cycle, moteur.duct, moteur.profile

    # 1. Le cycle sans pertes contre sa forme fermee.
    parfait = EngineCase(T04=cycle.engine.T04, fuel=cycle.engine.fuel,
                         eta_b=1.0, pi_d=1.0, pi_b=1.0, pi_n=1.0,
                         M_combustor=cycle.engine.M_combustor)
    numerique = run_cycle(cycle.flight, parfait)
    exact = ideal_cycle(cycle.flight, parfait)
    ecart_cycle = abs(numerique.specific_thrust / exact.specific_thrust - 1.0)

    # 2. La quantite de mouvement, sur la chambre — le seul troncon a pas
    #    constant ET reellement integre.
    ecart_qdm = momentum_residual(moteur.chamber, conduit)

    # 3. La masse : rho u A doit valoir le debit LOCAL partout — et le debit
    #    local grandit dans la chambre depuis que l'EDO porte le carburant.
    #    Comparer au debit d'ENTREE donnait 3,5e-02, c'est-a-dire « f » : le
    #    controle mesurait le carburant au lieu de mesurer une erreur.
    flux = profil.density * profil.velocity * profil.area
    ecart_masse = float(np.max(np.abs(flux / profil.mass_flow - 1.0)))

    # 4. Le raccord : l'EDO en fin de chambre contre la relation de section.
    zone = conduit.zone("chamber")
    par_section = float(mach_from_area_ratio(
        conduit.area(zone.x_end) / conduit.throat_area, "subsonic",
        cycle.flight.gamma))
    ecart_raccord = abs(float(moteur.chamber.mach[-1]) - par_section)

    return (
        Identity("cycle ideal contre forme fermee", ecart_cycle, 1e-12,
                 "etape 3"),
        Identity("quantite de mouvement (chambre)", ecart_qdm, 1e-8,
                 "etape 6"),
        Identity("conservation de la masse", ecart_masse, 1e-10, "etape 6"),
        Identity("raccord chambre -> tuyere", ecart_raccord, 1e-9, "etape 8"),
        Identity("point fixe de la boucle 0D <-> 1D", convergence.residual,
                 1e-10, "etape 9"),
    )


def _modele_d_entree(engine) -> str:
    """D'ou vient « pi_d ». Le rapport doit le dire : c'est le premier poste de
    perte du moteur, et il y a maintenant trois facons de le fixer."""
    if engine.pi_d is not None:
        return "valeur imposee"
    if engine.inlet_ramps is not None:
        pluriel = "s" if engine.inlet_ramps > 1 else ""
        return (f"{engine.inlet_ramps} rampe{pluriel} oblique{pluriel} "
                f"optimale{pluriel}, calculees")
    return "correlation MIL-E-5008B"


def _titre(texte: str) -> str:
    return f"\n{texte}\n{'-' * min(len(texte), LARGEUR)}"


def report(convergence, sweep=None) -> str:
    """Le rapport complet, en texte aligne."""
    from . import rayleigh
    from .coupling import _chamber_inlet_mach

    moteur = convergence.engine
    cycle, conduit, profil = moteur.cycle, moteur.duct, moteur.profile
    vol, cas = cycle.flight, cycle.engine

    lignes = [
        f"ramjet-1d {__version__} — rapport",
        "=" * LARGEUR,
        f"  vol      M0 = {vol.mach:g}   z = {vol.altitude:g} m",
        f"  moteur   T04 = {cas.T04:g} K   {cas.fuel_data.label}   "
        f"eta_b = {cas.eta_b:g}   M chambre vise = {cas.M_combustor:g}",
        f"  entree   pi_d = {cas.inlet_recovery(vol.mach):.4f}   "
        f"({_modele_d_entree(cas)})",
    ]

    lignes.append(_titre("CE QU'ON PEUT CROIRE — ecarts aux identites exactes"))
    controles = identities(convergence)
    lignes += [identite.line() for identite in controles]
    if all(identite.ok for identite in controles):
        lignes.append("  Les cinq identites tombent : le reste du rapport peut "
                      "etre lu.")
    else:
        lignes.append("  UNE IDENTITE NE TOMBE PAS : ne pas croire les "
                      "resultats ci-dessous.")

    lignes.append(_titre("LE CYCLE 0D, station par station"))
    for station in cycle.stations:
        lignes.append(station.describe())
    lignes.append(f"  richesse f = {cycle.fuel_air_ratio:.5f}   "
                  f"soit {100 * cycle.fuel_air_ratio / cas.fuel_data.f_stoich:.1f} "
                  f"% du stoechiometrique")

    lignes.append(_titre("LA BOUCLE 0D <-> 1D"))
    lignes.append(convergence.describe())
    chemin = " -> ".join(f"{valeur:.6f}" for valeur in convergence.history)
    lignes.append(f"  chemin                    {chemin}")

    lignes.append(_titre("LE CONDUIT ET L'ECOULEMENT"))
    lignes.append(conduit.describe())
    T02 = cycle.station("2").T0
    reel = _chamber_inlet_mach(moteur.mass_flow, conduit, T02,
                               cycle.station("2").p0, vol.gamma, vol.R)
    limite = float(rayleigh.critical_total_temperature(reel, T02))
    lignes += [
        f"  Mach de chambre           {cas.M_combustor:8.4f} vise, "
        f"{reel:.4f} reel (c'est le col qui le fixe)",
        f"  T04 maximale a ce Mach    {limite:8.1f} K   "
        f"(demande {cas.T04:g} K, marge {limite - cas.T04:.0f} K)",
        f"  Mach au col               {moteur.throat_mach:8.6f}",
        f"  Mach de sortie            {float(profil.mach[-1]):8.4f}",
        f"  p9 / p ambiant            {moteur.exit_pressure_ratio:8.4f}",
    ]

    lignes.append(_titre("LA PERFORMANCE"))
    lignes += [
        f"  poussee specifique        {moteur.specific_thrust:8.1f} N.s/kg "
        f"  (0D : {cycle.specific_thrust:.1f})",
        f"  poussee                   {moteur.thrust:8.1f} N",
        f"  consommation specifique   {cycle.tsfc * 1e6:8.2f} mg/(N.s)",
        f"  impulsion specifique      {cycle.specific_impulse:8.1f} s",
        f"  rendement thermique       {cycle.thermal_efficiency:8.4f}",
        f"  rendement propulsif       {cycle.propulsive_efficiency:8.4f}",
        f"  rendement global          {cycle.overall_efficiency:8.4f}",
    ]

    if sweep is not None:
        lignes.append(_titre("VALIDATION — accord d'ORDRE DE GRANDEUR, "
                             "pas une identite"))
        lignes += sweep_table(sweep)

    lignes.append("")
    return "\n".join(lignes)


def sweep_table(sweep) -> list[str]:
    """Le balayage en Mach, en table alignee, avec les fourchettes publiees."""
    from .cycle0d import (PUBLISHED_SPECIFIC_IMPULSE, PUBLISHED_SPECIFIC_THRUST,
                          PUBLISHED_THERMAL_EFFICIENCY)

    lignes = [
        f"  fourchettes publiees : F/mdot {PUBLISHED_SPECIFIC_THRUST[0]:.0f}"
        f"-{PUBLISHED_SPECIFIC_THRUST[1]:.0f} N.s/kg (sur M0 de 2 a 3,5), "
        f"Isp {PUBLISHED_SPECIFIC_IMPULSE[0]:.0f}"
        f"-{PUBLISHED_SPECIFIC_IMPULSE[1]:.0f} s, "
        f"eta_th {PUBLISHED_THERMAL_EFFICIENCY[0]:.2f}"
        f"-{PUBLISHED_THERMAL_EFFICIENCY[1]:.2f}",
        "",
        "    M0     pi_d      f      F/mdot     S        Isp     M9   eta_th",
        "  " + "-" * (LARGEUR - 4),
    ]
    for mach, cycle in zip(sweep.machs, sweep.cycles):
        lignes.append(
            f"  {mach:5.2f}   {cycle.engine.inlet_recovery(float(mach)):.4f}  "
            f"{cycle.fuel_air_ratio:.4f}  {cycle.specific_thrust:8.1f}  "
            f"{cycle.tsfc * 1e6:6.2f}  {cycle.specific_impulse:7.1f}  "
            f"{cycle.mach_exit:5.2f}  {cycle.thermal_efficiency:.4f}")
    if sweep.refused:
        lignes.append("")
        for mach, raison in sweep.refused:
            lignes.append(f"  M0 = {mach:.2f} REFUSE : {raison}")
    meilleur, valeur = sweep.best_thrust()
    lignes.append("")
    lignes.append(f"  maximum de poussee specifique : {valeur:.1f} N.s/kg "
                  f"a M0 = {meilleur:.2f}")
    return lignes


def write(text: str, path) -> Path:
    """Ecrit le rapport, en creant le dossier. Rend le chemin."""
    chemin = Path(path)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(text, encoding="utf-8")
    return chemin


def case_slug(flight, engine) -> str:
    """Un nom de dossier lisible et stable pour un cas : « M3.0_z11000_T2000 »."""
    return (f"M{flight.mach:g}_z{flight.altitude:g}_T{engine.T04:g}"
            f"_{engine.fuel}")
