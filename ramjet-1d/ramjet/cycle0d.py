"""Le cycle 0D : quatre stations, et une forme fermee pour les juger.

LA NUMEROTATION, ET CE QU'ELLE N'INVENTE PAS. Quatre stations seulement :

    0   amont, l'air non perturbe
    2   sortie du diffuseur = entree de la chambre
    4   sortie de la chambre = entree de la tuyere
    9   sortie de la tuyere

Les normes en numerotent davantage — 1 pour la levre d'entree, 3 pour le
diffuseur interne, 8 pour le col. Ce modele ne les distingue pas, donc il ne
les nomme pas : une station qui n'a pas d'equation propre est une station de
trop, et la numeroter laisserait croire qu'on sait ce qui s'y passe.

UNE FONCTION PAR ORGANE, PAS UN BLOC. « inlet », « combustor » et « nozzle »
sont separees parce qu'elles se testent separement, et parce que l'etape 9 les
appellera dans une boucle en ne changeant qu'un seul de leurs arguments. Un bloc
monolithique obligerait a tout recalculer pour reajuster « pi_b ».

L'ORACLE EST UNE FORME FERMEE, ET C'EST LE CRITERE DE L'ETAPE. Sans aucune
perte, le cycle se resout a la main : le rapport de pression totale valant 1 et
la detente etant complete, le Mach de sortie EGALE le Mach de vol, et

    F/mdot = V0 (sqrt(tau_lambda / tau_r) - 1)      avec tau_r = T02/T0,
                                                         tau_lambda = T04/T0

« ideal_cycle » calcule cela directement ; « run_cycle » y arrive par les
quatre stations. Les deux doivent tomber sur le meme nombre a la precision
machine, et c'est le seul endroit du projet ou le 0D peut etre verifie
EXACTEMENT — apres quoi il n'y aura plus que des ordres de grandeur.

LE DEBIT DE CARBURANT SORT D'UN BILAN D'ENERGIE, PAS D'UNE RICHESSE IMPOSEE.

    cp T02 + f eta_b PCI = (1 + f) cp T04    ->    f = cp(T04-T02)/(eta_b PCI - cp T04)

Deux consequences, et les deux sont des refus plutot que des resultats. Quand
T02 atteint T04, « f » s'annule : le moteur ne peut plus rien bruler parce que
la compression dynamique a deja tout chauffe. C'est une LIMITE PHYSIQUE, pas un
bug, et elle tombe a M0 = 6,42 pour T04 = 2 000 K. Et quand « f » depasse le
dosage stoechiometrique, le modele sort de son domaine : sans chimie, une
richesse superieure a 1 ne veut rien dire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import atmosphere
from .config import CaseError, EngineCase, FlightCase
from .thermo import (mach_from_total_pressure_ratio, speed_of_sound,
                     total_pressure_ratio, total_temperature_ratio, velocity)


@dataclass(frozen=True)
class Station:
    """L'etat total en un point du cycle. C'est tout ce que le 0D connait."""

    name: str
    T0: float          # temperature totale, K
    p0: float          # pression totale, Pa

    def describe(self) -> str:
        return (f"  station {self.name:<2}  T0 = {self.T0:8.2f} K"
                f"   p0 = {self.p0:12.1f} Pa")


@dataclass(frozen=True, eq=False)
class Cycle:
    """Le resultat complet d'un point de fonctionnement."""

    stations: tuple[Station, ...]
    fuel_air_ratio: float          # f
    specific_thrust: float         # F/mdot, N.s/kg
    tsfc: float                    # S, kg/(N.s)
    mach_exit: float
    velocity_exit: float
    velocity_flight: float
    thermal_efficiency: float
    propulsive_efficiency: float
    overall_efficiency: float
    equivalence_ratio: float
    flight: FlightCase
    engine: EngineCase

    def station(self, name: str) -> Station:
        for station in self.stations:
            if station.name == name:
                return station
        raise KeyError(f"station inconnue : {name!r}")

    @property
    def specific_impulse(self) -> float:
        """Impulsion specifique, en secondes : 1/(S g0).

        La grandeur que la litterature publie le plus souvent pour un
        statoreacteur, et celle qui permet de se comparer a un moteur-fusee.
        """
        return 1.0 / (self.tsfc * 9.80665)

    def describe(self) -> str:
        lignes = [station.describe() for station in self.stations]
        lignes.append(f"  f = {self.fuel_air_ratio:.5f}   richesse = "
                      f"{self.equivalence_ratio:.3f}")
        lignes.append(f"  F/mdot = {self.specific_thrust:.1f} N.s/kg   "
                      f"S = {self.tsfc * 1e6:.2f} mg/(N.s)   "
                      f"Isp = {self.specific_impulse:.0f} s")
        lignes.append(f"  eta_th = {self.thermal_efficiency:.4f}   "
                      f"eta_p = {self.propulsive_efficiency:.4f}   "
                      f"eta_o = {self.overall_efficiency:.4f}")
        return "\n".join(lignes)


# -- les organes, un par un -------------------------------------------------

def freestream(flight: FlightCase) -> Station:
    """Station 0 : l'atmosphere, plus la compression dynamique du vol."""
    T = atmosphere.temperature(flight.altitude)
    p = atmosphere.pressure(flight.altitude)
    return Station("0",
                   T0=T * float(total_temperature_ratio(flight.mach, flight.gamma)),
                   p0=p * float(total_pressure_ratio(flight.mach, flight.gamma)))


def inlet(flight: FlightCase, engine: EngineCase, upstream: Station) -> Station:
    """Station 2 : le diffuseur. ADIABATIQUE, donc T0 inchangee.

    Un diffuseur ne fait pas de travail et n'echange pas de chaleur : toute la
    compression dynamique est deja dans T0 a la station 0. Ce qu'il perd, il le
    perd en PRESSION totale, et c'est « pi_d » qui le dit.
    """
    return Station("2", T0=upstream.T0,
                   p0=upstream.p0 * engine.inlet_recovery(flight.mach))


def combustor(flight: FlightCase, engine: EngineCase,
              upstream: Station) -> tuple[Station, float]:
    """Station 4 : la chambre. Rend l'etat et le dosage « f ».

    « f » sort du bilan d'energie et non d'une richesse imposee, et les deux
    refus qui l'entourent sont des limites physiques — voir l'en-tete.
    """
    fuel = engine.fuel_data
    cp = flight.cp
    denominateur = engine.eta_b * fuel.lhv - cp * engine.T04
    if denominateur <= 0.0:
        # Inatteignable avec les carburants du registre — il faudrait
        # T04 > 41 700 K pour le kerosene — mais un signe faux ailleurs le
        # ferait apparaitre, et un « f » negatif silencieux serait pire.
        raise CaseError(
            f"T04 = {engine.T04:g} K depasse ce que {fuel.label} peut produire "
            f"(limite {engine.eta_b * fuel.lhv / cp:.0f} K)"
        )

    ratio = cp * (engine.T04 - upstream.T0) / denominateur
    if ratio <= 0.0:
        limite = mach_limit(flight, engine)
        raise CaseError(
            f"T02 = {upstream.T0:.1f} K atteint deja T04 = {engine.T04:g} K : "
            f"il n'y a plus rien a bruler. A cette temperature de fin de "
            f"combustion, le statoreacteur cesse de pousser a M0 = {limite:.2f} "
            f"(vol demande : M0 = {flight.mach:g})"
        )

    richesse = fuel.equivalence_ratio(ratio)
    if richesse > 1.0:
        raise CaseError(
            f"richesse {richesse:.2f} > 1 : le bilan d'energie demande "
            f"f = {ratio:.4f} quand {fuel.label} n'en accepte que "
            f"{fuel.f_stoich:.4f}. Sans chimie, ce modele n'a rien a dire "
            f"au-dela — baissez T04"
        )
    return Station("4", T0=engine.T04, p0=upstream.p0 * engine.pi_b), ratio


def nozzle(flight: FlightCase, engine: EngineCase, upstream: Station,
           ambient_pressure: float) -> tuple[Station, float]:
    """Station 9 : la tuyere, DETENDUE JUSQU'A LA PRESSION AMBIANTE.

    « p9 = p0 » est une hypothese de dimensionnement — la tuyere est adaptee —
    et c'est elle qui fixe le Mach de sortie par le seul rapport de detente.
    Une tuyere sous-detendue ajouterait un terme « (p9 - p0) A9 » a la poussee ;
    c'est le scope optionnel de l'etape 15.
    """
    exit_station = Station("9", T0=upstream.T0, p0=upstream.p0 * engine.pi_n)
    ratio = exit_station.p0 / ambient_pressure
    if ratio < 1.0:
        raise CaseError(
            f"pression totale de sortie {exit_station.p0:.0f} Pa inferieure a "
            f"l'ambiante {ambient_pressure:.0f} Pa : les pertes depassent la "
            f"compression dynamique, il n'y a pas de detente possible"
        )
    mach = float(mach_from_total_pressure_ratio(ratio, flight.gamma))
    return exit_station, mach


# -- le cycle complet -------------------------------------------------------

def run_cycle(flight: FlightCase, engine: EngineCase) -> Cycle:
    """Le cycle station par station. C'est lui que l'oracle doit valider."""
    ambient_T = atmosphere.temperature(flight.altitude)
    ambient_p = atmosphere.pressure(flight.altitude)

    station0 = freestream(flight)
    station2 = inlet(flight, engine, station0)
    station4, fuel_air = combustor(flight, engine, station2)
    station9, mach9 = nozzle(flight, engine, station4, ambient_p)

    T9 = station9.T0 / float(total_temperature_ratio(mach9, flight.gamma))
    V9 = float(velocity(mach9, T9, flight.gamma, flight.R))
    V0 = flight.mach * float(speed_of_sound(ambient_T, flight.gamma, flight.R))

    # Tuyere adaptee : « p9 = p0 », donc aucun terme de pression dans la
    # poussee. Le debit sortant vaut (1 + f) fois le debit d'air entrant.
    specific_thrust = (1.0 + fuel_air) * V9 - V0
    fuel = engine.fuel_data

    # Les trois rendements, definis par des PUISSANCES, ce qui rend
    # « eta_o = eta_th eta_p » vraie par construction plutot que par chance.
    power_fuel = fuel_air * fuel.lhv
    power_kinetic = 0.5 * ((1.0 + fuel_air) * V9 ** 2 - V0 ** 2)
    power_propulsive = specific_thrust * V0

    return Cycle(
        stations=(station0, station2, station4, station9),
        fuel_air_ratio=fuel_air,
        specific_thrust=specific_thrust,
        tsfc=fuel_air / specific_thrust,
        mach_exit=mach9, velocity_exit=V9, velocity_flight=V0,
        thermal_efficiency=power_kinetic / power_fuel,
        propulsive_efficiency=power_propulsive / power_kinetic,
        overall_efficiency=power_propulsive / power_fuel,
        equivalence_ratio=fuel.equivalence_ratio(fuel_air),
        flight=flight, engine=engine,
    )


# -- l'oracle ---------------------------------------------------------------

@dataclass(frozen=True)
class IdealCycle:
    """La solution analytique du statoreacteur sans pertes."""

    mach_exit: float
    fuel_air_ratio: float
    specific_thrust: float
    tsfc: float
    thermal_efficiency: float
    propulsive_efficiency: float
    overall_efficiency: float


def ideal_cycle(flight: FlightCase, engine: EngineCase) -> IdealCycle:
    """La forme fermee, sans une seule iteration.

    Sans pertes, le rapport de pression totale entre l'entree et la sortie vaut
    1 ; la detente etant complete, « p9 = p0 » impose alors M9 = M0. Le rapport
    des vitesses se reduit a « sqrt(T09/T00) », et tout le reste suit.

    Le rendement thermique vaut « 1 - 1/tau_r », la forme du cycle de Brayton
    dont le rapport de compression est celui du ram — et c'est EXACT, pas
    asymptotique. La premiere version de cette docstring annoncait le contraire
    (« vrai seulement a la limite f -> 0 ») ; le test l'a contredite a 5,5e-17,
    et l'algebre confirme :

        (1+f) V9^2 - V0^2 = 2 cp T0 (tau_r - 1) [(1+f) tau_lambda/tau_r - 1]
        f PCI             = cp T0 [(tau_lambda - tau_r) + f tau_lambda]

    Les crochets sont proportionnels, le rapport se simplifie en
    « (tau_r - 1)/tau_r », et « f » disparait. C'est donc une identite exacte,
    donc un oracle de plus.

    Le rendement PROPULSIF, lui, depend bien de f : la forme classique
    « 2/(V9/V0 + 1) » neglige le debit de carburant et s'ecarte de 2,1 % a
    M0 = 2, 2,6 % a M0 = 3 et 3,2 % a M0 = 4. C'est la forme complete qui est
    calculee ici.
    """
    T_ambient = atmosphere.temperature(flight.altitude)
    tau_r = float(total_temperature_ratio(flight.mach, flight.gamma))
    tau_lambda = engine.T04 / T_ambient
    if tau_lambda <= tau_r:
        raise CaseError(
            f"T04 = {engine.T04:g} K n'excede pas la temperature de compression "
            f"dynamique ({T_ambient * tau_r:.1f} K) : pas de cycle a M0 = "
            f"{flight.mach:g}"
        )

    V0 = flight.mach * float(speed_of_sound(T_ambient, flight.gamma, flight.R))
    ratio = np.sqrt(tau_lambda / tau_r)
    fuel = engine.fuel_data
    fuel_air = (flight.cp * T_ambient * (tau_lambda - tau_r)
                / (fuel.lhv - flight.cp * engine.T04))
    specific_thrust = (1.0 + fuel_air) * V0 * ratio - V0

    power_fuel = fuel_air * fuel.lhv
    power_kinetic = 0.5 * ((1.0 + fuel_air) * (V0 * ratio) ** 2 - V0 ** 2)
    power_propulsive = specific_thrust * V0
    return IdealCycle(
        mach_exit=flight.mach, fuel_air_ratio=fuel_air,
        specific_thrust=specific_thrust,
        tsfc=fuel_air / specific_thrust,
        thermal_efficiency=power_kinetic / power_fuel,
        propulsive_efficiency=power_propulsive / power_kinetic,
        overall_efficiency=power_propulsive / power_fuel,
    )


def ideal_engine(engine: EngineCase) -> EngineCase:
    """Le meme moteur, toutes pertes mises a 1. C'est ce que l'oracle suppose."""
    from dataclasses import replace

    return replace(engine, eta_b=1.0, pi_d=1.0, pi_b=1.0, pi_n=1.0)


def mach_limit(flight: FlightCase, engine: EngineCase) -> float:
    """Le Mach de vol ou la compression dynamique atteint deja T04.

    Au-dela, il n'y a plus rien a bruler et le moteur cesse de pousser. Vaut
    6,42 pour T04 = 2 000 K a 11 km, et c'est une formule, pas une mesure :

        M_lim = sqrt( 2/(gamma-1) (T04/T0 - 1) )
    """
    T_ambient = atmosphere.temperature(flight.altitude)
    if engine.T04 <= T_ambient:
        return 0.0
    return float(np.sqrt(2.0 / (flight.gamma - 1.0)
                         * (engine.T04 / T_ambient - 1.0)))


# -- balayage et validation -------------------------------------------------

# Fourchettes publiees pour un statoreacteur de croisiere. Ce sont des ordres
# de grandeur couvrant des moteurs de dessins differents : y tomber VALIDE le
# modele, ca ne le calibre pas. Elles ne servent a aucun calcul — seulement a
# comparer — et c'est pour ca qu'elles sont ici et pas dans une formule.
PUBLISHED_SPECIFIC_THRUST = (600.0, 900.0)      # N.s/kg
PUBLISHED_SPECIFIC_IMPULSE = (1500.0, 2500.0)   # s
PUBLISHED_THERMAL_EFFICIENCY = (0.25, 0.75)


@dataclass(frozen=True, eq=False)
class Sweep:
    """Un balayage en Mach de vol, a moteur fixe.

    Les points REFUSES sont gardes a part plutot qu'ecartes en silence : le
    balayage traverse la limite thermique des qu'il va assez loin, et savoir OU
    il s'arrete est une information, pas un incident.
    """

    machs: np.ndarray
    cycles: tuple[Cycle, ...]
    refused: tuple[tuple[float, str], ...]
    engine: EngineCase

    def column(self, name: str) -> np.ndarray:
        """Une grandeur du cycle, en tableau, dans l'ordre des Mach."""
        return np.array([getattr(cycle, name) for cycle in self.cycles])

    @property
    def specific_thrust(self) -> np.ndarray:
        return self.column("specific_thrust")

    @property
    def specific_impulse(self) -> np.ndarray:
        return np.array([cycle.specific_impulse for cycle in self.cycles])

    def best_thrust(self) -> tuple[float, float]:
        """(Mach du maximum de poussee specifique, valeur).

        Le maximum EXISTE, et c'est le fait marquant du statoreacteur : sous
        M0 = 2 la compression dynamique ne suffit pas, au-dessus de M0 = 3
        l'air arrive trop chaud pour qu'on puisse encore le chauffer.
        """
        if not self.cycles:
            raise CaseError("balayage vide : aucun point n'a abouti")
        valeurs = self.specific_thrust
        index = int(np.argmax(valeurs))
        return float(self.machs[index]), float(valeurs[index])

    def within(self, name: str, bounds: tuple[float, float]) -> np.ndarray:
        """Masque des points dont la grandeur tombe dans la fourchette."""
        lo, hi = bounds
        valeurs = (self.specific_impulse if name == "specific_impulse"
                   else self.column(name))
        return (valeurs >= lo) & (valeurs <= hi)


def sweep_mach(machs, engine: EngineCase, altitude: float = 11_000.0,
               gamma: float = 1.4, R: float = 287.05) -> Sweep:
    """Balaie le Mach de vol a moteur fixe. N'AJOUTE AUCUNE PHYSIQUE.

    Cette fonction mesure, elle ne modelise pas : c'est « run_cycle » qui
    travaille, et le balayage ne fait que l'appeler en enregistrant ce qui
    passe et ce qui est refuse.
    """
    retenus, refuses, machs_ok = [], [], []
    for mach in np.atleast_1d(np.asarray(machs, dtype=float)):
        try:
            cycle = run_cycle(FlightCase(mach=float(mach), altitude=altitude,
                                         gamma=gamma, R=R), engine)
        except CaseError as erreur:
            refuses.append((float(mach), str(erreur)))
            continue
        retenus.append(cycle)
        machs_ok.append(float(mach))
    return Sweep(machs=np.array(machs_ok), cycles=tuple(retenus),
                 refused=tuple(refuses), engine=engine)


# -- le cycle a proprietes variables -----------------------------------------
#
# CE QUE CE CYCLE PEUT, ET CE QU'IL NE FAUT PAS LUI DEMANDER. L'etape 14 a
# CHIFFRE l'erreur du modele a « cp » constant sans la corriger : elle valait
# 256 K sur T04 a carburant donne, et 5,6 % sur la poussee. Ce cycle-ci la
# corrige, pour le 0D.
#
# LE 1D, LUI, RESTE A GAMMA CONSTANT, ET C'EST UN CHOIX ASSUME. Ses trois
# oracles — l'isentropique, Rayleigh et Fanno — supposent tous « gamma »
# constant. Les etendre a « gamma(T) » ne demanderait pas d'ecrire une equation
# de plus : cela DETRUIRAIT la structure de verification de tout le projet, qui
# repose sur le fait que chaque terme source se compare a une forme fermee
# exacte. Un solveur 1D a proprietes variables serait plus juste et moins
# verifiable, et ce depot a choisi l'autre terme de l'echange — en le mesurant
# plutot qu'en le supposant.
#
# COMMENT CE CYCLE-CI EST VERIFIE, PUISQU'IL N'A PAS D'ORACLE A LUI. Par
# DEGENERESCENCE : on lui donne un gaz dont « cp » ne varie pas — l'argon, seul
# gaz monoatomique du jeu NASA — et il doit alors rendre exactement ce que le
# cycle a « cp » constant rend. Les melanges sont donc des ARGUMENTS et non des
# constantes du module : c'est ce qui rend le test possible.


@dataclass(frozen=True, eq=False)
class VariableCycle:
    """Le meme cycle, mais avec cp(T) et gamma(T) partout."""

    stations: tuple[Station, ...]
    fuel_air_ratio: float
    specific_thrust: float
    tsfc: float
    velocity_exit: float
    velocity_flight: float
    exit_temperature: float
    thermal_efficiency: float
    mean_cp_combustion: float      # le cp moyen reellement subi, J/(kg.K)

    def station(self, name: str) -> Station:
        for station in self.stations:
            if station.name == name:
                return station
        raise CaseError(f"station inconnue : {name!r}")

    @property
    def specific_impulse(self) -> float:
        return self.specific_thrust / (self.fuel_air_ratio * 9.80665)

    def describe(self) -> str:
        return "\n".join([
            f"  f = {self.fuel_air_ratio:.5f}   "
            f"F/mdot = {self.specific_thrust:.1f} N.s/kg   "
            f"Isp = {self.specific_impulse:.0f} s",
            f"  u9 = {self.velocity_exit:.1f} m/s   "
            f"T9 = {self.exit_temperature:.1f} K   "
            f"cp moyen de combustion = {self.mean_cp_combustion:.1f} J/(kg.K)",
        ])


def _total_temperature_variable(mixture, temperature: float,
                                velocity: float) -> float:
    """T0 tel que « h(T0) = h(T) + u^2/2 ». Pas de « 1 + (g-1)/2 M^2 » ici :
    cette formule suppose gamma constant, et c'est justement ce qu'on quitte."""
    return mixture.temperature_from_enthalpy(
        float(mixture.enthalpy(temperature)) + 0.5 * velocity ** 2)


def run_cycle_variable(flight: FlightCase, engine: EngineCase,
                       air_mixture=None, products=None) -> VariableCycle:
    """Le cycle 0D avec « cp(T) » et « gamma(T) », par enthalpie et entropie.

    Les compressions et detentes ne passent plus par « (p2/p1)^((g-1)/g) » mais
    par l'egalite des entropies, et les bilans d'energie par les enthalpies. Les
    deux melanges peuvent etre imposes — c'est ce qui permet de verifier le
    cycle par degenerescence sur un gaz a « cp » constant.
    """
    from . import caloric

    fuel = engine.fuel_data
    air = caloric.air() if air_mixture is None else air_mixture
    ambiant = atmosphere.state(flight.altitude)
    T0, p0 = ambiant["T"], ambiant["p"]

    # Station 0 : l'etat total du vol, par enthalpie et non par gamma.
    vitesse = flight.mach * float(np.sqrt(flight.gamma * flight.R * T0))
    T00 = _total_temperature_variable(air, T0, vitesse)
    p00 = p0 * float(np.exp((air.entropy(T00) - air.entropy(T0)) / air.R))

    # Station 2 : le diffuseur est adiabatique, seule la pression tombe.
    T02, p02 = T00, p00 * engine.inlet_recovery(flight.mach)

    # Combustion : « f » est implicite, car la composition des produits en
    # depend. Bissection sur « f », monotone parce que plus de carburant fait
    # monter T04 pour un apport donne.
    T04 = engine.T04
    if products is None:
        def temperature(f: float) -> float:
            return caloric.combustion_temperature(fuel, f, T02, engine.eta_b)
    else:
        def temperature(f: float) -> float:
            apport = f * engine.eta_b * fuel.lhv
            sensible = float(air.enthalpy(T02)
                             - air.enthalpy(caloric.T_REFERENCE))
            cible = ((apport + sensible) / (1.0 + f)
                     + float(products.enthalpy(caloric.T_REFERENCE)))
            return products.temperature_from_enthalpy(cible)

    # LA BORNE HAUTE EST LA STOECHIOMETRIE, PAS UN NOMBRE ROND. Au-dela, le
    # modele a reaction complete n'a plus rien a dire — il manquerait de
    # l'oxygene — et c'est une limite deja annoncee du perimetre.
    plafond = (fuel.f_stoich if products is not None
               else caloric.stoichiometric_ratio(fuel))
    # LA BORNE HAUTE PEUT ETRE INCALCULABLE, ET PAS SEULEMENT TROP RICHE. A la
    # stoechiometrie, un gaz a faible « cp » depasse les 5 000 K du domaine des
    # polynomes NASA. On retrecit donc la borne jusqu'a ce qu'elle soit
    # evaluable, exactement comme la recherche de debit de l'etape 8 retrecit
    # la sienne quand la chambre bloque : une exception n'a pas de signe, mais
    # elle a un sens.
    lo, hi = 1e-6, plafond * (1.0 - 1e-9)
    haut = None
    while hi > lo * 2.0:
        try:
            haut = temperature(hi)
            break
        except CaseError:
            hi *= 0.5
    if haut is None or haut < T04:
        raise CaseError(
            f"T04 = {T04:g} K est hors d'atteinte avec {fuel.label} : au plus "
            f"riche dosage exploitable (f = {hi:.4f}) la combustion ne depasse "
            f"pas {haut:.0f} K a proprietes variables"
            if haut is not None else
            f"aucun dosage exploitable pour {fuel.label} a T02 = {T02:.0f} K"
        )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if temperature(mid) < T04:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    gaz = caloric.combustion_products(fuel, f) if products is None else products
    p04 = p02 * engine.pi_b

    # Detente : entropie constante jusqu'a l'ambiante, tuyere adaptee.
    p09 = p04 * engine.pi_n
    T9 = gaz.isentropic_temperature(T04, p0 / p09)
    u9 = float(np.sqrt(2.0 * (gaz.enthalpy(T04) - gaz.enthalpy(T9))))

    poussee = (1.0 + f) * u9 - vitesse
    puissance_cinetique = 0.5 * ((1.0 + f) * u9 ** 2 - vitesse ** 2)
    return VariableCycle(
        stations=(Station("0", T00, p00), Station("2", T02, p02),
                  Station("4", T04, p04), Station("9", T04, p09)),
        fuel_air_ratio=f, specific_thrust=poussee, tsfc=f / poussee,
        velocity_exit=u9, velocity_flight=vitesse, exit_temperature=T9,
        thermal_efficiency=puissance_cinetique / (f * fuel.lhv),
        mean_cp_combustion=gaz.mean_cp(caloric.T_REFERENCE, T04),
    )
