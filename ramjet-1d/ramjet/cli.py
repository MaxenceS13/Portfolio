"""Interface en ligne de commande.

A ce stade du projet, la ligne de commande ne sait faire qu'une chose :
« --check », qui verifie l'environnement, le cloisonnement des dependances et
la construction d'un cas. Les modes « --validate » (balayage en Mach compare
aux fourchettes publiees) et « --figures » arrivent avec les etapes qui
produisent quelque chose a mesurer.

« --check » ne se contente pas de lister des versions. Il importe le noyau dans
un interpreteur NEUF et verifie qu'aucune des deux bibliotheques cloisonnees
n'a ete tiree au passage, puis recommence avec une contrainte plus forte sur
« config » et « fuels », qui doivent tenir sans NumPy. C'est la garantie qu'un
script d'optimisation exterieur pourra construire des milliers de cas sans
payer l'importation d'une bibliotheque de calcul.

CODES DE SORTIE. 0 tout va bien ; 1 erreur ; 2 un critere numerique n'est pas
atteint — c'est le code qu'un script d'integration doit surveiller ; 130
interrompu.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import (CORE_MODULES, FORBIDDEN_IN_CORE, STDLIB_ONLY_MODULES,
               __version__)
from .config import CaseError, EngineCase, FlightCase
from .solver1d import ChokedFlow
from .fuels import available as available_fuels

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sorties"

# Ecart tolere sur l'aller-retour « M -> A/A* -> M ». Mesure a l'etape 2 :
# 6,2e-15 en subsonique et 2,2e-15 en supersonique, loin des 1e-12 demandes par
# le plan. Le seuil garde trois ordres de marge sans etre complaisant.
INVERSION_TOL = 1e-12

# Ecart relatif tolere entre le cycle station par station et sa forme fermee.
# Mesure a l'etape 3 : 2,3e-13 en absolu sur la poussee specifique, soit 3e-16
# en relatif. Le seuil laisse trois ordres de marge.
CYCLE_TOL = 1e-12

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FAILED = 2
EXIT_INTERRUPTED = 130

# Role de chaque dependance, affiche par « --check ». Le booleen dit si son
# absence est bloquante : NumPy l'est, les deux autres ne le sont que pour la
# partie du projet qui les utilise, et le cloisonnement est fait pour ca.
DEPENDENCIES = (
    ("numpy", True, "thermo, cycle 0D, conduit, profils"),
    ("scipy", False, "solver1d.py (solve_ivp et ses evenements) et les tests"),
    ("matplotlib", False, "plots.py, les figures"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ramjet-1d",
        description="Statoreacteur : cycle 0D et ecoulement 1D avec combustion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemples :
  python3 main.py --check              # environnement et cloisonnement
  python3 main.py                      # le cas par defaut, rapport complet
  python3 main.py --mach 2.5 --t04 2200
  python3 main.py --rampes 3           # entree a 3 rampes au lieu de MIL
  python3 main.py --sweep              # le balayage en Mach, en table
  python3 main.py --validate           # balayage + fourchettes publiees
  python3 main.py --figures            # les cinq figures dans sorties/
  python3 main.py --m-comb 0.30 ; echo $?     # cas bloque : sort en 2

codes de sortie : 0 ok, 1 erreur, 2 critere numerique non atteint, 130 interrompu
""")

    modes = parser.add_argument_group("modes")
    modes.add_argument("--check", action="store_true",
                       help="verifie l'environnement et le cloisonnement, puis sort")
    modes.add_argument("--figures", action="store_true",
                       help="ecrit les cinq figures du point de fonctionnement")
    modes.add_argument("--sweep", action="store_true",
                       help="balaye le Mach de vol et rend une table texte")
    modes.add_argument("--validate", action="store_true",
                       help="balayage + comparaison aux fourchettes publiees")

    cas = parser.add_argument_group("cas de calcul")
    cas.add_argument("--mach", type=float, default=None,
                     metavar="M0", help="Mach de vol (defaut : 3)")
    cas.add_argument("--altitude", type=float, default=None, metavar="Z",
                     help="altitude en metres (defaut : 11000)")
    cas.add_argument("--t04", type=float, default=None, metavar="T",
                     help="temperature totale de fin de combustion, K "
                          "(defaut : 2000)")
    cas.add_argument("--fuel", type=str, default=None,
                     help=f"carburant : {', '.join(available_fuels())} "
                          f"(defaut : kerosene)")
    cas.add_argument("--m-comb", type=float, default=None, dest="m_comb",
                     metavar="M", help="Mach vise en entree de chambre "
                                       "(defaut : 0.20)")
    cas.add_argument("--rampes", type=int, default=None, metavar="N",
                     help="calcule pi_d pour une entree a N rampes obliques "
                          "optimales, au lieu de la correlation MIL-E-5008B")
    cas.add_argument("--range", type=float, nargs=2, default=(1.5, 5.0),
                     metavar=("MIN", "MAX"),
                     help="bornes du balayage (defaut : 1.5 5.0)")
    cas.add_argument("--points", type=int, default=15, metavar="N",
                     help="nombre de points du balayage (defaut : 15)")

    parser.add_argument("-o", "--output", type=Path, default=Path("sorties"),
                        help="dossier de sortie (defaut : sorties/)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="calcule et affiche, mais n'ecrit aucun fichier")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="affiche le detail des verifications")
    parser.add_argument("--version", action="version",
                        version=f"ramjet-1d {__version__}")
    return parser


# -- affichage ------------------------------------------------------------

def _line(ok: bool, label: str, detail: str = "") -> str:
    mark = f"{GREEN}ok{RESET}" if ok else f"{RED}echec{RESET}"
    tail = f"  {DIM}{detail}{RESET}" if detail else ""
    return f"  [{mark}] {label}{tail}"


def _note(label: str, detail: str = "") -> str:
    tail = f"  {DIM}{detail}{RESET}" if detail else ""
    return f"  [{YELLOW}--{RESET}] {label}{tail}"


# -- verifications --------------------------------------------------------

def check_dependencies() -> bool:
    ok = True
    for name, needed, role in DEPENDENCIES:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "?")
            print(_line(True, name, f"{version}  —  {role}"))
        except ImportError:
            if needed:
                print(_line(False, name, f"absent — indispensable ({role})"))
                ok = False
            else:
                print(_note(name, f"absent — {role} : indisponible"))
    return ok


def check_output_dir() -> bool:
    """Le dossier de sortie doit exister et etre inscriptible."""
    try:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = DEFAULT_OUTPUT_DIR / ".probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        print(_line(False, "dossier de sortie", f"{DEFAULT_OUTPUT_DIR} : {exc}"))
        return False
    print(_line(True, "dossier de sortie", str(DEFAULT_OUTPUT_DIR)))
    return True


def _imports_cleanly(modules, forbidden) -> tuple[bool, str]:
    """Importe des modules dans un interpreteur NEUF et regarde ce qu'ils tirent."""
    imports = "; ".join(f"import ramjet.{name}" for name in modules)
    banned = ", ".join(repr(name) for name in forbidden)
    code = (f"import sys; {imports}; "
            f"print(','.join(n for n in ({banned},) if n in sys.modules))")
    try:
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()
        return False, tail[-1] if tail else "import impossible"
    leaked = done.stdout.strip()
    if leaked:
        return False, f"ces modules ont importe {leaked}"
    return True, (f"{len(modules)} module(s), sans {', '.join(forbidden)}")


def core_imports_cleanly() -> tuple[bool, str]:
    return _imports_cleanly(CORE_MODULES, FORBIDDEN_IN_CORE)


def stdlib_only_holds() -> tuple[bool, str]:
    """La contrainte forte : « config » et « fuels » sans NumPy non plus."""
    return _imports_cleanly(STDLIB_ONLY_MODULES,
                            FORBIDDEN_IN_CORE + ("numpy",))


def check_config(verbose: bool = False) -> bool:
    """Construit les cas par defaut et verifie qu'ils refusent ce qu'il faut."""
    try:
        flight, engine = FlightCase(), EngineCase()
    except CaseError as exc:
        print(_line(False, "cas par defaut", str(exc)))
        return False
    print(_line(True, "cas de vol", flight.describe()))
    print(_line(True, "cas moteur", engine.describe()))

    refused = True
    cas_absurdes = (
        ("mach nul", FlightCase, dict(mach=0.0)),
        ("altitude hors modele", FlightCase, dict(altitude=50_000.0)),
        ("gamma inferieur a 1", FlightCase, dict(gamma=0.9)),
        ("T04 negative", EngineCase, dict(T04=-1.0)),
        ("rendement superieur a 1", EngineCase, dict(eta_b=1.2)),
        ("chambre supersonique", EngineCase, dict(M_combustor=1.5)),
        ("carburant inconnu", EngineCase, dict(fuel="eau")),
    )
    for label, classe, kwargs in cas_absurdes:
        try:
            classe(**kwargs)
        except CaseError:
            continue
        print(_line(False, "refus des cas absurdes", f"{label} a ete accepte"))
        refused = False
    if refused:
        print(_line(True, "refus des cas absurdes",
                    f"{len(cas_absurdes)} cas testes"))

    if verbose:
        print(f"  {DIM}carburants disponibles : "
              f"{', '.join(available_fuels())}{RESET}")
        print(f"  {DIM}cp deduit de gamma et R : {flight.cp:.1f} J/(kg.K){RESET}")
        print(f"  {DIM}recuperation du diffuseur a M0 = {flight.mach:g} : "
              f"{engine.inlet_recovery(flight.mach):.4f}{RESET}")
    return refused


def check_thermo(verbose: bool = False) -> bool:
    """LE CRITERE DE L'ETAPE 2 : l'aller-retour « M -> A/A* -> M ».

    C'est le seul endroit du module ou une iteration peut se tromper de
    branche, et une inversion fausse d'une branche donnerait un dimensionnement
    de section parfaitement plausible et faux. Le controle est ici plutot que
    dans les tests seuls parce qu'il ne coute rien et qu'il protege tout ce qui
    vient apres.
    """
    import numpy as np

    from .thermo import area_ratio, mach_from_area_ratio

    rng = np.random.default_rng(0)
    pire = 0.0
    for branche, lo, hi in (("subsonic", 0.05, 0.95), ("supersonic", 1.05, 6.0)):
        mach = rng.uniform(lo, hi, 200)
        retour = mach_from_area_ratio(area_ratio(mach), branche)
        ecart = float(np.max(np.abs(retour - mach)))
        pire = max(pire, ecart)
        if verbose:
            print(f"  {DIM}{branche} sur [{lo}, {hi}] : ecart max "
                  f"{ecart:.2e}{RESET}")
    ok = pire < INVERSION_TOL
    print(_line(ok, "inversion de A/A* sur les deux branches",
                f"ecart max {pire:.1e}, seuil {INVERSION_TOL:g}"))
    return ok


def check_atmosphere(verbose: bool = False) -> bool:
    """L'atmosphere contre les valeurs publiees aux ruptures de couche."""
    from .atmosphere import pressure, temperature

    publiees = ((0.0, 288.15, 101_325.0), (11_000.0, 216.65, 22_632.1),
                (20_000.0, 216.65, 5_474.89), (32_000.0, 228.65, 868.019))
    pire = 0.0
    for altitude, T, p in publiees:
        ecart = max(abs(temperature(altitude) - T) / T,
                    abs(pressure(altitude) - p) / p)
        pire = max(pire, ecart)
        if verbose:
            print(f"  {DIM}z = {altitude:6.0f} m : T = {temperature(altitude):7.2f} K, "
                  f"p = {pressure(altitude):9.2f} Pa   ecart {ecart:.1e}{RESET}")
    ok = pire < 1e-3
    print(_line(ok, "atmosphere standard ISA",
                f"ecart max {pire:.1e} sur {len(publiees)} altitudes"))
    return ok


def check_cycle(verbose: bool = False) -> bool:
    """LE CRITERE DE L'ETAPE 3 : le cycle contre sa forme fermee.

    Sans pertes, le calcul station par station doit retomber sur la solution
    analytique du statoreacteur ideal. C'est le seul controle EXACT du 0D, et
    il valide d'un coup les quatre stations, le bilan d'energie et la detente.
    """
    from .config import EngineCase, FlightCase
    from .cycle0d import ideal_cycle, ideal_engine, run_cycle

    pire = 0.0
    for mach in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        vol = FlightCase(mach=mach)
        moteur = ideal_engine(EngineCase())
        ferme = ideal_cycle(vol, moteur)
        stations = run_cycle(vol, moteur)
        ecart = abs(stations.specific_thrust - ferme.specific_thrust)
        pire = max(pire, ecart / abs(ferme.specific_thrust))
        if verbose:
            print(f"  {DIM}M0 = {mach}: F/mdot = {ferme.specific_thrust:8.3f} "
                  f"N.s/kg, M9 = {stations.mach_exit:.6f}, ecart "
                  f"{ecart:.1e}{RESET}")
    ok = pire < CYCLE_TOL
    print(_line(ok, "cycle ideal contre forme fermee",
                f"ecart relatif max {pire:.1e}, seuil {CYCLE_TOL:g}"))
    return ok


def check_choking(verbose: bool = False) -> bool:
    """LE CRITERE DE L'ETAPE 7 : la limite analytique et le refus qui va avec.

    Deux choses en une : la chaleur maximale se calcule AVANT d'integrer, et
    l'integration refuse effectivement au-dela. Un solveur qui rendrait un
    profil pour un ecoulement bloque produirait des courbes d'allure
    parfaitement ordinaire pour un moteur qui, en vrai, se desamorce.
    """
    from . import rayleigh
    from .duct import Duct, Zone
    from .solver1d import ChokedFlow, HeatRelease, integrate_duct

    mach_in, T0_in = 0.30, 300.0
    maximum = float(rayleigh.max_total_temperature_ratio(mach_in))
    tube = Duct((Zone("chambre", 0.0, 1.0, 0.02, 0.02),))

    passe = bloque = None
    try:
        profil = integrate_duct(tube, HeatRelease(T0_in, T0_in * maximum * 0.95,
                                                  0.0, 1.0), mach_in, 5.0)
        passe = float(profil.mach[-1])
    except ChokedFlow:
        passe = None
    try:
        integrate_duct(tube, HeatRelease(T0_in, T0_in * maximum * 1.05, 0.0, 1.0),
                       mach_in, 5.0)
    except ChokedFlow as erreur:
        bloque = erreur.x

    ok = passe is not None and passe < 1.0 and bloque is not None
    if verbose:
        print(f"  {DIM}M = {mach_in:g} : T0max/T0in = {maximum:.4f}, "
              f"q_max = {float(rayleigh.max_heat(mach_in, T0_in)) / 1e3:.1f} "
              f"kJ/kg{RESET}")
        if passe is not None:
            print(f"  {DIM}a 95 % du maximum : M de sortie = {passe:.4f}{RESET}")
    detail = (f"limite {maximum:.4f} ; sous elle M sort a {passe:.4f}, "
              f"au-dela blocage a x = {bloque:.4f} m"
              if ok else "le solveur n'a pas refuse un cas bloque")
    print(_line(ok, "blocage thermique detecte", detail))
    return ok


def check_friction(verbose: bool = False) -> bool:
    """L'ETAPE 12 : le troisieme terme, et son oracle.

    Le solveur a maintenant trois termes source et trois formes fermees pour
    les verifier separement — isentropique, Rayleigh, Fanno. C'est ce qui
    permet de savoir LEQUEL est faux quand un profil deraille.
    """
    import numpy as np

    from . import fanno
    from .duct import Duct, Zone
    from .solver1d import Friction, integrate_duct, momentum_residual, no_heat

    diametre, coefficient, longueur = 0.16, 0.004, 6.5
    section = float(np.pi * diametre ** 2 / 4.0)
    frottement = Friction(coefficient)
    conduit = Duct((Zone("tube", 0.0, longueur, section, section),))
    profil = integrate_duct(conduit, no_heat(300.0, longueur), 0.30, 5.0,
                            friction=frottement, n_points=201)
    exact = fanno.mach_after(0.30, longueur, diametre, coefficient)
    ecart = abs(float(profil.mach[-1]) - exact)
    residu = momentum_residual(profil, conduit, frottement)

    ok = ecart < 1e-12 and residu < 1e-12
    if verbose:
        identite = float(np.max(np.abs(
            fanno.total_pressure_ratio(np.array([0.2, 0.9, 3.0]))
            - __import__("ramjet.thermo", fromlist=["x"]).area_ratio(
                np.array([0.2, 0.9, 3.0])))))
        print(f"  {DIM}p0/p0* de Fanno == A/A* de l'isentropique : "
              f"ecart {identite:.1e}{RESET}")
        print(f"  {DIM}le frottement ACCELERE le subsonique : M 0,300 -> "
              f"{float(profil.mach[-1]):.4f} sur {longueur:g} m{RESET}")
    detail = (f"EDO contre Fanno : {ecart:.1e} ; quantite de mouvement avec "
              f"trainee parietale : {residu:.1e}"
              if ok else "le troisieme coefficient d'influence est faux")
    print(_line(ok, "frottement contre Fanno", detail))
    return ok


def check_caloric(verbose: bool = False) -> bool:
    """L'ETAPE 14 : ce que « cp constant » coute, chiffre.

    Le controle porte d'abord sur les DONNEES — cinquante coefficients recopies
    — par le raccord des deux polynomes a 1 000 K, qu'une seule faute de frappe
    casserait. Puis sur le resultat : a « f » donne, T04 tombe de 256 K.
    """
    from . import caloric
    from .cycle0d import run_cycle
    from .fuels import get as get_fuel

    vol, moteur = FlightCase(), EngineCase()
    cycle = run_cycle(vol, moteur)
    kerosene = get_fuel("kerosene")

    pire = 0.0
    for espece, (bas, haut) in caloric.NASA.items():
        def chaleur(a):
            T = caloric.T_JUNCTION
            return (a[0] + a[1] * T + a[2] * T ** 2 + a[3] * T ** 3
                    + a[4] * T ** 4)
        pire = max(pire, abs(chaleur(bas) / chaleur(haut) - 1.0))

    reel = caloric.combustion_temperature(kerosene, cycle.fuel_air_ratio,
                                          cycle.station("2").T0, moteur.eta_b)
    ecart = reel - moteur.T04
    # La flamme adiabatique depuis 298 K : ~2 390 K publies, sans dissociation.
    stoechio = caloric.stoichiometric_ratio(kerosene) * (1.0 - 1e-9)
    flamme = caloric.combustion_temperature(kerosene, stoechio, 298.15, 1.0)
    ok = (pire < 1e-5 and -300.0 < ecart < -200.0
          and abs(flamme / 2390.0 - 1.0) < 0.02)
    if verbose:
        air = caloric.air()
        print(f"  {DIM}air : cp {float(air.cp(300.0)):.1f} a 300 K, "
              f"{float(air.cp(2000.0)):.1f} a 2000 K ; gamma "
              f"{float(air.gamma(300.0)):.4f} -> "
              f"{float(air.gamma(2000.0)):.4f}{RESET}")
    detail = (f"raccord des polynomes a 1000 K : {pire:.1e} ; flamme "
              f"adiabatique {flamme:.0f} K (2390 publies) ; a f donne T04 vaut "
              f"{reel:.0f} K et non {moteur.T04:g} ({ecart:+.0f} K)"
              if ok else "les donnees ou le bilan d'enthalpie sont faux")
    print(_line(ok, "cp(T) et gamma(T)", detail))
    return ok


def check_engine(verbose: bool = False) -> bool:
    """LE CRITERE DE L'ETAPE 8 : le col est sonique et la poussee se retrouve.

    Trois choses qu'aucun des deux modeles ne pouvait dire seul. Le col est
    sonique — sans quoi la tuyere reste subsonique et le moteur ne pousse pas.
    La poussee calculee par un bilan de QUANTITE DE MOUVEMENT sur le profil 1D
    retrouve celle que le 0D tire d'un bilan d'ENTHALPIE : deux chemins qui ne
    partagent rien. Et le debit n'est pas celui qu'on avait demande, parce que
    c'est le col qui le fixe.
    """
    from .coupling import run_engine

    moteur = run_engine()
    ecart = abs(moteur.specific_thrust / moteur.cycle.specific_thrust - 1.0)
    ok = (abs(moteur.throat_mach - 1.0) < 1e-3
          and float(moteur.profile.mach[-1]) > 1.0
          and ecart < 5e-3)
    if verbose:
        print(f"  {DIM}debit demande 10,000 kg/s, impose par le col "
              f"{moteur.mass_flow:.3f}{RESET}")
        print(f"  {DIM}pi_b suppose {moteur.cycle.engine.pi_b:.4f}, subi "
              f"{moteur.combustor_pressure_ratio:.4f}{RESET}")
        print(f"  {DIM}p9 / p_ambiant = {moteur.exit_pressure_ratio:.4f}"
              f"{RESET}")
    detail = (f"M au col {moteur.throat_mach:.6f}, M9 "
              f"{float(moteur.profile.mach[-1]):.4f} ; F/mdot "
              f"{moteur.specific_thrust:.1f} contre "
              f"{moteur.cycle.specific_thrust:.1f} du 0D ({100 * ecart:.2f} %)"
              if ok else "l'assemblage 0D + geometrie + 1D ne tient pas")
    print(_line(ok, "statoreacteur complet", detail))
    return ok


def check_coupling(verbose: bool = False) -> bool:
    """LE CRITERE DE L'ETAPE 9 : la boucle converge, et elle corrige.

    Deux exigences, et la seconde est la vraie. Converger ne prouve rien si le
    point fixe est celui qu'on avait suppose ; ce qui compte est que le pi_b
    reellement subi DIFFERE de celui du 0D, sans quoi la boucle serait un
    ornement.
    """
    from .coupling import converge

    point = converge()
    ok = (point.calls <= 6 and point.residual < 1e-12
          and abs(point.correction) > 0.01)
    if verbose:
        chemin = " -> ".join(f"{v:.6f}" for v in point.history)
        print(f"  {DIM}pi_b : {chemin}{RESET}")
        print(f"  {DIM}marge avant blocage {point.choking_margin():.0f} K ; "
              f"debit {point.engine.mass_flow:.3f} kg/s{RESET}")
    detail = (f"pi_b {point.assumed_pressure_ratio:.4f} suppose -> "
              f"{point.combustor_pressure_ratio:.4f} subi "
              f"({100 * point.correction:+.2f} %) en {point.calls} appels, "
              f"residu {point.residual:.1e}"
              if ok else "la boucle 0D <-> 1D n'a pas tenu ses criteres")
    print(_line(ok, "boucle 0D <-> 1D", detail))
    return ok


def check_inlet(verbose: bool = False) -> bool:
    """L'ETAPE 13 : ce que la correlation du 0D decrivait, enfin nomme.

    Tout le cycle repose sur « pi_d = 0,8088 » a M0 = 3, tire de MIL-E-5008B,
    qui parle d'une entree « bien dessinee » sans dire laquelle. En calculant la
    recuperation de rampes optimales, on peut repondre : trois rampes.
    """
    import numpy as np

    from .thermo import normal_shock, optimal_ramp_angles, ramp_inlet

    mach = 3.0
    mil = EngineCase().inlet_recovery(mach)
    droit = float(normal_shock(mach).total_pressure_ratio)
    recuperations = [ramp_inlet(mach, optimal_ramp_angles(mach, n)
                                ).total_pressure_ratio for n in (1, 2, 3, 4)]
    proche = 1 + int(np.argmin([abs(r - mil) for r in recuperations]))

    ok = proche == 3 and droit < recuperations[0] < recuperations[-1]
    if verbose:
        entree = ramp_inlet(mach, optimal_ramp_angles(mach, 3))
        print(f"  {DIM}{entree.describe().replace(chr(10), chr(10) + '  ' + DIM)}"
              f"{RESET}")
    detail = (f"a M0 = 3, MIL donne {mil:.4f} ; choc droit {droit:.4f}, "
              f"{proche} rampes optimales {recuperations[proche - 1]:.4f} "
              f"({100 * (mil / recuperations[proche - 1] - 1):+.1f} %)"
              if ok else "l'identification de l'entree d'air a echoue")
    print(_line(ok, "entree a chocs obliques", detail))
    return ok


# -- les figures ------------------------------------------------------------

FIGURES = ("profils", "conduit", "cycle_ts", "balayage", "blocage",
           "adaptation")


def _etats_statiques(cycle, mach_chambre: float, mach_sortie_chambre: float):
    """Les quatre stations du cycle 0D, en grandeurs STATIQUES.

    POURQUOI PAS EN TOTALES. Une tuyere ne change pas la temperature totale :
    en totales, les stations 4 et 9 se confondent et la detente disparait du
    diagramme sans que rien ne le signale. Les Mach viennent du 0D lui-meme —
    M0 pour la station 0, le Mach de chambre pour la 2, celui de fin de chambre
    par conservation de la masse pour la 4, et le M9 du cycle pour la 9.
    """
    from .thermo import total_pressure_ratio, total_temperature_ratio

    gamma = cycle.flight.gamma
    machs = {"0": cycle.flight.mach, "2": mach_chambre,
             "4": mach_sortie_chambre, "9": cycle.mach_exit}
    noms, temperatures, pressions = [], [], []
    for station in cycle.stations:
        mach = machs[station.name]
        noms.append(station.name)
        temperatures.append(station.T0
                            / float(total_temperature_ratio(mach, gamma)))
        pressions.append(station.p0
                         / float(total_pressure_ratio(mach, gamma)))
    return noms, temperatures, pressions


def figures(output: Path, verbose: bool = False) -> int:
    """Les cinq figures du plan, depuis le point de fonctionnement converge."""
    import numpy as np

    from . import plots, rayleigh
    from .coupling import (SEPARATION_RATIO, _chamber_inlet_mach,
                           converge, flight_envelope)
    from .cycle0d import (PUBLISHED_SPECIFIC_IMPULSE,
                          PUBLISHED_SPECIFIC_THRUST, sweep_mach)
    from .duct import chamber_exit_mach

    print(f"{BOLD}ramjet-1d {__version__}{RESET}  figures")
    point = converge()
    moteur = point.engine
    cycle, conduit = moteur.cycle, moteur.duct
    profil = moteur.profile
    zones = [(z.name, z.x_start, z.x_end) for z in conduit.zones]

    ecrits = []
    ecrits.append(plots.save(plots.profile_figure(
        profil.x, profil.mach, profil.pressure, profil.temperature,
        profil.velocity, zones=zones, throat_x=conduit.throat_x),
        output / "profils.png"))
    ecrits.append(plots.save(plots.duct_figure(
        profil.x, profil.area, zones=zones, throat_x=conduit.throat_x),
        output / "conduit.png"))

    noms, temperatures, pressions = _etats_statiques(
        cycle, cycle.engine.M_combustor,
        chamber_exit_mach(conduit, cycle, moteur.mass_flow,
                          include_fuel_mass=False))
    ecrits.append(plots.save(plots.cycle_figure(
        noms, temperatures, pressions, cp=cycle.flight.cp, R=cycle.flight.R),
        output / "cycle_ts.png"))

    balayage = sweep_mach(np.linspace(1.5, 5.0, 80), cycle.engine)
    ecrits.append(plots.save(plots.sweep_figure(
        balayage.machs, balayage.specific_thrust, balayage.specific_impulse,
        thrust_band=PUBLISHED_SPECIFIC_THRUST,
        impulse_band=PUBLISHED_SPECIFIC_IMPULSE,
        best=balayage.best_thrust()), output / "balayage.png"))

    T02, p02 = cycle.station("2").T0, cycle.station("2").p0
    reel = _chamber_inlet_mach(moteur.mass_flow, conduit, T02, p02,
                               cycle.flight.gamma, cycle.flight.R)
    machs = np.linspace(0.05, 0.60, 240)
    ecrits.append(plots.save(plots.choking_figure(
        machs, rayleigh.critical_total_temperature(machs, T02),
        operating=(reel, cycle.station("4").T0),
        design_mach=cycle.engine.M_combustor), output / "blocage.png"))

    par_mach, par_altitude, _ = flight_envelope(
        conduit, machs=np.linspace(2.0, 4.0, 17),
        altitudes=np.linspace(0.0, 25000.0, 11),
        engine=cycle.engine)
    ecrits.append(plots.save(plots.adaptation_figure(
        [m for m, _ in par_mach], [a.loss for _, a in par_mach],
        [z for z, _ in par_altitude], [a.loss for _, a in par_altitude],
        pressure_ratio_by_mach=[a.exit_pressure_ratio for _, a in par_mach],
        separation=SEPARATION_RATIO, design_mach=cycle.flight.mach),
        output / "adaptation.png"))

    for chemin in ecrits:
        taille = chemin.stat().st_size / 1024
        print(_line(taille > 10.0, chemin.name, f"{taille:.0f} ko"))
    if verbose:
        print(f"  {DIM}Mach de chambre : {cycle.engine.M_combustor:g} vise, "
              f"{reel:.4f} reel{RESET}")
    print()
    print(f"{GREEN}{len(ecrits)} figures dans {output}{RESET}")
    return EXIT_OK


def check_adaptation(verbose: bool = False) -> bool:
    """L'ETAPE 15 : l'adaptation depend du MACH, pas de l'altitude.

    Le controle porte sur le fait contre-intuitif lui-meme : a geometrie figee
    et Mach fixe, le rapport « p9/p_amb » est rigoureusement le meme a 11, 15 et
    20 km — parce que la tropopause est isotherme et que la compression
    dynamique rend « p04 » proportionnelle a l'ambiante.
    """
    from .coupling import SEPARATION_RATIO, adaptation, converge, run_engine

    point = converge()
    conduit = point.engine.duct
    altitudes = [adaptation(run_engine(FlightCase(mach=3.0, altitude=z),
                                       EngineCase(), duct=conduit))
                 for z in (11000.0, 15000.0, 20000.0)]
    ecart = max(abs(a.exit_pressure_ratio - altitudes[0].exit_pressure_ratio)
                for a in altitudes)
    bas = adaptation(run_engine(FlightCase(mach=2.0), EngineCase(),
                                duct=conduit))
    haut = adaptation(run_engine(FlightCase(mach=4.0), EngineCase(),
                                 duct=conduit))

    ok = (ecart < 1e-12 and bas.separated and not haut.separated
          and bas.loss > 0.1 and haut.loss > 0.05)
    if verbose:
        print(f"  {DIM}M0 = 2 : p9/p_amb = {bas.exit_pressure_ratio:.4f} "
              f"(sous le seuil de decollement {SEPARATION_RATIO:g}) ; "
              f"M0 = 4 : {haut.exit_pressure_ratio:.4f}{RESET}")
    detail = (f"invariante en altitude a {ecart:.0e} sur 9 km ; perte "
              f"{100 * bas.loss:.1f} % a M0 = 2 et "
              f"{100 * haut.loss:.1f} % a M0 = 4"
              if ok else "l'analyse d'adaptation ne tient pas")
    print(_line(ok, "adaptation hors dessin", detail))
    return ok


def check_figure(verbose: bool = False) -> bool:
    """Le plan demande qu'une FIGURE soit produite pendant « --check ».

    Une figure qui ne se rend pas est une panne qu'aucun test de calcul
    n'attrape : le pipeline complet ne se verifie qu'en allant jusqu'au PNG.
    On ecrit dans un dossier temporaire, pour ne pas salir « sorties/ » avec un
    sous-produit de verification.
    """
    import tempfile

    import numpy as np

    from . import plots

    x = np.linspace(0.0, 1.5, 60)
    figure = plots.duct_figure(x, 0.027 - 0.009 * np.exp(-((x - 1.15) / 0.2) ** 2),
                               zones=[("chamber", 0.4, 1.0)], throat_x=1.15)
    with tempfile.TemporaryDirectory() as dossier:
        chemin = plots.save(figure, Path(dossier) / "essai.png")
        taille = chemin.stat().st_size
    ok = taille > 10_000
    if verbose:
        print(f"  {DIM}rendu dans un dossier temporaire, pyplot jamais "
              f"charge{RESET}")
    detail = (f"{taille / 1024:.0f} ko rendus par l'API objet de Matplotlib"
              if ok else f"figure trop petite ({taille} octets)")
    print(_line(ok, "rendu d'une figure", detail))
    return ok


def check(verbose: bool = False) -> int:
    print(f"{BOLD}ramjet-1d {__version__}{RESET}")
    ok = check_dependencies()
    ok = check_output_dir() and ok

    clean, detail = core_imports_cleanly()
    print(_line(clean, "cloisonnement des dependances", detail))
    ok = clean and ok

    stdlib, detail = stdlib_only_holds()
    print(_line(stdlib, "config et fuels sans NumPy", detail))
    ok = stdlib and ok

    ok = check_config(verbose) and ok
    ok = check_atmosphere(verbose) and ok
    ok = check_thermo(verbose) and ok
    ok = check_cycle(verbose) and ok
    ok = check_choking(verbose) and ok
    ok = check_friction(verbose) and ok
    ok = check_caloric(verbose) and ok
    ok = check_engine(verbose) and ok
    ok = check_coupling(verbose) and ok
    ok = check_inlet(verbose) and ok
    ok = check_adaptation(verbose) and ok
    ok = check_figure(verbose) and ok

    print()
    if not ok:
        print(f"{RED}Des verifications ont echoue.{RESET}")
        return EXIT_ERROR
    print(f"{GREEN}Tout est en place.{RESET}")
    return EXIT_OK


# -- les cas et les modes de calcul -----------------------------------------

def build_cases(args) -> tuple[FlightCase, EngineCase]:
    """Les deux dataclasses, depuis la ligne de commande.

    Les valeurs non fournies restent celles des defauts, qui sont dans
    « config.py » et pas ici : la ligne de commande ne doit pas devenir une
    seconde source de verite sur le cas de reference.
    """
    defaut_vol, defaut_moteur = FlightCase(), EngineCase()
    vol = FlightCase(
        mach=defaut_vol.mach if args.mach is None else args.mach,
        altitude=(defaut_vol.altitude if args.altitude is None
                  else args.altitude))
    moteur = EngineCase(
        T04=defaut_moteur.T04 if args.t04 is None else args.t04,
        fuel=defaut_moteur.fuel if args.fuel is None else args.fuel,
        M_combustor=(defaut_moteur.M_combustor if args.m_comb is None
                     else args.m_comb),
        inlet_ramps=args.rampes)
    return vol, moteur


def _diagnostic_bloque(flight: FlightCase, engine: EngineCase,
                       erreur: Exception) -> str:
    """Le message d'un cas refuse : il doit nommer le Mach ET donner le maximum.

    ET IL EN DONNE DEUX, PARCE QUE L'ETAPE 9 A MONTRE QUE LE PREMIER NE SUFFIT
    PAS. Le Mach de chambre VISE fixe une limite de Rayleigh — c'est le nombre
    qu'on attend — mais la chambre ne tourne jamais a ce Mach : le col fixe le
    debit, donc le Mach reel, et la limite applicable est la sienne. Annoncer
    seulement le premier ferait chercher le probleme au mauvais endroit.
    """
    from . import rayleigh
    from .coupling import _chamber_inlet_mach, choking_mass_flow
    from .cycle0d import run_cycle
    from .duct import from_cycle

    lignes = [f"{erreur}"]
    try:
        cycle = run_cycle(flight, engine)
        T02 = cycle.station("2").T0
        vise = float(rayleigh.critical_total_temperature(engine.M_combustor,
                                                         T02))
        lignes.append(
            f"Au Mach de chambre VISE ({engine.M_combustor:g}), la limite de "
            f"Rayleigh est T04 = {vise:.0f} K ; vous demandez "
            f"{engine.T04:g} K.")
        conduit = from_cycle(cycle, mass_flow=10.0)
        debit = choking_mass_flow(conduit, T02, cycle.station("2").p0,
                                  engine.T04, flight.gamma, flight.R)
        reel = _chamber_inlet_mach(debit, conduit, T02, cycle.station("2").p0,
                                   flight.gamma, flight.R)
        applicable = float(rayleigh.critical_total_temperature(reel, T02))
        lignes.append(
            f"Mais le col fixe le debit, donc le Mach REEL de chambre : "
            f"{reel:.4f}, ou la limite vaut {applicable:.0f} K. C'est celle-la "
            f"qui s'applique.")
    except CaseError:
        lignes.append("Le Mach reel de chambre n'a pas pu etre calcule : la "
                      "geometrie elle-meme ne tient pas.")
    return "\n".join(lignes)


def run_case(args) -> int:
    """Le mode par defaut : converger, rapporter, ecrire."""
    from .coupling import converge
    from .report import case_slug, report, write

    vol, moteur = build_cases(args)
    point = converge(vol, moteur)
    texte = report(point)
    print(texte)

    if not args.dry_run:
        chemin = write(texte, args.output / case_slug(vol, moteur)
                       / "rapport.txt")
        print(f"{GREEN}Rapport ecrit dans {chemin}{RESET}")
    else:
        print(f"{YELLOW}--dry-run : rien n'a ete ecrit.{RESET}")

    from .report import identities
    return EXIT_OK if all(i.ok for i in identities(point)) else EXIT_FAILED


def run_sweep(args, avec_validation: bool = False) -> int:
    """Le balayage en Mach, et — si demande — sa confrontation aux fourchettes."""
    import numpy as np

    from .cycle0d import (PUBLISHED_SPECIFIC_IMPULSE,
                          PUBLISHED_SPECIFIC_THRUST, sweep_mach)
    from .report import sweep_table, write

    vol, moteur = build_cases(args)
    bas, haut = args.range
    balayage = sweep_mach(np.linspace(bas, haut, args.points), moteur,
                          altitude=vol.altitude, gamma=vol.gamma, R=vol.R)
    lignes = sweep_table(balayage)
    print(f"{BOLD}ramjet-1d {__version__}{RESET}  balayage en Mach de vol")
    print("\n".join(lignes))

    if not avec_validation:
        if not args.dry_run:
            write("\n".join(lignes) + "\n", args.output / "balayage.txt")
        return EXIT_OK

    # LA VALIDATION EST UN ACCORD D'ORDRE DE GRANDEUR, ET LE DOMAINE OU ON
    # L'EXIGE EST CELUI DE L'ETAPE 4 : M0 de 2 a 3,5. Au-dela, la poussee sort
    # de la fourchette pour une raison physique — l'air arrive trop chaud pour
    # qu'on puisse encore le chauffer — et non parce que le modele derive.
    print()
    dedans = ((balayage.machs >= 2.0) & (balayage.machs <= 3.5))
    poussees = balayage.specific_thrust[dedans]
    impulsions = balayage.specific_impulse[dedans]
    controles = (
        ("poussee specifique sur M0 = 2 a 3,5", poussees,
         PUBLISHED_SPECIFIC_THRUST),
        ("impulsion specifique sur M0 = 2 a 3,5", impulsions,
         PUBLISHED_SPECIFIC_IMPULSE),
    )
    ok = True
    for nom, valeurs, (lo, hi) in controles:
        tient = bool(valeurs.size) and bool(np.all((valeurs >= lo)
                                                   & (valeurs <= hi)))
        ok = tient and ok
        print(_line(tient, nom,
                    f"{float(np.min(valeurs)):.1f} a "
                    f"{float(np.max(valeurs)):.1f}, fourchette {lo:.0f}-{hi:.0f}"
                    if valeurs.size else "aucun point dans le domaine"))

    from .coupling import converge
    point = converge(vol, moteur)
    boucle = abs(point.correction) > 0.01
    ok = boucle and ok
    print(_line(boucle, "la boucle 0D <-> 1D corrige quelque chose",
                f"pi_b {point.assumed_pressure_ratio:.4f} suppose -> "
                f"{point.combustor_pressure_ratio:.4f} subi "
                f"({100 * point.correction:+.2f} %)"))

    if not args.dry_run:
        chemin = write("\n".join(lignes) + "\n",
                       args.output / "validation.txt")
        print(f"\n{GREEN}Table ecrite dans {chemin}{RESET}")
    print()
    if ok:
        print(f"{GREEN}Le modele tient dans les fourchettes publiees.{RESET}")
        return EXIT_OK
    print(f"{RED}Le modele sort des fourchettes publiees.{RESET}")
    return EXIT_FAILED


# -- point d'entree -------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check:
            return check(args.verbose)
        if args.figures:
            return figures(args.output, args.verbose)
        if args.validate:
            return run_sweep(args, avec_validation=True)
        if args.sweep:
            return run_sweep(args)
        return run_case(args)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrompu.{RESET}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except ChokedFlow as exc:
        print(f"{RED}{_diagnostic_bloque(*build_cases(args), exc)}{RESET}",
              file=sys.stderr)
        return EXIT_FAILED
    except CaseError as exc:
        # UN CAS REFUSE PAR LA PHYSIQUE N'EST PAS UNE ERREUR D'USAGE. Le
        # blocage et l'effondrement du dessin sortent en 2 — « critere
        # numerique non atteint » — et non en 1 : le programme a fonctionne, et
        # c'est le point de fonctionnement qui n'existe pas.
        from .coupling import DesignCollapse
        if isinstance(exc, DesignCollapse):
            print(f"{RED}{_diagnostic_bloque(*build_cases(args), exc)}{RESET}",
                  file=sys.stderr)
            return EXIT_FAILED
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError) as exc:
        print(f"{RED}{type(exc).__name__} : {exc}{RESET}", file=sys.stderr)
        return EXIT_ERROR
