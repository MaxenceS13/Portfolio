"""Les deux cas d'un calcul : ce que voit le moteur, et ce que le moteur fait.

POURQUOI DEUX OBJETS ET PAS UN. « FlightCase » decrit le point de vol — Mach,
altitude, gaz — et « EngineCase » decrit le moteur — temperature de fin de
combustion, carburant, rendements. La separation n'est pas cosmetique : le
balayage en Mach de l'etape 4 fait varier le premier a moteur FIXE, et l'etude
de sensibilite aux pertes fait varier le second a vol fixe. Un objet unique
obligerait a recopier la moitie des champs a chaque variation, et c'est
exactement la ou les erreurs de recopie se logent.

GELES, DONC HACHABLES, DONC CACHABLES. La boucle de couplage 0D/1D de l'etape 9
resout le meme cas plusieurs fois ; le balayage de l'etape 4 en resout des
milliers. Les deux types sont geles et normalises, si bien que deux cas
EQUIVALENTS ont le meme hachage : « FlightCase(mach=3) » et
« FlightCase(mach=3.0) » sont le meme objet pour un cache. Sans la
normalisation, un entier et un flottant donneraient deux cles differentes pour
le meme calcul.

CE MODULE RESTE EN BIBLIOTHEQUE STANDARD, ET C'EST TESTE. Ni NumPy, ni SciPy,
ni Matplotlib. Il est importe par la ligne de commande, par les deux modeles et
par tous les tests ; le jour ou il tire NumPy, il tire tout le paquet avec lui,
et un script d'optimisation exterieur qui ne veut construire que des cas paie
une seconde d'importation pour rien.

LA VALIDATION EST ENTIERE ICI, SANS « bind » DIFFERE. C'est un ecart au plan,
et il vient d'une difference reelle avec « chaos-attractors » : la-bas, le
registre des systemes tirait NumPy, donc la configuration ne pouvait pas le
consulter et devait remettre la moitie de ses controles a plus tard. Ici le
registre des carburants est en bibliotheque standard, donc le carburant se
verifie a la construction, comme le plan le demandait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace as _replace
from functools import lru_cache

from .fuels import Fuel, FuelError
from .fuels import get as get_fuel

# Plafond d'altitude du modele d'atmosphere de l'etape 2 : au-dela, les couches
# ne sont plus decrites et extrapoler donnerait des nombres sans support.
MAX_ALTITUDE = 32_000.0

# Sous ce Mach, la compression dynamique ne suffit pas a faire fonctionner un
# statoreacteur : il n'a pas de compresseur, donc pas de poussee au point fixe.
# La borne n'interdit pas de calculer, elle interdit de le faire en silence.
MIN_USEFUL_MACH = 0.5

# Au-dela, une entree a rampes n'a plus de sens pratique : chaque rampe ajoutee
# rapporte de moins en moins (mesure a l'etape 13 : +0,25 puis +0,16 puis +0,09
# puis +0,05 de recuperation) et coute de la longueur, de la masse et de la
# trainee externe que ce modele ne compte pas.
MAX_RAMPS = 6


class CaseError(ValueError):
    """Parametres incoherents, detectes avant le moindre calcul."""


def _as_float(field: str, value: object) -> float:
    """Convertit en flottant fini, ou refuse en nommant le champ."""
    try:
        out = float(value)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise CaseError(f"{field} doit etre un nombre (recu {value!r})") from None
    if not math.isfinite(out):
        # « nan <= 0 » est faux : sans ce controle, un NaN traverserait toutes
        # les bornes ci-dessous sans en declencher une seule.
        raise CaseError(f"{field} doit etre un nombre fini (recu {out})")
    return out


def _as_ratio(field: str, value: object) -> float:
    """Un rendement ou un rapport de pression : dans ]0, 1]."""
    out = _as_float(field, value)
    if not 0.0 < out <= 1.0:
        raise CaseError(
            f"{field} doit etre dans ]0, 1] (recu {out:g}) : c'est un rapport "
            f"de pression totale ou un rendement, jamais un gain"
        )
    return out


@lru_cache(maxsize=256)
def _ramp_recovery(mach: float, ramps: int) -> float:
    """La recuperation d'une entree a « ramps » rampes optimales, mise en cache.

    ELLE COUTE 190 ms, ET UN BALAYAGE EN APPELLE QUINZE. Le calcul enchaine une
    section doree sur la force commune des chocs et une bissection par choc ;
    sans cache, « --sweep » y passerait trois secondes pour recalculer quinze
    fois des valeurs qui ne dependent que de (Mach, nombre de rampes). Le cache
    est correct parce que la fonction est pure et que ses deux arguments sont
    hachables.
    """
    from .thermo import optimal_ramp_angles, ramp_inlet

    angles = optimal_ramp_angles(mach, ramps)
    return float(ramp_inlet(mach, angles).total_pressure_ratio)


@dataclass(frozen=True)
class FlightCase:
    """Le point de vol : ce que le moteur voit arriver."""

    mach: float = 3.0
    altitude: float = 11_000.0        # m
    gamma: float = 1.4                # rapport des chaleurs specifiques
    R: float = 287.05                 # constante du gaz, J/(kg.K)

    def __post_init__(self) -> None:
        for field in ("mach", "altitude", "gamma", "R"):
            object.__setattr__(self, field, _as_float(field, getattr(self, field)))

        if self.mach <= 0.0:
            raise CaseError(f"mach doit etre positif (recu {self.mach:g})")
        if not 0.0 <= self.altitude <= MAX_ALTITUDE:
            raise CaseError(
                f"altitude doit etre dans [0, {MAX_ALTITUDE:g}] m "
                f"(recu {self.altitude:g}) : au-dela, le modele d'atmosphere "
                f"n'a plus de couche a decrire"
            )
        if not 1.0 < self.gamma < 2.0:
            raise CaseError(
                f"gamma doit etre dans ]1, 2[ (recu {self.gamma:g})"
            )
        if self.R <= 0.0:
            raise CaseError(f"R doit etre positif (recu {self.R:g})")

    @property
    def cp(self) -> float:
        """Chaleur specifique a pression constante, deduite de gamma et R.

        Deduite plutot que donnee : « cp = gamma R / (gamma - 1) » est une
        identite, et laisser l'utilisateur fournir les trois permettrait d'en
        donner trois incoherents.
        """
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def is_useful(self) -> bool:
        """Faux sous le Mach ou un statoreacteur n'a pas de sens physique."""
        return self.mach >= MIN_USEFUL_MACH

    def describe(self) -> str:
        return (f"vol   M0 = {self.mach:g}   z = {self.altitude:g} m   "
                f"gamma = {self.gamma:g}   R = {self.R:g}")


@dataclass(frozen=True)
class EngineCase:
    """Le moteur : ce qu'il fait de l'air qu'il recoit.

    « pi_d = None » signifie « prends la correlation MIL-E-5008B », qui depend
    du Mach de vol et n'a donc pas sa place dans un objet qui decrit le moteur
    seul. C'est la seule facon honnete de ranger une valeur qui depend de
    l'autre cas : ne pas la ranger.
    """

    T04: float = 2000.0               # temperature de fin de combustion, K
    fuel: str = "kerosene"
    eta_b: float = 0.98               # rendement de combustion
    pi_d: float | None = None         # None = correlation MIL-E-5008B
    pi_b: float = 0.95                # perte de pression totale en chambre
    pi_n: float = 0.98                # perte de pression totale en tuyere
    M_combustor: float = 0.20         # Mach vise a l'entree de la chambre
    inlet_ramps: int | None = None    # nombre de rampes de l'entree d'air

    def __post_init__(self) -> None:
        object.__setattr__(self, "T04", _as_float("T04", self.T04))
        object.__setattr__(self, "eta_b", _as_ratio("eta_b", self.eta_b))
        object.__setattr__(self, "pi_b", _as_ratio("pi_b", self.pi_b))
        object.__setattr__(self, "pi_n", _as_ratio("pi_n", self.pi_n))
        object.__setattr__(self, "M_combustor",
                           _as_float("M_combustor", self.M_combustor))
        if self.pi_d is not None:
            object.__setattr__(self, "pi_d", _as_ratio("pi_d", self.pi_d))

        if not isinstance(self.fuel, str) or not self.fuel:
            raise CaseError(f"fuel doit etre un nom non vide (recu {self.fuel!r})")
        try:
            get_fuel(self.fuel)
        except FuelError as erreur:
            raise CaseError(str(erreur)) from None

        if self.T04 <= 0.0:
            raise CaseError(f"T04 doit etre positive (recu {self.T04:g} K)")
        if not 0.0 < self.M_combustor < 1.0:
            raise CaseError(
                f"M_combustor doit etre dans ]0, 1[ (recu {self.M_combustor:g}) : "
                f"une chambre supersonique est un autre moteur, hors perimetre"
            )
        if self.inlet_ramps is not None:
            if self.pi_d is not None:
                raise CaseError(
                    "pi_d et inlet_ramps demandent tous deux la recuperation "
                    "d'entree : donnez l'un OU l'autre, pas les deux"
                )
            if not isinstance(self.inlet_ramps, int):
                raise CaseError(
                    f"inlet_ramps doit etre un entier "
                    f"(recu {self.inlet_ramps!r})"
                )
            if not 1 <= self.inlet_ramps <= MAX_RAMPS:
                raise CaseError(
                    f"inlet_ramps doit etre dans [1 ; {MAX_RAMPS}] "
                    f"(recu {self.inlet_ramps}) : au-dela, chaque rampe "
                    f"rapporte moins qu'elle ne coute en longueur et en masse"
                )

    def replace(self, **changes) -> "EngineCase":
        """Le meme cas, un champ change. Les controles du constructeur
        rejouent : la boucle de l'etape 9 s'en sert a chaque tour."""
        return _replace(self, **changes)

    @property
    def fuel_data(self) -> Fuel:
        """Le carburant lui-meme. Le nom est valide a la construction."""
        return get_fuel(self.fuel)

    def inlet_recovery(self, mach: float) -> float:
        """Recuperation de pression totale du diffuseur. TROIS MODELES.

        Par ordre de priorite : la valeur imposee « pi_d » si elle est donnee ;
        sinon l'entree a rampes si « inlet_ramps » est demande ; sinon la
        correlation MIL-E-5008B « 1 - 0,075 (M0 - 1)^1,35 », qui decrit une
        entree bien dessinee et vaut 0,809 a M0 = 3 — pres de 20 % de la
        pression totale perdue avant meme la chambre, ce qui en fait le premier
        poste de perte du moteur.

        L'ENTREE A RAMPES CALCULE CE QUE LA CORRELATION NE FAIT QUE RESUMER, et
        l'etape 13 a montre que MIL-E-5008B correspond a **trois rampes** a
        M0 = 3. Demander « inlet_ramps = 3 » donne 0,8361 au lieu de 0,8088 : la
        theorie des chocs ignore les pertes visqueuses qu'une correlation
        empirique contient, et l'ecart de 3,3 % est exactement cela.

        L'IMPORTATION DE « thermo » EST PARESSEUSE, ET C'EST DELIBERE. Ce module
        doit rester utilisable sans NumPy — un script d'optimisation exterieur
        construit des milliers de cas et ne doit pas payer l'importation. Un
        « import » place ici ne se declenche QUE si l'on demande reellement une
        entree a rampes ; construire un cas n'en tire rien, et un test le
        verifie dans un interpreteur neuf.
        """
        if self.pi_d is not None:
            return self.pi_d
        if mach <= 1.0:
            return 1.0
        if self.inlet_ramps is not None:
            return _ramp_recovery(mach, self.inlet_ramps)
        return 1.0 - 0.075 * (mach - 1.0) ** 1.35

    def describe(self) -> str:
        if self.pi_d is not None:
            recovery = f"{self.pi_d:g}"
        elif self.inlet_ramps is not None:
            recovery = (f"{self.inlet_ramps} rampe"
                        f"{'s' if self.inlet_ramps > 1 else ''} optimale"
                        f"{'s' if self.inlet_ramps > 1 else ''}")
        else:
            recovery = "MIL-E-5008B"
        return (f"moteur   T04 = {self.T04:g} K   {self.fuel_data.label}   "
                f"eta_b = {self.eta_b:g}   pi_d = {recovery}   "
                f"pi_b = {self.pi_b:g}   pi_n = {self.pi_n:g}   "
                f"M_chambre = {self.M_combustor:g}")
