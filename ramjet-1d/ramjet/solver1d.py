"""Le solveur quasi-1D : une EDO en M^2, et tout le reste reconstruit.

L'EQUATION, EN COEFFICIENTS D'INFLUENCE (Shapiro, ch. 8) :

    dM^2/M^2 = -2(1 + (g-1)/2 M^2)/(1 - M^2) dA/A
             + (1 + g M^2)(1 + (g-1)/2 M^2)/(1 - M^2) dT0/T0

Une seule variable est integree. Ce n'est pas une economie : c'est ce qui rend
la QUANTITE DE MOUVEMENT independante. « T » se deduit de la definition de T0,
« p » de la conservation de la masse, « u » de la definition du Mach — et la
quantite de mouvement n'entre nulle part dans cette reconstruction, alors
qu'elle est deja encodee dans l'EDO. Elle devient donc un controle qui ne
partage rien avec ce qu'il verifie, exactement comme la divergence nulle de
« cfd-cavity ». Le test compare « F(sortie) - F(entree) » a « integrale de
p dA », avec F = pA(1 + gM^2).

CE QUE CE MODULE INTEGRE, ET CE QU'IL N'INTEGRE PAS.

  - il integre a DEBIT CONSTANT. L'apport de masse du carburant (3,5 % sur le
    cas de reference) n'a pas de coefficient d'influence ici : c'est le choix
    qui rend l'ecoulement de Rayleigh utilisable comme oracle EXACT, et
    l'etape 5 a mesure ce qu'il coute — 0,0192 sur le Mach de sortie de
    chambre, contre 0,0132 pour la correction que l'etape 9 cherchera. Le
    couplage devra donc comparer ce qui est comparable, et le rapport le dira ;
  - il ne traverse PAS M = 1. Au col, « dA/dx » s'annule en meme temps que
    « 1 - M^2 » : le passage transsonique est un 0/0 dont la levee est le
    travail de l'etape 8. Ici, on integre des ecoulements qui restent d'un seul
    cote ;
  - il DETECTE le blocage thermique depuis l'etape 7, par un evenement
    terminal pose a M = 0,999. Sans lui, « solve_ivp » echoue sur « Required
    step size is less than spacing between numbers », un message qui ne parle
    pas de physique ; avec lui, l'integration s'arrete proprement et l'on sait
    ou et pourquoi. Le test compare les deux comportements.

LES TOLERANCES SONT SERREES, ET C'EST GRATUIT. « rtol = 1e-10, atol = 1e-12 »
sur une EDO scalaire integree sur un metre et demi coute quelques millisecondes,
et c'est ce qui permet de comparer a des formes fermees a 1e-12 plutot qu'a
1e-6. Un solveur regle a 1e-6 ne dirait plus rien de son propre schema.

LES GRANDEURS RECONSTRUITES AMPLIFIENT L'ERREUR DE L'EDO, ET IL FAUT LE SAVOIR
AVANT DE FIXER UNE TOLERANCE. La pression totale passe par « (1 + (g-1)/2 M^2)
^(g/(g-1)) », donc une erreur relative sur M s'y retrouve multipliee par cent a
mille. Mesure de la derive de p0 sur une tuyere isentropique, ou elle devrait
etre nulle :

    rtol        1e-8      1e-10     1e-12
    derive p0   8,9e-08   2,3e-09   2,9e-11
    ecart sur M 6,2e-11   2,0e-12   3,0e-14

La derive suit la tolerance a la proportionnelle : c'est bien de l'erreur
d'integration et non un defaut de physique. Les tests le verifient en faisant
varier « rtol » plutot qu'en posant un seuil absolu, ce qui distingue les deux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson, solve_ivp

from . import fanno, rayleigh
from .config import CaseError
from .thermo import total_pressure_ratio, total_temperature_ratio

RTOL, ATOL = 1e-10, 1e-12

# Seuil de detection du blocage. On s'arrete a 0,999 plutot qu'a 1 parce que
# l'equation est singuliere en 1 : la derivee y est infinie, et l'integrateur
# a pas adaptatif reduit son pas jusqu'a l'echec avant de l'atteindre. A 0,999,
# il s'arrete PROPREMENT et on peut dire ou et pourquoi.
MACH_EVENT = 0.999


class ChokedFlow(CaseError):
    """L'ecoulement se bloque : il n'existe pas de solution stationnaire.

    Ce n'est pas un echec numerique, c'est un resultat. L'exception porte donc
    de quoi comprendre : ou le blocage arrive, quel Mach a ete atteint, et
    quel rapport de temperature la section constante aurait autorise.
    """

    def __init__(self, message: str, x: float, mach: float,
                 max_ratio: float | None = None):
        super().__init__(message)
        self.x = x
        self.mach = mach
        self.max_ratio = max_ratio


# -- le depot de chaleur ----------------------------------------------------

@dataclass(frozen=True)
class HeatRelease:
    """T0(x) et sa derivee. La combustion, reduite a ce que le 1D en voit.

    Pas de chimie, pas d'especes, pas de temps de reaction : un profil de
    temperature totale impose, nul en dehors de la zone de chambre. C'est le
    perimetre annonce, et c'est aussi ce qui rend l'oracle de Rayleigh
    applicable — lui non plus ne connait qu'un apport de chaleur.
    """

    T0_in: float
    T0_out: float
    x_start: float
    x_end: float

    def __post_init__(self) -> None:
        if self.x_end <= self.x_start:
            raise CaseError(
                f"zone de combustion vide : [{self.x_start:g} ; {self.x_end:g}]"
            )
        if min(self.T0_in, self.T0_out) <= 0.0:
            raise CaseError("temperatures totales non positives")

    @property
    def rate(self) -> float:
        """dT0/dx dans la zone, constante : la chaleur est repartie uniformement."""
        return (self.T0_out - self.T0_in) / (self.x_end - self.x_start)

    def total_temperature(self, x):
        x = np.asarray(x, dtype=float)
        avance = np.clip(x, self.x_start, self.x_end) - self.x_start
        return self.T0_in + self.rate * avance

    def derivative(self, x):
        x = np.asarray(x, dtype=float)
        dedans = (x >= self.x_start) & (x <= self.x_end)
        return np.where(dedans, self.rate, 0.0)


def no_heat(T0: float, length: float) -> HeatRelease:
    """Un profil sans apport : T0 constante sur tout le conduit."""
    return HeatRelease(T0_in=T0, T0_out=T0, x_start=0.0, x_end=length)


# -- l'EDO ------------------------------------------------------------------

@dataclass(frozen=True)
class MassAddition:
    """L'injection de carburant, vue par le 1D : un debit qui grandit.

    CORRIGE LE DEFAUT LE PLUS DOCUMENTE DU PROJET. Jusqu'ici l'EDO conservait la
    masse et le « (1+f) » etait remis a la main dans le bilan de poussee. Cela
    coutait 7,4 % sur la poussee si on l'oubliait (etape 8) et, plus grave,
    rendait le terme de pression incoherent avec celui de quantite de mouvement
    — au point de faire paraitre une tuyere non adaptee meilleure qu'une adaptee
    (etape 15). Le debit est desormais une fonction de x.

    L'INJECTION EST TRANSVERSALE, ET C'EST UNE HYPOTHESE QU'IL FAUT DIRE. Le
    carburant entre sans quantite de mouvement axiale : il faut donc l'acceler
    aux depens de l'ecoulement principal, ce qui est une trainee. C'est le cas
    conservateur et le plus courant ; une injection axiale a la vitesse locale
    ne couterait rien et flatterait le moteur.

    LE COEFFICIENT VAUT EXACTEMENT LE DOUBLE DE CELUI DE LA CHALEUR. Derive a
    partir des cinq equations de conservation et verifie contre un solveur
    lineaire independant a 7,5e-16 sur dix-huit couples (gamma, M). Consequence
    remarquable : a temperature totale constante, ajouter de la masse revient
    EXACTEMENT a chauffer selon Rayleigh avec « T0 proportionnelle a mdot^2 »,
    ce qui donne au terme un oracle exact — comme les trois autres.
    """

    ratio: float = 0.0             # f : masse ajoutee, rapportee a l'entree
    x_start: float = 0.0
    x_end: float = 1.0

    def __post_init__(self) -> None:
        if self.ratio < 0.0:
            raise CaseError(
                f"apport de masse negatif : {self.ratio:g} (un carburant qui "
                f"sort du moteur n'existe pas)"
            )
        if self.active and self.x_end <= self.x_start:
            raise CaseError(
                f"zone d'injection vide : [{self.x_start:g} ; {self.x_end:g}]"
            )

    @property
    def active(self) -> bool:
        return self.ratio > 0.0

    def factor(self, x):
        """mdot(x)/mdot(entree) : croit lineairement dans la zone d'injection.

        Le meme profil que la chaleur, parce que c'est le meme carburant : ce
        qui brule est ce qui vient d'entrer.
        """
        x = np.asarray(x, dtype=float)
        if not self.active:
            return np.ones_like(x)
        avance = ((np.clip(x, self.x_start, self.x_end) - self.x_start)
                  / (self.x_end - self.x_start))
        return 1.0 + self.ratio * avance

    def derivative(self, x):
        """dmdot/dx, rapporte au debit d'ENTREE."""
        x = np.asarray(x, dtype=float)
        if not self.active:
            return np.zeros_like(x)
        dedans = (x >= self.x_start) & (x <= self.x_end)
        return np.where(dedans, self.ratio / (self.x_end - self.x_start), 0.0)

    def relative_derivative(self, x: float) -> float:
        """« (dmdot/dx)/mdot », le terme tel qu'il entre dans l'EDO."""
        if not self.active:
            return 0.0
        return float(self.derivative(x)) / float(self.factor(x))


NO_MASS_ADDITION = MassAddition(0.0)


@dataclass(frozen=True)
class Friction:
    """Le frottement parietal, reduit a ce que le 1D en voit : un « C_f ».

    ETAPE 12. Le coefficient de Fanning est suppose CONSTANT le long du
    conduit. C'est une approximation assumee : il depend en realite du nombre
    de Reynolds et de la rugosite, donc de x. Pour un conduit de statoreacteur
    a Reynolds eleve, 0,002 a 0,005 couvre le domaine usuel, et prendre une
    constante suffit a repondre a la question posee — de combien le frottement
    rapproche-t-il du blocage — sans pretendre calculer une couche limite que
    le perimetre exclut de toute facon.

    Le diametre est HYDRAULIQUE et varie avec la section : « D = sqrt(4 A/pi) »
    pour une section circulaire. Le terme source est donc « 4 C_f/D(x) », qui
    grandit la ou le conduit se resserre.
    """

    coefficient: float = 0.0        # C_f de Fanning
    circular: bool = True

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise CaseError(
                f"coefficient de frottement negatif : {self.coefficient:g} "
                f"(un frottement qui pousse n'existe pas)"
            )

    @property
    def active(self) -> bool:
        return self.coefficient > 0.0

    def hydraulic_diameter(self, area: float) -> float:
        """D = sqrt(4 A/pi), la section etant supposee circulaire."""
        if not self.circular:
            raise CaseError("seule la section circulaire est implementee")
        return float(np.sqrt(4.0 * area / np.pi))

    def source(self, area: float) -> float:
        """« 4 C_f / D » en 1/m : le terme tel qu'il entre dans l'EDO."""
        if not self.active:
            return 0.0
        return 4.0 * self.coefficient / self.hydraulic_diameter(area)


NO_FRICTION = Friction(0.0)


def dmach2_dx(x: float, mach2: float, duct, heat: HeatRelease,
              gamma: float = 1.4, friction: Friction = NO_FRICTION,
              mass: MassAddition = NO_MASS_ADDITION) -> float:
    """Les QUATRE coefficients d'influence. LA singularite est ecrite en clair.

    Tous portent le meme denominateur « 1 - M^2 », et trois d'entre eux portent
    le meme signe : la chaleur, le frottement et l'apport de masse poussent tous
    le Mach vers 1, par les deux cotes. Seule la section peut l'en eloigner.

    Le quatrieme terme, « 2 (1 + gamma M^2) (1 + (gamma-1)/2 M^2)/(1 - M^2) »
    multiplie par « (dmdot/dx)/mdot », est celui de l'injection. Il vaut
    EXACTEMENT le double de celui de la chaleur — resultat derive des cinq
    equations de conservation et verifie contre un solveur lineaire independant
    a 7,5e-16.
    """
    area = float(duct.area(x))
    d_area = float(duct.area_derivative(x))
    T0 = float(heat.total_temperature(x))
    dT0 = float(heat.derivative(x))

    denominateur = 1.0 - mach2
    facteur = 1.0 + 0.5 * (gamma - 1.0) * mach2
    chaleur = (1.0 + gamma * mach2) * facteur / denominateur
    return mach2 * (-2.0 * facteur / denominateur * (d_area / area)
                    + chaleur * (dT0 / T0)
                    + gamma * mach2 * facteur / denominateur
                    * friction.source(area)
                    + 2.0 * chaleur * mass.relative_derivative(x))


# -- le resultat ------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Profile:
    """Les grandeurs le long du conduit, reconstruites depuis M^2."""

    x: np.ndarray
    mach: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    density: np.ndarray
    total_temperature: np.ndarray
    total_pressure: np.ndarray
    area: np.ndarray
    mass_flow: np.ndarray          # mdot(x) : il GRANDIT si l'on injecte
    gamma: float

    @property
    def inlet_mass_flow(self) -> float:
        return float(np.asarray(self.mass_flow).reshape(-1)[0])

    @property
    def exit_mass_flow(self) -> float:
        return float(np.asarray(self.mass_flow).reshape(-1)[-1])

    @property
    def fuel_air_ratio(self) -> float:
        """« f » vu par le 1D : ce qu'il a reellement ajoute."""
        return self.exit_mass_flow / self.inlet_mass_flow - 1.0

    @property
    def impulse(self) -> np.ndarray:
        """F = p A (1 + gamma M^2), la fonction d'impulsion.

        Elle n'a servi a rien dans la reconstruction : c'est ce qui en fait un
        controle independant de l'EDO.
        """
        return self.pressure * self.area * (1.0 + self.gamma * self.mach ** 2)

    @property
    def mass_flux(self) -> np.ndarray:
        """rho u A. Doit valoir le debit LOCAL partout, par construction."""
        return self.density * self.velocity * self.area

    @property
    def total_pressure_ratio(self) -> float:
        """p0(sortie)/p0(entree) : le « pi » que le 0D suppose, et que le 1D
        calcule."""
        return float(self.total_pressure[-1] / self.total_pressure[0])

    def describe(self) -> str:
        return (f"  M {self.mach[0]:.4f} -> {self.mach[-1]:.4f}   "
                f"p {self.pressure[0] / 1e5:.3f} -> "
                f"{self.pressure[-1] / 1e5:.3f} bar   "
                f"T {self.temperature[0]:.0f} -> {self.temperature[-1]:.0f} K   "
                f"pi = {self.total_pressure_ratio:.4f}")


def reconstruct(x, mach2, duct, heat: HeatRelease, mass_flow: float,
                gamma: float, R: float,
                mass: MassAddition = NO_MASS_ADDITION) -> Profile:
    """Toutes les grandeurs depuis M^2. La quantite de mouvement N'Y SERT PAS.

    « T » vient de la definition de T0, « p » de la conservation de la masse,
    « rho » de la loi des gaz parfaits, « u » de la definition du Mach. Le debit
    n'est plus une constante : c'est « mdot_entree . facteur(x) ».
    """
    x = np.asarray(x, dtype=float)
    mach2 = np.asarray(mach2, dtype=float)
    if np.any(mach2 <= 0.0):
        raise CaseError("Mach nul ou negatif dans la reconstruction")

    mach = np.sqrt(mach2)
    area = np.asarray(duct.area(x), dtype=float)
    debit = mass_flow * np.asarray(mass.factor(x), dtype=float)
    T0 = np.asarray(heat.total_temperature(x), dtype=float)
    T = T0 / np.asarray(total_temperature_ratio(mach, gamma), dtype=float)
    p = debit * np.sqrt(R * T) / (area * mach * np.sqrt(gamma))
    return Profile(
        x=x, mach=mach, temperature=T, pressure=p,
        velocity=mach * np.sqrt(gamma * R * T),
        density=p / (R * T), total_temperature=T0,
        total_pressure=p * np.asarray(total_pressure_ratio(mach, gamma),
                                      dtype=float),
        area=area, mass_flow=debit, gamma=gamma,
    )


def _diagnostic(x_blocage: float, mach_atteint: float, mach_in: float,
                demande: float, maximum: float, duct, friction: Friction,
                depart: float, arrivee: float, gamma: float) -> str:
    """Le message de refus, qui doit NOMMER LA CAUSE et pas la supposer.

    ETAPE 12, ET C'EST UN DEFAUT QUE LE FROTTEMENT A REVELE. Tant que la
    chaleur etait le seul terme source, annoncer « blocage thermique » etait
    juste par construction. Des que le frottement existe, un conduit sans la
    moindre chaleur peut bloquer — et le message annoncait alors « la chaleur
    demandee porterait T0 d'un facteur 1,0000 [...] soit 0,79 fois trop », ce
    qui est a la fois faux et incomprehensible.

    On mesure donc les deux budgets separement : celui de Rayleigh, le rapport
    de temperature totale rapporte a son maximum, et celui de Fanno, la
    longueur reduite parcourue rapportee a celle qui restait. Chacun n'est
    exact qu'a section constante, ce que le message dit.
    """
    chauffe = demande > 1.0 + 1e-12
    budget_chaleur = demande / maximum if chauffe else 0.0

    budget_frottement = 0.0
    if friction.active:
        reste = float(fanno.friction_length(mach_in, gamma))
        section = float(duct.area(0.5 * (depart + arrivee)))
        consomme = friction.source(section) * (arrivee - depart)
        budget_frottement = consomme / reste if reste > 0.0 else float("inf")

    lignes = [f"blocage a x = {x_blocage:.4f} m (M = {mach_atteint:.4f}), "
              f"pour un ecoulement entrant a M = {mach_in:g}."]
    if chauffe:
        lignes.append(
            f" Chaleur : le rapport de temperature totale demande vaut "
            f"{demande:.4f} la ou {maximum:.4f} est le maximum de Rayleigh, "
            f"soit {budget_chaleur:.0%} du budget.")
    if friction.active:
        lignes.append(
            f" Frottement : le conduit consomme {budget_frottement:.0%} de la "
            f"longueur de Fanno disponible a C_f = {friction.coefficient:g}.")
    if chauffe and friction.active:
        lignes.append(
            " LES DEUX S'ADDITIONNENT : ils poussent tous deux le Mach vers 1, "
            "et aucun des deux budgets pris seul ne dit s'il y a blocage.")

    remedes = []
    if chauffe:
        remedes.append("baisser la temperature de fin de combustion")
    if friction.active:
        remedes.append("raccourcir le conduit ou l'elargir")
    remedes.append("ralentir l'ecoulement en entree")
    lignes.append(f" Remedes : {', '.join(remedes)}.")
    return "".join(lignes)


# -- l'integration ----------------------------------------------------------

def integrate_duct(duct, heat: HeatRelease, mach_in: float, mass_flow: float,
                   gamma: float = 1.4, R: float = 287.05, n_points: int = 400,
                   x_start: float | None = None, x_end: float | None = None,
                   rtol: float = RTOL, atol: float = ATOL,
                   friction: Friction = NO_FRICTION,
                   mass: MassAddition = NO_MASS_ADDITION) -> Profile:
    """Integre l'EDO d'un bout a l'autre et reconstruit tout le reste.

    Ne traverse pas M = 1 et ne detecte pas encore le blocage : voir l'en-tete.
    """
    if mach_in <= 0.0:
        raise CaseError(f"Mach d'entree non positif : {mach_in:g}")
    if abs(mach_in - 1.0) < 1e-6:
        raise CaseError(
            f"Mach d'entree {mach_in:g} : l'equation est singuliere en M = 1, "
            f"et le passage transsonique demande la levee du 0/0 (etape 8)"
        )
    depart = 0.0 if x_start is None else x_start
    arrivee = duct.length if x_end is None else x_end
    if arrivee <= depart:
        raise CaseError(f"intervalle vide : [{depart:g} ; {arrivee:g}]")

    # L'EVENEMENT, ET LE SENS DE SA DERIVEE. La fonction vaut la « marge »
    # jusqu'au seuil : positive tant qu'on est du bon cote, et DECROISSANTE en
    # s'en approchant. Elle est donc franchie dans le sens negatif dans les deux
    # cas, subsonique comme supersonique — d'ou « direction = -1 » partout.
    # Avec « direction = +1 », l'evenement ne se declenche JAMAIS et l'on
    # retombe sur l'echec cryptique de l'integrateur ; le test le verifie.
    subsonique = mach_in < 1.0
    seuil2 = (MACH_EVENT if subsonique else 2.0 - MACH_EVENT) ** 2

    def blocage(x, y):
        return (seuil2 - y[0]) if subsonique else (y[0] - seuil2)

    blocage.terminal = True
    blocage.direction = -1.0

    solution = solve_ivp(
        lambda x, y: [dmach2_dx(x, float(y[0]), duct, heat, gamma,
                                friction, mass)],
        (depart, arrivee), [mach_in ** 2],
        t_eval=np.linspace(depart, arrivee, n_points),
        rtol=rtol, atol=atol, dense_output=True, events=blocage,
    )

    if len(solution.t_events[0]) > 0:
        x_blocage = float(solution.t_events[0][0])
        mach_atteint = float(np.sqrt(solution.y_events[0][0][0]))
        maximum = float(rayleigh.max_total_temperature_ratio(mach_in, gamma))
        # Le rapport DEMANDE, celui du bout du conduit — pas celui atteint au
        # point de blocage, qui vaut le maximum par construction et ne dirait
        # donc rien a l'utilisateur.
        demande = (float(heat.total_temperature(arrivee))
                   / float(heat.total_temperature(depart)))
        raise ChokedFlow(
            _diagnostic(x_blocage, mach_atteint, mach_in, demande, maximum,
                        duct, friction, depart, arrivee, gamma),
            x=x_blocage, mach=mach_atteint, max_ratio=maximum,
        )
    if not solution.success:
        raise CaseError(
            f"l'integration a echoue a x = {solution.t[-1]:.4f} m : "
            f"{solution.message}"
        )
    return reconstruct(solution.t, solution.y[0], duct, heat, mass_flow,
                       gamma, R, mass)


def momentum_residual(profile: Profile, duct,
                      friction: Friction = NO_FRICTION) -> float:
    """L'ecart de l'identite « F(sortie) - F(entree) = integrale p dA », rapporte
    a l'impulsion elle-meme.

    LE controle independant du module. Rien dans l'integration ne le force :
    l'EDO n'integre que M^2, et « p » est reconstruite par la masse.

    DEUX CHOIX QUI ONT DEMANDE UNE MESURE.

    Le denominateur est l'impulsion MOYENNE, et non le saut. Sur un conduit a
    section constante — l'ecoulement de Rayleigh — les deux membres valent zero
    exactement (F y est conservee), et diviser par le saut rendait un residu de
    1,00 pour un ecart de 1e-10 : une division de zero par zero, deguisee.

    La quadrature est celle de Simpson, et non celle des trapezes. Mesure sur la
    tuyere subsonique a 2 000 points : trapezes 2,5e-07, Simpson 4e-14. L'erreur
    des trapezes va en (dx)^2, soit exactement les 2,5e-7 observes a
    dx = 5e-4 — c'etait donc la QUADRATURE qu'on mesurait, pas la solution.

    ET UN REFUS, AJOUTE A L'ETAPE 8. La formule de Simpson suppose un pas
    CONSTANT ; appliquee a un profil recolle de plusieurs troncons, elle rend un
    residu de 1,6e-05 la ou chaque troncon pris a part donne 1e-11. L'ecart
    n'est alors pas dans la solution mais dans la quadrature, et rien ne le
    disait. On refuse donc explicitement le pas variable : mieux vaut ne pas
    repondre que repondre a cote.
    """
    pas = np.diff(np.asarray(profile.x, dtype=float))
    if len(pas) < 2:
        raise CaseError("au moins trois points sont necessaires pour Simpson")
    if float(np.ptp(pas)) > 1e-9 * float(np.mean(pas)):
        raise CaseError(
            f"pas non constant (de {float(np.min(pas)):g} a "
            f"{float(np.max(pas)):g} m) : la formule de Simpson n'y est pas "
            f"valable et rendrait une erreur de quadrature deguisee en erreur "
            f"de solution. Verifiez chaque troncon separement"
        )
    impulse = profile.impulse
    saut = float(impulse[-1] - impulse[0])
    d_area = np.asarray(duct.area_derivative(profile.x), dtype=float)
    integrale = float(simpson(profile.pressure * d_area, x=profile.x))

    # LA FORCE DE FROTTEMENT ENTRE DANS LE BILAN, ET C'EST CE QUI FAIT DE CE
    # CONTROLE UNE VERIFICATION DU TROISIEME TERME. La contrainte parietale
    # integree sur le perimetre vaut « 1/2 rho u^2 . (4 C_f/D) . A » par metre :
    # exactement le terme source de l'EDO, mais atteint par un chemin
    # different — un bilan de quantite de mouvement, la ou l'EDO n'a integre
    # que M^2. Sans ce terme, le residu passerait de 1e-11 a 1e-2 des que le
    # frottement est actif, et l'on croirait le solveur casse.
    if friction.active:
        area = np.asarray(duct.area(profile.x), dtype=float)
        source = np.array([friction.source(float(a)) for a in area])
        trainee = 0.5 * profile.density * profile.velocity ** 2 * source * area
        integrale -= float(simpson(trainee, x=profile.x))
    return abs(saut - integrale) / float(np.mean(np.abs(impulse)))
