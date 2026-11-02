"""Les figures. SEUL module a importer Matplotlib, et il n'importe PAS pyplot.

POURQUOI PAS PYPLOT. « pyplot » tient un registre global des figures : chaque
« figure() » y reste jusqu'a ce qu'on la ferme, et une boucle qui en produit
cent les garde toutes en memoire. Il choisit aussi un backend, donc depend de
l'affichage — sur une machine sans ecran, l'importation seule peut echouer.
L'API objet — « Figure », puis « savefig » — n'a ni registre ni backend, et
rend le module utilisable depuis un script, un test ou un serveur sans
distinction. Un test verifie que « matplotlib.pyplot » n'est jamais charge.

FONCTIONS PURES, ET SUR DES TABLEAUX. Aucune fonction d'ici ne connait
« Cycle », « Engine » ou « Convergence » : elles prennent des tableaux et des
nombres. C'est ce qui permet de les tester sur des donnees fabriquees a la
main, sans lancer une seule integration, et ce qui garantit qu'aucune physique
ne se cache dans une figure.

LES COULEURS SONT CELLES D'OKABE-ITO. Huit teintes distinguables en vision
dichromate. La regle qui compte : deux courbes opposees ne sont JAMAIS rouge et
verte — ici bleu et orange, qui restent separees pour une deuterANOPIE comme
pour une protanopie. Le trait et le marqueur portent aussi l'information, de
sorte qu'une impression en noir et blanc reste lisible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .config import CaseError

# -- palette Okabe-Ito -------------------------------------------------------

NOIR = "#000000"
ORANGE = "#E69F00"
CIEL = "#56B4E9"
VERT = "#009E73"
JAUNE = "#F0E442"
BLEU = "#0072B2"
VERMILLON = "#D55E00"
POURPRE = "#CC79A7"

# Fonds de zone : les memes teintes, tres delavees, pour rester sous les
# courbes sans les concurrencer.
FONDS = {
    "diffuser": (CIEL, 0.13),
    "chamber": (ORANGE, 0.16),
    "converging": (VERT, 0.11),
    "diverging": (POURPRE, 0.11),
}
FOND_PAR_DEFAUT = (NOIR, 0.05)

NOMS_FR = {
    "diffuser": "diffuseur",
    "chamber": "chambre",
    "converging": "convergent",
    "diverging": "divergent",
}

DPI = 150


def _verifier(x, *series) -> np.ndarray:
    """Meme longueur, au moins deux points, rien d'infini."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise CaseError(f"il faut au moins deux points, recu {x.size}")
    for serie in series:
        serie = np.asarray(serie, dtype=float)
        if serie.shape != x.shape:
            raise CaseError(
                f"longueurs incompatibles : x fait {x.size} points, une serie "
                f"en fait {serie.size}"
            )
        if not np.all(np.isfinite(serie)):
            raise CaseError("une serie contient des valeurs non finies")
    return x


def _peindre_zones(ax, zones, avec_texte: bool = False) -> None:
    """Le fond colore des zones, et leur nom une seule fois, tout en haut."""
    for nom, debut, fin in zones:
        couleur, alpha = FONDS.get(nom, FOND_PAR_DEFAUT)
        ax.axvspan(debut, fin, color=couleur, alpha=alpha, linewidth=0,
                   zorder=0)
        if avec_texte:
            ax.annotate(NOMS_FR.get(nom, nom),
                        xy=(0.5 * (debut + fin), 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#444444")


def _marquer_col(ax, throat_x: float | None) -> None:
    if throat_x is None:
        return
    ax.axvline(throat_x, color=NOIR, linestyle=(0, (4, 3)), linewidth=1.0,
               alpha=0.65, zorder=3)


# -- 1. les quatre profils ---------------------------------------------------

def profile_figure(x, mach, pressure, temperature, velocity, zones=(),
                   throat_x: float | None = None,
                   title: str = "Profils le long du conduit") -> Figure:
    """M, p, T, u en quatre panneaux empiles qui PARTAGENT l'axe des x.

    L'axe des abscisses n'est ecrit qu'une fois, en bas : le repeter quatre fois
    prendrait la place de la courbe la plus basse et ne dirait rien de plus. Les
    zones ne sont nommees qu'en haut, pour la meme raison.
    """
    x = _verifier(x, mach, pressure, temperature, velocity)
    figure = Figure(figsize=(8.0, 9.0), layout="constrained")
    figure.suptitle(title, fontsize=13)
    axes = figure.subplots(4, 1, sharex=True)

    series = (
        (mach, "Mach", BLEU, None),
        (np.asarray(pressure, dtype=float) / 1e3, "pression (kPa)", ORANGE,
         None),
        (temperature, "temperature (K)", VERMILLON, None),
        (velocity, "vitesse (m/s)", VERT, None),
    )
    for index, (ax, (valeurs, etiquette, couleur, _)) in enumerate(
            zip(axes, series)):
        _peindre_zones(ax, zones, avec_texte=(index == 0))
        ax.plot(x, valeurs, color=couleur, linewidth=1.8, zorder=4)
        ax.set_ylabel(etiquette)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _marquer_col(ax, throat_x)

    # LE MACH VAUT 1 AU COL, ET C'EST LA SEULE LIGNE HORIZONTALE UTILE.
    axes[0].axhline(1.0, color=NOIR, linewidth=0.8, alpha=0.45,
                    linestyle=(0, (1, 2)))
    axes[-1].set_xlabel("x (m)")
    if throat_x is not None:
        axes[0].annotate("col, M = 1", xy=(throat_x, 1.0),
                         xytext=(6, -14), textcoords="offset points",
                         fontsize=8, color="#333333")
    return figure


# -- 2. le conduit, a l'echelle ----------------------------------------------

def duct_figure(x, area, zones=(), throat_x: float | None = None,
                title: str = "Le conduit, a l'echelle") -> Figure:
    """La section en MIROIR, avec des axes egaux, pour qu'on voie le moteur.

    Tracer « A(x) » donne une courbe ; tracer « +r » et « -r » avec
    « set_aspect('equal') » donne un moteur. Le rayon vient de « r = √(A/π) »,
    c'est-a-dire qu'on suppose la section circulaire — ce que le modele 1D ne
    dit pas et n'a pas besoin de dire, mais qu'il faut assumer pour dessiner.
    """
    x = _verifier(x, area)
    aire = np.asarray(area, dtype=float)
    if np.any(aire <= 0.0):
        raise CaseError("section nulle ou negative : rien a dessiner")
    rayon = np.sqrt(aire / np.pi)

    # LE CORPS EST GRIS, PAS BLEU. Les couleurs de zone portent l'information ;
    # un corps colore se melangeait a elles et la chambre paraissait verte.
    figure = Figure(figsize=(9.0, 3.0), layout="constrained")
    figure.suptitle(title, fontsize=13)
    ax = figure.subplots()
    _peindre_zones(ax, zones, avec_texte=True)
    ax.fill_between(x, -rayon, rayon, color="#9a9a9a", alpha=0.28,
                    linewidth=0, zorder=2)
    ax.plot(x, rayon, color=BLEU, linewidth=1.8, zorder=4)
    ax.plot(x, -rayon, color=BLEU, linewidth=1.8, zorder=4)
    ax.axhline(0.0, color=NOIR, linewidth=0.7, alpha=0.4,
               linestyle=(0, (1, 3)), zorder=3)
    _marquer_col(ax, throat_x)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("rayon (m)")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    return figure


# -- 3. le diagramme T-s -----------------------------------------------------

def cycle_figure(names, temperature, pressure, cp: float = 1004.5,
                 R: float = 287.05,
                 title: str = "Cycle 0D, diagramme T-s") -> Figure:
    """Les quatre stations en (s, T0), et la fermeture par l'atmosphere.

    L'ENTROPIE EST RELATIVE A LA STATION 0 : « s - s0 = cp ln(T0/T00) -
    R ln(p0/p00) ». Une entropie absolue demanderait une reference de gaz, et le
    cycle n'a besoin que des ecarts.

    PASSEZ DES ETATS STATIQUES, PAS TOTAUX — et c'est le piege de cette figure.
    Une tuyere ne change pas la temperature TOTALE : c'est sa definition. Trace
    en grandeurs totales, le cycle voit donc les stations 4 et 9 se confondre
    (2 000 K toutes les deux, a 12 J/(kg.K) l'une de l'autre) et la detente
    disparait purement et simplement. Le diagramme perd la moitie de son
    interet sans rien signaler. En statique, la detente descend de 1 900 a
    777 K et le cycle se referme visiblement.

    CE QUE LE TRACE NE DOIT PAS FAIRE CROIRE. Le 0D ne connait QUE les quatre
    stations : les segments entre elles sont des droites tracees faute de mieux,
    pas des chemins calcules. Seule la fermeture 9 -> 0 est une vraie courbe,
    parce qu'elle se fait a pression ambiante constante et que « s » y est alors
    une fonction connue de « T ». Le trait plein dit « je sais », le pointille
    dit « je relie ».
    """
    noms = list(names)
    T0 = np.asarray(temperature, dtype=float)
    p0 = np.asarray(pressure, dtype=float)
    if not (len(noms) == T0.size == p0.size) or T0.size < 2:
        raise CaseError(
            f"il faut autant de noms que de stations, au moins deux "
            f"(recu {len(noms)} noms, {T0.size} temperatures, "
            f"{p0.size} pressions)"
        )
    if np.any(T0 <= 0.0) or np.any(p0 <= 0.0):
        raise CaseError("temperatures et pressions totales doivent etre > 0")

    entropie = cp * np.log(T0 / T0[0]) - R * np.log(p0 / p0[0])

    figure = Figure(figsize=(7.0, 5.6), layout="constrained")
    figure.suptitle(title, fontsize=13)
    ax = figure.subplots()
    ax.plot(entropie, T0, color=NOIR, linewidth=1.2, linestyle=(0, (5, 3)),
            zorder=3, label="segments entre stations (non calcules)")
    ax.scatter(entropie, T0, s=64, color=BLEU, zorder=5, edgecolor="white",
               linewidth=1.2, label="stations du cycle")
    for nom, s, T in zip(noms, entropie, T0):
        ax.annotate(f"  {nom}", xy=(s, T), fontsize=10, va="center",
                    color="#222222")

    # La detente jusqu'a l'ambiante, puis le refroidissement a pression
    # constante : la seule branche que le 0D sait vraiment tracer.
    refroidissement = np.linspace(float(T0[-1]), float(T0[0]), 60)
    s_fin = float(entropie[-1])
    ax.plot(s_fin + cp * np.log(refroidissement / T0[-1]), refroidissement,
            color=ORANGE, linewidth=1.8, zorder=4,
            label="fermeture a pression ambiante")

    ax.set_xlabel("s - s0 (J/(kg.K))")
    ax.set_ylabel("temperature statique (K)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9, loc="upper left")
    return figure


# -- 4. les balayages --------------------------------------------------------

def sweep_figure(machs, specific_thrust, specific_impulse,
                 thrust_band: tuple[float, float] | None = None,
                 impulse_band: tuple[float, float] | None = None,
                 best: tuple[float, float] | None = None,
                 title: str = "Balayage en Mach de vol") -> Figure:
    """F/mdot et Isp contre le Mach de vol, avec les fourchettes publiees.

    LES BANDES SONT DES FOURCHETTES DE LITTERATURE, PAS DES BARRES D'ERREUR. Une
    courbe qui en sort n'est pas fausse : elle decrit un moteur hors du domaine
    ou ces ordres de grandeur ont ete releves. La legende le dit.
    """
    machs = _verifier(machs, specific_thrust, specific_impulse)
    figure = Figure(figsize=(8.0, 6.4), layout="constrained")
    figure.suptitle(title, fontsize=13)
    haut, bas = figure.subplots(2, 1, sharex=True)

    for ax, valeurs, bande, etiquette, couleur in (
            (haut, specific_thrust, thrust_band,
             "poussee specifique (N.s/kg)", BLEU),
            (bas, specific_impulse, impulse_band, "impulsion specifique (s)",
             ORANGE)):
        if bande is not None:
            ax.axhspan(bande[0], bande[1], color=VERT, alpha=0.16,
                       linewidth=0, zorder=0)
        ax.plot(machs, valeurs, color=couleur, linewidth=1.9, zorder=4)
        ax.set_ylabel(etiquette)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    if best is not None:
        haut.scatter([best[0]], [best[1]], s=70, color=VERMILLON, zorder=6,
                     edgecolor="white", linewidth=1.2)
        haut.annotate(f"maximum : M0 = {best[0]:.2f}", xy=best,
                      xytext=(10, 12), textcoords="offset points", fontsize=9,
                      color="#333333")

    bas.set_xlabel("Mach de vol")
    poignees = [Patch(facecolor=VERT, alpha=0.16,
                      label="fourchette publiee (ordre de grandeur)")]
    haut.legend(handles=poignees, fontsize=9, loc="lower left")
    return figure


# -- 5. la carte de blocage --------------------------------------------------

def choking_figure(machs, max_total_temperature,
                   operating: tuple[float, float] | None = None,
                   design_mach: float | None = None,
                   title: str = "Limite de blocage thermique") -> Figure:
    """T04 maximal contre Mach d'entree de chambre, et le point ou l'on est.

    LA FIGURE QUI JUSTIFIE LA CARTE : la limite s'effondre tres vite avec le
    Mach de chambre, et c'est pourquoi un statoreacteur y ralentit l'air a 0,2.
    Si le Mach de CONCEPTION est donne, il est marque aussi : l'ecart entre lui
    et le Mach reel est exactement ce que l'etape 9 a mesure, et il deplace la
    limite de plusieurs centaines de kelvins.
    """
    machs = _verifier(machs, max_total_temperature)
    figure = Figure(figsize=(7.4, 5.2), layout="constrained")
    figure.suptitle(title, fontsize=13)
    ax = figure.subplots()

    # L'ORDONNEE EST LOGARITHMIQUE, ET CE N'EST PAS UN ORNEMENT. La limite
    # diverge comme 1/M^2 quand le Mach tend vers zero : en echelle lineaire
    # elle vaut 51 000 K au bord gauche, ce qui ecrase toute la zone utile
    # (2 000 a 4 000 K) dans le dixieme inferieur de la figure et rend le point
    # de fonctionnement indiscernable de la courbe.
    limite = np.asarray(max_total_temperature, dtype=float)
    haut = float(limite.max()) * 1.3
    ax.fill_between(machs, limite, haut, color=VERMILLON, alpha=0.12,
                    linewidth=0, zorder=0, label="au-dela : pas de solution")
    ax.plot(machs, limite, color=VERMILLON, linewidth=2.0, zorder=4,
            label="T04 maximale (blocage)")
    ax.set_yscale("log")
    ax.set_ylim(float(limite.min()) * 0.7, haut)

    if operating is not None:
        ax.scatter([operating[0]], [operating[1]], s=80, color=BLEU, zorder=6,
                   edgecolor="white", linewidth=1.3,
                   label="point de fonctionnement")
        ax.annotate(f"M = {operating[0]:.4f}, T04 = {operating[1]:.0f} K",
                    xy=operating, xytext=(10, -18),
                    textcoords="offset points", fontsize=9, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.82))
    if design_mach is not None:
        ax.axvline(design_mach, color=NOIR, linestyle=(0, (4, 3)),
                   linewidth=1.1, alpha=0.6, zorder=3,
                   label=f"Mach de conception ({design_mach:g})")

    ax.set_xlabel("Mach d'entree de chambre")
    ax.set_ylabel("temperature totale de fin de combustion (K)")
    ax.grid(True, alpha=0.25, linewidth=0.6, which="both")
    ax.legend(fontsize=9, loc="lower left")
    return figure


# -- 6. l'adaptation hors du point de dessin ---------------------------------

def adaptation_figure(machs, loss_by_mach, altitudes, loss_by_altitude,
                      pressure_ratio_by_mach=None,
                      separation: float | None = None,
                      design_mach: float | None = None,
                      title: str = "Adaptation hors du point de dessin"
                      ) -> Figure:
    """La perte d'adaptation contre le Mach, PUIS contre l'altitude.

    LES DEUX PANNEAUX SONT LA POUR ETRE COMPARES, ET LE SECOND EST LA REPONSE.
    On s'attend a ce que l'adaptation d'une tuyere depende de l'altitude — c'est
    le cas pour une fusee, dont la pression de chambre ne suit pas l'ambiante.
    Pour un statoreacteur, la compression dynamique fait que « p04 » est
    PROPORTIONNELLE a l'ambiante : le rapport « p9/p_amb » s'en trouve
    independant, et le second panneau est plat. Les deux echelles verticales
    sont donc volontairement IDENTIQUES : rendre le panneau plat lisible en le
    dilatant ferait mentir la comparaison.
    """
    machs = _verifier(machs, loss_by_mach)
    altitudes = _verifier(altitudes, loss_by_altitude)
    figure = Figure(figsize=(8.0, 6.6), layout="constrained")
    figure.suptitle(title, fontsize=13)
    haut, bas = figure.subplots(2, 1)

    pertes = np.concatenate([np.asarray(loss_by_mach, dtype=float),
                             np.asarray(loss_by_altitude, dtype=float)])
    plafond = max(float(np.max(pertes)) * 1.15, 1e-3)

    if separation is not None and pressure_ratio_by_mach is not None:
        # LA FRONTIERE EST INTERPOLEE, PAS PRISE SUR LA GRILLE. Marquer
        # « jusqu'au dernier point decolle » donne une bande de largeur nulle
        # des qu'un seul point l'est, ce qui cache justement le cas qu'on
        # voulait montrer.
        rapports = np.asarray(pressure_ratio_by_mach, dtype=float)
        decolle = rapports < separation
        if bool(np.any(decolle)) and not bool(np.all(decolle)):
            dernier = int(np.max(np.flatnonzero(decolle)))
            frontiere = float(np.interp(
                separation, rapports[dernier:dernier + 2],
                machs[dernier:dernier + 2]))
            haut.axvspan(float(np.min(machs)), frontiere, color=VERMILLON,
                         alpha=0.14, linewidth=0, zorder=0,
                         label="ecoulement decolle : hors validite")
        elif bool(np.all(decolle)):
            haut.axvspan(float(np.min(machs)), float(np.max(machs)),
                         color=VERMILLON, alpha=0.14, linewidth=0, zorder=0,
                         label="ecoulement decolle : hors validite")
    haut.plot(machs, np.asarray(loss_by_mach, dtype=float) * 100.0,
              color=BLEU, linewidth=2.0, marker="o", markersize=4, zorder=4,
              label="perte d'adaptation")
    if design_mach is not None:
        haut.axvline(design_mach, color=NOIR, linestyle=(0, (4, 3)),
                     linewidth=1.1, alpha=0.6, zorder=3,
                     label=f"point de dessin (M0 = {design_mach:g})")
    haut.set_xlabel("Mach de vol")
    haut.set_ylabel("perte d'adaptation (%)")
    haut.set_ylim(0.0, plafond * 100.0)
    haut.grid(True, alpha=0.25, linewidth=0.6)
    haut.legend(fontsize=9, loc="upper center")

    bas.plot(np.asarray(altitudes, dtype=float) / 1000.0,
             np.asarray(loss_by_altitude, dtype=float) * 100.0, color=ORANGE,
             linewidth=2.0, marker="s", markersize=4, zorder=4)
    bas.set_xlabel("altitude (km)")
    bas.set_ylabel("perte d'adaptation (%)")
    bas.set_ylim(0.0, plafond * 100.0)
    bas.grid(True, alpha=0.25, linewidth=0.6)
    bas.annotate("meme echelle que ci-dessus : la courbe est PLATE",
                 xy=(0.5, 0.55), xycoords="axes fraction", ha="center",
                 fontsize=10, color="#444444")
    return figure


# -- l'ecriture --------------------------------------------------------------

def save(figure: Figure, path, dpi: int = DPI) -> Path:
    """Ecrit la figure, en creant le dossier si besoin. Rend le chemin."""
    chemin = Path(path)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=dpi)
    return chemin
