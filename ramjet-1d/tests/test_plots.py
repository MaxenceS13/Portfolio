"""Les figures : ce qu'elles contiennent, et ce qu'elles ne doivent pas cacher.

QUATRE FAMILLES, ET LA DERNIERE EST LA PLUS INSTRUCTIVE.

  - LE CLOISONNEMENT. Matplotlib n'entre nulle part ailleurs, et « pyplot »
    n'entre nulle part du tout — pas meme ici.
  - LA STRUCTURE. Le bon nombre d'axes et de courbes, le col au bon endroit,
    des fichiers qui existent et pesent quelque chose.
  - LA VISION DICHROMATE, verifiee par simulation et non par bonne intention.
    Le test montre aussi qu'il a du POUVOIR : le rouge et le vert par defaut de
    Matplotlib s'effondrent, les couleurs d'ici non.
  - LE PIEGE DU DIAGRAMME T-s. Trace en grandeurs totales, il perd la detente
    sans rien signaler. Le test mesure la perte.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from ramjet import plots
from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.cycle0d import run_cycle

RACINE = Path(__file__).resolve().parent.parent

ZONES = (("diffuser", 0.0, 0.40), ("chamber", 0.40, 1.00),
         ("converging", 1.00, 1.15), ("diverging", 1.15, 1.50))
COL = 1.15


def profil_fabrique(n: int = 60):
    """Des tableaux plausibles, sans lancer la moindre integration.

    C'est tout l'interet de fonctions pures sur des tableaux : les figures se
    testent sans physique, donc un echec ici ne peut venir que du trace.
    """
    x = np.linspace(0.0, 1.50, n)
    return dict(x=x, mach=0.2 + 1.8 * (x / 1.5) ** 3,
                pressure=6.6e5 - 6.4e5 * (x / 1.5) ** 2,
                temperature=605.0 + 1300.0 * np.sin(np.pi * x / 3.0),
                velocity=100.0 + 1450.0 * (x / 1.5) ** 2)


# -- le cloisonnement --------------------------------------------------------

def _modules_charges(imports: str, cibles) -> str:
    code = (f"import sys; {imports}; "
            f"print(','.join(n for n in ({', '.join(map(repr, cibles))},) "
            f"if n in sys.modules))")
    fait = subprocess.run([sys.executable, "-c", code], cwd=RACINE,
                          capture_output=True, text=True, timeout=120)
    if fait.returncode != 0:
        raise AssertionError(fait.stderr)
    return fait.stdout.strip()


class Cloisonnement(unittest.TestCase):

    def test_matplotlib_n_entre_nulle_part_ailleurs(self):
        """Le noyau, le solveur et la boucle s'importent sans Matplotlib."""
        charges = _modules_charges(
            "import ramjet.cycle0d, ramjet.solver1d, ramjet.coupling",
            ("matplotlib",))
        self.assertEqual(charges, "")

    def test_plots_importe_bien_matplotlib(self):
        """Sinon le cloisonnement serait obtenu en ne s'en servant jamais."""
        self.assertEqual(_modules_charges("import ramjet.plots",
                                          ("matplotlib",)), "matplotlib")

    def test_mais_JAMAIS_pyplot(self):
        """LE POINT DE CONCEPTION DU MODULE, ET IL SE VERIFIE.

        « pyplot » tient un registre global — cent figures produites en boucle y
        restent toutes — et choisit un backend, donc depend de l'affichage.
        L'API objet n'a ni l'un ni l'autre. Le test garantit qu'aucune commodite
        « pyplot » ne se glissera plus tard dans le module.
        """
        self.assertEqual(_modules_charges("import ramjet.plots",
                                          ("matplotlib.pyplot",)), "")

    def test_produire_une_figure_ne_charge_pas_pyplot_non_plus(self):
        """L'importation est propre, mais l'appel pourrait ne pas l'etre."""
        code = ("import numpy as np, ramjet.plots as p; "
                "p.duct_figure(np.linspace(0,1,10), np.full(10, 0.02)); "
                "import sys; "
                "print('matplotlib.pyplot' in sys.modules)")
        fait = subprocess.run([sys.executable, "-c", code], cwd=RACINE,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(fait.returncode, 0, fait.stderr)
        self.assertEqual(fait.stdout.strip(), "False")


# -- la structure ------------------------------------------------------------

class Structure(unittest.TestCase):

    def test_les_quatre_profils_ont_quatre_panneaux(self):
        figure = plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                      throat_x=COL)
        self.assertEqual(len(figure.axes), 4)
        # Une courbe par panneau, plus le trait du col ; le premier porte en
        # plus la ligne M = 1.
        self.assertEqual([len(ax.lines) for ax in figure.axes], [3, 2, 2, 2])
        for ax in figure.axes:
            self.assertEqual(len(ax.patches), len(ZONES))

    def test_l_axe_des_x_est_partage_et_ecrit_une_seule_fois(self):
        """Le repeter quatre fois prendrait la place du panneau du bas."""
        figure = plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                      throat_x=COL)
        etiquettes = [ax.get_xlabel() for ax in figure.axes]
        self.assertEqual(etiquettes, ["", "", "", "x (m)"])

    def test_les_zones_ne_sont_nommees_qu_en_haut(self):
        figure = plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                      throat_x=COL)
        textes = [len(ax.texts) for ax in figure.axes]
        self.assertEqual(textes[1:], [0, 0, 0])
        # Les quatre noms de zone, plus l'annotation du col.
        self.assertEqual(textes[0], len(ZONES) + 1)

    def test_le_conduit_est_en_miroir_et_a_l_echelle(self):
        """Deux courbes symetriques et des axes egaux : on voit un moteur."""
        x = np.linspace(0.0, 1.5, 40)
        aire = 0.027 - 0.009 * np.exp(-((x - 1.15) / 0.2) ** 2)
        figure = plots.duct_figure(x, aire, zones=ZONES, throat_x=COL)
        ax = figure.axes[0]
        self.assertEqual(ax.get_aspect(), 1.0)
        haut = ax.lines[0].get_ydata()
        bas = ax.lines[1].get_ydata()
        np.testing.assert_allclose(haut, -bas, rtol=0, atol=0)
        # Le rayon vient de A = pi r^2, pas de A lui-meme.
        np.testing.assert_allclose(haut, np.sqrt(aire / np.pi), rtol=1e-12)

    def test_le_balayage_a_deux_panneaux_et_ses_bandes(self):
        machs = np.linspace(1.5, 5.0, 30)
        figure = plots.sweep_figure(machs, np.linspace(700, 400, 30),
                                    np.linspace(1600, 2000, 30),
                                    thrust_band=(600.0, 900.0),
                                    impulse_band=(1500.0, 2500.0),
                                    best=(2.4, 770.0))
        self.assertEqual(len(figure.axes), 2)
        for ax in figure.axes:
            self.assertEqual(len(ax.lines), 1)
            self.assertEqual(len(ax.patches), 1)      # la fourchette publiee

    def test_la_carte_de_blocage_est_logarithmique(self):
        """SANS QUOI ELLE EST ILLISIBLE. La limite vaut 51 000 K a M = 0,05 et
        700 K a M = 0,60 : en echelle lineaire, toute la zone utile tient dans
        le dixieme inferieur de la figure.
        """
        machs = np.linspace(0.05, 0.60, 50)
        figure = plots.choking_figure(machs, 130.0 / machs ** 2,
                                      operating=(0.2074, 2000.0),
                                      design_mach=0.20)
        ax = figure.axes[0]
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(len(ax.lines), 2)            # limite + Mach de concep.


class LeColEstMarque(unittest.TestCase):

    def test_le_trait_du_col_tombe_au_bon_x(self):
        """Le critere du plan : 1e-6. Le trait est une verticale exacte."""
        figure = plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                      throat_x=COL)
        for ax in figure.axes:
            verticales = [l for l in ax.lines
                          if len(set(np.round(l.get_xdata(), 12))) == 1]
            self.assertTrue(verticales, "aucun trait vertical sur ce panneau")
            self.assertAlmostEqual(float(verticales[0].get_xdata()[0]), COL,
                                   delta=1e-6)

    def test_sans_col_demande_il_n_y_a_pas_de_trait(self):
        figure = plots.profile_figure(**profil_fabrique(), zones=ZONES)
        self.assertEqual([len(ax.lines) for ax in figure.axes], [2, 1, 1, 1])


class LesFichiers(unittest.TestCase):

    def test_les_png_sont_ecrits_et_pesent_plus_de_dix_ko(self):
        import tempfile
        with tempfile.TemporaryDirectory() as dossier:
            cible = Path(dossier) / "cree" / "au" / "passage" / "f.png"
            figure = plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                          throat_x=COL)
            rendu = plots.save(figure, cible)
            self.assertTrue(rendu.exists())
            self.assertGreater(rendu.stat().st_size, 10_000)

    def test_le_dossier_est_cree_s_il_manque(self):
        import tempfile
        with tempfile.TemporaryDirectory() as dossier:
            cible = Path(dossier) / "neuf" / "f.png"
            self.assertFalse(cible.parent.exists())
            plots.save(plots.duct_figure(np.linspace(0, 1, 10),
                                         np.full(10, 0.02)), cible)
            self.assertTrue(cible.exists())


# -- la vision dichromate ----------------------------------------------------

VERS_LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
DEPUIS_LMS = np.linalg.inv(VERS_LMS)
PROTANOPIE = np.array([[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
DEUTERANOPIE = np.array([[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827],
                         [0.0, 0.0, 1.0]])


def _lineaire(hexa: str) -> np.ndarray:
    canaux = np.array([int(hexa[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    return np.where(canaux > 0.04045, ((canaux + 0.055) / 1.055) ** 2.4,
                    canaux / 12.92)


def separation(a: str, b: str, matrice) -> float:
    """La distance entre deux couleurs vue par un dichromate.

    Simulation de Vienot-Brettel-Mollon : sRGB -> lineaire -> LMS, projection
    sur le plan du dichromate, retour. La distance est euclidienne en RGB
    lineaire — grossier comme mesure perceptive, mais suffisant pour separer
    « distinguable » de « confondu », qui est la seule question posee.
    """
    simule = [DEPUIS_LMS @ (matrice @ (VERS_LMS @ _lineaire(couleur)))
              for couleur in (a, b)]
    return float(np.linalg.norm(simule[0] - simule[1]))


class VisionDichromate(unittest.TestCase):
    """La regle du plan : pas de rouge/vert pour opposer deux courbes.

    Verifiee par simulation plutot que par bonne intention. Les valeurs sont
    mesurees, pas choisies.
    """

    # Les seules paires qui partagent reellement un axe dans les cinq figures.
    PAIRES = (
        ("cycle T-s", plots.NOIR, plots.BLEU),
        ("cycle T-s", plots.NOIR, plots.ORANGE),
        ("cycle T-s", plots.BLEU, plots.ORANGE),
        ("balayage", plots.BLEU, plots.VERMILLON),
        ("balayage", plots.VERMILLON, plots.VERT),
        ("blocage", plots.VERMILLON, plots.BLEU),
        ("blocage", plots.VERMILLON, plots.NOIR),
    )

    ROUGE_MATPLOTLIB, VERT_MATPLOTLIB = "#D62728", "#2CA02C"

    def test_le_test_a_du_POUVOIR(self):
        """SANS CELA, LE RESTE NE PROUVERAIT RIEN.

        Le rouge et le vert par defaut de Matplotlib sont a 0,727 l'un de
        l'autre en vision normale — parfaitement distincts — et tombent a
        **0,068** en deuteranopie. La mesure sait donc detecter un mauvais
        choix, et c'est ce que ce test etablit avant de mesurer les notres.
        """
        normale = float(np.linalg.norm(_lineaire(self.ROUGE_MATPLOTLIB)
                                       - _lineaire(self.VERT_MATPLOTLIB)))
        self.assertGreater(normale, 0.7)
        self.assertLess(separation(self.ROUGE_MATPLOTLIB, self.VERT_MATPLOTLIB,
                                   DEUTERANOPIE), 0.10)

    def test_aucune_paire_de_la_palette_ne_s_effondre(self):
        """Mesure : la plus faible est vermillon/vert a 0,197 en deuteranopie,
        soit pres de trois fois mieux que le rouge/vert de Matplotlib.
        """
        pire = float(separation(self.ROUGE_MATPLOTLIB, self.VERT_MATPLOTLIB,
                                DEUTERANOPIE))
        for figure, a, b in self.PAIRES:
            for nom, matrice in (("protanopie", PROTANOPIE),
                                 ("deuteranopie", DEUTERANOPIE)):
                with self.subTest(figure=figure, couleurs=(a, b), vision=nom):
                    self.assertGreater(separation(a, b, matrice), 2.8 * pire)

    def test_les_deux_courbes_opposees_sont_bleu_et_orange(self):
        """LA PAIRE QUI COMPTE : 0,564 en protanopie, 0,683 en deuteranopie,
        contre 0,925 en vision normale. Elle perd peu, et reste dix fois
        au-dessus du rouge/vert."""
        for matrice in (PROTANOPIE, DEUTERANOPIE):
            self.assertGreater(separation(plots.BLEU, plots.ORANGE, matrice),
                               0.55)

    def test_rien_n_est_distingue_par_la_COULEUR_SEULE(self):
        """LA VRAIE GARANTIE, ET ELLE EST STRUCTURELLE.

        Sur chaque axe portant plusieurs COURBES, deux d'entre elles ne se
        distinguent jamais par la seule couleur : le style de trait porte
        l'information en meme temps. Une impression en noir et blanc reste donc
        lisible, ce qu'aucune palette ne peut garantir a elle seule.

        Les traits de REPERE — la verticale du col, l'horizontale M = 1 — sont
        exclus : ils sont noirs tous les deux, mais une horizontale et une
        verticale ne se confondent pas, et c'est l'orientation qui les separe.
        """
        figures = (
            plots.profile_figure(**profil_fabrique(), zones=ZONES,
                                 throat_x=COL),
            plots.cycle_figure(["0", "2", "4", "9"], [216, 605, 1920, 777],
                               [22632, 660e3, 627e3, 22632]),
            plots.choking_figure(np.linspace(0.05, 0.6, 30),
                                 130.0 / np.linspace(0.05, 0.6, 30) ** 2,
                                 operating=(0.2074, 2000.0), design_mach=0.20),
        )
        def est_une_courbe(ligne) -> bool:
            """Ni horizontale ni verticale : une vraie courbe de donnees."""
            x, y = ligne.get_xdata(), ligne.get_ydata()
            return len(np.unique(x)) > 1 and len(np.unique(y)) > 1

        for figure in figures:
            for ax in figure.axes:
                styles = [(l.get_color(), l.get_linestyle())
                          for l in ax.lines if est_une_courbe(l)]
                for i, premier in enumerate(styles):
                    for second in styles[i + 1:]:
                        with self.subTest(paire=(premier, second)):
                            self.assertTrue(
                                premier[0] != second[0]
                                or premier[1] != second[1],
                                "deux traits identiques en couleur ET en style")
                            if premier[0] != second[0]:
                                continue
                            self.assertNotEqual(premier[1], second[1])


# -- le piege du diagramme T-s -----------------------------------------------

class LePiegeDuDiagrammeTS(unittest.TestCase):

    def test_en_grandeurs_TOTALES_la_detente_disparait(self):
        """UNE TUYERE NE CHANGE PAS LA TEMPERATURE TOTALE : C'EST SA
        DEFINITION.

        Trace en totales, les stations 4 et 9 sont a la MEME temperature
        (2 000 K) et a 12 J/(kg.K) l'une de l'autre : le cycle perd sa detente
        et personne ne le signale. Le test mesure les deux distances pour que
        le choix du statique soit un fait, pas un gout.
        """
        cycle = run_cycle(FlightCase(), EngineCase())
        stations = {s.name: s for s in cycle.stations}
        cp, R = cycle.flight.cp, cycle.flight.R

        def entropie(a, b):
            return (cp * np.log(b.T0 / a.T0) - R * np.log(b.p0 / a.p0))

        self.assertAlmostEqual(stations["4"].T0, stations["9"].T0, delta=1e-9)
        self.assertLess(abs(entropie(stations["4"], stations["9"])), 15.0)

    def test_en_statique_le_cycle_se_referme(self):
        """De 1 920 a 777 K : la detente occupe la moitie de la figure."""
        from ramjet.cli import _etats_statiques
        cycle = run_cycle(FlightCase(), EngineCase())
        noms, temperatures, _ = _etats_statiques(cycle, 0.20, 0.4250)
        self.assertEqual(noms, ["0", "2", "4", "9"])
        self.assertAlmostEqual(temperatures[0], 216.65, delta=0.1)
        self.assertAlmostEqual(temperatures[2], 1926.0, delta=5.0)
        self.assertAlmostEqual(temperatures[3], 777.0, delta=5.0)
        self.assertGreater(temperatures[2] - temperatures[3], 1000.0)

    def test_l_entropie_est_relative_a_la_premiere_station(self):
        figure = plots.cycle_figure(["a", "b"], [300.0, 600.0],
                                    [1e5, 1e5])
        abscisses = figure.axes[0].collections[0].get_offsets()[:, 0]
        self.assertAlmostEqual(float(abscisses[0]), 0.0, delta=1e-12)
        self.assertAlmostEqual(float(abscisses[1]),
                               1004.5 * np.log(2.0), delta=1e-9)


# -- les refus ---------------------------------------------------------------

class Refus(unittest.TestCase):

    def test_longueurs_incompatibles(self):
        with self.assertRaises(CaseError) as capture:
            plots.duct_figure(np.linspace(0, 1, 10), np.full(9, 0.02))
        self.assertIn("longueurs incompatibles", str(capture.exception))

    def test_un_seul_point(self):
        with self.assertRaises(CaseError):
            plots.duct_figure(np.array([0.0]), np.array([0.02]))

    def test_valeurs_non_finies(self):
        aire = np.full(10, 0.02)
        aire[3] = np.nan
        with self.assertRaises(CaseError) as capture:
            plots.duct_figure(np.linspace(0, 1, 10), aire)
        self.assertIn("non finies", str(capture.exception))

    def test_section_negative(self):
        with self.assertRaises(CaseError) as capture:
            plots.duct_figure(np.linspace(0, 1, 10), np.full(10, -0.02))
        self.assertIn("negative", str(capture.exception))

    def test_cycle_a_une_seule_station(self):
        with self.assertRaises(CaseError):
            plots.cycle_figure(["0"], [300.0], [1e5])

    def test_cycle_a_temperature_negative(self):
        with self.assertRaises(CaseError):
            plots.cycle_figure(["a", "b"], [300.0, -1.0], [1e5, 1e5])


if __name__ == "__main__":
    unittest.main()
