"""L'adaptation de la tuyere hors du point de dessin.

LA QUESTION OPERATIONNELLE : un statoreacteur vole sur un DOMAINE, pas en un
point, et sa geometrie est coulee dans le metal. Le rapport « A9/A_col » fixe
donc le Mach de sortie une fois pour toutes, et la pression de sortie ne s'ajuste
pas a l'ambiante.

LE RESULTAT EST CONTRE-INTUITIF, ET C'EST TOUT L'INTERET. On s'attend a ce que
l'adaptation depende de l'ALTITUDE — c'est le cas d'une fusee, dont la pression
de chambre ne suit pas l'ambiante. Pour un statoreacteur elle n'en depend
PRESQUE PAS : elle depend du MACH. Les tests mesurent les deux et montrent d'ou
vient la difference.

ET UN DEFAUT DU MODELE APPARAIT ICI, herite de l'etape 8 : la perte de poussee
devient legerement negative pres de l'adaptation, ce qui est impossible.
"""

import unittest

import numpy as np

from ramjet.atmosphere import state
from ramjet.config import EngineCase, FlightCase
from ramjet.coupling import (SEPARATION_RATIO, Adaptation, adaptation,
                             converge, flight_envelope, run_engine)

DESSIN_MACH, DESSIN_ALTITUDE = 3.0, 11000.0


class GeometrieFigee(unittest.TestCase):
    """Le conduit est dessine UNE fois, puis vole ailleurs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.point = converge(FlightCase(mach=DESSIN_MACH,
                                        altitude=DESSIN_ALTITUDE),
                             EngineCase())
        cls.duct = cls.point.engine.duct

    def au_point(self, mach: float = DESSIN_MACH,
                 altitude: float = DESSIN_ALTITUDE) -> Adaptation:
        return adaptation(run_engine(FlightCase(mach=mach, altitude=altitude),
                                     EngineCase(), duct=self.duct))


class LAdaptationNeDependPasDeLAltitude(GeometrieFigee):

    def test_dans_la_tropopause_elle_est_EXACTEMENT_constante(self):
        """LE RESULTAT DE L'ETAPE, ET IL EST EXACT, PAS APPROCHE.

        Entre 11 et 20 km, l'atmosphere standard est ISOTHERME. La compression
        dynamique rend alors « p04 » rigoureusement proportionnelle a
        l'ambiante, et le rapport « p9/p_amb » s'en trouve independant de
        l'altitude — au dernier chiffre, pas « a peu pres ». Mesure : 1,0204 a
        11, 15 et 20 km, et la meme poussee specifique.
        """
        valeurs = [self.au_point(altitude=z) for z in (11000.0, 15000.0,
                                                       20000.0)]
        for autre in valeurs[1:]:
            self.assertAlmostEqual(autre.exit_pressure_ratio,
                                   valeurs[0].exit_pressure_ratio, delta=1e-12)
            self.assertAlmostEqual(autre.engine.specific_thrust,
                                   valeurs[0].engine.specific_thrust,
                                   delta=1e-9)

    def test_et_la_temperature_est_bien_la_SEULE_cause_du_reste(self):
        """Sous 11 km et au-dessus de 20, T_ambiante varie — et c'est la seule
        chose qui bouge. Mesure : 1,0308 au sol contre 1,0204 en tropopause,
        soit 1 % sur 11 km d'altitude. La pression, elle, a change d'un
        facteur 4,5 sans effet.
        """
        sol, tropopause = self.au_point(altitude=0.0), self.au_point()
        self.assertAlmostEqual(sol.exit_pressure_ratio, 1.0308, delta=1e-3)
        self.assertLess(abs(sol.exit_pressure_ratio
                            / tropopause.exit_pressure_ratio - 1.0), 0.02)
        # Et pourtant la pression ambiante a ete divisee par 4,5.
        self.assertGreater(state(0.0)["p"] / state(11000.0)["p"], 4.0)

    def test_la_perte_est_plate_sur_tout_le_domaine_d_altitude(self):
        """Moins de 0,05 % de 0 a 25 km : c'est ce que la figure montre en
        gardant la MEME echelle verticale que le panneau en Mach."""
        pertes = [self.au_point(altitude=z).loss
                  for z in (0.0, 5000.0, 11000.0, 20000.0, 25000.0)]
        self.assertLess(max(pertes), 5e-4)


class MaisElleDependDuMach(GeometrieFigee):

    def test_la_perte_passe_par_un_minimum_pres_du_point_de_dessin(self):
        """23,2 % a M0 = 2, 0,003 % a M0 = 3, 7,6 % a M0 = 4. La courbe est en
        U, et son fond est la ou le moteur a ete dessine."""
        pertes = {mach: self.au_point(mach=mach).loss
                  for mach in (2.0, 2.5, 3.0, 3.5, 4.0)}
        self.assertAlmostEqual(pertes[2.0], 0.2321, delta=5e-3)
        self.assertAlmostEqual(pertes[4.0], 0.0759, delta=5e-3)
        self.assertLess(pertes[3.0], 1e-3)
        for mach in (2.0, 2.5, 3.5, 4.0):
            with self.subTest(mach=mach):
                self.assertGreater(pertes[mach], pertes[3.0])

    def test_mais_le_minimum_n_est_pas_EXACTEMENT_au_point_de_dessin(self):
        """ET LA RAISON REMONTE A L'ETAPE 8.

        « p9 = p_ambiant » tombe a **M0 = 2,98**, pas a 3,00. La geometrie a ete
        dimensionnee par le 0D pour 10,000 kg/s, et le col en impose 10,351 :
        A9 est donc legerement trop petite pour le debit reel, et il faut voler
        un peu moins vite pour que la detente se termine juste.
        """
        machs = np.arange(2.90, 3.05, 0.02)
        rapports = np.array([self.au_point(mach=float(m)).exit_pressure_ratio
                             for m in machs])
        croisement = float(np.interp(1.0, rapports, machs))
        self.assertAlmostEqual(croisement, 2.985, delta=0.01)
        self.assertLess(croisement, DESSIN_MACH)

    def test_sous_le_Mach_de_dessin_la_tuyere_est_SUR_detendue(self):
        """p9 < p_amb : le terme de pression devient NEGATIF et mange la
        poussee. A M0 = 2, il vaut -73 % de la poussee."""
        basse = self.au_point(mach=2.0)
        self.assertFalse(basse.under_expanded)
        self.assertLess(basse.pressure_term, 0.0)
        self.assertLess(basse.pressure_term / basse.engine.thrust, -0.5)

    def test_au_dessus_elle_est_SOUS_detendue(self):
        """p9 > p_amb : la detente n'est pas finie a la sortie. Le terme est
        positif, mais on perd quand meme — la poussee qu'on aurait eue avec un
        divergent plus long est plus grande."""
        haute = self.au_point(mach=4.0)
        self.assertTrue(haute.under_expanded)
        self.assertGreater(haute.pressure_term, 0.0)
        self.assertGreater(haute.loss, 0.05)
        # Il faudrait une section de sortie 2,3 fois plus grande.
        self.assertLess(haute.area_ratio_to_adapted, 0.5)


class LeDecollementBorneLaValidite(GeometrieFigee):

    def test_sous_M0_egal_2_3_le_modele_decrit_un_ecoulement_INEXISTANT(self):
        """LE POINT LE PLUS IMPORTANT DU FICHIER.

        A M0 = 2, le modele calcule tranquillement un terme de pression de
        -73 % de la poussee. Mais « p9/p_amb = 0,246 » est bien sous le seuil de
        decollement de Summerfield (0,4) : la couche limite se decolle du
        divergent, et l'ecoulement calcule n'existe pas. Le decollement est un
        effet visqueux 2D, explicitement hors perimetre — le modele ne peut pas
        le calculer, mais il peut DIRE ou il cesse d'etre valable.
        """
        for mach, decolle in ((2.0, True), (2.2, True), (2.4, False),
                              (3.0, False)):
            with self.subTest(mach=mach):
                self.assertIs(self.au_point(mach=mach).separated, decolle)

    def test_et_il_le_DIT_dans_son_compte_rendu(self):
        texte = self.au_point(mach=2.0).describe()
        self.assertIn("ATTENTION", texte)
        self.assertIn("decolle", texte)
        self.assertIn("n'existe pas", texte)
        self.assertNotIn("ATTENTION", self.au_point().describe())

    def test_la_frontiere_tombe_vers_M0_egal_2_33(self):
        machs = np.arange(2.0, 2.6, 0.05)
        rapports = np.array([self.au_point(mach=float(m)).exit_pressure_ratio
                             for m in machs])
        frontiere = float(np.interp(SEPARATION_RATIO, rapports, machs))
        self.assertAlmostEqual(frontiere, 2.33, delta=0.05)


class LeDefautDeLEtape8_EST_CORRIGE(GeometrieFigee):
    """CE FICHIER A D'ABORD SERVI A DOCUMENTER UN DEFAUT ; IL ATTESTE
    MAINTENANT SA CORRECTION.

    Tant que le « (1+f) » etait pose a la main, la perte de poussee devenait
    NEGATIVE de 0,019 % vers M0 = 2,94, ce qui est thermodynamiquement
    impossible : une tuyere adaptee est optimale. La cause etait que le terme de
    pression se calculait avec le debit d'AIR et celui de quantite de mouvement
    avec le debit de SORTIE — 3,5 % d'ecart. Le quatrieme terme source de l'EDO
    a ferme le trou.
    """

    def test_la_perte_de_poussee_est_POSITIVE_partout(self):
        """Y COMPRIS DANS LA FENETRE OU ELLE ETAIT NEGATIVE.

        Mesure sur [2,90 ; 3,09] au pas de 0,01 : le minimum vaut
        **+2,8e-06**, atteint pres du point ou « p9 = p_amb ». Pas un seul
        point negatif.
        """
        pertes = [self.au_point(mach=float(m)).loss
                  for m in np.arange(2.90, 3.10, 0.01)]
        self.assertGreater(min(pertes), 0.0)
        self.assertLess(min(pertes), 1e-4)

    def test_et_celle_sur_l_IMPULSION_aussi(self):
        """Les deux lectures de la meme question sont maintenant d'accord :
        celle qui fait intervenir la vitesse de vol et celle qui ne la fait pas.
        """
        for mach in (2.5, 2.9, 2.94, 2.98, 3.0, 3.5, 4.0):
            with self.subTest(mach=mach):
                valeur = self.au_point(mach=mach)
                self.assertGreaterEqual(valeur.impulse_loss, 0.0)
                self.assertGreaterEqual(valeur.loss, 0.0)

    def test_le_debit_de_SORTIE_est_celui_qui_sert_aux_deux_termes(self):
        """LA CAUSE DE LA CORRECTION, VERIFIEE DIRECTEMENT.

        Le terme de quantite de mouvement et le terme de pression doivent
        parler du meme debit. Le profil porte desormais « mdot(x) », et sa
        derniere valeur vaut « mdot_entree (1+f) » exactement.
        """
        valeur = self.au_point()
        moteur = valeur.engine
        self.assertAlmostEqual(
            float(moteur.profile.mass_flow[-1]) / moteur.mass_flow,
            1.0 + moteur.cycle.fuel_air_ratio, delta=1e-12)


class LeBalayageDuDomaine(GeometrieFigee):

    def test_il_rend_les_deux_coupes_et_signale_les_refus(self):
        par_mach, par_altitude, refuses = flight_envelope(self.duct)
        self.assertEqual(len(par_mach), 5)
        self.assertEqual(len(par_altitude), 6)
        self.assertEqual(refuses, ())
        for _, valeur in par_mach + par_altitude:
            self.assertIsInstance(valeur, Adaptation)

    def test_un_point_impossible_est_ECARTE_et_nomme_pas_propage(self):
        """Comme le balayage 0D de l'etape 4 : savoir OU le domaine s'arrete
        est une information, pas un incident.

        LE DOMAINE NE CASSE QUE TRES HAUT, et pas la ou je le croyais. J'avais
        parie sur M0 = 1,2 ; ce point PASSE. La premiere borne dure est
        **M0 = 6,5**, ou « T02 = 2 047 K » depasse deja le « T04 » demande : il
        n'y a plus rien a bruler. C'est la limite thermique du statoreacteur,
        deja calculee analytiquement a l'etape 4.
        """
        par_mach, _, refuses = flight_envelope(self.duct, machs=(3.0, 6.5),
                                               altitudes=(11000.0,))
        self.assertEqual(len(par_mach), 1)
        self.assertEqual(len(refuses), 1)
        self.assertIn("M0 = 6.5", refuses[0][0])
        self.assertIn("plus rien a bruler", refuses[0][1])

    def test_sous_M0_egal_1_4_le_moteur_FREINE_au_lieu_de_pousser(self):
        """ET LE MODELE LE DIT CORRECTEMENT, CE QUI EST RASSURANT.

        A M0 = 1,05, la poussee specifique vaut **-648 N.s/kg** : la sur-detente
        est telle que le terme de pression mange toute l'impulsion, et le moteur
        est un frein. C'est physiquement juste pour un statoreacteur tres en
        dessous de son point de dessin — et le modele signale de surcroit que
        l'ecoulement y est decolle, donc que le nombre lui-meme n'est pas a
        prendre au pied de la lettre.
        """
        for mach, pousse in ((1.05, False), (1.2, False), (1.5, True),
                             (3.0, True)):
            with self.subTest(mach=mach):
                valeur = adaptation(run_engine(FlightCase(mach=mach),
                                               EngineCase(), duct=self.duct))
                self.assertIs(valeur.engine.specific_thrust > 0.0, pousse)
                if not pousse:
                    self.assertTrue(valeur.separated)


class LaFigure(unittest.TestCase):

    def test_les_deux_panneaux_partagent_l_echelle_verticale(self):
        """SANS QUOI LE PANNEAU PLAT MENTIRAIT. Dilater l'axe du bas rendrait
        la courbe d'altitude « interessante » alors que le fait a montrer est
        justement qu'elle ne l'est pas."""
        from ramjet import plots

        figure = plots.adaptation_figure(
            [2.0, 3.0, 4.0], [0.21, 0.0002, 0.08],
            [0.0, 11000.0, 20000.0], [3e-4, 2e-4, 2e-4],
            pressure_ratio_by_mach=[0.25, 1.02, 3.5], separation=0.4,
            design_mach=3.0)
        haut, bas = figure.axes
        self.assertEqual(haut.get_ylim(), bas.get_ylim())

    def test_la_zone_de_decollement_est_INTERPOLEE_pas_prise_sur_la_grille(self):
        """UN BOGUE ATTRAPE EN REGARDANT LA FIGURE.

        Marquer « jusqu'au dernier point decolle » donne une bande de largeur
        NULLE des qu'un seul point de la grille l'est — c'est-a-dire exactement
        dans le cas qu'on voulait montrer. La frontiere est donc interpolee.
        """
        from ramjet import plots

        figure = plots.adaptation_figure(
            [2.0, 3.0, 4.0], [0.21, 0.0002, 0.08],
            [0.0, 20000.0], [3e-4, 2e-4],
            pressure_ratio_by_mach=[0.25, 1.02, 3.5], separation=0.4,
            design_mach=3.0)
        bandes = figure.axes[0].patches
        self.assertEqual(len(bandes), 1)
        self.assertGreater(float(bandes[0].get_width()), 0.1)
        # La frontiere interpolee tombe entre les deux points de grille qui
        # encadrent le seuil, et pas sur l'un d'eux.
        droite = float(bandes[0].get_x() + bandes[0].get_width())
        self.assertGreater(droite, 2.0)
        self.assertLess(droite, 3.0)


if __name__ == "__main__":
    unittest.main()
