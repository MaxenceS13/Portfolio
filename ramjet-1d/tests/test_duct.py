"""La geometrie : continuite, derivee, dimensionnement, et un piege trouve ici.

DEUX FAMILLES DE TESTS. Les premiers verifient la GEOMETRIE elle-meme —
continuite, derivee analytique, monotonies — et ne demandent aucune physique.
Les seconds verifient le DIMENSIONNEMENT depuis le 0D, et c'est la que se
trouve le pont entre les deux modeles du projet.
"""

import unittest

import numpy as np

from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.cycle0d import run_cycle
from ramjet.duct import (DEFAULT_LENGTHS, Duct, Zone, area_for_mach,
                         chamber_exit_mach, choked_area, from_cycle,
                         mass_flow_parameter, static_state)
from ramjet.thermo import area_ratio

DEBIT = 10.0


def conduit(**kwargs) -> tuple:
    cycle = run_cycle(FlightCase(), EngineCase(**kwargs))
    return cycle, from_cycle(cycle, mass_flow=DEBIT)


class Continuite(unittest.TestCase):
    """Les raccords cubiques, et ce qu'ils garantissent."""

    def setUp(self):
        self.cycle, self.duct = conduit()

    def test_la_section_est_continue_aux_jonctions(self):
        for x in self.duct.stations[1:-1]:
            with self.subTest(x=x):
                avant, apres = float(self.duct.area(x - 1e-9)), float(
                    self.duct.area(x + 1e-9))
                self.assertAlmostEqual(avant / apres, 1.0, delta=1e-12)

    def test_la_derivee_est_EXACTEMENT_nulle_a_toutes_les_jonctions(self):
        """C'est plus fort que la continuite demandee par le plan. Le
        « smoothstep » a une derivee nulle aux deux bouts de chaque zone, donc
        le raccord est C1 sans qu'on ait rien a imposer — et la chambre a
        section constante s'y raccorde sans rupture.
        """
        for x in self.duct.stations:
            with self.subTest(x=x):
                self.assertEqual(abs(float(self.duct.area_derivative(x))), 0.0)

    def test_la_derivee_analytique_contre_les_differences_finies(self):
        """Mesure : ecart relatif maximal 1,5e-10 sur 200 points, pour 1e-8
        demande. Une derivee fausse ne se verrait pas sur le trace de A(x), et
        elle entre pourtant DIRECTEMENT dans l'equation du solveur 1D."""
        xs = np.linspace(0.001, self.duct.length - 0.001, 200)
        h = 1e-7
        analytique = self.duct.area_derivative(xs)
        finies = (self.duct.area(xs + h) - self.duct.area(xs - h)) / (2 * h)
        ecart = np.abs(analytique - finies) / (1.0 + np.abs(analytique))
        self.assertLess(float(ecart.max()), 1e-8)


class Monotonies(unittest.TestCase):
    """Ce que la forme d'un statoreacteur impose."""

    def setUp(self):
        self.cycle, self.duct = conduit()

    def test_le_diffuseur_se_resserre(self):
        xs = np.linspace(0.0, 0.40, 50)
        self.assertTrue(np.all(np.diff(self.duct.area(xs)) <= 0.0))

    def test_la_chambre_est_a_section_constante_AU_BIT_PRES(self):
        xs = np.linspace(0.40, 1.00, 200)
        self.assertEqual(float(np.ptp(self.duct.area(xs))), 0.0)

    def test_la_tuyere_se_resserre_puis_s_ouvre(self):
        avant = self.duct.area(np.linspace(1.00, self.duct.throat_x, 50))
        apres = self.duct.area(np.linspace(self.duct.throat_x,
                                           self.duct.length, 50))
        self.assertTrue(np.all(np.diff(avant) <= 0.0))
        self.assertTrue(np.all(np.diff(apres) >= 0.0))

    def test_le_col_est_le_minimum_de_section(self):
        """Le minimum d'une grille de 1 000 points depasse le col de 2,5e-8 :
        c'est l'echantillonnage qui rate x = 1,15, pas le conduit. Le test
        verifie donc les deux choses justes — aucun point sous le col, et le
        col atteint des qu'on l'echantillonne.
        """
        xs = np.linspace(0.0, self.duct.length, 1000)
        sections = self.duct.area(xs)
        self.assertGreaterEqual(float(sections.min()), self.duct.throat_area)
        self.assertAlmostEqual(float(sections.min()) / self.duct.throat_area,
                               1.0, delta=1e-5)
        self.assertAlmostEqual(self.duct.throat_x, 1.15, delta=1e-9)
        self.assertEqual(float(self.duct.area(self.duct.throat_x)),
                         self.duct.throat_area)

    def test_aucune_section_n_est_nulle_ou_negative(self):
        sections = self.duct.area(np.linspace(0.0, self.duct.length, 1000))
        self.assertGreater(float(sections.min()), 0.0)


class Dimensionnement(unittest.TestCase):
    """LE CRITERE DE L'ETAPE : « mdot = rho u A » a chaque section imposee."""

    def setUp(self):
        self.cycle, self.duct = conduit()

    def test_la_section_de_capture_porte_le_debit_demande(self):
        etat = static_state(self.cycle, self.cycle.flight.mach, "0")
        debit = etat["rho"] * etat["u"] * float(self.duct.area(0.0))
        self.assertAlmostEqual(debit, DEBIT, delta=1e-9)

    def test_la_chambre_porte_le_debit_au_mach_vise(self):
        etat = static_state(self.cycle, self.cycle.engine.M_combustor, "2")
        debit = etat["rho"] * etat["u"] * float(self.duct.area(0.40))
        self.assertAlmostEqual(debit, DEBIT, delta=1e-9)

    def test_le_col_bloque_EXACTEMENT_le_debit_chaud(self):
        """Le col se deduit du debit bloque, qui est analytique — jamais d'une
        inversion de A/A*, mal conditionnee pres de M = 1 (lecon de l'etape 2).
        """
        etat = static_state(self.cycle, 1.0, "4")
        debit = etat["rho"] * etat["u"] * self.duct.throat_area
        attendu = DEBIT * (1.0 + self.cycle.fuel_air_ratio)
        self.assertAlmostEqual(debit / attendu, 1.0, delta=1e-10)

    def test_la_section_de_sortie_donne_le_mach_de_sortie_du_cycle(self):
        """A9/A_col doit valoir A/A*(M9) : mesure 3,5419 des deux cotes."""
        rapport = float(self.duct.area(self.duct.length)) / self.duct.throat_area
        attendu = float(area_ratio(self.cycle.mach_exit, self.cycle.flight.gamma))
        self.assertAlmostEqual(rapport, attendu, delta=1e-12)
        self.assertAlmostEqual(rapport, 3.5419, delta=1e-3)

    def test_les_dimensions_du_cas_de_reference(self):
        """Mesure a 10 kg/s : capture 310,4 cm2, chambre 268,6, col 179,3,
        sortie 635,1."""
        for x, attendu in ((0.0, 310.4), (0.7, 268.6), (1.15, 179.3),
                           (1.5, 635.1)):
            with self.subTest(x=x):
                self.assertAlmostEqual(float(self.duct.area(x)) * 1e4, attendu,
                                       delta=0.2)

    def test_les_sections_grandissent_avec_le_debit(self):
        """Le debit est la SEULE echelle de la geometrie : doubler le debit
        doit doubler toutes les sections."""
        double = from_cycle(self.cycle, mass_flow=2.0 * DEBIT)
        for x in (0.0, 0.7, 1.15, 1.5):
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    float(double.area(x)) / float(self.duct.area(x)), 2.0,
                    delta=1e-12)


class LApportDeMasseDuCarburant(unittest.TestCase):
    """Le piege trouve a cette etape, et il pese plus que ce qu'on cherchait.

    La relation de Rayleigh, qui servira d'oracle au solveur 1D, suppose le
    debit CONSTANT. Le moteur, lui, ajoute 3,5 % de masse. L'ecart sur le Mach
    de sortie de chambre atteint 4,5 % — soit PLUS que l'effet de pi_b que
    l'etape 9 cherchera a mesurer.
    """

    def setUp(self):
        # pi_b pris egal a la valeur que Rayleigh calcule pour ce cas, de sorte
        # que les deux modeles decrivent la MEME chambre.
        self.cycle, self.duct = conduit(pi_b=0.9280)

    def test_sans_l_apport_de_masse_la_geometrie_est_d_accord_avec_Rayleigh(self):
        """Mesure : M4 = 0,4250 contre 0,4249 pour Rayleigh — 1e-4 d'ecart.
        La geometrie est donc juste ; c'est bien la masse qui fait la
        difference."""
        mach = chamber_exit_mach(self.duct, self.cycle, DEBIT,
                                 include_fuel_mass=False)
        self.assertAlmostEqual(mach, 0.4249, delta=1e-3)

    def test_avec_l_apport_de_masse_l_ecart_atteint_quatre_pour_cent(self):
        """Mesure : 0,4442 au lieu de 0,4250."""
        avec = chamber_exit_mach(self.duct, self.cycle, DEBIT)
        sans = chamber_exit_mach(self.duct, self.cycle, DEBIT,
                                 include_fuel_mass=False)
        self.assertAlmostEqual(avec, 0.4442, delta=1e-3)
        self.assertGreater((avec - sans) / sans, 0.04)

    def test_il_pese_plus_que_la_correction_que_l_etape_9_appliquera(self):
        """Et la comparaison a d'abord ete faite contre le mauvais intervalle.

        Premier jet : « l'apport de masse pese plus qu'un balayage de pi_b de
        0,95 a 0,90 ». Faux — mesure, ce balayage large vaut 0,0314 contre
        0,0192 pour la masse. Mais ce n'est pas l'intervalle pertinent :
        l'etape 9 corrigera pi_b de 0,95 (suppose) a 0,928 (calcule), soit
        0,0132 sur le Mach de sortie de chambre. L'apport de masse, lui, en
        vaut 0,0192 — donc une fois et demie la correction recherchee.

        C'est ce qui rend la remarque utile : negliger la masse fausserait le
        couplage davantage que ce qu'il pretend corriger.
        """
        cycle_suppose, duct_suppose = conduit(pi_b=0.95)
        correction_pi_b = abs(
            chamber_exit_mach(duct_suppose, cycle_suppose, DEBIT)
            - chamber_exit_mach(self.duct, self.cycle, DEBIT))
        effet_masse = abs(chamber_exit_mach(self.duct, self.cycle, DEBIT)
                          - chamber_exit_mach(self.duct, self.cycle, DEBIT,
                                              include_fuel_mass=False))
        self.assertAlmostEqual(correction_pi_b, 0.0132, delta=1e-3)
        self.assertAlmostEqual(effet_masse, 0.0192, delta=1e-3)
        self.assertGreater(effet_masse, correction_pi_b)

    def test_le_mach_de_chambre_monte_de_l_entree_a_la_sortie(self):
        """L'apport de chaleur accelere un ecoulement subsonique : c'est le sens
        de Rayleigh, et le Mach doit monter de 0,20 a 0,42-0,44."""
        entree = self.cycle.engine.M_combustor
        sortie = chamber_exit_mach(self.duct, self.cycle, DEBIT)
        self.assertGreater(sortie, entree)
        self.assertLess(sortie, 1.0)


class FonctionsDeDimensionnement(unittest.TestCase):
    """Les trois briques analytiques, testees seules."""

    def test_le_parametre_de_debit_bloque(self):
        """Gamma = 0,0404149 pour gamma = 1,4 et R = 287,05 — la constante qui
        permet de dimensionner un col sans iterer, et celle que les tables de
        debit bloque publient a 0,0404."""
        self.assertAlmostEqual(mass_flow_parameter(1.4, 287.05), 0.0404149,
                               delta=1e-7)

    def test_la_section_bloquee_est_proportionnelle_au_debit(self):
        une = choked_area(1.0, 2000.0, 8e5, 1.4, 287.05)
        deux = choked_area(2.0, 2000.0, 8e5, 1.4, 287.05)
        self.assertAlmostEqual(deux / une, 2.0, delta=1e-15)

    def test_la_section_a_mach_un_est_la_section_bloquee(self):
        """« area_for_mach » a M = 1 doit rendre exactement « choked_area » :
        A/A*(1) vaut 1."""
        bloquee = choked_area(5.0, 1500.0, 3e5, 1.4, 287.05)
        self.assertAlmostEqual(area_for_mach(5.0, 1.0, 1500.0, 3e5, 1.4, 287.05),
                               bloquee, delta=1e-15)

    def test_un_debit_non_positif_est_refuse(self):
        for debit in (0.0, -1.0):
            with self.subTest(debit=debit):
                with self.assertRaises(CaseError):
                    choked_area(debit, 1000.0, 1e5, 1.4, 287.05)


class Refus(unittest.TestCase):
    """Ce que la geometrie refuse de construire."""

    def test_un_mach_de_chambre_trop_grand_supprime_le_convergent(self):
        """A M = 0,45, la chambre devient plus etroite que le col : il n'y a
        plus de convergent, donc plus de blocage, donc plus de tuyere."""
        cycle = run_cycle(FlightCase(), EngineCase())
        for mach in (0.45, 0.60):
            with self.subTest(mach=mach):
                with self.assertRaises(CaseError) as leve:
                    from_cycle(cycle, mass_flow=DEBIT, mach_combustor=mach)
                self.assertIn("convergent", str(leve.exception))

    def test_une_abscisse_hors_du_conduit_est_refusee(self):
        _, duct = conduit()
        for x in (-0.1, duct.length + 0.1):
            with self.subTest(x=x):
                with self.assertRaises(CaseError) as leve:
                    duct.area(x)
                self.assertIn("hors du conduit", str(leve.exception))

    def test_des_zones_qui_ne_se_touchent_pas_sont_refusees(self):
        with self.assertRaises(CaseError) as leve:
            Duct((Zone("a", 0.0, 1.0, 1.0, 2.0), Zone("b", 1.5, 2.0, 2.0, 3.0)))
        self.assertIn("ne se touchent pas", str(leve.exception))

    def test_un_saut_de_section_est_refuse(self):
        with self.assertRaises(CaseError) as leve:
            Duct((Zone("a", 0.0, 1.0, 1.0, 2.0), Zone("b", 1.0, 2.0, 5.0, 6.0)))
        self.assertIn("saut de section", str(leve.exception))

    def test_une_zone_de_longueur_ou_de_section_nulle_est_refusee(self):
        with self.assertRaises(CaseError):
            Duct((Zone("a", 0.0, 0.0, 1.0, 1.0),))
        with self.assertRaises(CaseError):
            Duct((Zone("a", 0.0, 1.0, 0.0, 1.0),))

    def test_des_longueurs_incompletes_sont_refusees_en_les_nommant(self):
        cycle = run_cycle(FlightCase(), EngineCase())
        with self.assertRaises(CaseError) as leve:
            from_cycle(cycle, mass_flow=DEBIT, lengths=dict(diffuser=0.4))
        message = str(leve.exception)
        self.assertIn("chamber", message)
        self.assertIn("diverging", message)


class Description(unittest.TestCase):
    def test_le_conduit_se_decrit(self):
        _, duct = conduit()
        texte = duct.describe()
        for attendu in ("diffuser", "chamber", "converging", "diverging", "col"):
            self.assertIn(attendu, texte)

    def test_les_longueurs_par_defaut_font_un_metre_et_demi(self):
        self.assertAlmostEqual(sum(DEFAULT_LENGTHS.values()), 1.5, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
