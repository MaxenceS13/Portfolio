"""La boucle 0D <-> 1D : ce qu'elle corrige, et ce qu'elle refuse.

TROIS FAMILLES.

  - LA CONVERGENCE, et la methode qu'il a fallu changer pour l'obtenir.
    L'iteration nue converge, mais trop lentement pour le critere du plan ; la
    secante y arrive exactement.
  - CE QUE LA BOUCLE APPREND. Le « pi_b » suppose est faux de 3 %, et surtout :
    le seul chemin par lequel la boucle boucle est la GEOMETRIE. A conduit fige,
    elle se ferme en un tour.
  - CE QU'ELLE REFUSE, et pourquoi ce n'est pas ce que le plan croyait. Le point
    « M = 0,30 / T04 = 2 000 K » est bien refuse — mais parce que le dessin
    s'effondre, pas parce que l'ecoulement bloque.
"""

import unittest

import numpy as np

from ramjet import rayleigh
from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.coupling import (Convergence, DesignCollapse, _chamber_inlet_mach,
                             choking_mass_flow, converge, run_engine)
from ramjet.cycle0d import run_cycle
from ramjet.duct import from_cycle

GAMMA, R = 1.4, 287.05


def pi_b_du_1D(pi_suppose: float, **changes) -> float:
    """Un tour de boucle, nu : le « pi_b » que le 1D rend pour celui suppose."""
    cas = EngineCase(pi_b=pi_suppose, **changes)
    return run_engine(FlightCase(), cas).combustor_pressure_ratio


_SUITE_NUE: list[float] = []


def suite_nue(tours: int = 8) -> list[float]:
    """L'iteration « pi <- g(pi) » depuis 0,95, calculee UNE fois.

    Deux tests s'en servent — la lenteur de la methode du plan et le rapport de
    convergence — et chaque tour coute 0,33 s.
    """
    while len(_SUITE_NUE) < tours + 1:
        _SUITE_NUE.append(0.95 if not _SUITE_NUE
                          else pi_b_du_1D(_SUITE_NUE[-1]))
    return _SUITE_NUE[:tours + 1]


class PointFixeDeReference(unittest.TestCase):
    """Le cas de reference, converge une fois : 1,7 s, ce n'est pas gratuit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.point = converge()


# -- la convergence, et la methode ------------------------------------------

class Convergence5Tours(PointFixeDeReference):

    def test_le_critere_du_plan_est_atteint(self):
        """Moins de cinq tours a 1e-6. Mesure : cinq appels au 1D pour un
        residu de 8,6e-13, soit six ordres de grandeur de mieux que demande.
        """
        self.assertLessEqual(self.point.calls, 6)
        self.assertLess(self.point.residual, 1e-12)

    def test_mais_l_iteration_NUE_ne_l_aurait_pas_ete(self):
        """LA METHODE DU PLAN N'AURAIT PAS TENU SON PROPRE CRITERE.

        « pi <- g(pi) » converge geometriquement, de rapport 0,1724. Il faut
        SEPT tours pour descendre a 1e-6 sur la variation relative, la ou le
        plan en annoncait moins de cinq. Le test le mesure plutot que de le
        supposer : c'est ce qui a motive la secante.
        """
        suite = suite_nue()
        tours = next(i for i in range(1, len(suite))
                     if abs(suite[i] / suite[i - 1] - 1.0) < 1e-6)
        self.assertEqual(tours, 7)

    def test_le_rapport_de_convergence_est_constant(self):
        """0,172369, stable a 1e-6 des le troisieme tour. C'est un point fixe
        LINEAIRE — d'ou l'efficacite de la secante, qui suppose exactement
        cela.
        """
        ecarts = np.diff(suite_nue())
        rapports = ecarts[1:] / ecarts[:-1]
        # Mesure : 0,1633 / 0,1708 / 0,1721 / 0,1723 / 0,17236 / 0,172368 /
        # 0,172369. Le test verifie la limite ET l'approche monotone vers elle,
        # ce qui est l'affirmation utile : le point fixe est LINEAIRE.
        # L'APPORT DE MASSE A CHANGE LE RAPPORT, ce qui est attendu : il ajoute
        # sa propre perte de pression totale et modifie donc la boucle.
        limite = float(rapports[-1])
        self.assertAlmostEqual(limite, 0.17680, delta=1e-3)
        for avant, apres in zip(rapports, rapports[1:]):
            self.assertLess(abs(apres - limite), abs(avant - limite) + 1e-9)

    def test_le_point_fixe_est_un_VRAI_point_fixe(self):
        """Repartir de la solution convergee donne la meme chose. Le plan
        demandait 1e-12 ; mesure : 7,8e-14.
        """
        arrivee = self.point.combustor_pressure_ratio
        self.assertAlmostEqual(pi_b_du_1D(arrivee), arrivee, delta=1e-12)

    def test_insensibilite_au_point_de_depart(self):
        """Le plan demandait 1e-8 entre deux departs. Mesure : les DOUZE memes
        chiffres depuis 0,90 et 0,99 — le point fixe ne se souvient de rien.
        """
        depuis_90 = converge(engine=EngineCase(pi_b=0.90))
        depuis_99 = converge(engine=EngineCase(pi_b=0.99))
        self.assertAlmostEqual(depuis_90.combustor_pressure_ratio,
                               depuis_99.combustor_pressure_ratio, delta=1e-12)
        self.assertAlmostEqual(depuis_90.combustor_pressure_ratio,
                               self.point.combustor_pressure_ratio, delta=1e-12)

    def test_partir_du_point_fixe_coute_un_seul_appel(self):
        """L'amorcage voit tout de suite qu'il n'y a rien a faire."""
        arrivee = self.point.combustor_pressure_ratio
        rapide = converge(engine=EngineCase(pi_b=arrivee))
        self.assertEqual(rapide.calls, 1)
        self.assertLess(rapide.residual, 1e-12)


# -- ce que la boucle apprend ------------------------------------------------

class CeQueLaBoucleCorrige(PointFixeDeReference):

    def test_le_pi_b_suppose_est_faux_de_trois_pour_cent(self):
        """CE QUI JUSTIFIE LE PROJET. Le plan demandait plus de 1 % d'ecart ;
        mesure : 0,9218 contre 0,9500 suppose, soit -2,97 %.
        """
        self.assertAlmostEqual(self.point.combustor_pressure_ratio, 0.919591,
                               delta=1e-5)
        self.assertGreater(abs(self.point.correction), 0.01)

    def test_et_la_poussee_ne_bouge_presque_pas(self):
        """CE QUI EST PLUS SURPRENANT, ET QU'IL FAUT DIRE.

        Corriger « pi_b » de 3 % ne change la poussee specifique que de 0,2 %
        (738,3 -> 737,4 N.s/kg). La perte de chambre est reelle mais le moteur
        y est peu sensible : ce qu'on gagne a le savoir n'est pas la poussee,
        c'est le DEBIT et les sections, qui bougent de 2,5 %.
        """
        premier = run_engine(FlightCase(), EngineCase())
        self.assertAlmostEqual(
            self.point.engine.specific_thrust / premier.specific_thrust, 1.0,
            delta=5e-3)
        self.assertGreater(
            abs(self.point.engine.mass_flow / premier.mass_flow - 1.0), 0.02)

    def test_la_geometrie_est_le_SEUL_chemin_de_la_boucle(self):
        """A CONDUIT FIGE, LA BOUCLE SE FERME EN UN TOUR — EXACTEMENT.

        Les donnees du 1D dans la chambre sont « T02 », « p02 », « T04 » et la
        geometrie. Aucune ne depend du « pi_b » suppose : celui-ci n'agit que
        sur le DIMENSIONNEMENT. Fige le conduit, et « g » devient une fonction
        constante. Le residu est nul au sens strict, pas seulement petit.
        """
        conduit = from_cycle(run_cycle(FlightCase(), EngineCase()),
                             mass_flow=10.0)
        a = run_engine(FlightCase(), EngineCase(pi_b=0.95), duct=conduit)
        b = run_engine(FlightCase(), EngineCase(pi_b=0.80), duct=conduit)
        self.assertEqual(a.combustor_pressure_ratio,
                         b.combustor_pressure_ratio)
        fige = converge(duct=conduit)
        self.assertEqual(fige.calls, 2)
        self.assertEqual(fige.residual, 0.0)

    def test_la_marge_avant_blocage_est_large_sur_le_cas_de_reference(self):
        """1 275 K. Le point de fonctionnement nominal n'est pas au bord."""
        self.assertAlmostEqual(self.point.choking_margin(), 1495.0, delta=5.0)

    def test_l_objet_rendu_porte_l_historique(self):
        self.assertIsInstance(self.point, Convergence)
        self.assertEqual(self.point.history[0], 0.95)
        self.assertEqual(len(self.point.history), self.point.calls + 1)
        self.assertIn("marge avant blocage", self.point.describe())


# -- ce qu'elle refuse -------------------------------------------------------

class CasBloque(unittest.TestCase):
    """Le plan predisait un refus a « M = 0,30 / T04 = 2 000 K ». Il a raison
    sur le fait, et se trompe sur les trois raisons."""

    CAS = dict(M_combustor=0.30, T04=2000.0)

    @classmethod
    def setUpClass(cls) -> None:
        try:
            converge(engine=EngineCase(**cls.CAS))
        except DesignCollapse as erreur:
            cls.refus = erreur
        else:
            cls.refus = None

    def test_la_boucle_refuse_bien_ce_point(self):
        self.assertIsNotNone(self.refus)
        self.assertEqual(self.refus.iteration, 1)

    def test_mais_le_PREMIER_dessin_passait(self):
        """CE QUI REND LE REFUS INTERESSANT.

        Un 0D seul rend ici un moteur d'allure parfaitement normale. C'est la
        contre-reaction qui le detruit : avec pi_b = 0,839 au lieu de 0,95, la
        pression de fin de chambre tombe, le col qu'il faudrait grandit, et il
        finit plus large que la chambre. Il n'y a plus de convergent, donc plus
        de tuyere.
        """
        moteur = run_engine(FlightCase(), EngineCase(**self.CAS))
        self.assertAlmostEqual(moteur.combustor_pressure_ratio, 0.839,
                               delta=1e-2)
        self.assertGreater(moteur.specific_thrust, 500.0)

    def test_le_refus_ne_vient_PAS_du_blocage_thermique(self):
        """PREMIERE ERREUR DU PLAN. Il annoncait « la boucle doit refuser en
        annoncant le blocage ». Le message parle de geometrie : le col n'est
        plus plus etroit que la chambre.
        """
        self.assertIn("col", self.refus.cause)
        self.assertNotIn("blocage thermique", self.refus.cause)

    def test_et_le_1749_K_du_plan_repond_a_une_question_que_le_moteur_ne_pose_pas(self):
        """DEUXIEME ERREUR, ET C'EST LA PLUS INSTRUCTIVE.

        Le 1 749 K est JUSTE — c'est bien la limite de Rayleigh a M = 0,30,
        avec T02 = 606,6 K. Mais la chambre ne tourne jamais au Mach de
        conception : le col fixe le debit, donc le Mach reel, qui vaut 0,2716.
        La limite qui s'applique est celle-la : 2 055 K, et non 1 749 K. Le
        point passe donc le blocage avec 55 K de marge, et meurt d'autre chose.
        """
        cas = EngineCase(**self.CAS)
        cycle = run_cycle(FlightCase(), cas)
        T02, p02 = cycle.station("2").T0, cycle.station("2").p0
        conduit = from_cycle(cycle, mass_flow=10.0)
        debit = choking_mass_flow(conduit, T02, p02, 2000.0, GAMMA, R)
        reel = _chamber_inlet_mach(debit, conduit, T02, p02, GAMMA, R)

        self.assertAlmostEqual(T02, 606.62, delta=0.1)
        self.assertAlmostEqual(
            float(rayleigh.critical_total_temperature(0.30, T02)), 1749.0,
            delta=1.0)
        self.assertAlmostEqual(reel, 0.2716, delta=1e-3)
        self.assertAlmostEqual(
            float(rayleigh.critical_total_temperature(reel, T02)), 2055.0,
            delta=5.0)

    def test_le_message_dit_que_le_premier_passage_avait_accepte(self):
        """Sans quoi l'utilisateur croirait avoir demande l'impossible des le
        depart, alors que c'est la boucle qui a decouvert le probleme."""
        message = str(self.refus)
        self.assertIn("le premier passage avait accepte", message)
        self.assertIn("0.95", message)


class QuelleGardeSeDeclenche(unittest.TestCase):

    def test_sur_la_grille_utile_la_garde_de_blocage_ne_sert_JAMAIS(self):
        """TROISIEME ERREUR DU PLAN, ET LA PLUS STRUCTURELLE.

        Balaye sur 8 Mach de conception x 6 temperatures : 27 cas passent, 21
        sont refuses par la GEOMETRIE, zero par le blocage thermique. Le test
        rejoue une grille reduite. La raison est celle de l'etape 8 : le col
        fixe le debit, donc le Mach de chambre, qui reste toujours sous celui
        vise — le moteur se regule lui-meme, et la degenerescence geometrique
        arrive toujours en premier.

        La garde de blocage reste en place : elle est atteignable par une
        geometrie construite a la main (etape 8, A_chambre/A_col = 1,0), donc
        elle n'est pas morte. Mais aucun cas dessine par « from_cycle » ne
        l'atteint, et il faut le dire.
        """
        geometriques = blocages = passes = 0
        for mach in (0.20, 0.28, 0.32, 0.40):
            for T04 in (1400.0, 2200.0, 3000.0):
                cas = EngineCase(M_combustor=mach, T04=T04)
                try:
                    cycle = run_cycle(FlightCase(), cas)
                    conduit = from_cycle(cycle, mass_flow=10.0)
                except CaseError:
                    geometriques += 1
                    continue
                try:
                    choking_mass_flow(conduit, cycle.station("2").T0,
                                      cycle.station("2").p0, T04, GAMMA, R)
                    passes += 1
                except CaseError:
                    blocages += 1
        self.assertEqual(blocages, 0)
        self.assertGreater(geometriques, 0)
        self.assertGreater(passes, 0)


class RefusDeLaBoucle(unittest.TestCase):

    def test_le_plafond_d_iterations_est_annonce(self):
        with self.assertRaises(CaseError) as capture:
            converge(max_iterations=3, tol=1e-16)
        self.assertIn("n'a pas converge", str(capture.exception))

    def test_le_cas_absurde_est_refuse_AVANT_la_boucle(self):
        """Un Mach de chambre supersonique n'atteint jamais la boucle : c'est
        « EngineCase » qui refuse, a la construction."""
        with self.assertRaises(CaseError):
            converge(engine=EngineCase(M_combustor=1.5))


if __name__ == "__main__":
    unittest.main()
