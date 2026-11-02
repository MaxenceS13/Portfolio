"""Validation 0D : le modele contre les ordres de grandeur publies.

VERIFIER ET VALIDER NE SONT PAS LA MEME CHOSE, ET CE FICHIER FAIT LA SECONDE.
L'etape 3 a compare le cycle a sa propre forme fermee : c'etait de la
verification, exacte a 1e-16, et elle ne disait rien de la justesse du MODELE.
Ici on compare a des fourchettes publiees, larges, qui couvrent des moteurs de
dessins differents. Y tomber valide le modele ; ca ne le calibre pas, et aucune
de ces bornes n'a de sens a mieux que 10 %.

LE PLAN SE CONTREDISAIT SUR CE POINT, ET LA MESURE A TRANCHE. Il annonce
« F/mdot dans [600 ; 900] sur M0 dans [2 ; 4] » puis liste, parmi ses valeurs
mesurees, 576 N.s/kg a M0 = 4 — hors de sa propre bande. Ce n'est pas le modele
qui a tort : a T04 = 2 000 K, un statoreacteur a Mach 4 n'a plus assez d'ecart
de temperature pour bruler. La bande tient donc sur [2 ; 3,5], et le test qui
suit montre que porter T04 a 2 200 K ramene M0 = 4 dedans — ce qui vaut mieux
qu'un intervalle retreci en silence.
"""

import unittest

import numpy as np

from ramjet.config import EngineCase, FlightCase
from ramjet.cycle0d import (PUBLISHED_SPECIFIC_IMPULSE,
                            PUBLISHED_SPECIFIC_THRUST,
                            PUBLISHED_THERMAL_EFFICIENCY,
                            ideal_engine, run_cycle, sweep_mach)


class LeCritereDeLEtape(unittest.TestCase):
    """Les deux fourchettes publiees, sur le domaine ou elles s'appliquent."""

    def test_la_poussee_specifique_est_dans_la_bande_publiee(self):
        """Mesure a T04 = 2 000 K : 762,0 / 774,5 / 739,0 / 669,7 N.s/kg pour
        M0 = 2 / 2,5 / 3 / 3,5."""
        balayage = sweep_mach(np.arange(2.0, 3.51, 0.1), EngineCase())
        dedans = balayage.within("specific_thrust", PUBLISHED_SPECIFIC_THRUST)
        self.assertTrue(np.all(dedans),
                        f"hors bande : "
                        f"{balayage.machs[~dedans]} -> "
                        f"{balayage.specific_thrust[~dedans]}")

    def test_l_impulsion_specifique_est_dans_la_bande_publiee(self):
        """Mesure : 1 642 s a M0 = 1,5, 2 167 s a M0 = 3,5, 1 983 s a M0 = 5."""
        balayage = sweep_mach(np.arange(1.5, 5.01, 0.25), EngineCase())
        dedans = balayage.within("specific_impulse", PUBLISHED_SPECIFIC_IMPULSE)
        self.assertTrue(np.all(dedans),
                        f"hors bande : {balayage.machs[~dedans]}")

    def test_le_rendement_thermique_est_dans_la_bande_publiee(self):
        balayage = sweep_mach(np.arange(1.5, 5.01, 0.25), EngineCase())
        dedans = balayage.within("thermal_efficiency",
                                 PUBLISHED_THERMAL_EFFICIENCY)
        self.assertTrue(np.all(dedans))


class LaChuteAMachQuatre(unittest.TestCase):
    """Le point ou le plan se contredisait, traite comme un resultat."""

    def test_a_T04_egale_2000_K_la_poussee_sort_de_la_bande_a_M0_egale_4(self):
        """576,4 N.s/kg, sous les 600 de la bande. Ce n'est pas une erreur du
        modele : c'est ce que fait un statoreacteur a qui il ne reste plus
        d'ecart de temperature."""
        cycle = run_cycle(FlightCase(mach=4.0), EngineCase())
        self.assertAlmostEqual(cycle.specific_thrust, 576.4, delta=0.5)
        self.assertLess(cycle.specific_thrust, PUBLISHED_SPECIFIC_THRUST[0])

    def test_et_une_chambre_plus_chaude_l_y_ramene(self):
        """C'est la demonstration que la chute est physique et non numerique :
        671,5 N.s/kg a T04 = 2 200 K, 763,7 a 2 400 K. Un statoreacteur de
        Mach 4 se dessine plus chaud, et le modele le dit."""
        for T04, attendu in ((2200.0, 671.5), (2400.0, 763.7)):
            with self.subTest(T04=T04):
                cycle = run_cycle(FlightCase(mach=4.0), EngineCase(T04=T04))
                self.assertAlmostEqual(cycle.specific_thrust, attendu, delta=1.0)
                self.assertGreater(cycle.specific_thrust,
                                   PUBLISHED_SPECIFIC_THRUST[0])

    def test_la_cause_est_le_dosage_qui_s_effondre(self):
        """f passe de 0,0405 a M0 = 2 a 0,0274 a M0 = 4 : un tiers de moins de
        carburant admissible, a T04 fixee."""
        dosages = [run_cycle(FlightCase(mach=m), EngineCase()).fuel_air_ratio
                   for m in (2.0, 3.0, 4.0)]
        self.assertAlmostEqual(dosages[0], 0.0405, delta=1e-4)
        self.assertAlmostEqual(dosages[2], 0.0274, delta=1e-4)
        self.assertLess(dosages[2], 0.7 * dosages[0])


class LeMaximumDePousseeSpecifique(unittest.TestCase):
    """Le fait marquant du statoreacteur : il a un Mach de predilection."""

    def test_le_maximum_existe_et_tombe_entre_deux_et_trois(self):
        """Sur grille fine (0,01), le maximum est a M0 = 2,36 et non 2,5 : le
        2,5 du plan venait d'un balayage par pas de 0,5. Le test cherche donc
        l'argmax plutot que de verifier une valeur tabulee.
        """
        balayage = sweep_mach(np.arange(1.5, 6.01, 0.01), EngineCase())
        mach, poussee = balayage.best_thrust()
        self.assertAlmostEqual(mach, 2.36, delta=0.05)
        self.assertAlmostEqual(poussee, 776.5, delta=1.0)
        self.assertGreater(mach, 2.0)
        self.assertLess(mach, 3.0)

    def test_il_se_deplace_vers_les_grands_mach_quand_la_chambre_chauffe(self):
        """LA contre-epreuve : un modele insensible a son parametre principal
        donnerait le meme maximum partout. Mesure : 2,28 a 1 800 K, 2,36 a
        2 000 K, 2,53 a 2 500 K, 2,68 a 3 000 K.
        """
        machs = []
        for T04 in (1800.0, 2000.0, 2500.0, 3000.0):
            balayage = sweep_mach(np.arange(1.5, 6.01, 0.01), EngineCase(T04=T04))
            machs.append(balayage.best_thrust()[0])
        self.assertAlmostEqual(machs[0], 2.28, delta=0.05)
        self.assertAlmostEqual(machs[-1], 2.68, delta=0.05)
        for avant, apres in zip(machs, machs[1:]):
            self.assertLess(avant, apres)

    def test_la_poussee_croit_partout_quand_la_chambre_chauffe(self):
        """Second volet de la contre-epreuve, sur les valeurs et non sur
        l'argmax."""
        for mach in (2.0, 3.0, 4.0):
            with self.subTest(mach=mach):
                froide = run_cycle(FlightCase(mach=mach), EngineCase(T04=2000.0))
                chaude = run_cycle(FlightCase(mach=mach), EngineCase(T04=2500.0))
                self.assertGreater(chaude.specific_thrust,
                                   froide.specific_thrust)


class TendancesQuiNeDemandentAucuneTable(unittest.TestCase):
    """Ce que le modele doit faire, independamment de toute valeur publiee."""

    def setUp(self):
        self.balayage = sweep_mach(np.arange(1.5, 5.01, 0.25), EngineCase())

    def test_le_rendement_thermique_croit_sur_le_domaine_UTILE(self):
        """Mesure : 0,281 a M0 = 1,5 et 0,709 a M0 = 4,5, strictement croissant
        entre les deux. Il croit alors meme que la poussee s'effondre —
        confondre les deux est l'erreur classique sur ce moteur.
        """
        balayage = sweep_mach(np.arange(1.5, 4.51, 0.01), EngineCase())
        rendements = balayage.column("thermal_efficiency")
        self.assertTrue(np.all(np.diff(rendements) > 0.0))
        self.assertAlmostEqual(rendements[0], 0.281, delta=1e-3)
        self.assertAlmostEqual(rendements[-1], 0.709, delta=1e-3)

    def test_mais_il_a_un_MAXIMUM_lui_aussi_et_le_plan_l_ignorait(self):
        """Le plan annoncait « strictement croissante sur [1,5 ; 5] ». C'est
        faux pour le moteur REEL : le rendement thermique passe par un maximum
        de 0,7129 a M0 = 4,79, puis s'effondre a 0,464 a M0 = 6.

        La cause est mesurable : la recuperation du diffuseur tombe de 0,925 a
        M0 = 2 a 0,513 a M0 = 5, et au-dela de M0 = 4,8 cette perte l'emporte
        sur le gain de compression dynamique. Ce n'est donc pas un defaut du
        modele, c'est ce que fait une entree d'air a Mach 5.
        """
        balayage = sweep_mach(np.arange(1.5, 6.4, 0.01), EngineCase())
        rendements = balayage.column("thermal_efficiency")
        index = int(np.argmax(rendements))
        self.assertAlmostEqual(float(balayage.machs[index]), 4.79, delta=0.05)
        self.assertAlmostEqual(float(rendements[index]), 0.7129, delta=1e-3)
        self.assertFalse(np.all(np.diff(rendements) > 0.0))

    def test_le_rendement_IDEAL_lui_est_bien_monotone(self):
        """Le contraste qui identifie la cause : sans aucune perte, le
        rendement vaut « 1 - 1/tau_r » (identite exacte de l'etape 3), et
        celui-la croit strictement de 0,310 a 0,891 sur toute la plage.
        Toute la non-monotonie du moteur reel vient donc de ses pertes.
        """
        balayage = sweep_mach(np.arange(1.5, 6.4, 0.01),
                              ideal_engine(EngineCase()))
        rendements = balayage.column("thermal_efficiency")
        self.assertTrue(np.all(np.diff(rendements) > 0.0))
        self.assertAlmostEqual(float(rendements[0]), 0.310, delta=1e-3)
        self.assertAlmostEqual(float(rendements[-1]), 0.891, delta=1e-3)

    def test_le_rendement_propulsif_croit_aussi(self):
        """Il tend vers 1 quand la vitesse d'ejection rejoint celle du vol —
        c'est-a-dire quand il n'y a plus de poussee."""
        rendements = self.balayage.column("propulsive_efficiency")
        self.assertTrue(np.all(np.diff(rendements) > 0.0))
        self.assertLess(rendements[-1], 1.0)

    def test_le_rendement_croit_pendant_que_la_poussee_s_effondre(self):
        """Les deux ne disent pas la meme chose, et c'est le piege du sujet :
        entre M0 = 3 et M0 = 4,5, le rendement thermique gagne 12 points
        pendant que la poussee specifique perd 37 %."""
        balayage = sweep_mach(np.arange(3.0, 4.51, 0.05), EngineCase())
        self.assertTrue(np.all(np.diff(
            balayage.column("thermal_efficiency")) > 0.0))
        self.assertTrue(np.all(np.diff(balayage.specific_thrust) < 0.0))

    def test_la_poussee_vient_de_la_VITESSE_et_pas_du_MACH(self):
        """Ce test a d'abord ete ecrit a l'envers, et l'erreur est instructive.

        Premier jet : « M9 > M0, sinon il n'y aurait pas de poussee ». C'est
        faux — mesure, M9 est INFERIEUR a M0 partout sur [1,5 ; 5] pour le
        moteur reel (4,391 contre 5,0), et la poussee est pourtant positive
        partout. La raison : les gaz sortent chauds, donc leur vitesse du son
        vaut de 1,4 a 2,6 fois celle de l'air ambiant, et « V9 = M9 a9 »
        depasse « V0 = M0 a0 » malgre un Mach plus faible.

        Dans le cas ideal, c'est encore plus net : M9 = M0 EXACTEMENT (etape 3)
        et la poussee vaut quand meme 777 N.s/kg.
        """
        vitesses_sortie = self.balayage.column("velocity_exit")
        vitesses_vol = self.balayage.column("velocity_flight")
        self.assertTrue(np.all(vitesses_sortie > vitesses_vol))
        self.assertTrue(np.all(self.balayage.specific_thrust > 0.0))
        # Et le Mach, lui, ne monte pas.
        self.assertTrue(np.all(self.balayage.column("mach_exit")
                               < self.balayage.machs))
        rapport_sons = ((vitesses_sortie / self.balayage.column("mach_exit"))
                        / (vitesses_vol / self.balayage.machs))
        self.assertGreater(float(rapport_sons.min()), 1.3)


class LeBalayageLuiMeme(unittest.TestCase):
    """Le balayage n'ajoute aucune physique : il enregistre."""

    def test_il_garde_les_points_refuses_au_lieu_de_les_ecarter(self):
        """Savoir OU le balayage s'arrete est une information. Mesure : les six
        derniers points d'un balayage jusqu'a M0 = 7 sont refuses, le premier
        a M0 = 6,5."""
        balayage = sweep_mach(np.linspace(1.5, 7.0, 56), EngineCase())
        self.assertEqual(len(balayage.cycles) + len(balayage.refused), 56)
        self.assertGreater(len(balayage.refused), 0)
        premier_refus = balayage.refused[0]
        self.assertAlmostEqual(premier_refus[0], 6.5, delta=0.11)
        self.assertIn("plus rien a bruler", premier_refus[1])

    def test_il_rend_les_memes_valeurs_que_run_cycle(self):
        """Il appelle « run_cycle » et n'en modifie rien."""
        balayage = sweep_mach([2.0, 3.0, 4.0], EngineCase())
        for mach, cycle in zip(balayage.machs, balayage.cycles):
            with self.subTest(mach=mach):
                direct = run_cycle(FlightCase(mach=float(mach)), EngineCase())
                self.assertEqual(cycle.specific_thrust, direct.specific_thrust)

    def test_un_balayage_entierement_refuse_le_dit(self):
        from ramjet.config import CaseError

        balayage = sweep_mach([6.6, 6.8, 7.0], EngineCase())
        self.assertEqual(len(balayage.cycles), 0)
        with self.assertRaises(CaseError):
            balayage.best_thrust()

    def test_les_colonnes_ont_la_longueur_des_points_retenus(self):
        balayage = sweep_mach(np.linspace(1.5, 7.0, 56), EngineCase())
        for nom in ("specific_thrust", "tsfc", "fuel_air_ratio", "mach_exit"):
            with self.subTest(colonne=nom):
                self.assertEqual(len(balayage.column(nom)), len(balayage.cycles))


if __name__ == "__main__":
    unittest.main()
