"""Les deux cas : ce qu'ils acceptent, ce qu'ils refusent, et leur hachage.

UN CAS REFUSE DOIT DIRE LEQUEL DE SES CHAMPS EST EN CAUSE. C'est la seule chose
qu'on lit d'un message d'erreur trois semaines plus tard, et c'est donc ce que
ces tests verifient — pas seulement qu'une exception sort, mais qu'elle nomme
le champ et rappelle ses bornes.
"""

import unittest
from dataclasses import FrozenInstanceError

from ramjet.config import (MAX_ALTITUDE, MIN_USEFUL_MACH, CaseError, EngineCase,
                           FlightCase)


class ConstructionDuVol(unittest.TestCase):
    def test_les_valeurs_par_defaut_sont_le_cas_de_reference_du_plan(self):
        vol = FlightCase()
        self.assertEqual((vol.mach, vol.altitude), (3.0, 11_000.0))
        self.assertEqual((vol.gamma, vol.R), (1.4, 287.05))

    def test_cp_est_DEDUIT_de_gamma_et_R(self):
        """« cp = gamma R / (gamma - 1) » est une identite : la donner en plus
        permettrait d'en fournir trois incoherents."""
        vol = FlightCase()
        self.assertAlmostEqual(vol.cp, 1004.675, delta=1e-3)
        self.assertAlmostEqual(FlightCase(gamma=1.3).cp, 1.3 * 287.05 / 0.3,
                               delta=1e-9)

    def test_un_mach_nul_ou_negatif_est_refuse_en_le_nommant(self):
        for mach in (0.0, -1.0):
            with self.subTest(mach=mach):
                with self.assertRaises(CaseError) as leve:
                    FlightCase(mach=mach)
                self.assertIn("mach", str(leve.exception))

    def test_une_altitude_hors_du_modele_est_refusee_avec_ses_bornes(self):
        for altitude in (-1.0, MAX_ALTITUDE + 1.0):
            with self.subTest(altitude=altitude):
                with self.assertRaises(CaseError) as leve:
                    FlightCase(altitude=altitude)
                message = str(leve.exception)
                self.assertIn("altitude", message)
                self.assertIn(f"{MAX_ALTITUDE:g}", message)

    def test_les_bornes_d_altitude_elles_memes_sont_acceptees(self):
        """Une borne exclue par erreur est un bug qu'on ne voit jamais."""
        for altitude in (0.0, MAX_ALTITUDE):
            with self.subTest(altitude=altitude):
                self.assertEqual(FlightCase(altitude=altitude).altitude, altitude)

    def test_un_gamma_absurde_est_refuse(self):
        for gamma in (0.9, 1.0, 2.0, 2.5):
            with self.subTest(gamma=gamma):
                with self.assertRaises(CaseError) as leve:
                    FlightCase(gamma=gamma)
                self.assertIn("gamma", str(leve.exception))

    def test_un_nan_ne_traverse_pas_les_bornes(self):
        """« nan <= 0 » est faux : sans controle de finitude, un NaN passerait
        toutes les comparaisons sans en declencher une seule."""
        for champ in ("mach", "altitude", "gamma", "R"):
            with self.subTest(champ=champ):
                with self.assertRaises(CaseError) as leve:
                    FlightCase(**{champ: float("nan")})
                self.assertIn("fini", str(leve.exception))

    def test_un_infini_est_refuse_aussi(self):
        with self.assertRaises(CaseError):
            FlightCase(mach=float("inf"))

    def test_ce_qui_n_est_pas_un_nombre_est_refuse_en_le_nommant(self):
        with self.assertRaises(CaseError) as leve:
            FlightCase(mach="trois")
        self.assertIn("mach", str(leve.exception))

    def test_le_seuil_d_utilite_est_signale_sans_etre_impose(self):
        """Un statoreacteur n'a pas de poussee au point fixe. Le modele doit
        pouvoir calculer quand meme — c'est le dire qui compte."""
        self.assertFalse(FlightCase(mach=0.2).is_useful)
        self.assertTrue(FlightCase(mach=MIN_USEFUL_MACH).is_useful)
        self.assertTrue(FlightCase(mach=3.0).is_useful)


class ConstructionDuMoteur(unittest.TestCase):
    def test_les_valeurs_par_defaut_sont_le_cas_de_reference_du_plan(self):
        moteur = EngineCase()
        self.assertEqual(moteur.T04, 2000.0)
        self.assertEqual(moteur.fuel, "kerosene")
        self.assertIsNone(moteur.pi_d)

    def test_une_temperature_negative_est_refusee(self):
        with self.assertRaises(CaseError) as leve:
            EngineCase(T04=-5.0)
        self.assertIn("T04", str(leve.exception))

    def test_un_rendement_hors_de_zero_un_est_refuse_avec_l_explication(self):
        """Un rapport de pression totale superieur a 1 serait un GAIN de
        pression, ce qu'aucun des trois organes ne peut produire."""
        for champ in ("eta_b", "pi_b", "pi_n", "pi_d"):
            for valeur in (0.0, -0.1, 1.2):
                with self.subTest(champ=champ, valeur=valeur):
                    with self.assertRaises(CaseError) as leve:
                        EngineCase(**{champ: valeur})
                    message = str(leve.exception)
                    self.assertIn(champ, message)
                    self.assertIn("]0, 1]", message)

    def test_un_rendement_egal_a_un_est_accepte(self):
        """Le cas ideal du critere de l'etape 3 met tous les rendements a 1 :
        les refuser rendrait l'oracle inatteignable."""
        moteur = EngineCase(eta_b=1.0, pi_b=1.0, pi_n=1.0, pi_d=1.0)
        self.assertEqual(moteur.inlet_recovery(3.0), 1.0)

    def test_une_chambre_supersonique_est_refusee_comme_hors_perimetre(self):
        for mach in (1.0, 1.5):
            with self.subTest(mach=mach):
                with self.assertRaises(CaseError) as leve:
                    EngineCase(M_combustor=mach)
                self.assertIn("M_combustor", str(leve.exception))

    def test_un_carburant_inconnu_est_refuse_avec_la_liste_des_choix(self):
        with self.assertRaises(CaseError) as leve:
            EngineCase(fuel="eau")
        message = str(leve.exception)
        self.assertIn("eau", message)
        self.assertIn("kerosene", message)

    def test_le_carburant_est_accessible_depuis_le_cas(self):
        moteur = EngineCase(fuel="hydrogen")
        self.assertEqual(moteur.fuel_data.name, "hydrogen")
        self.assertGreater(moteur.fuel_data.lhv, 100e6)


class RecuperationDuDiffuseur(unittest.TestCase):
    """« pi_d = None » veut dire « la correlation », pas « pas de perte »."""

    def test_la_correlation_MIL_vaut_ce_qu_annonce_le_plan(self):
        """Mesure : 0,8088 a M0 = 3 — pres de 20 % de pression totale perdue
        avant la chambre, le premier poste de perte du moteur."""
        moteur = EngineCase()
        self.assertAlmostEqual(moteur.inlet_recovery(3.0), 0.8088, delta=1e-4)
        self.assertAlmostEqual(moteur.inlet_recovery(2.0), 0.9250, delta=1e-4)
        self.assertAlmostEqual(moteur.inlet_recovery(5.0), 0.5126, delta=1e-4)

    def test_elle_vaut_exactement_un_en_subsonique(self):
        """La correlation ne decrit qu'une entree supersonique : sous M = 1 il
        n'y a pas de choc, donc pas de perte a ce titre."""
        moteur = EngineCase()
        for mach in (0.5, 0.9, 1.0):
            with self.subTest(mach=mach):
                self.assertEqual(moteur.inlet_recovery(mach), 1.0)

    def test_elle_decroit_strictement_avec_le_mach(self):
        moteur = EngineCase()
        valeurs = [moteur.inlet_recovery(m) for m in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)]
        for avant, apres in zip(valeurs, valeurs[1:]):
            self.assertGreater(avant, apres)

    def test_une_valeur_donnee_l_emporte_sur_la_correlation(self):
        moteur = EngineCase(pi_d=0.90)
        for mach in (0.5, 3.0, 5.0):
            with self.subTest(mach=mach):
                self.assertEqual(moteur.inlet_recovery(mach), 0.90)


class GeleEtHachable(unittest.TestCase):
    """Ce qui rend la boucle de couplage de l'etape 9 memorisable."""

    def test_les_deux_cas_sont_immuables(self):
        with self.assertRaises(FrozenInstanceError):
            FlightCase().mach = 2.0
        with self.assertRaises(FrozenInstanceError):
            EngineCase().T04 = 1500.0

    def test_ils_se_hachent(self):
        self.assertIsInstance(hash(FlightCase()), int)
        self.assertIsInstance(hash(EngineCase()), int)

    def test_deux_cas_identiques_ont_le_meme_hachage(self):
        self.assertEqual(hash(FlightCase(mach=2.5)), hash(FlightCase(mach=2.5)))
        self.assertEqual(FlightCase(mach=2.5), FlightCase(mach=2.5))

    def test_un_entier_et_un_flottant_donnent_le_MEME_cas(self):
        """Sans normalisation, « mach=3 » et « mach=3.0 » seraient deux cles de
        cache differentes pour exactement le meme calcul."""
        self.assertEqual(FlightCase(mach=3), FlightCase(mach=3.0))
        self.assertEqual(hash(FlightCase(mach=3)), hash(FlightCase(mach=3.0)))
        self.assertEqual(hash(EngineCase(T04=2000)), hash(EngineCase(T04=2000.0)))

    def test_deux_cas_differents_se_hachent_differemment(self):
        self.assertNotEqual(hash(FlightCase(mach=2.5)), hash(FlightCase(mach=3.0)))
        self.assertNotEqual(hash(EngineCase(pi_b=0.95)), hash(EngineCase(pi_b=0.90)))

    def test_pi_d_a_None_se_distingue_de_pi_d_a_un(self):
        """Les deux donnent la meme perte en subsonique et une perte tres
        differente en supersonique : ce ne sont pas les memes cas."""
        self.assertNotEqual(EngineCase(pi_d=None), EngineCase(pi_d=1.0))


class Descriptions(unittest.TestCase):
    """La premiere ligne d'un rapport doit se lire sans decodeur."""

    def test_le_vol_se_decrit(self):
        texte = FlightCase().describe()
        self.assertIn("M0 = 3", texte)
        self.assertIn("11000", texte)

    def test_le_moteur_se_decrit_et_nomme_sa_correlation(self):
        texte = EngineCase().describe()
        self.assertIn("2000", texte)
        self.assertIn("MIL-E-5008B", texte)
        self.assertIn("kerosene", texte)

    def test_un_pi_d_donne_apparait_a_la_place_de_la_correlation(self):
        texte = EngineCase(pi_d=0.9).describe()
        self.assertIn("0.9", texte)
        self.assertNotIn("MIL", texte)


if __name__ == "__main__":
    unittest.main()
