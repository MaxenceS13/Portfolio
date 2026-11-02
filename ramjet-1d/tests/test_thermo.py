"""Les relations qui serviront d'oracles a tout le reste du projet.

CE FICHIER EST LE PLUS EXIGEANT DU PROJET, ET C'EST VOULU. Chacune de ces
relations a une forme fermee, donc chacune est verifiable contre une table
publiee ou contre une valeur rationnelle EXACTE. Ce qui se verifie ici a 1e-15
n'aura pas a etre soupconne trois etapes plus loin, quand plus rien ne sera
exact.

Quatre valeurs de A/A* sont des rationnels exacts pour gamma = 1,4, ce qui
permet de les comparer a 1e-15 au lieu des cinq decimales d'une table :

    A/A*(1/2) = 2 (7/8)^3   = 1,33984375       exactement
    A/A*(1)   = 1                              exactement
    A/A*(2)   = (1/2)(3/2)^3 = 1,6875          exactement
    A/A*(3)   = (1/3)(7/3)^3 = 343/81          exactement
    A/A*(5)   = (1/5) 5^3    = 25              exactement
"""

import unittest

import numpy as np
from scipy.optimize import brentq

from ramjet.thermo import (BISECTION_STEPS, MACH_MAX, NormalShock, ThermoError,
                           area_ratio, area_ratio_derivative,
                           mach_from_area_ratio,
                           mach_from_total_pressure_ratio, normal_shock,
                           speed_of_sound, strong_shock_limit,
                           total_density_ratio, total_pressure_ratio,
                           total_temperature_ratio, velocity)

GAMMA = 1.4


class RelationsIsentropiques(unittest.TestCase):
    """Les trois rapports totaux, et leur coherence entre eux."""

    def test_a_mach_nul_tous_les_rapports_valent_un(self):
        for fonction in (total_temperature_ratio, total_pressure_ratio,
                         total_density_ratio):
            with self.subTest(fonction=fonction.__name__):
                self.assertEqual(float(fonction(0.0)), 1.0)

    def test_les_valeurs_a_mach_un(self):
        """T0/T = 1,2 exactement pour gamma = 1,4 : c'est un rationnel."""
        self.assertEqual(float(total_temperature_ratio(1.0)), 1.2)
        self.assertAlmostEqual(float(total_pressure_ratio(1.0)), 1.892929,
                               delta=1e-6)

    def test_les_trois_rapports_sont_coherents(self):
        """p0/p = (T0/T)^(g/(g-1)) et rho0/rho = (T0/T)^(1/(g-1)), donc
        « p0/p = (T0/T)(rho0/rho) » — une identite, donc un vrai controle."""
        mach = np.linspace(0.05, 6.0, 200)
        gauche = total_pressure_ratio(mach)
        droite = total_temperature_ratio(mach) * total_density_ratio(mach)
        np.testing.assert_allclose(gauche, droite, rtol=1e-14)

    def test_le_rapport_de_pression_a_mach_trois(self):
        """p0/p = 36,73 a M = 3 : c'est la compression dynamique qui fait
        marcher un statoreacteur sans compresseur."""
        self.assertAlmostEqual(float(total_pressure_ratio(3.0)), 36.7327,
                               delta=1e-4)


class RapportDeSection(unittest.TestCase):
    """A/A*, ses valeurs exactes, et son minimum."""

    def test_les_valeurs_rationnelles_exactes(self):
        for mach, exact in ((0.5, 1.33984375), (1.0, 1.0), (2.0, 1.6875),
                            (3.0, 343.0 / 81.0), (5.0, 25.0)):
            with self.subTest(mach=mach):
                self.assertAlmostEqual(float(area_ratio(mach)), exact, delta=1e-14)

    def test_il_vaut_exactement_un_au_col(self):
        self.assertEqual(float(area_ratio(1.0)), 1.0)

    def test_c_est_une_fonction_a_MINIMUM_et_pas_monotone(self):
        """Deux branches, donc deux reponses a l'inversion : c'est la source de
        toute la difficulte du module."""
        mach = np.array([0.2, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0])
        valeurs = area_ratio(mach)
        self.assertEqual(float(valeurs.min()), 1.0)
        self.assertTrue(np.all(np.diff(valeurs[:4]) < 0))    # decroit avant 1
        self.assertTrue(np.all(np.diff(valeurs[3:]) > 0))    # croit apres

    def test_la_derivee_s_annule_exactement_au_col(self):
        """C'est la forme analytique du probleme de conditionnement."""
        self.assertEqual(float(area_ratio_derivative(1.0)), 0.0)

    def test_la_derivee_analytique_contre_les_differences_finies(self):
        h = 1e-6
        for mach in (0.3, 0.9, 1.1, 3.0):
            with self.subTest(mach=mach):
                analytique = float(area_ratio_derivative(mach))
                finies = (float(area_ratio(mach + h))
                          - float(area_ratio(mach - h))) / (2 * h)
                self.assertAlmostEqual(analytique, finies, delta=1e-8)

    def test_un_mach_nul_ou_negatif_est_refuse(self):
        for mach in (0.0, -1.0):
            with self.subTest(mach=mach):
                with self.assertRaises(ThermoError):
                    area_ratio(mach)


class InversionDuRapportDeSection(unittest.TestCase):
    """LE CRITERE DE L'ETAPE : l'aller-retour exact sur les deux branches."""

    def test_l_aller_retour_est_exact_sur_les_deux_branches(self):
        """Mesure : 6,2e-15 en subsonique, 2,2e-15 en supersonique, sur cent
        valeurs tirees au hasard. Le plan demandait 1e-12."""
        rng = np.random.default_rng(0)
        for branche, lo, hi in (("subsonic", 0.05, 0.95),
                                ("supersonic", 1.05, 6.0)):
            with self.subTest(branche=branche):
                mach = rng.uniform(lo, hi, 100)
                retour = mach_from_area_ratio(area_ratio(mach), branche)
                np.testing.assert_allclose(retour, mach, rtol=0.0, atol=1e-12)

    def test_la_degradation_pres_du_col_est_INTRINSEQUE_et_mesuree(self):
        """A/A* ~ 1 + (gamma+1)/4 (M-1)^2 : la fonction est PLATE au col, donc
        une erreur eps sur A/A* donne une erreur en sqrt(eps) sur M. Aucun
        algorithme n'y peut rien — l'information n'est pas dans le nombre.

        Mesure : 1,1e-12 a M = 1,0001 ; 2,0e-13 a 1,001 ; 3,3e-14 a 1,01 ;
        2,9e-15 a 1,05. C'est pourquoi le dimensionnement d'un col ne passera
        JAMAIS par cette inversion : c'est la section qui se deduit du Mach.
        """
        ecarts = []
        for mach in (1.0001, 1.001, 1.01, 1.05):
            retour = float(mach_from_area_ratio(area_ratio(mach), "supersonic"))
            ecarts.append(abs(retour - mach))
        self.assertLess(ecarts[0], 1e-11)
        for grossier, fin in zip(ecarts, ecarts[1:]):
            self.assertGreater(grossier, fin)         # ca s'ameliore en s'eloignant

    def test_la_bissection_maison_vaut_brentq(self):
        """SciPy n'entre pas dans le paquet, mais il arbitre dans le test.

        Mesure : ecart maximal 1,0e-15 entre la bissection ecrite ici et
        « brentq » regle a la precision machine.
        """
        for mach in (0.3, 0.7, 1.5, 3.0, 5.0):
            with self.subTest(mach=mach):
                branche = "supersonic" if mach > 1 else "subsonic"
                cible = float(area_ratio(mach))
                maison = float(mach_from_area_ratio(cible, branche))
                lo, hi = (1e-9, 1.0) if mach < 1 else (1.0, 60.0)
                reference = brentq(lambda m: float(area_ratio(m)) - cible,
                                   lo, hi, xtol=1e-15, rtol=8.9e-16)
                self.assertAlmostEqual(maison, reference, delta=1e-12)

    def test_les_deux_branches_donnent_deux_reponses_differentes(self):
        """Et les deux sont valides : c'est la geometrie et la pression aval qui
        tranchent, jamais le rapport de section seul."""
        cible = 2.0
        subsonique = float(mach_from_area_ratio(cible, "subsonic"))
        supersonique = float(mach_from_area_ratio(cible, "supersonic"))
        self.assertLess(subsonique, 1.0)
        self.assertGreater(supersonique, 1.0)
        self.assertAlmostEqual(float(area_ratio(subsonique)), cible, delta=1e-12)
        self.assertAlmostEqual(float(area_ratio(supersonique)), cible, delta=1e-12)

    def test_un_rapport_inferieur_a_un_est_refuse_avec_l_explication(self):
        with self.assertRaises(ThermoError) as leve:
            mach_from_area_ratio(0.5, "subsonic")
        self.assertIn("section critique", str(leve.exception))

    def test_une_branche_inconnue_est_refusee_avec_les_choix(self):
        with self.assertRaises(ThermoError) as leve:
            mach_from_area_ratio(2.0, "transsonique")
        message = str(leve.exception)
        self.assertIn("transsonique", message)
        self.assertIn("supersonic", message)

    def test_un_divergent_demesure_est_refuse_plutot_que_cherche(self):
        with self.assertRaises(ThermoError) as leve:
            mach_from_area_ratio(1e9, "supersonic")
        self.assertIn(f"{MACH_MAX:g}", str(leve.exception))

    def test_la_bissection_a_de_quoi_converger(self):
        """Quatre-vingts iterations amenent [1, 100] sous la resolution des
        flottants : le test protege la constante contre une baisse distraite."""
        self.assertGreaterEqual(BISECTION_STEPS, 60)

    def test_l_inversion_est_vectorisee(self):
        mach = np.linspace(1.2, 4.0, 17)
        groupe = mach_from_area_ratio(area_ratio(mach), "supersonic")
        self.assertEqual(groupe.shape, mach.shape)
        for un, attendu in zip(groupe, mach):
            self.assertAlmostEqual(float(un), float(attendu), delta=1e-12)


class InversionDuRapportDePression(unittest.TestCase):
    """Celle-ci est analytique, et c'est elle que la tuyere utilisera."""

    def test_l_aller_retour_est_exact(self):
        mach = np.linspace(0.05, 6.0, 200)
        retour = mach_from_total_pressure_ratio(total_pressure_ratio(mach))
        np.testing.assert_allclose(retour, mach, rtol=1e-13)

    def test_a_rapport_un_le_mach_est_nul(self):
        self.assertEqual(float(mach_from_total_pressure_ratio(1.0)), 0.0)

    def test_un_rapport_inferieur_a_un_est_refuse(self):
        with self.assertRaises(ThermoError) as leve:
            mach_from_total_pressure_ratio(0.9)
        self.assertIn("pression totale", str(leve.exception))

    def test_elle_n_a_pas_le_probleme_de_conditionnement_de_A_sur_Astar(self):
        """p0/p est strictement croissante et de derivee non nulle : l'inversion
        y garde tous ses chiffres, meme au voisinage de M = 1."""
        for mach in (0.9999, 1.0, 1.0001):
            with self.subTest(mach=mach):
                retour = float(mach_from_total_pressure_ratio(
                    total_pressure_ratio(mach)))
                self.assertAlmostEqual(retour, mach, delta=1e-14)


class Vitesses(unittest.TestCase):
    def test_la_vitesse_du_son_au_niveau_de_la_mer(self):
        self.assertAlmostEqual(float(speed_of_sound(288.15)), 340.29, delta=0.02)

    def test_la_vitesse_utilise_la_temperature_STATIQUE(self):
        """L'erreur la plus frequente de ce calcul est de prendre la totale.

        La vitesse va en sqrt(T), donc l'erreur vaut exactement sqrt(T0/T) :
        a M = 3, sqrt(2,8) = 1,673, soit une vitesse SURESTIMEE DE 67 %. Elle
        ne casse rien d'autre et ne se voit sur aucune figure — d'ou ce test.
        """
        statique, mach = 216.65, 3.0
        totale = statique * float(total_temperature_ratio(mach))
        juste = float(velocity(mach, statique))
        faux = float(velocity(mach, totale))
        self.assertAlmostEqual(juste, 885.2, delta=0.5)
        self.assertAlmostEqual(faux / juste, np.sqrt(totale / statique),
                               delta=1e-12)
        self.assertAlmostEqual(faux / juste, 1.6733, delta=1e-4)

    def test_une_temperature_non_positive_est_refusee(self):
        with self.assertRaises(ThermoError):
            speed_of_sound(0.0)


class ChocDroit(unittest.TestCase):
    """Rankine-Hugoniot, ses valeurs tabulees et ses deux limites."""

    def test_les_valeurs_tabulees(self):
        """Tables standard, gamma = 1,4."""
        for mach, m2, p_ratio, p0_ratio in (
                (1.5, 0.70109, 2.45833, 0.92979),
                (2.0, 0.57735, 4.50000, 0.72087),
                (3.0, 0.47519, 10.33333, 0.32834),
                (5.0, 0.41523, 29.00000, 0.06172)):
            with self.subTest(mach=mach):
                choc = normal_shock(mach)
                self.assertAlmostEqual(choc.mach_downstream, m2, delta=1e-5)
                self.assertAlmostEqual(choc.pressure_ratio, p_ratio, delta=1e-5)
                self.assertAlmostEqual(choc.total_pressure_ratio, p0_ratio,
                                       delta=1e-5)

    def test_un_choc_de_force_nulle_est_EXACTEMENT_l_identite(self):
        """A M = 1 les trois rapports doivent valoir 1 au bit pres, et pas
        0,9999999 : les numerateurs et denominateurs sont egaux terme a terme."""
        choc = normal_shock(1.0)
        self.assertEqual(choc.mach_downstream, 1.0)
        self.assertEqual(choc.pressure_ratio, 1.0)
        self.assertEqual(choc.density_ratio, 1.0)
        self.assertEqual(choc.temperature_ratio, 1.0)
        self.assertEqual(choc.total_pressure_ratio, 1.0)

    def test_la_limite_du_choc_fort(self):
        """M2 tend vers sqrt((gamma-1)/2gamma) = 0,377964, et la convergence est
        en 1/M1^2 : ecart 9,7e-5 a M1 = 100, 9,7e-7 a M1 = 1 000."""
        limite = strong_shock_limit()
        self.assertAlmostEqual(limite, 0.377964, delta=1e-6)
        self.assertAlmostEqual(normal_shock(1e3).mach_downstream, limite,
                               delta=1e-6)
        ecarts = [abs(normal_shock(m).mach_downstream - limite)
                  for m in (1e2, 1e3, 1e4)]
        for grossier, fin in zip(ecarts, ecarts[1:]):
            self.assertAlmostEqual(grossier / fin, 100.0, delta=5.0)

    def test_la_temperature_totale_traverse_le_choc_INCHANGEE(self):
        """Controle INDEPENDANT : « T02 = T01 » ne sert a aucune des formules
        du module, c'est une consequence de la conservation de l'energie.
        Mesure : 1,000000000000000 pour cinq Mach amont.
        """
        for mach in (1.5, 2.0, 3.0, 5.0, 10.0):
            with self.subTest(mach=mach):
                choc = normal_shock(mach)
                rapport = (choc.temperature_ratio
                           * float(total_temperature_ratio(choc.mach_downstream))
                           / float(total_temperature_ratio(mach)))
                self.assertAlmostEqual(rapport, 1.0, delta=1e-14)

    def test_le_flux_de_masse_traverse_le_choc_INCHANGE(self):
        """Second controle independant : « rho2 u2 = rho1 u1 »."""
        for mach in (1.5, 2.0, 3.0, 5.0, 10.0):
            with self.subTest(mach=mach):
                choc = normal_shock(mach)
                flux = (choc.density_ratio * choc.mach_downstream
                        * np.sqrt(choc.temperature_ratio) / mach)
                self.assertAlmostEqual(float(flux), 1.0, delta=1e-14)

    def test_un_choc_ralentit_toujours_et_perd_toujours_de_la_pression_totale(self):
        for mach in (1.01, 1.5, 2.0, 3.0, 5.0, 10.0):
            with self.subTest(mach=mach):
                choc = normal_shock(mach)
                self.assertLess(choc.mach_downstream, 1.0)
                self.assertGreater(choc.pressure_ratio, 1.0)
                self.assertLess(choc.total_pressure_ratio, 1.0)

    def test_les_monotonies(self):
        machs = (1.5, 2.0, 3.0, 5.0, 8.0)
        chocs = [normal_shock(m) for m in machs]
        for avant, apres in zip(chocs, chocs[1:]):
            self.assertGreater(avant.mach_downstream, apres.mach_downstream)
            self.assertLess(avant.pressure_ratio, apres.pressure_ratio)
            self.assertGreater(avant.total_pressure_ratio,
                               apres.total_pressure_ratio)

    def test_un_choc_subsonique_est_refuse_au_nom_du_SECOND_PRINCIPE(self):
        """Les formules donneraient un resultat — elles sont algebriquement
        valides — mais avec « p02/p01 > 1 », c'est-a-dire un choc de detente,
        qui n'existe pas. Rendre un nombre laisserait croire le contraire.
        """
        for mach in (0.5, 0.99):
            with self.subTest(mach=mach):
                with self.assertRaises(ThermoError) as leve:
                    normal_shock(mach)
                self.assertIn("second principe", str(leve.exception))

    def test_le_choc_se_decrit(self):
        texte = normal_shock(3.0).describe()
        self.assertIn("3.000", texte)
        self.assertIn("p02/p01", texte)
        self.assertIsInstance(normal_shock(2.0), NormalShock)


if __name__ == "__main__":
    unittest.main()
