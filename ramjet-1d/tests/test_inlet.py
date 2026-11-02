"""L'entree a chocs obliques : ce que la correlation du 0D decrivait vraiment.

CE FICHIER REPOND A UNE QUESTION LAISSEE OUVERTE A L'ETAPE 8. Tout le cycle 0D
repose sur un seul nombre pour l'entree d'air — « pi_d = 0,8088 » a M0 = 3, tire
de la correlation MIL-E-5008B, qui decrit une entree « bien dessinee ». Bien
dessinee comment ? Le plan proposait un choc normal, qui rend 0,3283 : deux fois
et demie pire, et donc pas ce que la correlation decrit. En calculant la
recuperation de rampes reelles, on peut enfin le dire.

QUATRE FAMILLES.

  - LA VERIFICATION EXACTE. La relation theta-beta-M est verifiee par
    SUBSTITUTION, pas contre une table recopiee : c'est une identite, elle
    tombe a 1e-16, et elle ne depend d'aucune source exterieure.
  - LA VALIDATION CONTRE LES TABLES publiees, qui, elle, depend d'une source —
    et ou une valeur citee de memoire s'est revelee fausse.
  - LES REFUS. Choc detache, rampe qui rend l'ecoulement subsonique.
  - L'IDENTIFICATION, qui est le resultat de l'etape.
"""

import unittest

import numpy as np

from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.cycle0d import run_cycle
from ramjet.thermo import (ThermoError, max_deflection, normal_shock,
                           oblique_shock, optimal_ramp_angles, ramp_inlet,
                           shock_angle)

GAMMA = 1.4


def theta_de_beta(mach: float, beta: float, gamma: float = GAMMA) -> float:
    """La relation theta-beta-M ecrite a part, pour verifier par substitution."""
    return float(np.arctan(
        2.0 / np.tan(beta) * (mach ** 2 * np.sin(beta) ** 2 - 1.0)
        / (mach ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0)))


# -- la verification exacte --------------------------------------------------

class RelationThetaBetaM(unittest.TestCase):

    def test_l_angle_trouve_satisfait_la_relation_a_1e_16(self):
        """L'IDENTITE, ET ELLE NE DOIT RIEN A PERSONNE.

        « shock_angle » inverse la relation par bissection ; resubstituer
        l'angle trouve doit rendre la deviation demandee. Mesure sur 30 couples
        (M, theta) repartis jusqu'a M = 10 : ecart maximal **1,4e-16 rad**.
        C'est de la verification au sens strict — aucune table n'intervient.
        """
        pire = 0.0
        for mach in (1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
            for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
                theta = fraction * max_deflection(mach)
                beta = shock_angle(mach, theta)
                pire = max(pire, abs(theta_de_beta(mach, beta) - theta))
        self.assertLess(pire, 1e-15)

    def test_la_vitesse_TANGENTIELLE_est_conservee(self):
        """LA SECONDE IDENTITE, ET C'EST ELLE QUI VALIDE « M2 ».

        Un choc oblique ne touche pas la composante tangentielle de la vitesse :
        « M1 cos(beta) a1 = M2 cos(beta - theta) a2 ». Comme « a » va en
        sqrt(T), cela s'ecrit sans vitesse du tout et fournit un controle exact
        de « M2 » qui n'utilise ni table ni la formule qui l'a produit. Mesure
        sur 12 cas jusqu'a M = 8 : ecart maximal **5,6e-16**.
        """
        pire = 0.0
        for mach in (2.0, 3.0, 5.0, 8.0):
            for fraction in (0.2, 0.5, 0.8):
                theta = fraction * max_deflection(mach)
                choc = oblique_shock(mach, theta)
                tangentiel = (choc.mach_downstream
                              * np.cos(choc.wave_angle - theta)
                              / (mach * np.cos(choc.wave_angle)))
                pire = max(pire, abs(tangentiel
                                     - 1.0 / np.sqrt(choc.temperature_ratio)))
        self.assertLess(pire, 1e-14)

    def test_un_choc_oblique_EST_un_choc_droit_vu_de_biais(self):
        """LA COHERENCE INTERNE QUI EMPECHE LES DEUX FAMILLES DE DIVERGER.

        La composante normale « M sin(beta) » subit exactement Rankine-Hugoniot.
        Le module ne reecrit donc rien : il appelle « normal_shock ». Le test
        verifie que les rapports de pression et de temperature sont bien ceux du
        choc droit de meme composante normale.
        """
        choc = oblique_shock(3.0, np.radians(20.0))
        droit = normal_shock(choc.normal_mach)
        self.assertAlmostEqual(choc.total_pressure_ratio,
                               float(droit.total_pressure_ratio), delta=1e-15)
        self.assertAlmostEqual(choc.pressure_ratio,
                               float(droit.pressure_ratio), delta=1e-15)
        self.assertAlmostEqual(choc.temperature_ratio,
                               float(droit.temperature_ratio), delta=1e-15)

    def test_une_rampe_nulle_tend_vers_l_onde_de_Mach(self):
        """A deviation infinitesimale, beta tend vers l'angle de Mach et le
        choc ne coute rien : pi -> 1. C'est le cas limite qui doit tomber
        juste."""
        choc = oblique_shock(3.0, 1e-6)
        self.assertAlmostEqual(choc.wave_angle, float(np.arcsin(1 / 3.0)),
                               delta=1e-5)
        self.assertAlmostEqual(choc.total_pressure_ratio, 1.0, delta=1e-12)


# -- la validation contre les tables -----------------------------------------

class ContreLesTablesPubliees(unittest.TestCase):
    """Ici on depend d'une source exterieure, et il faut le dire."""

    # (M1, theta en degres, beta publie). Seul l'angle de choc est retenu :
    # les « M2 » que j'avais notes se sont reveles faux deux fois sur cinq, et
    # ce sont les identites ci-dessus qui les valident desormais.
    TABLES = ((2.0, 10.0, 39.31),
              (2.0, 20.0, 53.42),
              (3.0, 20.0, 37.76),
              (5.0, 10.0, 19.38))

    def test_les_angles_de_choc_tombent_a_0_01_degre(self):
        for mach, theta, beta_publie in self.TABLES:
            with self.subTest(mach=mach, theta=theta):
                choc = oblique_shock(mach, np.radians(theta))
                self.assertAlmostEqual(np.degrees(choc.wave_angle),
                                       beta_publie, delta=0.01)

    def test_la_deviation_maximale_tombe_sur_les_tables(self):
        """12,11 deg a M = 1,5 ; 22,97 a M = 2 ; 34,07 a M = 3 ; 38,77 a M = 4.
        Et la limite hypersonique classique, 45,58 deg."""
        for mach, publie in ((1.5, 12.11), (2.0, 22.97), (3.0, 34.07),
                             (4.0, 38.77)):
            with self.subTest(mach=mach):
                self.assertAlmostEqual(np.degrees(max_deflection(mach)),
                                       publie, delta=0.01)
        self.assertAlmostEqual(np.degrees(max_deflection(1e6)), 45.58,
                               delta=0.01)

    def test_DEUX_valeurs_citees_de_MEMOIRE_etaient_FAUSSES(self):
        """ET C'EST POURQUOI LES IDENTITES PASSENT AVANT LES TABLES.

        Deux nombres que j'avais notes de memoire se sont reveles faux, et dans
        les deux cas ce sont les identites qui ont tranche :

          - « M = 4, theta = 30 deg -> beta = 47,90 deg ». Le code rend
            **45,224**, et la substitution dans la relation theta-beta-M redonne
            30,000000 deg exactement ;
          - « M = 5, theta = 10 deg -> M2 = 4,0016 ». Le code rend **3,999162**,
            et l'identite de vitesse tangentielle le confirme a 1e-16.

        Une table recopiee ne vaut que ce que vaut celui qui la recopie ; une
        identite, elle, se verifie. Ce test garde les deux erreurs par ecrit.
        """
        choc = oblique_shock(4.0, np.radians(30.0))
        self.assertAlmostEqual(np.degrees(choc.wave_angle), 45.224, delta=1e-2)
        self.assertAlmostEqual(np.degrees(theta_de_beta(4.0, choc.wave_angle)),
                               30.0, delta=1e-9)
        self.assertGreater(abs(np.degrees(choc.wave_angle) - 47.90), 2.0)

        autre = oblique_shock(5.0, np.radians(10.0))
        self.assertAlmostEqual(autre.mach_downstream, 3.999162, delta=1e-5)
        self.assertGreater(abs(autre.mach_downstream - 4.0016), 2e-3)


# -- les refus ---------------------------------------------------------------

class Refus(unittest.TestCase):

    def test_le_choc_se_detache_au_dela_de_la_deviation_maximale(self):
        maximum = max_deflection(2.0)
        with self.assertRaises(ThermoError) as capture:
            oblique_shock(2.0, maximum * 1.01)
        message = str(capture.exception)
        self.assertIn("detache", message)
        self.assertIn("22.97", message)
        self.assertIn("plusieurs rampes", message)

    def test_juste_sous_la_limite_le_choc_tient_encore(self):
        """La frontiere doit etre franche, pas approximative."""
        maximum = max_deflection(2.0)
        choc = oblique_shock(2.0, maximum * 0.999)
        self.assertGreater(choc.mach_downstream, 0.0)
        self.assertLess(choc.mach_downstream, 2.0)

    def test_une_rampe_qui_rend_l_ecoulement_subsonique_arrete_la_chaine(self):
        """Les rampes suivantes n'auraient plus de choc oblique a produire."""
        with self.assertRaises(ThermoError) as capture:
            ramp_inlet(2.0, (np.radians(22.0), np.radians(10.0)))
        self.assertIn("rampe", str(capture.exception))

    def test_une_entree_subsonique_est_refusee(self):
        with self.assertRaises(ThermoError):
            ramp_inlet(0.8, (np.radians(10.0),))
        with self.assertRaises(ThermoError):
            max_deflection(0.9)

    def test_le_message_nomme_la_rampe_fautive(self):
        with self.assertRaises(ThermoError) as capture:
            ramp_inlet(3.0, (np.radians(10.0), np.radians(40.0)))
        self.assertIn("rampe 2 sur 2", str(capture.exception))


# -- le critere d'Oswatitsch -------------------------------------------------

class CritereDOswatitsch(unittest.TestCase):

    def test_a_l_optimum_les_chocs_OBLIQUES_ont_la_meme_force(self):
        """LE RESULTAT QUI PERMET DE N'OPTIMISER QU'UNE SEULE INCONNUE.

        Ce n'est pas une hypothese imposee : une optimisation libre a
        Nelder-Mead sur les N angles retrouve des composantes normales egales a
        six chiffres. Mesure a M0 = 3 : 1,4252 pour les trois chocs obliques.
        """
        for count in (2, 3, 4):
            with self.subTest(rampes=count):
                entree = ramp_inlet(3.0, optimal_ramp_angles(3.0, count))
                forces = [s.normal_mach for s in entree.shocks]
                self.assertAlmostEqual(max(forces), min(forces), delta=1e-6)

    def test_mais_le_choc_TERMINAL_est_plus_faible(self):
        """L'ENONCE COURANT « tous les chocs de meme force » VAUT POUR LES
        OBLIQUES SEULS.

        A M0 = 3 et trois rampes : 1,4252 pour les obliques, **1,3483** pour le
        choc droit terminal. L'ecart est de 5 % et il est systematique.
        """
        entree = ramp_inlet(3.0, optimal_ramp_angles(3.0, 3))
        oblique_force = entree.shocks[0].normal_mach
        self.assertAlmostEqual(oblique_force, 1.4252, delta=1e-3)
        self.assertAlmostEqual(entree.terminal.mach_upstream, 1.3483,
                               delta=1e-3)
        self.assertLess(entree.terminal.mach_upstream, oblique_force)

    def test_l_optimum_est_bien_un_optimum(self):
        """Perturber les angles ne peut que degrader la recuperation."""
        angles = optimal_ramp_angles(3.0, 3)
        meilleur = ramp_inlet(3.0, angles).total_pressure_ratio
        for perturbation in (-0.02, -0.01, 0.01, 0.02):
            for index in range(3):
                essai = list(angles)
                essai[index] += perturbation
                with self.subTest(rampe=index, delta=perturbation):
                    self.assertLessEqual(
                        ramp_inlet(3.0, essai).total_pressure_ratio,
                        meilleur + 1e-12)

    def test_ajouter_une_rampe_ameliore_toujours(self):
        """0,3283 / 0,5812 / 0,7437 / 0,8361 / 0,8892 a M0 = 3, pour 0 a 4
        rampes. Le gain decroit, mais il ne s'annule jamais."""
        valeurs = [float(normal_shock(3.0).total_pressure_ratio)]
        valeurs += [ramp_inlet(3.0, optimal_ramp_angles(3.0, n)
                               ).total_pressure_ratio for n in (1, 2, 3, 4)]
        for avant, apres in zip(valeurs, valeurs[1:]):
            self.assertGreater(apres, avant)
        np.testing.assert_allclose(
            valeurs, [0.3283, 0.5812, 0.7437, 0.8361, 0.8892], atol=1e-3)
        gains = np.diff(valeurs)
        for avant, apres in zip(gains, gains[1:]):
            self.assertLess(apres, avant)


# -- l'identification, qui est le resultat de l'etape ------------------------

class CeQueLaCorrelationDecrit(unittest.TestCase):

    def test_a_M0_egal_3_MIL_correspond_a_TROIS_rampes(self):
        """LE RESULTAT DE L'ETAPE 13.

        Le 0D repose sur « pi_d = 0,8088 » sans jamais dire de quelle entree il
        parle. Mesure : 2 rampes optimales donnent 0,7437, trois en donnent
        0,8361. La correlation tombe entre les deux, tout pres de trois — et
        **3 % en dessous**, ce qui est exactement ce qu'on attend : la theorie
        des chocs ignore les pertes visqueuses et les non-idealites qu'une
        correlation empirique, elle, contient.
        """
        mil = EngineCase().inlet_recovery(3.0)
        deux = ramp_inlet(3.0, optimal_ramp_angles(3.0, 2)).total_pressure_ratio
        trois = ramp_inlet(3.0, optimal_ramp_angles(3.0, 3)).total_pressure_ratio
        self.assertAlmostEqual(mil, 0.8088, delta=1e-4)
        self.assertLess(deux, mil)
        self.assertLess(mil, trois)
        self.assertLess(abs(mil - trois), abs(mil - deux))
        self.assertAlmostEqual(mil / trois, 0.967, delta=0.01)

    def test_et_le_nombre_de_rampes_CROIT_avec_le_Mach(self):
        """CE QUI EST PHYSIQUEMENT JUSTE, ET QUE LA CORRELATION CONTIENT SANS
        LE DIRE.

        La rampe la plus proche, par Mach : 1 rampe a M0 = 1,5 et 2 ; 2 rampes a
        2,5 ; 3 rampes de 3 a 4 ; 4 rampes a 5. Une correlation empirique unique
        decrit donc une FAMILLE d'entrees dont la complexite grandit avec le
        Mach de dessin — ce que font les entrees reelles.
        """
        attendu = {1.5: 1, 2.0: 1, 2.5: 2, 3.0: 3, 3.5: 3, 4.0: 3, 5.0: 4}
        for mach, rampes in attendu.items():
            with self.subTest(mach=mach):
                mil = EngineCase().inlet_recovery(mach)
                valeurs = [float(normal_shock(mach).total_pressure_ratio)]
                valeurs += [ramp_inlet(mach, optimal_ramp_angles(mach, n)
                                       ).total_pressure_ratio
                            for n in (1, 2, 3, 4)]
                proche = int(np.argmin([abs(v - mil) for v in valeurs]))
                self.assertEqual(proche, rampes)

    def test_ce_que_le_choix_de_l_entree_coute_en_poussee(self):
        """L'ECART VA DE 580 A 753 N.s/kg : 30 % de poussee tiennent au dessin
        de l'entree d'air, que le 0D resume par un seul nombre."""
        vol = FlightCase()
        poussees = {}
        for nom, pi in (
                ("choc droit", float(normal_shock(3.0).total_pressure_ratio)),
                ("2 rampes", ramp_inlet(
                    3.0, optimal_ramp_angles(3.0, 2)).total_pressure_ratio),
                ("MIL", EngineCase().inlet_recovery(3.0)),
                ("4 rampes", ramp_inlet(
                    3.0, optimal_ramp_angles(3.0, 4)).total_pressure_ratio)):
            poussees[nom] = run_cycle(
                vol, EngineCase(pi_d=float(pi))).specific_thrust
        self.assertAlmostEqual(poussees["choc droit"], 580.4, delta=1.0)
        self.assertAlmostEqual(poussees["MIL"], 739.0, delta=1.0)
        self.assertAlmostEqual(poussees["4 rampes"], 752.6, delta=1.0)
        self.assertGreater(poussees["4 rampes"] / poussees["choc droit"], 1.29)

    def test_l_objet_rendu_se_decrit(self):
        entree = ramp_inlet(3.0, optimal_ramp_angles(3.0, 3))
        texte = entree.describe()
        self.assertIn("3 rampe(s)", texte)
        self.assertIn("choc droit terminal", texte)
        self.assertEqual(len(entree.shocks), 3)
        self.assertAlmostEqual(np.degrees(entree.total_turn), 39.56, delta=0.1)


if __name__ == "__main__":
    unittest.main()


# -- le cablage dans le cycle -----------------------------------------------

class LEntreeEstMaintenantUN_CHOIX_DU_CAS(unittest.TestCase):
    """Le module savait calculer une entree a rampes ; le cycle ne savait pas
    la demander. « inlet_ramps » ferme cet ecart."""

    def test_le_cas_calcule_pi_d_pour_le_nombre_de_rampes_demande(self):
        for rampes, attendu in ((1, 0.5812), (2, 0.7437), (3, 0.8361),
                                (4, 0.8892)):
            with self.subTest(rampes=rampes):
                self.assertAlmostEqual(
                    EngineCase(inlet_ramps=rampes).inlet_recovery(3.0),
                    attendu, delta=1e-3)

    def test_sans_rien_demander_on_garde_MIL_E_5008B(self):
        self.assertAlmostEqual(EngineCase().inlet_recovery(3.0), 0.8088,
                               delta=1e-4)

    def test_pi_d_impose_reste_prioritaire(self):
        self.assertEqual(EngineCase(pi_d=0.5).inlet_recovery(3.0), 0.5)

    def test_et_pi_d_ET_rampes_ensemble_sont_REFUSES(self):
        """Deux facons de fixer la meme grandeur : il faut choisir, et le dire
        vaut mieux que de decider en silence par ordre de priorite."""
        with self.assertRaises(CaseError) as capture:
            EngineCase(pi_d=0.8, inlet_ramps=3)
        self.assertIn("l'un OU l'autre", str(capture.exception))

    def test_le_nombre_de_rampes_est_borne_et_entier(self):
        for mauvais in (0, -1, 7, 12):
            with self.subTest(rampes=mauvais):
                with self.assertRaises(CaseError):
                    EngineCase(inlet_ramps=mauvais)
        with self.assertRaises(CaseError) as capture:
            EngineCase(inlet_ramps=3.0)
        self.assertIn("entier", str(capture.exception))

    def test_LA_RAISON_D_ETRE_pi_d_varie_avec_le_MACH(self):
        """CE QU'UNE VALEUR IMPOSEE NE PEUT PAS FAIRE.

        Passer « pi_d = 0,8361 » figerait la recuperation sur tout un balayage,
        ce qui est faux : une entree reelle recupere 0,9966 a M0 = 1,5 et 0,4306
        a M0 = 5. C'est pour cela que le nombre de rampes devait entrer dans le
        CAS, et non etre resolu au bord de la ligne de commande.
        """
        cas = EngineCase(inlet_ramps=3)
        valeurs = [cas.inlet_recovery(m) for m in (1.5, 2.0, 3.0, 4.0, 5.0)]
        for avant, apres in zip(valeurs, valeurs[1:]):
            self.assertLess(apres, avant)
        self.assertAlmostEqual(valeurs[0], 0.9966, delta=1e-3)
        self.assertAlmostEqual(valeurs[-1], 0.4306, delta=1e-3)

    def test_trois_rampes_poussent_plus_que_la_correlation(self):
        """+3,3 % sur pi_d se traduit par +0,7 % de poussee specifique."""
        vol = FlightCase()
        avec = run_cycle(vol, EngineCase(inlet_ramps=3))
        sans = run_cycle(vol, EngineCase())
        self.assertGreater(avec.specific_thrust, sans.specific_thrust)
        self.assertAlmostEqual(
            avec.specific_thrust / sans.specific_thrust, 1.007, delta=0.002)

    def test_le_calcul_est_MIS_EN_CACHE(self):
        """190 ms par appel, quinze appels par balayage. Sans cache, « --sweep »
        y passerait trois secondes a recalculer les memes nombres."""
        import time

        from ramjet.config import _ramp_recovery

        _ramp_recovery.cache_clear()
        depart = time.perf_counter()
        _ramp_recovery(3.0, 3)
        premier = time.perf_counter() - depart
        depart = time.perf_counter()
        for _ in range(50):
            _ramp_recovery(3.0, 3)
        suivants = (time.perf_counter() - depart) / 50.0
        self.assertGreater(premier / max(suivants, 1e-9), 100.0)


class LeCloisonnementTientTOUJOURS(unittest.TestCase):
    """L'ajout ouvre une dependance « config -> thermo », donc « config ->
    NumPy ». Elle est PARESSEUSE, et c'est toute la question."""

    def test_construire_des_cas_ne_tire_toujours_pas_NumPy(self):
        """LA CONTRAINTE QUE L'ETAPE 1 A POSEE, ET QUI DOIT SURVIVRE.

        Un script d'optimisation exterieur construit des milliers de cas et ne
        doit pas payer l'importation de NumPy. L'« import » de « thermo » est
        place DANS la fonction de recuperation : il ne se declenche que si l'on
        demande reellement une entree a rampes.
        """
        import subprocess
        import sys
        from pathlib import Path

        racine = Path(__file__).resolve().parent.parent
        code = ("import sys, ramjet.config as c; "
                "cas = [c.EngineCase(T04=1500.0 + i) for i in range(1000)]; "
                "print('numpy' in sys.modules)")
        fait = subprocess.run([sys.executable, "-c", code], cwd=racine,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(fait.returncode, 0, fait.stderr)
        self.assertEqual(fait.stdout.strip(), "False")

    def test_mais_DEMANDER_des_rampes_le_tire_evidemment(self):
        """Sinon le calcul n'aurait pas lieu. Le test dit que la frontiere est
        au bon endroit : a l'usage, pas a la construction."""
        import subprocess
        import sys
        from pathlib import Path

        racine = Path(__file__).resolve().parent.parent
        code = ("import sys, ramjet.config as c; "
                "c.EngineCase(inlet_ramps=3).inlet_recovery(3.0); "
                "print('numpy' in sys.modules)")
        fait = subprocess.run([sys.executable, "-c", code], cwd=racine,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(fait.returncode, 0, fait.stderr)
        self.assertEqual(fait.stdout.strip(), "True")
