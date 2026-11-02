"""Le frottement : la forme fermee de Fanno, et le troisieme terme du solveur.

ETAPE 12. Le solveur avait deux termes et deux oracles — l'isentropique pour la
section, Rayleigh pour la chaleur. Le troisieme terme en a maintenant un aussi,
et c'est ce qui permet de savoir LEQUEL est faux quand un profil deraille : on
peut allumer les trois separement, chacun contre sa propre forme fermee.

QUATRE FAMILLES.

  - LA FORME FERMEE elle-meme, et une identite remarquable qui la relie a un
    module ecrit separement.
  - LE SOLVEUR CONTRE ELLE, a 1e-13, y compris le bilan de quantite de
    mouvement une fois la trainee parietale remise dedans.
  - CE QUE LE FROTTEMENT FAIT VRAIMENT, mesure : il accelere le subsonique, il
    rapproche du blocage, et il ne change presque rien a ce moteur-ci.
  - LES REFUS.
"""

import unittest

import numpy as np

from ramjet import fanno
from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.coupling import converge
from ramjet.cycle0d import run_cycle
from ramjet.duct import Duct, Zone, from_cycle
from ramjet.solver1d import (NO_FRICTION, ChokedFlow, Friction, HeatRelease,
                             integrate_duct, momentum_residual, no_heat)
from ramjet.thermo import area_ratio

GAMMA, R = 1.4, 287.05
DIAMETRE, CF = 0.16, 0.004
SECTION = np.pi * DIAMETRE ** 2 / 4.0


def tube(length: float, area: float = SECTION) -> Duct:
    return Duct((Zone("tube", 0.0, length, area, area),))


# -- la forme fermee ---------------------------------------------------------

class FormeFermee(unittest.TestCase):

    def test_la_longueur_reduite_s_annule_en_M_egal_1(self):
        self.assertAlmostEqual(float(fanno.friction_length(1.0)), 0.0,
                               delta=1e-15)

    def test_elle_croit_des_DEUX_cotes(self):
        """LE FAIT PHYSIQUE CENTRAL : le frottement pousse le Mach vers 1 par
        les deux bords, exactement comme la chaleur."""
        for machs in (np.linspace(0.05, 0.99, 40),
                      np.linspace(1.01, 6.0, 40)):
            valeurs = fanno.friction_length(machs)
            pentes = np.diff(valeurs)
            with self.subTest(regime="subsonique" if machs[0] < 1 else "super"):
                # Decroissante en subsonique, croissante en supersonique :
                # dans les deux cas on va vers zero en approchant de M = 1.
                self.assertTrue(bool(np.all(pentes < 0.0))
                                or bool(np.all(pentes > 0.0)))
                self.assertGreater(float(np.min(valeurs)), 0.0)

    def test_les_valeurs_mesurees(self):
        attendu = {0.10: 66.9216, 0.50: 1.0691, 0.80: 0.0723, 2.00: 0.3050}
        for mach, valeur in attendu.items():
            with self.subTest(mach=mach):
                self.assertAlmostEqual(float(fanno.friction_length(mach)),
                                       valeur, delta=1e-4)

    def test_la_limite_supersonique_vaut_0_8215(self):
        """« -1/gamma + (gamma+1)/(2 gamma) ln((gamma+1)/(gamma-1)) ».

        UN ECOULEMENT SUPERSONIQUE NE PEUT PAS ALLER PLUS LOIN QUE CA, si vite
        qu'il aille. C'est le pendant exact de la limite « 1 - 1/gamma^2 » de
        Rayleigh, et l'asymetrie avec le subsonique est enorme : a M = 0,10 il
        reste 66,92, soit 900 fois plus.
        """
        limite = fanno.supersonic_limit()
        self.assertAlmostEqual(limite, 0.8215, delta=1e-4)
        self.assertLess(float(fanno.friction_length(1e6)), limite)
        self.assertAlmostEqual(float(fanno.friction_length(1e6)), limite,
                               delta=1e-6)
        self.assertGreater(float(fanno.friction_length(0.10)) / limite, 80.0)

    def test_p0_sur_p0_etoile_EST_A_sur_A_etoile(self):
        """L'IDENTITE REMARQUABLE, ET C'EST UN CONTROLE CROISE GRATUIT.

        Le rapport de pression totale de Fanno est litteralement la meme
        fonction que le rapport de section de l'isentropique. Les deux modules
        ont ete ecrits separement, a des etapes differentes, et rendent le meme
        nombre au bit pres : ecart **0,0 exactement** sur cinq Mach.
        """
        machs = np.array([0.2, 0.5, 0.9, 1.5, 3.0])
        np.testing.assert_array_equal(fanno.total_pressure_ratio(machs),
                                      area_ratio(machs))

    def test_les_rapports_valent_1_en_M_egal_1(self):
        for nom in ("temperature_ratio", "pressure_ratio",
                    "total_pressure_ratio", "velocity_ratio"):
            with self.subTest(rapport=nom):
                self.assertAlmostEqual(
                    float(getattr(fanno, nom)(1.0)), 1.0, delta=1e-15)

    def test_l_inversion_retrouve_le_Mach_sur_les_deux_branches(self):
        for branche, machs in (("subsonic", np.linspace(0.05, 0.98, 30)),
                               ("supersonic", np.linspace(1.02, 8.0, 30))):
            with self.subTest(branche=branche):
                retour = fanno.mach_from_friction_length(
                    fanno.friction_length(machs), branche)
                self.assertLess(float(np.max(np.abs(retour - machs))), 1e-10)


# -- le solveur contre la forme fermee ---------------------------------------

class LeSolveurContreFanno(unittest.TestCase):

    def test_le_troisieme_terme_tombe_a_1e_13(self):
        """LA VERIFICATION DE L'ETAPE : section constante, pas de chaleur,
        frottement seul. Mesure : 2,4e-15 a 1 m, 1,5e-13 a 6,5 m — l'ecart
        croit avec la longueur parcourue, ce qui est l'accumulation normale
        d'une integration, pas une erreur de formule.
        """
        frottement = Friction(CF)
        for longueur in (1.0, 3.0, 5.0, 6.5):
            with self.subTest(longueur=longueur):
                conduit = tube(longueur)
                profil = integrate_duct(conduit, no_heat(300.0, longueur), 0.30,
                                        5.0, friction=frottement, n_points=200)
                exact = fanno.mach_after(0.30, longueur, DIAMETRE, CF)
                self.assertAlmostEqual(float(profil.mach[-1]), exact,
                                       delta=1e-12)

    def test_le_diametre_hydraulique_est_bien_celui_du_tube(self):
        self.assertAlmostEqual(Friction(CF).hydraulic_diameter(SECTION),
                               DIAMETRE, delta=1e-15)

    def test_le_bilan_de_quantite_de_mouvement_EXIGE_la_trainee(self):
        """SANS ELLE, LE CONTROLE SEMBLERAIT DIRE QUE LE SOLVEUR EST CASSE.

        L'identite d'impulsion devient « F(sortie) - F(entree) = integrale p dA
        - integrale tau_w dA_paroi ». La trainee parietale vaut
        « 1/2 rho u^2 (4 C_f/D) A » par metre : c'est exactement le terme source
        de l'EDO, mais atteint par un bilan de quantite de mouvement, chemin que
        l'integration n'a pas emprunte. Mesure : **2,4e-15 avec la trainee,
        2,3e-02 sans** — quatorze ordres de grandeur.
        """
        conduit = tube(6.5)
        frottement = Friction(CF)
        profil = integrate_duct(conduit, no_heat(300.0, 6.5), 0.30, 5.0,
                                friction=frottement, n_points=201)
        self.assertLess(momentum_residual(profil, conduit, frottement), 1e-12)
        sans = momentum_residual(profil, conduit, NO_FRICTION)
        self.assertGreater(sans, 1e-3)
        self.assertGreater(sans / momentum_residual(profil, conduit,
                                                    frottement), 1e10)

    def test_sans_frottement_le_solveur_est_inchange(self):
        """LE TERME AJOUTE NE DOIT RIEN CASSER : a C_f = 0, l'EDO doit rendre
        exactement ce qu'elle rendait avant, au bit pres."""
        conduit = tube(1.0)
        chaleur = HeatRelease(300.0, 600.0, 0.0, 1.0)
        avec = integrate_duct(conduit, chaleur, 0.30, 5.0,
                              friction=Friction(0.0), n_points=50)
        sans = integrate_duct(conduit, chaleur, 0.30, 5.0, n_points=50)
        np.testing.assert_array_equal(avec.mach, sans.mach)


# -- ce que le frottement fait vraiment --------------------------------------

class CeQueLeFrottementFait(unittest.TestCase):

    def test_il_ACCELERE_un_ecoulement_subsonique(self):
        """CE QUI SURPREND, ET QUI EST VRAI.

        On attend qu'il freine. Il fait chuter la pression, donc la vitesse
        debitante — mais la temperature chute aussi, donc la vitesse du son, et
        le rapport des deux monte. Mesure : M passe de 0,300 a 0,315 sur 6,5 m.
        """
        conduit = tube(6.5)
        profil = integrate_duct(conduit, no_heat(300.0, 6.5), 0.30, 5.0,
                                friction=Friction(CF), n_points=100)
        self.assertGreater(float(profil.mach[-1]), 0.30)
        self.assertAlmostEqual(float(profil.mach[-1]), 0.3149, delta=1e-3)
        # Et il decelere le supersonique : le meme fait, vu de l'autre cote.
        # Sur 2 m seulement, parce qu'a M = 2 la longueur de Fanno n'est que de
        # 3,05 m — un ecoulement supersonique va vite au blocage.
        court = tube(2.0)
        amont = integrate_duct(court, no_heat(300.0, 2.0), 2.0, 5.0,
                               friction=Friction(CF), n_points=100)
        self.assertLess(float(amont.mach[-1]), 2.0)

    def test_le_supersonique_bloque_EXACTEMENT_a_la_longueur_de_Fanno(self):
        """UN CONTROLE CROISE QUE LE TEST PRECEDENT A REVELE EN ECHOUANT.

        A M = 2, C_f = 0,004 et D = 0,16 m, la forme fermee donne
        « L_max = 0,3050 x 0,16/(4 x 0,004) = 3,050 m ». Le solveur, qui ne
        connait pas cette formule, declenche son evenement de blocage a
        **x = 3,0500 m**. Deux chemins independants, le meme nombre a 1e-4.
        """
        maximum = fanno.max_length(2.0, DIAMETRE, CF)
        self.assertAlmostEqual(maximum, 3.050, delta=1e-3)
        with self.assertRaises(ChokedFlow) as capture:
            integrate_duct(tube(6.5), no_heat(300.0, 6.5), 2.0, 5.0,
                           friction=Friction(CF), n_points=100)
        self.assertAlmostEqual(capture.exception.x, maximum, delta=1e-3)

    def test_il_RAPPROCHE_du_blocage_et_l_effet_va_avec_la_LONGUEUR(self):
        """LE RESULTAT DEMANDE PAR LE PLAN, MESURE PLUTOT QU'AFFIRME.

        La T04 maximale que la chambre laisse passer, a M d'entree 0,20 :

            longueur    sans frottement    avec C_f = 0,005
            0,6 m       3 495 K            3 444 K   (-1,5 %)
            2,0 m       3 495 K            3 330 K   (-4,7 %)
            5,0 m       3 495 K            3 110 K   (-11,0 %)

        La carte de l'etape 7 se deplace donc, mais **peu pour cette
        chambre-ci** : 0,6 m ne fait que 3,2 diametres. C'est un effet de
        longueur, et il faut une chambre huit fois plus longue pour qu'il
        devienne notable.
        """
        cycle = run_cycle(FlightCase(), EngineCase())
        T02 = cycle.station("2").T0
        section = from_cycle(cycle, mass_flow=10.0).zone("chamber").area_start

        def T04_max(longueur: float, coefficient: float) -> float:
            conduit = tube(longueur, section)
            lo, hi = 700.0, 12000.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                try:
                    integrate_duct(conduit,
                                   HeatRelease(T02, mid, 0.0, longueur), 0.20,
                                   10.0, friction=Friction(coefficient),
                                   n_points=20)
                    lo = mid
                except (ChokedFlow, CaseError):
                    hi = mid
            return 0.5 * (lo + hi)

        sans = T04_max(0.6, 0.0)
        self.assertAlmostEqual(sans, 3495.3, delta=2.0)
        for longueur, attendu in ((0.6, -1.47), (2.0, -4.73), (5.0, -11.02)):
            with self.subTest(longueur=longueur):
                avec = T04_max(longueur, 0.005)
                self.assertAlmostEqual(100.0 * (avec / sans - 1.0), attendu,
                                       delta=0.3)

    def test_sans_frottement_la_limite_du_SOLVEUR_est_celle_de_RAYLEIGH(self):
        """Le controle croise qui valide la bissection ci-dessus : 3 495,3 K
        des deux cotes, l'un integre, l'autre en forme fermee."""
        from ramjet import rayleigh
        cycle = run_cycle(FlightCase(), EngineCase())
        T02 = cycle.station("2").T0
        self.assertAlmostEqual(
            float(rayleigh.critical_total_temperature(0.20, T02)), 3495.3,
            delta=1.0)

    def test_sur_CE_moteur_le_frottement_ne_change_presque_rien(self):
        """UN RESULTAT NEGATIF, ET IL FAUT LE DIRE AUSSI NETTEMENT.

        Au point fixe, a C_f = 0,005 — le haut de la fourchette realiste —
        « pi_b » perd 0,56 % et la poussee specifique **0,11 %**. Deux raisons
        se composent : la chambre est courte, et l'etape 9 avait deja mesure que
        le moteur est peu sensible a « pi_b » (3 % sur pi_b -> 0,2 % sur la
        poussee).
        """
        sans = converge()
        avec = converge(friction=Friction(0.005))
        self.assertAlmostEqual(
            100.0 * (avec.combustor_pressure_ratio
                     / sans.combustor_pressure_ratio - 1.0), -0.56, delta=0.05)
        self.assertAlmostEqual(
            100.0 * (avec.engine.specific_thrust
                     / sans.engine.specific_thrust - 1.0), -0.111, delta=0.02)

    def test_et_il_ne_peut_PAS_deplacer_le_debit_au_point_fixe(self):
        """UNE SIMPLIFICATION EXACTE, ET ELLE S'EST SIMPLIFIEE ENCORE.

        « from_cycle » dimensionne le col pour « 10 (1+f) » sous le « pi_b »
        suppose ; « choking_mass_flow » cherche le debit de SORTIE qui bloque ce
        col sous le « pi_b » subi. Au POINT FIXE les deux « pi_b » sont le meme
        nombre : il se simplifie, et il reste **mdot_entree = 10,000000**
        exactement — le debit meme pour lequel la geometrie a ete dessinee.

        Le frottement n'agissant que sur « pi_b », il ne peut pas y toucher.
        Verifie a 1e-10 pour C_f de 0 a 0,02.

        *Avant que l'EDO ne porte le carburant, cette identite s'ecrivait
        « mdot = 10 (1+f) » et laissait 0,94 % d'ecart au dessin ; c'etait le
        « (1+f) » pose a la main qui manquait d'un cote.*
        """
        for coefficient in (0.0, 0.005, 0.02):
            with self.subTest(cf=coefficient):
                point = converge(friction=Friction(coefficient))
                self.assertAlmostEqual(point.engine.mass_flow, 10.0,
                                       delta=1e-10)
                self.assertAlmostEqual(
                    float(point.engine.profile.mass_flow[-1]),
                    10.0 * (1.0 + point.engine.cycle.fuel_air_ratio),
                    delta=1e-10)


# -- les refus ---------------------------------------------------------------

class Refus(unittest.TestCase):

    def test_un_coefficient_negatif_est_refuse(self):
        with self.assertRaises(CaseError) as capture:
            Friction(-0.001)
        self.assertIn("qui pousse", str(capture.exception))

    def test_un_conduit_plus_long_que_L_max_est_refuse(self):
        maximum = fanno.max_length(0.30, DIAMETRE, CF)
        with self.assertRaises(CaseError) as capture:
            fanno.mach_after(0.30, maximum * 1.05, DIAMETRE, CF)
        message = str(capture.exception)
        self.assertIn("blocage par frottement", message)
        self.assertIn("1.05 fois trop", message)
        self.assertIn(f"{maximum:.4f}", message)

    def test_le_message_du_SOLVEUR_nomme_la_cause_et_ne_la_suppose_pas(self):
        """UN DEFAUT QUE LE FROTTEMENT A REVELE, ET QU'IL A FALLU CORRIGER.

        Tant que la chaleur etait le seul terme source, annoncer « blocage
        thermique » etait juste par construction. Des que le frottement existe,
        un conduit SANS LA MOINDRE CHALEUR peut bloquer — et le message
        annoncait alors « la chaleur demandee porterait T0 d'un facteur 1,0000
        [...] soit 0,79 fois trop », a la fois faux et incomprehensible. Le
        diagnostic mesure desormais les deux budgets separement.
        """
        # Frottement seul : pas un mot sur la chaleur.
        with self.assertRaises(ChokedFlow) as capture:
            integrate_duct(tube(6.5), no_heat(300.0, 6.5), 2.0, 5.0,
                           friction=Friction(CF))
        message = str(capture.exception)
        self.assertIn("Frottement", message)
        self.assertNotIn("Chaleur", message)
        self.assertIn("raccourcir", message)

        # Chaleur seule : pas un mot sur le frottement.
        conduit = tube(1.0)
        with self.assertRaises(ChokedFlow) as capture:
            integrate_duct(conduit, HeatRelease(300.0, 1200.0, 0.0, 1.0), 0.30,
                           5.0)
        message = str(capture.exception)
        self.assertIn("Chaleur", message)
        self.assertNotIn("Frottement", message)

    def test_quand_les_DEUX_agissent_aucun_budget_seul_ne_suffit(self):
        """LE CAS QUI JUSTIFIE LE DIAGNOSTIC SEPARE.

        Mesure : 92 % du budget de chaleur et 7 % du budget de frottement — et
        l'ecoulement bloque. Ni l'un ni l'autre pris seul ne l'aurait annonce,
        et c'est precisement ce que le message dit.
        """
        conduit = tube(3.0)
        with self.assertRaises(ChokedFlow) as capture:
            integrate_duct(conduit, HeatRelease(300.0, 800.0, 0.0, 3.0), 0.30,
                           5.0, friction=Friction(0.005))
        message = str(capture.exception)
        self.assertIn("92%", message)
        self.assertIn("7%", message)
        self.assertIn("S'ADDITIONNENT", message)
        # Sans frottement, le meme depot de chaleur passe.
        profil = integrate_duct(conduit, HeatRelease(300.0, 800.0, 0.0, 3.0),
                                0.30, 5.0)
        self.assertLess(float(profil.mach[-1]), 1.0)

    def test_juste_sous_L_max_le_conduit_passe_encore(self):
        maximum = fanno.max_length(0.30, DIAMETRE, CF)
        mach = fanno.mach_after(0.30, maximum * 0.999, DIAMETRE, CF)
        self.assertLess(mach, 1.0)
        self.assertGreater(mach, 0.9)

    def test_la_branche_supersonique_refuse_au_dela_de_sa_limite(self):
        with self.assertRaises(CaseError) as capture:
            fanno.mach_from_friction_length(0.9, "supersonic")
        self.assertIn("limite supersonique", str(capture.exception))

    def test_branche_inconnue_et_longueur_negative(self):
        with self.assertRaises(CaseError):
            fanno.mach_from_friction_length(0.1, "transsonique")
        with self.assertRaises(CaseError):
            fanno.mach_from_friction_length(-0.1, "subsonic")

    def test_diametre_ou_coefficient_non_positif(self):
        with self.assertRaises(CaseError):
            fanno.max_length(0.3, 0.0, CF)
        with self.assertRaises(CaseError):
            fanno.max_length(0.3, DIAMETRE, 0.0)


if __name__ == "__main__":
    unittest.main()
