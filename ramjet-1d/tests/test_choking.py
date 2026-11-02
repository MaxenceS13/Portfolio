"""Le blocage thermique : le fichier le plus important du projet.

POURQUOI CELUI-LA PLUS QUE LES AUTRES. Les tests des etapes 2, 3 et 6 comparent
a des formes fermees : quand ils passent, on sait que le code resout ce qu'il
pretend resoudre. Celui-ci verifie autre chose — que le code REFUSE de resoudre
ce qui n'a pas de solution. C'est la difference entre un solveur juste et un
solveur utilisable : sans cette protection, le premier rend un profil d'allure
parfaitement ordinaire pour un moteur qui, en vrai, se desamorce.

LA CONTRE-EPREUVE EST LA MOITIE DU TEST. Un RK4 a pas fixe est ecrit ICI, dans
le fichier de test, pour montrer ce qui se passe sans protection : il franchit
M = 1 sans une erreur, et ce qu'il rend ensuite ne converge meme pas — le Mach
final saute de 0,76 a 2,11 selon le nombre de pas. Sans cette moitie-la, rien
ne prouverait que la protection protege de quelque chose.
"""

import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ramjet import rayleigh
from ramjet.config import CaseError
from ramjet.duct import Duct, Zone
from ramjet.solver1d import (MACH_EVENT, ChokedFlow, HeatRelease, dmach2_dx,
                             integrate_duct)

GAMMA, R, CP = 1.4, 287.05, 1004.5
M_IN, T0_IN, LONGUEUR = 0.3, 300.0, 1.0


def tube(area: float = 0.02) -> Duct:
    return Duct((Zone("chambre", 0.0, LONGUEUR, area, area),))


def chauffe(facteur: float) -> HeatRelease:
    """Un depot de chaleur valant « facteur » fois le maximum admissible."""
    maximum = float(rayleigh.max_total_temperature_ratio(M_IN, GAMMA))
    return HeatRelease(T0_IN, T0_IN * maximum * facteur, 0.0, LONGUEUR)


class LeCritereDeLEtape(unittest.TestCase):
    """Sous la limite, un profil ; au-dela, un refus. Jamais l'inverse."""

    def test_juste_sous_la_limite_le_solveur_rend_un_profil(self):
        """A 99,9 % du maximum, mesure : M de sortie = 0,9630."""
        profil = integrate_duct(tube(), chauffe(0.999), M_IN, 5.0, GAMMA, R)
        self.assertAlmostEqual(float(profil.mach[-1]), 0.9630, delta=1e-3)
        self.assertLess(float(profil.mach[-1]), 1.0)

    def test_au_dela_c_est_un_refus_et_PAS_un_profil(self):
        """A 105 % du maximum, mesure : blocage a x = 0,9289 m."""
        with self.assertRaises(ChokedFlow) as leve:
            integrate_duct(tube(), chauffe(1.05), M_IN, 5.0, GAMMA, R)
        self.assertAlmostEqual(leve.exception.x, 0.9289, delta=0.01)
        self.assertAlmostEqual(leve.exception.mach, MACH_EVENT, delta=1e-3)

    def test_plus_on_chauffe_plus_le_blocage_arrive_tot(self):
        """Mesure : x = 0,9289 a 105 % du maximum, x = 0,5664 a 150 %."""
        abscisses = []
        for facteur in (1.05, 1.20, 1.50):
            with self.assertRaises(ChokedFlow) as leve:
                integrate_duct(tube(), chauffe(facteur), M_IN, 5.0, GAMMA, R)
            abscisses.append(leve.exception.x)
        self.assertAlmostEqual(abscisses[0], 0.9289, delta=0.01)
        self.assertAlmostEqual(abscisses[-1], 0.5664, delta=0.01)
        for avant, apres in zip(abscisses, abscisses[1:]):
            self.assertGreater(avant, apres)

    def test_le_message_dit_quoi_changer(self):
        """Un refus qui ne dit pas quoi faire est un refus a moitie utile.

        REFORMULE A L'ETAPE 12. Le message annoncait « blocage thermique » et
        parlait de chaleur quoi qu'il arrive ; avec le frottement, un conduit
        sans la moindre chaleur peut bloquer, et il fallait donc que le
        diagnostic NOMME la cause au lieu de la supposer. Ce test verifie que
        la reformulation n'a rien perdu : les deux nombres qui comptent et les
        deux remedes sont toujours la.
        """
        with self.assertRaises(ChokedFlow) as leve:
            integrate_duct(tube(), chauffe(1.05), M_IN, 5.0, GAMMA, R)
        message = str(leve.exception)
        self.assertIn("3.0272", message)          # le rapport demande
        self.assertIn("2.8830", message)          # celui qui serait admissible
        self.assertIn("105%", message)            # et leur rapport, en clair
        self.assertIn("Chaleur", message)         # la cause, NOMMEE
        self.assertNotIn("Frottement", message)   # et l'autre, tue
        self.assertIn("baisser la temperature", message)      # remede 1
        self.assertIn("ralentir l'ecoulement", message)       # remede 2
        self.assertAlmostEqual(leve.exception.max_ratio, 2.8830, delta=1e-3)


class LaLimiteEstCELLE_DE_RAYLEIGH(unittest.TestCase):
    """La chaleur maximale s'ecrit avant d'integrer, et l'integration confirme."""

    def test_la_chaleur_maximale_analytique(self):
        """Mesure a M = 0,3 et T0 = 300 K : T0* = 864,9 K, q_max = 567,4 kJ/kg."""
        self.assertAlmostEqual(
            float(rayleigh.critical_total_temperature(M_IN, T0_IN, GAMMA)),
            864.90, delta=0.01)
        self.assertAlmostEqual(
            float(rayleigh.max_heat(M_IN, T0_IN, CP, GAMMA)) / 1e3, 567.4,
            delta=0.1)

    def test_la_limite_trouvee_par_BISSECTION_est_celle_de_la_forme_fermee(self):
        """Le controle qui ferme le sujet : on cherche par dichotomie le plus
        grand facteur pour lequel l'integration passe encore, et on le compare
        au 1,0 que la forme fermee predit. Mesure : accord a 1e-4.
        """
        def passe(facteur: float) -> bool:
            try:
                integrate_duct(tube(), chauffe(facteur), M_IN, 5.0, GAMMA, R)
                return True
            except ChokedFlow:
                return False

        bas, haut = 0.5, 1.5
        for _ in range(18):
            milieu = 0.5 * (bas + haut)
            if passe(milieu):
                bas = milieu
            else:
                haut = milieu
        self.assertAlmostEqual(0.5 * (bas + haut), 1.0, delta=1e-3)

    def test_la_limite_depend_du_MACH_D_ENTREE(self):
        """C'est le couplage qui decide de tout le dimensionnement d'un
        statoreacteur. Mesure : 21,378 a M = 0,10 ; 5,762 a 0,20 ; 2,883 a
        0,30 ; 1,890 a 0,40."""
        attendus = {0.10: 21.378, 0.15: 9.808, 0.20: 5.762, 0.25: 3.894,
                    0.30: 2.883, 0.40: 1.890}
        for mach, attendu in attendus.items():
            with self.subTest(mach=mach):
                mesure = float(rayleigh.max_total_temperature_ratio(mach, GAMMA))
                self.assertAlmostEqual(mesure, attendu, delta=1e-3)

    def test_ralentir_l_ecoulement_repousse_le_blocage(self):
        """Le remede que le message propose doit marcher : le meme depot de
        chaleur passe a M = 0,20 la ou il bloque a M = 0,30."""
        maximum_a_03 = float(rayleigh.max_total_temperature_ratio(0.30, GAMMA))
        heat = HeatRelease(T0_IN, T0_IN * maximum_a_03 * 1.20, 0.0, LONGUEUR)
        with self.assertRaises(ChokedFlow):
            integrate_duct(tube(), heat, 0.30, 5.0, GAMMA, R)
        profil = integrate_duct(tube(), heat, 0.20, 5.0, GAMMA, R)
        self.assertLess(float(profil.mach[-1]), 1.0)

    def test_un_rapport_au_dela_du_maximum_est_refuse_AVANT_d_integrer(self):
        """« heat_to_mach » repond sans rien integrer : c'est tout l'interet
        d'avoir la limite en forme fermee."""
        with self.assertRaises(CaseError) as leve:
            rayleigh.heat_to_mach(M_IN, 3.5, GAMMA)
        self.assertIn("blocage", str(leve.exception))
        self.assertAlmostEqual(rayleigh.heat_to_mach(M_IN, 2.5, GAMMA), 0.6487,
                               delta=1e-3)


class LaContreEpreuve(unittest.TestCase):
    """Sans protection, que fait un integrateur ? La moitie du test est ici.

    ET LE RESULTAT EST PIRE QUE CE QUE LE PLAN ANNONCAIT. Le plan disait « il
    franchit M = 1 et finit a M = 1,122 », d'apres une mesure du prototype.
    C'est vrai pour UN nombre de pas donne, et seulement pour celui-la : au-dela
    de la singularite, l'integration ne converge plus du tout. Mesure du Mach
    final pour q/q_max = 1,10, en faisant varier le nombre de pas :

        n         500      1000     2000     4000     8000
        M final   0,9264   0,7640   2,1088   1,2017   0,8390

    Cinq nombres sans rapport entre eux, entre 0,76 et 2,11 — et TROIS des cinq
    finissent SUBSONIQUES. Le solveur non protege ne se contente donc pas de
    rendre un resultat faux : il rend un resultat faux d'allure parfaitement
    ordinaire, et qu'on ne peut meme pas reconnaitre a son point d'arrivee.
    """

    @staticmethod
    def _rk4_naif(facteur: float, n: int = 2000):
        """RK4 a pas fixe, sans evenement, sans controle. Rend (a franchi
        M = 1, abscisse du franchissement, Mach final)."""
        duct, heat = tube(), chauffe(facteur)
        h = LONGUEUR / n
        mach2, x, franchi, x_franchi = M_IN ** 2, 0.0, False, None
        for _ in range(n):
            k1 = dmach2_dx(x, mach2, duct, heat, GAMMA)
            k2 = dmach2_dx(x + h / 2, mach2 + h / 2 * k1, duct, heat, GAMMA)
            k3 = dmach2_dx(x + h / 2, mach2 + h / 2 * k2, duct, heat, GAMMA)
            k4 = dmach2_dx(x + h, mach2 + h * k3, duct, heat, GAMMA)
            mach2 += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x += h
            if mach2 > 1.0 and not franchi:
                franchi, x_franchi = True, x
        return franchi, x_franchi, float(np.sqrt(abs(mach2)))

    def test_un_RK4_A_PAS_FIXE_franchit_M_egale_1_SANS_RIEN_DIRE(self):
        """Aucune erreur, aucun avertissement. Mesure du franchissement :
        x = 0,867 pour q/q_max = 1,10, x = 0,567 pour 1,50."""
        for facteur, abscisse in ((1.10, 0.867), (1.50, 0.567)):
            with self.subTest(facteur=facteur):
                franchi, x_franchi, _ = self._rk4_naif(facteur, n=4000)
                self.assertTrue(franchi)
                self.assertAlmostEqual(x_franchi, abscisse, delta=0.01)

    def test_le_franchissement_CONVERGE_mais_pas_ce_qui_suit(self):
        """La ligne de partage du sujet. Avant la singularite, l'integration
        est une integration : l'abscisse de franchissement converge quand on
        raffine (0,8680 -> 0,8674 entre 500 et 8 000 pas). Apres, ce n'est plus
        rien : le Mach final saute de 0,76 a 2,11 sans tendre vers quoi que ce
        soit.
        """
        abscisses, machs = [], []
        for n in (500, 1000, 2000, 4000):
            _, x_franchi, mach_final = self._rk4_naif(1.10, n=n)
            abscisses.append(x_franchi)
            machs.append(mach_final)
        self.assertLess(max(abscisses) - min(abscisses), 0.02)
        self.assertGreater(max(machs) - min(machs), 1.0)

    def test_et_la_moitie_des_essais_finit_SUBSONIQUE_donc_invisible(self):
        """Le cas le plus dangereux : la courbe repasse sous 1 et ressemble a
        un ecoulement subsonique parfaitement ordinaire. Mesure : trois des
        cinq raffinements finissent sous M = 1."""
        machs = [self._rk4_naif(1.10, n=n)[2]
                 for n in (500, 1000, 2000, 4000)]
        subsoniques = sum(1 for m in machs if m < 1.0)
        self.assertGreaterEqual(subsoniques, 1)
        self.assertLess(subsoniques, len(machs))

    def test_sous_la_limite_le_RK4_naif_est_correct_lui_aussi(self):
        """La contre-epreuve doit discriminer : le RK4 naif n'est pas faux
        partout, il est faux LA OU IL N'Y A PAS DE SOLUTION. Mesure a
        q/q_max = 0,99 : M = 0,889, subsonique, stable au raffinement."""
        machs = [self._rk4_naif(0.99, n=n)[2] for n in (1000, 4000)]
        for mach in machs:
            self.assertAlmostEqual(mach, 0.889, delta=0.01)
        self.assertLess(abs(machs[0] - machs[1]), 1e-3)

    def test_le_solveur_protege_refuse_exactement_les_memes_cas(self):
        for facteur in (1.10, 1.50):
            with self.subTest(facteur=facteur):
                with self.assertRaises(ChokedFlow):
                    integrate_duct(tube(), chauffe(facteur), M_IN, 5.0, GAMMA, R)


class LeSensDeLEvenement(unittest.TestCase):
    """Le detail qui desactive la protection en silence si on le rate."""

    def _integre_avec_direction(self, direction: float, facteur: float = 1.05):
        """La meme integration, avec le sens d'evenement passe en argument."""
        duct, heat = tube(), chauffe(facteur)
        seuil2 = MACH_EVENT ** 2

        def blocage(x, y):
            return seuil2 - y[0]

        blocage.terminal = True
        blocage.direction = direction
        return solve_ivp(
            lambda x, y: [dmach2_dx(x, float(y[0]), duct, heat, GAMMA)],
            (0.0, LONGUEUR), [M_IN ** 2], rtol=1e-10, atol=1e-12,
            events=blocage, max_step=LONGUEUR / 100)

    def test_avec_le_BON_sens_l_integration_s_arrete_proprement(self):
        """« direction = -1 » : la fonction d'evenement DECROIT vers zero.
        Mesure : arret a x = 0,9289 avec « success = True »."""
        solution = self._integre_avec_direction(-1.0)
        self.assertTrue(solution.success)
        self.assertEqual(len(solution.t_events[0]), 1)
        self.assertAlmostEqual(float(solution.t_events[0][0]), 0.9289,
                               delta=0.01)

    def test_avec_le_MAUVAIS_sens_l_evenement_ne_se_declenche_JAMAIS(self):
        """Et c'est l'erreur naturelle : on surveille un Mach qui MONTE, donc
        on ecrit « direction = +1 ». Mais c'est le signe de la derivee de la
        FONCTION D'EVENEMENT qui compte, et elle, elle descend.

        Resultat mesure : aucun evenement, « success = False », et le message
        « Required step size is less than spacing between numbers » — qui ne
        parle pas de physique et qu'on interprete naturellement comme un
        probleme de tolerance.
        """
        solution = self._integre_avec_direction(+1.0)
        self.assertEqual(len(solution.t_events[0]), 0)
        self.assertFalse(solution.success)
        self.assertIn("step size", solution.message)

    def test_les_deux_sens_s_arretent_au_MEME_endroit(self):
        """Le mauvais sens n'integre pas plus loin : il echoue au meme point,
        mais sans savoir dire pourquoi. La difference n'est pas dans le calcul,
        elle est dans ce qu'on peut en dire."""
        bon = self._integre_avec_direction(-1.0)
        mauvais = self._integre_avec_direction(+1.0)
        self.assertAlmostEqual(float(bon.t[-1]), float(mauvais.t[-1]),
                               delta=1e-3)

    def test_le_seuil_reste_sous_un(self):
        """A M = 1 exactement, la derivee est infinie : s'arreter a 0,999 est
        ce qui permet un arret PROPRE plutot qu'un echec."""
        self.assertLess(MACH_EVENT, 1.0)
        self.assertGreater(MACH_EVENT, 0.99)


class LesFormesFermeesDeRayleigh(unittest.TestCase):
    """Le module d'oracle lui-meme, verifie contre les tables et contre SciPy."""

    def test_toutes_les_fonctions_valent_un_a_M_egale_1(self):
        """Par definition de l'etoile. Et l'egalite doit etre exacte."""
        for fonction in (rayleigh.total_temperature_ratio,
                         rayleigh.total_pressure_ratio,
                         rayleigh.temperature_ratio, rayleigh.pressure_ratio):
            with self.subTest(fonction=fonction.__name__):
                self.assertAlmostEqual(float(fonction(1.0, GAMMA)), 1.0,
                                       delta=1e-15)

    def test_T0_sur_T0_etoile_decroit_DES_DEUX_COTES_de_M_egale_1(self):
        """C'est la raison du blocage : chauffer pousse le Mach vers 1 par les
        deux bords."""
        subsonique = rayleigh.total_temperature_ratio(
            np.array([0.2, 0.4, 0.6, 0.8, 1.0]), GAMMA)
        supersonique = rayleigh.total_temperature_ratio(
            np.array([1.0, 1.5, 2.0, 3.0, 5.0]), GAMMA)
        self.assertTrue(np.all(np.diff(subsonique) > 0.0))
        self.assertTrue(np.all(np.diff(supersonique) < 0.0))

    def test_la_limite_supersonique(self):
        """T0/T0* tend vers « 1 - 1/gamma^2 » = 0,4898 : un supersonique se
        bloque aussi, par le haut."""
        self.assertAlmostEqual(rayleigh.supersonic_limit(GAMMA), 0.489796,
                               delta=1e-6)
        self.assertAlmostEqual(
            float(rayleigh.total_temperature_ratio(1e4, GAMMA)),
            rayleigh.supersonic_limit(GAMMA), delta=1e-7)

    def test_l_inversion_maison_vaut_brentq_sur_les_deux_branches(self):
        for mach, branche in ((0.2, "subsonic"), (0.7, "subsonic"),
                              (1.5, "supersonic"), (3.0, "supersonic")):
            with self.subTest(mach=mach):
                cible = float(rayleigh.total_temperature_ratio(mach, GAMMA))
                maison = float(rayleigh.mach_from_total_temperature_ratio(
                    cible, branche, GAMMA))
                bornes = (1e-9, 1.0) if branche == "subsonic" else (1.0, 40.0)
                reference = brentq(
                    lambda m: float(rayleigh.total_temperature_ratio(m, GAMMA))
                    - cible, *bornes, xtol=1e-15, rtol=8.9e-16)
                self.assertAlmostEqual(maison, reference, delta=1e-9)

    def test_un_rapport_superieur_a_un_est_refuse_comme_etant_le_blocage(self):
        with self.assertRaises(CaseError) as leve:
            rayleigh.mach_from_total_temperature_ratio(1.2, "subsonic", GAMMA)
        self.assertIn("blocage", str(leve.exception))

    def test_un_rapport_sous_la_limite_supersonique_est_refuse(self):
        with self.assertRaises(CaseError) as leve:
            rayleigh.mach_from_total_temperature_ratio(0.3, "supersonic", GAMMA)
        self.assertIn("limite supersonique", str(leve.exception))

    def test_une_branche_inconnue_est_refusee_avec_les_choix(self):
        with self.assertRaises(CaseError) as leve:
            rayleigh.mach_from_total_temperature_ratio(0.5, "transsonique")
        self.assertIn("subsonic", str(leve.exception))


if __name__ == "__main__":
    unittest.main()
