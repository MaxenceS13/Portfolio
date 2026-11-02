"""Le solveur 1D, du plus exact au moins exact.

TROIS FAMILLES, ET ELLES NE SE VALENT PAS. Les deux premieres comparent
l'integration a des formes fermees — Rayleigh et l'isentropique — qui sont
exactes : c'est de la verification au sens strict, et elle tombe a 1e-12. La
troisieme est le controle de QUANTITE DE MOUVEMENT, qui ne compare a rien
d'exterieur mais verifie une identite que l'integration ne force pas. La
quatrieme famille est faite de cas triviaux, ou le solveur ne doit rien faire
du tout — et c'est souvent la que casse un terme source qui fuit.
"""

import unittest

import numpy as np
from scipy.optimize import brentq

from ramjet.config import CaseError
from ramjet.duct import Duct, Zone
from ramjet.solver1d import (ATOL, RTOL, HeatRelease, dmach2_dx, integrate_duct,
                             momentum_residual, no_heat, reconstruct)
from ramjet.thermo import area_ratio, mach_from_area_ratio

GAMMA, R = 1.4, 287.05


def rayleigh_T0(mach: float, g: float = GAMMA) -> float:
    return (((g + 1) * mach ** 2 * (2 + (g - 1) * mach ** 2))
            / ((1 + g * mach ** 2) ** 2))


def rayleigh_p0(mach: float, g: float = GAMMA) -> float:
    return (((1 + g) / (1 + g * mach ** 2))
            * ((2 + (g - 1) * mach ** 2) / (g + 1)) ** (g / (g - 1)))


def tube(area: float = 0.02, length: float = 1.0) -> Duct:
    """Un conduit a section constante : le domaine de Rayleigh."""
    return Duct((Zone("chambre", 0.0, length, area, area),))


def nozzle(a_in: float = 0.01, a_out: float = 0.025) -> Duct:
    return Duct((Zone("divergent", 0.0, 1.0, a_in, a_out),))


class ContreRayleigh(unittest.TestCase):
    """CRITERE, VOLET 1 : section constante, chaleur repartie."""

    M_IN, T0_IN = 0.3, 300.0
    RATIOS = (1.5, 2.0, 2.5, 2.87)

    def _integre(self, ratio: float):
        heat = HeatRelease(self.T0_IN, self.T0_IN * ratio, 0.0, 1.0)
        return integrate_duct(tube(), heat, self.M_IN, mass_flow=5.0,
                              gamma=GAMMA, R=R)

    def _exact(self, ratio: float) -> float:
        cible = ratio * rayleigh_T0(self.M_IN)
        return brentq(lambda m: rayleigh_T0(m) - cible, 1e-9, 1.0)

    def test_le_mach_de_sortie_tombe_sur_la_forme_fermee(self):
        """Mesure : 2,7e-12 a T0r = 1,5 et 3,8e-11 a 2,87, pour 1e-10 demande."""
        for ratio in self.RATIOS:
            with self.subTest(ratio=ratio):
                mesure = float(self._integre(ratio).mach[-1])
                self.assertAlmostEqual(mesure, self._exact(ratio), delta=1e-10)

    def test_l_ecart_CROIT_quand_on_approche_du_blocage(self):
        """Et ce n'est pas du bruit : c'est la singularite qui se rapproche.

        Mesure : 2,7e-12 / 7,7e-12 / 1,8e-11 / 3,8e-11 pour T0r = 1,5 / 2 /
        2,5 / 2,87 — le dernier etant a 99,5 % du rapport de blocage. Masquer
        cette croissance sous une tolerance unique serait perdre l'information
        la plus utile du test.
        """
        ecarts = [abs(float(self._integre(r).mach[-1]) - self._exact(r))
                  for r in self.RATIOS]
        for petit, grand in zip(ecarts, ecarts[1:]):
            self.assertGreater(grand, petit)
        self.assertGreater(ecarts[-1] / ecarts[0], 5.0)

    def test_la_perte_de_pression_totale_tombe_aussi(self):
        """C'est le « pi_b » que le 0D suppose et que le 1D calcule : premiere
        apparition du nombre autour duquel tourne l'etape 9. Mesure : accord a
        1e-11 avec la forme fermee."""
        for ratio in self.RATIOS:
            with self.subTest(ratio=ratio):
                mesure = self._integre(ratio).total_pressure_ratio
                exact = rayleigh_p0(self._exact(ratio)) / rayleigh_p0(self.M_IN)
                self.assertAlmostEqual(mesure, exact, delta=1e-10)

    def test_l_impulsion_est_CONSERVEE_a_section_constante(self):
        """Propriete de la droite de Rayleigh : « p + rho u^2 » est constante,
        donc « F = pA(1 + gamma M^2) » l'est aussi quand A ne bouge pas. Rien
        dans le solveur ne l'impose."""
        profil = self._integre(2.5)
        impulsion = profil.impulse
        self.assertLess(float(np.ptp(impulsion) / np.mean(impulsion)), 1e-10)

    def test_le_mach_MONTE_quand_on_chauffe_un_ecoulement_subsonique(self):
        profil = self._integre(2.5)
        self.assertTrue(np.all(np.diff(profil.mach) > 0.0))
        self.assertGreater(float(profil.mach[-1]), self.M_IN)
        self.assertTrue(np.all(np.diff(profil.pressure) < 0.0))


class ContreLIsentropique(unittest.TestCase):
    """CRITERE, VOLET 2 : section variable, chaleur nulle, les deux branches."""

    def _integre(self, mach_in: float, duct: Duct):
        return integrate_duct(duct, no_heat(300.0, duct.length), mach_in,
                              mass_flow=1.0, gamma=GAMMA, R=R)

    def test_la_branche_subsonique(self):
        """Mesure : 2,0e-12."""
        duct = nozzle()
        profil = self._integre(0.3, duct)
        critique = 0.01 / float(area_ratio(0.3, GAMMA))
        exact = float(mach_from_area_ratio(0.025 / critique, "subsonic", GAMMA))
        self.assertAlmostEqual(float(profil.mach[-1]), exact, delta=1e-10)

    def test_la_branche_supersonique(self):
        """Mesure : 1,9e-11."""
        duct = nozzle()
        profil = self._integre(2.5, duct)
        critique = 0.01 / float(area_ratio(2.5, GAMMA))
        exact = float(mach_from_area_ratio(0.025 / critique, "supersonic", GAMMA))
        self.assertAlmostEqual(float(profil.mach[-1]), exact, delta=1e-9)

    def test_un_divergent_RALENTIT_le_subsonique_et_ACCELERE_le_supersonique(self):
        """L'inversion de comportement de part et d'autre de M = 1, qui est
        toute la raison d'etre d'une tuyere convergente-divergente."""
        duct = nozzle()
        self.assertLess(float(self._integre(0.3, duct).mach[-1]), 0.3)
        self.assertGreater(float(self._integre(2.5, duct).mach[-1]), 2.5)

    def test_la_pression_totale_est_CONSERVEE_sans_chaleur(self):
        """Sans apport ni frottement, l'ecoulement est isentropique : p0 ne doit
        pas bouger. Mesure a la tolerance de travail : 2,3e-09 en subsonique,
        4,6e-09 en supersonique."""
        for mach_in in (0.3, 2.5):
            with self.subTest(mach=mach_in):
                profil = self._integre(mach_in, nozzle())
                variation = float(np.ptp(profil.total_pressure)
                                  / np.mean(profil.total_pressure))
                self.assertLess(variation, 1e-7)

    def test_et_sa_derive_suit_la_TOLERANCE_de_l_EDO(self):
        """Le test qui distingue une erreur d'integration d'un defaut de
        physique — et le premier jet posait un seuil absolu de 1e-10 qui
        echouait sans dire pourquoi.

        La pression totale passe par « (1 + 0,2 M^2)^3,5 » : une erreur
        relative sur M s'y retrouve multipliee par cent a mille. Mesure de la
        derive sur la tuyere supersonique : 2,2e-07 a rtol = 1e-8, 4,6e-09 a
        1e-10, 4,4e-11 a 1e-12. Elle suit donc la tolerance a la
        proportionnelle, ce qui identifie sa cause sans ambiguite.
        """
        duct = nozzle()
        derives = []
        for rtol in (1e-8, 1e-10, 1e-12):
            profil = integrate_duct(duct, no_heat(300.0, 1.0), 2.5, 1.0,
                                    GAMMA, R, rtol=rtol, atol=rtol * 1e-2)
            derives.append(float(np.ptp(profil.total_pressure)
                                 / np.mean(profil.total_pressure)))
        for grossier, fin in zip(derives, derives[1:]):
            self.assertGreater(grossier / fin, 20.0)


class LaQuantiteDeMouvement(unittest.TestCase):
    """LE controle independant : rien dans l'integration ne le force.

    L'EDO n'integre que M^2 ; « p » est reconstruite par la conservation de la
    masse. L'identite « F(sortie) - F(entree) = integrale p dA » n'a donc servi
    nulle part, et elle doit pourtant tenir.
    """

    CONFIGURATIONS = (
        ("Rayleigh", tube(), HeatRelease(300.0, 750.0, 0.0, 1.0), 0.3),
        ("divergent subsonique", nozzle(), no_heat(300.0, 1.0), 0.3),
        ("divergent supersonique", nozzle(), no_heat(300.0, 1.0), 2.5),
        ("convergent supersonique", Duct((Zone("c", 0.0, 1.0, 0.025, 0.012),)),
         no_heat(300.0, 1.0), 2.5),
    )

    def test_l_identite_tient_sur_les_quatre_configurations(self):
        """Mesure : de 2,5e-13 a 2,0e-11, pour 1e-8 demande."""
        for nom, duct, heat, mach_in in self.CONFIGURATIONS:
            with self.subTest(configuration=nom):
                profil = integrate_duct(duct, heat, mach_in, 2.0, GAMMA, R,
                                        n_points=2001)
                self.assertLess(momentum_residual(profil, duct), 1e-8)

    def test_le_residu_ne_depend_PAS_du_nombre_de_points(self):
        """C'est ce qui prouve qu'on mesure la SOLUTION et non la quadrature.

        Premier jet : integrale par trapezes, residu 2,5e-07 a 2 000 points et
        1,0e-05 a 400 — soit exactement l'erreur des trapezes en (dx)^2, donc
        la quadrature qui se mesurait elle-meme. Avec Simpson, le residu de
        Rayleigh vaut 9,29e-12 aux deux resolutions.
        """
        duct, heat = tube(), HeatRelease(300.0, 750.0, 0.0, 1.0)
        residus = [momentum_residual(
            integrate_duct(duct, heat, 0.3, 2.0, GAMMA, R, n_points=n), duct)
            for n in (401, 2001)]
        self.assertAlmostEqual(residus[0] / residus[1], 1.0, delta=0.5)

    def test_il_est_normalise_par_l_impulsion_et_non_par_son_saut(self):
        """Autre erreur du premier jet. A section constante, les DEUX membres
        de l'identite valent zero — l'impulsion de Rayleigh est conservee —, et
        diviser par le saut rendait un residu de 1,00 pour un ecart de 1e-10 :
        un zero divise par zero, deguise en echec.
        """
        duct, heat = tube(), HeatRelease(300.0, 750.0, 0.0, 1.0)
        profil = integrate_duct(duct, heat, 0.3, 2.0, GAMMA, R)
        saut = float(profil.impulse[-1] - profil.impulse[0])
        self.assertLess(abs(saut) / float(np.mean(profil.impulse)), 1e-10)
        self.assertLess(momentum_residual(profil, duct), 1e-10)


class ConservationEtCasTriviaux(unittest.TestCase):
    """Ce que le solveur doit faire sans qu'on le lui demande."""

    def test_le_debit_est_conserve_a_la_precision_machine(self):
        """C'est un test de la RECONSTRUCTION, pas de l'EDO : « p » est deduite
        de « rho u A = mdot », donc l'egalite est vraie par construction — et
        c'est justement pour ca qu'elle attrape une erreur de reconstruction.
        Mesure : 8,9e-16."""
        profil = integrate_duct(tube(), HeatRelease(300.0, 750.0, 0.0, 1.0),
                                0.3, 5.0, GAMMA, R)
        self.assertLess(float(np.ptp(profil.mass_flux)) / 5.0, 1e-12)

    def test_sans_chaleur_ni_variation_de_section_RIEN_ne_bouge(self):
        """Test bete, et le premier a casser si un terme source fuit. Mesure :
        variation exactement nulle sur les cinq grandeurs."""
        profil = integrate_duct(tube(), no_heat(300.0, 1.0), 0.3, 5.0, GAMMA, R)
        for nom in ("mach", "temperature", "pressure", "velocity", "density"):
            with self.subTest(grandeur=nom):
                self.assertEqual(float(np.ptp(getattr(profil, nom))), 0.0)

    def test_deux_executions_identiques_donnent_les_memes_bits(self):
        arguments = (tube(), HeatRelease(300.0, 750.0, 0.0, 1.0), 0.3, 5.0,
                     GAMMA, R)
        a = integrate_duct(*arguments)
        b = integrate_duct(*arguments)
        np.testing.assert_array_equal(a.mach, b.mach)
        np.testing.assert_array_equal(a.pressure, b.pressure)

    def test_les_tolerances_restent_serrees(self):
        """Un solveur regle a 1e-6 ne dirait plus rien de son propre schema :
        le test protege les constantes contre une baisse distraite."""
        self.assertLessEqual(RTOL, 1e-9)
        self.assertLessEqual(ATOL, 1e-11)


class LeDepotDeChaleur(unittest.TestCase):
    """« HeatRelease » : un profil impose, nul en dehors de la zone."""

    def test_T0_est_constante_avant_et_apres_la_zone(self):
        heat = HeatRelease(300.0, 900.0, 0.4, 1.0)
        self.assertEqual(float(heat.total_temperature(0.0)), 300.0)
        self.assertEqual(float(heat.total_temperature(0.2)), 300.0)
        self.assertEqual(float(heat.total_temperature(1.0)), 900.0)
        self.assertEqual(float(heat.total_temperature(1.4)), 900.0)

    def test_la_derivee_est_nulle_hors_de_la_zone_et_constante_dedans(self):
        heat = HeatRelease(300.0, 900.0, 0.4, 1.0)
        self.assertEqual(float(heat.derivative(0.2)), 0.0)
        self.assertEqual(float(heat.derivative(1.2)), 0.0)
        self.assertAlmostEqual(float(heat.derivative(0.7)), 1000.0, delta=1e-9)

    def test_sans_apport_la_derivee_est_nulle_partout(self):
        heat = no_heat(500.0, 1.5)
        xs = np.linspace(0.0, 1.5, 50)
        np.testing.assert_array_equal(heat.derivative(xs), np.zeros(50))
        np.testing.assert_array_equal(heat.total_temperature(xs),
                                      np.full(50, 500.0))

    def test_une_zone_vide_ou_une_temperature_absurde_sont_refusees(self):
        with self.assertRaises(CaseError):
            HeatRelease(300.0, 900.0, 1.0, 1.0)
        with self.assertRaises(CaseError):
            HeatRelease(-1.0, 900.0, 0.0, 1.0)


class Refus(unittest.TestCase):
    """Ce que le solveur refuse d'integrer, et pourquoi."""

    def test_un_mach_d_entree_a_un_est_refuse_comme_singulier(self):
        """L'equation a un « 1 - M^2 » au denominateur : demarrer a M = 1
        demande la levee du 0/0, qui est le travail de l'etape 8."""
        with self.assertRaises(CaseError) as leve:
            integrate_duct(tube(), no_heat(300.0, 1.0), 1.0, 5.0, GAMMA, R)
        self.assertIn("singuliere", str(leve.exception))

    def test_un_mach_d_entree_non_positif_est_refuse(self):
        for mach in (0.0, -0.5):
            with self.subTest(mach=mach):
                with self.assertRaises(CaseError):
                    integrate_duct(tube(), no_heat(300.0, 1.0), mach, 5.0,
                                   GAMMA, R)

    def test_un_intervalle_vide_est_refuse(self):
        with self.assertRaises(CaseError):
            integrate_duct(tube(), no_heat(300.0, 1.0), 0.3, 5.0, GAMMA, R,
                           x_start=0.5, x_end=0.5)

    def test_la_reconstruction_refuse_un_mach_carre_negatif(self):
        with self.assertRaises(CaseError):
            reconstruct(np.array([0.0, 1.0]), np.array([0.09, -0.01]), tube(),
                        no_heat(300.0, 1.0), 5.0, GAMMA, R)


class LEquationElleMeme(unittest.TestCase):
    """Les deux coefficients d'influence, pris separement."""

    def test_sans_chaleur_ni_section_variable_la_derivee_est_nulle(self):
        self.assertEqual(dmach2_dx(0.5, 0.09, tube(), no_heat(300.0, 1.0)), 0.0)

    def test_le_signe_du_terme_de_section_s_inverse_en_M_egal_un(self):
        """En subsonique, un divergent ralentit ; en supersonique, il accelere.
        C'est le signe de « 1 - M^2 » qui le dit, et il est ecrit en clair."""
        duct, heat = nozzle(), no_heat(300.0, 1.0)
        self.assertLess(dmach2_dx(0.5, 0.09, duct, heat), 0.0)
        self.assertGreater(dmach2_dx(0.5, 6.25, duct, heat), 0.0)

    def test_le_signe_du_terme_de_chaleur_s_inverse_aussi(self):
        """Chauffer accelere un subsonique et RALENTIT un supersonique : les
        deux poussent le Mach vers 1, ce qui est la raison du blocage."""
        duct = tube()
        heat = HeatRelease(300.0, 900.0, 0.0, 1.0)
        self.assertGreater(dmach2_dx(0.5, 0.09, duct, heat), 0.0)
        self.assertLess(dmach2_dx(0.5, 6.25, duct, heat), 0.0)


if __name__ == "__main__":
    unittest.main()
