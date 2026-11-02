"""L'atmosphere standard : contre les valeurs publiees, et aux jonctions.

DEUX FAMILLES DE TESTS, ET ELLES NE DISENT PAS LA MEME CHOSE. Les valeurs
publiees verifient que le modele est le BON — les couches, les gradients, la
gravite. La continuite aux jonctions verifie qu'il est bien PROGRAMME : c'est
la que se logent les erreurs d'indice, et une discontinuite de pression de
0,1 % a 11 km ne se verrait sur aucune courbe.
"""

import unittest

from ramjet.atmosphere import (G0, LAYERS, R_ISA, TOP_ALTITUDE,
                               AtmosphereError, _base_pressure,
                               _pressure_within, density, pressure,
                               speed_of_sound, state, temperature)

# Valeurs de l'atmosphere standard 1976, telles que publiees.
PUBLIEES = (
    (0.0, 288.150, 101_325.0),
    (5_000.0, 255.650, 54_019.9),
    (11_000.0, 216.650, 22_632.1),
    (15_000.0, 216.650, 12_044.6),
    (20_000.0, 216.650, 5_474.89),
    (25_000.0, 221.650, 2_511.02),
    (32_000.0, 228.650, 868.019),
)


class ValeursPubliees(unittest.TestCase):
    """Mesure : ecart maximal 4e-6 en relatif, la ou le plan demandait 1e-3."""

    def test_les_temperatures(self):
        for altitude, T, _ in PUBLIEES:
            with self.subTest(altitude=altitude):
                self.assertAlmostEqual(temperature(altitude), T, delta=1e-3)

    def test_les_pressions(self):
        for altitude, _, p in PUBLIEES:
            with self.subTest(altitude=altitude):
                self.assertAlmostEqual(pressure(altitude) / p, 1.0, delta=1e-4)

    def test_le_cas_de_reference_du_projet(self):
        """11 km : le point de vol de tous les cas de reference du plan."""
        etat = state(11_000.0)
        self.assertAlmostEqual(etat["T"], 216.65, delta=1e-6)
        self.assertAlmostEqual(etat["p"], 22_632.0, delta=1.0)
        self.assertAlmostEqual(etat["rho"], 0.36392, delta=1e-5)
        self.assertAlmostEqual(etat["a"], 295.07, delta=0.01)
        self.assertIn("tropopause", etat["layer"])


class Continuite(unittest.TestCase):
    """Aux jonctions de couches, rien ne doit sauter."""

    def test_la_temperature_est_continue(self):
        for couche in LAYERS[1:]:
            with self.subTest(altitude=couche.base_altitude):
                z = couche.base_altitude
                self.assertAlmostEqual(temperature(z - 1e-6), temperature(z + 1e-6),
                                       delta=1e-8)

    def test_la_pression_est_continue_a_LA_VARIATION_PHYSIQUE_PRES(self):
        """Et c'est le test qui a d'abord ete ecrit faux.

        Premier jet : « p(z-eps)/p(z+eps) = 1 a 1e-12 pres ». Il echouait a
        3,15e-10 — non pas parce que le modele saute, mais parce que la
        pression VARIE entre les deux points. Le gradient hydrostatique vaut
        « dp/p = -g dz/(R T) », soit 1,577e-4 par metre a 216,65 K : sur les
        2 microns d'echantillonnage, exactement 3,15e-10.

        Le test compare donc l'ecart mesure a l'ecart ATTENDU. Il verifie ainsi
        deux choses d'un coup : qu'il n'y a pas de saut, et que le gradient
        local est le bon.
        """
        delta = 1e-6
        for couche in LAYERS[1:]:
            with self.subTest(altitude=couche.base_altitude):
                z = couche.base_altitude
                mesure = pressure(z - delta) / pressure(z + delta) - 1.0
                attendu = 2.0 * delta * G0 / (R_ISA * temperature(z))
                self.assertAlmostEqual(mesure / attendu, 1.0, delta=1e-3)

    def test_les_deux_couches_donnent_la_MEME_pression_a_la_jonction(self):
        """La continuite au sens strict, sans echantillonnage : la formule de
        la couche du dessous, evaluee a la jonction, doit rendre exactement la
        pression de base de celle du dessus."""
        for basse, haute in zip(LAYERS, LAYERS[1:]):
            with self.subTest(jonction=haute.base_altitude):
                depuis_le_bas = _pressure_within(
                    basse, _base_pressure(basse), haute.base_altitude)
                depuis_le_haut = _base_pressure(haute)
                self.assertAlmostEqual(depuis_le_bas / depuis_le_haut, 1.0,
                                       delta=1e-15)

    def test_la_pression_a_la_base_d_une_couche_est_celle_de_sa_declaration(self):
        """Les pressions de base sont calculees de proche en proche et non
        tabulees : une table serait une seconde source de verite, et c'est la
        que naissent les incoherences de sixieme decimale."""
        for couche in LAYERS[1:]:
            with self.subTest(couche=couche.name):
                self.assertAlmostEqual(
                    temperature(couche.base_altitude), couche.base_temperature,
                    delta=1e-9)


class Monotonies(unittest.TestCase):
    """Ce que l'atmosphere fait, et qui ne demande aucune valeur tabulee."""

    def test_la_pression_et_la_densite_decroissent_partout(self):
        altitudes = [0.0, 2_000.0, 5_000.0, 11_000.0, 15_000.0, 20_000.0,
                     26_000.0, 32_000.0]
        pressions = [pressure(z) for z in altitudes]
        densites = [density(z) for z in altitudes]
        for avant, apres in zip(pressions, pressions[1:]):
            self.assertGreater(avant, apres)
        for avant, apres in zip(densites, densites[1:]):
            self.assertGreater(avant, apres)

    def test_la_temperature_descend_puis_stagne_puis_remonte(self):
        """Les trois couches du modele, et c'est ce qui interdit l'exponentielle
        unique : une atmosphere isotherme donnerait 5 % d'erreur a 20 km."""
        self.assertGreater(temperature(0.0), temperature(10_000.0))
        self.assertAlmostEqual(temperature(11_000.0), temperature(19_000.0),
                               delta=1e-9)
        self.assertGreater(temperature(30_000.0), temperature(20_000.0))

    def test_la_vitesse_du_son_suit_la_temperature(self):
        self.assertGreater(speed_of_sound(0.0), speed_of_sound(11_000.0))
        self.assertGreater(speed_of_sound(32_000.0), speed_of_sound(20_000.0))


class LoiDesGazParfaits(unittest.TestCase):
    def test_la_densite_decoule_de_p_et_T(self):
        """« rho = p/(R T) » : la densite n'est pas une quatrieme donnee
        independante, et un test le verifie plutot qu'une docstring."""
        for altitude, _, _ in PUBLIEES:
            with self.subTest(altitude=altitude):
                attendue = pressure(altitude) / (R_ISA * temperature(altitude))
                self.assertAlmostEqual(density(altitude), attendue, delta=1e-15)


class Domaine(unittest.TestCase):
    def test_hors_du_modele_c_est_un_refus_et_pas_une_extrapolation(self):
        for altitude in (-1.0, TOP_ALTITUDE + 1.0, 80_000.0):
            with self.subTest(altitude=altitude):
                with self.assertRaises(AtmosphereError) as leve:
                    temperature(altitude)
                self.assertIn(f"{TOP_ALTITUDE:g}", str(leve.exception))

    def test_les_bornes_elles_memes_sont_acceptees(self):
        for altitude in (0.0, TOP_ALTITUDE):
            with self.subTest(altitude=altitude):
                self.assertGreater(pressure(altitude), 0.0)

    def test_le_plafond_du_modele_est_celui_de_la_configuration(self):
        """Les deux constantes doivent rester d'accord : « config » refuse a
        32 km parce que « atmosphere » s'y arrete, et pas l'inverse."""
        from ramjet.config import MAX_ALTITUDE

        self.assertEqual(MAX_ALTITUDE, TOP_ALTITUDE)


if __name__ == "__main__":
    unittest.main()
