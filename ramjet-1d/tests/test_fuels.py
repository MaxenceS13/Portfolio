"""Les carburants : deux nombres, un registre, et un garde-fou.

Le registre est teste comme celui des systemes de « chaos-attractors » : un
carburant enregistre DANS le fichier de test doit apparaitre sans qu'une ligne
du paquet ait change. C'est la preuve de l'extensibilite, pas son affirmation.
"""

import unittest
from dataclasses import FrozenInstanceError

from ramjet.fuels import REGISTRY, Fuel, FuelError, available, get, register


class Declaration(unittest.TestCase):
    def test_un_carburant_mal_declare_est_refuse(self):
        for label, kwargs in (
                ("sans nom", dict(name="", lhv=1e6, f_stoich=0.06)),
                ("PCI nul", dict(name="x", lhv=0.0, f_stoich=0.06)),
                ("PCI negatif", dict(name="x", lhv=-1e6, f_stoich=0.06)),
                ("dosage nul", dict(name="x", lhv=1e6, f_stoich=0.0)),
                ("dosage superieur a 1", dict(name="x", lhv=1e6, f_stoich=1.5)),
        ):
            with self.subTest(label):
                with self.assertRaises(FuelError):
                    Fuel(**kwargs)

    def test_le_libelle_prend_le_nom_par_defaut(self):
        self.assertEqual(Fuel("x", 1e6, 0.06).label, "x")
        self.assertEqual(Fuel("x", 1e6, 0.06, "un carburant").label,
                         "un carburant")

    def test_un_carburant_est_immuable(self):
        with self.assertRaises(FrozenInstanceError):
            get("kerosene").lhv = 1.0


class ValeursLivrees(unittest.TestCase):
    def test_le_kerosene_porte_la_valeur_du_plan(self):
        """42,8 MJ/kg est la valeur de Mattingly, celle des cas de reference."""
        kerosene = get("kerosene")
        self.assertAlmostEqual(kerosene.lhv, 42.8e6, delta=1.0)
        self.assertAlmostEqual(kerosene.f_stoich, 0.0679, delta=1e-4)

    def test_l_hydrogene_a_bien_trois_fois_plus_d_energie_par_kilogramme(self):
        self.assertGreater(get("hydrogen").lhv, 2.5 * get("kerosene").lhv)

    def test_tous_les_carburants_livres_sont_coherents(self):
        for nom in available():
            with self.subTest(carburant=nom):
                carburant = get(nom)
                self.assertGreater(carburant.lhv, 1e6)
                self.assertTrue(0.0 < carburant.f_stoich < 1.0)


class Richesse(unittest.TestCase):
    """Le garde-fou : un bilan d'energie ne sait pas qu'il y a une limite."""

    def test_la_richesse_vaut_un_au_dosage_stoechiometrique(self):
        kerosene = get("kerosene")
        self.assertAlmostEqual(kerosene.equivalence_ratio(kerosene.f_stoich),
                               1.0, delta=1e-12)

    def test_le_dosage_du_cas_de_reference_est_pauvre(self):
        """f = 0,0351 a M0 = 3 (mesure au prototype), soit une richesse de 0,52.
        Un statoreacteur de croisiere brule pauvre, et c'est ce qui laisse de la
        marge avant la limite du modele."""
        richesse = get("kerosene").equivalence_ratio(0.0351)
        self.assertAlmostEqual(richesse, 0.517, delta=0.01)
        self.assertLess(richesse, 1.0)


class Registre(unittest.TestCase):
    def test_un_carburant_enregistre_ici_apparait_sans_toucher_au_paquet(self):
        self.addCleanup(REGISTRY.pop, "essai", None)
        register(Fuel("essai", 30e6, 0.05))
        self.assertIn("essai", available())
        self.assertEqual(get("essai").lhv, 30e6)

    def test_deux_carburants_de_meme_nom_sont_refuses(self):
        with self.assertRaises(FuelError):
            register(get("kerosene"))

    def test_les_noms_sont_tries(self):
        self.assertEqual(available(), tuple(sorted(available())))

    def test_un_carburant_inconnu_est_refuse_et_les_choix_sont_donnes(self):
        with self.assertRaises(FuelError) as leve:
            get("eau")
        message = str(leve.exception)
        self.assertIn("eau", message)
        for nom in available():
            self.assertIn(nom, message)


if __name__ == "__main__":
    unittest.main()
