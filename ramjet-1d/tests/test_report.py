"""Le rapport : ce qu'il dit, et surtout DANS QUEL ORDRE.

L'ORDRE EST LA SEULE CHOSE QUI COMPTE VRAIMENT ICI. Un rapport qui annonce
738 N.s/kg en tete et relegue en annexe le fait qu'une identite ne tombe pas
ment par ordonnancement, sans ecrire un seul nombre faux. Les tests verifient
donc que les ecarts aux identites precedent tout resultat, et qu'un lecteur qui
s'arrete apres dix lignes sait deja s'il peut croire la suite.

LE SECOND POINT : LES NOMBRES DOIVENT ETRE ANALYSABLES. Le plan demande que
« grep » sur une ligne d'ecart rende un nombre. Un rapport qu'on ne peut pas
depouiller automatiquement est un rapport qu'on ne relira pas.
"""

import re
import tempfile
import unittest
from pathlib import Path

from ramjet.config import EngineCase, FlightCase
from ramjet.coupling import converge
from ramjet.cycle0d import sweep_mach
from ramjet.report import (Identity, case_slug, identities, report,
                           sweep_table, write)

NOMBRE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# Ce qu'un « grep » attraperait sur une ligne d'ecart : la notation
# scientifique, et elle seule. Chercher « le premier nombre de la ligne »
# ramassait le « 0 » de « 0D » dans « boucle 0D <-> 1D ».
SCIENTIFIQUE = re.compile(r"\d(?:\.\d+)?e[-+]\d+")


class RapportDeReference(unittest.TestCase):
    """Converge une fois : 1,7 s, partage par toute la classe."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.point = converge()
        cls.texte = report(cls.point)
        cls.lignes = cls.texte.splitlines()


class LOrdreDesChoses(RapportDeReference):

    def test_les_identites_viennent_AVANT_le_moindre_resultat(self):
        """LE TEST QUI PORTE TOUTE L'INTENTION DU FICHIER.

        « CE QU'ON PEUT CROIRE » doit apparaitre avant « LA PERFORMANCE », et
        avant meme le cycle. Sinon le rapport ment par ordonnancement.
        """
        position = {titre: self.texte.index(titre)
                    for titre in ("CE QU'ON PEUT CROIRE", "LE CYCLE 0D",
                                  "LA PERFORMANCE")}
        self.assertLess(position["CE QU'ON PEUT CROIRE"],
                        position["LE CYCLE 0D"])
        self.assertLess(position["LE CYCLE 0D"], position["LA PERFORMANCE"])

    def test_un_lecteur_presse_sait_avant_la_premiere_station(self):
        """Les cinq identites ET leur verdict tiennent dans les treize
        premieres lignes, donc avant la premiere station du cycle. Le test
        compare aux positions reelles plutot qu'a un compte fixe, pour ne pas
        casser au premier mot ajoute a l'en-tete."""
        verdict = next(i for i, l in enumerate(self.lignes)
                       if "Les cinq identites tombent" in l)
        station = next(i for i, l in enumerate(self.lignes)
                       if "station 0" in l)
        self.assertLess(verdict, station)
        self.assertLessEqual(verdict, 13)

    def test_la_validation_est_NOMMEE_comme_telle(self):
        """« accord d'ORDRE DE GRANDEUR, pas une identite ». Melanger les deux
        est l'erreur que tout ce projet cherche a eviter."""
        balayage = sweep_mach([2.0, 3.0], EngineCase())
        avec = report(self.point, sweep=balayage)
        self.assertIn("accord d'ORDRE DE GRANDEUR, pas une identite", avec)
        self.assertGreater(avec.index("ORDRE DE GRANDEUR"),
                           avec.index("CE QU'ON PEUT CROIRE"))


class LesIdentites(RapportDeReference):

    def test_il_y_en_a_cinq_et_elles_tombent_toutes(self):
        controles = identities(self.point)
        self.assertEqual(len(controles), 5)
        for identite in controles:
            with self.subTest(identite=identite.name):
                self.assertIsInstance(identite, Identity)
                self.assertTrue(identite.ok,
                                f"{identite.name} : {identite.value:.2e} "
                                f"au-dessus de {identite.threshold:.0e}")

    def test_chacune_nomme_l_etape_qui_l_a_mesuree(self):
        """Un seuil sans provenance est un seuil qu'on n'osera pas toucher."""
        for identite in identities(self.point):
            with self.subTest(identite=identite.name):
                self.assertRegex(identite.source, r"etape \d+")

    def test_les_valeurs_mesurees(self):
        """3,3e-16 / 1,5e-11 / 4,4e-16 / 1,3e-15 / 8,6e-13 — et chacune a au
        moins deux ordres de grandeur de marge sous son seuil."""
        for identite in identities(self.point):
            with self.subTest(identite=identite.name):
                self.assertLess(identite.value, identite.threshold / 100.0)

    def test_un_ecart_au_dessus_du_seuil_est_annonce_en_TETE(self):
        """Le rapport doit basculer son verdict, pas seulement une ligne."""
        faux = Identity("essai", 1.0, 1e-12, "etape 0")
        self.assertFalse(faux.ok)
        self.assertIn("ECHEC", faux.line())


class LesNombresSontAnalysables(RapportDeReference):

    def test_chaque_ligne_d_identite_rend_un_nombre(self):
        """Le critere du plan : « grep » sur une ligne d'ecart doit rendre un
        nombre exploitable, pas une phrase."""
        lignes = [l for l in self.lignes if "seuil" in l]
        self.assertEqual(len(lignes), 5)
        for ligne in lignes:
            with self.subTest(ligne=ligne.strip()[:50]):
                nombres = SCIENTIFIQUE.findall(ligne)
                # L'ecart mesure, puis le seuil : deux nombres, dans cet ordre.
                self.assertEqual(len(nombres), 2)
                self.assertGreater(float(nombres[0]), 0.0)
                self.assertLess(float(nombres[0]), float(nombres[1]))

    def test_les_lignes_de_performance_aussi(self):
        for etiquette in ("poussee specifique", "rendement thermique",
                          "impulsion specifique"):
            with self.subTest(grandeur=etiquette):
                ligne = next(l for l in self.lignes if etiquette in l)
                self.assertTrue(NOMBRE.findall(ligne))

    def test_le_rapport_est_aligne(self):
        """Aucune ligne ne depasse une largeur lisible en terminal."""
        for ligne in self.lignes:
            self.assertLessEqual(len(ligne), 110, ligne)


class LeContenu(RapportDeReference):

    def test_les_quatre_stations_y_sont(self):
        for nom in ("station 0", "station 2", "station 4", "station 9"):
            self.assertIn(nom, self.texte)

    def test_les_deux_Mach_de_chambre_sont_donnes(self):
        """LE RESULTAT DE L'ETAPE 9 DOIT ETRE VISIBLE DANS LE RAPPORT.

        Le Mach vise et le Mach reel sont deux nombres differents — 0,20 et
        0,2074 — et c'est le second qui fixe la limite de blocage. N'afficher
        que le premier ferait chercher la marge au mauvais endroit.
        """
        ligne = next(l for l in self.lignes if "Mach de chambre" in l)
        self.assertIn("vise", ligne)
        self.assertIn("reel", ligne)
        self.assertIn("c'est le col qui le fixe", ligne)

    def test_la_marge_avant_blocage_est_chiffree(self):
        ligne = next(l for l in self.lignes if "T04 maximale" in l)
        self.assertIn("marge", ligne)
        self.assertIn("1495 K", ligne)

    def test_le_chemin_de_la_boucle_est_donne_en_entier(self):
        ligne = next(l for l in self.lignes if "chemin" in l)
        self.assertEqual(ligne.count("->"), len(self.point.history) - 1)


class LeBalayage(unittest.TestCase):

    def test_la_table_a_une_ligne_par_Mach(self):
        balayage = sweep_mach([1.5, 2.0, 2.5, 3.0], EngineCase())
        lignes = sweep_table(balayage)
        donnees = [l for l in lignes if l.strip().startswith(("1.", "2.", "3."))]
        self.assertEqual(len(donnees), 4)

    def test_elle_annonce_les_fourchettes_ET_leur_domaine(self):
        """« sur M0 de 2 a 3,5 » : sans le domaine, la fourchette est fausse —
        c'est la correction de l'etape 4."""
        lignes = sweep_table(sweep_mach([2.0, 3.0], EngineCase()))
        entete = "\n".join(lignes[:2])
        self.assertIn("600-900", entete)
        self.assertIn("M0 de 2 a 3,5", entete)

    def test_le_maximum_est_annonce(self):
        lignes = sweep_table(sweep_mach([2.0, 2.5, 3.0], EngineCase()))
        self.assertTrue(any("maximum de poussee" in l for l in lignes))


class LEcriture(unittest.TestCase):

    def test_le_dossier_est_cree_et_le_fichier_relisible(self):
        with tempfile.TemporaryDirectory() as dossier:
            cible = Path(dossier) / "cas" / "rapport.txt"
            rendu = write("essai\naccentue : rien\n", cible)
            self.assertTrue(rendu.exists())
            self.assertEqual(rendu.read_text(encoding="utf-8"),
                             "essai\naccentue : rien\n")

    def test_le_nom_de_cas_est_lisible_et_stable(self):
        slug = case_slug(FlightCase(), EngineCase())
        self.assertEqual(slug, "M3_z11000_T2000_kerosene")
        self.assertEqual(slug, case_slug(FlightCase(), EngineCase()))
        autre = case_slug(FlightCase(mach=2.5), EngineCase(T04=2200.0))
        self.assertEqual(autre, "M2.5_z11000_T2200_kerosene")

    def test_deux_cas_differents_ne_partagent_pas_de_dossier(self):
        self.assertNotEqual(case_slug(FlightCase(), EngineCase()),
                            case_slug(FlightCase(mach=2.0), EngineCase()))


if __name__ == "__main__":
    unittest.main()
