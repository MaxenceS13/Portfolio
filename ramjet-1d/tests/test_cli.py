"""Ligne de commande : options, codes de sortie, verifications de « --check »."""

import importlib.util
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from ramjet import cli
from ramjet.config import CaseError

NUMPY = importlib.util.find_spec("numpy") is not None


def quiet(argv):
    """Lance la ligne de commande sans polluer la sortie des tests."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue() + err.getvalue()


class Analyse(unittest.TestCase):
    def test_version_et_aide_sortent_en_zero(self):
        for option in ("--version", "--help"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit) as leve, \
                        redirect_stdout(io.StringIO()):
                    cli.build_parser().parse_args([option])
                self.assertEqual(leve.exception.code, 0)

    def test_la_version_affichee_est_celle_du_paquet(self):
        from ramjet import __version__

        out = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(out):
            cli.build_parser().parse_args(["--version"])
        self.assertIn(__version__, out.getvalue())

    def test_option_inconnue_refusee(self):
        with self.assertRaises(SystemExit) as leve, redirect_stderr(io.StringIO()):
            cli.build_parser().parse_args(["--statoreacteur-magique"])
        self.assertEqual(leve.exception.code, 2)

    def test_sans_mode_le_cas_par_defaut_est_calcule(self):
        """CHANGEMENT DE COMPORTEMENT A L'ETAPE 11, ET IL EST VOULU.

        Tant qu'il n'y avait rien a integrer, afficher l'aide etait la reponse
        la plus utile. Maintenant qu'il y a un moteur, « python3 main.py » doit
        le calculer : un outil qui ne fait rien par defaut se fait oublier.
        """
        code, texte = quiet(["--dry-run"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("CE QU'ON PEUT CROIRE", texte)
        self.assertIn("poussee specifique", texte)


class Verification(unittest.TestCase):
    @unittest.skipUnless(NUMPY, "NumPy absent")
    def test_check_sort_en_zero(self):
        code, texte = quiet(["--check"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("numpy", texte)

    @unittest.skipUnless(NUMPY, "NumPy absent")
    def test_check_verbeux_donne_le_detail(self):
        _, court = quiet(["--check"])
        _, long = quiet(["--check", "-v"])
        self.assertGreater(len(long), len(court))
        self.assertIn("carburants disponibles", long)

    def test_le_cloisonnement_est_verifie_dans_un_interpreteur_neuf(self):
        propre, detail = cli.core_imports_cleanly()
        self.assertTrue(propre, detail)

    def test_la_contrainte_sans_numpy_est_verifiee_aussi(self):
        propre, detail = cli.stdlib_only_holds()
        self.assertTrue(propre, detail)

    def test_une_dependance_indispensable_absente_fait_echouer(self):
        """NumPy manquant doit se voir tout de suite, pas trois etapes plus loin."""
        with mock.patch.object(cli, "DEPENDENCIES",
                               (("numpy_absent_pour_de_bon", True, "role"),)):
            with redirect_stdout(io.StringIO()):
                self.assertFalse(cli.check_dependencies())

    def test_une_dependance_optionnelle_absente_ne_fait_pas_echouer(self):
        """Le cloisonnement existe precisement pour ca : sans Matplotlib, la
        physique tourne encore."""
        with mock.patch.object(cli, "DEPENDENCIES",
                               (("matplotlib_absent_pour_de_bon", False, "role"),)):
            with redirect_stdout(io.StringIO()):
                self.assertTrue(cli.check_dependencies())

    def test_check_config_construit_et_refuse(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(cli.check_config(verbose=True))

    def test_check_config_echoue_si_un_cas_absurde_passe(self):
        """Contre-epreuve : un controle toujours vert ne prouverait rien.

        On remplace la validation de « FlightCase » par une qui accepte tout,
        et « check_config » doit s'en apercevoir.
        """
        class ToutAccepter(cli.FlightCase):
            def __post_init__(self):
                pass

        with mock.patch.object(cli, "FlightCase", ToutAccepter):
            with redirect_stdout(io.StringIO()):
                self.assertFalse(cli.check_config())


class CodesDeSortie(unittest.TestCase):
    """Les quatre codes du depot : 0 / 1 / 2 / 130."""

    def test_interruption_sort_en_130(self):
        with mock.patch.object(cli, "check", side_effect=KeyboardInterrupt):
            code, _ = quiet(["--check"])
        self.assertEqual(code, cli.EXIT_INTERRUPTED)

    def test_cas_invalide_sort_en_1_en_nommant_le_champ(self):
        with mock.patch.object(cli, "check",
                               side_effect=CaseError("mach doit etre positif")):
            code, texte = quiet(["--check"])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("mach", texte)

    def test_erreur_systeme_sort_en_1(self):
        with mock.patch.object(cli, "check", side_effect=OSError("disque plein")):
            code, texte = quiet(["--check"])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("disque plein", texte)

    def test_une_verification_ratee_sort_en_1(self):
        with mock.patch.object(cli, "check_output_dir", return_value=False):
            code, _ = quiet(["--check"])
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_les_codes_sont_distincts_et_conformes_au_depot(self):
        codes = (cli.EXIT_OK, cli.EXIT_ERROR, cli.EXIT_FAILED,
                 cli.EXIT_INTERRUPTED)
        self.assertEqual(len(set(codes)), 4)
        self.assertEqual(codes, (0, 1, 2, 130))


if __name__ == "__main__":
    unittest.main()


# -- l'etape 11 : les cas, les modes, les codes de sortie --------------------

class ConstructionDesCas(unittest.TestCase):
    """La ligne de commande ne doit pas devenir une seconde source de verite."""

    def test_sans_option_on_retombe_sur_les_defauts_de_config(self):
        from ramjet.config import EngineCase, FlightCase

        args = cli.build_parser().parse_args([])
        vol, moteur = cli.build_cases(args)
        self.assertEqual(vol, FlightCase())
        self.assertEqual(moteur, EngineCase())

    def test_chaque_option_atteint_le_bon_champ(self):
        args = cli.build_parser().parse_args(
            ["--mach", "2.5", "--altitude", "15000", "--t04", "2200",
             "--fuel", "hydrogen", "--m-comb", "0.25"])
        vol, moteur = cli.build_cases(args)
        self.assertEqual(vol.mach, 2.5)
        self.assertEqual(vol.altitude, 15000.0)
        self.assertEqual(moteur.T04, 2200.0)
        self.assertEqual(moteur.fuel, "hydrogen")
        self.assertEqual(moteur.M_combustor, 0.25)

    def test_un_cas_absurde_est_refuse_par_config_pas_par_le_CLI(self):
        """Les controles vivent dans les dataclasses, et une seule fois."""
        args = cli.build_parser().parse_args(["--mach", "-1"])
        with self.assertRaises(CaseError):
            cli.build_cases(args)


@unittest.skipUnless(NUMPY, "NumPy absent")
class ModesDeCalcul(unittest.TestCase):

    def test_le_cas_par_defaut_sort_en_zero_et_rapporte(self):
        code, texte = quiet(["--dry-run"])
        self.assertEqual(code, cli.EXIT_OK)
        for attendu in ("CE QU'ON PEUT CROIRE", "LE CYCLE 0D",
                        "LA BOUCLE 0D <-> 1D", "LA PERFORMANCE"):
            self.assertIn(attendu, texte)

    def test_le_balayage_rend_une_table(self):
        code, texte = quiet(["--sweep", "--dry-run", "--points", "5"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("maximum de poussee", texte)
        self.assertEqual(len([l for l in texte.splitlines()
                              if l.strip().startswith(("1.", "2.", "3.", "4.",
                                                       "5."))]), 5)

    def test_la_validation_confronte_aux_fourchettes(self):
        code, texte = quiet(["--validate", "--dry-run", "--points", "6"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("fourchette 600-900", texte)
        self.assertIn("la boucle 0D <-> 1D corrige quelque chose", texte)
        self.assertIn("tient dans les fourchettes publiees", texte)

    def test_le_domaine_de_validation_est_celui_de_l_etape_4(self):
        """« M0 de 2 a 3,5 », et pas « 2 a 4 » : c'est la correction mesuree a
        l'etape 4, ou 576,4 N.s/kg a M0 = 4 sortait de la fourchette."""
        _, texte = quiet(["--validate", "--dry-run", "--points", "6"])
        self.assertIn("M0 = 2 a 3,5", texte)


@unittest.skipUnless(NUMPY, "NumPy absent")
class LeCasBloqueSortEnDeux(unittest.TestCase):
    """Le critere du plan, et il demande DEUX nombres, pas un."""

    def setUp(self) -> None:
        self.code, self.texte = quiet(["--m-comb", "0.30", "--dry-run"])

    def test_le_code_de_sortie_est_2_et_pas_1(self):
        """UN CAS REFUSE PAR LA PHYSIQUE N'EST PAS UNE ERREUR D'USAGE.

        Le programme a parfaitement fonctionne ; c'est le point de
        fonctionnement qui n'existe pas. D'ou le code 2, « critere numerique
        non atteint », et non le code 1.
        """
        self.assertEqual(self.code, cli.EXIT_FAILED)

    def test_le_message_nomme_le_Mach_de_chambre_et_donne_le_T04_maximal(self):
        self.assertIn("0.3", self.texte)
        self.assertIn("1749 K", self.texte)
        self.assertIn("2000 K", self.texte)

    def test_et_il_donne_le_SECOND_Mach_celui_qui_s_applique(self):
        """CE QUE L'ETAPE 9 A ETABLI, ET QUE LE MESSAGE DOIT PORTER.

        La chambre ne tourne jamais au Mach vise : le col fixe le debit, donc
        le Mach reel — 0,2716 — et c'est SA limite, 2 055 K, qui s'applique.
        N'annoncer que les 1 749 K du Mach vise ferait chercher le probleme au
        mauvais endroit.
        """
        self.assertIn("0.2716", self.texte)
        self.assertIn("2055 K", self.texte)
        self.assertIn("C'est celle-la qui s'applique", self.texte)

    def test_le_message_dit_aussi_que_le_premier_passage_avait_accepte(self):
        self.assertIn("le premier passage avait accepte", self.texte)


@unittest.skipUnless(NUMPY, "NumPy absent")
class DryRun(unittest.TestCase):

    def _dossier_vide(self, argv):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as base:
            cible = Path(base) / "sorties"
            quiet(argv + ["-o", str(cible)])
            return cible

    def test_dry_run_n_ecrit_RIEN(self):
        for argv in (["--dry-run"], ["--sweep", "--dry-run", "--points", "3"],
                     ["--validate", "--dry-run", "--points", "3"]):
            with self.subTest(mode=argv[0]):
                cible = self._dossier_vide(argv)
                self.assertFalse(cible.exists(),
                                 f"{argv[0]} a cree {cible}")

    def test_sans_dry_run_le_rapport_est_ECRIT_et_relisible(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as base:
            cible = Path(base) / "sorties"
            code, texte = quiet(["-o", str(cible)])
            self.assertEqual(code, cli.EXIT_OK)
            fichiers = list(cible.rglob("rapport.txt"))
            self.assertEqual(len(fichiers), 1)
            contenu = fichiers[0].read_text(encoding="utf-8")
            self.assertIn("CE QU'ON PEUT CROIRE", contenu)
            # Le fichier et la sortie ecran disent la meme chose.
            self.assertIn(contenu.strip(), texte)

    def test_le_dossier_porte_le_nom_du_cas(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as base:
            cible = Path(base) / "sorties"
            quiet(["-o", str(cible), "--mach", "2.5"])
            self.assertTrue((cible / "M2.5_z11000_T2000_kerosene"
                             / "rapport.txt").exists())
