"""Cloisonnement : le noyau sans SciPy ni Matplotlib, la config sans NumPy.

Ce test est un garde-fou, pas une formalite. Le jour ou une commodite SciPy se
glisse dans « config.py », un script d'optimisation qui ne veut construire que
des cas paie une seconde d'importation par appel, et rien d'autre ne le signale.
La liste CORE_MODULES s'allonge a chaque etape du plan ; « solver1d », « plots »
et « report » n'y entreront jamais.
"""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

from ramjet import (CORE_MODULES, FORBIDDEN_IN_CORE,
                    SERVICE_MODULES, SOLVER_MODULES,
                    STDLIB_ONLY_MODULES)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _importe_dans_un_interpreteur_neuf(modules, interdits):
    """Rend la liste des bibliotheques interdites qui ont ete tirees.

    Un sous-processus, parce que les autres fichiers de tests ont deja importe
    NumPy et SciPy : dans CET interpreteur, la question n'a aucun sens.
    """
    imports = "; ".join(f"import ramjet.{n}" for n in modules)
    banned = ", ".join(repr(n) for n in interdits)
    code = (f"import sys; {imports}; "
            f"print(','.join(n for n in ({banned},) if n in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=120)
    return done


class Cloisonnement(unittest.TestCase):
    def test_les_modules_du_noyau_existent(self):
        for name in CORE_MODULES:
            with self.subTest(module=name):
                importlib.import_module(f"ramjet.{name}")

    def test_le_noyau_n_importe_ni_scipy_ni_matplotlib(self):
        done = _importe_dans_un_interpreteur_neuf(CORE_MODULES, FORBIDDEN_IN_CORE)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "",
                         f"le noyau a importe {done.stdout.strip()}")

    def test_config_et_fuels_tournent_meme_sans_NumPy(self):
        """La contrainte forte, et c'est celle qui compte pour ces deux-la.

        Ils sont importes par la ligne de commande, par les deux modeles et par
        tous les tests. Le jour ou ils tirent NumPy, ils tirent tout le paquet
        avec eux.
        """
        done = _importe_dans_un_interpreteur_neuf(
            STDLIB_ONLY_MODULES, FORBIDDEN_IN_CORE + ("numpy",))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "",
                         f"ces modules ont importe {done.stdout.strip()}")

    def test_solver1d_est_le_SEUL_module_a_importer_scipy(self):
        """Le cloisonnement se verifie dans les deux sens : le noyau reste
        propre, et « solver1d » est bien celui qui paie SciPy — sinon le
        cloisonnement serait obtenu en n'utilisant jamais la bibliotheque."""
        done = _importe_dans_un_interpreteur_neuf(("solver1d",), ("scipy",))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "scipy")

    def test_solver1d_n_importe_pas_matplotlib(self):
        done = _importe_dans_un_interpreteur_neuf(("solver1d",), ("matplotlib",))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "")

    def test_les_bibliotheques_cloisonnees_sont_les_deux_attendues(self):
        self.assertEqual(set(FORBIDDEN_IN_CORE), {"scipy", "matplotlib"})

    def test_AUCUN_module_n_echappe_au_classement(self):
        """AJOUTE A L'ETAPE 8, APRES AVOIR AJOUTE UN MODULE.

        « coupling.py » importe SciPy. Rien ne l'aurait signale : le test de
        cloisonnement ne regarde que les modules NOMMES, et un fichier neuf
        n'est nomme nulle part. Ce test parcourt le repertoire et exige que
        chaque module soit classe — noyau, couche 1D, ou service. Il echoue
        donc au prochain fichier ajoute, ce qui est exactement le but.
        """
        classes = set(CORE_MODULES) | set(SOLVER_MODULES) | set(SERVICE_MODULES)
        trouves = {f.stem for f in (PROJECT_ROOT / "ramjet").glob("*.py")
                   if not f.stem.startswith("__")}
        # LA CONTRAINTE NE VA QUE DANS UN SENS. Un module declare mais pas
        # encore ecrit est normal — « plots » et « report » sont les etapes 10
        # et 11. Un module ECRIT mais pas declare ne l'est jamais.
        self.assertEqual(trouves - classes, set(),
                         f"modules non classes : {sorted(trouves - classes)}")
        self.assertTrue(classes - trouves <= {"plots", "report"},
                        f"declares sans exister : {sorted(classes - trouves)}")

    def test_les_trois_classes_sont_disjointes(self):
        self.assertFalse(set(CORE_MODULES) & set(SOLVER_MODULES))
        self.assertFalse(set(CORE_MODULES) & set(SERVICE_MODULES))
        self.assertFalse(set(SOLVER_MODULES) & set(SERVICE_MODULES))

    def test_coupling_paie_scipy_comme_solver1d(self):
        """Il appartient a la couche 1D : il DOIT tirer SciPy, sinon il ne
        s'en sert pas et n'avait rien a y faire."""
        done = _importe_dans_un_interpreteur_neuf(("coupling",), ("scipy",))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "scipy")

    def test_la_contrainte_forte_porte_bien_sur_config_et_fuels(self):
        self.assertEqual(set(STDLIB_ONLY_MODULES), {"config", "fuels"})
        for name in STDLIB_ONLY_MODULES:
            self.assertIn(name, CORE_MODULES)


if __name__ == "__main__":
    unittest.main()
