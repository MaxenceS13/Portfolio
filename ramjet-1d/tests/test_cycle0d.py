"""Le cycle 0D, juge d'abord par une forme fermee, ensuite par des identites.

L'ORDRE DES CLASSES EST CELUI DES CERTITUDES. Le premier test compare le calcul
station par station a la solution analytique du statoreacteur ideal : c'est le
seul endroit du 0D ou la reponse est connue EXACTEMENT, et il ne reviendra pas.
Viennent ensuite trois identites — le rendement de Brayton, le produit des
rendements, le bilan d'energie global — qui tiennent avec les pertes. Puis les
valeurs de reference, qui ne sont que des ordres de grandeur. Puis les refus.
"""

import unittest

from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.cycle0d import (Cycle, IdealCycle, combustor, freestream,
                            ideal_cycle, ideal_engine, inlet, mach_limit,
                            run_cycle)
from ramjet.thermo import total_temperature_ratio

MACHS = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


def ideal(mach: float) -> tuple[FlightCase, EngineCase]:
    return FlightCase(mach=mach), ideal_engine(EngineCase())


class LeCritereDeLEtape(unittest.TestCase):
    """Station par station contre forme fermee, sans aucune perte.

    Ce test valide d'un coup les quatre stations, le bilan d'energie et la
    detente. Mesure : 2,3e-13 sur la poussee specifique, pour 1e-12 demande.
    """

    def test_la_poussee_specifique_tombe_sur_la_forme_fermee(self):
        for mach in MACHS:
            with self.subTest(mach=mach):
                vol, moteur = ideal(mach)
                ferme = ideal_cycle(vol, moteur).specific_thrust
                stations = run_cycle(vol, moteur).specific_thrust
                self.assertAlmostEqual(stations, ferme, delta=1e-9)
                self.assertLess(abs(stations - ferme) / abs(ferme), 1e-12)

    def test_le_mach_de_sortie_EGALE_le_mach_de_vol(self):
        """Sans perte, le rapport de pression totale vaut 1 et la detente est
        complete : M9 = M0 est alors une consequence, pas un reglage."""
        for mach in MACHS:
            with self.subTest(mach=mach):
                vol, moteur = ideal(mach)
                self.assertAlmostEqual(run_cycle(vol, moteur).mach_exit, mach,
                                       delta=1e-12)

    def test_le_dosage_et_la_consommation_tombent_aussi(self):
        for mach in MACHS:
            with self.subTest(mach=mach):
                vol, moteur = ideal(mach)
                ferme, stations = ideal_cycle(vol, moteur), run_cycle(vol, moteur)
                self.assertAlmostEqual(stations.fuel_air_ratio,
                                       ferme.fuel_air_ratio, delta=1e-15)
                self.assertAlmostEqual(stations.tsfc / ferme.tsfc, 1.0,
                                       delta=1e-12)

    def test_les_trois_rendements_tombent_aussi(self):
        for mach in (2.0, 3.0, 5.0):
            with self.subTest(mach=mach):
                vol, moteur = ideal(mach)
                ferme, stations = ideal_cycle(vol, moteur), run_cycle(vol, moteur)
                for nom in ("thermal_efficiency", "propulsive_efficiency",
                            "overall_efficiency"):
                    self.assertAlmostEqual(getattr(stations, nom),
                                           getattr(ferme, nom), delta=1e-12)

    def test_l_oracle_est_bien_un_objet_a_part(self):
        vol, moteur = ideal(3.0)
        self.assertIsInstance(ideal_cycle(vol, moteur), IdealCycle)
        self.assertIsInstance(run_cycle(vol, moteur), Cycle)


class IdentitesExactes(unittest.TestCase):
    """Trois egalites qui doivent tenir a la precision machine."""

    def test_le_rendement_thermique_ideal_vaut_EXACTEMENT_1_moins_1_sur_tau_r(self):
        """Et l'exactitude est le resultat : la docstring du module annoncait
        d'abord que l'egalite n'etait vraie qu'a la limite « f -> 0 ». Le test
        l'a contredite a 5,5e-17, et l'algebre confirme — « f » se simplifie
        entre le numerateur et le denominateur.
        """
        for mach in (2.0, 3.0, 4.0, 5.0):
            with self.subTest(mach=mach):
                vol, moteur = ideal(mach)
                cycle = run_cycle(vol, moteur)
                brayton = 1.0 - 1.0 / float(total_temperature_ratio(mach, vol.gamma))
                self.assertAlmostEqual(cycle.thermal_efficiency, brayton,
                                       delta=1e-15)
                self.assertGreater(cycle.fuel_air_ratio, 0.005)   # f n'est PAS nul

    def test_le_rendement_propulsif_DEPEND_du_debit_de_carburant(self):
        """La forme classique « 2/(V9/V0 + 1) » neglige f. Ecart mesure :
        2,1 % a M0 = 2, 2,6 % a M0 = 3, 3,2 % a M0 = 4. C'est la forme
        complete qui est calculee, et l'ecart est teste pour qu'on ne
        confonde pas les deux."""
        attendus = {2.0: 0.0207, 3.0: 0.0259, 4.0: 0.0317}
        for mach, ecart_attendu in attendus.items():
            with self.subTest(mach=mach):
                cycle = run_cycle(*ideal(mach))
                sans_f = 2.0 / (cycle.velocity_exit / cycle.velocity_flight + 1.0)
                ecart = abs(cycle.propulsive_efficiency - sans_f) / sans_f
                self.assertAlmostEqual(ecart, ecart_attendu, delta=1e-3)

    def test_le_produit_des_rendements(self):
        """« eta_o = eta_th eta_p » : les trois sont calcules separement, a
        partir de trois puissances, donc l'identite est un vrai controle."""
        for mach in (2.0, 3.0, 5.0):
            with self.subTest(mach=mach):
                cycle = run_cycle(FlightCase(mach=mach), EngineCase())
                self.assertAlmostEqual(
                    cycle.overall_efficiency,
                    cycle.thermal_efficiency * cycle.propulsive_efficiency,
                    delta=1e-15)

    def test_le_bilan_d_energie_global(self):
        """Refait depuis les temperatures RENDUES, pas depuis les
        intermediaires : « (1+f) cp T09 - cp T00 = f eta_b PCI ».
        Mesure : 3,2e-16 en relatif, pour 1e-10 demande."""
        for mach in (2.0, 3.0, 5.0):
            with self.subTest(mach=mach):
                vol, moteur = FlightCase(mach=mach), EngineCase()
                cycle = run_cycle(vol, moteur)
                f = cycle.fuel_air_ratio
                gauche = ((1.0 + f) * vol.cp * cycle.station("9").T0
                          - vol.cp * cycle.station("0").T0)
                droite = f * moteur.eta_b * moteur.fuel_data.lhv
                self.assertAlmostEqual(gauche / droite, 1.0, delta=1e-12)


class ValeursDeReference(unittest.TestCase):
    """Le cas du plan, ideal et reel. Ce ne sont que des ordres de grandeur."""

    def test_le_cas_ideal_a_mach_trois(self):
        cycle = run_cycle(*ideal(3.0))
        self.assertAlmostEqual(cycle.specific_thrust, 777.3, delta=0.1)
        self.assertAlmostEqual(cycle.fuel_air_ratio, 0.03432, delta=1e-5)
        self.assertAlmostEqual(cycle.tsfc * 1e6, 44.15, delta=0.01)
        self.assertAlmostEqual(cycle.specific_impulse, 2309.0, delta=1.0)
        self.assertAlmostEqual(cycle.thermal_efficiency, 0.6429, delta=1e-4)

    def test_le_cas_reel_a_mach_trois(self):
        cycle = run_cycle(FlightCase(), EngineCase())
        self.assertAlmostEqual(cycle.station("2").T0, 606.6, delta=0.1)
        self.assertAlmostEqual(cycle.fuel_air_ratio, 0.03505, delta=1e-5)
        self.assertAlmostEqual(cycle.specific_thrust, 738.97, delta=0.1)
        self.assertAlmostEqual(cycle.tsfc * 1e6, 47.44, delta=0.01)
        self.assertAlmostEqual(cycle.mach_exit, 2.812, delta=1e-3)
        self.assertAlmostEqual(cycle.thermal_efficiency, 0.5882, delta=1e-4)

    def test_les_pertes_coutent_de_la_poussee(self):
        reel = run_cycle(FlightCase(), EngineCase())
        parfait = run_cycle(*ideal(3.0))
        self.assertLess(reel.specific_thrust, parfait.specific_thrust)
        self.assertGreater(reel.tsfc, parfait.tsfc)


class Monotonies(unittest.TestCase):
    """Trois tendances, et aucune ne demande de valeur tabulee."""

    def setUp(self):
        self.cycles = [run_cycle(FlightCase(mach=m), EngineCase())
                       for m in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)]

    def test_la_temperature_de_fin_de_diffuseur_croit_avec_le_mach(self):
        temperatures = [c.station("2").T0 for c in self.cycles]
        for avant, apres in zip(temperatures, temperatures[1:]):
            self.assertLess(avant, apres)

    def test_le_dosage_DECROIT_avec_le_mach_a_T04_fixee(self):
        """C'est le mecanisme qui tue le statoreacteur en hypersonique : plus
        l'air arrive chaud, moins il reste d'ecart a T04 pour bruler."""
        dosages = [c.fuel_air_ratio for c in self.cycles]
        for avant, apres in zip(dosages, dosages[1:]):
            self.assertGreater(avant, apres)

    def test_la_recuperation_du_diffuseur_decroit_avec_le_mach(self):
        moteur = EngineCase()
        recuperations = [moteur.inlet_recovery(m)
                         for m in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)]
        for avant, apres in zip(recuperations, recuperations[1:]):
            self.assertGreater(avant, apres)

    def test_la_pression_totale_ne_remonte_jamais_d_une_station_a_l_autre(self):
        """Aucun des trois organes ne fait de travail : la pression totale ne
        peut que descendre."""
        for cycle in self.cycles:
            with self.subTest(mach=cycle.flight.mach):
                pressions = [station.p0 for station in cycle.stations]
                for avant, apres in zip(pressions, pressions[1:]):
                    self.assertLessEqual(apres, avant)


class Stations(unittest.TestCase):
    """Ce que chaque organe fait, et ce qu'il ne fait pas."""

    def test_le_diffuseur_est_adiabatique(self):
        """Il ne fait pas de travail : T0 le traverse inchangee, exactement."""
        vol, moteur = FlightCase(), EngineCase()
        amont = freestream(vol)
        aval = inlet(vol, moteur, amont)
        self.assertEqual(aval.T0, amont.T0)
        self.assertLess(aval.p0, amont.p0)

    def test_la_chambre_est_le_seul_organe_qui_chauffe(self):
        cycle = run_cycle(FlightCase(), EngineCase())
        self.assertEqual(cycle.station("0").T0, cycle.station("2").T0)
        self.assertGreater(cycle.station("4").T0, cycle.station("2").T0)
        self.assertEqual(cycle.station("9").T0, cycle.station("4").T0)

    def test_sans_perte_les_quatre_pressions_totales_sont_EGALES(self):
        cycle = run_cycle(*ideal(3.0))
        pressions = [station.p0 for station in cycle.stations]
        for pression in pressions[1:]:
            self.assertEqual(pression, pressions[0])

    def test_une_station_inconnue_est_refusee(self):
        with self.assertRaises(KeyError):
            run_cycle(FlightCase(), EngineCase()).station("3")

    def test_le_cycle_se_decrit(self):
        texte = run_cycle(FlightCase(), EngineCase()).describe()
        for attendu in ("station 0", "station 9", "F/mdot", "eta_th", "Isp"):
            self.assertIn(attendu, texte)


class LimitesEtRefus(unittest.TestCase):
    """Ce que le modele refuse de calculer, et pourquoi."""

    def test_la_limite_thermique_est_analytique(self):
        """« M_lim = sqrt(2/(gamma-1) (T04/T0 - 1)) » : 6,4154 pour T04 =
        2 000 K a 11 km. C'est une formule, pas une mesure."""
        self.assertAlmostEqual(mach_limit(FlightCase(), EngineCase()), 6.4154,
                               delta=1e-4)
        self.assertGreater(mach_limit(FlightCase(), EngineCase(T04=2500.0)),
                           mach_limit(FlightCase(), EngineCase(T04=2000.0)))

    def test_au_dela_de_la_limite_c_est_un_REFUS_et_pas_une_poussee_negative(self):
        limite = mach_limit(FlightCase(), EngineCase())
        self.assertGreater(run_cycle(FlightCase(mach=limite - 0.05),
                                     EngineCase()).fuel_air_ratio, 0.0)
        with self.assertRaises(CaseError) as leve:
            run_cycle(FlightCase(mach=limite + 0.01), EngineCase())
        message = str(leve.exception)
        self.assertIn("plus rien a bruler", message)
        self.assertIn(f"{limite:.2f}", message)

    def test_une_richesse_superieure_a_un_est_refusee(self):
        """Mesure : a M0 = 2, la richesse passe 1 vers T04 = 3 050 K. Sans
        chimie, le modele n'a rien a dire au-dela."""
        self.assertLess(run_cycle(FlightCase(mach=2.0),
                                  EngineCase(T04=2900.0)).equivalence_ratio, 1.0)
        with self.assertRaises(CaseError) as leve:
            run_cycle(FlightCase(mach=2.0), EngineCase(T04=3100.0))
        message = str(leve.exception)
        self.assertIn("richesse", message)
        self.assertIn("kerosene", message)

    def test_une_T04_au_dela_de_ce_que_le_carburant_peut_donner(self):
        """Inatteignable en pratique — il faudrait 41 700 K pour le kerosene —
        mais un signe faux ailleurs le ferait apparaitre."""
        vol = FlightCase()
        moteur = EngineCase(T04=50_000.0)
        amont = inlet(vol, moteur, freestream(vol))
        with self.assertRaises(CaseError) as leve:
            combustor(vol, moteur, amont)
        self.assertIn("peut produire", str(leve.exception))

    def test_une_tuyere_qui_ne_peut_plus_detendre_est_refusee(self):
        """Quand les pertes depassent la compression dynamique, il n'y a pas de
        detente possible : le modele le dit au lieu de rendre un Mach imaginaire.
        """
        with self.assertRaises(CaseError) as leve:
            run_cycle(FlightCase(mach=0.6), EngineCase(pi_b=0.5))
        self.assertIn("detente", str(leve.exception))

    def test_l_oracle_refuse_aussi_hors_de_son_domaine(self):
        with self.assertRaises(CaseError):
            ideal_cycle(FlightCase(mach=7.0), ideal_engine(EngineCase()))


class MoteurIdeal(unittest.TestCase):
    def test_il_met_les_quatre_pertes_a_un_et_ne_touche_a_rien_d_autre(self):
        depart = EngineCase(T04=2200.0, fuel="hydrogen", M_combustor=0.25)
        parfait = ideal_engine(depart)
        self.assertEqual((parfait.eta_b, parfait.pi_d, parfait.pi_b, parfait.pi_n),
                         (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(parfait.T04, depart.T04)
        self.assertEqual(parfait.fuel, depart.fuel)
        self.assertEqual(parfait.M_combustor, depart.M_combustor)


if __name__ == "__main__":
    unittest.main()
