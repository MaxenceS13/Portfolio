"""cp(T) et gamma(T) : chiffrer l'erreur du modele a cp constant.

CE FICHIER TESTE D'ABORD DES DONNEES, PAS DU CODE. Cinquante coefficients
recopies sont cinquante occasions de se tromper, et ce projet a deja attrape
deux valeurs citees de memoire qui etaient fausses — le coefficient
« (gamma+1)/4 » de l'etape 8 et deux entrees de tables de chocs a l'etape 13.
Trois controles independants protegent celles-ci, et le second est decisif.

ENSUITE SEULEMENT VIENT LE RESULTAT : de combien le modele a cp constant se
trompe. La reponse depend de ce qu'on tient fixe, et les deux cadrages ne
donnent pas le meme signe. C'est le fait le plus interessant de l'etape.
"""

import unittest

import numpy as np

from ramjet import caloric
from ramjet.atmosphere import state
from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.cycle0d import run_cycle, run_cycle_variable
from ramjet.fuels import get as get_fuel

RU = caloric.RU


class LesDonneesSeVerifient(unittest.TestCase):

    def test_cp_a_298K_contre_les_valeurs_de_reference(self):
        """Premier controle : accord a 0,4 %. Les ecarts residuels sont la
        precision connue de ces ajustements, pas des fautes de frappe."""
        reference = {"N2": 29.12, "O2": 29.38, "CO2": 37.13, "H2O": 33.58,
                     "Ar": 20.786}
        for espece, valeur in reference.items():
            with self.subTest(espece=espece):
                mesure = float(caloric.cp_molar(espece, 298.15))
                self.assertAlmostEqual(mesure / valeur, 1.0, delta=0.005)

    def test_LE_RACCORD_A_1000K_est_exact(self):
        """LE CONTROLE DECISIF, ET IL NE DOIT RIEN A UNE SOURCE EXTERIEURE.

        Chaque espece a DEUX polynomes, ajustes separement sur [200 ; 1 000] et
        [1 000 ; 5 000], mais construits pour se rejoindre. Une seule faute de
        frappe dans l'un des dix jeux casse la continuite. Mesure : **0,000 %
        sur les cinq especes**, ce qui valide les cinquante nombres d'un coup.
        """
        for espece, (bas, haut) in caloric.NASA.items():
            with self.subTest(espece=espece):
                def cp(a):
                    T = caloric.T_JUNCTION
                    return RU * (a[0] + a[1] * T + a[2] * T ** 2
                                 + a[3] * T ** 3 + a[4] * T ** 4)
                self.assertAlmostEqual(cp(bas) / cp(haut), 1.0, delta=1e-5)

    def test_l_enthalpie_est_continue_a_2e_7_pres_et_PAS_MIEUX(self):
        """ET J'AI D'ABORD MESURE LE GRADIENT EN CROYANT MESURER LE SAUT.

        Comparer h(999,999) a h(1000,001) donne un ecart de 3e-6 relatif — mais
        c'est « cp dT » sur l'intervalle d'echantillonnage, pas une
        discontinuite. La meme erreur qu'a l'etape 1 sur l'atmosphere. Il faut
        retrancher la variation ATTENDUE :

            saut = | h(T+e) - h(T-e) - cp(T) . 2e |

        Mesure, en relatif a h : 2,4e-07 sur N2, 1,8e-07 sur O2, 2,1e-07 sur
        CO2, 6,8e-08 sur H2O. Les ajustements NASA sont donc continus a leur
        propre precision, pas exactement — et il faut le dire plutot que de
        choisir une tolerance qui cache la difference.

        L'ARGON TRANCHE : ses deux jeux de coefficients sont litteralement
        identiques, et le saut y vaut **1,8e-16**. Ce test mesure donc bien la
        DONNEE, et non l'arithmetique du module.
        """
        for espece in caloric.NASA:
            with self.subTest(espece=espece):
                epsilon = 1e-3
                juste_avant = float(caloric.enthalpy_molar(
                    espece, caloric.T_JUNCTION - epsilon))
                juste_apres = float(caloric.enthalpy_molar(
                    espece, caloric.T_JUNCTION + epsilon))
                attendu = float(caloric.cp_molar(
                    espece, caloric.T_JUNCTION)) * 2.0 * epsilon
                reference = abs(float(caloric.enthalpy_molar(
                    espece, caloric.T_JUNCTION)))
                saut = abs(juste_apres - juste_avant - attendu) / reference
                self.assertLess(saut, 1e-6)
                if espece == "Ar":
                    self.assertLess(saut, 1e-14)

    def test_R_de_l_air_retrouve_celui_que_le_projet_code_en_dur(self):
        """Troisieme controle croise : la masse molaire deduite de la
        composition donne R = 287,048, la ou tout le reste du projet code
        287,05. Les deux jeux de donnees sont donc coherents."""
        self.assertAlmostEqual(caloric.air().R, 287.05, delta=0.01)
        self.assertAlmostEqual(caloric.air().molar_mass * 1000.0, 28.9647,
                               delta=1e-3)

    def test_la_composition_de_l_air_se_ferme(self):
        self.assertAlmostEqual(sum(caloric.AIR_MOLE_FRACTIONS.values()), 1.0,
                               delta=1e-6)

    def test_le_f_stoechiometrique_deduit_colle_a_celui_du_registre(self):
        """QUATRIEME CONTROLE, ENTRE DEUX MODULES ECRITS SEPAREMENT.

        « fuels.py » stocke un « f_stoich » ; la composition atomique en donne
        un autre. Accord : 0,37 % sur le kerosene, 0,15 % sur le methane,
        0,14 % sur l'hydrogene.
        """
        for nom in ("kerosene", "methane", "hydrogen"):
            with self.subTest(carburant=nom):
                carburant = get_fuel(nom)
                deduit = caloric.stoichiometric_ratio(carburant)
                self.assertAlmostEqual(deduit / carburant.f_stoich, 1.0,
                                       delta=0.005)


class LAirSeComporteCommeLesTables(unittest.TestCase):

    def setUp(self) -> None:
        self.air = caloric.air()

    def test_cp_et_gamma_a_300K(self):
        """1 003,1 J/(kg.K) et 1,4009, contre les 1 004,5 et 1,4 du projet."""
        self.assertAlmostEqual(float(self.air.cp(300.0)), 1003.1, delta=1.0)
        self.assertAlmostEqual(float(self.air.gamma(300.0)), 1.4009,
                               delta=1e-3)

    def test_cp_monte_et_gamma_descend_avec_la_temperature(self):
        """LE FAIT QUI REND L'ETAPE UTILE. De 300 a 2 000 K, cp monte de 25 %
        et gamma tombe de 1,401 a 1,298."""
        temperatures = np.linspace(250.0, 3000.0, 40)
        chaleurs = self.air.cp(temperatures)
        gammas = self.air.gamma(temperatures)
        self.assertTrue(bool(np.all(np.diff(chaleurs) > 0.0)))
        self.assertTrue(bool(np.all(np.diff(gammas) < 0.0)))
        self.assertAlmostEqual(float(self.air.cp(2000.0)) / 1004.5, 1.245,
                               delta=0.01)
        self.assertAlmostEqual(float(self.air.gamma(2000.0)), 1.2978,
                               delta=1e-3)

    def test_un_gaz_MONOATOMIQUE_garde_gamma_egal_a_5_sur_3(self):
        """LE CAS OU LA REPONSE EST CONNUE D'AVANCE, DONC LE MEILLEUR TEST.

        L'argon n'a pas de degre de liberte interne : « cp = 5/2 Ru » et
        « gamma = 5/3 » a toute temperature. Si le code se trompait sur la
        conversion molaire/massique ou sur R, ce test tomberait.
        """
        argon = caloric.Mixture("argon", (("Ar", 1.0),))
        for T in (200.0, 1000.0, 4000.0):
            with self.subTest(T=T):
                self.assertAlmostEqual(float(argon.cp(T)),
                                       2.5 * RU / argon.molar_mass, delta=1e-6)
                self.assertAlmostEqual(float(argon.gamma(T)), 5.0 / 3.0,
                                       delta=1e-9)

    def test_la_detente_isentropique_redonne_la_loi_en_puissance(self):
        """SUR UN GAZ A GAMMA CONSTANT, LES DEUX CHEMINS DOIVENT COINCIDER.

        L'argon a gamma = 5/3 exactement, donc « T2/T1 = (p2/p1)^((g-1)/g) »
        y est exact. La bissection sur l'entropie doit le retrouver : c'est la
        verification de « isentropic_temperature », et elle ne depend d'aucune
        donnee.
        """
        argon = caloric.Mixture("argon", (("Ar", 1.0),))
        gamma = 5.0 / 3.0
        for rapport in (0.05, 0.2, 0.5, 2.0):
            with self.subTest(rapport=rapport):
                exact = 1500.0 * rapport ** ((gamma - 1.0) / gamma)
                self.assertAlmostEqual(
                    argon.isentropic_temperature(1500.0, rapport) / exact, 1.0,
                    delta=1e-9)

    def test_le_cp_MOYEN_n_est_pas_le_cp_a_la_temperature_moyenne(self):
        """ET L'ECART EST PLUS GRAND QUE JE NE LE CROYAIS : 1,6 %, PAS 0,3 %.

        « (h2-h1)/(T2-T1) » vaut 1 148,0 J/(kg.K) entre 300 et 2 000 K, la ou
        « cp(1 150 K) » en donne 1 166,4 : **18,3 de moins**. C'est le premier
        qui conserve l'energie, et c'est celui qu'un cycle demande. Prendre le
        second introduirait a lui seul une erreur du meme ordre que celle que
        toute cette etape cherche a mesurer.
        """
        moyen = self.air.mean_cp(300.0, 2000.0)
        au_milieu = float(self.air.cp(1150.0))
        self.assertAlmostEqual(moyen, 1148.0, delta=1.0)
        self.assertAlmostEqual(au_milieu, 1166.4, delta=1.0)
        self.assertAlmostEqual(100.0 * (moyen / au_milieu - 1.0), -1.57,
                               delta=0.1)

    def test_l_inversion_de_l_enthalpie_est_exacte(self):
        for T in (250.0, 800.0, 1500.0, 3000.0):
            with self.subTest(T=T):
                h = float(self.air.enthalpy(T))
                self.assertAlmostEqual(
                    self.air.temperature_from_enthalpy(h), T, delta=1e-8)


class LesProduitsDeCombustion(unittest.TestCase):

    def test_la_composition_se_ferme_et_l_oxygene_reste_en_exces(self):
        kerosene = get_fuel("kerosene")
        produits = caloric.combustion_products(kerosene, 0.03505)
        fractions = dict(produits.mole_fractions)
        self.assertAlmostEqual(sum(fractions.values()), 1.0, delta=1e-9)
        self.assertGreater(fractions["O2"], 0.0)
        self.assertGreater(fractions["CO2"], 0.0)
        self.assertGreater(fractions["H2O"], 0.0)
        # A moitie de la stoechiometrie, il reste beaucoup d'air.
        self.assertGreater(fractions["N2"], 0.7)

    def test_a_la_stoechiometrie_l_oxygene_est_juste_epuise(self):
        kerosene = get_fuel("kerosene")
        produits = caloric.combustion_products(
            kerosene, caloric.stoichiometric_ratio(kerosene))
        self.assertAlmostEqual(dict(produits.mole_fractions)["O2"], 0.0,
                               delta=1e-9)

    def test_au_dela_le_modele_REFUSE_au_lieu_d_inventer(self):
        """Un modele a reaction complete n'a rien a dire sur un melange riche :
        il faudrait de la chimie, que le perimetre exclut."""
        kerosene = get_fuel("kerosene")
        with self.assertRaises(CaseError) as capture:
            caloric.combustion_products(kerosene, 0.09)
        self.assertIn("trop riche", str(capture.exception))

    def test_les_produits_ont_un_cp_plus_grand_que_l_air(self):
        """CO2 et H2O sont des molecules a plus de degres de liberte."""
        kerosene = get_fuel("kerosene")
        produits = caloric.combustion_products(kerosene, 0.03505)
        self.assertGreater(float(produits.cp(2000.0)),
                           float(caloric.air().cp(2000.0)))


class LeResultatDeLEtape(unittest.TestCase):
    """De combien le modele a cp constant se trompe — et le signe depend du
    cadrage, ce qui est le fait le plus interessant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vol, cls.cas = FlightCase(), EngineCase()
        cls.cycle = run_cycle(cls.vol, cls.cas)
        cls.T02 = cls.cycle.station("2").T0
        cls.f = cls.cycle.fuel_air_ratio
        cls.kerosene = get_fuel("kerosene")

    def poussee(self, fuel_air_ratio: float, T04: float) -> float:
        """F/mdot a proprietes variables, detente isentropique comprise."""
        produits = caloric.combustion_products(self.kerosene, fuel_air_ratio)
        ambiante = state(self.vol.altitude)["p"]
        T9 = produits.isentropic_temperature(
            T04, ambiante / self.cycle.station("9").p0)
        vitesse = float(np.sqrt(2.0 * (produits.enthalpy(T04)
                                       - produits.enthalpy(T9))))
        return (1.0 + fuel_air_ratio) * vitesse - self.cycle.velocity_flight

    def test_a_f_donne_T04_tombe_de_256_K(self):
        """LE PLAN ANNONCAIT 100 A 200 K. C'EST 256, ET CELA CROIT AVEC T04.

        Mesure : -109 K a T04 = 1 400, -256 a 2 000, -443 a 2 600. La cause
        s'ecrit : le cp moyen des produits de 298 a 2 000 K vaut
        1 214 J/(kg.K), soit **1,21 fois** les 1 004,5 supposes.
        """
        reel = caloric.combustion_temperature(self.kerosene, self.f, self.T02,
                                              self.cas.eta_b)
        self.assertAlmostEqual(reel, 1743.6, delta=2.0)
        self.assertLess(reel - self.cas.T04, -200.0)
        produits = caloric.combustion_products(self.kerosene, self.f)
        self.assertAlmostEqual(produits.mean_cp(298.15, 2000.0) / 1004.5,
                               1.209, delta=0.01)

    def test_et_l_ecart_croit_avec_la_temperature_visee(self):
        attendu = {1400.0: -108.7, 1800.0: -202.4, 2200.0: -314.6,
                   2600.0: -442.3}
        for T04, ecart in attendu.items():
            with self.subTest(T04=T04):
                cycle = run_cycle(self.vol, EngineCase(T04=T04))
                reel = caloric.combustion_temperature(
                    self.kerosene, cycle.fuel_air_ratio, self.T02,
                    self.cas.eta_b)
                self.assertAlmostEqual(reel - T04, ecart, delta=2.0)

    def test_a_f_donne_la_poussee_est_SUResstimee_de_5_6_pour_cent(self):
        """LE CADRAGE DU PLAN, ET IL AVAIT RAISON : « quelques pour cent ».

        On dose du carburant, pas des kelvins. A « f » impose, T04 tombe a
        1 744 K et la poussee specifique passe de 739,0 a 697,5 N.s/kg.
        """
        reel = caloric.combustion_temperature(self.kerosene, self.f, self.T02,
                                              self.cas.eta_b)
        poussee = self.poussee(self.f, reel)
        self.assertAlmostEqual(poussee, 697.5, delta=2.0)
        self.assertAlmostEqual(
            100.0 * (poussee / self.cycle.specific_thrust - 1.0), -5.6,
            delta=0.5)

    def test_mais_a_T04_donne_elle_est_SOUSestimee_de_10_pour_cent(self):
        """LE FAIT QUI SURPREND, ET QUI N'EST PAS UNE CONTRADICTION.

        Si l'on IMPOSE T04 = 2 000 K — donc si l'on accepte de bruler les 27 %
        de carburant supplementaires qu'il faut vraiment — alors le cp plus
        grand rend PLUS d'enthalpie a la detente : u9 passe de 1 569 a
        1 643 m/s et la poussee de 739 a 816 N.s/kg.

        Les deux cadrages disent des choses differentes parce qu'ils ne
        tiennent pas la meme chose fixe. Le second flatte le moteur ; le
        premier decrit ce qui se passe reellement quand on ouvre le robinet
        d'une quantite donnee.
        """
        poussee = self.poussee(self.f, 2000.0)
        self.assertAlmostEqual(poussee, 815.6, delta=2.0)
        self.assertGreater(poussee, self.cycle.specific_thrust)

    def test_mais_l_IMPULSION_SPECIFIQUE_est_surestimee_dans_LES_DEUX(self):
        """L'ENONCE INVARIANT, ET DONC LE SEUL A RETENIR.

        Le signe de l'erreur sur la poussee depend du cadrage ; celui sur
        l'impulsion specifique, non. -5,6 % a « f » impose, -11,2 % a T04
        impose. Le modele a cp constant flatte la consommation dans tous les
        cas.
        """
        reel = caloric.combustion_temperature(self.kerosene, self.f, self.T02,
                                              self.cas.eta_b)
        a_f_impose = self.poussee(self.f, reel) / self.f / 9.80665

        lo, hi = 0.001, 0.06
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if caloric.combustion_temperature(self.kerosene, mid, self.T02,
                                              self.cas.eta_b) < 2000.0:
                lo = mid
            else:
                hi = mid
        f_vrai = 0.5 * (lo + hi)
        a_T04_impose = self.poussee(f_vrai, 2000.0) / f_vrai / 9.80665

        self.assertLess(a_f_impose, self.cycle.specific_impulse)
        self.assertLess(a_T04_impose, self.cycle.specific_impulse)
        self.assertAlmostEqual(f_vrai / self.f, 1.274, delta=0.01)


class Refus(unittest.TestCase):

    def test_espece_inconnue(self):
        with self.assertRaises(CaseError) as capture:
            caloric.cp_molar("CH4", 300.0)
        self.assertIn("espece inconnue", str(capture.exception))

    def test_temperature_hors_domaine(self):
        for T in (100.0, 6000.0):
            with self.subTest(T=T):
                with self.assertRaises(CaseError) as capture:
                    caloric.cp_molar("N2", T)
                self.assertIn("hors du domaine", str(capture.exception))

    def test_composition_qui_ne_se_ferme_pas(self):
        with self.assertRaises(CaseError) as capture:
            caloric.Mixture("faux", (("N2", 0.5), ("O2", 0.3)))
        self.assertIn("et non 1", str(capture.exception))

    def test_carburant_sans_composition_atomique_declaree(self):
        from ramjet.fuels import Fuel

        inconnu = Fuel("ammoniac", 18.6e6, 0.155)
        with self.assertRaises(CaseError) as capture:
            caloric.stoichiometric_ratio(inconnu)
        self.assertIn("composition atomique inconnue", str(capture.exception))

    def test_rendement_de_combustion_absurde(self):
        with self.assertRaises(CaseError):
            caloric.combustion_temperature(get_fuel("kerosene"), 0.03, 600.0,
                                           efficiency=1.5)


if __name__ == "__main__":
    unittest.main()


# -- le cycle 0D a proprietes variables --------------------------------------

class LeCycleVariable(unittest.TestCase):
    """Le cycle complet avec cp(T) et gamma(T), et sa verification.

    IL N'A PAS D'ORACLE A LUI, ET C'EST LE PROBLEME DE FOND. Les trois formes
    fermees du projet — isentropique, Rayleigh, Fanno — supposent toutes gamma
    constant. Un cycle a proprietes variables ne peut donc pas se comparer a
    l'une d'elles. Il se verifie autrement : par DEGENERESCENCE.
    """

    ARGON = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.ARGON = caloric.Mixture("argon", (("Ar", 1.0),))

    def test_LA_VERIFICATION_par_degenerescence_est_exacte(self):
        """FOURNI UN GAZ A cp CONSTANT, IL DOIT RENDRE LA FORME FERMEE.

        L'argon est le seul gaz monoatomique du jeu NASA : « cp = 5/2 Ru » et
        « gamma = 5/3 » a toute temperature. Le cycle a proprietes variables,
        nourri d'argon des deux cotes de la combustion, doit alors reproduire
        EXACTEMENT le cycle a cp constant calcule a la main — a condition
        d'utiliser la meme reference d'enthalpie, 298,15 K.

        Mesure sur six grandeurs : **ecart maximal 3,1e-15**. C'est ce qui rend
        les melanges des ARGUMENTS de la fonction et non des constantes du
        module : sans cette injection, la verification serait impossible.
        """
        gamma = 5.0 / 3.0
        R = self.ARGON.R
        cp = gamma * R / (gamma - 1.0)
        vol = FlightCase(mach=3.0, gamma=gamma, R=R)
        cas = EngineCase(T04=2000.0, pi_d=0.85)
        obtenu = run_cycle_variable(vol, cas, air_mixture=self.ARGON,
                                    products=self.ARGON)

        ambiant = state(vol.altitude)
        T0, p0 = ambiant["T"], ambiant["p"]
        u0 = vol.mach * np.sqrt(gamma * R * T0)
        T02 = T0 + u0 ** 2 / (2.0 * cp)
        p02 = p0 * (T02 / T0) ** (gamma / (gamma - 1.0)) * cas.pi_d
        Tref = caloric.T_REFERENCE
        f = ((cp * (cas.T04 - Tref) - cp * (T02 - Tref))
             / (cas.eta_b * cas.fuel_data.lhv - cp * (cas.T04 - Tref)))
        p09 = p02 * cas.pi_b * cas.pi_n
        T9 = cas.T04 * (p0 / p09) ** ((gamma - 1.0) / gamma)
        u9 = np.sqrt(2.0 * cp * (cas.T04 - T9))

        for nom, exact, calcule in (
                ("T02", T02, obtenu.station("2").T0),
                ("p02", p02, obtenu.station("2").p0),
                ("f", f, obtenu.fuel_air_ratio),
                ("T9", T9, obtenu.exit_temperature),
                ("u9", u9, obtenu.velocity_exit),
                ("F/mdot", (1.0 + f) * u9 - u0, obtenu.specific_thrust)):
            with self.subTest(grandeur=nom):
                self.assertAlmostEqual(calcule / exact, 1.0, delta=1e-13)

    def test_le_bilan_d_energie_de_la_combustion_se_referme(self):
        """Le controle qui ne depend d'aucun gamma : l'enthalpie entrante
        egale la sortante, a l'apport chimique pres."""
        vol, cas = FlightCase(), EngineCase()
        cycle = run_cycle_variable(vol, cas)
        f = cycle.fuel_air_ratio
        air = caloric.air()
        produits = caloric.combustion_products(cas.fuel_data, f)
        Tref = caloric.T_REFERENCE
        gauche = f * cas.eta_b * cas.fuel_data.lhv + float(
            air.enthalpy(cycle.station("2").T0) - air.enthalpy(Tref))
        droite = (1.0 + f) * float(produits.enthalpy(cas.T04)
                                   - produits.enthalpy(Tref))
        self.assertAlmostEqual(gauche / droite, 1.0, delta=1e-9)

    def test_LE_RESULTAT_l_impulsion_specifique_chute_de_11_pour_cent(self):
        """LE CHIFFRE QUI COMPTE, MAINTENANT SUR LE CYCLE ENTIER.

        L'etape 14 l'avait estime sur un seul bilan ; le voici sur les quatre
        stations, avec la detente resolue par egalite des entropies :

            f       0,0351 -> 0,0448   (+27,9 %)
            F/mdot   739,0 ->  836,7   (+13,2 %)
            Isp      2 150 ->  1 904   (-11,5 %)
            eta_th  0,5882 -> 0,5354   (-9,0 %)

        La poussee MONTE parce que T04 est impose : on brule plus, donc il y a
        plus d'enthalpie a detendre. Ce que le modele a cp constant flatte,
        c'est la CONSOMMATION.
        """
        vol, cas = FlightCase(), EngineCase()
        constant = run_cycle(vol, cas)
        variable = run_cycle_variable(vol, cas)
        self.assertAlmostEqual(variable.fuel_air_ratio / constant.fuel_air_ratio,
                               1.279, delta=0.01)
        self.assertAlmostEqual(
            variable.specific_thrust / constant.specific_thrust, 1.132,
            delta=0.01)
        self.assertAlmostEqual(
            variable.specific_impulse / constant.specific_impulse, 0.885,
            delta=0.01)
        self.assertLess(variable.thermal_efficiency,
                        constant.thermal_efficiency)

    def test_le_cp_moyen_de_combustion_est_celui_annonce(self):
        """1 232 J/(kg.K) de 298 a 2 000 K, soit 1,23 fois les 1 004,5
        supposes. C'est de la que vient tout l'ecart."""
        cycle = run_cycle_variable(FlightCase(), EngineCase())
        self.assertAlmostEqual(cycle.mean_cp_combustion, 1231.9, delta=2.0)
        self.assertAlmostEqual(cycle.mean_cp_combustion / 1004.5, 1.226,
                               delta=0.01)

    def test_la_compression_dynamique_chauffe_MOINS_qu_a_cp_constant(self):
        """601,3 K contre 606,6 : un cp plus grand absorbe la meme energie
        cinetique avec moins d'elevation de temperature."""
        constant = run_cycle(FlightCase(), EngineCase())
        variable = run_cycle_variable(FlightCase(), EngineCase())
        self.assertLess(variable.station("2").T0, constant.station("2").T0)
        self.assertAlmostEqual(variable.station("2").T0, 601.3, delta=0.5)

    def test_une_T04_hors_d_atteinte_est_REFUSEE_avec_son_maximum(self):
        """Le refus doit dire jusqu'ou on peut aller, pas seulement que non."""
        with self.assertRaises(CaseError) as capture:
            run_cycle_variable(FlightCase(), EngineCase(T04=4200.0))
        message = str(capture.exception)
        self.assertIn("hors d'atteinte", message)
        self.assertIn("plus riche dosage exploitable", message)
        self.assertIn("2557 K", message)

    def test_LA_VALIDATION_la_flamme_adiabatique_tombe_sur_la_litterature(self):
        """UNE VALIDATION, ET NON UNE VERIFICATION — MAIS ELLE EST DECISIVE.

        La temperature de flamme adiabatique du kerosene dans l'air, au dosage
        stoechiometrique et depuis 298 K, est publiee autour de **2 390 K** sans
        dissociation. Le modele a proprietes variables donne **2 394 K**.

        Le modele a cp constant, lui, donnerait 3 286 K depuis T02 = 607 K la ou
        les proprietes variables donnent 2 598 K : **26 % de trop**. Autrement
        dit, ce n'est pas une correction de raffinement — c'est la difference
        entre une temperature de flamme plausible et une qui ne l'est pas.
        """
        kerosene = get_fuel("kerosene")
        stoechio = caloric.stoichiometric_ratio(kerosene) * (1.0 - 1e-9)
        depuis_298 = caloric.combustion_temperature(kerosene, stoechio, 298.15,
                                                    1.0)
        self.assertAlmostEqual(depuis_298, 2394.0, delta=25.0)

        T02 = run_cycle(FlightCase(), EngineCase()).station("2").T0
        variable = caloric.combustion_temperature(kerosene, stoechio, T02, 1.0)
        constant = ((stoechio * kerosene.lhv + 1004.5 * T02)
                    / ((1.0 + stoechio) * 1004.5))
        self.assertAlmostEqual(variable, 2598.0, delta=10.0)
        self.assertAlmostEqual(constant, 3286.0, delta=10.0)
        self.assertGreater(constant / variable, 1.25)

    def test_l_objet_rendu_se_decrit(self):
        cycle = run_cycle_variable(FlightCase(), EngineCase())
        texte = cycle.describe()
        self.assertIn("cp moyen de combustion", texte)
        self.assertEqual(len(cycle.stations), 4)
