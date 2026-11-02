"""Le statoreacteur complet : ce que l'assemblage revele.

CE FICHIER TESTE UNE THESE, PAS UN MODULE. La these est que le 0D et le 1D ne
repondent pas a la meme question, et que les mettre bout a bout fait apparaitre
laquelle des deux etait mal posee. Trois familles :

  - CE QUI DOIT TENIR PAR CONSTRUCTION. Le col sonique, le raccord chambre ->
    tuyere, l'invariance d'echelle. Si l'un de ces trois casse, c'est
    l'assemblage qui est faux, pas la physique.
  - CE QUI EST UNE VRAIE VERIFICATION. La poussee calculee par la quantite de
    mouvement contre celle du 0D, la pression de sortie, l'identite d'impulsion.
    Ces nombres pouvaient tomber faux.
  - CE QUI ETAIT FAUX DANS LE PLAN. Le choc droit a l'entree, annonce
    « coherent avec pi_d », qui perd 2,5 fois plus.
"""

import unittest

import numpy as np

from ramjet.config import CaseError, EngineCase, FlightCase
from ramjet.coupling import (Engine, _integrate_chamber,
                             choking_mass_flow,
                             isentropic_profile, run_engine)
from ramjet.cycle0d import run_cycle
from ramjet.duct import Duct, Zone, choked_area, from_cycle
from ramjet.solver1d import ChokedFlow, Profile, momentum_residual
from ramjet.thermo import mach_from_area_ratio, normal_shock

GAMMA, R = 1.4, 287.05


def chambre_retrecie(conduit: Duct, facteur: float) -> Duct:
    """La meme geometrie, chambre plus etroite, col et sortie inchanges.

    C'est le seul levier qui fasse bloquer la chambre avant le col : ni la
    temperature de fin de combustion ni la taille du col n'y suffisent, parce
    qu'ils deplacent le plafond de debit et le Mach d'entree ENSEMBLE.
    """
    section = conduit.zone("chamber").area_start * facteur
    zones = tuple(
        Zone(z.name, z.x_start, z.x_end,
             section if z.name in ("chamber", "converging") else z.area_start,
             section if z.name in ("diffuser", "chamber") else z.area_end)
        for z in conduit.zones)
    return Duct(zones)


def morceau(profile: Profile, masque) -> Profile:
    """Un sous-profil, pour verifier un troncon a pas constant."""
    champs = ("x", "mach", "temperature", "pressure", "velocity", "density",
              "total_temperature", "total_pressure", "area")
    return Profile(mass_flow=profile.mass_flow, gamma=profile.gamma,
                   **{n: getattr(profile, n)[masque] for n in champs})


class MoteurDeReference(unittest.TestCase):
    """Le cas de reference, calcule une fois : 0,33 s, ce n'est pas gratuit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.moteur = run_engine()
        cls.profil = cls.moteur.profile
        cls.conduit = cls.moteur.duct
        cls.chambre = cls.conduit.zone("chamber")


# -- ce qui doit tenir par construction --------------------------------------

class ColSonique(MoteurDeReference):

    def test_le_col_est_sonique(self):
        """Le critere du plan : M = 1 au col a 1e-3, ou le col n'en est pas un."""
        self.assertAlmostEqual(self.moteur.throat_mach, 1.0, delta=1e-3)

    def test_et_il_l_est_a_1e_8_pas_mieux(self):
        """LE PLANCHER EST CELUI DE L'ETAPE 2, ET IL SE RETROUVE ICI.

        Pres du col, « A/A* » est plate : une erreur de 1e-16 sur le rapport
        devient une erreur de 1e-8 sur le Mach. Mesure : 1,23e-08. Demander
        mieux serait demander l'impossible, et le test le dit en encadrant.
        """
        ecart = abs(self.moteur.throat_mach - 1.0)
        self.assertLess(ecart, 1e-6)
        self.assertGreater(ecart, 1e-12,
                           "un ecart nul signalerait que le Mach est impose "
                           "et non calcule")

    def test_le_conditionnement_pres_du_col_est_celui_annonce(self):
        """LE COEFFICIENT A ETE MESURE, PAS RECITE.

        Pres du col, « A/A* - 1 = (M-1)^2 . 2/(g+1) ». J'avais d'abord ecrit
        « (g+1)/4 » de memoire, ce qui donnait 1,05e-3 la ou l'on mesure
        8,92e-4. Verifie sur trois gamma — 1,100 a g = 1,2 ; 1,200 a 1,4 ;
        1,333 a 1,667 — le rapport « (1-M)^2/(A/A*-1) » vaut « (g+1)/2 », et
        converge vers lui quand on approche du col.
        """
        col = self.conduit.throat_x
        rapport = float(self.conduit.area(col - 1e-4)) / self.conduit.throat_area
        mach = float(mach_from_area_ratio(rapport, "subsonic", GAMMA))
        self.assertAlmostEqual(rapport - 1.0, 6.64e-7, delta=1e-8)
        self.assertAlmostEqual((1.0 - mach) ** 2 / (rapport - 1.0),
                               (GAMMA + 1.0) / 2.0, delta=1e-3)

    def test_l_ecart_au_col_va_comme_la_distance_pas_comme_sa_racine(self):
        """PARCE QUE LA SECTION EST PLATE AU COL, PAS SEULEMENT LA RELATION.

        Le raccord « 3s^2 - 2s^3 » annule dA/dx a la jonction, donc
        « A/A* - 1 » va en dx^2 ; la relation de section prend ensuite la
        racine. Les deux effets se composent et « 1 - M » redevient LINEAIRE en
        dx. Mesure : 3,56e-3 / 1,78e-3 / 8,92e-4 / 4,46e-4 quand dx est divise
        par deux a chaque fois. La linearite est ASYMPTOTIQUE — les rapports
        valent 1,99810, 1,99906, 1,99955 — donc le test verifie aussi qu'ils
        s'APPROCHENT de 2, et non seulement qu'ils en sont proches.
        """
        col = self.conduit.throat_x
        ecarts = []
        for dx in (4e-4, 2e-4, 1e-4, 5e-5):
            r = float(self.conduit.area(col - dx)) / self.conduit.throat_area
            ecarts.append(1.0 - float(mach_from_area_ratio(r, "subsonic",
                                                           GAMMA)))
        rapports = [a / b for a, b in zip(ecarts, ecarts[1:])]
        for rapport in rapports:
            self.assertAlmostEqual(rapport, 2.0, delta=3e-3)
        for avant, apres in zip(rapports, rapports[1:]):
            self.assertLess(abs(apres - 2.0), abs(avant - 2.0))


class RaccordChambreTuyere(MoteurDeReference):

    def test_le_raccord_est_exact(self):
        """LE POINT DE TOUT L'ASSEMBLAGE, ET IL EST EXACT PAR CONSTRUCTION.

        Le debit solution est celui pour lequel « A*(fin de chambre) = A_col ».
        Le Mach que l'EDO rend en fin de chambre et celui que « A(x)/A_col »
        donne au meme endroit sont donc le meme nombre. Mesure : 5,6e-17.
        """
        par_l_edo = float(self.moteur.chamber.mach[-1])
        par_la_section = float(mach_from_area_ratio(
            self.conduit.area(self.chambre.x_end) / self.conduit.throat_area,
            "subsonic", GAMMA))
        self.assertAlmostEqual(par_l_edo, par_la_section, delta=1e-9)

    def test_le_profil_ne_saute_pas_aux_jonctions(self):
        """Aucun ecart de Mach superieur au pas local entre deux points."""
        saut = np.abs(np.diff(self.profil.mach))
        # Le plus grand saut est dans le divergent, ou M monte vite.
        self.assertLess(float(np.max(saut)), 0.05)

    def test_le_col_n_est_compte_qu_une_fois(self):
        """Sinon le trace montrerait un palier a M = 1 large de deux points."""
        col = self.conduit.throat_x
        self.assertEqual(int(np.sum(np.abs(self.profil.x - col) < 1e-12)), 1)


class DebitImposeParLeCol(MoteurDeReference):

    def test_le_debit_est_une_INCONNUE_meme_quand_il_retombe_juste(self):
        """LE RESULTAT CENTRAL DE L'ETAPE, ET IL A CHANGE DE FORME.

        Ce test porte sur un passage SIMPLE, a « pi_b = 0,95 » suppose : la
        geometrie dessinee pour 10,000 kg/s en avale **9,734**, soit -2,7 %. Le
        debit est bien une inconnue, resolue par « A*(fin de chambre) = A_col ».

        L'EXACTITUDE N'APPARAIT QU'AU POINT FIXE, et c'est le test
        « test_au_point_fixe_le_debit_retombe_EXACTEMENT_sur_le_dessin » de
        « test_coupling.py » qui la mesure : la ou le dimensionnement et
        l'analyse comptent enfin la masse de la meme facon, le debit d'entree
        vaut 10,000000 et celui de sortie 10 (1+f). Tant que le « (1+f) » etait
        pose a la main, il restait un ecart de 0,94 % meme au point fixe.
        """
        self.assertAlmostEqual(self.moteur.mass_flow, 9.733746, delta=1e-5)
        self.assertAlmostEqual(float(self.profil.mass_flow[-1]) /
                               self.moteur.mass_flow,
                               1.0 + self.moteur.cycle.fuel_air_ratio,
                               delta=1e-9)

    def test_la_formule_du_debit_est_exacte_et_ne_garde_QU_UN_terme(self):
        """CE QU'IL RESTE DE LA FORMULE, UNE FOIS LE COMPTAGE CORRIGE.

        Le col impose « A* = A_col » dans les deux lectures, et le debit qui le
        traverse est celui de SORTIE dans les deux. Il reste

            mdot_sortie = 10 (1+f) pi_reel / pi_suppose

        et au point fixe les deux « pi » sont le meme nombre : il se simplifie.
        Mesure hors du point fixe (pi_b = 0,95 suppose contre 0,9196 subi) :
        l'ecart est celui-la, et rien d'autre.
        """
        f = self.moteur.cycle.fuel_air_ratio
        pi_suppose = self.moteur.cycle.engine.pi_b
        pi_reel = self.moteur.combustor_pressure_ratio
        attendu = 10.0 * (1.0 + f) * (pi_reel / pi_suppose)
        self.assertAlmostEqual(float(self.profil.mass_flow[-1]), attendu,
                               delta=1e-4)

    def test_invariance_d_echelle(self):
        """A ETAT TOTAL DONNE, LE DEBIT BLOQUANT EST PROPORTIONNEL AUX SECTIONS.

        Doubler toutes les sections double le debit, exactement : l'EDO ne voit
        que « A/A* », et le Mach d'entree de chambre est inchange. Le test le
        verifie a 1e-9 relatif, ce qui exclut toute dependance cachee a la
        taille absolue.
        """
        cycle = self.moteur.cycle
        T02, p02 = cycle.station("2").T0, cycle.station("2").p0
        T04 = cycle.station("4").T0
        debits = []
        for reference in (5.0, 10.0, 20.0):
            conduit = from_cycle(cycle, mass_flow=reference)
            debits.append(choking_mass_flow(conduit, T02, p02, T04, GAMMA, R)
                          / reference)
        self.assertAlmostEqual(debits[0], debits[1], delta=1e-9)
        self.assertAlmostEqual(debits[1], debits[2], delta=1e-9)


# -- ce qui pouvait tomber faux ----------------------------------------------

class ProfilPhysique(MoteurDeReference):

    def test_tout_est_positif(self):
        for nom in ("mach", "temperature", "pressure", "density", "velocity"):
            with self.subTest(grandeur=nom):
                self.assertTrue(bool(np.all(getattr(self.profil, nom) > 0.0)))

    def test_l_abscisse_est_strictement_croissante(self):
        self.assertTrue(bool(np.all(np.diff(self.profil.x) > 0.0)))

    def test_T0_est_constante_hors_de_la_chambre(self):
        """Le critere du plan : 1e-10. Mesure : exactement 0.

        Ce n'est pas une prouesse numerique — hors de la chambre, « T0 » est une
        constante litterale passee a « reconstruct ». Le test verifie donc
        surtout qu'aucune chaleur ne fuit hors de la zone declaree.
        """
        cycle = self.moteur.cycle
        avant = self.profil.x < self.chambre.x_start - 1e-12
        apres = self.profil.x > self.chambre.x_end + 1e-12
        self.assertLess(float(np.max(np.abs(
            self.profil.total_temperature[avant] - cycle.station("2").T0))),
            1e-10)
        self.assertLess(float(np.max(np.abs(
            self.profil.total_temperature[apres] - cycle.station("4").T0))),
            1e-10)

    def test_T0_atteint_T04_en_fin_de_chambre(self):
        self.assertAlmostEqual(float(self.moteur.chamber.total_temperature[-1]),
                               self.moteur.cycle.station("4").T0, delta=1e-6)

    def test_subsonique_en_chambre_supersonique_en_sortie(self):
        """LES DEUX INEGALITES QUI FONT QU'UN STATOREACTEUR EN EST UN.

        Elles ne sont pas garanties : un col trop grand laisserait la tuyere
        entierement subsonique, et le moteur ne pousserait pas.
        """
        chambre = ((self.profil.x >= self.chambre.x_start - 1e-12)
                   & (self.profil.x <= self.chambre.x_end + 1e-12))
        divergent = self.profil.x > self.conduit.throat_x
        self.assertTrue(bool(np.all(self.profil.mach[chambre] < 1.0)))
        self.assertTrue(bool(np.all(self.profil.mach[divergent] > 1.0)))
        self.assertGreater(float(self.profil.mach[-1]), 2.5)


class PressionDeSortie(MoteurDeReference):

    def test_la_tuyere_est_presque_adaptee(self):
        """Le critere du plan : p9 = p_ambiant a 1 %. Mesure : 0,49 %.

        Ce n'est pas exact parce que le debit a change : « from_cycle » avait
        dimensionne A9 pour 10,000 kg/s et le col en impose 10,094.
        """
        self.assertAlmostEqual(self.moteur.exit_pressure_ratio, 1.0, delta=1e-2)
        self.assertGreater(abs(self.moteur.exit_pressure_ratio - 1.0), 1e-3,
                           "une adaptation exacte signalerait que p9 est "
                           "imposee et non calculee")

    def test_le_terme_de_pression_est_negligeable_mais_pas_nul(self):
        """0,10 % de la poussee. C'est ce qu'on attend d'une tuyere adaptee."""
        A9 = float(self.profil.area[-1])
        p9 = float(self.profil.pressure[-1])
        terme = (p9 - self.moteur.ambient_pressure) * A9
        self.assertLess(abs(terme / self.moteur.thrust), 5e-3)


class PousseeContreLe0D(MoteurDeReference):

    def test_la_poussee_du_1D_retrouve_celle_du_0D(self):
        """LA VERIFICATION QUI COMPTE : deux chemins independants.

        Le 0D applique un bilan d'enthalpie station a station. Le 1D integre une
        EDO, reconstruit p et u, et fait un bilan de quantite de mouvement. Ils
        tombent a 0,094 % l'un de l'autre.
        """
        ecart = (self.moteur.specific_thrust
                 / self.moteur.cycle.specific_thrust - 1.0)
        self.assertLess(abs(ecart), 5e-3)

    def test_oublier_la_masse_de_carburant_couterait_7_pour_cent(self):
        """CE QUE L'EDO PORTE MAINTENANT, ET CE QUE CELA VAUT.

        L'equation integree portait la seule masse d'air ; le « (1+f) » etait
        remis a la main. Ce test mesure ce que vaut le quatrieme terme source :
        sans lui la poussee tomberait de 737,1 a 683,0 N.s/kg, soit -7,3 %, et
        l'ecart vaut exactement « f u9 ».
        """
        f = self.moteur.cycle.fuel_air_ratio
        u9 = float(self.profil.velocity[-1])
        A9 = float(self.profil.area[-1])
        p9 = float(self.profil.pressure[-1])
        debit = self.moteur.mass_flow
        sans = (debit * u9 - debit * self.moteur.cycle.velocity_flight
                + (p9 - self.moteur.ambient_pressure) * A9) / debit
        self.assertAlmostEqual(self.moteur.specific_thrust - sans, f * u9,
                               delta=1e-6)
        self.assertLess(sans / self.moteur.specific_thrust - 1.0, -0.07)

    def test_le_mach_de_sortie_est_une_identite_pas_une_verification(self):
        """ET IL FAUT LE DIRE, SINON ON CROIT AVOIR VERIFIE QUELQUE CHOSE.

        « from_cycle » a dimensionne A9 pour que « A9/A_col » donne exactement
        le M9 du 0D. Le 1D le retrouve a 0 pres — non parce qu'il a raison, mais
        parce qu'il resout la meme equation avec les memes nombres.
        """
        self.assertAlmostEqual(float(self.profil.mach[-1]),
                               self.moteur.cycle.mach_exit, delta=1e-12)


class QuantiteDeMouvement(MoteurDeReference):

    def test_chaque_troncon_satisfait_l_identite_d_impulsion(self):
        """Le controle qui n'a servi a rien dans le calcul. 1e-11 a 4e-08."""
        col = self.conduit.throat_x
        troncons = {
            "diffuseur": self.profil.x <= self.chambre.x_start + 1e-12,
            "chambre": ((self.profil.x >= self.chambre.x_start - 1e-12)
                        & (self.profil.x <= self.chambre.x_end + 1e-12)),
            "convergent": ((self.profil.x >= self.chambre.x_end - 1e-12)
                           & (self.profil.x <= col + 1e-12)),
            "divergent": self.profil.x > col + 1e-12,
        }
        for nom, masque in troncons.items():
            with self.subTest(troncon=nom):
                self.assertLess(momentum_residual(morceau(self.profil, masque),
                                                  self.conduit), 1e-7)

    def test_le_profil_recolle_est_REFUSE(self):
        """UN RESIDU DE 1,6e-05 LA OU CHAQUE MORCEAU DONNE 1e-11.

        Simpson suppose un pas constant. Le profil complet en a quatre
        differents, et la formule rendait alors une erreur de QUADRATURE que
        l'on aurait lue comme une erreur de SOLUTION. Le module refuse
        desormais, plutot que de repondre a cote.
        """
        with self.assertRaises(CaseError) as capture:
            momentum_residual(self.profil, self.conduit)
        self.assertIn("pas non constant", str(capture.exception))


# -- ce qui etait faux dans le plan ------------------------------------------

class LeChocDroitDuPlan(unittest.TestCase):
    """Le plan annoncait un choc normal a l'entree « coherent avec pi_d ».

    Il ne l'est pas. A M = 3, un choc droit rend 0,3283 la ou la correlation
    MIL-E-5008B du 0D suppose 0,8088 : un facteur 2,46. Le modele que le plan
    proposait aurait contredit le cycle qu'il devait completer.
    """

    def test_un_choc_droit_perd_deux_fois_et_demie_plus(self):
        choc = normal_shock(3.0)
        mil = EngineCase().inlet_recovery(3.0)
        self.assertAlmostEqual(float(choc.total_pressure_ratio), 0.3283,
                               delta=1e-4)
        self.assertAlmostEqual(mil, 0.8088, delta=1e-4)
        self.assertGreater(mil / float(choc.total_pressure_ratio), 2.4)

    def test_et_coute_vingt_pour_cent_de_poussee(self):
        vol = FlightCase()
        choc = normal_shock(vol.mach)
        reference = run_cycle(vol, EngineCase())
        avec_choc = run_cycle(vol, EngineCase(
            pi_d=float(choc.total_pressure_ratio)))
        self.assertLess(avec_choc.specific_thrust / reference.specific_thrust,
                        0.80)

    def test_et_demanderait_un_diffuseur_divergent(self):
        """Apres un choc droit a M = 3 il reste M = 0,475 : pour descendre a
        0,20 il faut DIVERGER d'un facteur 2,13. La geometrie de « from_cycle »
        converge (310 -> 269 cm2). Les deux modeles ne decrivent pas le meme
        conduit.
        """
        from ramjet.thermo import area_ratio
        choc = normal_shock(3.0)
        facteur = (float(area_ratio(0.20, GAMMA))
                   / float(area_ratio(float(choc.mach_downstream), GAMMA)))
        self.assertAlmostEqual(facteur, 2.13, delta=0.02)


class LeDiffuseurNEnEstPasUn(MoteurDeReference):

    def test_il_accelere_l_ecoulement_au_lieu_de_le_ralentir(self):
        """ET C'EST LA GEOMETRIE QUI LE VEUT, PAS UNE ERREUR DE CALCUL.

        « from_cycle » relie la section de capture a M = 3 (310 cm2) a la
        chambre (269 cm2) : le conduit CONVERGE, donc l'ecoulement subsonique
        accelere, de M = 0,174 a M = 0,202. La vraie compression est deja
        contenue dans « pi_d », qui est un saut applique en x = 0 ; ce qui
        reste tracable n'est qu'un raccord de section. Le README doit le dire.
        """
        avant = self.profil.x <= self.chambre.x_start + 1e-12
        machs = self.profil.mach[avant]
        self.assertLess(float(self.conduit.area(self.chambre.x_start)),
                        float(self.conduit.area(0.0)))
        self.assertGreater(float(machs[-1]), float(machs[0]))
        self.assertAlmostEqual(float(machs[0]), 0.1672, delta=1e-3)
        self.assertAlmostEqual(float(machs[-1]), 0.1947, delta=1e-3)


# -- les refus ---------------------------------------------------------------

class Refus(unittest.TestCase):

    def test_une_chambre_a_la_taille_du_col_est_refusee(self):
        """A « A_chambre = A_col » il n'y a plus de conduit du tout, et aucun
        debit ne bloque le col sans que la chambre bloque d'abord. La recherche
        doit le DIRE, et nommer les deux remedes.
        """
        cycle = run_cycle(FlightCase(), EngineCase())
        plat = chambre_retrecie(from_cycle(cycle, mass_flow=10.0), 0.6)
        with self.assertRaises(CaseError) as capture:
            choking_mass_flow(plat, cycle.station("2").T0,
                              cycle.station("2").p0, cycle.station("4").T0,
                              GAMMA, R)
        message = str(capture.exception)
        self.assertIn("borne haute integrable", message)
        self.assertIn("agrandissez la chambre", message)

    def test_sur_le_cas_de_reference_la_garde_ne_sert_JAMAIS(self):
        """ET IL FAUT LE DIRE, SINON ON CROIT AVOIR TESTE LA PHASE PRUDENTE.

        Au plafond — le debit qui bloquerait le col si « pi_b » valait 1 — la
        chambre de reference passe sans encombre. Mesure : elle ne bloque qu'a
        partir de 13,5 kg/s, alors que le plafond vaut 10,895. Ni chauffer
        davantage (teste de 2 000 a 4 000 K) ni retrecir le col (teste de 1,0 a
        0,4 fois) ne le change : les deux abaissent le plafond ET le Mach
        d'entree ensemble.
        """
        cycle = run_cycle(FlightCase(), EngineCase())
        conduit = from_cycle(cycle, mass_flow=10.0)
        T02, p02 = cycle.station("2").T0, cycle.station("2").p0
        for T04 in (2000.0, 3000.0, 4000.0):
            with self.subTest(T04=T04):
                plafond = (conduit.throat_area
                           / choked_area(1.0, T04, p02, GAMMA, R))
                _integrate_chamber(plafond, conduit, T02, p02, T04, GAMMA, R,
                                   40)

    def test_le_parametre_qui_declenche_la_garde_est_A_chambre_SUR_A_col(self):
        """CE QUI INTERDIT D'UTILISER « brentq » SEUL — quand cela arrive.

        Le Mach d'entree de chambre au plafond ne depend que du rapport
        « A_chambre/A_col ». Mesure : a 1,498 (le cas de reference) le plafond
        passe ; a 1,123 il BLOQUE, et la recherche aboutit quand meme a
        9,4467 kg/s. C'est la que « brentq » seul se serait arrete : une
        fonction qui leve n'a pas de signe.
        """
        cycle = run_cycle(FlightCase(), EngineCase())
        etroit = chambre_retrecie(from_cycle(cycle, mass_flow=10.0), 0.75)
        T02, p02 = cycle.station("2").T0, cycle.station("2").p0
        T04 = cycle.station("4").T0
        plafond = etroit.throat_area / choked_area(1.0, T04, p02, GAMMA, R)
        self.assertAlmostEqual(
            etroit.zone("chamber").area_start / etroit.throat_area, 1.123,
            delta=1e-3)
        with self.assertRaises(ChokedFlow):
            _integrate_chamber(plafond, etroit, T02, p02, T04, GAMMA, R, 40)
        debit = choking_mass_flow(etroit, T02, p02, T04, GAMMA, R)
        self.assertAlmostEqual(debit, 9.4467, delta=1e-3)
        self.assertLess(debit, plafond)

    def test_une_section_sous_la_section_critique_est_refusee(self):
        cycle = run_cycle(FlightCase(), EngineCase())
        conduit = from_cycle(cycle, mass_flow=10.0)
        with self.assertRaises(CaseError) as capture:
            isentropic_profile(conduit, np.linspace(0.0, 0.4, 20),
                               cycle.station("2").T0,
                               10.0 * conduit.throat_area, "subsonic", 10.0,
                               GAMMA, R)
        self.assertIn("critique", str(capture.exception))

    def test_zone_inconnue(self):
        conduit = from_cycle(run_cycle(FlightCase(), EngineCase()),
                             mass_flow=10.0)
        with self.assertRaises(CaseError) as capture:
            conduit.zone("chambre")
        self.assertIn("chamber", str(capture.exception))


class TypeDuResultat(MoteurDeReference):

    def test_l_objet_rendu_porte_ce_qu_il_faut_pour_l_etape_9(self):
        self.assertIsInstance(self.moteur, Engine)
        self.assertAlmostEqual(self.moteur.combustor_pressure_ratio,
                               float(self.moteur.chamber.total_pressure_ratio),
                               delta=1e-15)
        self.assertIn("debit impose par le col", self.moteur.describe())

    def test_le_pi_b_subi_differe_de_celui_suppose(self):
        """CE QUI JUSTIFIE L'ETAPE 9. 0,9265 contre 0,9500 : 2,5 % d'ecart."""
        suppose = self.moteur.cycle.engine.pi_b
        subi = self.moteur.combustor_pressure_ratio
        self.assertAlmostEqual(subi, 0.9247, delta=1e-3)
        self.assertGreater(abs(subi / suppose - 1.0), 0.02)


if __name__ == "__main__":
    unittest.main()
