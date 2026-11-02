"""Les carburants : pouvoir calorifique et rapport stoechiometrique.

CE FICHIER EST ARRIVE A L'ETAPE 1 ET NON A L'ETAPE 2, ET C'EST DELIBERE. Le
plan le rangeait avec la thermodynamique, en suivant le motif de
« chaos-attractors » ou le registre des systemes ne pouvait pas etre importe par
la configuration — parce qu'il tirait NumPy. Ici la raison n'existe pas : un
carburant, c'est deux nombres et un nom, donc de la bibliotheque standard pure.
« config.py » peut donc valider le carburant A LA CONSTRUCTION, comme le plan le
demandait, sans la ceremonie d'un « bind » differe.

CE QUE CES DEUX NOMBRES DISENT, ET CE QU'ILS NE DISENT PAS. Le PCI donne
l'energie liberee par kilogramme de carburant, et le rapport stoechiometrique
dit combien de carburant l'air peut bruler. Ni l'un ni l'autre ne dit en
combien de temps : ce projet n'a pas de cinetique, donc la longueur de chambre
est une donnee geometrique et jamais un temps de reaction.

LE RAPPORT STOECHIOMETRIQUE SERT DE GARDE-FOU, PAS DE CALCUL. Le debit de
carburant sort d'un bilan d'energie, qui ne sait pas qu'il y a une limite : rien
n'empeche le modele de demander une richesse de 1,5. C'est « f_stoich » qui
permet de refuser, et refuser vaut mieux que de rendre une poussee tiree d'une
combustion impossible.
"""

from __future__ import annotations

from dataclasses import dataclass


class FuelError(ValueError):
    """Carburant inconnu, ou declare de facon incoherente."""


@dataclass(frozen=True)
class Fuel:
    """Un carburant, reduit a ce dont les deux modeles ont besoin."""

    name: str
    lhv: float           # pouvoir calorifique inferieur, J/kg
    f_stoich: float      # rapport carburant/air stoechiometrique, sans unite
    label: str = ""      # libelle lisible, pour les rapports

    def __post_init__(self) -> None:
        if not self.name:
            raise FuelError("un carburant doit avoir un nom")
        if self.lhv <= 0.0:
            raise FuelError(f"{self.name} : le PCI doit etre positif "
                            f"(recu {self.lhv:g})")
        if not 0.0 < self.f_stoich < 1.0:
            raise FuelError(f"{self.name} : le rapport stoechiometrique doit "
                            f"etre dans ]0, 1[ (recu {self.f_stoich:g})")
        if not self.label:
            object.__setattr__(self, "label", self.name)

    def equivalence_ratio(self, f: float) -> float:
        """Richesse : le rapport du dosage au dosage stoechiometrique."""
        return f / self.f_stoich


# Le registre. Ajouter un carburant coute trois lignes et rien d'autre : ni
# dans la configuration, ni dans la ligne de commande, ni dans les modeles.
REGISTRY: dict[str, Fuel] = {}


def register(fuel: Fuel) -> Fuel:
    if fuel.name in REGISTRY:
        raise FuelError(f"deux carburants s'appellent {fuel.name!r}")
    REGISTRY[fuel.name] = fuel
    return fuel


def get(name: str) -> Fuel:
    """Le carburant enregistre sous ce nom.

    Leve « FuelError » plutot que « KeyError » : demander un carburant qui
    n'existe pas est une erreur d'utilisateur, du meme ordre qu'un parametre
    hors bornes, et la ligne de commande la traite deja comme telle.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise FuelError(
            f"carburant inconnu : {name!r} (au choix : {', '.join(available())})"
        ) from None


def available() -> tuple[str, ...]:
    """Les noms enregistres, tries."""
    return tuple(sorted(REGISTRY))


# PCI et dosages stoechiometriques usuels. Le kerosene est la valeur de
# Mattingly (hPR = 42 800 kJ/kg), celle qui sert aux cas de reference du plan.
register(Fuel("kerosene", 42.8e6, 0.0679, "kerosene (Jet-A / JP-8)"))
register(Fuel("hydrogen", 120.0e6, 0.0292, "hydrogene"))
register(Fuel("methane", 50.0e6, 0.0581, "methane"))
