"""Statoreacteur : cycle 0D et ecoulement 1D avec combustion repartie.

Deux modeles du meme moteur, et l'interet est dans ce qu'ils se disent l'un a
l'autre. Le 0D, un cycle de Brayton station par station, donne la poussee
specifique et les rendements ; il suppose une perte de pression dans la chambre.
Le 1D, les equations de conservation le long de l'axe, la CALCULE — et refuse
des points de fonctionnement que le 0D accepte sans broncher, parce qu'il
connait le blocage thermique et que l'autre l'ignore.

Le fil conducteur : tout ce qui est affirme ici est verifie contre une solution
en FORME FERMEE avant d'etre valide contre la litterature. Le statoreacteur
ideal, l'ecoulement de Rayleigh et les relations isentropiques en fournissent
trois, exactes, et c'est ce qui rend le sujet interessant a traiter proprement.
"""

__version__ = "0.1.0"

# Modules qui doivent tourner sans SciPy ni Matplotlib. La liste s'allonge a
# chaque etape du plan ; « solver1d », « plots » et « report » n'y entreront
# jamais. Le test tests/test_architecture.py les importe dans un interpreteur
# NEUF et verifie qu'aucune des deux bibliotheques interdites n'a ete tiree au
# passage.
CORE_MODULES = (
    "config",
    "fuels",
    "atmosphere",
    "thermo",
    "cycle0d",
    "duct",
    "rayleigh",
    "fanno",
    "caloric",
)

# La couche 1D : elle a le droit d'importer SciPy, et elle est la SEULE avec
# « plots » a avoir un droit d'importation elargi. Declarer la liste sert a
# l'exhaustivite : le test verifie qu'aucun module du paquet n'echappe au
# classement, faute de quoi un fichier neuf pourrait n'etre ni noyau ni couche
# 1D, donc soumis a aucune regle.
SOLVER_MODULES = ("solver1d", "coupling")

# Les modules de service, hors du classement : ligne de commande, figures,
# rapport. Ils ont le droit de tout importer.
SERVICE_MODULES = ("cli", "plots", "report")

# Bibliotheques que les modules ci-dessus n'ont pas le droit d'importer.
FORBIDDEN_IN_CORE = ("scipy", "matplotlib")

# Contrainte PLUS FORTE, et sur ces deux modules seulement : ils ne doivent pas
# tirer NumPy non plus. Ce sont ceux qu'un script d'optimisation exterieur
# appellera en boucle pour construire des milliers de cas, et le cout
# d'importation de NumPy y serait pur gaspillage. C'est aussi ce qui garantit
# que la validation des cas ne depend d'aucun calcul.
STDLIB_ONLY_MODULES = ("config", "fuels")
