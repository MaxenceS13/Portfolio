# ramjet-1d — statoréacteur : cycle 0D et écoulement 1D avec combustion

## Contexte

Un statoréacteur n'a aucune pièce mobile : il comprime l'air par sa seule
vitesse, brûle, et détend. C'est le moteur le plus simple qu'on puisse écrire —
et c'est précisément pour ça qu'il est un bon sujet de portfolio : il n'y a
nulle part où cacher une approximation. Deux modèles, et ce qu'ils se disent
l'un à l'autre :

- le **0D**, un cycle de Brayton station par station, qui donne la poussée
  spécifique, la consommation spécifique et les rendements ;
- le **1D**, les équations de conservation le long de l'axe avec section
  variable et apport de chaleur réparti, qui donne les profils de Mach, de
  pression et de température.

**L'intérêt n'est pas d'obtenir des courbes, c'est que les deux modèles se
contredisent quand l'un des deux a tort.** Le 0D suppose une perte de pression
dans la chambre — typiquement `π_b = 0,95`, un chiffre qu'on recopie. Le 1D la
**calcule**, et mesure sur le prototype elle vaut 0,984 ou 0,896 selon le Mach
d'entrée de chambre. Mieux : le 0D accepte sans broncher des cas que le 1D
déclare **physiquement impossibles**, parce qu'il ignore le blocage thermique.
Boucler les deux est le résultat du projet.

Comme dans `cfd-cavity`, le plan sépare deux choses que les projets étudiants
confondent presque toujours :

- la **vérification** — « est-ce que je résous correctement les équations que
  j'ai écrites ? » → comparaison à des **solutions en forme fermée** qui
  existent toutes ici : relations isentropiques, écoulement de Rayleigh,
  statoréacteur idéal. Mesuré sur le prototype : accord à **1e-12** et mieux ;
- la **validation** — « est-ce que ces équations décrivent le bon moteur ? » →
  comparaison aux ordres de grandeur publiés pour un statoréacteur réel.

C'est la vérification qui fait la différence, et c'est elle qui ordonne les
étapes : chaque brique a un oracle exact **avant** que le premier profil
n'existe.

---

## Décisions techniques et pourquoi

### Le piège central : la singularité en M = 1, et elle est silencieuse

L'équation qui gouverne le 1D s'écrit, en coefficients d'influence (Shapiro) :

```
dM²/M² = −2(1 + (γ−1)/2 M²)/(1 − M²) · dA/A
         + (1 + γM²)(1 + (γ−1)/2 M²)/(1 − M²) · dT₀/T₀
```

Le `1 − M²` au dénominateur n'est pas une difficulté numérique, c'est de la
physique : **l'apport de chaleur pousse le Mach vers 1 par les deux côtés**, et
au-delà d'une certaine chaleur il n'existe plus de solution stationnaire. Le
débit se bloque, l'entrée d'air se désamorce. Cette chaleur maximale est
analytique — c'est le `T₀*` de Rayleigh — donc il n'y a **aucune excuse** à
intégrer au-delà.

**Ce que fait un intégrateur qui ne le sait pas, mesuré sur le prototype.** RK4
à pas fixe, 2 000 pas, chambre à section constante, `M_in = 0,3` :

| `q/q_max` | ce qui se passe |
|---|---|
| 0,99 | reste subsonique, finit à M = 0,889 — correct |
| **1,10** | **franchit M = 1 à x = 0,868, puis n'importe quoi** |
| **1,50** | **franchit M = 1 à x = 0,567, puis n'importe quoi** |

Aucune erreur, aucun avertissement, et un profil qui a l'air d'un profil.

*L'étape 7 a mesuré ce « n'importe quoi », et c'est PIRE que ce que ce tableau
annonçait d'abord.* Le plan écrivait « finit à M = 1,122 ». Cette valeur
n'existe pas : elle dépend entièrement du pas de discrétisation. À
`q/q_max = 1,10`, en faisant varier le nombre de pas de 500 à 8 000, le Mach
final vaut successivement **0,9264 / 0,7640 / 2,1088 / 1,2017 / 0,8390** — et
**trois de ces cinq valeurs sont subsoniques**, donc le résultat faux ne se
reconnaît même pas à son point d'arrivée. L'abscisse de franchissement, elle,
converge proprement (0,8680 → 0,8674) : avant la singularité, une intégration
reste une intégration. Le solveur non protégé ne rend donc pas seulement un
résultat faux, mais un résultat faux, d'allure ordinaire, et **non
reproductible**.

`solve_ivp` avec pas adaptatif fait mieux, mais à peine : il **échoue** avec
`Required step size is less than spacing between numbers`, un message qui ne
parle pas de physique et qu'on interprète naturellement comme un problème de
tolérance.

**La solution retenue, et elle tient en trois lignes :** un événement terminal
de `solve_ivp` posé à `M = 0,999`. Mesuré, il arrête proprement l'intégration à
`x = 0,9289` avec `success = True`, ce qui permet de dire *où* et *pourquoi* ça
bloque. Attention au sens : la fonction d'événement `0,999² − M²` **décroît**
vers zéro, donc `direction = −1`. Avec `direction = +1` — l'erreur naturelle —
**l'événement ne se déclenche jamais** et on retombe sur le message cryptique.
Le test qui protège ça vérifie les deux sens.

### Le couplage qui décide de tout : le Mach d'entrée de chambre

C'est le résultat que le 0D seul ne peut pas donner. À `M₀ = 3`, la compression
dynamique porte l'air à `T₀₂ = 606,6 K`. Pour atteindre `T₀₄ = 2 000 K` il faut
donc un rapport `T₀₄/T₀₂ = 3,3`. Or ce rapport est **plafonné** par le Mach
auquel le diffuseur a ralenti l'air (mesuré, γ = 1,4) :

| `M` entrée chambre | `T₀max/T₀in` | `T₀₄` maximal | `π_b` à `T₀r = 3,3` |
|---|---|---|---|
| 0,10 | 21,38 | 12 968 K | 0,9835 |
| 0,15 | 9,81 | 5 949 K | 0,9617 |
| 0,20 | 5,76 | 3 495 K | 0,9280 |
| **0,25** | **3,89** | **2 362 K** | **0,8751** |
| 0,30 | 2,88 | 1 749 K | **BLOQUÉ** |
| 0,40 | 1,89 | 1 147 K | **BLOQUÉ** |

Le prix du confort est visible dans la dernière colonne : ralentir davantage
repousse le blocage mais la perte de pression totale, elle, vient de nulle part
— c'est le compromis central du dessin d'un statoréacteur, et il tombe du modèle
plutôt que d'être posé à la main.

**Attention à la conclusion qu'on tire de ce tableau : je l'ai tirée fausse.**
Le plan concluait qu'à `M = 0,30` le cas `T₀₄ = 2 000 K` « n'existe pas ». Les
chiffres sont justes — c'est bien la limite de Rayleigh à `M = 0,30` — mais
**la chambre ne tourne jamais au Mach de conception**. Le col fixe le débit
(étape 8), donc le Mach réel d'entrée de chambre, qui vaut **0,2716** et non
0,30 : la limite applicable est **2 055 K**, et le point franchit le blocage
avec 55 K de marge. Il est bien refusé, mais pour une tout autre raison —
l'effondrement géométrique du dessin quand on réinjecte le `π_b` réel
(étape 9). Ce tableau décrit une chambre à Mach imposé ; le moteur, lui,
choisit son Mach.

### Deux modèles, et ce que chacun ne peut pas dire

Le 0D ne connaît pas la géométrie : il ne peut donc rien dire du blocage, ni de
l'endroit où la chaleur est déposée, ni de la longueur nécessaire. Le 1D ne
connaît pas le rendement de combustion ni le débit de carburant : il reçoit un
profil de `T₀(x)` et l'intègre. Les deux se rejoignent sur `π_b` :

```
   0D                                        1D
   π_b = 0,95 (supposé)      <-----------    π_b calculé par Rayleigh
   T₀₄ = 2000 K (imposé)     ----------->    dT₀/dx réparti sur la chambre
```

**La boucle de cohérence est le test décisif du projet** : partir du 0D avec un
`π_b` supposé, faire tourner le 1D, récupérer le `π_b` qu'il calcule, le
réinjecter dans le 0D, et itérer.

*Mesuré à l'étape 9 : « deux ou trois tours suffisent » était optimiste.*
L'itération nue converge **linéairement**, de rapport 0,172369, et demande
**sept** tours pour descendre à 1e-6. La secante sur `h(π) = g(π) − π` atteint
le point fixe exactement en **cinq appels**, et c'est elle qui est
implémentée. Surtout, la boucle ne boucle que par la **géométrie** : à conduit
figé, `π_b` sort du 1D sans dépendre du `π_b` d'entrée, et la « boucle » se
ferme en un tour avec un résidu nul au sens strict.

### Coefficients d'influence, pas volumes finis

Le 1D pourrait s'écrire en volumes finis sur les variables conservatives
`(ρ, ρu, ρE)`. Ce n'est pas le bon outil ici, pour trois raisons :

1. l'écoulement est **stationnaire** : marcher en espace sur une EDO est direct,
   marcher en temps jusqu'à convergence coûte cent fois plus ;
2. la forme en coefficients d'influence rend la **singularité visible** — le
   `1 − M²` est écrit dans le code, pas caché dans une matrice jacobienne ;
3. elle donne accès aux formes fermées comme cas limites, donc à des oracles
   exacts : `dT₀ = 0` redonne l'isentropique, `dA = 0` redonne Rayleigh.

Le prix : cette formulation ne sait pas traverser un choc. C'est assumé et
explicite — le diffuseur n'est pas intégré, il est traité comme un **saut**.

*Précision de l'étape 8, qui a changé la nature de ce saut.* Le plan prévoyait
un choc normal par Rankine-Hugoniot, « cohérent avec `π_d` ». Il ne l'est pas :
à `M₀ = 3` un choc droit rend **0,3283** là où la corrélation MIL-E-5008B du 0D
suppose **0,8088**, un facteur 2,46 — l'adopter aurait coûté 21,5 % de poussée
et contredit le cycle qu'il devait compléter. Le saut appliqué est donc celui de
`π_d`, qui décrit une entrée à rampes multiples ; le choc normal reste dans
`thermo.py`, utilisé comme oracle et comme point de comparaison chiffré.

### Reconstruire, puis vérifier par un chemin qui n'a pas servi

L'EDO n'intègre qu'une seule variable, `M²`. Tout le reste se reconstruit
algébriquement :

```
T  = T₀ / (1 + (γ−1)/2 M²)          (définition de T₀)
p  = ṁ √(RT) / (A M √γ)              (conservation de la masse)
u  = M √(γRT)
```

La **quantité de mouvement n'est utilisée nulle part** dans cette
reconstruction : elle est déjà encodée dans l'EDO. Elle devient donc un
contrôle **indépendant**, et c'est le meilleur test du solveur 1D — l'équivalent
de la divergence nulle de `cfd-cavity`. La fonction d'impulsion
`F = pA(1 + γM²)` doit vérifier `dF = p dA` le long du conduit, soit :

```
F(sortie) − F(entrée) = ∫ p (dA/dx) dx
```

Les deux membres se calculent depuis la solution, et rien dans l'intégration ne
les force à être égaux.

### Le modèle de combustion, et ce qu'il exclut explicitement

La combustion est un **apport de chaleur**, pas une chimie. En 0D, un bilan
d'énergie sur la chambre :

```
f · η_b · PCI = (1 + f) cp T₀₄ − cp T₀₂      →      f = cp(T₀₄ − T₀₂) / (η_b·PCI − cp T₀₄)
```

En 1D, le même total réparti sur une zone du conduit par un profil `dT₀/dx`
imposé. Deux conséquences à assumer et à écrire dans le README :

- la richesse est **libre** : rien n'empêche le modèle de demander plus de
  carburant que la stœchiométrie n'en permet. Un contrôle explicite compare `f`
  à `f_st ≈ 0,068` pour le kérosène et refuse au-delà, parce qu'une richesse de
  1,5 ne veut rien dire dans un modèle sans chimie ;
- `T₀₄` est un **paramètre**, pas un résultat : la dissociation, qui plafonne
  la température réelle vers 2 400–2 700 K, n'est pas modélisée. Le plafond est
  donc imposé de l'extérieur, et le dire est plus honnête que de le calculer
  avec une chimie qu'on n'a pas.

### Ce que le projet ne fera pas, et qu'il faut dire

- **pas de cinétique chimique** : ni mécanisme réactionnel, ni espèces, ni temps
  caractéristique de réaction. La longueur de chambre n'est donc pas dimensionnée
  par la chimie, et le projet ne prétend pas qu'elle l'est ;
- **pas de 2D ni de 3D** : pas de couche limite, pas de décollement, pas de
  profil radial. Le frottement pariétal peut entrer comme terme source 1D
  (scope optionnel), mais ce n'est pas une couche limite ;
- **pas d'instationnaire** : ni instabilité de combustion, ni buzz, ni
  désamorçage dynamique. Le blocage thermique est détecté comme une **limite
  d'existence** de la solution stationnaire, ce qui est le maximum qu'un modèle
  stationnaire puisse dire honnêtement ;
- **pas de gaz réel** : gaz parfait, `γ` et `cp` constants par zone (froide et
  chaude). La variation continue `cp(T)` est en scope optionnel.

---

## Conventions à respecter

Reprises telles quelles des sept projets existants du dépôt :

- Python 3.10+, `from __future__ import annotations` en tête de chaque module,
  unions PEP 604, `dataclasses` pour les structures partagées ;
- **README.md et plan.md en français avec accents** ; **docstrings et
  commentaires en français sans accents** ; **identifiants en anglais** ;
- CLI `argparse` avec couleurs ANSI, `--check`, `--validate`, `--figures`,
  `--dry-run`, `-v/--verbose`, codes de sortie `0` ok / `1` erreur /
  `2` critère numérique non atteint / `130` interrompu ;
- tests `unittest` : `python3 -m unittest discover -s tests`, aucun réseau.
  Budget annoncé : sous trente secondes — **mesuré à 223 s pour 468 tests** au
  terme du projet, soit un facteur sept. Le dépassement est réel et assumé : à
  partir de l'étape 8, un test coûte une intégration (0,33 s) ou une boucle
  complète (1,7 s), et les mutualiser par `setUpClass` a déjà divisé le coût par
  trois. Un budget de trente secondes n'était pas atteignable pour ce projet, et
  il valait mieux le mesurer que le tenir ;
- `sorties/` et `.venv/` dans `.gitignore`.

**Cloisonnement des dépendances**, comme `cfd-cavity` et `chaos-attractors`, et
vérifié par un test :

```
numpy>=1.24        obligatoire — thermo, cycle 0D, conduit, solveur 1D
scipy>=1.10        UNIQUEMENT dans solver1d.py (solve_ivp et ses evenements)
                   et dans les tests (brentq pour inverser les formes fermees)
matplotlib>=3.7    UNIQUEMENT dans plots.py
```

Conséquence à tenir : **le cycle 0D entier tourne avec NumPy seul**. Un test
importe `thermo`, `cycle0d`, `duct` et vérifie que ni SciPy ni Matplotlib
n'apparaissent dans `sys.modules`. Ce n'est pas une coquetterie : le 0D est la
partie qu'on veut pouvoir appeler depuis un script d'optimisation, et une
dépendance de tracé n'y a rien à faire.

Détail rencontré sur le prototype : `scipy.integrate.solve_ivp` attend
l'attribut `terminal` **posé sur la fonction** d'événement, et son `direction`
suit le signe de la dérivée de cette fonction, pas celui de la grandeur
surveillée. Les deux se trompent facilement, et les deux sont testés.

---

## Arborescence

```
ramjet-1d/
├── plan.md                    ← ce document
├── main.py                    point d'entree CLI, 8 lignes
├── README.md                  doc francaise
├── requirements.txt           banniere commentee, cloisonnement explicite
├── .gitignore                 sorties/ __pycache__/ .venv/
├── ramjet/
│   ├── __init__.py            docstring + __version__
│   ├── __main__.py            python3 -m ramjet
│   ├── cli.py                 argparse, couleurs ANSI, --check/--validate
│   ├── config.py              dataclasses FlightCase et EngineCase, gelees
│   ├── atmosphere.py          atmosphere standard ISA jusqu'a 32 km
│   ├── thermo.py              gaz parfait, relations isentropiques, choc normal
│   ├── rayleigh.py            formes fermees de Rayleigh — oracle de la chaleur
│   ├── fanno.py               formes fermees de Fanno — oracle du frottement
│   ├── caloric.py             polynomes NASA-7 : cp(T), gamma(T), entropie
│   ├── fuels.py               PCI, f_stoechiometrique, un dataclass par carburant
│   ├── cycle0d.py             le cycle station par station
│   ├── duct.py                geometrie : A(x), dA/dx, zones, profil de chaleur
│   ├── solver1d.py            l'EDO quasi-1D (SEUL module a importer scipy)
│   ├── coupling.py            la boucle 0D <-> 1D
│   ├── plots.py               figures (SEUL module a importer matplotlib)
│   └── report.py              rapport texte, tableaux alignes
├── tests/                     un test_<module>.py par module
└── sorties/                   gitignore — figures, rapports
```

---

## Spécification numérique

### Le cas de référence

Statoréacteur de croisière, kérosène, atmosphère standard à 11 km :

| grandeur | valeur |
|---|---|
| altitude | 11 000 m — `T₀ = 216,65 K`, `p₀ = 22 632 Pa` |
| Mach de vol | `M₀ = 3` (balayage de 1,5 à 5 pour les courbes) |
| carburant | kérosène, `PCI = 42,8 MJ/kg`, `f_st = 0,0679` |
| température de fin de combustion | `T₀₄ = 2 000 K` |
| rendement de combustion | `η_b = 0,98` |
| pertes | `π_d` = MIL-E-5008B, `π_b = 0,95` (0D), `π_n = 0,98` |
| gaz | `γ = 1,4`, `R = 287,05`, `cp = 1004,5 J/(kg·K)` |

`π_d = 1 − 0,075 (M₀ − 1)^1,35` pour `M₀ > 1` est la corrélation de récupération
de pression totale de la norme MIL-E-5008B. Elle vaut 0,809 à M₀ = 3 : **près de
20 % de la pression totale est perdue dans l'entrée d'air**, ce qui en fait le
premier poste de perte du moteur, loin devant la chambre.

### Les formes fermées qui servent d'oracles

Tout le projet est vérifiable contre des solutions exactes. C'est rare, et c'est
la raison pour laquelle ce sujet est un bon sujet.

*Le plan en annonçait quatre. Le projet en a six, et le compte final vaut la
peine d'être écrit : **chacun des quatre termes source de l'équation quasi-1D a
fini par avoir le sien**, ce qui permet de savoir lequel est faux quand un
profil déraille.*

1. **Relations isentropiques** — `T₀/T`, `p₀/p`, `A/A*` en fonction de `M`.
   Oracle du **terme de section**. Mesuré : 1,2e-12.
2. **Choc normal** — relations de Rankine-Hugoniot, pour le diffuseur
   supersonique. Sert aussi de brique aux chocs obliques de l'étape 13, qui
   n'en sont que la composante normale.
3. **Écoulement de Rayleigh** — section constante avec apport de chaleur :
   `T₀/T₀*`, `p₀/p₀*`, `p/p*`. Oracle du **terme de chaleur** (9,8e-15), et
   c'est lui qui donne la chaleur maximale.
4. **Statoréacteur idéal** — sans aucune perte, le cycle se résout à la main :

```
M₉ = M₀        et        F/ṁ = V₀ (√(τ_λ/τ_r) − 1)      avec τ_r = 1 + (γ−1)/2 M₀²,  τ_λ = T₀₄/T₀
```

C'est l'oracle du 0D : le code station par station doit retomber dessus **à la
précision machine** quand on met toutes les pertes à 1.

5. **Écoulement de Fanno** *(ajouté à l'étape 12)* — section constante avec
   frottement seul : `4C_f L*/D` en fonction de `M`. Oracle du **terme de
   frottement** (2,4e-15). Son `p₀/p₀*` est **littéralement** la même fonction
   que `A/A*` de l'isentropique, ce qui donne un contrôle croisé gratuit entre
   deux modules écrits à six étapes d'intervalle : écart **0,0 exactement**.
6. **Équivalence masse-chaleur** *(ajoutée après les quinze étapes)* — le
   coefficient d'apport de masse valant exactement le double de celui de la
   chaleur, ajouter de la masse à `T₀` constante revient *exactement* à chauffer
   selon Rayleigh avec `T₀ ∝ ṁ²`. Oracle du **terme d'injection** (1,2e-13).

**Et une septième vérification, qui n'est pas un oracle** : le cycle 0D à
`cp(T)` n'a aucune forme fermée à laquelle se comparer, puisque toutes celles
ci-dessus supposent `γ` constant. Il se vérifie par **dégénérescence** — nourri
d'argon, dont le `cp` ne varie pas, il doit reproduire le cycle à `cp` constant.
Mesuré : 3,1e-15.

### Valeurs de référence, mesurées sur le prototype

**Statoréacteur idéal**, `M₀ = 3`, `T₀₄ = 2 000 K`, `PCI = 42,8 MJ/kg` :

| grandeur | valeur |
|---|---|
| `f` | 0,03431 |
| poussée spécifique `F/ṁ` | **777,3 N·s/kg** |
| consommation spécifique `S` | 44,15 mg/(N·s) = 0,159 kg/(N·h) |
| impulsion spécifique | 2 309 s |
| `η_th` / `η_p` / `η_o` | 0,643 / 0,710 / 0,457 |
| `T₀₂` / `p₀₂` | 606,6 K / 8,313 bar (36,7 × `p₀`) |

Forme fermée contre calcul station par station : **écart 2,3e-13** à M₀ = 2, 3
et 4. C'est l'oracle, et il est exact.

**Statoréacteur réel**, mêmes conditions, avec les pertes du tableau ci-dessus :

| `M₀` | `π_d` | `T₀₂` (K) | `p₀₂/p₀` | `f` | `F/ṁ` (N·s/kg) | `S` (mg/(N·s)) | `M₉` | `η_th` |
|---|---|---|---|---|---|---|---|---|
| 1,5 | 0,971 | 314 | 3,6 | 0,0424 | 682,9 | 62,10 | 1,43 | 0,281 |
| 2,0 | 0,925 | 390 | 7,2 | 0,0405 | 762,0 | 53,15 | 1,90 | 0,406 |
| **2,5** | 0,870 | 488 | 14,9 | 0,0381 | **774,5** | 49,13 | 2,37 | 0,509 |
| 3,0 | 0,809 | 607 | 29,7 | 0,0351 | 739,0 | **47,43** | 2,81 | 0,588 |
| 3,5 | 0,742 | 747 | 56,6 | 0,0315 | 669,7 | 47,05 | 3,24 | 0,646 |
| 4,0 | 0,670 | 910 | 101,7 | 0,0274 | 576,4 | 47,57 | 3,65 | 0,686 |
| 5,0 | 0,513 | 1 300 | 271,2 | 0,0176 | 342,5 | 51,41 | 4,39 | 0,710 |

Trois choses à retrouver, et ce sont les critères de validation de l'étape 4 :

- la poussée spécifique **passe par un maximum à M₀ = 2,36** (776,5 N·s/kg) et
  s'effondre ensuite. Ce n'est pas un artefact : à mesure que `M₀` monte, `T₀₂`
  monte, et il reste de moins en moins d'écart à `T₀₄` pour brûler — `f` tombe
  de 0,042 à 0,018. *(Le tableau ci-dessus, au pas de 0,5, plaçait ce maximum à
  2,5 ; c'était le pas de la grille qu'on lisait, pas le maximum. Étape 4.)* ;
- la **limite thermique est analytique** : `f = 0` quand `T₀₂ = T₀₄`, soit
  `M₀ = 6,42` pour `T₀₄ = 2 000 K`. Le modèle doit y annoncer une poussée nulle
  et non une poussée négative ;
- le rendement thermique **continue de croître** quand la poussée s'effondre :
  les deux ne disent pas la même chose, et confondre les deux est l'erreur
  classique sur ce moteur. *(Il ne croît pas indéfiniment pour autant : mesuré à
  l'étape 4, il passe par un maximum de 0,7129 à M₀ = 4,79 puis retombe à 0,464
  à M₀ = 6. Le cycle idéal, lui, est bien monotone — la non-monotonie vient
  entièrement des pertes.)*

**Ordres de grandeur publiés**, pour la validation : poussée spécifique
600–900 N·s/kg **entre M 2 et 3,5**, impulsion spécifique 1 500–2 500 s,
rendement thermique 25–75 %.

*Correction de l'étape 4 : le plan disait « entre M 2 et 4 » et « le tableau
tombe dedans partout », alors que son propre tableau porte 576,4 N·s/kg à
M₀ = 4 — sous la borne basse. La bande a donc été resserrée à [2 ; 3,5], et un
test montre qu'à `T₀₄ = 2 200 K` le point M₀ = 4 remonte à 671,5 et rentre dans
la fourchette : ce n'était pas le modèle qui sortait du domaine, c'était le
domaine qui dépend de `T₀₄`. La fourchette de rendement a été élargie à
25–75 %, celle du code.*

### Vérification du solveur 1D, mesurée sur le prototype

**Contre Rayleigh** (section constante, `M_in = 0,3`, chaleur linéaire) :

| `T₀_sortie/T₀_entrée` | `M` par l'EDO | `M` exact | écart |
|---|---|---|---|
| 1,5 | 0,395047597 | 0,395047597 | **9,8e-15** |
| 2,0 | 0,501618038 | 0,501618038 | 7,8e-13 |
| 2,5 | 0,648686082 | 0,648686082 | 1,3e-11 |
| 2,87 | 0,923399529 | 0,923399530 | 3,8e-11 |

L'écart **croît en approchant du blocage**, et c'est attendu : c'est la
singularité qui se rapproche. À citer dans le test plutôt qu'à masquer par une
tolérance unique.

**Contre l'isentropique** (section variable, sans chaleur, `A_out/A_in = 2,5`) :
branche subsonique **1,5e-12**, branche supersonique **1,2e-12**.

**Chaleur maximale** à `M_in = 0,3`, `T₀_in = 300 K` : `T₀* = 864,9 K`, soit
`q_max = 567,4 kJ/kg` et un rapport `T₀` maximal de **2,8830**.

---

# SCOPE PRINCIPAL

Chaque étape s'arrête sur un critère numérique vérifiable. Ne pas passer à la
suivante avant que le critère soit atteint.

### 1. Squelette, configuration, CLI minimal

**À faire.** `main.py`, `__init__.py`, `__main__.py`, `.gitignore`,
`requirements.txt` avec sa bannière commentée. `config.py` : deux dataclasses
gelées et leur exception `CaseError`.

```python
@dataclass(frozen=True)
class FlightCase:          # ce que voit le moteur
    mach: float = 3.0
    altitude: float = 11_000.0
    gamma: float = 1.4
    R: float = 287.05

@dataclass(frozen=True)
class EngineCase:          # ce que le moteur fait
    T04: float = 2000.0    # temperature de fin de combustion, K
    fuel: str = "kerosene"
    eta_b: float = 0.98
    pi_d: float | None = None   # None = correlation MIL-E-5008B
    pi_b: float = 0.95
    pi_n: float = 0.98
```

Validation à la construction : `mach > 0`, `0 ≤ altitude ≤ 32 000`,
`T04 > 0`, `0 < η_b ≤ 1`, `0 < π ≤ 1` pour chaque rendement de pression,
carburant présent dans le registre. Les deux types sont **hachables**, comme le
`RunCase` de `chaos-attractors` : c'est ce qui permettra de mémoriser un point
de fonctionnement dans la boucle de couplage de l'étape 9 sans sérialisation.

**Vérification.** `test_config.py` : chaque champ hors bornes lève `CaseError`
avec un message qui **nomme le champ et rappelle ses bornes** ; les objets sont
immuables (`FrozenInstanceError`) ; `hash()` fonctionne et deux cas identiques
ont le même hachage ; `python3 main.py --version` sort en 0 ;
`test_architecture.py` vérifie dans un **sous-processus** que `config` s'importe
sans tirer NumPy.

**Fait.** 64 tests en 1,6 s. `test_config.py` (28), `test_fuels.py` (13),
`test_cli.py` (16), `test_architecture.py` (5, dont les deux contrôles
d'importation en sous-processus). `--check` vérifie les dépendances, le dossier
de sortie, les deux niveaux de cloisonnement et la construction des cas, en
moins d'une seconde.

**`fuels.py` est arrivé à l'étape 1 et non à l'étape 2, et c'est un écart
raisonné.** Le plan reprenait le motif de `chaos-attractors`, où la validation
se coupait en deux — contrôles autonomes dans `__post_init__`, contrôles
dépendant du registre dans un `bind()` différé. Cette coupure y était imposée
par une raison précise : le registre des systèmes tirait NumPy, donc la
configuration ne pouvait pas le consulter. **Ici la raison n'existe pas** : un
carburant est un nom et deux nombres, donc de la bibliothèque standard pure.
`config.py` valide donc le carburant **à la construction**, comme le texte de
l'étape le demandait, et il n'y a pas de `bind()` à écrire ni à se rappeler
d'appeler.

**Le cloisonnement se lit à deux niveaux, et c'est nouveau par rapport aux
projets précédents.** `FORBIDDEN_IN_CORE` interdit SciPy et Matplotlib à tout
le noyau — NumPy y est légitime, `thermo` en aura besoin. Mais
`STDLIB_ONLY_MODULES` ajoute une contrainte **plus forte sur `config` et
`fuels` seuls** : pas de NumPy non plus. Ce sont les deux modules qu'un script
d'optimisation extérieur appellera en boucle pour construire des milliers de
cas, et l'importation de NumPy y serait payée pour rien. Les deux niveaux sont
vérifiés dans un interpréteur neuf, par `--check` et par un test.

**Trois décisions de conception prises en écrivant.**

*Deux cas plutôt qu'un.* `FlightCase` (Mach, altitude, gaz) et `EngineCase`
(T₀₄, carburant, rendements) sont séparés parce que le balayage de l'étape 4
fait varier le premier à moteur fixe, et l'étude de sensibilité aux pertes fait
varier le second à vol fixe. Un objet unique obligerait à recopier la moitié
des champs à chaque variation.

*`cp` est déduit, pas donné.* `cp = γR/(γ−1)` est une identité ; laisser
fournir les trois permettrait d'en donner trois incohérents. C'est une
propriété, pas un champ.

*`pi_d = None` veut dire « la corrélation », pas « pas de perte ».* La
récupération de pression du diffuseur dépend du **Mach de vol**, qui appartient
à l'autre cas : la seule façon honnête de ranger une valeur qui dépend d'un
objet qu'on n'a pas est de ne pas la ranger. `inlet_recovery(mach)` applique
MIL-E-5008B quand le champ est `None`, et un test vérifie les trois valeurs du
plan : **0,9250 à M₀ = 2, 0,8088 à M₀ = 3, 0,5126 à M₀ = 5**.

**Deux tests qui ont failli manquer.** Le premier vérifie que les **bornes
elles-mêmes** sont acceptées — `altitude = 0` et `altitude = 32 000` —, parce
qu'une borne exclue par un `<` au lieu d'un `<=` est un bug qu'on ne voit
jamais. Le second vérifie qu'un rendement **égal à 1** passe : le cas idéal qui
sert d'oracle à l'étape 3 met `π_d = π_b = π_n = η_b = 1`, et les refuser
rendrait l'oracle inatteignable.

**Et une contre-épreuve sur `--check` lui-même** : en remplaçant `FlightCase`
par une sous-classe qui accepte tout, `check_config` doit **échouer**. Sans
elle, la ligne « refus des cas absurdes : 7 cas testés » ne prouverait rien —
elle serait verte même si la validation ne validait rien.

**Bibliothèques.** stdlib.

### 2. Thermodynamique : les relations qui serviront d'oracles

**À faire.** `atmosphere.py` : atmosphère standard ISA par couches (troposphère
à gradient −6,5 K/km, stratosphère isotherme puis à gradient positif), jusqu'à
32 km. `thermo.py` : relations isentropiques `T₀/T`, `p₀/p`, `ρ₀/ρ`, `A/A*`,
vitesse du son, et l'inversion `M(A/A*)` sur chacune des deux branches. Choc
normal : `M₂(M₁)`, `p₂/p₁`, `p₀₂/p₀₁`.

`fuels.py` : un dataclass par carburant (`kerosene`, `hydrogen`), avec PCI et
rapport stœchiométrique, dans un petit registre — même motif que le registre de
systèmes de `chaos-attractors`, pour que le second carburant coûte trois lignes.

**Vérification.** `test_thermo.py`, et c'est ici que se construit la confiance :

- *valeurs tabulées* : `A/A*` vaut 1,0 à M = 1 exactement, 1,6875 à M = 2,
  4,2346 à M = 3 (tables standard, γ = 1,4) à 1e-9 ;
- **aller-retour exact** : `M(A/A*(M))` redonne `M` à **1e-12** sur les deux
  branches, pour 100 valeurs de M tirées dans [0,05 ; 0,95] et [1,05 ; 6]. Ce
  test seul ferme l'inversion, qui est le seul endroit du module où une
  bissection peut se tromper de branche ;
- *choc normal* : la relation est **involutive au sens de Prandtl**,
  `M₁ M₂ ... ` — plus simplement, `M₂(M₁) < 1 < M₁`, `p₀₂/p₀₁ < 1` strictement
  pour `M₁ > 1`, et `M₂ → 0,378` quand `M₁ → ∞` (limite analytique
  `√((γ−1)/2γ)`), à 1e-6 ;
- *choc de force nulle* : à `M₁ = 1`, toutes les relations doivent rendre
  l'identité **exactement** (`p₂/p₁ = 1`, `p₀₂/p₀₁ = 1`, `M₂ = 1`) ;
- *atmosphère* : `T`, `p` aux altitudes de rupture des couches (11, 20, 32 km)
  contre les valeurs ISA publiées, à 0,1 % ; continuité de `T` et `p` aux
  jonctions à 1e-12 ;
- *carburants* : le registre voit un carburant enregistré **dans le fichier de
  test**, sans qu'une ligne du paquet change.

**Fait.** 113 tests en 1,5 s, dont 35 pour `thermo.py` et 14 pour
`atmosphere.py`. **Le critère de l'étape est atteint mille fois mieux que
demandé : l'aller-retour `M → A/A* → M` est exact à 6,2e-15 en subsonique et
1,6e-15 en supersonique**, sur 200 valeurs tirées au hasard dans chaque
intervalle, là où le plan demandait 1e-12.

**Quatre valeurs de `A/A*` sont des rationnels exacts pour γ = 1,4**, ce qui a
permis de les tester à **1e-14** au lieu des cinq décimales d'une table :

```
A/A*(1/2) = 2 (7/8)^3   = 1,33984375      A/A*(2) = (1/2)(3/2)^3 = 1,6875
A/A*(1)   = 1                             A/A*(3) = (1/3)(7/3)^3 = 343/81
                                          A/A*(5) = (1/5) 5^3    = 25
```

**Le conditionnement près du col est intrinsèque, et il est mesuré plutôt que
masqué.** `A/A*` a un **minimum** en M = 1 : elle y est plate, et son
développement s'écrit `A/A* ≈ 1 + (γ+1)/4 (M−1)²`. Une erreur ε sur `A/A*`
donne donc une erreur en `√ε` sur M, et **aucun algorithme n'y peut rien** —
l'information n'est pas dans le nombre. Mesuré :

| M | 1,0001 | 1,001 | 1,01 | 1,05 |
|---|---|---|---|---|
| `A/A* − 1` | 8,3e-9 | 8,3e-7 | 8,3e-5 | 2,0e-3 |
| erreur sur M | **1,1e-12** | 2,0e-13 | 3,3e-14 | 2,9e-15 |

Conséquence écrite dans le module : **le dimensionnement d'un col ne passera
jamais par cette inversion**. C'est la section qui se déduit du Mach, et pas
l'inverse — l'étape 5 en tiendra compte.

**La bissection maison vaut `brentq`, et c'est mesuré.** SciPy est réservé à
`solver1d.py`, donc l'inversion est écrite à la main — bissection vectorisée,
80 itérations. Le test la compare à `scipy.optimize.brentq` réglé à la
précision machine : **écart maximal 1,0e-15**. On perd une ligne de code et on
gagne un arbitre indépendant.

**Deux contrôles du choc droit ne servent à aucune des formules du module**, et
ce sont donc de vrais tests : la température totale traverse le choc
**inchangée** (`T02/T01 = 1,000000000000000`) et le flux de masse aussi
(`ρ₂u₂/ρ₁u₁ = 1,000000000000000`), pour cinq Mach amont. La limite du choc fort
converge en 1/M₁² vers `√((γ−1)/2γ) = 0,377964` : écart 9,7e-5 à M₁ = 100,
9,7e-7 à M₁ = 1 000. Et le **choc de force nulle est exactement l'identité** —
les cinq rapports valent `1.0` au bit près à M₁ = 1, pas 0,9999999.

**Un choc subsonique est refusé au nom du second principe.** Les formules
donneraient un résultat, algébriquement valide, avec `p02/p01 > 1` —
c'est-à-dire un choc de **détente**, qui n'existe pas. Rendre un nombre
laisserait croire le contraire.

**L'atmosphère tombe à 3,7e-6 des valeurs publiées**, là où le plan demandait
1e-3, sur les quatre altitudes de rupture de couche. Deux décisions : les
pressions de base sont **calculées de proche en proche** et non tabulées — une
table serait une seconde source de vérité, et c'est là que naissent les
incohérences de sixième décimale ; et la couche isotherme utilise
l'**exponentielle** et non la forme en puissance, qui donnerait 0/0 à gradient
nul. Ce n'est pas sa limite numérique, c'est sa limite mathématique.

**Un test a été écrit faux, et sa correction vaut d'être notée.** Premier jet
de la continuité de pression : « `p(z−ε)/p(z+ε) = 1` à 1e-12 près ». Il
échouait à **3,15e-10** — non pas parce que le modèle saute, mais parce que la
pression **varie** entre les deux points : le gradient hydrostatique vaut
`dp/p = −g dz/(RT)`, soit 1,577e-4 par mètre à 216,65 K, donc exactement
3,15e-10 sur les 2 µm d'échantillonnage. Le test compare maintenant l'écart
mesuré à l'écart **attendu**, ce qui vérifie deux choses au lieu d'une : pas de
saut, et le bon gradient local. Un second test vérifie la continuité au sens
strict, sans échantillonnage, en évaluant la formule de la couche du dessous à
la jonction.

**Une erreur de docstring corrigée par le test qui l'accompagnait.** Le module
annonçait qu'utiliser la température totale au lieu de la statique surestime la
vitesse « de 25 % à M = 3 ». C'est faux : la vitesse va en `√T`, donc l'erreur
vaut `√(T0/T) = √2,8 = 1,673`, soit **67 %**. Le chiffre est maintenant celui
que le test mesure.

**`--check` porte les deux critères de l'étape** : l'atmosphère contre les
valeurs publiées et l'aller-retour d'inversion sur les deux branches, dans le
même esprit que le contrôle de jacobienne de `chaos-attractors`. Coût : 0,3 s.

**Bibliothèques.** NumPy ; SciPy dans les tests uniquement, pour vérifier
l'inversion par `brentq` indépendamment.

### 3. Modèle 0D : le cycle station par station

**À faire.** `cycle0d.py`. Une fonction par station plutôt qu'un bloc
monolithique — `inlet()`, `combustor()`, `nozzle()` — et un `Cycle` gelé qui
porte le résultat : `T0` et `p0` aux stations 0, 2, 4, 9, plus `f`, `F/ṁ`, `S`,
`M9`, `V9`, `η_th`, `η_p`, `η_o`.

Numérotation retenue, et elle doit être écrite dans le module : **0** amont,
**2** sortie diffuseur, **4** sortie chambre, **9** sortie tuyère. Les stations
1, 3 et 8 des normes n'existent pas dans ce modèle et ne sont pas inventées.

Deux points de méthode :

- `f` sort du **bilan d'énergie**, `f = cp(T₀₄ − T₀₂)/(η_b·PCI − cp·T₀₄)`, et
  non d'une richesse imposée. Le dénominateur s'annule quand
  `T₀₄ = η_b·PCI/cp ≈ 41 700 K` — inatteignable, mais le contrôle est écrit
  parce qu'un signe faux ailleurs le ferait apparaître ;
- la tuyère détend **jusqu'à `p₉ = p₀`**, ce qui définit `M₉` par le rapport de
  pression totale. Une tuyère sous-détendue est en scope optionnel.

**Vérification.** `test_cycle0d.py` :

- **le critère de l'étape** — *le cas idéal contre sa forme fermée* : avec
  `π_d = π_b = π_n = η_b = 1`, le calcul station par station doit rendre
  `M₉ = M₀` et `F/ṁ = V₀(√(τ_λ/τ_r) − 1)` à **1e-12**. Mesuré sur le
  prototype : **2,3e-13** à M₀ = 2, 3 et 4. Ce test valide d'un coup les quatre
  stations, le bilan d'énergie et la détente ;
- *monotonies* : `T₀₂` croît avec `M₀` ; `f` **décroît** avec `M₀` à `T₀₄` fixé ;
  `π_d` décroît. Trois tendances, trois signes, et aucune ne demande de valeur
  tabulée ;
- *limite thermique* : à `T₀₄` fixé, `f → 0` quand `T₀₂ → T₀₄`, soit
  `M₀ = 6,42` pour `T₀₄ = 2 000 K` (analytique). Le modèle doit y rendre `f ≤ 0`
  détecté et **refusé** avec un message qui donne le Mach limite, pas une
  poussée négative ;
- *cohérence des rendements* : `η_o = η_th · η_p` à 1e-12, les trois étant
  calculés séparément. C'est une identité, donc un vrai contrôle ;
- *bilan d'énergie global* : `(1+f)·cp·T₀₉ − cp·T₀₀ = f·η_b·PCI` à 1e-10, en
  repartant des températures rendues et non des intermédiaires.

**Fait.** 141 tests en 1,6 s, dont 28 pour le cycle. **Le critère de l'étape est
atteint : le calcul station par station retombe sur la forme fermée à
2,3e-13 en absolu sur la poussée spécifique, soit 5,9e-16 en relatif**, pour six
Mach de vol entre 1,5 et 6. `M9 = M0` y est exact à 1e-12, et `f`, `S` et les
trois rendements suivent.

**Le cas de référence, retrouvé au chiffre près.** Idéal à M₀ = 3 :
`F/ṁ = 777,3 N·s/kg`, `S = 44,15 mg/(N·s)`, `Isp = 2 309 s`, `η_th = 0,6429`.
Réel : `f = 0,03505`, `F/ṁ = 738,97`, `S = 47,44`, `M9 = 2,812`,
`η_th = 0,5882`. Le tableau du plan est reproduit à la dernière décimale.

**Une affirmation du plan était fausse, et le test l'a corrigée.** La docstring
annonçait que `η_th = 1 − 1/τ_r` n'était vraie qu'à la limite `f → 0`. Mesuré :
l'égalité tient à **5,5e-17** avec `f = 0,034`. L'algèbre confirme —

```
(1+f) V9² − V0² = 2 cp T0 (τ_r − 1) [(1+f) τ_λ/τ_r − 1]
f · PCI         = cp T0 [(τ_λ − τ_r) + f τ_λ]
```

les crochets sont proportionnels, `f` se simplifie, et il reste `(τ_r−1)/τ_r`.
C'est donc une **identité exacte**, donc un oracle de plus, et le test l'exige
maintenant à 1e-15 en vérifiant au passage que `f` n'est pas nul.

**Le rendement propulsif, lui, dépend bien de `f`.** La forme classique
`2/(V9/V0 + 1)` néglige le débit de carburant : écart mesuré **2,1 % à M₀ = 2,
2,6 % à M₀ = 3, 3,2 % à M₀ = 4**. C'est la forme complète qui est calculée, et
l'écart est testé pour qu'on ne confonde pas les deux — le `η_p = 0,710` du
tableau du plan était la forme approchée ; la forme complète donne **0,7287**.

**Les trois rendements sont définis par des PUISSANCES**, ce qui rend
`η_o = η_th · η_p` vraie *par construction* plutôt que par chance : puissance
carburant `f·PCI`, gain d'énergie cinétique, puissance propulsive `F·V0`.
Mesuré : écart nul ou 1,1e-16. Et le bilan d'énergie global, refait depuis les
températures **rendues**, tombe à **3,2e-16** pour 1e-10 demandé.

**Quatre refus plutôt que quatre résultats douteux.**

*La limite thermique* est analytique — `M_lim = √(2/(γ−1)·(T₀₄/T₀ − 1))` = **6,4154**
à T₀₄ = 2 000 K. À M₀ = 6,42, le modèle refuse en donnant le Mach limite, au
lieu de rendre une poussée négative.

*La richesse* : le bilan d'énergie ne sait pas qu'il y a une limite. À M₀ = 2,
elle passe 1 vers **T₀₄ = 3 050 K**, et au-delà le modèle sort de son domaine —
sans chimie, une richesse de 1,1 ne veut rien dire.

*Une `T₀₄` au-delà de ce que le carburant peut donner* (41 700 K pour le
kérosène) : inatteignable, mais un signe faux ailleurs le ferait apparaître, et
un `f` négatif silencieux serait pire.

*Une tuyère qui ne peut plus détendre* : quand les pertes dépassent la
compression dynamique, `p09 < p0` et il n'y a pas de détente. Mesuré à M₀ = 0,6
avec `π_b = 0,5`.

**Une décision de nommage qui a un fond.** Quatre stations seulement — 0, 2, 4,
9. Les normes en numérotent davantage (1, 3, 8) ; ce modèle ne les distingue
pas, donc il ne les nomme pas. Une station sans équation propre est une station
de trop, et la numéroter laisserait croire qu'on sait ce qui s'y passe. Le test
vérifie qu'appeler `station("3")` échoue.

**Bibliothèques.** NumPy.

### 4. Validation 0D contre les ordres de grandeur publiés

**À faire.** Un module de balayage — `sweep_mach()` — qui produit le tableau de
la section « Valeurs de référence » et le compare aux fourchettes publiées.
Aucune fonction nouvelle de physique : cette étape **mesure**, elle n'ajoute
rien.

**Vérification.** `test_validation0d.py` :

- **le critère de l'étape** : sur `M₀ ∈ [2 ; 4]`, la poussée spécifique tombe
  dans **[600 ; 900] N·s/kg** et l'impulsion spécifique dans **[1 500 ; 2 500] s**.
  Mesuré : 762, 774, 739, 670, 576 N·s/kg et 1 920 à 2 150 s ;
- *maximum de poussée spécifique* : il existe, et il est entre `M₀ = 2` et
  `M₀ = 3`. Mesuré : **2,5**. Le test cherche l'argmax sur une grille fine
  plutôt que de vérifier une valeur, parce que la position dépend de `T₀₄` ;
- *rendement thermique croissant* : `η_th(M₀)` est strictement croissante sur
  [1,5 ; 5]. Mesuré : 0,281 → 0,710 ;
- *contre-épreuve* : à `T₀₄` doublé, la poussée spécifique doit croître partout
  et le maximum se déplacer vers les grands `M₀`. Un modèle qui ne bougerait pas
  serait insensible à son paramètre principal.

**Fait.** 160 tests en 2,0 s, dont 19 pour la validation. Les trois fourchettes
publiées sont retrouvées, **et trois affirmations de cette étape du plan étaient
fausses** — c'est le propre d'une étape de validation que de le découvrir.

**1. La bande de poussée ne tient pas jusqu'à M₀ = 4, et le plan le disait
lui-même sans le voir.** Il annonce `F/ṁ ∈ [600 ; 900]` sur `M₀ ∈ [2 ; 4]`, puis
liste parmi ses valeurs mesurées **576 N·s/kg à M₀ = 4** — hors de sa propre
bande. Le modèle n'a pas tort : à `T₀₄ = 2 000 K`, un statoréacteur à Mach 4 n'a
plus assez d'écart de température pour brûler (`f` tombe de 0,0405 à 0,0274, un
tiers de moins). La bande tient donc sur **[2 ; 3,5]**, et plutôt que de
rétrécir l'intervalle en silence, un test montre que **porter `T₀₄` à 2 200 K
ramène M₀ = 4 dedans** (671,5 N·s/kg), et à 2 400 K donne 763,7. Un
statoréacteur de Mach 4 se dessine plus chaud, et le modèle le dit.

**2. Le maximum de poussée spécifique est à M₀ = 2,36, pas 2,5.** Le 2,5 du plan
venait d'un balayage par pas de 0,5. Sur grille fine (0,01), le maximum vaut
**776,5 N·s/kg à M₀ = 2,36**. La conception du test l'avait anticipé — « le test
cherche l'argmax sur une grille fine plutôt que de vérifier une valeur » — et
c'est ce qui a permis de le voir.

La contre-épreuve fonctionne : le maximum se déplace vers les grands Mach quand
la chambre chauffe, **2,28 → 2,36 → 2,53 → 2,68** pour `T₀₄` = 1 800, 2 000,
2 500 et 3 000 K.

**3. Le rendement thermique n'est PAS strictement croissant, contrairement à ce
qu'annonçait le plan.** Il passe par un **maximum de 0,7129 à M₀ = 4,79**, puis
s'effondre à 0,464 à M₀ = 6. La cause est mesurable et le contraste l'identifie :
avec le moteur **idéal**, `η_th = 1 − 1/τ_r` (identité exacte de l'étape 3) croît
strictement de 0,310 à 0,891 sur toute la plage. Toute la non-monotonie vient
donc des **pertes** — la récupération du diffuseur tombe de 0,925 à M₀ = 2 à
0,513 à M₀ = 5, et au-delà de M₀ = 4,8 cette perte l'emporte sur le gain de
compression dynamique. Le test affirme désormais la croissance sur le domaine
**utile** [1,5 ; 4,5] et *mesure* le maximum au lieu de le nier.

**Un quatrième test a été écrit à l'envers, et l'erreur est la plus instructive
des quatre.** Premier jet : « M₉ > M₀, sinon il n'y aurait pas de poussée ».
Faux — mesuré, **M₉ est inférieur à M₀ partout** sur [1,5 ; 5] pour le moteur
réel (4,391 contre 5,0), et la poussée est pourtant positive partout. La raison :
les gaz sortent chauds, donc leur vitesse du son vaut **1,4 à 2,6 fois** celle de
l'air ambiant, et `V₉ = M₉a₉` dépasse `V₀ = M₀a₀` malgré un Mach plus faible.
Dans le cas idéal c'est encore plus net : `M₉ = M₀` exactement, et la poussée
vaut quand même 777 N·s/kg. **La poussée vient de la vitesse, pas du Mach**, et
c'est maintenant ce que le test dit.

**Le balayage garde les points refusés** au lieu de les écarter en silence : un
balayage jusqu'à M₀ = 7 en retient 50 et en refuse 6, le premier à M₀ = 6,5 avec
le message de la limite thermique. Savoir *où* le balayage s'arrête est une
information, pas un incident.

**Bibliothèques.** NumPy.

### 5. Géométrie du conduit

**À faire.** `duct.py` : une classe `Duct` qui porte `A(x)`, `dA/dx` et le
découpage en zones (diffuseur, chambre, tuyère convergente, tuyère divergente).
La section est décrite par des **raccords lisses** — polynômes de degré 3 à
dérivée imposée aux jonctions — et non par des segments linéaires : `dA/dx`
entre directement dans l'EDO, et une discontinuité de pente y ferait un saut de
`dM²/dx` que l'intégrateur paierait en pas rejetés.

Un constructeur `from_cycle(cycle, M_combustor)` qui **dimensionne** le conduit
depuis le résultat 0D : section de capture par le débit, section de chambre par
le Mach visé, col de tuyère par la condition de blocage, section de sortie par
`M₉`. C'est le pont entre les deux modèles.

**Vérification.** `test_duct.py` :

- *continuité* : `A(x)` et `dA/dx` sont continues aux quatre jonctions à
  1e-12 — et c'est un vrai test, parce que le raccord cubique est le seul
  endroit du module où une erreur d'indice se cache ;
- *dérivée* : `dA/dx` analytique contre différences finies centrées, écart
  relatif < 1e-8 sur 200 points. Même motif que la jacobienne de
  `chaos-attractors` : une dérivée fausse ne se voit pas sur le tracé de `A(x)` ;
- *monotonies attendues* : `A` décroît dans le diffuseur, est constante dans la
  chambre (à 1e-14), décroît jusqu'au col puis croît ;
- *dimensionnement* : `from_cycle` place le col à la section qui bloque
  exactement le débit, vérifié par `ṁ = ρ*u*A*` à 1e-10 ;
- *aires positives* : aucune section nulle ou négative, sur 1 000 points.

**Fait.** 190 tests en 1,9 s, dont 30 pour la géométrie. **Le critère de l'étape
est atteint : `ṁ = ρuA` à 1e-9 aux sections de capture et de chambre, et le col
bloque exactement le débit chaud à 1e-10.** Le rapport `A₉/A_col = 3,5419`
égale `A/A*(M₉ = 2,812)` à 1e-12. Dimensions du cas de référence à 10 kg/s :
capture **310,4 cm²**, chambre **268,6**, col **179,3**, sortie **635,1**, sur
1,50 m.

**Les raccords cubiques donnent plus que la continuité demandée.** Le plan
demandait `A` et `dA/dx` continues à 1e-12 ; le `smoothstep` `3s² − 2s³`, dont
la dérivée s'annule aux deux bouts de chaque zone, rend `dA/dx` **exactement
nulle à toutes les jonctions** — donc C¹ partout sans rien imposer, et la
chambre à section constante s'y raccorde sans rupture. La dérivée analytique
tombe à **1,5e-10** des différences finies, pour 1e-8 demandé.

**Le dimensionnement respecte la leçon de l'étape 2** : il ne passe jamais par
l'inversion de `A/A*`. Le col se déduit du **débit bloqué**, qui est analytique
(`Γ = 0,0404149`), et la section de sortie du Mach de sortie. L'inversion ne
sert qu'à *lire* le Mach de chambre, où `A/A*` vaut 1,5 et où elle est
parfaitement conditionnée.

**Un piège a été trouvé ici, et il pèse plus que ce que l'étape 9 cherchera à
corriger.** La relation de Rayleigh, qui servira d'oracle au solveur 1D, suppose
le débit **constant** : elle ne connaît qu'un apport de chaleur. Le moteur, lui,
ajoute 3,5 % de masse. Mesuré sur le cas de référence, chambre dimensionnée à
`M = 0,20` :

| | `A/A*` | `M₄` |
|---|---|---|
| sans l'apport de masse | 1,5146 | **0,4250** |
| avec l'apport de masse | 1,4633 | **0,4442** |
| Rayleigh (débit constant) | 1,5148 | 0,4249 |

Sans l'apport, l'accord avec Rayleigh est à **1e-4** — la géométrie est donc
juste, et c'est bien la masse qui fait la différence. Avec, l'écart atteint
4,5 %. Or la correction que l'étape 9 appliquera — passer `π_b` de 0,95 supposé
à 0,928 calculé — ne vaut que **0,0132** sur le même Mach, contre **0,0192**
pour l'apport de masse. **Négliger la masse fausserait donc le couplage plus
que ce qu'il prétend corriger**, et l'étape 6 devra dire explicitement lequel
des deux problèmes elle intègre. Un drapeau `include_fuel_mass` existe pour que
la comparaison à Rayleigh se fasse à débit constant.

**Deux tests ont été écrits contre le mauvais repère.** Le premier comparait
l'apport de masse à un balayage de `π_b` de 0,95 à 0,90 — mais cet intervalle
large (0,0314) n'est pas celui que le couplage corrigera ; contre le bon
intervalle, la conclusion s'inverse et tient. Le second cherchait le minimum de
section sur une grille de 1 000 points et le trouvait **2,5e-8 au-dessus** du
col : c'est l'échantillonnage qui rate `x = 1,15`, pas le conduit. Le test
vérifie maintenant les deux choses vraies — aucun point sous le col, et le col
atteint dès qu'on l'échantillonne.

**Le col est un minimum, donc `dA/dx` y est nul — et c'est la seconde
singularité du projet.** L'équation du 1D porte un `1 − M²` au dénominateur ;
au col, le terme `dA/A` s'annule aussi. Le passage transsonique est donc un
0/0, dont la levée est le vrai travail de l'étape 6. Ce module ne le résout
pas : il fournit la géométrie et signale où elle pose le problème.

**Bibliothèques.** NumPy.

### 6. Le solveur 1D

**À faire.** `solver1d.py`. L'EDO en `M²` avec ses deux coefficients
d'influence, intégrée par `solve_ivp` (`RK45`, `rtol = 1e-10`, `atol = 1e-12`,
`dense_output = True`). Le profil `dT₀/dx` est fourni par l'appelant : nul
partout sauf sur la zone de chambre, où il vaut la constante qui porte `T₀` de
`T₀₂` à `T₀₄`. Reconstruction algébrique de `T`, `p`, `u`, `ρ` depuis `M²`,
`T₀(x)`, `A(x)` et le débit.

**Vérification.** `test_solver1d.py`, du plus exact au moins exact :

- **le critère de l'étape, premier volet** — *contre Rayleigh* : section
  constante, chaleur répartie, comparaison à la forme fermée. Mesuré :
  **9,8e-15** à `T₀r = 1,5`, 7,8e-13 à 2,0, 1,3e-11 à 2,5, 3,8e-11 à 2,87.
  Tolérance retenue : 1e-10, et le test **vérifie que l'écart croît** avec le
  rapport de température — c'est la singularité qui approche, et le masquer
  sous une tolérance unique serait perdre l'information ;
- **le critère de l'étape, second volet** — *contre l'isentropique* : section
  variable, chaleur nulle, comparaison à `A/A*` sur les **deux branches**.
  Mesuré : 1,5e-12 en subsonique, 1,2e-12 en supersonique ;
- **le contrôle indépendant** — *quantité de mouvement* : la fonction
  d'impulsion doit vérifier `ΔF = ∫ p dA` à 1e-8 en relatif. Rien dans
  l'intégration ne le force : l'EDO n'intègre que `M²`, et `p` est reconstruite
  par la masse ;
- *conservation de la masse* : `ρuA` constant le long du conduit à 1e-12, ce
  qui est vrai par construction et donc un test de la **reconstruction**, pas
  de l'EDO ;
- *chaleur nulle et section constante* : tout doit rester rigoureusement
  constant, à la précision machine. Test bête, et le premier à casser si un
  terme source fuit ;
- *reproductibilité* : deux exécutions identiques donnent des tableaux
  **bit à bit** identiques.

**Fait.** 220 tests en 4,4 s, dont 28 pour le solveur. **Les deux volets du
critère sont atteints, et l'ordre de grandeur est meilleur que demandé.**

*Contre Rayleigh* (section constante, chaleur répartie, `M_in = 0,3`) :

| `T₀_sortie/T₀_entrée` | 1,5 | 2,0 | 2,5 | 2,87 |
|---|---|---|---|---|
| écart sur M | 2,7e-12 | 7,7e-12 | 1,8e-11 | **3,8e-11** |
| écart sur `π` | 5,5e-12 | 1,0e-11 | 1,3e-11 | 4,4e-12 |

L'écart **croît** avec le rapport de température, et le test l'exige : c'est la
singularité qui se rapproche, le dernier point étant à 99,5 % du rapport de
blocage. La perte de pression totale tombe elle aussi sur la forme fermée à
1e-11 — première apparition du `π_b` autour duquel tournera l'étape 9.

*Contre l'isentropique* (section variable, chaleur nulle) : **2,0e-12** sur la
branche subsonique, **1,9e-11** sur la supersonique.

**Le contrôle indépendant de quantité de mouvement tombe à 1e-11 à 1e-13**, pour
1e-8 demandé — mais il a fallu deux corrections pour qu'il mesure ce qu'il
prétend mesurer.

*La normalisation.* Premier jet : écart rapporté au **saut** d'impulsion. Or à
section constante les deux membres valent zéro — l'impulsion de Rayleigh est
conservée, ce qui est d'ailleurs un contrôle exact de plus, mesuré à 1e-10 —
et diviser par le saut rendait un résidu de **1,00** pour un écart de 1e-10 :
un zéro divisé par zéro, déguisé en échec. Le dénominateur est maintenant
l'impulsion moyenne.

*La quadrature.* Premier jet : intégrale par trapèzes, résidu **2,5e-07** à
2 000 points. L'erreur des trapèzes va en `(dx)²`, soit exactement les 2,5e-7
observés à `dx = 5e-4` — c'était donc la **quadrature** qu'on mesurait, pas la
solution. Avec Simpson, le résidu tombe à 2,5e-13 et **ne dépend plus du nombre
de points**, ce qu'un test vérifie explicitement.

**Une troisième correction, sur la conservation de `p₀`.** Le test posait un
seuil absolu de 1e-10 sur la dérive de pression totale en écoulement
isentropique ; mesuré, elle vaut 4,6e-09. La cause est identifiée en faisant
varier la tolérance de l'EDO :

| `rtol` | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|
| dérive de `p₀` | 2,2e-07 | 4,6e-09 | 4,4e-11 |
| écart sur M | 2,1e-09 | 1,9e-11 | 3,6e-13 |

La dérive suit la tolérance **à la proportionnelle**, et vaut 100 à 1 000 fois
l'erreur sur M : `p₀` passe par `(1 + 0,2M²)^3,5`, qui amplifie. C'est donc de
l'erreur d'intégration et non un défaut de physique — et le test le prouve en
faisant varier `rtol` plutôt qu'en posant un seuil, ce qui distingue les deux
causes.

**Ce que le module intègre, et ce qu'il refuse.** À **débit constant** —
l'apport de masse du carburant n'a pas de coefficient d'influence ici, c'est ce
qui rend Rayleigh utilisable comme oracle exact, et l'étape 5 a mesuré ce que
ce choix coûte. Il **ne traverse pas M = 1** : un Mach d'entrée à 1 est refusé
en nommant la singularité, et le passage transsonique reste le travail de
l'étape 8. Et il **ne détecte pas encore le blocage** : la séparation d'avec
l'étape 7 est délibérée, puisqu'elle seule permet de mesurer ce que fait le
solveur non protégé.

**Les cas triviaux sont exacts** : sans chaleur ni variation de section, les
cinq grandeurs ont une variation **exactement nulle**, et le débit `ρuA` est
conservé à 8,9e-16. Le cloisonnement est vérifié dans les deux sens :
`solver1d` tire SciPy — et pas Matplotlib —, ce qui prouve que le cloisonnement
n'est pas obtenu en n'utilisant jamais la bibliothèque.

**Bibliothèques.** NumPy, SciPy.

### 7. Blocage thermique

**À faire.** Trois choses, et l'ordre compte :

1. `rayleigh.py` : `T0_max(M_in)` et `q_max(M_in, T0_in)` en forme fermée, donc
   **avant** toute intégration ;
2. dans `solver1d.py`, un **événement terminal** de `solve_ivp` posé à
   `M = 0,999`, avec `direction = −1` ;
3. une exception `ChokedFlow` qui porte l'abscisse du blocage, le Mach atteint
   et la chaleur maximale admissible, pour que le message dise quoi changer.

**Vérification.** `test_choking.py`, et c'est le fichier le plus important du
projet :

- **le critère de l'étape** : à `q/q_max = 1,05`, le solveur doit lever
  `ChokedFlow` en annonçant `x = 0,929` à ±0,01, et **surtout pas** rendre un
  profil. À `q/q_max = 0,999`, il doit rendre un profil valide, mesuré
  `M_sortie = 0,963` ;
- *la chaleur maximale est celle de Rayleigh* : `q_max` calculée par la forme
  fermée contre la plus grande chaleur pour laquelle l'intégration passe,
  cherchée par bissection. Accord à 1e-4 en relatif ;
- **la contre-épreuve, et elle est indispensable** : un RK4 à pas fixe écrit
  dans le fichier de test doit, lui, **franchir M = 1 sans rien dire**. Mesuré :
  franchissement à `x = 0,868` pour `q/q_max = 1,10`, à `x = 0,567` pour 1,50.
  Sans ce test, rien ne prouve que la protection protège de quelque chose.
  *Les valeurs de sortie que ce plan annonçait — `M = 1,122` et `M = 0,916` —
  n'existent pas : l'étape 7 a montré qu'elles dépendent entièrement du pas.
  Le test porte donc sur les trois faits reproductibles : l'abscisse de
  franchissement converge, la valeur finale non, et trois fois sur cinq elle
  est subsonique* ;
- **le sens de l'événement** : avec `direction = +1`, l'événement ne se
  déclenche jamais et `solve_ivp` échoue sur `Required step size is less than
  spacing between numbers`. Le test vérifie les deux sens et documente le
  message qu'on obtient quand on se trompe — c'est ce message qu'on lira dans
  six mois ;
- *le blocage dépend du Mach d'entrée* : le tableau `M_in → T₀max/T₀in` du
  prototype (21,4 à `M = 0,10`, 2,88 à `M = 0,30`) est retrouvé à 1e-6.

**Fait.** 245 tests en 16,7 s, dont 25 pour le blocage. **Le critère est
atteint : à 99,9 % du maximum le solveur rend un profil (M de sortie 0,9630), à
105 % il lève `ChokedFlow` en annonçant `x = 0,9289 m`** — et jamais l'inverse.
La limite trouvée par bissection sur l'intégration coïncide avec la forme
fermée à **1e-3**, et le tableau `M_in → T₀max/T₀in` est retrouvé à 1e-3
(21,378 / 9,808 / 5,762 / 3,894 / 2,883 / 1,890).

**La contre-épreuve donne un résultat PIRE que ce que le plan annonçait, et
c'est le meilleur résultat de l'étape.** Le plan disait, d'après le prototype :
« le RK4 naïf franchit M = 1 à x = 0,868 et finit à M = 1,122 ». La première
moitié est exacte et reproductible ; la seconde ne l'est pas du tout. Mesuré en
faisant varier le nombre de pas, à `q/q_max = 1,10` :

| n | 500 | 1 000 | 2 000 | 4 000 | 8 000 |
|---|---|---|---|---|---|
| abscisse de franchissement | 0,8680 | 0,8860 | 0,8680 | 0,8672 | 0,8674 |
| **M final** | **0,9264** | **0,7640** | **2,1088** | **1,2017** | **0,8390** |

L'abscisse de franchissement **converge** — avant la singularité, une
intégration reste une intégration. Ce qui suit ne converge pas du tout : cinq
nombres sans rapport, entre 0,76 et 2,11. Et **trois des cinq finissent
subsoniques**, donc le résultat faux ne se reconnaît même pas à son point
d'arrivée. Le `1,122` du plan était une valeur parmi d'autres, tirée au sort par
le pas de discrétisation.

Le solveur non protégé ne rend donc pas seulement un résultat faux : il rend un
résultat faux, d'allure ordinaire, et **non reproductible**. Les trois tests de
cette famille vérifient les trois choses séparément, et un quatrième vérifie que
sous la limite le RK4 naïf est correct — la contre-épreuve doit discriminer,
pas condamner en bloc.

**Le sens de l'événement est vérifié dans les deux sens, et l'erreur naturelle
est documentée.** On surveille un Mach qui *monte*, donc on écrit
`direction = +1` — mais c'est le signe de la dérivée de la **fonction
d'événement** qui compte, et elle, elle descend. Mesuré :

- `direction = −1` : arrêt propre à `x = 0,9289`, `success = True` ;
- `direction = +1` : **aucun événement**, `success = False`, et le message
  *« Required step size is less than spacing between numbers »* — qui ne parle
  pas de physique et qu'on interprète naturellement comme un problème de
  tolérance.

Un test vérifie de plus que les deux sens s'arrêtent **au même endroit** : le
mauvais sens n'intègre pas plus loin, il échoue au même point sans savoir dire
pourquoi. La différence n'est pas dans le calcul, elle est dans ce qu'on peut en
dire.

**Une subtilité de conception que le plan n'avait pas vue** : la fonction
d'événement doit changer de forme selon le côté d'où l'on approche. Écrite comme
la *marge* jusqu'au seuil — `(1−ε)² − M²` en subsonique, `M² − (1+ε)²` en
supersonique — elle est positive tant qu'on est du bon côté et décroissante en
s'en approchant **dans les deux cas**, ce qui donne `direction = −1` partout.

**Le message de refus dit quoi changer**, et lui aussi a demandé une
correction : il annonçait d'abord le rapport de température atteint *au point de
blocage*, qui vaut le maximum par construction — donc « 2,8830 là où on
n'autorise que 2,8830 ». Il annonce maintenant le rapport **demandé** au bout du
conduit : « porterait T₀ d'un facteur 3,0272 là où M = 0,3 n'en autorise que
2,8830, soit 1,05 fois trop ». Et le remède qu'il propose — ralentir
l'écoulement — est testé : le même dépôt de chaleur qui bloque à `M = 0,30`
passe à `M = 0,20`.

**Bibliothèques.** NumPy, SciPy.

### 8. Profils sur le statoréacteur complet

**À faire.** Assembler : géométrie de l'étape 5, solveur de l'étape 6,
protection de l'étape 7. `run_engine(flight, engine, duct)` qui intègre de
l'entrée à la sortie et rend un objet `Profile` portant `x`, `A`, `M`, `T`, `p`,
`u`, `ρ`, `T₀`, `p₀`.

Le diffuseur supersonique est traité par **un choc normal placé à l'entrée**,
puis intégration subsonique : c'est le modèle le plus simple qui soit honnête,
et il est cohérent avec `π_d`. Le choc n'est pas intégré, il est appliqué par
les relations de saut.

**Vérification.** `test_engine.py` :

- *le profil existe et est physique* : `M > 0`, `p > 0`, `T > 0` partout ;
  `T₀` constant hors de la chambre à 1e-10 ; `T₀` atteint `T₀₄` en fin de
  chambre à 1e-6 ;
- *le col est bien sonique* : `M = 1` au col à 1e-3, ou le col n'en est pas un ;
- *l'écoulement est subsonique dans la chambre et supersonique en sortie* —
  deux inégalités, mais ce sont celles qui font qu'un statoréacteur est un
  statoréacteur ;
- *pression de sortie* : `p₉ = p₀` à 1 % pour la géométrie dimensionnée par
  `from_cycle` — c'est ce que le dimensionnement visait ;
- *la poussée par la quantité de mouvement* : `F = ṁ₉u₉ − ṁ₀u₀ + (p₉ − p₀)A₉`
  calculée depuis le profil, contre `F/ṁ` du 0D. C'est le début de l'étape 9.

**Fait.** 281 tests en 24,1 s, dont 33 pour l'assemblage. `run_engine` vit dans
`coupling.py` — l'arborescence n'avait pas prévu de fichier pour lui, et
l'étape 9 s'y greffe naturellement. **Tous les critères du plan sont atteints** :
`M = 1,000000` au col, `T₀` constante hors chambre à **0 exactement**,
subsonique en chambre et `M₉ = 2,81` en sortie, `p₉/p_ambiant = 0,9951` (0,49 %,
sous le 1 % demandé), et la poussée par la quantité de mouvement retrouve celle
du 0D à **0,094 %**.

**Mais le résultat de l'étape n'est aucun de ces cinq-là. C'est que l'assemblage
a montré que le 0D posait mal la question :**

> **À géométrie fixée, le débit n'est pas un paramètre. C'est le col qui le
> fixe.**

> ⚠ **Corrigé après les quinze étapes.** Les 10,094 kg/s ci-dessous étaient un
> artefact du `(1+f)` posé à la main. Depuis que l'EDO porte le carburant, le
> point fixe rend **exactement 10,000000** en entrée. Le fait de fond — le débit
> est une inconnue — tient toujours. Voir « Après les quinze étapes ».

Un statoréacteur avale ce que son col laisse passer. Le débit est donc
l'inconnue d'une équation — `A*(fin de chambre) = A_col` — et pas une donnée.
La géométrie dessinée pour 10,000 kg/s en avale **10,094**. Et l'écart s'écrit
en forme fermée : égaler les deux lectures du col donne

```
ṁ_1D = ṁ_0D (1+f) π_b,réel / π_b,supposé = 10 × 1,0344 × 0,9265/0,95 = 10,094064
```

soit le débit mesuré **à 1e-6 près**. Les deux corrections vont en sens
contraire — +3,4 % de masse de carburant que le 0D comptait au col, −2,5 % de
perte de chambre qu'il sous-estimait — et leur net de +0,94 % n'est ni assez
grand pour se voir, ni assez petit pour être nul.

**Le passage transsonique : on ne lève pas le 0/0, on ne le rencontre pas.** Le
profil est calculé en trois morceaux dont un seul est intégré. La chambre passe
par l'EDO, parce que la chaleur y est répartie et que personne d'autre ne sait
faire. Le diffuseur et la tuyère sont **algébriques** : sans chaleur, la
solution de l'EDO *est* la relation de section — l'étape 6 l'a vérifié à
1e-12 — et la relation de section, elle, n'a pas de singularité en `M = 1`. On
traverse donc le col en forme fermée, sur les deux branches, sans jamais
approcher le point critique.

**Et le raccord tombe juste par construction** : le débit solution est
précisément celui pour lequel `A*(fin de chambre) = A_col`, donc le Mach que
l'EDO rend en fin de chambre et celui que `A(x)/A_col` donne au même endroit
sont le même nombre. Mesuré : **5,6e-17**. Rien n'est ajusté.

**Le plancher de précision au col est celui annoncé à l'étape 2, et il se
retrouve ici** : `M = 0,9999999877`, soit 1,2e-8. Un test l'*encadre* — moins de
1e-6, mais plus de 1e-12 — parce qu'un écart nul signalerait que le Mach est
imposé et non calculé.

**Un coefficient que j'avais récité au lieu de le mesurer.** J'ai d'abord écrit
`A/A* − 1 ≈ (γ+1)/4 · (M−1)²`, ce qui prédisait 1,05e-3 là où l'on mesure
8,92e-4. Le bon coefficient est **`2/(γ+1)`**, vérifié sur trois gamma : le
rapport `(1−M)²/(A/A*−1)` vaut 1,100 à γ = 1,2, 1,200 à 1,4 et 1,333 à 1,667,
c'est-à-dire `(γ+1)/2`. Et parce que le raccord `3s²−2s³` annule aussi `dA/dx`
au col, les deux effets se composent et `1 − M` redevient **linéaire** en
distance au col : 3,56e-3 / 1,78e-3 / 8,92e-4 / 4,46e-4 quand on divise `dx`
par deux à chaque fois.

**Ce que le plan proposait pour le diffuseur était faux.** Il annonçait « un
choc normal placé à l'entrée […] cohérent avec `π_d` ». Il ne l'est pas : à
`M₀ = 3` un choc droit rend **0,3283** là où la corrélation MIL-E-5008B du 0D
suppose **0,8088**, un facteur 2,46. Adopter ce modèle aurait coûté **21,5 %**
de poussée spécifique et contredit le cycle qu'il devait compléter — et il
aurait de plus exigé un diffuseur **divergent** d'un facteur 2,13 (pour ramener
`M = 0,475` à 0,20), alors que la géométrie de `from_cycle` converge. Le
diffuseur reste donc un **saut**, celui que le 0D définit ; c'est de la
compression externe, hors du conduit, et le 1D n'a rien à en dire. Trois tests
chiffrent ce qu'on a évité.

Cela rend visible une bizarrerie qu'il faut assumer plutôt que masquer : dans le
profil, le « diffuseur » **accélère** l'écoulement, de `M = 0,174` à `M = 0,202`,
parce que `from_cycle` relie la section de capture (310 cm²) à la chambre
(269 cm²) et que le conduit converge donc. La vraie compression est déjà tout
entière dans `π_d`. Le README doit le dire.

**Deux silences fermés en passant.**

`momentum_residual` rendait **1,6e-05** sur le profil complet là où chaque
tronçon pris à part donne 1e-11 à 4e-08. La formule de Simpson suppose un pas
constant, le profil recollé en a quatre différents, et l'on aurait lu une erreur
de **quadrature** comme une erreur de **solution**. Le module refuse désormais
le pas variable.

Et le test d'architecture ne regardait que les modules *nommés* : `coupling.py`
aurait pu importer SciPy sans que rien ne le signale. Il parcourt maintenant le
répertoire et exige que chaque module soit classé — noyau, couche 1D, ou
service. La contrainte ne va que dans un sens : un module déclaré mais pas
encore écrit est normal (`plots` et `report` sont les étapes 10 et 11), un
module écrit mais non classé ne l'est jamais.

**La recherche du débit ne pouvait pas être confiée à `brentq`**, et il a fallu
mesurer *quand* pour le dire honnêtement. Au-delà d'un certain débit la chambre
bloque thermiquement et l'intégration **lève** : une fonction qui lève n'a pas
de signe. La recherche bissecte donc à la main tant qu'une borne est bloquée —
une exception a un sens, « trop de débit » — puis passe la main à `brentq`. Ce
qui décide n'est **ni la température de fin de combustion ni la taille du col**
(testés de 2 000 à 4 000 K et de 1,0 à 0,4 fois : ils déplacent le plafond de
débit *et* le Mach d'entrée ensemble), mais le seul rapport
**`A_chambre/A_col`** : à 1,498 — le cas de référence — la phase prudente ne
sert jamais ; à 1,123 elle est indispensable ; à 1,000 la recherche refuse en
nommant les deux remèdes. Un test dit explicitement que sur le cas de référence
la garde ne se déclenche pas, sans quoi on croirait l'avoir testée.

Le passage en deux phases a par ailleurs rendu `run_engine` **5,3 fois plus
rapide** (1,775 s → 0,332 s) *et* plus précis (1e-12 → 1e-14), après avoir
mesuré que `n_points` ne coûte rien : le temps est dans `solve_ivp`, 27 ms
quelle que soit la sortie demandée.

**Enfin, un test dit ce qui n'est PAS une vérification.** `M₉` du 1D égale celui
du 0D à 0 près — non parce que le 1D a raison, mais parce que `from_cycle` a
dimensionné `A₉` pour ce Mach-là. C'est une identité. La vérification, c'est la
poussée : deux bilans qui ne partagent rien, l'enthalpie d'un côté et la
quantité de mouvement de l'autre, qui tombent à 0,094 %. Et l'écart de 7,4 %
qu'on observe si l'on oublie le `(1+f)` vaut exactement `f·u₉` — l'EDO conserve
la masse et ne peut structurellement pas porter ce terme, qui est remis à la
main dans `run_engine`.

**Bibliothèques.** NumPy, SciPy.

### 9. La boucle 0D ↔ 1D

**À faire.** `coupling.py` : `converge(flight, engine)` qui alterne 0D et 1D
jusqu'à ce que `π_b` ne bouge plus. Le 0D donne `T₀₄` et le débit, le 1D rend le
`π_b` réellement subi, on réinjecte, on recommence. Critère d'arrêt sur la
variation relative de `π_b`, plafond d'itérations, et un refus argumenté si le
1D déclare le point de fonctionnement bloqué.

**Vérification.** `test_coupling.py` :

- **le critère de l'étape** : la boucle converge en **moins de cinq itérations**
  à 1e-6 sur `π_b`, pour le cas de référence ;
- *le point fixe est un vrai point fixe* : repartir de la solution convergée
  donne la même chose à 1e-12 — l'itération est idempotente au point fixe ;
- **le résultat qui justifie le projet** : le `π_b` convergé **diffère** de la
  valeur supposée de 0,95. Mesuré sur le prototype par Rayleigh seul :
  0,9280 à `M_in = 0,20` et `T₀r = 3,3`, 0,8751 à `M_in = 0,25`. Le test vérifie
  que l'écart au 0,95 supposé dépasse 1 %, c'est-à-dire que la boucle **sert à
  quelque chose** ;
- *insensibilité au point de départ* : partir de `π_b = 0,90` ou de `π_b = 0,99`
  donne le même point fixe à 1e-8 ;
- *cas bloqué* : à `M_in = 0,30` et `T₀₄ = 2 000 K`, la boucle doit **refuser**
  en annonçant le blocage et le `T₀₄` maximal (1 749 K mesuré), et non converger
  vers un point de fonctionnement imaginaire.

**Fait.** 300 tests en 50,9 s, dont 19 pour la boucle. **Les cinq critères sont
atteints, et plus largement que demandé** : convergence en **5 appels** au 1D
pour un résidu de 8,6e-13 ; point fixe idempotent à **7,8e-14** (1e-12
demandé) ; `π_b` convergé à **0,921823** contre 0,95 supposé, soit **−2,97 %**
(plus de 1 % demandé) ; et l'insensibilité au départ tient aux **douze
chiffres** depuis 0,80, 0,90, 0,95 et 0,99, là où le plan se contentait de 1e-8.

**Mais la méthode que le plan décrivait n'aurait pas tenu le critère du plan.**
L'itération nue `π ← g(π)` converge **linéairement**, de rapport **0,172369** —
mesuré sur huit tours, stable à 1e-6 dès le sixième — et il lui faut **sept**
tours pour descendre à 1e-6, pas « moins de cinq ». La secante sur
`h(π) = g(π) − π` atteint le point fixe *exactement*, en cinq appels. Un test
mesure les sept tours de la méthode naïve plutôt que de les supposer : c'est ce
qui a motivé le changement.

**Le résultat structurel de l'étape : la géométrie est le SEUL chemin par lequel
la boucle boucle.** Les données du 1D dans la chambre sont `T₀₂`, `p₀₂`, `T₀₄`
et la géométrie ; aucune ne dépend du `π_b` supposé, qui n'agit que sur le
*dimensionnement*. Figez le conduit et `g` devient une **fonction constante** :
`π_b(0,95) == π_b(0,80)` au sens de l'égalité stricte, et la « boucle » se ferme
en un tour avec un résidu de **0,0 exactement**. La boucle n'est une boucle que
parce qu'elle redessine le moteur à chaque tour — elle répond à « quel conduit
est cohérent », pas à « que fait ce conduit ».

**Ce que la correction achète n'est pas la poussée.** Corriger `π_b` de 3 % ne
déplace la poussée spécifique que de **0,2 %** (738,3 → 737,4 N·s/kg). En
revanche le débit bouge de **2,5 %** (10,094 → 10,351 kg/s) et les sections avec
lui. Le moteur est peu sensible à la perte de chambre pour ce qu'il produit, et
très sensible pour ce qu'il faut construire.

### Le cas bloqué : le plan a raison sur le fait et se trompe sur les trois raisons

Le point `M = 0,30 / T₀₄ = 2 000 K` **est bien refusé**, dès le tour 1. Mais :

**1. Ce n'est pas un blocage thermique, c'est un effondrement du dessin.** En
réinjectant `π_b = 0,839` au lieu de 0,95, la pression de fin de chambre tombe,
le col qu'il faudrait grandit — et il finit **plus large que la chambre**. Il
n'y a plus de convergent, donc plus de tuyère. C'est la garde géométrique de
l'étape 5 qui parle, pas celle de l'étape 7. D'où une exception dédiée,
`DesignCollapse`, dont le message dit explicitement que **le premier passage
avait accepté ce point** : sans cela l'utilisateur croirait avoir demandé
l'impossible dès le départ, alors que c'est la contre-réaction qui l'a
découvert. Un 0D seul rend ici un moteur d'allure parfaitement normale, à plus
de 500 N·s/kg.

**2. Le 1 749 K est juste, mais répond à une question que le moteur ne pose
pas.** *(⚠ Le Mach réel de 0,2716 était lui aussi gonflé par le `(1+f)` ; au
point fixe corrigé la chambre tourne exactement au Mach visé. L'énoncé reste
vrai partout ailleurs — voir « Après les quinze étapes ».)* C'est bien la limite de Rayleigh à `M = 0,30` avec `T₀₂ = 606,6 K`,
vérifié à 1 K près. Seulement **la chambre ne tourne jamais au Mach de
conception** : le col fixe le débit, donc le Mach réel, qui vaut **0,2716**. La
limite applicable est celle-là — **2 055 K** — et le point franchit donc le
blocage avec 55 K de marge avant de mourir d'autre chose.

**3. Sur toute la grille utile, la garde de blocage ne sert jamais.** Balayé sur
8 Mach de conception × 6 températures : **27 cas passent, 21 sont refusés par la
géométrie, zéro par le blocage thermique.** La raison est exactement le résultat
de l'étape 8 — le col fixe le débit, donc le Mach de chambre, qui reste
*toujours* sous celui visé : le moteur se régule lui-même, et la dégénérescence
géométrique arrive systématiquement en premier. À `M = 0,30`, la marge au
blocage se referme (55 K, puis 28, puis 2 à `T₀₄ = 2 100 K`) mais la géométrie
refuse à 2 150 K avant qu'elle n'atteigne zéro.

La garde de l'étape 7 **reste en place et n'est pas morte** : l'étape 8 l'atteint
avec une géométrie construite à la main (`A_chambre/A_col = 1,0`). Mais aucun cas
dessiné par `from_cycle` ne l'atteint, et un test le dit — sans quoi on croirait
l'avoir éprouvée.

**La marge au point nominal est large** : 1 275 K entre `T₀₄ = 2 000 K` et la
limite de Rayleigh au Mach réel d'entrée de chambre. Le cas de référence n'est
pas au bord.

**Bibliothèques.** NumPy, SciPy.

### 10. Visualisation des profils

**À faire.** `plots.py`, fonctions pures rendant des figures Matplotlib, aucune
dépendance au reste de la chaîne au-delà des tableaux. Cinq figures :

1. les quatre profils `M(x)`, `p(x)`, `T(x)`, `u(x)` en panneaux empilés
   partageant l'axe `x`, avec les zones (diffuseur, chambre, tuyère) en fond
   coloré et le col marqué ;
2. la section `A(x)`, tracée **à l'échelle** et en miroir, pour qu'on voie le
   moteur et pas une courbe ;
3. le diagramme `T-s` du cycle 0D, avec les quatre stations ;
4. les balayages `F/ṁ(M₀)` et `S(M₀)`, avec les fourchettes publiées en bande ;
5. la carte de blocage : `T₀₄` maximal contre Mach d'entrée de chambre, avec le
   point de fonctionnement dessus.

Trois points de mise en œuvre : les axes des profils partagent `x` et ne le
répètent pas ; les zones sont annotées une fois en haut ; les couleurs restent
distinguables en vision dichromate — pas de rouge/vert pour opposer deux
courbes.

**Vérification.** `test_plots.py` : aucune importation de Matplotlib hors de
`plots.py` (vérifié dans un sous-processus) ; chaque figure a le nombre d'axes
et de courbes attendu ; les fichiers PNG sont écrits et font plus de 10 ko ; la
figure de profils marque le col au bon `x` à 1e-6.

**Fait.** 327 tests en 55,9 s, dont 27 pour les figures. Les cinq figures sont
écrites par `python3 main.py --figures` : 105, 35, 68, 80 et 71 ko. Tous les
critères du plan sont atteints, et **`plots.py` n'importe pas `pyplot`** — un
cran de plus que demandé.

**Pourquoi pas `pyplot`.** Il tient un registre global des figures : cent
figures produites en boucle y restent toutes. Il choisit aussi un backend, donc
dépend de l'affichage. L'API objet — `Figure`, puis `savefig` — n'a ni l'un ni
l'autre. Deux tests le garantissent : l'un vérifie que l'*importation* du module
ne charge pas `pyplot`, l'autre qu'un *appel* ne le charge pas non plus, parce
qu'une importation propre ne dit rien de ce que fait le code.

**Les fonctions ne connaissent aucun objet du projet** — ni `Cycle`, ni
`Engine`, ni `Convergence` : rien que des tableaux et des nombres. Les 27 tests
tournent donc **sans lancer une seule intégration**, et un échec ne peut venir
que du tracé. L'assemblage vit dans `cli.py`, qui est un module de service.

### Trois pièges, dont un physique

**Le diagramme T-s en grandeurs totales perd la détente, et ne le dit pas.** Une
tuyère ne change pas la température totale — c'est sa définition. Tracées en
totales, les stations 4 et 9 tombent donc à la **même température** (2 000 K
toutes les deux) et à 12 J/(kg·K) l'une de l'autre : le cycle se réduit à un
triangle, la détente a disparu, et rien ne le signale. En statique, elle
descend de **1 926 à 777 K** et occupe la moitié de la figure. Le module
documente le piège, la fonction s'appelle sur des états statiques, et un test
mesure les deux distances pour que le choix soit un fait et non un goût.

**La carte de blocage exige une échelle logarithmique.** La limite diverge comme
`1/M²` : 51 000 K à `M = 0,05` contre 700 K à `M = 0,60`. En échelle linéaire,
toute la zone utile (2 000 à 4 000 K) tient dans le dixième inférieur de la
figure et le point de fonctionnement est indiscernable de la courbe. Un test
vérifie l'échelle, avec les deux nombres qui la justifient.

**Le corps du conduit est gris, pas bleu.** Coloré, il se mélangeait aux fonds
de zone et la chambre paraissait verte. Les couleurs de zone portent
l'information ; le corps ne porte que la forme.

### La vision dichromate est vérifiée par simulation, et le test a du pouvoir

La règle du plan — « pas de rouge/vert pour opposer deux courbes » — est
mesurée, pas affirmée. Simulation de Viénot-Brettel-Mollon : sRGB → linéaire →
LMS, projection sur le plan du dichromate, retour, puis distance euclidienne.

**Le test établit d'abord qu'il sait détecter un mauvais choix** : le rouge et
le vert par défaut de Matplotlib sont à **0,727** l'un de l'autre en vision
normale — parfaitement distincts — et tombent à **0,068** en deutéranopie.
Sans cette contre-épreuve, mesurer nos propres couleurs ne prouverait rien.

Mesuré ensuite sur les sept paires qui partagent réellement un axe : la plus
faible est vermillon/vert à **0,197**, soit près de trois fois mieux ; et la
paire qui oppose les deux courbes du balayage, bleu/orange, tient à **0,564** en
protanopie et **0,683** en deutéranopie contre 0,925 en vision normale — dix
fois au-dessus du rouge/vert.

**Mais la vraie garantie est structurelle** : un test vérifie qu'aucune figure
ne distingue deux *courbes* par la seule couleur — le style de trait porte
l'information en même temps, donc une impression en noir et blanc reste lisible.
Aucune palette ne peut garantir cela à elle seule. Les traits de repère
(verticale du col, horizontale `M = 1`) sont exclus du test : ils sont noirs
tous les deux, mais une horizontale et une verticale ne se confondent pas.

**Bibliothèques.** NumPy, Matplotlib.

### 11. CLI, rapport, README

**À faire.** `cli.py` : `--mach`, `--altitude`, `--t04`, `--fuel`, `--m-comb`
(Mach visé en chambre), `--figures`, plus les trois modes du dépôt :

- `--check` : la chaîne complète en moins de dix secondes — relations
  thermodynamiques, cycle idéal contre forme fermée, un profil 1D, le contrôle
  de quantité de mouvement, et une figure. Sortie 0 ou 2 ;
- `--validate` : le balayage en Mach, la comparaison aux fourchettes publiées,
  et la boucle de couplage. Produit le tableau du README ;
- `--sweep` : le balayage seul, en table texte.

`report.py` : rapport texte aligné dans `sorties/<cas>/rapport.txt`, qui annonce
**d'abord les écarts aux identités exactes** — cycle idéal contre forme fermée,
quantité de mouvement, masse — puis les résultats. Un lecteur pressé lit les
trois premières lignes et sait s'il peut croire le reste.

README : le contexte, les deux modèles, le tableau mesuré contre la
littérature, la section « ce qui est vérifié et ce qui est validé », la carte de
blocage, et les limites connues.

**Vérification.** `test_cli.py` : `--check` sort en 0 ; un cas bloqué sort en 2
avec un message qui nomme le Mach de chambre et donne le `T₀₄` maximal ;
`--dry-run` n'écrit rien dans `sorties/` ; le rapport contient les lignes
attendues et des nombres analysables (`grep "ecart a l'identite"` doit rendre un
nombre).

**Fait.** **406 tests en 110 s**, dont 31 pour la ligne de commande et 20 pour
le rapport. `--check` fait ses **quatorze** contrôles en **4,5 s**, sous le
budget de dix secondes annoncé — dépendances, cloisonnement, config,
atmosphère, thermo, cycle, blocage, frottement contre Fanno, moteur complet,
boucle, entrée à chocs obliques, et le rendu d'une figure. Tous les critères du
plan sont atteints.

### Le rapport dit les écarts avant les résultats, et c'est testé

Cinq identités, toutes vérifiées sur le point convergé et affichées **avant le
moindre résultat** :

```
  ok   cycle ideal contre forme fermee         3.33e-16   seuil 1e-12   etape 3
  ok   quantite de mouvement (chambre)         1.46e-11   seuil 1e-08   etape 6
  ok   conservation de la masse                4.44e-16   seuil 1e-10   etape 6
  ok   raccord chambre -> tuyere               1.33e-15   seuil 1e-09   etape 8
  ok   point fixe de la boucle 0D <-> 1D       8.58e-13   seuil 1e-10   etape 9
```

Un test vérifie l'**ordre** — `CE QU'ON PEUT CROIRE` avant `LE CYCLE 0D` avant
`LA PERFORMANCE` — parce qu'un rapport qui annonce 737 N·s/kg en tête et relègue
en annexe le fait qu'une identité ne tombe pas **ment par ordonnancement**, sans
écrire un seul nombre faux. Un autre vérifie que le verdict tombe avant la
première station du cycle. Chaque seuil **nomme l'étape** qui l'a mesuré : un
seuil sans provenance est un seuil qu'on n'osera pas toucher. Chacun garde deux
ordres de grandeur de marge, ce qu'un test exige aussi — un seuil serré au
dernier chiffre casse à la première version de SciPy.

Et la validation contre la littérature est **nommée comme telle** dans le
rapport : « accord d'ORDRE DE GRANDEUR, pas une identité ». Les confondre est
exactement ce que tout ce dépôt cherche à éviter.

**Les nombres sont analysables**, comme le plan le demandait — mais le test
d'abord écrit attrapait le « 0 » de « boucle **0**D » avant l'écart. Il cherche
désormais la notation scientifique, ce qu'un `grep` ferait, et exige deux
nombres par ligne dans l'ordre écart-puis-seuil.

### Le cas bloqué sort en 2, et donne DEUX nombres

Un point refusé par la physique **n'est pas une erreur d'usage** : le programme
a parfaitement fonctionné, c'est le moteur qui n'existe pas. D'où le code 2, et
non 1 — un test vérifie les deux codes séparément.

Le message donne les deux limites, parce que l'étape 9 a montré que la première
ne suffit pas :

```
Au Mach de chambre VISE (0.3), la limite de Rayleigh est T04 = 1749 K ;
vous demandez 2000 K.
Mais le col fixe le debit, donc le Mach REEL de chambre : 0.2716, ou la
limite vaut 2055 K. C'est celle-la qui s'applique.
```

N'annoncer que les 1 749 K du Mach visé ferait chercher le problème au mauvais
endroit. Le message rappelle aussi que **le premier passage avait accepté ce
point** : c'est la contre-réaction qui l'a détruit, pas la demande initiale.

### Un changement de comportement assumé

`python3 main.py` sans mode **calcule le cas par défaut** au lieu d'afficher
l'aide. Tant qu'il n'y avait rien à intégrer, l'aide était la réponse la plus
utile ; maintenant qu'il y a un moteur, un outil qui ne fait rien par défaut se
fait oublier. Le test correspondant a été réécrit plutôt que supprimé, et dit
pourquoi.

`--dry-run` est vérifié sur les **trois** modes qui écrivent, et le test
n'inspecte pas le contenu du dossier : il vérifie que le dossier **n'existe
pas**. Un test qui cherche un dossier vide passe aussi quand le dossier a été
créé pour rien.

### Le README

Il suit l'ordre du rapport : les cinq identités d'abord, la distinction
vérification/validation ensuite, les résultats après. Il porte les quatre
figures, le tableau de validation avec son domaine corrigé (`M₀` de 2 à **3,5**,
et pourquoi), la table d'identification de l'entrée d'air, et les huit limites
connues — dont trois découvertes en route : le diffuseur qui accélère, la masse
de carburant que le 1D ne porte pas, et le débit fixé par le col.

Sa dernière section dit ce que ce dépôt a de particulier : **les tests gardent
par écrit ce qui a été cru puis mesuré.** Le coefficient `(γ+1)/4` récité de
mémoire au lieu de `2/(γ+1)`, deux valeurs de tables de chocs obliques fausses,
la valeur post-singularité qui n'existe pas — chacune est conservée à côté du
nombre qui l'a démentie.

**Bibliothèques.** stdlib, plus tout le reste.

---

# SCOPE OPTIONNEL

À faire seulement si le scope principal est terminé et le README écrit.

### 12. Frottement pariétal

Le troisième coefficient d'influence, `4 C_f dx/D`, qui transforme l'EDO en
Fanno-Rayleigh. Oracle disponible : l'écoulement de **Fanno** en forme fermée
(`4 C_f L*/D` contre `M`), donc vérifiable exactement comme le reste. Intérêt
réel : le frottement **rapproche du blocage**, et la carte de l'étape 7 se
déplace. À mesurer plutôt qu'à affirmer.

**Fait.** 372 tests en 83 s, dont 25 pour le frottement. Nouveau module
`fanno.py` — l'arborescence n'en prévoyait pas, et c'est le test
d'exhaustivité ajouté à l'étape 8 qui a exigé qu'il soit déclaré. Le solveur a
désormais **trois termes source et trois formes fermées** pour les vérifier
séparément : isentropique pour la section, Rayleigh pour la chaleur, Fanno pour
le frottement. C'est ce qui permet de savoir *lequel* est faux quand un profil
déraille. *(Un quatrième terme — l'injection de carburant — et son oracle sont
venus après les quinze étapes ; voir la section finale.)*

**La vérification tombe à 1e-13**, et une identité remarquable la double : le
rapport `p₀/p₀*` de Fanno est **littéralement** la même fonction que `A/A*` de
l'isentropique. Deux modules écrits séparément, à des étapes différentes,
rendent le même nombre — écart **0,0 exactement**, testé par
`assert_array_equal`. Ce n'est pas une coïncidence : les deux expriment que le
débit critique passe par une section critique donnée.

**Le contrôle de quantité de mouvement EXIGE la traînée pariétale**, et c'est ce
qui en fait une vérification du troisième terme. L'identité devient
`F(sortie) − F(entrée) = ∫p dA − ∫τ_w dA_paroi`, la traînée valant
`½ρu²·(4C_f/D)·A` par mètre — exactement le terme source de l'EDO, mais atteint
par un bilan que l'intégration n'a pas emprunté. Mesuré : **2,4e-15 avec la
traînée, 2,3e-02 sans**. Quatorze ordres de grandeur : sans ce terme on
croirait le solveur cassé.

**Le frottement ACCÉLÈRE un écoulement subsonique**, ce qui surprend et qui est
vrai : M passe de 0,300 à 0,3149 sur 6,5 m. Il fait chuter la pression, donc la
vitesse débitante — mais la température chute aussi, donc la vitesse du son, et
le rapport des deux monte. En supersonique il décélère. Les deux comportements
sont le même fait : le Mach va vers 1, comme avec la chaleur.

**L'asymétrie des deux branches est énorme.** La longueur réduite subsonique
n'est bornée par rien — 66,92 à `M = 0,10` — tandis que la supersonique plafonne
à **0,8215**, soit 80 fois moins. Un écoulement supersonique ne peut jamais
parcourir qu'une longueur bornée avant de bloquer, si vite qu'il aille.

### Ce que le plan demandait de mesurer, mesuré

La `T₀₄` maximale que la chambre laisse passer, à `M` d'entrée 0,20 :

| longueur | sans frottement | avec `C_f = 0,005` |
|---|---|---|
| 0,6 m (la chambre) | 3 495 K | 3 444 K (**−1,5 %**) |
| 2,0 m | 3 495 K | 3 330 K (−4,7 %) |
| 5,0 m | 3 495 K | 3 110 K (−11,0 %) |

La carte de l'étape 7 se déplace donc, mais **peu pour cette chambre-ci** :
0,6 m ne font que 3,2 diamètres. C'est un **effet de longueur**, et il faut une
chambre huit fois plus longue pour qu'il devienne notable. Sans frottement, la
limite trouvée par le solveur retrouve exactement celle de Rayleigh en forme
fermée : 3 495,3 K des deux côtés.

**Sur ce moteur, le frottement ne change presque rien — et il faut le dire aussi
nettement.** Au point fixe, à `C_f = 0,005`, `π_b` perd 0,56 % et la poussée
spécifique **0,11 %**. Deux raisons se composent : la chambre est courte, et
l'étape 9 avait déjà mesuré que le moteur est peu sensible à `π_b`.

**Et le débit ne peut structurellement PAS bouger.** `from_cycle` dimensionne le
col avec `A_col ∝ 10(1+f)/π_b` et `choking_mass_flow` résout
`A*(ṁ) ∝ ṁ/π_b = A_col`. **Au point fixe les deux `π_b` sont le même nombre :
il se simplifie, et il reste `ṁ = 10(1+f)` exactement.** Vérifié à 1e-11 pour
`C_f` de 0 à 0,02. Le frottement n'agissant que sur `π_b`, il ne peut pas
toucher au débit — une simplification exacte trouvée en mesurant, pas en
raisonnant.

### Un défaut que le frottement a révélé

Le message de refus annonçait « **blocage thermique** » et parlait de chaleur
quoi qu'il arrive. C'était juste tant que la chaleur était le seul terme
source. Dès que le frottement existe, un conduit **sans la moindre chaleur**
peut bloquer — et le message annonçait alors :

> « la chaleur demandée porterait T₀ d'un facteur **1,0000** là où M = 2 n'en
> autorise que 1,2604, soit **0,79 fois trop** »

À la fois faux et incompréhensible. Le diagnostic mesure désormais les **deux
budgets séparément**, et ne nomme que les causes actives. Le cas le plus
instructif est celui où les deux agissent : **92 % du budget de chaleur et 7 %
du budget de frottement — et l'écoulement bloque.** Aucun des deux pris seul ne
l'aurait annoncé, et le message le dit.

Enfin, un contrôle croisé révélé par un test qui échouait : à `M = 2`, la forme
fermée donne `L_max = 3,050 m`, et le solveur — qui ne connaît pas cette
formule — déclenche son événement de blocage à **`x = 3,0500 m`**.

### 13. Diffuseur à chocs obliques

Remplacer le choc normal par une rampe à deux ou trois chocs obliques. La
récupération de pression totale passe de 0,81 à ~0,9 à M₀ = 3, ce qui est
précisément la raison d'être des entrées d'air à rampes. Le test est la
comparaison à `π_d` de MIL-E-5008B, qui décrit justement une entrée bien
dessinée.

**Fait.** 20 tests, dans `thermo.py` (relations de chocs obliques) et
`tests/test_inlet.py`. *Le « de 0,81 à ~0,9 » du plan mélangeait deux choses :
0,81 est la valeur MIL, pas celle du choc normal, qui vaut **0,3283**. Le vrai
écart à combler allait donc de 0,33 à 0,84.*

### Le résultat : ce que la corrélation du 0D décrivait vraiment

Tout le cycle repose sur `π_d = 0,8088` à `M₀ = 3`, tiré de MIL-E-5008B, qui
parle d'une entrée « bien dessinée » **sans dire laquelle**. En calculant la
récupération de rampes optimales, on peut enfin répondre. À `M₀ = 3` :

| entrée | `π_d` |
|---|---|
| choc droit seul | 0,3283 |
| 1 rampe optimale | 0,5812 |
| 2 rampes | 0,7437 |
| **MIL-E-5008B** | **0,8088** |
| **3 rampes** | **0,8361** |
| 4 rampes | 0,8892 |

**La corrélation tombe entre deux et trois rampes, tout près de trois — et
3,3 % en dessous.** C'est exactement ce qu'on attend : la théorie des chocs
ignore les pertes visqueuses et les non-idéalités qu'une corrélation empirique,
elle, contient.

**Et le nombre de rampes croît avec le Mach** : 1 rampe à `M₀ = 1,5` et 2, deux
rampes à 2,5, **trois de 3 à 4**, quatre à 5. Une corrélation empirique unique
décrit donc une **famille** d'entrées dont la complexité grandit avec le Mach de
dessin — ce que font les entrées réelles, et que le nombre seul ne laissait pas
voir.

**Ce que le choix de l'entrée coûte en poussée** : `F/ṁ` va de **580,4** (choc
droit) à **739,0** (MIL) à **752,6 N·s/kg** (4 rampes). **30 % de la poussée
tiennent au dessin de l'entrée d'air**, que le 0D résume par un seul nombre.

### Le critère d'Oswatitsch, retrouvé plutôt que supposé

À l'optimum, **tous les chocs obliques ont exactement la même force** — la même
composante normale de Mach, 1,4252 pour les trois obliques à `M₀ = 3`. Ce n'est
pas une hypothèse imposée : une optimisation libre à Nelder-Mead sur les N
angles la retrouve **à six chiffres**. Le code s'en sert ensuite pour ramener
une optimisation à N paramètres à **une seule inconnue**, cette force commune.

Précision qui compte : **le choc terminal, lui, est plus faible** — 1,3483
contre 1,4252, soit 5 % — et l'écart est systématique. L'énoncé courant « tous
les chocs de même force » vaut donc pour les **obliques seuls**.

### Deux valeurs citées de mémoire, toutes deux fausses

C'est le fait le plus utile de l'étape, et il justifie l'ordre des
vérifications. La relation θ-β-M est d'abord vérifiée **par substitution** —
resubstituer l'angle trouvé doit redonner la déviation demandée : **1,4e-16 sur
30 couples** — et `M₂` par l'**identité de vitesse tangentielle**
`M₁cos β·a₁ = M₂cos(β−θ)·a₂`, soit **5,6e-16**. Aucune source extérieure
n'intervient. Ensuite seulement viennent les tables. Deux nombres que j'avais
notés de mémoire s'y sont révélés faux :

- « `M = 4`, `θ = 30°` → `β = 47,90°` » : le code rend **45,224**, et la
  substitution redonne 30,000000° ;
- « `M = 5`, `θ = 10°` → `M₂ = 4,0016` » : le code rend **3,999162**, confirmé
  par l'identité tangentielle à 1e-16.

Une table recopiée ne vaut que ce que vaut celui qui la recopie ; une identité,
elle, se vérifie. Les quatre valeurs de table qui *tiennent* le font à 0,01°, et
un test garde les deux erreurs par écrit.

### Le refus qui manquait

Au-delà d'une déviation maximale, le choc **se détache** et il n'existe plus de
solution oblique — même famille de limite que le blocage thermique. Mesuré :
`θ_max` vaut 12,11° à `M = 1,5`, 22,97° à 2, 34,07° à 3, 38,77° à 4, et plafonne
à **45,58°**. Le message nomme la rampe fautive et propose de répartir l'angle
sur plusieurs rampes.

### Complément, après les quinze étapes : l'entrée est devenue un choix du cas

`thermo.py` savait calculer une entrée à rampes, mais le cycle ne savait pas la
*demander* : il fallait calculer `π_d` à la main et le passer en dur. `EngineCase`
porte désormais **`inlet_ramps`**, et `python3 main.py --rampes 3` fait tourner
tout le moteur sur une entrée calculée. 459 tests.

**Ce qu'une valeur imposée ne pouvait pas faire : `π_d` varie avec le Mach.**
Figer 0,8361 sur un balayage serait faux — une entrée à trois rampes récupère
**0,9966 à `M₀` = 1,5** et **0,4306 à `M₀` = 5**. C'est précisément pour cela que
le nombre de rampes devait entrer dans le **cas**, et non être résolu au bord de
la ligne de commande.

**Le cloisonnement de l'étape 1 survit, et c'est la partie délicate.**
`config.py` ne doit pas tirer NumPy — un script d'optimisation extérieur
construit des milliers de cas et ne doit pas payer l'importation. L'`import` de
`thermo` est donc placé **dans** la fonction de récupération : construire mille
cas ne tire toujours rien, et un test le vérifie dans un interpréteur neuf ;
*demander* des rampes le tire, évidemment, et un second test le vérifie aussi.
La frontière est à l'usage, pas à la construction.

Le calcul coûtant 190 ms et un balayage en demandant quinze, il est **mis en
cache** — un test mesure le facteur, supérieur à 100.

Deux garde-fous ajoutés : `pi_d` et `inlet_ramps` ensemble sont **refusés** (deux
façons de fixer la même grandeur : il faut choisir, et le dire vaut mieux que de
trancher en silence par ordre de priorité), et le nombre de rampes est borné à
6 — au-delà, chaque rampe rapporte moins qu'elle ne coûte en longueur et en
masse, ce que ce modèle ne compte pas.

### 14. `cp(T)` et `γ(T)`

Polynômes NASA à sept coefficients pour l'air et les produits de combustion.
Effet attendu, à mesurer : `T₀₄` réel plus bas de 100 à 200 K à `f` donné, donc
poussée surestimée de quelques pour cent par le modèle à `cp` constant. C'est le
premier poste d'erreur du 0D, et le mesurer vaut mieux que de le supposer.

**Fait.** 432 tests en 116 s, dont 26 pour les propriétés caloriques. Nouveau
module `caloric.py`, déclaré dans `CORE_MODULES` — le test d'exhaustivité de
l'étape 8 l'a de nouveau exigé. **Le module ne remplace pas le modèle à `cp`
constant : il le chiffre.**

### Cinquante coefficients recopiés, donc cinquante occasions de se tromper

Ce projet a déjà attrapé deux valeurs citées de mémoire qui étaient fausses — le
coefficient `(γ+1)/4` de l'étape 8 et deux entrées de tables de chocs à l'étape
13. Les données NASA sont protégées par **quatre contrôles indépendants**, et le
second est décisif :

1. `cp` à 298,15 K contre les valeurs de référence : accord à **0,4 %** ;
2. **le raccord des deux polynômes à 1 000 K.** Chaque espèce est ajustée
   séparément sur [200 ; 1 000] et [1 000 ; 5 000], mais les deux jeux sont
   construits pour se rejoindre : **une seule faute de frappe dans l'un des dix
   casse la continuité**. Mesuré : **0,000 % sur les cinq espèces**, ce qui
   valide les cinquante nombres d'un coup ;
3. la masse molaire déduite de la composition de l'air donne
   `R = 287,048 J/(kg·K)`, là où tout le reste du projet code 287,05 ;
4. le `f_stoechiométrique` déduit de la composition atomique colle à celui que
   `fuels.py` stocke depuis l'étape 1 : **0,37 %** sur le kérosène, 0,15 % sur
   le méthane, 0,14 % sur l'hydrogène. Deux jeux de données écrits à onze
   étapes d'intervalle, et ils sont cohérents.

Un cinquième contrôle ne doit rien aux données : **l'argon**. Ses deux jeux de
coefficients sont littéralement identiques et il n'a aucun degré de liberté
interne, donc `γ = 5/3` exactement à toute température, et la détente
isentropique y suit `(p₂/p₁)^((γ−1)/γ)` en forme fermée. La bissection sur
l'entropie la retrouve à **1e-9**. Le module est donc juste indépendamment de
la justesse des polynômes.

### Le résultat, et le plan avait raison dans SON cadrage

**À `f` donné — le cadrage du plan, et celui qui compte, parce qu'on dose du
carburant et pas des kelvins :**

| grandeur | `cp` constant | `cp(T)` | écart |
|---|---|---|---|
| `T₀₄` | 2 000 K | **1 743,6 K** | **−256 K** |
| `F/ṁ` | 739,0 N·s/kg | 697,5 | **−5,6 %** |
| `Isp` | 2 150 s | 2 029 s | −5,6 % |

Le plan annonçait « plus bas de 100 à 200 K » et « poussée surestimée de
quelques pour cent ». **Le second est juste ; le premier est sous-estimé de
moitié** — et l'écart croît avec la température visée : −109 K à
`T₀₄ = 1 400 K`, −256 à 2 000, −443 à 2 600. La cause s'écrit : le `cp` moyen
des produits de 298 à 2 000 K vaut **1 214 J/(kg·K)**, soit **1,21 fois** les
1 004,5 supposés.

### Mais le signe dépend du cadrage, et c'est le fait le plus instructif

Si l'on **impose `T₀₄` = 2 000 K** — donc si l'on accepte de brûler les
**+27,4 %** de carburant qu'il faut vraiment — alors le `cp` plus grand rend
**plus** d'enthalpie à la détente : `u₉` passe de 1 569 à 1 643 m/s et la
poussée spécifique de 739 à **816 N·s/kg**, soit **+10,4 %**.

Les deux cadrages ne se contredisent pas : ils ne tiennent pas la même chose
fixe. Le second flatte le moteur ; le premier décrit ce qui se passe quand on
ouvre le robinet d'une quantité donnée.

**L'énoncé invariant, et donc le seul à retenir : dans les deux cadrages,
l'impulsion spécifique est SURESTIMÉE** — de 5,6 % à `f` imposé, de 11,2 % à
`T₀₄` imposé. Le modèle à `cp` constant flatte la consommation dans tous les
cas, et c'est cela qu'il faut dire au lecteur.

### Deux erreurs de ma part, gardées par écrit

**J'ai mesuré un gradient en croyant mesurer un saut** — exactement l'erreur de
l'étape 1 sur l'atmosphère. Comparer `h(999,999)` à `h(1000,001)` donne 3e-6
relatif, mais c'est `cp·dT` sur l'intervalle d'échantillonnage. Après
retranchement de la variation attendue, le vrai saut vaut **2,4e-07** au pire —
les ajustements NASA sont continus *à leur propre précision*, pas exactement. Et
l'argon, dont les deux jeux sont identiques, donne **1,8e-16** : le test mesure
donc bien la donnée et non l'arithmétique.

**Et j'ai écrit « 0,3 % » pour l'écart entre `cp` moyen et `cp` à la température
moyenne.** C'est **1,57 %** : 1 148,0 contre 1 166,4 J/(kg·K) entre 300 et
2 000 K. Prendre le second introduirait à lui seul une erreur du même ordre que
celle que toute l'étape cherche à mesurer.

### 15. Tuyère sous-détendue et adaptation

`p₉ ≠ p₀` : le terme de pression `(p₉ − p₀)A₉` entre dans la poussée. Permet de
tracer la perte d'adaptation contre l'altitude à géométrie fixée — la vraie
question opérationnelle d'un statoréacteur, qui vole sur un domaine et non en un
point.

**Fait.** 449 tests en 161 s, dont 17 pour l'adaptation. Sixième figure,
`adaptation.png`. Et la réponse à la question posée est **non** :

> **L'adaptation d'un statoréacteur à géométrie figée ne dépend pas de
> l'altitude. Elle dépend du Mach.**

Entre 11 et 20 km l'atmosphère standard est **isotherme**, la compression
dynamique rend `p₀₄` rigoureusement proportionnelle à l'ambiante, et le rapport
`p₉/p_amb` s'en trouve indépendant de l'altitude — **au dernier chiffre**
(mesuré : écart 1e-15 sur 9 km), pas « à peu près ». La pression ambiante a
pourtant été divisée par 4,5. C'est l'inverse d'une fusée, dont la pression de
chambre ne suit pas l'ambiante, et c'est pour cela que le plan attendait une
courbe contre l'altitude.

Ce qui reste de dépendance en altitude vient **uniquement de la température** :
1,0308 au sol contre 1,0204 en tropopause, soit 1 % sur 11 km. La figure met les
deux panneaux à la **même échelle verticale** — dilater celui du bas rendrait la
courbe d'altitude « intéressante » alors que le fait à montrer est justement
qu'elle ne l'est pas.

### Contre le Mach, la perte est en U et son fond est au point de dessin

| `M₀` | 2,0 | 2,5 | 3,0 | 3,5 | 4,0 |
|---|---|---|---|---|---|
| `p₉/p_amb` | 0,246 | 0,508 | **1,020** | 1,957 | 3,546 |
| perte d'adaptation | **20,9 %** | 3,1 % | **0,02 %** | 2,7 % | 8,1 % |

**Mais le minimum n'est pas exactement à `M₀` = 3**, et la raison remonte à
l'étape 8 : `p₉ = p_amb` tombe à **`M₀` = 2,985**. La géométrie a été
dimensionnée par le 0D pour 10,000 kg/s et le col en impose 10,351 ; `A₉` est
donc légèrement trop petite pour le débit réel.

### Le modèle calculait un écoulement qui n'existe pas, et il le dit maintenant

À `M₀` = 2 il produisait tranquillement un terme de pression de **−73 % de la
poussée**. Mais `p₉/p_amb = 0,246` est bien sous le seuil de décollement de
Summerfield (0,4) : la couche limite se décolle du divergent et l'écoulement
calculé n'existe pas. Le décollement est un effet visqueux 2D, **explicitement
hors périmètre** — le modèle ne peut pas le calculer, mais il peut dire où il
cesse d'être valable, et c'est le minimum. La frontière tombe à **`M₀` = 2,33**,
et la figure grise la zone.

Sous `M₀` ≈ 1,4 le moteur **freine** au lieu de pousser : −648 N·s/kg à
`M₀` = 1,05. C'est physiquement juste, et le modèle signale de surcroît que
l'écoulement y est décollé.

*Deux paris que j'avais faits et qui étaient faux.* Je croyais que le domaine de
vol casserait à bas Mach : il passe, avec une poussée négative. La première
borne dure est **`M₀` = 6,5**, où `T₀₂ = 2 047 K` dépasse déjà le `T₀₄` demandé —
la limite thermique calculée analytiquement à l'étape 4. Et ma bande de
décollement, tracée « jusqu'au dernier point décollé », avait une largeur
**nulle** dès qu'un seul point de la grille l'était, c'est-à-dire exactement
dans le cas qu'elle devait montrer ; la frontière est désormais interpolée.

### Un défaut du modèle, révélé par cette étape — et corrigé depuis

> ⚠ **Ce défaut est réparé.** Il a motivé la correction du `(1+f)` décrite dans
> « Après les quinze étapes ». La perte de poussée est désormais positive
> partout, minimum **+2,8e-06**.

**La perte de poussée devenait légèrement négative près de l'adaptation** —
−0,019 % vers `M₀` = 2,94 — ce qui est thermodynamiquement impossible : une
tuyère adaptée est optimale. La cause est le `(1+f)` posé à la main à l'étape 8.
L'EDO conserve la masse, donc le terme de pression est calculé avec le débit
**d'air** tandis que le terme de quantité de mouvement porte le carburant ; les
deux diffèrent de 3,5 %, et cela suffit à faire paraître une tuyère non adaptée
meilleure qu'une adaptée.

L'objet rendu porte donc **deux** pertes : celle en poussée, qui est la grandeur
qu'on veut, et celle sur l'**impulsion de sortie par kilogramme de gaz**, qui est
la quantité que l'optimalité gouverne réellement. La seconde est **positive
partout**, avec un minimum de 1,4e-06 exactement là où `p₉ = p_amb`. Le défaut
est donc bien dans le comptage de la masse, et non dans la tuyère — et il vaut
mieux l'écrire que le laisser trouver.

---

## Ordre d'exécution

```
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
                        ┊
            [ arret propre possible apres 4 : le 0D seul se tient ]
                        ┊
                  12 · 13 · 14 · 15
```

Les étapes 4 et 5 sont indépendantes l'une de l'autre. L'étape 7 peut se faire
avant la 6 si l'on écrit `rayleigh.py` en premier — c'est même recommandé, parce
que ses formes fermées servent d'oracle à la 6.

**Quatre points de contrôle où il ne faut pas tricher :**

- **fin de l'étape 2** — l'aller-retour `M(A/A*(M))` est exact à 1e-12 sur les
  deux branches. Sans cela, tout dimensionnement de section sera faux d'une
  branche, et le résultat aura l'air raisonnable ;
- **fin de l'étape 3** — le cycle station par station retombe sur la forme
  fermée du statoréacteur idéal à 1e-12. C'est le seul endroit où le 0D peut
  être vérifié **exactement**, et il ne reviendra pas ;
- **fin de l'étape 7** — le solveur refuse `q > q_max` et le RK4 naïf du test,
  lui, franchit M = 1. Les deux moitiés comptent : sans la contre-épreuve, la
  protection n'est pas prouvée ;
- **fin de l'étape 9** — la boucle 0D ↔ 1D converge et le `π_b` obtenu diffère
  du 0,95 supposé. À partir de là, les deux modèles ne se contredisent plus, et
  c'est le résultat du projet.

Les étapes 1 à 9 ne produisent **aucune image** et sont entièrement testables :
c'est là qu'est la totalité du risque technique. Les étapes 10 et 11 habillent
une chaîne déjà prouvée. Résister à la tentation de tracer les profils en
premier parce que c'est gratifiant : ils auront exactement la même allure avec
un solveur faux.

---

## Vérification globale

**Automatique** — **mesuré : 468 tests en 223 s**, au terme du projet. Le budget
annoncé était de trente secondes ; l'essentiel du dépassement vient des
étapes 8 et suivantes, où chaque test appelle une intégration, une boucle ou un
balayage de domaine :

```bash
cd ramjet-1d
python3 -m unittest discover -s tests
```

**Bout en bout :**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --check          # 20 controles, 4,5 s, sortie 0
.venv/bin/python main.py                  # le cas par defaut, rapport complet
.venv/bin/python main.py --rampes 3       # entree a rampes au lieu de MIL
.venv/bin/python main.py --validate       # balayage + fourchettes publiees
.venv/bin/python main.py --figures        # les six figures dans sorties/
.venv/bin/python main.py --m-comb 0.30 ; echo $?   # doit rendre 2
```

Le dernier cas sort en **2** — critère numérique non atteint, pas erreur
d'usage — en annonçant l'**effondrement du dessin** et non le blocage thermique
(étape 9), et en donnant les **deux** limites : celle du Mach de chambre visé
(1 749 K) et celle du Mach réel, qui est la seule applicable (2 055 K).

**Contrôles à l'œil, sur `sorties/` :**

- le profil de Mach **monte** dans le diffuseur (0,178 → 0,207), monte encore
  dans la chambre (jusqu'à 0,43), passe par 1 **exactement au col** et monte
  ensuite jusqu'à 2,81. *Le plan annonçait qu'il « descend » dans le diffuseur ;
  c'est faux, et pas par erreur de calcul : `from_cycle` relie la section de
  capture (310 cm²) à la chambre (269 cm²), donc le conduit CONVERGE et
  l'écoulement subsonique accélère. La vraie compression est tout entière dans
  le saut `π_d` ; ce qui reste traçable n'est qu'un raccord de section
  (étape 8).* ;
- la pression fait le contraire, et sa chute dans la chambre est visible :
  653 → 540 kPa, soit **−17,2 %** — c'est la perte de Rayleigh, celle que le 0D
  supposait ;
- la température monte **en rampe quasi linéaire** sur la zone de chambre
  (601 → 1 923 K, à 0,8 % d'une droite) et redescend dans la tuyère. *Le plan
  disait « en marche d'escalier » : le dépôt de chaleur est réparti
  linéairement en `x`, donc `T₀` monte en rampe, et `T` la suit de près tant que
  le Mach reste bas ;*
- sur la carte de blocage, le point de fonctionnement est **sous** la courbe
  limite, et on voit de combien.

**Les deux contrôles décisifs :**

```bash
.venv/bin/python main.py --check | grep "forme fermee"   # < 1e-12 (mesure 5,9e-16)
.venv/bin/python main.py --check | grep "complet"        # F/mdot a 0,09 % du 0D
```

*Le contrôle de quantité de mouvement ne peut pas se lire sur le profil complet
: la formule de Simpson exige un pas constant, et le profil est recollé de
quatre tronçons de pas différents. Mesuré, elle rend 1,6e-05 là où chaque
tronçon pris à part donne 1e-11 à 4e-08 — une erreur de QUADRATURE qu'on aurait
lue comme une erreur de SOLUTION. `momentum_residual` refuse désormais le pas
variable (étape 8).*

Un écart de quantité de mouvement à 1e-3 « qui a l'air de marcher » est une
reconstruction de pression fausse, et tous les profils publiés le seront aussi.

---

## Limites connues, à documenter dans le README

- **Pas de cinétique** : la longueur de chambre est un paramètre géométrique,
  pas un temps de réaction. Le modèle ne peut donc pas dire si la combustion a
  le temps de se faire — seulement ce qui se passe si elle se fait.
- **`T₀₄` est imposé, pas calculé.** La dissociation plafonne la température
  réelle vers 2 400–2 700 K et ce modèle l'ignore : demander `T₀₄ = 3 500 K`
  donnera un résultat, et il sera faux.
- **La richesse n'est pas bornée par la physique**, seulement par un contrôle
  explicite à `f_st`. Un modèle sans chimie n'a rien à dire au-delà.
- **Le diffuseur n'est pas modélisé : c'est un saut.** La récupération vient de
  la corrélation MIL-E-5008B (`π_d = 0,809` à M₀ = 3), qui décrit une entrée à
  rampes multiples bien dessinée. *Le plan prévoyait un choc normal ; il a été
  rejeté à l'étape 8, parce qu'il rend 0,3283 et contredirait le cycle d'un
  facteur 2,46.* Conséquence à assumer : le modèle ne dit rien du système de
  chocs, et le tronçon « diffuseur » du profil n'est qu'un raccord de section,
  où l'écoulement **accélère** au lieu de ralentir. Un vrai diffuseur à chocs
  obliques reste l'étape 13.
- **Le 1D ne porte pas la masse de carburant.** L'équation intégrée conserve la
  masse : elle ne peut structurellement pas porter le `(1+f)`. Le terme est
  remis à la main dans le bilan de poussée, et il vaut **7,4 %** — exactement
  `f·u₉`. L'oublier est l'erreur la plus coûteuse du projet.
- **À géométrie fixée, le débit n'est pas un paramètre : le col le fixe.** La
  géométrie dessinée pour 10,000 kg/s en avale 10,094, puis 10,351 une fois la
  boucle convergée. Un `run_engine` auquel on « impose » un débit décrirait un
  moteur qui n'existe pas.
- **La garde de blocage thermique n'est atteinte par aucun cas dessiné par
  `from_cycle`.** Balayé sur 8 Mach × 6 températures : 27 passent, 21 sont
  refusés par la géométrie, **zéro** par le blocage. Le moteur se régule
  lui-même. La garde reste en place — une géométrie construite à la main
  l'atteint — mais il faut savoir que c'est une assurance, pas un garde-fou
  quotidien.
- **Pas de couche limite**, donc pas de blocage visqueux, pas de décollement, et
  une section géométrique confondue avec la section utile.
- **Stationnaire seulement.** Le blocage thermique est détecté comme une limite
  d'existence, ce qui ne dit rien de ce qui se passe **après** — le désamorçage
  est un phénomène instationnaire, hors périmètre.
- **Gaz parfait, `cp` constant par zone.** Erreur attendue de quelques pour cent
  sur la poussée, à mesurer à l'étape 14 plutôt qu'à supposer.
- **L'accord avec la littérature est un accord d'ordre de grandeur.** Les
  fourchettes publiées couvrent des moteurs de dessins différents ; tomber dedans
  valide le modèle, ça ne le calibre pas.

---

## Après les quinze étapes : les deux réécritures

Le plan était terminé. Restaient deux défauts que le projet avait lui-même
nommés et chiffrés, et que ni l'un ni l'autre ne pouvait corriger par un ajout.

### A. Le `(1+f)` est entré dans l'EDO

**Un quatrième coefficient d'influence**, celui de l'apport de masse. Je l'ai
**dérivé** des cinq équations de conservation plutôt que de le citer, puis
vérifié contre un **solveur linéaire indépendant** : accord **7,5e-16** sur
dix-huit couples (γ, M). Il vaut

```
2 (1 + γM²) (1 + (γ−1)/2 M²) / (1 − M²)  ·  (dṁ/dx)/ṁ
```

c'est-à-dire **exactement le double du coefficient de chaleur**. Conséquence
remarquable : à `T₀` constante, ajouter de la masse revient *exactement* à
chauffer selon Rayleigh avec `T₀ ∝ ṁ²`. **Le quatrième terme a donc lui aussi
un oracle exact** — vérifié à **1e-13**, comme les trois autres.

L'injection est supposée **transversale** : le carburant entre sans quantité de
mouvement axiale, donc il faut l'accélérer aux dépens de l'écoulement. C'est le
cas conservateur, et il faut le dire.

**Ce que la correction a fait apparaître, et qui change trois résultats
antérieurs :**

| au point fixe | avant | après |
|---|---|---|
| débit d'entrée | 10,3505 kg/s | **10,000000** |
| débit de sortie | — (non porté) | **10,350546 = 10 (1+f)** |
| Mach réel de chambre | 0,2074 | **0,200000** |
| perte d'adaptation minimale | **−0,019 %** | **+2,8e-06** |
| `π_b` convergé | 0,9218 | 0,9196 |
| marge avant blocage | 1 275 K | 1 495 K |

Les 0,94 % d'écart de l'étape 8 et le décalage de Mach de l'étape 9 étaient
donc **des erreurs de comptage, pas des faits physiques**. Le dimensionnement et
l'analyse comptent enfin la masse de la même façon, et le point fixe reproduit
exactement le dessin — c'est ce que « cohérent » doit vouloir dire. Les tests
concernés ont été réécrits plutôt que renumérotés, et disent ce qui a changé.

**Un bogue attrapé au passage, et il rejouait le défaut qu'on venait de
corriger.** `_concatenate` recopiait `mass_flow` depuis le premier tronçon —
correct tant que c'était un scalaire. Devenu tableau, cela donnait la poussée
**683 N·s/kg**, c'est-à-dire exactement la valeur *sans* carburant.

### B. Un cycle 0D à `cp(T)` et `γ(T)`

**Et une limite qu'il faut énoncer avant le résultat : le 1D reste à `γ`
constant, délibérément.** Ses trois oracles supposent tous `γ` constant.
Étendre le solveur à `γ(T)` ne demanderait pas une équation de plus : cela
**détruirait la structure de vérification** du projet, où chaque terme source se
compare à une forme fermée exacte. Le dépôt a choisi l'autre terme de l'échange,
et le mesure au lieu de le supposer.

**Le cycle variable n'a donc pas d'oracle à lui. Il se vérifie par
DÉGÉNÉRESCENCE** : nourri d'un gaz dont `cp` ne varie pas — l'argon, seul
monoatomique du jeu NASA — il doit rendre *exactement* le cycle à `cp` constant
calculé à la main. Mesuré sur six grandeurs : **écart maximal 3,1e-15**. C'est
ce qui justifie que les mélanges soient des **arguments** de la fonction et non
des constantes du module ; sans cette injection, la vérification serait
impossible.

**Puis une validation, et elle est décisive.** La température de flamme
adiabatique du kérosène dans l'air, au dosage stœchiométrique depuis 298 K, est
publiée autour de **2 390 K** sans dissociation. Le modèle à propriétés
variables donne **2 394 K**. Le modèle à `cp` constant donnerait **3 286 K** là
où les propriétés variables donnent 2 598 K depuis `T₀₂` : **26 % de trop**. Ce
n'est pas un raffinement — c'est la différence entre une température de flamme
plausible et une qui ne l'est pas.

**Le résultat, sur le cycle entier** (à `T₀₄` = 2 000 K imposé) :

| | `cp` constant | `cp(T)` | écart |
|---|---|---|---|
| `f` | 0,0351 | 0,0448 | **+27,9 %** |
| `F/ṁ` | 739,0 | 836,7 N·s/kg | +13,2 % |
| `Isp` | 2 150 | 1 904 s | **−11,5 %** |
| `η_th` | 0,5882 | 0,5354 | −9,0 % |
| `T₀₂` | 606,6 | 601,3 K | −0,9 % |

La poussée **monte** parce que `T₀₄` est imposé : on brûle plus, donc il y a
plus d'enthalpie à détendre. Ce que le modèle à `cp` constant flatte, c'est la
**consommation**. Et la compression dynamique chauffe *moins* qu'annoncé, un
`cp` plus grand absorbant la même énergie cinétique avec moins d'élévation de
température.

**Deux détails d'implémentation qui ont demandé une mesure.** L'équilibre
d'énergie du cycle à `cp` constant référence implicitement l'enthalpie à
**0 K**, alors que le PCI est défini à 298,15 K ; la version variable utilise la
bonne référence, et l'écart entre les deux conventions vaut `f·cp·T_ref`, soit
0,7 % de l'apport. Et la recherche de `f` doit **rétrécir sa borne haute** quand
elle est incalculable — à la stœchiométrie, un gaz à faible `cp` dépasse les
5 000 K du domaine des polynômes — exactement comme la recherche de débit de
l'étape 8 rétrécit la sienne quand la chambre bloque.

**Total : 468 tests en 223 s**, `--check` fait vingt contrôles.

---

## Bilan

**Ce qui a été construit.** 16 modules, 19 fichiers de tests, 11 400 lignes.
468 tests en 223 s, vingt contrôles dans `--check`, six figures, un rapport
texte. Les quinze étapes du plan sont faites — les onze du scope principal et
les quatre optionnelles — plus deux réécritures que le projet s'est lui-même
prescrites.

**Ce qui a été vérifié, et ce qui a été validé.** Six formes fermées servent
d'oracle, une par terme source de l'équation quasi-1D plus deux pour les cycles,
et les écarts vont de 1e-12 à 1e-16. La validation contre la littérature, elle,
est un accord d'**ordre de grandeur** : les fourchettes publiées de poussée et
d'impulsion spécifiques, la corrélation MIL-E-5008B, et la température de flamme
adiabatique du kérosène. Le rapport nomme les deux différemment, et dans cet
ordre.

**Ce que ce plan a appris en se trompant.** Quatorze affirmations écrites ici
avant mesure se sont révélées fausses, et chacune est conservée à côté du nombre
qui l'a démentie :

- le coefficient `(γ+1)/4` près du col, récité de mémoire au lieu de `2/(γ+1)` ;
- deux valeurs de tables de chocs obliques, β = 47,90° et M₂ = 4,0016 ;
- la valeur post-singularité « M = 1,122 », qui **n'existe pas** — elle dépend
  du pas ;
- le maximum de poussée « vers M₀ = 2,5 », qui est à 2,36 : c'était le pas de la
  grille qu'on lisait ;
- le rendement thermique « croissant », qui passe par un maximum à M₀ = 4,79 ;
- la fourchette « 600–900 sur M₀ de 2 à 4 », que le tableau du plan violait
  lui-même à M₀ = 4 ;
- le choc normal du diffuseur, « cohérent avec `π_d` », qui en est loin d'un
  facteur 2,46 ;
- « deux ou trois tours suffisent » pour la boucle, qui en demande sept ;
- le cas bloqué de l'étape 9, refusé pour une tout autre raison que celle
  prédite ;
- le Mach qui « descend » dans le diffuseur, alors qu'il monte ;
- la température « en marche d'escalier », qui est une rampe ;
- « `T₀₄` plus bas de 100 à 200 K », qui est 256 ;
- l'écart de débit de 0,94 % et le Mach de chambre de 0,2074, qui étaient des
  **artefacts de comptage** et non des faits ;
- et un budget de tests de trente secondes, tenu à un facteur sept près.

C'est la partie du projet qui a demandé le plus de travail, et c'est celle qui
vaut le plus. Un plan qui ne se trompe jamais est un plan qui n'a rien mesuré.

**Ce qui reste ouvert, et qui est assumé.** Le solveur 1D garde `γ` constant :
l'étendre détruirait la structure d'oracles qui fait tout l'intérêt du dépôt, et
le coût est chiffré plutôt que caché. Pas de cinétique, donc rien à dire sur le
temps de réaction. Pas de couche limite, donc le décollement de tuyère est
**signalé** et non calculé. `T₀₄` reste imposé, et la dissociation ignorée.
Chacune de ces limites est dans le README, avec le nombre qui la mesure.

---

## Références

- P. Hill & C. Peterson, *Mechanics and Thermodynamics of Propulsion*, 2ᵉ éd.,
  Addison-Wesley (1992) — chapitres 5 et 6, cycle et statoréacteur.
- J. D. Mattingly, *Elements of Gas Turbine Propulsion*, McGraw-Hill (1996) —
  chapitre 5, forme fermée du statoréacteur idéal.
- A. H. Shapiro, *The Dynamics and Thermodynamics of Compressible Fluid Flow*,
  vol. 1, Ronald Press (1953) — chapitre 8, coefficients d'influence ;
  chapitre 7, écoulement de Rayleigh.
- J. D. Anderson, *Modern Compressible Flow*, 3ᵉ éd., McGraw-Hill (2003) —
  relations de choc et écoulements avec apport de chaleur.
- MIL-E-5008B — corrélation de récupération de pression totale des entrées
  d'air supersoniques.
- U.S. Standard Atmosphere (1976) — couches et gradients.
