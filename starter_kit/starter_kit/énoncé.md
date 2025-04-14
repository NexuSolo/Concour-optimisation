# Cartographie optimale

Dans le cadre de l'amélioration de la vie urbaine, votre ville souhaite collecter des mesures (qualité de l'air, bruit, état de la route, ...) dans toutes les rues de l'agglomération. Pour cela, un robot autonome a été développé et embarque à son bord tous les capteurs nécessaires.

Sur les essais en laboratoire, le robot fonctionne parfaitement et arrive à naviguer en toute autonomie, mais il reste un problème de taille : sa batterie limitée l'oblige à rentrer à la base régulièrement pour se recharger !

Comme le temps est limité, vous avez été chargé·e de mettre en place un algorithme visant à optimiser les trajets du robot pour parcourir un maximum de rues différentes dans le temps imparti sans tomber en panne de batterie.

On vous fournit plusieurs jeux de données au format [JSON](https://fr.wikipedia.org/wiki/JavaScript_Object_Notation) correspondant à différents réseaux routiers.

Pour chacune de ces villes, vous devez choisir où placer la borne de recharge du robot, puis proposer un itinéraire de parcours du robot. Le robot est activé au début du premier jour avec une batterie complète, et suit l'itinéraire programmé. À chaque fois qu'il passe par l'intersection où se situe sa borne de chargement, il s'arrête pour recharger complètement sa batterie puis reprend la suite de son itinéraire le lendemain matin. Votre objectif est d'avoir parcouru un maximum de rues différentes dans le nombre de jours qui vous est imparti.

Chaque jeu de données vous rapportera jusqu'à 1 million de points pour le classement (voir détail plus bas).

## Entrée

Le schéma routier de chaque ville vous est fourni sous forme de [graphe](https://fr.wikipedia.org/wiki/Th%C3%A9orie_des_graphes), avec un ensemble d'intersections (les sommets) et de routes (les arêtes).

Le format JSON des fichiers fournis est détaillé ci-dessous :

```json
{
    "comment": "Exemple de réseau routier", // Description courte du dataset
    "batteryCapacity": 70,                  // Distance maximale que peut parcourir le robot entre deux recharges
    "numDays": 2,                           // Nombre de jours de fonctionnement du robot
    "intersections": [
        {
            "id": 0,                        // Identifiant de l'intersection (nombre entier)
            "lat": 5.0,                     // Latitude (nombre flottant)
            "lng": -8.66                    // Longitude (nombre flottant)
        },
        {
            "id": 1,
            "lat": 10.0,
            "lng": 0.0
        },
        ...
    ],
    "roads": [
        {
            "intersectionId1": 0,           // Identifiant de l'intersection à une extrémité de la route
            "intersectionId2": 1,           // Identifiant de l'intersection à l'autre extrémité de la route
            "isOneWay": false,              // Indique si la rue est en sens unique : ici, on peut la parcourir dans les deux sens.
            "length": 10                    // Longueur de la route (nombre entier)
        },
        {
            "intersectionId1": 0,
            "intersectionId2": 6,
            "isOneWay": true                // Ici, la rue est à sens unique : on peut uniquement la parcourir dans le sens intersectionId1 vers intersectionId2 (0->6).
        },
        ...
    ]
}
```

**Note :** La latitude et la longitude des intersections sont fournies à titre indicatif, pour vous aider à mettre en place une visualisation des données. La distance utilisée pour le calcul de la batterie et du score sera toujours le nombre entier fourni dans la liste "roads".

## Sortie

Pour chaque jeu de données, vous devez soumettre votre solution sous la forme d'un fichier JSON contenant deux éléments : l'identifiant de l'intersection où vous souhaitez placer la borne de recharge, et le trajet souhaité du robot.

```json
{
    "chargeStationId": 1,                               // Identifiant de l'intersection où vous placerez la station de chargement
    "itinerary": [1, 0, 6, 2, 1, 6, 2, 3, 4, 5, 0]      // Liste des intersections sur le trajet du robot
}
```

Notez que le robot doit commencer son trajet sur la station de recharge, mais peut **terminer sur n'importe quelle intersection**.

Si votre robot tombe en panne de batterie ou dépasse le nombre de jours autorisés, le score de votre solution sera tout de même décompté jusqu'au premier déplacement interdit. Pensez à regarder les messages d'avertissement.

## Calcul du score

### Distance unique

Cet exercice est un problème d'optimisation : il n'existe pas d'algorithme efficace connu pour le résoudre de manière exacte, vous devrez donc essayer d'approcher la plus petite distance possible sans forcément atteindre la solution parfaite.

On vous demande de proposer une solution qui maximise la **distance unique** parcourue : parcourir une rue de longueur L vous rapportera L points, mais seulement la première fois que vous la parcourez. Une rue déjà visitée précédemment (même en sens inverse) ne vous rapportera aucun point.

Le score de votre solution est égal à cette distance unique parcourue.

### Bonus de rapidité

Si vous parvenez à visiter **toutes** les rues de la ville avant la fin du temps imparti, on vous accordera un bonus proportionnel au temps restant.

Par exemple, vous avez 5 jours pour parcourir toutes les routes d'une ville dont les routes ont une longueur totale de 3000, avec une capacité de batterie de 1500 unités. Votre algorithme est si performant que le robot parvient à finir le parcours lors du 4e jour après avoir couvert toutes les rues, alors que sa batterie est encore chargée à 50% (soit 750 unités).

Le bonus est calculé ainsi : (nb_jours_complets_restants + (batterie_restante / batterie_max)) / nb_jours_total, soit (1 + 750/1500) / 5 = 0.3 = 30% dans notre exemple.

Dans ce cas, vous aurez un bonus de score égal à 30%, le score de votre solution sera donc de 3900.

### Classement global

Pour chaque jeu de données, le joueur avec la meilleure solution remporte 1 million de points de classement. Le score des autres joueurs est calculé en fonction de la solution du meilleur joueur avec la formule suivante : `1000000 * score_joueur / score_meilleur`.

Par exemple, si votre solution couvre 2 fois moins de distance unique que votre meilleur adversaire, votre score sur ce jeu de données sera de 500 000 points. Le classement est déterminé en fonction du total de points de classement obtenus.

Il est donc possible que votre score diminue au cours de la compétition, si l'un de vos adversaires parvient à battre la meilleure solution sur un jeu de données.

Seule votre meilleure soumission sur chaque jeu de données sera prise en compte dans le classement.
