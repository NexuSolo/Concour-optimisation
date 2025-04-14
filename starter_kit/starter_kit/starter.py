import json
import random
import networkx as nx
import test_solution
import datetime


def solve(dataset_txt):
    # Lecture du dataset
    dataset = json.loads(dataset_txt)

    # Choix de la station de charge : on prend l'intersection 0 par défaut. Il y a peut-être mieux à faire !
    base_id = 0

    # On crée le graphe du réseau avec networkx. On aurait pu le faire aussi avec un simple dictionnaire Python
    # networkx permet cependant d'avoir accès à certaines fonctions comme le calcul des distances plutôt que de les recoder
    G = nx.DiGraph()
    for edge in dataset['roads']:
        if edge['isOneWay']:
            G.add_edge(edge['intersectionId1'], edge['intersectionId2'], length=edge['length'], one_way=True)
        else:
            G.add_edge(edge['intersectionId1'], edge['intersectionId2'], length=edge['length'], one_way=False)
            G.add_edge(edge['intersectionId2'], edge['intersectionId1'], length=edge['length'], one_way=False)


    # Crée un dictionnaire des distances de chaque noeud à la base
    dist_to_base = nx.shortest_path_length(G, source=None, target=base_id, weight='length')    

    visited_roads = set()  # Ensemble des routes visitées
    curr_node = base_id  # Noeud courant, initialisé à la base
    path = [base_id]  # Chemin parcouru, initialisé avec la base
    score = 0  # Score initialisé à 0

    # Boucle sur chaque jour
    for day_i in range(dataset['numDays']):
        battery_remaining = dataset['batteryCapacity']  # Réinitialisation de la batterie au maximum

        # Boucle principale pour parcourir les routes
        while True:
            # On mélange les voisins pour éviter de toujours prendre le même chemin
            neighbors = list(G.neighbors(curr_node))
            random.shuffle(neighbors)
            
            
            # L'agorithme du choix du prochain noeud est là !
            # Commencez par trouver un meilleur algorithme que celui-ci
            # -------------------------------------------
            # On choisit le prochain noeud à visiter
            # On prend le premier voisin qui permet de rentrer à la base    
            for nxt in neighbors:
                edge_len = G[curr_node][nxt]['length']  # Longueur de l'arête vers le voisin

                # Si la batterie nous permet de visiter l'intersection et de rentrer a la base, on la choisit
                if battery_remaining >= edge_len + dist_to_base[nxt]:            
                    next_node = nxt
                    break

            # -------------------------------------------


            # Mise à jour du score si la route n'a pas été visitée
            if (curr_node, next_node) not in visited_roads:
                score += G[curr_node][next_node]['length']

            # Ajout des routes visitées dans les deux sens
            visited_roads.add((curr_node, next_node))
            visited_roads.add((next_node, curr_node))

            # Mise à jour de la batterie restante
            battery_remaining -= G[curr_node][next_node]['length']

            # Ajout du prochain noeud au chemin
            path.append(next_node)
            curr_node = next_node

            # On termine la journée si on est de retour à la base
            if curr_node == base_id:
                break

        print(f'End day {day_i+1} with {battery_remaining} battery')  # Message de fin de jour


    # Message de fin de routage
    print(f'Visited {len(visited_roads) // 2} / {len(dataset["roads"])} roads')
    print(f'Expected score: {score:_}')

    # Retour du résultat sous forme de chaîne JSON
    return json.dumps({"chargeStationId": base_id, "itinerary": path})



dataset_file = "1_example"
dataset = open(f'.\\datasets\\{dataset_file}.json').read()

print('---------------------------------')
print(f'Solving {dataset_file}')
solution = solve(dataset)
print('---------------------------------')
score, is_valid, message = test_solution.getSolutionScore(solution, dataset)

if is_valid:
    print('✅ Solution is valid!')
    print(f'Message: {message}')
    print(f'Score: {score:_}')
    
    save = input('Save solution? (y/n): ')
    if save.lower() == 'y':
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'{dataset_file}_{score}_{date}'
        
        with open(f'.\\solutions\\{file_name}.json', 'w') as f:
            f.write(solution)
        print('Solution saved')
    else:
        print('Solution not saved')
    
else:
    print('❌ Solution is invalid')
    print(f'Message: {message}')


