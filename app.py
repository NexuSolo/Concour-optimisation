import json
import networkx as nx
import random
import heapq
from datetime import datetime

def create_graph_from_json(file_path):
    # Lecture du fichier JSON
    with open(file_path, 'r') as dataset:
        data = json.load(dataset)
    
        G = nx.DiGraph()
        for edge in data['roads']:
            if edge['isOneWay']:
                G.add_edge(edge['intersectionId1'], edge['intersectionId2'], length=edge['length'], one_way=True, visits=0)
            else:
                G.add_edge(edge['intersectionId1'], edge['intersectionId2'], length=edge['length'], one_way=False, visits=0)
                G.add_edge(edge['intersectionId2'], edge['intersectionId1'], length=edge['length'], one_way=False, visits=0)

        return G, data['batteryCapacity'], data['numDays']

def next_step(graph, current_node, battery_remaining, start_node):
    # Création d'une liste d'arêtes disponibles avec leurs attributs
    available_edges = []
    regular_edges = []  # Arêtes ne menant pas au départ
    start_node_edges = []  # Arêtes menant au départ
    
    for neighbor in graph.neighbors(current_node):
        edge_data = graph[current_node][neighbor]
        # On vérifie si l'arête peut être empruntée avec la batterie restante
        if edge_data['length'] <= battery_remaining:
            # Plus une arête a été visitée, moins elle aura de chance d'être choisie
            # On utilise visits + 1 pour éviter une division par zéro
            weight = 1 / (edge_data['visits'] + 0.1)
            edge_info = {
                'to_node': neighbor,
                'length': edge_data['length'],
                'weight': weight,
                'visits': edge_data['visits']
            }
            
            if neighbor == start_node:
                start_node_edges.append(edge_info)
            else:
                regular_edges.append(edge_info)
    
    # On retourne les arêtes régulières si elles existent, sinon les arêtes vers le départ
    return regular_edges if regular_edges else start_node_edges

def generate_path(graph, start_node, battery_capacity, days_remaining):
    current_node = start_node
    battery_remaining = battery_capacity
    path = [current_node]
    
    while True:
        available_edges = next_step(graph, current_node, battery_remaining, start_node)
        if not available_edges:
            break
            
        # Sélection d'une arête basée sur les poids
        weights = [edge['weight'] for edge in available_edges]
        next_edge = random.choices(available_edges, weights=weights)[0]
        
        # Si c'est le dernier jour, on continue jusqu'à épuisement de la batterie
        if days_remaining == 0:
            if next_edge['length'] <= battery_remaining:
                # Mise à jour des visites sur l'arête
                graph[current_node][next_edge['to_node']]['visits'] += 1
                
                # Mise à jour de la position et de la batterie
                current_node = next_edge['to_node']
                battery_remaining -= next_edge['length']
                path.append(current_node)
            else:
                break
            continue
        
        # Sinon, on vérifie si on peut rentrer à la base
        temp_battery = battery_remaining - next_edge['length']
        can_return, return_path, return_cost = can_return_to_base(
            graph, 
            next_edge['to_node'], 
            start_node, 
            temp_battery
        )
        
        if not can_return:
            # Si on ne peut pas rentrer, on essaie de rentrer depuis la position actuelle
            _, emergency_return_path, _ = can_return_to_base(
                graph,
                current_node,
                start_node,
                battery_remaining
            )
            if emergency_return_path:
                # On ajoute le chemin de retour d'urgence
                path.extend(emergency_return_path[1:])  # [1:] pour éviter de dupliquer le noeud actuel
            break
        
        # Mise à jour des visites sur l'arête
        graph[current_node][next_edge['to_node']]['visits'] += 1
        
        # Mise à jour de la position et de la batterie
        current_node = next_edge['to_node']
        battery_remaining = temp_battery
        path.append(current_node)
    
    return path

def save_solution(start_node, paths, dataset_name):
    solution = {
        "chargeStationId": start_node,
        "itinerary": [node for path in paths for node in path]  # Aplatir tous les chemins en un seul itinéraire
    }
    
    # Création du nom de fichier avec la date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solutions/{dataset_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(solution, f)
    
    return filename

def modified_astar(graph, start, goal, battery_remaining):
    # File de priorité pour stocker les noeuds à explorer
    frontier = []
    # Le coût pour atteindre chaque noeud
    cost_so_far = {start: 0}
    # Pour reconstruire le chemin
    came_from = {start: None}
    
    # Ajout du noeud initial
    heapq.heappush(frontier, (0, start))
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        
        if current == goal:
            break
            
        for neighbor in graph.neighbors(current):
            edge_data = graph[current][neighbor]
            # Calculer le nouveau coût en tenant compte des visites précédentes
            visit_penalty = edge_data['visits'] * 5  # Pénalité pour les arêtes déjà visitées
            new_cost = cost_so_far[current] + edge_data['length'] + visit_penalty
            
            # Vérifier si on a assez de batterie pour emprunter cette arête
            if new_cost <= battery_remaining:
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    # L'heuristique est une estimation simple de la distance restante
                    priority = new_cost + edge_data['length']
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
    
    # Reconstruction du chemin
    if goal not in came_from:
        return None, float('inf')
        
    path = []
    current = goal
    total_cost = cost_so_far[goal]
    
    while current is not None:
        path.append(current)
        current = came_from[current]
    
    return path[::-1][:-1:], total_cost

def can_return_to_base(graph, current_node, start_node, battery_remaining):
    # Utiliser A* modifié pour trouver le chemin le plus court vers la base
    path, cost = modified_astar(graph, current_node, start_node, battery_remaining)
    return path is not None and cost <= battery_remaining, path, cost

# Test avec le fichier example
if __name__ == "__main__":
    dataset_names = [
        "1_example",
        "2_pacman",
        "3_efrei",
        "4_manhattan",
        "5_gta",
        "6_paris",
        "7_london",
    ]
    # dataset_name = "3_efrei"

    for dataset_name in dataset_names:
        graph, battery_capacity, num_days = create_graph_from_json(f"datasets/{dataset_name}.json")
        
        # Choix d'un nœud de départ aléatoire
        start_node = random.choice(list(graph.nodes()))
        all_paths = []
        
        # Génération des chemins pour chaque jour
        for day in range(num_days):
            # Génération du chemin pour ce jour
            days_remaining = num_days - day - 1
            path = generate_path(graph, start_node, battery_capacity, days_remaining)
            all_paths.append(path)
        
        # Sauvegarde de la solution
        output_file = save_solution(start_node, all_paths, dataset_name)
        print(f"Solution saved to {output_file}")