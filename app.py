import json
import networkx as nx
import random
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

def next_step(graph, current_node, battery_remaining):
    # Création d'une liste d'arêtes disponibles avec leurs attributs
    available_edges = []
    for neighbor in graph.neighbors(current_node):
        edge_data = graph[current_node][neighbor]
        # On vérifie si l'arête peut être empruntée avec la batterie restante
        if edge_data['length'] <= battery_remaining:
            # Plus une arête a été visitée, moins elle aura de chance d'être choisie
            # On utilise visits + 1 pour éviter une division par zéro
            weight = 1 / (edge_data['visits'] + 1)
            available_edges.append({
                'to_node': neighbor,
                'length': edge_data['length'],
                'weight': weight,
                'visits': edge_data['visits']
            })
    
    return available_edges

def generate_path(graph, start_node, battery_capacity):
    current_node = start_node
    battery_remaining = battery_capacity
    path = [current_node]
    
    while True:
        available_edges = next_step(graph, current_node, battery_remaining)
        if not available_edges:
            break
            
        # Sélection d'une arête basée sur les poids
        weights = [edge['weight'] for edge in available_edges]
        next_edge = random.choices(available_edges, weights=weights)[0]
        
        # Mise à jour des visites sur l'arête
        graph[current_node][next_edge['to_node']]['visits'] += 1
        
        # Mise à jour de la position et de la batterie
        current_node = next_edge['to_node']
        battery_remaining -= next_edge['length']
        path.append(current_node)
    
    return path

def save_solution(start_node, path, dataset_name):
    solution = {
        "chargeStationId": start_node,
        "itinerary": path
    }
    
    # Création du nom de fichier avec la date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solutions/{dataset_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(solution, f)
    
    return filename

# Test avec le fichier example
if __name__ == "__main__":
    dataset_name = "1_example"
    graph, battery_capacity, num_days = create_graph_from_json(f"datasets/{dataset_name}.json")
    
    # Choix d'un nœud de départ aléatoire
    start_node = random.choice(list(graph.nodes()))
    
    # Génération du chemin
    path = generate_path(graph, start_node, battery_capacity)
    
    # Sauvegarde de la solution
    output_file = save_solution(start_node, path, dataset_name)
    print(f"Solution saved to {output_file}")