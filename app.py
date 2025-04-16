import argparse
import datetime
import itertools
import json
import multiprocessing
import os
import random
import time
import networkx as nx

from test_solution import getSolutionScore

def load_dataset(file_path):
    """Charge le fichier dataset JSON."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier dataset '{file_path}' n'a pas été trouvé.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Erreur: Le fichier dataset '{file_path}' n'est pas un JSON valide.")
        exit(1)

def save_solution(solution, dataset_name, score, output_dir="solutions"):
    """Sauvegarde la solution dans un fichier JSON nommé."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    score_int = int(round(score)) if isinstance(score, (int, float)) else 0
    filename = f"{dataset_name}_{timestamp}_score{score_int}.json"
    file_path = os.path.join(output_dir, filename)

    try:
        with open(file_path, 'w') as f:
            json.dump(solution, f, indent=4)
        print(f"Solution sauvegardée dans : {file_path}")
        return file_path
    except IOError as e:
        print(f"Erreur lors de la sauvegarde de la solution : {e}")
        return None
    except TypeError as e:
        print(f"Erreur lors de la sérialisation JSON de la solution : {e}")
        return None


def build_graph(dataset):
    """Construit un graphe NetworkX orienté (DiGraph) à partir du dataset."""
    G = nx.DiGraph()
    # Garder les données des intersections pour lat/lon si besoin plus tard
    all_intersections_data = {inter['id']: inter for inter in dataset['intersections']}
    G.add_nodes_from(all_intersections_data.keys()) # Ajouter tous les nœuds

    # Ajouter lat/lon comme attributs de nœud (optionnel, pour référence)
    for node_id, data in all_intersections_data.items():
        G.nodes[node_id]['lat'] = data['lat']
        G.nodes[node_id]['lng'] = data['lng']

    all_physical_edges = set() # Pour le scoring (arêtes physiques uniques)
    edge_lengths_map = {} # Map: tuple(sorted(id1, id2)) -> length

    for road in dataset['roads']:
        id1, id2 = road['intersectionId1'], road['intersectionId2']
        length = road['length']
        is_one_way = road['isOneWay']

        # Vérifier si les intersections existent avant d'ajouter l'arête
        if id1 not in all_intersections_data or id2 not in all_intersections_data:
             print(f"Avertissement: Route {id1}-{id2} référence une intersection inconnue.")
             continue

        G.add_edge(id1, id2, length=length)
        if not is_one_way:
            G.add_edge(id2, id1, length=length)

        edge_key = tuple(sorted((id1, id2)))
        all_physical_edges.add(edge_key)
        if edge_key not in edge_lengths_map:
             edge_lengths_map[edge_key] = length

    # Précalculer et stocker le degré (nombre total de voisins entrants/sortants) pour chaque nœud
    degrees = dict(G.degree())
    nx.set_node_attributes(G, degrees, 'degree')

    return G, all_physical_edges, edge_lengths_map


# --- Algorithme Principal (avec Backtracking Simple) ---

def generate_solution(dataset, graph: nx.DiGraph, all_physical_edges_keys, edge_lengths_map):
    """Génère une solution avec choix de base amélioré, stratégie greedy et backtracking."""
    intersections = list(graph.nodes())
    if not intersections:
        return {"chargeStationId": -1, "itinerary": []}

    # --- 1. Choix Amélioré de la Station de Charge (Basé sur Degré) ---
    # Prendre un échantillon plus large
    sample_size = min(len(intersections), max(20, int(len(intersections) * 0.1))) # Ex: 10% ou 20 nœuds min
    candidate_nodes_sample = random.sample(intersections, sample_size)

    # Récupérer le degré pour chaque candidat
    candidates_with_degree = []
    for node_id in candidate_nodes_sample:
        degree = graph.nodes[node_id].get('degree', 0)
        candidates_with_degree.append((node_id, degree))

    # Trier les candidats par degré décroissant
    candidates_with_degree.sort(key=lambda item: item[1], reverse=True)

    # Sélectionner le top K candidats (ex: top 10 ou 20% de l'échantillon)
    top_k_count = min(len(candidates_with_degree), max(10, int(sample_size * 0.2)))
    top_candidates = [node_id for node_id, degree in candidates_with_degree[:top_k_count]]

    # Choisir aléatoirement parmi les meilleurs candidats
    if top_candidates:
        charge_station_id = random.choice(top_candidates)
    else:
        # Fallback si l'échantillon était vide ou si aucun degré n'a pu être récupéré
        charge_station_id = random.choice(intersections)


    # --- Précalcul des distances de retour (inchangé) ---
    memoized_return_paths = {}
    def get_return_distance(node_id):
        if node_id == charge_station_id:
             return 0
        if node_id in memoized_return_paths:
             return memoized_return_paths[node_id]
        try:
            if node_id not in graph or charge_station_id not in graph:
                 memoized_return_paths[node_id] = float('inf')
                 return float('inf')
            dist = nx.shortest_path_length(graph, source=node_id, target=charge_station_id, weight='length')
            memoized_return_paths[node_id] = dist
            return dist
        except nx.NetworkXNoPath:
            memoized_return_paths[node_id] = float('inf')
            return float('inf')
        except nx.NodeNotFound:
             memoized_return_paths[node_id] = float('inf')
             return float('inf')

    # Initialisation de l'état global
    itinerary = [charge_station_id]
    visited_physical_edges = set() # Arêtes physiques visitées au total
    current_day = 1
    battery_capacity = dataset['batteryCapacity']
    num_days = dataset['numDays']

    # --- Planification multi-jours : découpage des arêtes en groupes (proximité à la base si souhaité) ---
    all_edges_list = list(all_physical_edges_keys)
    random.shuffle(all_edges_list)
    num_days = dataset['numDays']
    day_edge_groups = [set() for _ in range(num_days)]
    for idx, edge in enumerate(all_edges_list):
        day_edge_groups[idx % num_days].add(edge)

    # Constantes pour le backtracking
    MAX_RETRIES_PER_DAY = 2 # Nombre max de tentatives supplémentaires par jour
    MIN_SCORE_RATIO_THRESHOLD = 0.15 # Seuil de ratio score/distance pour déclencher une tentative

    # --- Boucle principale des jours ---
    while current_day <= num_days:
        # Sauvegarder l'état au début du jour
        start_of_day_visited_edges = visited_physical_edges.copy()
        start_of_day_itinerary_len = len(itinerary)
        start_of_day_random_state = random.getstate()

        num_retries_today = 0

        # --- Boucle de tentatives pour la journée actuelle ---
        while True:
            # Réinitialiser l'état pour cette tentative
            current_node = charge_station_id
            remaining_battery = battery_capacity
            daily_path = [charge_station_id] # Chemin construit pour cette tentative
            daily_distance_travelled = 0
            # Arêtes visitées PENDANT cette tentative (commence par celles déjà visitées)
            current_attempt_visited_edges = start_of_day_visited_edges.copy()

            # Restaurer l'état RNG si ce n'est pas la première tentative
            if num_retries_today > 0:
                random.setstate(start_of_day_random_state)
                # random.jumpahead(num_retries_today) # Optionnel

            # --- Simulation de la journée (boucle interne) ---
            while True:
                possible_moves = []
                neighbors = list(graph.successors(current_node))

                if not neighbors:
                     break # Impasse

                # Définir le groupe d'arêtes à cibler aujourd'hui (pour la portée locale)
                target_edges_today = day_edge_groups[current_day-1]

                for neighbor in neighbors:
                    length = graph.edges[current_node, neighbor]['length']
                    can_move = remaining_battery >= length
                    estimated_return_cost = get_return_distance(neighbor)
                    can_return_if_needed = (remaining_battery - length >= estimated_return_cost) or \
                                           (neighbor == charge_station_id) or \
                                           (current_day == num_days)

                    if can_move and can_return_if_needed:
                        physical_edge_key = tuple(sorted((current_node, neighbor)))
                        is_visited = physical_edge_key in current_attempt_visited_edges
                        # --- Accent sur les arêtes non visitées du groupe du jour proches du robot ---
                        is_today_target = physical_edge_key in target_edges_today
                        # Calcul de la distance à la base pour pondérer la priorité (plus c'est proche, mieux c'est)
                        try:
                            dist_to_base = nx.shortest_path_length(graph, source=neighbor, target=charge_station_id, weight='length')
                        except Exception:
                            dist_to_base = 99999
                        # Bonus si arête non visitée, dans le groupe du jour, et proche
                        bonus = 0
                        if not is_visited and is_today_target:
                            # Plus la distance à la base est faible, plus le bonus est grand
                            bonus = max(0, 1000 - dist_to_base)
                        # --- Lookahead sur 3 coups : score potentiel sur 3 mouvements ---
                        def lookahead_score(node, visited_edges, depth, max_depth):
                            if depth > max_depth:
                                return 0
                            gain = 0
                            for n2 in graph.successors(node):
                                edge2 = tuple(sorted((node, n2)))
                                if edge2 not in visited_edges:
                                    gain2 = graph.edges[node, n2]['length']
                                    visited_edges2 = visited_edges | {edge2}
                                    gain += gain2 + lookahead_score(n2, visited_edges2, depth+1, max_depth)
                            return gain
                        lookahead_gain = 0
                        visited_for_lookahead = current_attempt_visited_edges.copy()
                        if not is_visited:
                            lookahead_gain += length
                            visited_for_lookahead.add(physical_edge_key)
                        lookahead_gain += lookahead_score(neighbor, visited_for_lookahead, 1, 3)
                        # ---
                        priority_score = (
                            is_visited,
                            -bonus, # bonus fort pour les arêtes non visitées du groupe du jour proches
                            -lookahead_gain,
                            -length if not is_visited else length
                        )
                        possible_moves.append({
                            "priority": priority_score, "neighbor": neighbor,
                            "length": length, "edge_key": physical_edge_key,
                            "is_visited": is_visited
                        })

                if not possible_moves:
                     break # Bloqué

                possible_moves.sort(key=lambda x: x["priority"])

                # Stratégie de Sélection (inchangée)
                chosen_move = None
                if random.random() < 0.9 or len(possible_moves) == 1:
                    chosen_move = possible_moves[0]
                else:
                    top_moves = possible_moves[:min(len(possible_moves), 5)]
                    unvisited_in_top = [m for m in top_moves if not m['is_visited']]
                    if unvisited_in_top and random.random() < 0.7:
                        chosen_move = random.choice(unvisited_in_top)
                    else:
                        chosen_move = random.choice(top_moves)

                # Mettre à jour l'état de la tentative
                next_node = chosen_move["neighbor"]
                move_length = chosen_move["length"]
                edge_key_tuple = chosen_move["edge_key"]

                remaining_battery -= move_length
                daily_path.append(next_node)
                daily_distance_travelled += move_length
                current_attempt_visited_edges.add(edge_key_tuple) # Mettre à jour la copie locale
                current_node = next_node

                if current_node == charge_station_id:
                     break # Retour à la base
            # --- Fin de la simulation de la journée ---

            # Évaluer la journée simulée
            new_edges_today = current_attempt_visited_edges - start_of_day_visited_edges
            daily_score_gain = sum(edge_lengths_map.get(edge_key, 0) for edge_key in new_edges_today)

            # Critère de Backtrack
            perform_retry = False
            if daily_distance_travelled > 5: # Éviter retry si on a à peine bougé
                score_ratio = daily_score_gain / daily_distance_travelled if daily_distance_travelled > 0 else 0
                if score_ratio < MIN_SCORE_RATIO_THRESHOLD and num_retries_today < MAX_RETRIES_PER_DAY:
                    perform_retry = True

            if perform_retry:
                num_retries_today += 1
            else:
                # Accepter cette journée
                if len(daily_path) > 1:
                    itinerary = itinerary[:start_of_day_itinerary_len]
                    itinerary.extend(daily_path[1:])
                visited_physical_edges = current_attempt_visited_edges
                break # Sortir de la boucle de tentatives

        # Passer au jour suivant
        current_day += 1
    # --- Fin de la boucle principale des jours ---

    return {
        "chargeStationId": charge_station_id,
        "itinerary": itinerary
    }


# --- Worker Function ---
def worker_task(args_tuple):
    """Génère et évalue une solution. Prend un tuple d'arguments."""
    process_id, dataset, graph, all_physical_edges_keys, edge_lengths_map, dataset_txt = args_tuple
    random.seed(os.getpid() * int(time.time()) + process_id)
    try:
        # Passer edge_lengths_map à generate_solution
        solution = generate_solution(dataset, graph, all_physical_edges_keys, edge_lengths_map)
        if not solution or 'chargeStationId' not in solution or 'itinerary' not in solution or not isinstance(solution['itinerary'], list):
             return -1, False, "Solution invalide générée", None, process_id

        solution_json_str = json.dumps(solution)
        score, is_valid, message = getSolutionScore(solution_json_str, dataset_txt)
        return score, is_valid, message, solution, process_id
    except Exception as e:
        # print(f"Erreur dans worker {process_id}: {e}") # Décommenter pour debug
        return -1, False, f"Erreur dans worker {process_id}: {e}", None, process_id

# --- Boucle Principale ---

def main():
    parser = argparse.ArgumentParser(description="Optimisation de trajet pour robot collecteur (Multiprocessing + NetworkX + Backtracking + Base Améliorée + Relance).")
    parser.add_argument("dataset_file", help="Chemin vers le fichier dataset JSON.")
    parser.add_argument("-s", "--score", type=int, default=0, help="Score minimum à atteindre. 0 pour sauvegarder la première solution valide.")
    parser.add_argument("-o", "--output", default="solutions", help="Répertoire de sortie pour les solutions.")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Nombre de processus workers à utiliser.")
    parser.add_argument("--max-stale-time", type=int, default=0, help="Temps max (secondes) sans amélioration avant de relancer la recherche. 0 = jamais relancer.")
    parser.add_argument("--max-total-time", type=int, default=0, help="Temps total max (secondes) pour la recherche (0 = illimité).")


    args = parser.parse_args()

    if not os.path.exists(args.dataset_file):
        print(f"Erreur: Le fichier dataset '{args.dataset_file}' n'existe pas.")
        exit(1)

    print(f"Chargement du dataset: {args.dataset_file}")
    dataset = load_dataset(args.dataset_file)
    print("Construction du graphe NetworkX...")
    graph, all_physical_edges_keys, edge_lengths_map = build_graph(dataset)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_file))[0]

    if not graph or not nx.is_weakly_connected(graph):
         print("Avertissement: Le graphe est vide ou non (faiblement) connecté.")

    # Variables globales pour la recherche sur plusieurs relances
    best_score = -1
    best_solution_dict = None
    total_attempts = 0
    global_start_time = time.time()


    try:
        with open(args.dataset_file, 'r') as f:
            dataset_txt = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier dataset pour le scoring: {e}")
        exit(1)

    num_workers = args.workers
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    print(f"Utilisation de {num_workers} processus workers.")
    print(f"Relance auto si pas d'amélioration pendant {args.max_stale_time}s.")
    if args.max_total_time > 0:
        print(f"Temps total max: {args.max_total_time}s.")


    target_score = args.score
    search_iteration = 0

    # Boucle externe pour gérer les relances
    while True:
        search_iteration += 1
        print(f"\n--- Démarrage Itération de Recherche #{search_iteration} ---")
        iteration_start_time = time.time()
        last_improvement_time = iteration_start_time
        iteration_attempts = 0
        found_target_in_iteration = False

        # Vérifier le temps total avant de démarrer une nouvelle itération
        if args.max_total_time > 0 and (time.time() - global_start_time) > args.max_total_time:
             print("\nTemps total maximum de recherche atteint.")
             break

        # Générateur d'arguments (recréé pour chaque itération si nécessaire, mais ici partagé)
        def argument_generator():
            for i in itertools.count(total_attempts): # Compteur global
                yield (i, dataset, graph, all_physical_edges_keys, edge_lengths_map, dataset_txt)

        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results_iterator = pool.imap_unordered(worker_task, argument_generator(), chunksize=10)

                for result in results_iterator:
                    if result is None:
                         continue

                    score, is_valid, message, solution_dict, worker_id = result
                    total_attempts += 1
                    iteration_attempts += 1

                    if is_valid and solution_dict is not None:
                        current_best_score = best_score # Stocker avant mise à jour
                        if score > best_score:
                            best_score = score
                            best_solution_dict = solution_dict
                            last_improvement_time = time.time() # Mettre à jour le temps de la dernière amélioration
                            # Affichage amélioré
                            elapsed_total_time = time.time() - global_start_time
                            rate = total_attempts / elapsed_total_time if elapsed_total_time > 0 else 0
                            print(f"\rIter {search_iteration}, Tentative ~{total_attempts}: Nouveau meilleur score = {best_score} (Station: {solution_dict.get('chargeStationId', 'N/A')}) ({rate:.1f} att/s)      ", end="")

                        # Vérifier si le score cible est atteint
                        if target_score > 0 and best_score >= target_score:
                            print(f"\nScore cible ({target_score}) atteint ou dépassé !")
                            found_target_in_iteration = True
                            break # Sortir de la boucle for (et terminera le pool)

                        # Gérer le cas score cible = 0
                        if target_score == 0 and best_solution_dict is not None and current_best_score == -1:
                             print(f"\nPremière solution valide trouvée (score={best_score}).")
                             found_target_in_iteration = True
                             break # Sortir de la boucle for

                    # Vérifier le temps écoulé sans amélioration pour relancer
                    current_time = time.time()
                    # Désactiver la relance auto si max_stale_time == 0
                    if args.max_stale_time > 0 and current_time - last_improvement_time > args.max_stale_time:
                        print(f"\nINFO: Aucune amélioration du score depuis {args.max_stale_time}s. Relance de la recherche...")
                        break # Sortir de la boucle for pour terminer ce pool et relancer

                    # Vérifier le temps total écoulé
                    if args.max_total_time > 0 and (current_time - global_start_time) > args.max_total_time:
                         print("\nTemps total maximum de recherche atteint.")
                         found_target_in_iteration = True # Pour sortir de la boucle while externe
                         break

                    # Affichage de la progression moins fréquent
                    if iteration_attempts % (num_workers * 50) == 0:
                        elapsed_total_time = time.time() - global_start_time
                        rate = total_attempts / elapsed_total_time if elapsed_total_time > 0 else 0
                        print(f"\rIter {search_iteration}, Tentative ~{total_attempts}... Meilleur score: {best_score} ({elapsed_total_time:.1f}s, {rate:.1f} att/s)      ", end="")

        except KeyboardInterrupt:
            print("\nRecherche interrompue par l'utilisateur.")
            found_target_in_iteration = True # Pour sortir de la boucle while externe
        except Exception as e:
            print(f"\nUne erreur générale est survenue pendant l'itération {search_iteration}: {e}")
            import traceback
            traceback.print_exc()
            # On pourrait décider de s'arrêter ou de continuer avec la prochaine itération
            # Pour l'instant, on arrête.
            found_target_in_iteration = True # Pour sortir de la boucle while externe
        finally:
            # Le context manager s'occupe de fermer le pool
            print(f"\n--- Fin Itération de Recherche #{search_iteration} ---")

        # Sortir de la boucle while externe si le score cible est atteint,
        # si le temps total est dépassé, ou si interrompu
        if found_target_in_iteration:
            break

    # --- Sauvegarde Finale ---
    print("-" * 30)
    if best_solution_dict:
        analyse_post_solution(best_solution_dict, graph, all_physical_edges_keys, edge_lengths_map)
        # Recalculer le score final pour être sûr
        final_score, is_valid_final, msg_final = getSolutionScore(json.dumps(best_solution_dict), dataset_txt)
        if is_valid_final:
             print(f"Meilleur score final trouvé après {total_attempts} tentatives: {final_score}")
             save_solution(best_solution_dict, dataset_name, final_score, args.output)
        else:
             print(f"Erreur: La meilleure solution enregistrée est invalide ({msg_final}). Sauvegarde annulée.")
    else:
        print(f"Aucune solution valide n'a été trouvée après {total_attempts} tentatives.")
    print(f"Temps total de recherche: {time.time() - global_start_time:.2f} secondes.")
    print("-" * 30)

def analyse_post_solution(solution_dict, graph, all_physical_edges_keys, edge_lengths_map):
    """Affiche un résumé de la couverture des rues par la solution."""
    if not solution_dict or "itinerary" not in solution_dict:
        print("Aucune solution à analyser.")
        return

    itinerary = solution_dict["itinerary"]
    visited_edges = set()
    for u, v in zip(itinerary, itinerary[1:]):
        edge_key = tuple(sorted((u, v)))
        if edge_key in all_physical_edges_keys:
            visited_edges.add(edge_key)

    total_edges = len(all_physical_edges_keys)
    visited_count = len(visited_edges)
    non_visited = all_physical_edges_keys - visited_edges
    non_visited_count = len(non_visited)
    percent_visited = 100 * visited_count / total_edges if total_edges else 0

    score_possible = sum(edge_lengths_map[e] for e in all_physical_edges_keys)
    score_obtenu = sum(edge_lengths_map[e] for e in visited_edges)

    print("\n--- Analyse post-solution ---")
    print(f"Nombre total de rues (arêtes physiques) : {total_edges}")
    print(f"Nombre de rues visitées : {visited_count} ({percent_visited:.2f}%)")
    print(f"Nombre de rues non visitées : {non_visited_count}")
    print(f"Score obtenu (distance unique) : {score_obtenu}")
    print(f"Score maximal possible : {score_possible}")
    if non_visited_count > 0:
        print("Exemples de rues non visitées :", list(non_visited)[:10])
    print("--- Fin de l'analyse ---\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()