import tkinter as tk
from tkinter import messagebox
import json
import os
import networkx as nx

# --- Paramètres ---
DATASET_PATH = os.path.join("datasets", "2_pacman.json")  # Change selon le dataset voulu
NODE_RADIUS = 15
WIDTH, HEIGHT = 1500, 950  # Fenêtre agrandie
MARGIN = 60

# --- Chargement du dataset ---
def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def build_graph(dataset):
    G = nx.DiGraph()
    pos = {}
    for inter in dataset["intersections"]:
        G.add_node(inter["id"])
        pos[inter["id"]] = (inter["lng"], inter["lat"])
    for road in dataset["roads"]:
        G.add_edge(road["intersectionId1"], road["intersectionId2"], length=road["length"], oneway=road["isOneWay"])
        if not road["isOneWay"]:
            G.add_edge(road["intersectionId2"], road["intersectionId1"], length=road["length"], oneway=road["isOneWay"])
    return G, pos

def scale_positions(pos):
    # Met à l'échelle les positions pour l'affichage
    xs = [lng for lng, lat in pos.values()]
    ys = [lat for lng, lat in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    def scale(lng, lat):
        x = MARGIN + (lng - min_x) / (max_x - min_x + 1e-6) * (WIDTH - 2*MARGIN)
        y = HEIGHT - (MARGIN + (lat - min_y) / (max_y - min_y + 1e-6) * (HEIGHT - 2*MARGIN))
        return x, y
    return {nid: scale(lng, lat) for nid, (lng, lat) in pos.items()}

class MapGame:
    def __init__(self, root, dataset):
        self.dataset = dataset
        self.G, self.pos = build_graph(dataset)
        self.pos = scale_positions(self.pos)
        self.base_id = None  # À choisir par l'utilisateur
        self.current = None
        self.battery = dataset["batteryCapacity"]
        self.battery_max = dataset["batteryCapacity"]
        self.days = dataset["numDays"]
        self.day = 1
        self.visited_edges = set()
        self.itinerary = []
        self.history = []  # Pour l'annulation
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()
        self.info = tk.Label(root, text="", font=("Arial", 14))
        self.info.pack()
        self.undo_btn = tk.Button(root, text="Annuler", font=("Arial", 12), command=self.undo_move)
        self.undo_btn.pack(pady=8)
        self.draw()
        self.info.config(text="Cliquez sur une intersection pour choisir la base de départ.")
        self.canvas.bind("<Button-1>", self.choose_base)

    def choose_base(self, event):
        for nid, (x, y) in self.pos.items():
            if (event.x - x)**2 + (event.y - y)**2 <= NODE_RADIUS**2:
                self.base_id = nid
                self.current = nid
                self.itinerary = [nid]
                self.draw()
                self.update_info()
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<Button-1>", self.on_click)
                return

    def draw(self):
        self.canvas.delete("all")
        # Dessiner les arêtes
        for u, v, data in self.G.edges(data=True):
            key = tuple(sorted((u, v)))
            color = "red" if key in self.visited_edges else "#888"
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=4 if color=="red" else 2)
            # Si sens unique, dessiner une flèche
            if data.get("oneway", False):
                import math
                fx = x1 + 0.66 * (x2 - x1)
                fy = y1 + 0.66 * (y2 - y1)
                angle = math.atan2(y2 - y1, x2 - x1)
                arrow_len = 16
                p1 = (fx, fy)
                p2 = (fx - arrow_len * math.cos(angle - math.pi/8), fy - arrow_len * math.sin(angle - math.pi/8))
                p3 = (fx - arrow_len * math.cos(angle + math.pi/8), fy - arrow_len * math.sin(angle + math.pi/8))
                self.canvas.create_polygon(p1, p2, p3, fill=color, outline=color)
        # Dessiner les nœuds
        for nid, (x, y) in self.pos.items():
            # Vérifie si toutes les arêtes incidentes sont visitées
            all_visited = True
            for neighbor in self.G.neighbors(nid):
                key = tuple(sorted((nid, neighbor)))
                if key not in self.visited_edges:
                    all_visited = False
                    break
            for neighbor in self.G.predecessors(nid):
                key = tuple(sorted((nid, neighbor)))
                if key not in self.visited_edges:
                    all_visited = False
                    break
            if all_visited and self.G.degree(nid) > 0:
                fill = "#FFD700"  # Jaune/or pour les nœuds "complets"
            elif nid == self.base_id:
                fill = "blue"
            elif nid == self.current:
                fill = "green"
            else:
                fill = "white"
            outline = "black"
            self.canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS, fill=fill, outline=outline, width=3)
            self.canvas.create_text(x, y, text=str(nid), font=("Arial", 12, "bold"))

    def update_info(self):
        self.info.config(text=f"Jour {self.day}/{self.days} | Batterie : {self.battery}/{self.battery_max} | Position : {self.current} | Base : {self.base_id}")

    def save_state(self):
        # Sauvegarde l'état courant pour pouvoir annuler
        state = {
            'current': self.current,
            'battery': self.battery,
            'day': self.day,
            'visited_edges': set(self.visited_edges),
            'itinerary': list(self.itinerary)
        }
        self.history.append(state)

    def undo_move(self):
        if not self.history:
            messagebox.showinfo("Annuler", "Aucun déplacement à annuler.")
            return
        state = self.history.pop()
        self.current = state['current']
        self.battery = state['battery']
        self.day = state['day']
        self.visited_edges = set(state['visited_edges'])
        self.itinerary = list(state['itinerary'])
        self.draw()
        self.update_info()

    def on_click(self, event):
        # Cherche le nœud cliqué
        for nid, (x, y) in self.pos.items():
            if (event.x - x)**2 + (event.y - y)**2 <= NODE_RADIUS**2:
                if nid == self.current:
                    return  # Clique sur soi-même
                if not self.G.has_edge(self.current, nid):
                    messagebox.showinfo("Déplacement impossible", "Pas de route directe !")
                    return
                length = self.G[self.current][nid]["length"]
                if length > self.battery:
                    messagebox.showinfo("Batterie insuffisante", "Pas assez de batterie pour ce déplacement !")
                    self.end_game()  # Enregistre la solution si batterie insuffisante
                    return
                # Sauvegarde l'état avant de bouger
                self.save_state()
                # Déplacement OK
                self.battery -= length
                self.itinerary.append(nid)
                key = tuple(sorted((self.current, nid)))
                self.visited_edges.add(key)
                self.current = nid
                # Recharge si sur la base
                if self.current == self.base_id:
                    if self.day < self.days:
                        self.battery = self.battery_max
                        self.day += 1
                        messagebox.showinfo("Recharge", f"Recharge complète ! Début du jour {self.day}.")
                    else:
                        messagebox.showinfo("Fin du jeu", "Nombre de jours maximum atteint.")
                        self.end_game()
                        return
                self.draw()
                self.update_info()
                # Vérifie si toutes les arêtes sont visitées
                if len(self.visited_edges) == len({tuple(sorted((u, v))) for u, v in self.G.edges()}):
                    messagebox.showinfo("Bravo !", "Toutes les rues ont été visitées !")
                    self.end_game()
                return
        # Clique ailleurs : rien

    def end_game(self):
        self.canvas.unbind("<Button-1>")
        score = sum(self.G[u][v]["length"] for u, v in self.visited_edges if self.G.has_edge(u, v))
        messagebox.showinfo("Score final", f"Score (distance unique) : {score}\nItinéraire : {self.itinerary}")
        # Sauvegarde du résultat au format attendu
        result = {
            "chargeStationId": self.base_id,
            "itinerary": self.itinerary
        }
        import datetime
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
        filename = f"solutions/{dataset_name}_manual_{now}_score{score}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        messagebox.showinfo("Fichier sauvegardé", f"Résultat sauvegardé dans :\n{filename}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Cartographie optimale - Jeu interactif")
    dataset = load_dataset(DATASET_PATH)
    game = MapGame(root, dataset)
    root.mainloop()
