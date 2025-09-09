import heapq
import math
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import random
import string
import itertools


def heuristic(node, goal, coordinates):
    x1, y1 = coordinates.get(node, (0, 0))
    x2, y2 = coordinates.get(goal, (0, 0))
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#DFS
def dfs(graph, start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    visited_order = []
    while stack:
        current, path, cost = stack.pop()
        if current not in visited:
            visited.add(current)
            visited_order.append(current)
            if current == goal:
                return path, cost, len(visited), visited_order
            for neighbor, weight in sorted(graph.get(current, []), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + weight))
    return None, float('inf'), len(visited), visited_order

#BFS
def bfs(graph, start, goal):
    queue = deque([(start, [start], 0)])
    visited = {start}
    visited_order = [start]
    while queue:
        current, path, cost = queue.popleft()
        if current == goal:
            return path, cost, len(visited), visited_order
        for neighbor, weight in sorted(graph.get(current, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                queue.append((neighbor, path + [neighbor], cost + weight))
    return None, float('inf'), len(visited), visited_order

#Dijkstra
def dijkstra(graph, start, goal):
    pq = [(0, start, [start])]
    visited = {}
    tracker = set()
    while pq:
        cost, current, path = heapq.heappop(pq)
        if current not in tracker:
            tracker.add(current)
        if current in visited and visited[current] < cost:
            continue
        visited[current] = cost
        if current == goal:
            return path, cost, len(tracker), list(tracker)
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited or visited[neighbor] > cost + weight:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
    return None, float('inf'), len(tracker), list(tracker)

#A*
def a_star(graph, start, goal, coordinates):
    pq = [(heuristic(start, goal, coordinates), 0, start, [start])]
    visited = {}
    tracker = set()
    while pq:
        f_score, g_score, current, path = heapq.heappop(pq)
        if current not in tracker:
            tracker.add(current)
        if current in visited and visited[current] <= g_score:
            continue
        visited[current] = g_score
        if current == goal:
            return path, g_score, len(tracker), list(tracker)
        for neighbor, weight in graph.get(current, []):
            new_g = g_score + weight
            if neighbor not in visited or visited[neighbor] > new_g:
                heapq.heappush(pq, (new_g + heuristic(neighbor, goal, coordinates), new_g, neighbor, path + [neighbor]))
    return None, float('inf'), len(tracker), list(tracker)

#  Graph Drawing 
def draw_graph(graph, pos, start_node, goal_node, path=None, title="Pipe Network"):
    G = nx.Graph()
    edge_labels = {}
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)
            if (node, neighbor) not in edge_labels:
                edge_labels[(node, neighbor)] = str(int(weight))
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color='orange', width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_node], node_color='red', node_size=700)
    plt.title(title)
    plt.show(block=False)
    plt.pause(1)

# Manual Input with Error Handling
def get_user_input():
    graph = {}
    coordinates = {}

    while True:
        try:
            num_pipes = int(input("Enter total number of pipes: "))
            if num_pipes > 0:
                break
            else:
                print(" Must be positive.")
        except ValueError:
            print(" Invalid input, enter a number.")

    print("Enter each pipe as: Junction1 Junction2 Cost")
    for i in range(num_pipes):
        while True:
            try:
                j1, j2, cost = input(f"Pipe #{i+1}: ").split()
                cost = float(cost)
                if cost <= 0:
                    print(" Cost must be > 0")
                    continue
                graph.setdefault(j1, []).append((j2, cost))
                graph.setdefault(j2, []).append((j1, cost))
                break
            except ValueError:
                print(" Format must be: A B 5")

    print("\nEnter coordinates for each junction (X Y):")
    for j in graph.keys():
        while True:
            try:
                x, y = map(float, input(f"Coordinates of {j}: ").split())
                coordinates[j] = (x, y)
                break
            except ValueError:
                print(" Must be two numbers, e.g., 0 5")

    while True:
        start = input("Enter start junction: ")
        if start in graph:
            break
        else:
            print(" Junction not found. Try again.")

    while True:
        goal = input("Enter goal junction: ")
        if goal in graph:
            break
        else:
            print(" Junction not found. Try again.")

    return graph, coordinates, start, goal

# Random Graph


def generate_random_graph():
    
    num_junctions = random.randint(10, 20)  # always at least 10
    junctions = list(string.ascii_uppercase)
    if num_junctions > len(junctions):  # extend if > 26
        junctions += [''.join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
    junctions = junctions[:num_junctions]

    # initialize graph + coordinates
    graph = {j: [] for j in junctions}
    coords = {j: (random.randint(0, 100), random.randint(0, 100)) for j in junctions}

    # Step 1: Ensure connectivity with a spanning tree
    for i in range(num_junctions - 1):
        a, b = junctions[i], junctions[i+1]
        cost = random.randint(5, 20)
        graph[a].append((b, cost))
        graph[b].append((a, cost))

    # Step 2: Add random extra edges for complexity
    extra_edges = random.randint(num_junctions, num_junctions * 3)  # lots of loops
    for _ in range(extra_edges):
        a, b = random.sample(junctions, 2)
        cost = random.randint(5, 20)
        if all(neigh != b for neigh, _ in graph[a]):  # avoid duplicate
            graph[a].append((b, cost))
            graph[b].append((a, cost))

    # Step 3: Pick random start & goal
    start, goal = random.sample(junctions, 2)

    print(f" Generated complex graph with {num_junctions} junctions "
          f"and {sum(len(v) for v in graph.values()) // 2} pipes.")
    print(f"Start: {start}, Goal: {goal}")

    return graph, coords, start, goal

# Main
if __name__ == "__main__":
    
    while True:
        choice = input("Choose input method (1=Manual, 2=Random): ")
        if choice in ["1", "2"]:
            break
        else:
            print(" Invalid choice, enter 1 or 2.")

    if choice == "1":
        graph, coords, start, goal = get_user_input()
    else:
        graph, coords, start, goal = generate_random_graph()

    print(f"\n Rat starts at {start}, needs to reach {goal}\n")

    results = {
        "DFS": dfs(graph, start, goal),
        "BFS": bfs(graph, start, goal),
        "Dijkstra": dijkstra(graph, start, goal),
        "A*": a_star(graph, start, goal, coords)
    }

    draw_graph(graph, coords, start, goal, title="Full Pipe Network")

    for name, (path, cost, visited, order) in results.items():
        print(f"\n--- {name} ---")
        if path:
            print(f"Path: {' -> '.join(path)}")
            print(f"Cost: {cost}")
            print(f"Visited: {visited}")
            draw_graph(graph, coords, start, goal, path, title=f"{name} Path")
        else:
            print("No path found")
        plt.show()                                              
    
