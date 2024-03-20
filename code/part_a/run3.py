import math
from typing import List, Tuple
import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import os
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from typing import List, Tuple

def create_graph_from_matrix(dist_matrix):
    print(dist_matrix)
    G = nx.Graph()
    n = len(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i][j])
    return G


def find_minimum_spanning_tree(G):
    return nx.minimum_spanning_tree(G)


def find_odd_degree_vertices(T):
    return [v for v in T.nodes() if T.degree(v) % 2 == 1]


def minimum_weight_perfect_matching(G, odd_vertices):
    subgraph = G.subgraph(odd_vertices)
    num_vertices = len(subgraph)
    weight_matrix = np.zeros((num_vertices, num_vertices))
    for i, u in enumerate(subgraph.nodes()):
        for j, v in enumerate(subgraph.nodes()):
            if u != v:
                try:
                    # Try to get the weight from the subgraph
                    weight = subgraph[u][v]['weight']
                except KeyError:
                    # If not present, use the original graph's weight
                    weight = G[u][v]['weight']

                weight_matrix[i][j] = weight

            else:
                weight_matrix[i][j] = -100000000000

    print("Weight matrix", weight_matrix)
    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    print("row_ind", row_ind, "col_ind", col_ind)
    matching = [(odd_vertices[row_ind[i]], odd_vertices[col_ind[i]])
                for i in range(len(row_ind))]

    return matching


def create_eulerian_multigraph(T, matching):
    multigraph = nx.MultiGraph(T)
    multigraph.add_edges_from(matching)
    print(matching, "in eulerian multigraph")
    print(multigraph.edges())
    for v in multigraph.nodes():
        if multigraph.degree(v) % 2 != 0:
            print(f"Vertex {v} has an odd degree.")
    if not nx.is_eulerian(multigraph):
        print("The multigraph is not Eulerian.")

    return multigraph


def find_eulerian_tour(multigraph):
    return list(nx.eulerian_circuit(multigraph))


def held_karp(dist_matrix, eulerian_tour):
    n = len(dist_matrix)
    all_sets = [frozenset({0, 2}) for i in range(1, n)]
    memo = {}

    # Base case: distance from starting node to itself is 0
    memo[(frozenset([0]), 0)] = 0

    # Initialize entries for subsets of size 1
    for k in range(1, n):
        memo[(frozenset([0, k]), k)] = dist_matrix[0][k]

    # Initialize entries for subsets containing the starting node
    for r in range(2, n + 1):
        for subset in itertools.combinations(range(1, n), r - 1):
            subset = frozenset(subset) | {0}
            for k in subset - {0}:
                memo[(subset, k)] = float('inf')

    # Main loop
    for r in range(2, n + 1):
        for subset in itertools.combinations(range(1, n), r - 1):
            subset = frozenset(subset) | {0}
            for k in subset - {0}:
                memo[(subset, k)] = min(
                    memo[(subset - {k}, m)] + dist_matrix[m][k] for m in subset if m != k
                )

    # Find the minimum Hamiltonian cycle guided by the Eulerian tour
    min_cycle_length = float('inf')
    min_cycle_path = []
    for i in range(len(eulerian_tour) - 1):
        u, v = eulerian_tour[i], eulerian_tour[i + 1]
        length = memo[(all_sets, u)] + dist_matrix[u][v]
        if length < min_cycle_length:
            min_cycle_length = length
            min_cycle_path = [u, v]

    # Reconstruct the Hamiltonian cycle
    last_node = min_cycle_path[-1]
    remaining_nodes = set(range(n))
    remaining_nodes.remove(last_node)
    while remaining_nodes:
        next_node = min(remaining_nodes, key=lambda x: memo[(
            all_sets, x)] + dist_matrix[x][last_node])
        min_cycle_path.append(next_node)
        remaining_nodes.remove(next_node)
        last_node = next_node

    return min_cycle_length, min_cycle_path


def christofides_algorithm(dist_matrix):
    for i in range(len(dist_matrix)):
        # or a very large number if `np.inf` causes issues
        dist_matrix[i][i] = float('inf')

    G = create_graph_from_matrix(dist_matrix)
    T = find_minimum_spanning_tree(G)
    odd_vertices = find_odd_degree_vertices(T)
    print("odd vertices", odd_vertices)
    matching = minimum_weight_perfect_matching(G, odd_vertices)
    # check for odd verticies in matching
    print("matching", matching)

    unique_matching = set([tuple(sorted([u, v])) for u, v in matching])

    multigraph = create_eulerian_multigraph(T, list(unique_matching))
    eulerian_tour = find_eulerian_tour(multigraph)

    path = held_karp(dist_matrix, eulerian_tour)

    # Convert Eulerian tour to Hamiltonian path (TSP solution) by skipping visited nodes
    # path = []
    # for u, v in eulerian_tour:
    #     if u not in path:
    #         path.append(u)
    # path.append(path[0])  # to make it a cycle

    return calculate_distance_from_path(dist_matrix, path), path


def calculate_distance_from_path(dist_matrix, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_matrix[path[i]][path[i + 1]]
    return distance

# Read the dataset


def read_dataset(filepath):
    return pd.read_csv(filepath)

# Define the haversine function to calculate distances


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(delta_lambda/2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

# Function to create the distance matrix


def create_distance_matrix(data):
    # Add depot as the first node
    num_customers = data.shape[0]
    num_nodes = num_customers + 1  # Customers + 1 depot
    dist_matrix = np.zeros((num_nodes, num_nodes))

    # Get depot coordinates
    depot_lat = data['depot_lat'].iloc[0]
    depot_lng = data['depot_lng'].iloc[0]

    # Fill in the distances from the depot to each customer and vice versa
    for i, customer in data.iterrows():
        dist_matrix[0][i+1] = haversine(depot_lat,
                                        depot_lng, customer['lat'], customer['lng'])
        dist_matrix[i+1][0] = dist_matrix[0][i+1]  # Symmetric distance

    # Fill in the distances between each pair of customer locations
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                dist_matrix[i+1][j+1] = haversine(data.iloc[i]['lat'], data.iloc[i]['lng'],
                                                  data.iloc[j]['lat'], data.iloc[j]['lng'])

    return dist_matrix


# Dynamic Programming approach to solve the TSP problem
#
def tsp_dp(dist_matrix):
    n = dist_matrix.shape[0]  # Number of nodes
    memo = [[-1] * (1 << n) for _ in range(n)]
    parent = [[-1] * (1 << n) for _ in range(n)]

    def visit(city, visited):
        if visited == ((1 << n) - 1):
            return dist_matrix[city][0]  # Return to the starting city

        if memo[city][visited] != -1:
            return memo[city][visited]

        min_dist = float('inf')
        for next_city in range(n):
            if not visited & (1 << next_city):
                new_visited = visited | (1 << next_city)
                dist_to_next_city = dist_matrix[city][next_city] + \
                    visit(next_city, new_visited)
                if dist_to_next_city < min_dist:
                    min_dist = dist_to_next_city
                    parent[city][visited] = next_city

        memo[city][visited] = min_dist
        return min_dist

    # Start the TSP from the depot, which is city 0
    min_tour_cost = visit(0, 1 << 0)
    path = [0]
    last = 0
    visited = 1 << 0

    # Construct the path using the parent pointers
    while True:
        last = parent[last][visited]
        if last == -1:
            break  # Completed the cycle

        path.append(last)
        visited |= (1 << last)

        if visited == ((1 << n) - 1):
            break

    path.append(0)  # End the path at the depot
    return min_tour_cost, path


def greedy_tsp(dist_matrix):
    n = dist_matrix.shape[0]  # Number of cities
    visited = np.zeros(n, dtype=int)  # Tracks visited cities
    path = [0]  # Starting city
    visited[0] = 1  # Mark the starting city as visited
    cost = 0  # Accumulated cost

    current_city = 0
    while len(path) < n:
        next_city = None
        min_dist = np.inf
        for city in range(n):
            if not visited[city] and dist_matrix[current_city][city] < min_dist:
                min_dist = dist_matrix[current_city][city]
                next_city = city
        if next_city is not None:
            path.append(next_city)
            visited[next_city] = 1
            cost += min_dist
            current_city = next_city

    # Add the cost to return to the starting city
    cost += dist_matrix[current_city][0]
    path.append(0)  # Complete the cycle by returning to the start

    return cost, path


def write_best_route_distance(input_filepath, distance, output_directory):
    # Construct the output file path
    output_filepath = os.path.join(
        output_directory, 'part_a_best_routes_distance_travelled.csv')
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    # Append the new data
    with open(output_filepath, 'a') as f:
        f.write(f"{base_filename},{distance}\n")


def write_output(data, sequence, output_filepath):
    # Add a new column for the sequence
    # data['sequence'] = pd.Series(sequence, index=data.index)
    # Save the dataframe to a CSV file
    data.to_csv(output_filepath, index=False)

# The main function

import math
from typing import List, Tuple

class LagrangianRelaxation:
    def __init__(self, distance_matrix: List[List[float]]):
        self.distance = distance_matrix
        self.number_of_nodes = len(distance_matrix)
        self.UB = 0.0

    def print_path(self) -> str:
        return ""

    def upper_bound(self) -> float:
        opt = 0.0
        path = ""

        # You need to implement the ApproxTSP class separately
        approx_tsp = ApproxTSP(self.distance)
        opt, _ = approx_tsp.silent_solve()
        
        return opt

    def lagrange_sub_gradient(self, G, best_zero_tree_cost: float):
        t_1 = 0.0
        t_k = 0.0
        k = 0
        M = ((self.number_of_nodes * self.number_of_nodes) // 50) + self.number_of_nodes + 16
        constraint1 = 2.0 * (M - 1.0) * (M - 2.0)
        constraint2 = M * (2.0 * M - 3.0)

        # You need to implement the Kruskal class separately
        kruskal = Kruskal()

        π = [0.0] * self.number_of_nodes
        best_zero_tree = None

        while k < M:
            k += 1
            current_cardinality = k

            zero_tree = kruskal.solve(G)

            if not zero_tree.edges:
                break

            d_k = zero_tree.degree()

            if k == 1:
                d_k_prev = d_k.copy()

            zero_tree_cost = zero_tree.cost()

            for i in range(self.number_of_nodes):
                zero_tree_cost += π[i] * 2.0

            if zero_tree_cost > best_zero_tree_cost or k == 1:
                best_zero_tree_cost = zero_tree_cost
                best_zero_tree = zero_tree

                t_1 = 0.01 * zero_tree_cost

                if zero_tree_cost > self.UB:
                    break

            if zero_tree.has_cycle():
                break

            t_k = t_1 * ((k * k - 3.0 * (M - 1.0) * k + constraint2) / constraint1)

            for i, node in enumerate(zero_tree.nodes):
                π[node.id] += 0.6 * t_k * (2 - d_k[node.id]) + 0.4 * t_k * (2 - d_k_prev[node.id])

            d_k_prev = d_k.copy()

            for edge in G.edges:
                edge.cost = self.distance[edge.from_node.id][edge.to_node.id] - π[edge.from_node.id] - π[edge.to_node.id]

        return best_zero_tree

    def solve(self) -> Tuple[float, str]:
        G = Graph(self.number_of_nodes)
        G.make_connected(self.distance)

        self.UB = self.upper_bound()
        LB = 0.0
        zero_tree = self.lagrange_sub_gradient(G, LB)

        opt = LB
        path = self.print_path()

        return opt, path


class HeldKarp:
    def __init__(self, distance_matrix: List[List[float]]):
        self.distance = distance_matrix
        self.number_of_nodes = len(distance_matrix)
        self.C = [{} for _ in range(self.number_of_nodes)]
        self.max_cardinality = self.number_of_nodes
        self.current_cardinality = 0


    def powered2_code(self, s: List[int]) -> int:
        code = 0
        unique_values = set()
        for i in s:
            if i not in unique_values:
                code += 1 << i
                unique_values.add(i)
        return code

    def add_new_to_queue(self):
        self.C.insert(0, {})

    def combinations_free_mem(self, Q: List[int], S: List[int], K: int, N: int, s_cur: int):
        i = len(Q)
        s = Q[-1]

        while len(Q) > 0:
            s = Q.pop()

            while s < N:
                s += 1
                if i == 0 and s > s_cur:
                    return

                S[i] = s
                Q.append(s)
                i += 1

                if i == K:
                    code = self.powered2_code(S)
                    if code in self.C[0]:
                        del self.C[0][code]

                    break

    def combinations(self, K: int, N: int):
        Q = [0]
        while Q:  # While Q is not empty
            i = len(Q)
            s = Q[-1]

            while s < N:
                if i == 0 and s > 0:
                    self.combinations_free_mem(Q, [0] * (K - 1), K - 1, N, s)

                s += 1
                Q.append(s)
                i += 1

                if i == K:
                    code = self.powered2_code(Q)
                    temp_C_back = self.C[0][code] if code in self.C[0] else {}
                    for k in Q:
                        π = 0
                        opt = math.inf

                        code_k = self.powered2_code(Q) - (1 << k)
                        temp_C_k = self.C[1][code_k] if code_k in self.C[1] else {
                        }

                        for m in Q:
                            if m != k and m in temp_C_k:
                                tmp = temp_C_k[m]['cost'] + self.distance[m][k]

                                if tmp < opt:
                                    opt = tmp
                                    π = m

                        if π != 0:
                            temp_C_back[k] = dict(
                                path=temp_C_k[π]['path'] + [π], cost=opt)

                    self.C[0][code] = temp_C_back
                    break

            Q.pop()  # Remove the last element from Q to avoid infinite loop

    def solve(self) -> Tuple[float, List[int]]:
        # Initialize C for k = 2
        self.add_new_to_queue()
        for k in range(1, self.number_of_nodes):
            self.C[0][1 << k] = {
                k: {'path': [0, k], 'cost': self.distance[0][k]}}

        # Loop over cardinalities from 3 to n
        for self.current_cardinality in range(2, self.number_of_nodes):
            self.add_new_to_queue()
            self.combinations(self.current_cardinality,
                              self.number_of_nodes - 1)
            self.C.pop(0)

        # Find optimal path
        opt = 10000000
        path = []
        full_set = list(range(1, self.number_of_nodes))
        code = self.powered2_code(full_set)
        for k in range(1, self.number_of_nodes):
            if self.C[0][code].get(k) is not None:
                tmp = self.C[0][code][k]['cost'] + self.distance[k][0]
                if tmp < opt:
                    opt = tmp
                    path = self.C[0][code][k]['path'] + [0]

        return opt, path


def main(input_filepath):
    # Parse the input filepath to generate the output filepath
    output_filepath = input_filepath.replace('input', 'output')
    output_directory = os.path.dirname(output_filepath)
    print(output_directory)
    print(input_filepath)
    print(output_filepath)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the dataset
    data = read_dataset(input_filepath)

    # Create the distance matrix
    distance_matrix = create_distance_matrix(data)
    # print(distance_matrix)

    # Solve the TSP
    # if len(distance_matrix) <= 13:
    #     total_distance, sequence = tsp_dp(distance_matrix)
    # else:
    #     total_distance, sequence = christofides_algorithm(distance_matrix)

    tsp_solver = LagrangianRelaxation(distance_matrix)

    # Solve the TSP
    total_distance, sequence = tsp_solver.solve()

    for i, a in enumerate(sequence):
        if a == 0:
            continue

        data.at[sequence[i] - 1, "sequence"] = i

    # print("data", data)

    data.to_csv(output_filepath, index=False)
    print(total_distance, sequence)

    # Write the best route distance to the summary CSV file
    write_best_route_distance(input_filepath, total_distance, output_directory)


# Filepath for example, replace with parameter when calling main
# main('input_datasets/part_a/part_a_input_dataset_1.csv')
if __name__ == '__main__':
    for (i) in range(1, 6):
        main(f'input_datasets/part_a/part_a_input_dataset_{i}.csv')
