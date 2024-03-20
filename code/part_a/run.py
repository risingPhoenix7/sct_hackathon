import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import os
import sys


class HeldKarp:
    def __init__(self, distance_matrix):
        self.distance = distance_matrix
        self.n = len(distance_matrix)  # Number of nodes
        self.C = []
        self.init_base_cases()

    def init_base_cases(self):
        # Initialize base cases: C({k}, k) = d[0,k]
        base_cases = dict()
        for k in range(1, self.n):
            base_cases[(1 << k, k)] = {
                'cost': self.distance[0][k], 'path': [0, k]}
        self.C.append(base_cases)

    def solve(self):
        # Dynamic programming approach
        for s in range(2, self.n):
            new_cases = dict()
            for subset in self.generate_subsets(s, self.n):
                for k in subset:
                    if k == 0:
                        continue
                    best_cost = float('inf')
                    best_prev = -1
                    subset_without_k = subset - {k}
                    subset_without_k_code = self.subset_to_code(
                        subset_without_k)
                    for m in subset:
                        if m == k:
                            continue
                        prev_cost = self.C[-1].get((subset_without_k_code, m), {
                                                   'cost': float('inf')})['cost']
                        cost_with_m_to_k = prev_cost + self.distance[m][k]
                        if cost_with_m_to_k < best_cost:
                            best_cost = cost_with_m_to_k
                            best_prev = m
                    new_cases[(self.subset_to_code(subset), k)] = {
                        'cost': best_cost, 'path': self.C[-1][(subset_without_k_code, best_prev)]['path'] + [k]}
            self.C.append(new_cases)

        # Find optimal solution
        full_set = set(range(1, self.n))
        full_set_code = self.subset_to_code(full_set)
        opt = float('inf')
        final_path = []
        for k in range(1, self.n):
            cost_with_return = self.C[-1][(full_set_code, k)
                                          ]['cost'] + self.distance[k][0]
            if cost_with_return < opt:
                opt = cost_with_return
                final_path = self.C[-1][(full_set_code, k)]['path'] + [0]

        return opt, final_path

    def generate_subsets(self, size, n):
        # Generate all subsets of size `size` from the set {1, ..., n-1}
        from itertools import combinations
        return [set(s) for s in combinations(range(1, n), size)]

    def subset_to_code(self, subset):
        # Convert a subset to its code representation
        code = 0
        for element in subset:
            code |= 1 << element
        return code


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
# Function to find the minimum key vertex from the set of vertices not yet included in MST


def findMinKey(key, mstSet, V):
    min_val = float('inf')
    min_index = -1
    for v in range(V):
        if not mstSet[v] and key[v] < min_val:
            min_val = key[v]
            min_index = v
    return min_index

# Function to perform Prim's algorithm to find the Minimum Spanning Tree (MST)


def primMST(graph, V):
    parent = [-1] * V
    key = [float('inf')] * V
    mstSet = [False] * V
    key[0] = 0  # Start from vertex 0

    for _ in range(V - 1):
        u = findMinKey(key, mstSet, V)
        mstSet[u] = True

        for v in range(V):
            if graph[u][v] and not mstSet[v] and graph[u][v] < key[v]:
                parent[v] = u
                key[v] = graph[u][v]
    return parent

# Function to find preorder traversal of the tree based on MST


def findPreorderTraversal(parent, node, visited, path, V):
    visited[node] = True
    path.append(node)
    for i in range(V):
        if parent[i] == node and not visited[i]:
            findPreorderTraversal(parent, i, visited, path, V)

# Main function for the Traveling Salesperson Approximation Algorithm


def tspApproximation(graph, V):
    # Find the Minimum Spanning Tree using Prim's Algorithm
    parent = primMST(graph, V)

    # Find preorder traversal of the MST to approximate Hamiltonian cycle
    visited = [False] * V
    path = []
    findPreorderTraversal(parent, 0, visited, path, V)
    path.append(0)  # Adding the start node at the end to complete the cycle

    # Calculate the cost of the Hamiltonian path
    cost = 0
    for i in range(1, len(path)):
        cost += graph[path[i - 1]][path[i]]

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


def main(input_filepath):
    # Parse the input filepath to generate the output filepath
    output_filepath = input_filepath.replace('input', 'output')
    output_directory = os.path.dirname(output_filepath)
    # print(output_directory)
    # print(input_filepath)
    # print(output_filepath)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the dataset
    data = read_dataset(input_filepath)

    # Create the distance matrix
    distance_matrix = create_distance_matrix(data)
    # print(distance_matrix)

    # Solve the TSP
    if len(distance_matrix) <= 13:
        total_distance, sequence = tsp_dp(distance_matrix)

    elif len(distance_matrix) <= 20:
        solver = HeldKarp(distance_matrix)
        total_distance, sequence = solver.solve()
    else:
        total_distanceG, sequenceG = greedy_tsp(distance_matrix)
        total_distanceA, sequenceA = tspApproximation(
            distance_matrix, len(distance_matrix))

        total_distance = min(total_distanceA, total_distanceG)
        if total_distanceA < total_distanceG:

            sequence = sequenceA

        else:
            sequence = sequenceG

    for i, a in enumerate(sequence):
        if a == 0:
            continue

        data.at[sequence[i] - 1, "sequence"] = i

    # print("data", data)

    data.to_csv(output_filepath, index=False)
    # print(total_distance, sequence)

    # Write the best route distance to the summary CSV file
    write_best_route_distance(input_filepath, total_distance, output_directory)


# Filepath for example, replace with parameter when calling main
# main('input_datasets/part_a/part_a_input_dataset_1.csv')
if __name__ == '__main__':
    # for (i) in range(1, 6):
    #     main(f'input_datasets/part_a/part_a_input_dataset_{i}.csv')
    if len(sys.argv) != 2:
        print("Usage: python code/part_a/run.py <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    main(input_file_path)
