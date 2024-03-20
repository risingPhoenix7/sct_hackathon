import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import os
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

import numpy as np


class Graph:
    # Placeholder for the Graph class
    def __init__(self, number_of_nodes, edges=[]):
        self.number_of_nodes = number_of_nodes
        self.edges = edges  # List of tuples (from, to, cost)

    def make_connected(self, distance_matrix):
        # Example implementation to create a connected graph
        pass

    def cost(self):
        # Calculate cost of the graph
        return sum(edge[2] for edge in self.edges)

    def degree(self):
        # Calculate degree of each node
        degree_dict = {node: 0 for node in range(self.number_of_nodes)}
        for edge in self.edges:
            degree_dict[edge[0]] += 1
            degree_dict[edge[1]] += 1
        return degree_dict

    def has_cycle(self):
        # Placeholder for cycle detection
        return False


def kruskal_solve(graph):
    # Placeholder for Kruskal's algorithm
    return Graph(graph.number_of_nodes)


def approx_tsp_solve(distance_matrix):
    # Placeholder for ApproxTSP solution
    return 0, ""  # Return cost and path


class LagrangianRelaxation:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.number_of_nodes = len(distance_matrix)
        self.UB = float('inf')  # Upper bound

    def upper_bound(self):
        opt, path = approx_tsp_solve(self.distance_matrix)
        return opt

    def lagrange_subgradient(self, graph):
        best_zero_tree_cost = float('-inf')
        best_zero_tree = None
        π = np.zeros(self.number_of_nodes)
        d_k_prev = {}
        M = (self.number_of_nodes ** 2 // 50) + self.number_of_nodes + 16
        t_1 = t_k = 0

        for k in range(1, M+1):
            zero_tree = kruskal_solve(graph)
            if not zero_tree.edges:
                break

            d_k = zero_tree.degree()
            zero_tree_cost = zero_tree.cost() + 2 * np.dot(π, list(d_k.values()))

            if zero_tree_cost > best_zero_tree_cost or k == 1:
                best_zero_tree_cost = zero_tree_cost
                best_zero_tree = zero_tree
                t_1 = 0.01 * zero_tree_cost
                if zero_tree_cost > self.UB:
                    break

            if zero_tree.has_cycle():
                break

            # Update π and edge costs for next iteration
            # Placeholder for updates

        return best_zero_tree

    def solve(self):
        self.UB = self.upper_bound()
        G = Graph(self.number_of_nodes)
        G.make_connected(self.distance_matrix)
        LB = 0
        zero_tree = self.lagrange_subgradient(G)
        opt = LB
        path = ""  # Placeholder for path reconstruction
        return opt, path


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
    if len(distance_matrix) <= 13:
        total_distance, sequence = tsp_dp(distance_matrix)
    else:
        solver = LagrangianRelaxation(distance_matrix)
        total_distance, sequence = solver.solve()

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
