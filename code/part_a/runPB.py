import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import os
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


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

    # Convert Eulerian tour to Hamiltonian path (TSP solution) by skipping visited nodes
    path = []
    for u, v in eulerian_tour:
        if u not in path:
            path.append(u)
    path.append(path[0])  # to make it a cycle
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
    total_distance, sequence = christofides_algorithm(distance_matrix)

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
    main(f'input_datasets/part_a/part_a_input_dataset_3.csv')

    # for (i) in range(1, 6):
    #     main(f'input_datasets/part_a/part_a_input_dataset_{i}.csv')
