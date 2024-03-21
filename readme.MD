
# Walmart Hackathon Supply Chain (SCT) Hackathon - Round 1: Route Optimization

## Team Members
- Achinthya Hebbar
- Yash Pandey
- Ujjwal Aggarwal

## Overview
In this project, we tackle the Route Optimization challenge presented in Round 1 of the Walmart Supply Chain Technology Hackathon. Our goal is to find the best delivery route for an eCommerce organization, like Walmart, to efficiently deliver packages from a depot/hub to customer locations and then return to the depot. We focus on optimizing the delivery process by ensuring each customer location is visited exactly once, minimizing the total travel distance.

## Problem Statement
Given a set of customer locations and a depot, we aim to find the shortest possible route that starts at the depot, visits each customer location exactly once, and returns to the depot. This problem is known as the Travelling Salesman Problem (TSP), a well-known NP-hard problem in combinatorial optimization.

## Solution Approach
We employed three different algorithms to address the TSP based on the size of the dataset:

### 1. Dynamic Programming (DP)
**Applicability:** Smaller datasets (Datasets 1 & 2)

**Description:** Dynamic programming solves the TSP by breaking it into smaller subproblems, solving each once, and storing their solutions to avoid redundant calculations. It iteratively computes the shortest path for all possible subsets of cities, optimizing the sequence of visits to minimize total distance.

**Why it works:** DP is efficient for smaller datasets as it explores all possible city sequences and stores optimal solutions to subproblems, allowing for the computation of the overall optimal solution.

### 2. Held-Karp Algorithm
**Applicability:** Mid-length datasets (Dataset 3)

**Description:** An optimized dynamic programming approach, the Held-Karp algorithm reduces the number of subproblems by exploiting the TSP's properties, such as the triangular inequality. It iteratively computes the optimal solution for each city subset, employing optimizations to improve efficiency.

**Why it works:** By exploiting the TSP's structural properties, the Held-Karp algorithm prunes the search space, reducing time complexity and enhancing efficiency over a naive DP approach.

### 3. Greedy Algorithm
**Applicability:** Larger datasets (Datasets 4 & 5)

**Description:** The greedy algorithm selects the next city based on local information, typically choosing the nearest unvisited city. It iteratively constructs a tour until all cities are visited and the tour returns to the starting city.

**Why it works:** Its simplicity and efficiency make the greedy algorithm suitable for small to medium-sized TSP instances, producing good solutions quickly. However, it may not find optimal solutions for larger or more complex instances.

