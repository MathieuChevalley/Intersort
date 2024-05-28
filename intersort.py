"""
Copyright (C) 2024  GSK plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import scipy
import networkx as nx
from sklearn.preprocessing import StandardScaler


def _sort_ranking(score_matrix, lmbda):
    flat_array = score_matrix.flatten()
    G = nx.DiGraph()
    
    # Argsort on the flattened array
    sorted_flat_indices = np.argsort(-flat_array)

    # Mapping flat indices back to (i, j) format
    rows, cols = score_matrix.shape
    G.add_nodes_from(range(cols))
    sorted_indices_ij = np.unravel_index(sorted_flat_indices, (rows, cols))
    for k in range(len(sorted_indices_ij[0])):
        i, j = sorted_indices_ij[0][k], sorted_indices_ij[1][k]
        if i != j:
            score = score_matrix[i, j]
            if score > lmbda: 
                G.add_edge(i, j)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(i, j)
                
    return G

def _compute_scores(data, inter, d):
    score_matrix = np.zeros((d, d))
    for node in range(d):
        for var in range(d):
            if node != var: 
                data_obs = data[inter == -1, var]
                data_inter = data[inter == node, var]
                if len(data_inter) > 0:
                    w_dist = scipy.stats.wasserstein_distance(data_obs, data_inter)
                    score_matrix[node, var] = w_dist
    return score_matrix
   
def score_ordering(topological_order, score_matrix, d, eps=0.3):
    """ Score an causal order based on the observed distances"""
    tot = 0
    after = list(range(d))
    for i in topological_order:
        after.remove(i)
        if np.any(score_matrix[i, :] > 0.0):
            positive = np.sum(score_matrix[i, after] - eps)
            tot += positive
    return tot

def move_variable(perm, from_index, to_index):
    """Move a variable from from_index to to_index in the permutation."""
    if from_index == to_index:  # No move needed
        return perm
    new_perm = perm.copy()
    new_perm.insert(to_index, new_perm.pop(from_index))
    return new_perm

def generate_all_possible_moves(perm):
    """Generate all possible moves of a variable to any position."""
    moves = []
    for i in range(len(perm)):
        for j in range(len(perm)):
            if i != j:
                # Generate a move by placing i-th element to j-th position
                moved_perm = move_variable(perm, i, j)
                moves.append(moved_perm)
    return moves

def local_search_extended(initial_perm, score_matrix, d, eps=0.3):
    """Perform local search around neighborhood of initial solution."""
    current_perm = initial_perm
    current_score = score_ordering(current_perm, score_matrix, d, eps=eps)
    while True:
        all_moves = generate_all_possible_moves(current_perm)
        next_perm = None
        for move in all_moves:
            move_score = score_ordering(move, score_matrix, d)
            if move_score > current_score:  # Assuming we want to maximize the score
                next_perm = move
                current_score = move_score
                break  # Exit early if a better move is found
        if next_perm is None:
            break  # No improvement found
        current_perm = next_perm
    return current_perm


def intersort(
    data: np.array,
    interventions: np.array,
    eps: float
):      
    """Main frunction of Intersort to derive the causal order from single variable interventional data

    Args:
        data (np.array): n x d array of samples
        interventions (np.array): n x 1 array denoting for each sample whether it is observational (-1), or which variables was intervened on
        eps (float): regularization term epsilon

    Returns:
        List: predicted causal ordering of the variables (size d)
    """
    obs_indices = interventions == -1
    data_obs = data[obs_indices, :]
    scaler = StandardScaler()
    scaler.fit(data_obs)
    data = scaler.transform(data)
    d = data.shape[1]
    score_matrix = _compute_scores(data, interventions, d)
    init_solution = _sort_ranking(score_matrix, eps)
    topological_order_sortranking = list(nx.topological_sort(init_solution)) 
    causal_order = local_search_extended(topological_order_sortranking, score_matrix, d, eps=eps)
    return causal_order

