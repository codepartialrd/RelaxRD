import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import product
import heapq
import time


def build_equivalence_dict(r, lhs, rhs):
    """Builds equivalence classes based on LHS → RHS, with early termination on excessive conflicts."""
    node_count = set()
    eq_dict = defaultdict(lambda: defaultdict(list))
    for idx, t in enumerate(r):
        lhs_val = tuple(t[k] for k in lhs)
        rhs_val = tuple(t[k] for k in rhs)
        eq_dict[lhs_val][rhs_val].append(idx)
        if len(eq_dict[lhs_val]) > 1:
            for rhs_key in eq_dict[lhs_val]:
                node_count.update(eq_dict[lhs_val][rhs_key])
    return eq_dict


def find_largest_cached_subset(fd_set, fd_to_fdset_map):
    """Finds the largest cached FD subset contained in the current FD set."""
    candidate_sets = defaultdict(int)
    for fd in fd_set:
        for fs in fd_to_fdset_map.get(fd, []):
            candidate_sets[fs] += 1
    max_subset = set()
    for fs, count in candidate_sets.items():
        if count == len(fs) and len(fs) > len(max_subset):
            max_subset = fs
    return max_subset

def merge_graphs_with_cache(fd_set, graphs, r, laji_fds, graph_cache, fd_to_fdset_map):
    """Merges conflict graphs of the given FD set, using cache if possible."""
    fd_set_frozen = frozenset(fd_set)
    max_cached_subset = find_largest_cached_subset(fd_set_frozen, fd_to_fdset_map)
    graph_tmp = graph_cache[max_cached_subset].copy() if max_cached_subset else nx.Graph()
    used_fds = max_cached_subset if max_cached_subset else set()

    for fd in fd_set_frozen - used_fds:
        lhs_name = fd[0][0]
        rhs_name = fd[1]
        if fd in laji_fds:
            continue
        if lhs_name in graphs and rhs_name in graphs[lhs_name]:
            G = graphs[lhs_name][rhs_name]
        else:
            eq_dict = build_equivalence_dict(r, [lhs_name], [rhs_name])
            if not eq_dict:
                laji_fds.append(fd)
                continue
            G = nx.Graph()
            for rhs_groups in eq_dict.values():
                if len(rhs_groups) <= 1:
                    continue
                groups = list(rhs_groups.values())
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        G.add_edges_from(product(groups[i], groups[j]))
            graphs.setdefault(lhs_name, {})[rhs_name] = G
        graph_tmp.add_edges_from(G.edges())

    graph_cache[fd_set_frozen] = graph_tmp.copy()
    for fd in fd_set_frozen:
        fd_to_fdset_map[fd].add(fd_set_frozen)
    return graph_tmp


def estimate_merged_cost(df, afd_set, index_list):
    """Estimates total cost of a merged FD set on a subset of tuples."""
    merged = defaultdict(set)
    for lhs, rhs in afd_set:
        merged[lhs].add(rhs)
    dg = 0
    mf=len(index_list)
    for lhs, rhs_set in merged.items():
        dX = df.loc[index_list, list(lhs)].drop_duplicates().shape[0]
        dg+=(mf-dX)*len(rhs_set)-dX*len(lhs)
    return dg


def compute_max_independent_set(graph_tmp, n):
    """Computes a maximal independent set using minimum degree greedy strategy."""
    V = set(range(n))
    T_prime = set()
    conflict_nodes = set(graph_tmp.nodes())
    degree_map = {node: graph_tmp.degree[node] for node in conflict_nodes}
    heap = [(deg, node) for node, deg in degree_map.items()]
    heapq.heapify(heap)
    deleted = set()
    while heap:
        _, u = heapq.heappop(heap)
        if u in deleted:
            continue
        T_prime.add(u)
        deleted.update([u] + list(graph_tmp.neighbors(u)))
    T_prime.update(V - conflict_nodes)
    return T_prime


def computedg(df, afd_set, T_prime):
    """Computes total size after applying FD set and removing conflicting tuples."""
    dg = estimate_merged_cost(df, afd_set, list(T_prime))
    return dg

def cache_and_index(input_fd_sets, df, cols):
    """Main loop to evaluate each FD set and return the one with maximum gain."""
    # pdb.set_trace()
    r = df.to_dict(orient="records")
    n = len(r)
    best_afd = []
    best_dg = 0
    original_fds = set(fd for fd_set in input_fd_sets for fd in fd_set)
    graphs, graph_cache, laji_fds = {}, {}, []
    fd_to_fdset_map = defaultdict(set)

    t0 = time.time()
    for lhs, rhs in original_fds:
        eq_dict = build_equivalence_dict(r, list(lhs), [rhs])
        if not eq_dict:
            laji_fds.append((lhs, rhs))
            continue
        G = nx.Graph()
        for rhs_groups in eq_dict.values():
            if len(rhs_groups) <= 1:
                continue
            groups = list(rhs_groups.values())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    G.add_edges_from(product(groups[i], groups[j]))
        graphs.setdefault(lhs, {})[rhs] = G


    print("\nTime of Construct subgraphs:", round(time.time() - t0, 2), "s")
    print("== Merge and compute dg ==")
    for i, fd_set in enumerate(input_fd_sets, 1):
        t1 = time.time()
        effective_fd_set = [fd for fd in fd_set if fd not in laji_fds]
        if not effective_fd_set:
            print("All FDs skipped as laji.")
            continue
        graph_tmp = merge_graphs_with_cache(effective_fd_set, graphs, r, laji_fds, graph_cache, fd_to_fdset_map)
        T_prime = compute_max_independent_set(graph_tmp, n)
        dg = computedg(df, effective_fd_set, T_prime)
        print(f"AFD Set {i}: dg = {dg}, time = {round(time.time() - t1, 2)}s")
        if dg > best_dg:
            best_dg = dg
            best_afd = effective_fd_set
    return best_afd, best_dg


