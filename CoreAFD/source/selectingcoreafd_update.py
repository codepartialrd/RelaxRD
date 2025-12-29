import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
from collections import defaultdict

def build_equivalence_dict_vectorized(df, lhs, rhs):
    rhs_col = rhs[0]
    grouped = df.groupby(lhs + [rhs_col]).size().reset_index(name='count')
    lhs_rhs_counts = grouped.groupby(lhs).size().reset_index(name='rhs_count')
    valid_lhs = lhs_rhs_counts[lhs_rhs_counts['rhs_count'] > 1]
    grouped = grouped.merge(valid_lhs[lhs], on=lhs, how='inner')
    grouped['max_count'] = grouped.groupby(lhs)['count'].transform('max')
    grouped['is_max'] = grouped['count'] == grouped['max_count']
    grouped = grouped.reset_index(drop=False).rename(columns={'index': 'row_id'})
    first_max_rows = grouped[grouped['is_max']].groupby(lhs)['row_id'].first().reset_index()
    first_max_rows['keep'] = True
    grouped = grouped.merge(first_max_rows, on=lhs + ['row_id'], how='left')
    grouped['keep'] = grouped['keep'].fillna(False).astype(bool)
    conflict_rhs_rows = grouped[~grouped['keep']]
    df['orig_index'] = df.index
    merge_cols = lhs + [rhs_col]
    conflict_matches = df.merge(conflict_rhs_rows[merge_cols], on=merge_cols, how='inner')
    t_set = set(conflict_matches['orig_index'])
    return 1, t_set

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

def computedg(df, afd_set, T_prime):
    """Computes total size after applying FD set and removing conflicting tuples."""
    dg = estimate_merged_cost(df, afd_set, list(T_prime))
    return dg

def get_t_set_for_fd_set(fd_set, afd_sets, total_count):
    conflict_union = set()
    for lhs, rhs in fd_set:
        key = (tuple(lhs), rhs)
        if key in afd_sets:
            conflict_union.update(afd_sets[key])
    all_indices = set(range(total_count))
    return all_indices - conflict_union

def cache_and_index(input_fd_sets, df, cols):
    """Main loop to evaluate each FD set and return the one with maximum gain."""
    r = df.to_dict(orient="records")
    n = len(r)
    best_afd = []
    best_dg = 0
    original_fds = set(fd for fd_set in input_fd_sets for fd in fd_set)
    laji_fds = []

    fd_t_sets = {}
    t1= time.time()
    for lhs, rhs in original_fds:
        flag, t_set = build_equivalence_dict_vectorized(df, list(lhs), [rhs])
        if flag==0:
            laji_fds.append((lhs, rhs))
        else:
            fd_t_sets[(lhs, rhs)] = t_set

    for fd_set in input_fd_sets:
        effective_fd_set = [fd for fd in fd_set if fd not in laji_fds]
        if not effective_fd_set:
            continue
        T_prime = get_t_set_for_fd_set(effective_fd_set, fd_t_sets, n)
        dg = computedg(df, effective_fd_set, T_prime)
        if dg > best_dg:
            best_dg = dg
            best_afd = effective_fd_set
    print(f"[Efficiency] Time of selecting core AFD set: {time.time() - t1:.4f} seconds")
    return best_afd, best_dg


