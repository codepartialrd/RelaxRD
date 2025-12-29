import os
import time
import argparse
import pickle as pkl
import pandas as pd


from fca.defs.patterns.hypergraphs import TrimmedPartitionPattern
from itertools import combinations

try:
    from functools import reduce
except ImportError:
    pass

def read_db_from_csv(file_path, sep=','):
    df = pd.read_csv(
        file_path,
        sep=sep,
        header=0,
        encoding='utf-8-sig',
        keep_default_na=False
    )

    columns = df.columns.tolist()
    df.columns = list(range(len(columns)))

    hashes = {}

    for t, row in df.iterrows():
        for i, s in row.items():
            s = str(s)
            hashes.setdefault(i, {}).setdefault(s, set()).add(t)

    patterns = [
        PPattern.fix_desc(list(hashes[k].values()))
        for k in sorted(hashes.keys())
    ]

    return df, patterns, len(df), columns

def tostr(atts):
    return ''.join([chr(65 + i) for i in atts])


class PPattern(TrimmedPartitionPattern):
    @classmethod
    def intersection(cls, desc1, desc2):
        new_desc = []
        T = {}
        S = {}
        for i, k in enumerate(desc1):
            for t in k:
                T[t] = i
            S[i] = set([])
        for i, k in enumerate(desc2):
            for t in k:
                if T.get(t, None) is not None:
                    S[T[t]].add(t)
            for t in k:
                if T.get(t, None) is not None:
                    if len(S[T[t]]) > 1:
                        new_desc.append(S[T[t]])
                    S[T[t]] = set([])
        return new_desc


class PartitionsManager(object):
    def __init__(self, T):
        self.T = T
        self.cache = {0: None, 1: {(i,): j for i, j in enumerate(T)}}
        self.current_level = 1

    def new_level(self):
        self.current_level += 1
        self.cache[self.current_level] = {}

    def purge_old_level(self):
        del self.cache[self.current_level - 2]

    def register_partition(self, X, X0, X1):
        self.cache[len(X)][X] = PPattern.intersection(self.cache[len(X0)][X0], self.cache[len(X1)][X1])

    def check_fd(self, X, XY, r, error):
        if not bool(X):
            return False
        return bool(calculate_e(X, XY, range(r), self) <= error)

    def is_superkey(self, X):
        return not bool(self.cache[len(X)][X])


class rdict(dict):
    def __init__(self, *args, **kwargs):
        super(rdict, self).__init__(*args, **kwargs)
        self.itemlist = super(rdict, self).keys()

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.recursive_search(key)
        return super(rdict, self).__getitem__(key)

    def recursive_search(self, key):
        return reduce(set.intersection, [self[tuple(key[:i] + key[i + 1:])] for i in range(len(key))])


def pdep_self(df, y):
    return (df.loc[:, y].value_counts() / df.shape[0]).pow(2).sum()


def pdep(df, lhs, rhs):
    if not isinstance(lhs, list):
        lhs = [lhs]
    xy_counts = df.groupby(lhs + rhs).size().reset_index(name="xy_count")
    x_counts = df.groupby(lhs).size().reset_index(name="x_count")
    counts = xy_counts.merge(x_counts, on=lhs)
    return 1 - (1 / df.shape[0]) * (counts["xy_count"].pow(2) / counts["x_count"]).sum()


def calculate_e(X, XA, R, checker):
    if not bool(X):
        return 1
    A = set(XA) - set(X)
    error = pdep(data_pd, list(X), list(A))
    return error


def prefix_blocks(L):
    blocks = {}
    for atts in L:
        blocks.setdefault(atts[:-1], []).append(atts)
    return blocks.values()


class TANE(object):
    def __init__(self, T):
        self.T = T
        self.rules = []
        self.error_dict = {}
        self.pmgr = PartitionsManager(T)
        self.R = range(len(T))
        self.Cplus = rdict()
        self.Cplus[tuple([])] = set(self.R)
        self.n=len(data_pd)

    def compute_dependencies(self, L, r, error):
        for X in L:
            for y in self.Cplus[X].intersection(X):
                a = X.index(y)
                LHS = X[:a] + X[a + 1:]
                ee=calculate_e(LHS, X, range(r), self)
                if ee <= error:
                    self.rules.append((LHS, y))
                    print(LHS, y, ee)
                    self.error_dict[(LHS, y)] = ee
                    self.Cplus[X].remove(y)
                    if calculate_e(LHS, X, range(r), self.pmgr) == 0:
                        map(self.Cplus[X].remove, filter(lambda i: i not in X, self.Cplus[X]))

    def prune(self, L):
        clean_idx = set([])
        for X in L:
            if not bool(self.Cplus[X]):
                clean_idx.add(X)
            dx=len(data_pd[list(X)].drop_duplicates())
            if dx==self.n:
                for y in filter(lambda x: x not in X, self.Cplus[X]):
                    self.rules.append((X, y))
                    self.error_dict[(X, y)] = 0
                clean_idx.add(X)
        for X in clean_idx:
            L.remove(X)

    def prefix_blocks(self, L):
        blocks = {}
        for atts in L:
            blocks.setdefault(atts[:-1], []).append(atts)
        return blocks.values()

    def generate_next_level(self, L):
        self.pmgr.new_level()
        next_L = set([])
        for k in prefix_blocks(L):
            for i, j in combinations(k, 2):
                if i[-1] < j[-1]:
                    X = i + (j[-1],)
                else:
                    X = j + (i[-1],)
                if all(X[:a] + X[a + 1:] in L for a, x in enumerate(X)):
                    next_L.add(X)
        return next_L
    def memory_wipe(self):
        self.pmgr.purge_old_level()
    def run(self, r, error):
        L1 = set([tuple([i]) for i in self.R])
        L = [None, L1]
        l = 1
        while bool(L[l]):
            self.compute_dependencies(L[l], r, error)
            self.prune(L[l])
            L.append(self.generate_next_level(L[l]))
            l = l + 1
            L[l - 1] = None
            self.memory_wipe()
def main(args):
    t_start = time.time()

    file_path = args.data_dir
    output_dir = args.output_dir
    error = args.error

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_sigma.pkl")

    # Load data
    data_pd, T, r, columns = read_db_from_csv(file_path, sep=',')

    tane = TANE(T)
    t0 = time.time()
    tane.run(r, error)

    print(f"[AFD Discovery] Execution Time: {time.time() - t0:.4f} seconds")
    print(f"[AFD Discovery] {len(tane.rules)} rules found")

    # Convert attribute indices to names
    afd_list = []
    error_dict = {}

    for lhs, rhs in tane.rules:
        lhs_names = tuple(columns[i] for i in lhs)
        rhs_name = columns[rhs]
        afd_list.append((lhs_names, rhs_name))

        if (lhs, rhs) in tane.error_dict:
            error_dict[(lhs_names, rhs_name)] = tane.error_dict[(lhs, rhs)]

    # Save result
    with open(output_path, "wb") as f:
        pkl.dump({
            "fds_dict_list": afd_list,
            "error_dict": error_dict
        }, f)

    print(f"[Output] AFDs saved to {output_path}")
    print(f"[Total Time] {time.time() - t_start:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFD Discovery via TANE")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to input CSV dataset")
    parser.add_argument("--output_dir", type=str, default="AFD_discovery/Sigma",
                        help="Directory to store discovered AFDs")
    parser.add_argument("--error", type=float, default=0.05,
                        help="Error threshold ε for approximate FDs")

    args = parser.parse_args()
    main(args)

