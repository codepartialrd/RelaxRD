from typing import Tuple, Dict
from collections import defaultdict
import pandas as pd
import time
import pickle as pkl


class QuickCore:
    def __init__(self, df: pd.DataFrame, sigma, err: Dict[Tuple[Tuple[str, ...], str], float]):
        self.df = df
        self.sigma = sigma
        self.err = err
        self.m = len(df)
        self._DIST_CACHE = {}
        self._up_memo = {}
        self._low_memo = {}

    def quickcore(self):
        t1 = time.time()
        print(f"[Effectiveness] Number of AFD in Σ: {len(self.sigma)}")
        self.sigma = self._filter_out_non_duplicate_afds(self.sigma)
        tf=time.time()
        print(f"[Effectiveness] Number of AFD after filtering, |Σ|_NaiveCore_1: {len(self.sigma)}")
        self.sigma = self._filter_out_using_rhs_uniqueness(self.sigma)
        print(f"[Effectiveness] Number of AFD after filtering, |Σ|_NaiveCore_2: {len(self.sigma)}")
        t2=time.time()
        S, dg_S_l, dg_S_u = [], 0.0, 0.0
        L1 = [(tuple(afd),) for afd in self.sigma]
        L = [None, L1]
        l = 1
        while L[l]:
            next_level_seeds = set()
            for F in L[l]:
                if not self._is_redundancy(F):
                    next_level_seeds.add(F)
                    up, low = self._compute_up_low(F)
                    if up <= dg_S_l:
                        continue
                    if low > dg_S_u:
                        S, dg_S_l, dg_S_u = [F], low, up
                    else:
                        S = [F2 for F2 in S if self._up_memo[F2] >= low]
                        S.append(F)
                        dg_S_l = max(dg_S_l, low)
                        dg_S_u = max(dg_S_u, up)
            L.append(self._generate_next_level(next_level_seeds))
            l += 1
        print(f"[Effectiveness] Number of candidate AFD subsets after QuickCore, |S_QuickCore|: {len(S)}")
        print('-------------------------------------------------------------------------------------')
        print(f"[Efficiency] Time of filtering using NaiveCore_1, Σ_NaiveCore_1: {tf-t1:.4f} seconds")
        print(f"[Efficiency] Time of filtering using NaiveCore_2, Σ_NaiveCore_2: {t2-tf:.4f} seconds")
        print(f"[Efficiency] Time of pruning: {time.time()-t2:.4f} seconds")
        print('-------------------------------------------------------------------------------------')
        return S

    def compute_closure(self, lhs, fd_list):
        closure = set(lhs)
        changed = True
        while changed:
            changed = False
            for l, r in fd_list:
                if set(l).issubset(closure) and r not in closure:
                    closure.add(r)
                    changed = True
        return closure
    def _deduplicate_equivalent_fd_sets(self, fd_list):
        lhs_to_rhs_set = defaultdict(set)
        for lhs, rhs in fd_list:
            lhs_key = tuple(sorted(lhs))
            lhs_to_rhs_set[lhs_key].add(rhs)

        closure_map = {}
        for lhs in lhs_to_rhs_set:
            closure_map[lhs] = self.compute_closure(lhs, fd_list)

        removed_lhs = set()
        lhs_keys = list(lhs_to_rhs_set.keys())
        for i in range(len(lhs_keys)):
            for j in range(i + 1, len(lhs_keys)):
                lhs1, lhs2 = lhs_keys[i], lhs_keys[j]

                if lhs1 in removed_lhs or lhs2 in removed_lhs:
                    continue
                if closure_map[lhs1] >= set(lhs2) and closure_map[lhs2] >= set(lhs1):
                    if lhs1 < lhs2:
                        removed_lhs.add(lhs2)
                    else:
                        removed_lhs.add(lhs1)

        cleaned_fd = []
        for lhs, rhs in fd_list:
            lhs_key = tuple(sorted(lhs))
            if lhs_key not in removed_lhs:
                cleaned_fd.append((lhs, rhs))

        return cleaned_fd
    def _d(self, X):
        X = tuple(X)
        if X not in self._DIST_CACHE:
            self._DIST_CACHE[X] = self.df[list(X)].drop_duplicates().shape[0]
        return self._DIST_CACHE[X]

    def _compute_up_low(self, F):
        if F in self._up_memo:
            return self._up_memo[F], self._low_memo[F]
        m, up, low = self.m, 0, 0
        lhs2rhs = defaultdict(set)
        for X, A in F:
            lhs2rhs[X].add(A)
        for X, Y in lhs2rhs.items():
            y_cnt = len(Y)
            dX = self._d(X)
            up += m * y_cnt - dX * (y_cnt + len(X))
            err_sum = sum(self.err[(X2, A)] for (X2, A) in F if A in Y)
            low += (1 - err_sum) * m * y_cnt - dX * (y_cnt + len(X))
        self._up_memo[F], self._low_memo[F] = up, low
        return up, low

    def _dg_up_single(self, X, Y):
        m, dX = self.m, self._d(X)
        e_max = max(self.err[(X, A)] for A in Y)
        return (1 - e_max) * m * len(Y) - dX * (len(X) + len(Y))

    def _is_redundancy(self, F):
        rhs_seen = set()
        for _, A in F:
            if A in rhs_seen:
                return True
            rhs_seen.add(A)
        g = defaultdict(set)
        for lhs, A in F:
            for a in lhs:
                g[a].add(A)
        seen = set()
        def dfs(v, stk):
            seen.add(v)
            stk.add(v)
            for nxt in g.get(v, ()):
                if nxt in stk or (nxt not in seen and dfs(nxt, stk)):
                    return True
            stk.remove(v)
            return False
        return any(dfs(n, set()) for n in list(g) if n not in seen)

    def _filter_out_non_duplicate_afds(self, sigma):
        sigma=self._deduplicate_equivalent_fd_sets(sigma)
        lhs2rhs = defaultdict(set)
        for lhs, rhs in sigma:
            lhs2rhs[lhs].add(rhs)
        bad = {X for X, Y in lhs2rhs.items() if self._dg_up_single(X, Y) < 0}
        return [(l, r) for l, r in sigma if l not in bad]

    def _filter_out_using_rhs_uniqueness(self, sigma, max_iter=5):
        fd_set = self._generate_dict(sigma)
        iter_num = 0
        confirmed_attrs = {}
        inverted_index, distinct_cache, lhs_counter = self._init_inverted_index(fd_set, confirmed_attrs)
        while len(fd_set) != 0 and iter_num < max_iter:
            inverted_index, fd_set = self._filter_dominated_lhs(inverted_index, fd_set)
            confirmed_attrs, fd_set = self._confirm_and_update_fd(inverted_index, confirmed_attrs, fd_set)
            inverted_index = self._update_inverted_index(fd_set, confirmed_attrs, distinct_cache, lhs_counter)
            iter_num += 1
        filter_afds = [(lhs, rhs) for rhs, lhs in confirmed_attrs.items()]
        for lhs, rhs_list in fd_set.items():
            for rhs in rhs_list:
                filter_afds.append((lhs, rhs))
        return filter_afds

    def _generate_dict(self, results):
        fd_set = defaultdict(list)
        for lhs, rhs in results:
            fd_set[tuple(lhs)].append(rhs)
        return fd_set

    def _generate_next_level(self, seeds):
        seeds = [tuple(sorted(s)) for s in seeds]
        if not seeds:
            return []
        buckets = defaultdict(list)
        for s in seeds:
            prefix = s[:-1]
            buckets[prefix].append(s)
        next_level = set()
        for bucket in buckets.values():
            bucket.sort()
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    merged = tuple(sorted(set(a) | set(b)))
                    next_level.add(merged)
        return next_level

    def _confirm_and_update_fd(self, updated_inverted_index, original_confirmed_attrs, fd_set):
        for attr, gain_list in updated_inverted_index.items():
            if len(gain_list) == 1:
                X, _ = gain_list[0]
                original_confirmed_attrs[attr] = X
        confirmed_set = set(original_confirmed_attrs)
        new_fd_set = {}
        for lhs, rhs_list in fd_set.items():
            filtered_rhs = [rhs for rhs in rhs_list if rhs not in confirmed_set]
            if filtered_rhs:
                new_fd_set[lhs] = filtered_rhs
        return original_confirmed_attrs, new_fd_set

    def _filter_dominated_lhs(self, inverted_index, fd_set):
        new_inv = {}
        for rhs_attr, pair_list in inverted_index.items():
            ranges = []
            for lhs, gain in pair_list:
                if isinstance(gain, tuple):
                    low, high = gain
                else:
                    low = high = gain
                ranges.append((lhs, low, high))
            kept = []
            for lhs_i, low_i, high_i in ranges:
                other_lows = [low_j for lhs_j, low_j, _ in ranges if lhs_j != lhs_i]
                max_other_low = max(other_lows) if other_lows else float('-inf')
                if high_i >= max_other_low:
                    kept.append((lhs_i, (low_i, high_i) if low_i != high_i else low_i))
                else:
                    if lhs_i in fd_set and rhs_attr in fd_set[lhs_i]:
                        fd_set[lhs_i].remove(rhs_attr)
                        if not fd_set[lhs_i]:
                            del fd_set[lhs_i]
            new_inv[rhs_attr] = kept
        return new_inv, fd_set

    def _compute_gain(self, lhs_t, d_X, count, confirmed_lhs, err_self):
        e_sum = self._sum_max_rhs_errors_from_dict()
        m = self.m
        if lhs_t in confirmed_lhs:
            lower = m * (1 - e_sum) - min(d_X, m * (1 - e_sum))
            upper = m * (1 - err_self) - min(d_X, m * (1 - err_self)) * (1 - err_self)
            return (lower, upper) if upper >= 0 else None
        lower = m * (1 - e_sum) - (1 + len(lhs_t)) * min(d_X, m * (1 - e_sum))
        upper = m * (1 - err_self) - (1 + len(lhs_t)) * min(d_X, m * (1 - err_self)) * (1 - err_self)
        return (lower, upper) if upper >= 0 else None

    def _sum_max_rhs_errors_from_dict(self):
        rhs_max_error = {}
        for lhs, rhs_list in self._generate_dict(self.sigma).items():
            for rhs in rhs_list:
                key = (tuple(lhs), rhs)
                if key not in self.err:
                    continue
                err = self.err[key]
                if rhs not in rhs_max_error or err > rhs_max_error[rhs]:
                    rhs_max_error[rhs] = err
        return sum(rhs_max_error.values())

    def _init_inverted_index(self, fd_set, confirmed_attrs):
        gain_map = defaultdict(list)
        distinct_cache = {}
        lhs_counter = defaultdict(int)
        confirmed_lhs = {tuple(v) for v in confirmed_attrs.values()}
        for lhs, rhs_list in fd_set.items():
            lhs_t = tuple(lhs)
            lhs_counter[lhs_t] += len(rhs_list)
        for lhs, rhs_list in fd_set.items():
            lhs_t = tuple(lhs)
            d_X = self.df[list(lhs_t)].drop_duplicates().shape[0]
            distinct_cache[lhs_t] = d_X
            count = lhs_counter[lhs_t]
            for rhs in rhs_list:
                gain = self._compute_gain(lhs_t, d_X, count, confirmed_lhs, self.err[(lhs_t, rhs)])
                if gain is not None:
                    gain_map[rhs].append((lhs_t, gain))
        return gain_map, distinct_cache, lhs_counter

    def _update_inverted_index(self, fd_set, confirmed_attrs, distinct_cache, lhs_counter):
        gain_map = defaultdict(list)
        confirmed_lhs = {tuple(v) for v in confirmed_attrs.values()}
        lhs_counter.clear()
        for lhs, rhs_list in fd_set.items():
            lhs_t = tuple(lhs)
            lhs_counter[lhs_t] += len(rhs_list)
        for lhs, rhs_list in fd_set.items():
            lhs_t = tuple(lhs)
            d_X = distinct_cache.get(lhs_t)
            if d_X is None:
                d_X = self.df[list(lhs_t)].drop_duplicates().shape[0]
                distinct_cache[lhs_t] = d_X
            count = lhs_counter[lhs_t]
            for rhs in rhs_list:
                gain = self._compute_gain(lhs_t, d_X, count, confirmed_lhs, self.err[(lhs_t, rhs)])
                if gain is not None:
                    gain_map[rhs].append((lhs_t, gain))
        return gain_map
    def save_coreafd_as_pkl(self, coreafd, output_pkl_path):
        lhs_to_rhs = defaultdict(list)

        for lhs, rhs in coreafd:
            lhs_key = tuple(lhs)  # 保证 hashable / 稳定
            lhs_to_rhs[lhs_key].append(rhs)
        afd_list = []
        for lhs, rhs_list in lhs_to_rhs.items():
            afd_list.append({
                "lhs": list(lhs),
                "rhs": sorted(rhs_list)  # 排序，保证可复现
            })

        with open(output_pkl_path, "wb") as f:
            pkl.dump(afd_list, f)

        print(f"[AFD Save] Saved {len(afd_list)} AFD entries to {output_pkl_path}")
