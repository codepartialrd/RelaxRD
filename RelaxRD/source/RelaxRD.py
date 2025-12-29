import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd


def load_relation(file_path):
    r_df = pd.read_csv(file_path)
    r_df = r_df.drop_duplicates()
    return r_df
def build_equivalence_dict_vectorized(df, lhs, rhs, t_set):
    grouped = df.groupby(lhs + [rhs]).size().reset_index(name='count')
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
    merge_cols = lhs + [rhs]
    conflict_matches = df.merge(conflict_rhs_rows[merge_cols], on=merge_cols, how='inner')
    t_set.update(conflict_matches['orig_index'])


def maximal_afd_tuples(df, afd_list):
    conflict_tuples=set()
    for afd in afd_list:
        lhs, rhss = afd["lhs"], afd["rhs"]
        for rhs in rhss:
            build_equivalence_dict_vectorized(df, lhs, rhs, conflict_tuples)
    return conflict_tuples

def RelaxRD(r_all, afd_list, relation_schemas):
    tuple_conflict = maximal_afd_tuples(r_all, afd_list)
    result = {}
    r0_df = (
        r_all.loc[r_all.index.isin(tuple_conflict)]
        .drop(columns=["orig_index"], errors="ignore")
    )
    r_df = r_all[~r_all.index.isin(tuple_conflict)]
    for idx, schema in enumerate(relation_schemas):
        sub_df = r_df[schema].drop_duplicates()
        result[f"r{idx + 1}"] = sub_df.to_dict(orient='records')
    result["r0"] = r0_df.to_dict(orient="records")
    return result