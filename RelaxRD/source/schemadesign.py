from typing import List, Dict, Set

def compute_closure(X: Set[str], afd_list: List[Dict[str, List[str]]]) -> Set[str]:
    """
    Compute the closure of attribute set X based on given AFDs.
    """
    closure = set(X)
    changed = True
    # Convert AFDs to tuple format to avoid recreating sets
    fd_pairs = [(set(fd['lhs']), set(fd['rhs'])) for fd in afd_list]

    while changed:
        changed = False
        for lhs, rhs in fd_pairs:
            if lhs.issubset(closure) and not rhs.issubset(closure):
                closure.update(rhs)
                changed = True
    return closure

def print_schemas(relation_schemas, title="Generated Schemas"):
    print(f"\n=== {title} ===")
    for i, schema in enumerate(relation_schemas):
        schema_str = ", ".join(schema)
        print(f"R{i}: ({schema_str})")
    print("=" * 100)

def SchemaDesign(afd_list: List[Dict[str, List[str]]], all_attributes: List[str]) -> List[List[str]]:
    """
    Generate relation schemas based on AFDs and full attribute list.
    Includes individual AFD-based schemas and a pruned fact schema.
    """
    relation_schemas = []
    all_lhs, all_rhs = set(), set()

    # Construct schema for each AFD and collect LHS and RHS attributes
    for fd in afd_list:
        lhs = fd['lhs']
        rhs = fd['rhs']
        all_lhs.update(lhs)
        all_rhs.update(rhs)
        relation_schemas.append(sorted(set(lhs + rhs)))

    # Initial candidate fact schema: all LHS + attributes not in RHS
    candidate_fact_schema = all_lhs.union(set(all_attributes) - all_rhs)

    # Prune fact schema by removing attributes that can be inferred from others
    final_fact_schema = set()
    for attr in candidate_fact_schema:
        others = candidate_fact_schema - {attr}
        closure = compute_closure(others, afd_list)
        if attr not in closure:
            final_fact_schema.add(attr)

    relation_schemas.append(sorted(final_fact_schema))
    print_schemas(relation_schemas)
    return relation_schemas


