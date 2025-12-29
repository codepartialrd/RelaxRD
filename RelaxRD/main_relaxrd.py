import argparse
import os
import time
import pickle as pkl
import pandas as pd
from RelaxRD.source.RelaxRD import load_relation
from RelaxRD.source.schemadesign import SchemaDesign
from RelaxRD.source.RelaxRD import RelaxRD



def main(data_path, coreafd_path, output_dir):
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # -------- . loading the core AFD set --------
    with open(coreafd_path, "rb") as f:
        afd_list = pkl.load(f)

    # -------- 3. loading relation --------
    r_df = load_relation(data_path)

    # -------- 4. Schema design + instantiation --------
    t_start = time.time()
    relation_schemas = SchemaDesign(afd_list, r_df.columns)
    decomposed_relations = RelaxRD(r_df, afd_list, relation_schemas)
    t_end = time.time()

    # -------- 5. printing results --------
    for name, rows in decomposed_relations.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(dataset_output_dir, f"{name}.csv"),index=False)


    print(f"[Output] Decomposed relations stored in: {dataset_output_dir}")
    print(f"[Efficiency] Relation instantiation time: {t_end - t_start:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RelaxRD decomposition")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to input dataset (CSV file)"
    )

    parser.add_argument(
        "--coreafd",
        type=str,
        required=True,
        help="Path to core AFD set (.pkl)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="RelaxRD/output",
        help="Root directory for output relations"
    )

    args = parser.parse_args()

    main(
        data_path=args.data_dir,
        coreafd_path=args.coreafd,
        output_dir=args.output_dir
    )
