import os
import pandas as pd
import pickle as pkl
import time
import argparse
from CoreAFD.source.quickcore import QuickCore
from CoreAFD.source.naivecore import NaiveCore
from CoreAFD.source.selectingcoreafd_update import cache_and_index

def main_quickcore(CSV_PATH, AFD_PKL, output_dir="output"):
    with open(AFD_PKL, 'rb') as f:
        data = pkl.load(f)

    sigma = data["fds_dict_list"]
    error = data["error_dict"]

    df = pd.read_csv(CSV_PATH, na_values=' ', sep=',', header=0)

    qc = QuickCore(df, sigma, error)
    S_quickcore = qc.quickcore()
    coreafd, dg = cache_and_index(S_quickcore, df, df.columns)

    print(f"Core AFD set: {coreafd}")

    dataset_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
    output_path = os.path.join(output_dir, f"{dataset_name}_coreafd.pkl")

    qc.save_coreafd_as_pkl(coreafd, output_path)

def main_naivecore(CSV_PATH, AFD_PKL, output_dir="output"):
    with open(AFD_PKL, 'rb') as f:
        data = pkl.load(f)

    sigma = data["fds_dict_list"]
    error = data["error_dict"]

    df = pd.read_csv(CSV_PATH, na_values=' ', sep=',', header=0)

    qc = NaiveCore(df, sigma, error)
    S_naivecore = qc.naivecore()
    coreafd, dg=cache_and_index(S_naivecore, df, df.columns)

    print(f"Core AFD set: {coreafd}")

    dataset_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
    output_path = os.path.join(output_dir, f"{dataset_name}_coreafd.pkl")

    qc.save_coreafd_as_pkl(coreafd, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Core AFD Set Selection")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the input CSV dataset"
    )
    parser.add_argument(
        "--Sigma",
        type=str,
        required=True,
        help="Path to the discovered AFD pickle file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="quickcore",
        choices=["quickcore", "naivecore"],
        help="Core AFD selection method (quickcore or naivecore)"
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    t_start = time.time()

    if args.method == "quickcore":
        main_quickcore(args.data_dir, args.Sigma)
    else:
        main_naivecore(args.data_dir, args.Sigma)

    t_end = time.time()
    print(f"[Total Time] Time of total: {t_end - t_start:.4f} seconds")



