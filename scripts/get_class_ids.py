import argparse
import numpy as np
import pandas as pd

from pathlib import Path


def get_class_ids(df, output_root, label_name, class_name):
    df = df.drop_duplicates(subset=[label_name, class_name])
    class_ids = df[[label_name, class_name]].values
    # save class_ids
    class_ids_filepath = Path(output_root, f"class-ids.npy")
    np.save(class_ids_filepath, class_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate tarball and entries file for pretraining dataset.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the csv file with samples labels.")
    parser.add_argument("-o", "--output_root", type=str, required=True, help="Path to the output directory where dataset.tar and entries.npy will be saved.")
    parser.add_argument("-l", "--label_name", type=str, default="label", help="Name of the column holding the labels.")
    parser.add_argument("-c", "--class_name", type=str, default="class", help="Name of the column holding the class names.")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    get_class_ids(df, args.output_root, args.label_name, args.class_name)


if __name__ == "__main__":

    main()
