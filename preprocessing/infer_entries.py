import tqdm
import tarfile
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, List


def infer_entries_from_tarball(
    tarball_path,
    output_root,
    keep_filenames: Optional[List[str]] = None,
    remove_filenames: Optional[List[str]] = None,
    name: Optional[str] = None,
    suffix: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    file_name: str = "filename",
    label_name: str = "label",
    class_name: str = "class",
):
    entries = []
    file_index = 0
    file_indices = {}  # store filenames with an index

    if df is not None:
        if class_name not in df.columns:
            df[class_name] = df[label_name].apply(lambda x: f"class_{x}")

    # convert keep filenames to a set for efficient lookup, if provided
    keep_filenames = set(keep_filenames) if keep_filenames else None

    # convert remove filenames to a set for efficient lookup, if provided
    remove_filenames = set(remove_filenames) if remove_filenames else None

    with tarfile.open(tarball_path, "r:*") as tar:
        with tqdm.tqdm(
            tar.getmembers(),
            desc=f"Parsing {Path(tarball_path).name}",
            leave=True,
        ) as t:
            for member in t:
                if member.isfile():  # Process only files (not directories, symlinks, etc.)
                    filename = Path(member.name).name
                    start_offset = member.offset_data
                    end_offset = member.offset_data + member.size

                    # add file name to the dictionary and use the index in entries
                    file_indices[file_index] = filename

                    class_index = 0
                    keep_file = keep_filenames is None or len(keep_filenames.intersection(set([filename]))) > 0
                    remove_file = (
                        remove_filenames is not None and len(remove_filenames.intersection(set([filename]))) > 0
                    )
                    if keep_file and not remove_file:
                        if df is not None:
                            class_index = df[df[file_name] == filename][label_name].values[0]

                        entries.append([class_index, file_index, start_offset, end_offset])

                    file_index += 1

    # save entries
    if name:
        if suffix:
            entries_filepath = Path(output_root, f"{name}_entries_{suffix}.npy")
        else:
            entries_filepath = Path(output_root, f"{name}_entries.npy")
    elif suffix:
        entries_filepath = Path(output_root, f"entries_{suffix}.npy")
    else:
        entries_filepath = Path(output_root, "entries.npy")
    np.save(entries_filepath, np.array(entries, dtype=np.uint64))

    # optionally save file indices
    file_indices_filepath = (
        Path(output_root, f"{name}_file_indices.npy") if name else Path(output_root, "file_indices.npy")
    )
    if not file_indices_filepath.exists():
        np.save(file_indices_filepath, file_indices)

    if df is not None:
        # optionally save class index mapping
        df = df.drop_duplicates(subset=[label_name, class_name])
        class_ids = df[[label_name, class_name]].values
        class_ids_filepath = Path(output_root, "class-ids.npy")
        if not class_ids_filepath.exists():
            np.save(class_ids_filepath, class_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate entries file for pretraining dataset.")
    parser.add_argument(
        "--tarball_path",
        type=str,
        required=True,
        help="Path to the tarball file.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory where dataset.tar and entries.npy will be saved.",
    )
    parser.add_argument("--name", type=str, help="Name to put in front of the *.npy file names.")
    parser.add_argument(
        "--keep",
        type=str,
        help="Path to a .csv file with the filenames of the patches you want to keep.",
    )
    parser.add_argument(
        "--remove",
        type=str,
        help="Path to a .csv file with the filenames of the patches you want to remove.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix to append to the entries.npy file name.",
    )
    parser.add_argument("--csv", type=str, help="Path to the csv file with samples labels.")
    parser.add_argument(
        "--file_name",
        type=str,
        default="filename",
        help="Name of the column holding the file names.",
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default="label",
        help="Name of the column holding the labels.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="class",
        help="Name of the column holding the class names.",
    )

    args = parser.parse_args()
    assert not (
        isinstance(args.keep, str) and isinstance(args.remove, str)
    ), "Both 'keep' and 'remove' flags were specified, but they are mutually exclusive.\nPlease provide one or the other."

    keep_filenames = None
    if args.keep:
        print(f"Parsing {args.keep}...")
        with open(args.keep, "r") as f:
            keep_filenames = [line.strip() for line in f]
        print(f"Done: {len(set(keep_filenames))} unique files found")

    remove_filenames = None
    if args.remove:
        print(f"Parsing {args.remove}...")
        with open(args.remove, "r") as f:
            remove_filenames = [line.strip() for line in f]
        print(f"Done: {len(set(remove_filenames))} unique files found")

    name = f"{args.name}" if args.name else None
    suffix = f"{args.suffix}" if args.suffix else None

    df = pd.read_csv(args.csv) if args.csv else None
    infer_entries_from_tarball(
        args.tarball_path,
        args.output_root,
        keep_filenames,
        remove_filenames,
        name,
        suffix,
        df,
        args.file_name,
        args.label_name,
        args.class_name,
    )


if __name__ == "__main__":
    main()
