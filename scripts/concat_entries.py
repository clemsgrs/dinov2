import tqdm
import argparse
import numpy as np

from pathlib import Path
from typing import Optional


def concat_entries(
    entries_paths,
    output_root,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    try:
        concat_entries = np.load(entries_paths[0])
        # load and concatenate the rest of the files
        with tqdm.tqdm(
            entries_paths[1:],
            desc="Concatenating entries",
            unit=" file",
            initial=1,
            total=len(entries_paths),
            leave=True,
        ) as t:
            for e in t:
                data = np.load(e)
                concat_entries = np.concatenate((concat_entries, data))

        # save entries
        if prefix:
            if suffix:
                entries_filepath = Path(output_root, f"{prefix}_entries_{suffix}.npy")
            else:
                entries_filepath = Path(output_root, f"{prefix}_entries.npy")
        elif suffix:
            entries_filepath = Path(output_root, f"entries_{suffix}.npy")
        else:
            entries_filepath = Path(output_root, "entries.npy")
        np.save(entries_filepath, np.array(concat_entries, dtype=np.uint64))
        print(f"Concatenated entries saved to: {entries_filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Concatenate entriy files from multiple datasets.")
    parser.add_argument(
        "-r", "--root", type=str, required=True, help="Folder where entry files to concatenate are located."
    )
    parser.add_argument(
        "-o",
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory where the concatenated entry will be saved.",
    )
    parser.add_argument("-p", "--prefix", type=str, help="Prefix to append to the concatenated file name.")
    parser.add_argument("-s", "--suffix", type=str, help="Suffix to append to the concatenated file name.")

    args = parser.parse_args()

    prefix = f"{args.prefix}" if args.prefix else None
    suffix = f"{args.suffix}" if args.restrict and args.suffix else None

    # grab all entries
    entries_paths = [fp for fp in Path(args.root).glob("*.npy") if "entries" in str(fp)]
    assert len(entries_paths) > 0, f"0 entry file found under {args.root}"
    print(f"{len(entries_paths)} entry files found!")

    concat_entries(entries_paths, args.output_root, prefix, suffix)


if __name__ == "__main__":
    main()
