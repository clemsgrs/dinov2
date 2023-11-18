import os
import argparse
import numpy as np

from pathlib import Path
from typing import Optional, List


def parse_tar_header(header_bytes):
    # Parse the header of a tar file
    name = header_bytes[0:100].decode("utf-8").rstrip("\0")
    size = int(header_bytes[124:136].decode("utf-8").strip("\0"), 8)
    return name, size


def infer_entries_from_tarball(
    tarball_path,
    output_root,
    restrict_filepaths: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    entries = []
    file_index = 0
    file_indices = {}  # store filenames with an index

    # convert restrict filepaths to a set for efficient lookup, if provided
    restrict_filepaths = set(restrict_filepaths) if restrict_filepaths else None

    with open(tarball_path, "rb") as f:
        while True:
            header_bytes = f.read(512)  # read header
            if not header_bytes.strip(b"\0"):
                break  # end of archive

            name, size = parse_tar_header(header_bytes)
            if size == 0 and file_index == 0:
                # skip first entry if empty
                file_index += 1
                continue

            # add file name to the dictionary and use the index in entries
            file_indices[file_index] = name

            if restrict_filepaths is None or name in restrict_filepaths:
                start_offset = f.tell()
                end_offset = start_offset + size
                entries.append([0, file_index, start_offset, end_offset])  # dummy class index 0

            file_index += 1

            f.seek(size, os.SEEK_CUR)  # skip to the next header
            if size % 512 != 0:
                f.seek(512 - (size % 512), os.SEEK_CUR)  # adjust for padding

    # save entries
    entries_filepath = (
        Path(output_root, f"{prefix}_entries_{suffix}.npy") if suffix else Path(output_root, "entries.npy")
    )
    np.save(entries_filepath, np.array(entries, dtype=np.uint64))

    # optionally save file indices
    file_indices_filepath = Path(output_root, f"{prefix}_file_indices.npy")
    if not file_indices_filepath.exists():
        np.save(file_indices_filepath, file_indices)


def main():
    parser = argparse.ArgumentParser(description="Generate tarball and entries file for pretraining dataset.")
    parser.add_argument("-t", "--tarball_path", type=str, required=True, help="Path to the tarball file.")
    parser.add_argument(
        "-o",
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory where dataset.tar and entries.npy will be saved.",
    )
    parser.add_argument("-p", "--prefix", type=str, help="Prefix to append to the *.npy file names.")
    parser.add_argument(
        "-r", "--restrict", type=str, help="Path to a .txt file with the filenames for a specific fold."
    )
    parser.add_argument("-s", "--suffix", type=str, help="Suffix to append to the entries.npy file name.")

    args = parser.parse_args()

    restrict_filepaths = None
    if args.restrict:
        with open(args.restrict, "r") as f:
            restrict_filepaths = [line.strip() for line in f]

    prefix = f"{args.prefix}" if args.suffix else None
    suffix = f"{args.suffix}" if args.restrict and args.suffix else None

    infer_entries_from_tarball(args.tarball_path, args.output_root, restrict_filepaths, prefix, suffix)


if __name__ == "__main__":
    main()
