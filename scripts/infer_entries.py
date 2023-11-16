import os
import argparse
import numpy as np

from pathlib import Path


def parse_tar_header(header_bytes):
    # Parse the header of a tar file
    name = header_bytes[0:100].decode('utf-8').rstrip('\0')
    size = int(header_bytes[124:136].decode('utf-8').strip('\0'), 8)
    return name, size


def infer_entries_from_tarball(tarball_path, output_root):
    entries = []
    entry_count = 0
    with open(tarball_path, 'rb') as f:
        while True:
            header_bytes = f.read(512)  # read header
            if not header_bytes.strip(b'\0'):
                break  # end of archive

            name, size = parse_tar_header(header_bytes)
            if size == 0 and entry_count == 0:
                # skip first entry if empty
                entry_count += 1
                continue

            start_offset = f.tell()
            end_offset = start_offset + size

            entries.append([0, name, start_offset, end_offset])  # dummy class index 0
            entry_count += 1

            f.seek(size, os.SEEK_CUR)  # Skip to the next header
            if size % 512 != 0:
                f.seek(512 - (size % 512), os.SEEK_CUR)  # Adjust for padding

    # save entries
    np.save(Path(output_root, "entries.npy"), np.array(entries, dtype=object))


def main():
    parser = argparse.ArgumentParser(description="Generate tarball and entries file for pretraining dataset.")
    parser.add_argument("-t", "--tarball_path", type=str, required=True, help="Path to the tarball file.")
    parser.add_argument("-o", "--output_root", type=str, required=True, help="Path to the output directory where dataset.tar and entries.npy will be saved.")

    args = parser.parse_args()

    infer_entries_from_tarball(args.tarball_path, args.output_root)


if __name__ == "__main__":

    main()
