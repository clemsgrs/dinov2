import os
import tarfile
import argparse
import numpy as np

from pathlib import Path


def create_tarball_and_entries(image_root, output_root):
    entries = []
    current_offset = 0  # initialize offset

    with tarfile.open(Path(output_root, "dataset.tar"), "w") as tar:
        for img_path in sorted(Path(image_root).iterdir()):
            img_name = img_path.stem

            # tarfile headers are typically 512 bytes, and files are padded to 512 bytes
            header_size = 512
            file_size = os.path.getsize(img_path)
            padded_file_size = (file_size + 511) // 512 * 512  # round up to nearest 512

            start_offset = current_offset + header_size
            end_offset = start_offset + file_size

            tar.add(img_path, arcname=img_name)
            entries.append([0, img_name, start_offset, end_offset])  # dummy class index 0

            current_offset += header_size + padded_file_size

    # save entries
    np.save(Path(output_root, "entries.npy"), np.array(entries, dtype=object))


def main():
    parser = argparse.ArgumentParser(description="Generate tarball and entries file for pretraining dataset.")
    parser.add_argument("-i", "image_root", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("-o", "output_root", type=str, required=True, help="Path to the output directory where dataset.tar and entries.npy will be saved.")

    args = parser.parse_args()

    create_tarball_and_entries(args.image_root, args.output_root)


if __name__ == "__main__":

    main()
