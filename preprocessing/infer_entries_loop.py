import argparse
import subprocess
from typing import Optional
from pathlib import Path
from natsort import natsorted


def infer_entries(
    source_dir,
    output_dir,
    keep: Optional[str] = None,
    remove: Optional[str] = None,
    suffix: Optional[str] = None,
    csv: Optional[str] = None,
    file_name: Optional[str] = None,
    label_name: Optional[str] = None,
    class_name: Optional[str] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    tarball_files = natsorted(list(Path(source_dir).glob("*.tar")))
    for fp in tarball_files:
        cmd = [
            "/usr/local/bin/python3",
            "preprocessing/infer_entries.py",
            "--tarball_path",
            f"{fp}",
            "--output_root",
            f"{output_dir}",
            "--name",
            f"{fp.stem}",
        ]
        if keep is not None:
            cmd.append("--keep")
            cmd.append(f"{keep}")
        if remove is not None:
            cmd.append("--remove")
            cmd.append(f"{remove}")
        if suffix is not None:
            cmd.append("--suffix")
            cmd.append(f"{suffix}")
        if csv is not None:
            cmd.append("--csv")
            cmd.append(f"{csv}")
        if file_name is not None:
            cmd.append("--file_name")
            cmd.append(f"{file_name}")
        if label_name is not None:
            cmd.append("--label_name")
            cmd.append(f"{label_name}")
        if class_name is not None:
            cmd.append("--class_name")
            cmd.append(f"{class_name}")
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entries file for multiple tarball files at once.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source tarball file directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory where the npy files will be saved.",
    )
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

    infer_entries(
        args.source,
        args.output,
        args.keep,
        args.remove,
        args.suffix,
        args.csv,
        args.file_name,
        args.label_name,
        args.class_name,
    )
