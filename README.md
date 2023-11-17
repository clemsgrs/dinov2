# DINOv2 for Computational Pathology

## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

Clone the repository and then create and activate a `dinov2` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate dinov2
```

For dense tasks (depth estimation and semantic segmentation), there are additional dependencies (specific versions of `mmcv` and `mmsegmentation`) which are captured in the `extras` dependency specifications:

```shell
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

## Data preparation

You need to organize data in a tarball file:

1. ensure images are all in one directory
2. create a single large tarball file that contains all images

    ```shell
    tar -cvf dataset.tar /path/to/image/folder
    ```

### Single Fold

3. infer the auxiliary files `entries.npy` and `file_indices.npy`

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/dataset.tar \
      --output_root /path/to/output/folder
    ```

    The `entries.npy` file will record:
    - a dummy class index (we set it to 0 for all images since we’re not using classes)
    - a unique filename index for each image
    - the start and end offsets of each image within the tarball file

    The `file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename.

4. (optional) doublecheck the `entries.npy` file matches the tarball file

    ```shell
    python3 scripts/test_entries.py \
      --image_root /path/to/image/folder \
      --tarball_path /path/to/dataset.tar \
      --entries_path /path/to/entries.npy \
      --file_indices_path /path/to/file_indices.npy
    ```

### Multiple Folds

For each individual fold `i`, repeat the following steps:

  3. dump the image filepaths of the fold in a `.txt` file (e.g. `fold_i.txt`)

  4. infer the corresponding auxiliary files `entries_i.npy`

      ```shell
      python3 scripts/infer_entries.py \
        --tarball_path /path/to/dataset.tar \
        --output_root /path/to/output/folder \
        --restrict /path/to/fold_0.txt \
        --suffix 0
      ```

      The `entries_i.npy` file will record:
      - a dummy class index (we set it to 0 for all images since we’re not using classes)
      - a unique filename index for each image listed in `fold_i.txt`
      - the start and end offsets of each image within the tarball file

      A generic `file_indices.npy` file will be saved the first time you run this command.<br>
      It consists in a dictionnary mapping filename index to corresponding filename for the entire tarball file.

  4. (optional) doublecheck the `entries_i.npy` file matches the tarball file

      ```shell
      python3 scripts/test_entries.py \
        --image_root /path/to/image/folder \
        --tarball_path /path/to/dataset.tar \
        --entries_path /path/to/entries_i.npy \
        --file_indices_path /path/to/file_indices.npy
      ```

In the end, this will have created one entries file per fold, which will get dynamically loaded during pretraining.

<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/dinov2"
```

## Training

### Training a ViT-S/16

Update `dinov2/configs/train/vits16.yaml` and run:

```shell
python dinov2/train/train.py \
    --config-file dinov2/configs/train/vits16.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR>
```