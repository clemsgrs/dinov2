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

You need to wrap up your data in a tarball file:

1. Ensure images are all in one directory
2. Create a single large tarball file that contains all images and name it `pretrain_dataset.tar` :

    ```shell
    tar -chf pretrain_dataset.tar /path/to/image/folder
    ```

### Using whole dataset

  1. Infer the auxiliary files `pretrain_entries.npy` and `pretrain_file_indices.npy` :

      ```shell
      python scripts/infer_entries.py \
          --tarball_path /path/to/pretrain_dataset.tar \
          --output_root /path/to/output/folder \
          --name pretrain
      ```

      The `pretrain_entries.npy` file will record:
      - a dummy class index (we set it to 0 for all images since we’re not using classes)
      - a unique filename index for each image
      - the start and end offsets of each image within the tarball file

      The `pretrain_file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename.

  2. Dump `pretrain_dataset.tar`, `pretrain_entries.npy` and `pretrain_file_indices.npy` in a common folder (e.g. `/root/data`)

### Restricting to a subset

You may not want to use all the patches of a cohort, but only a subset of them (e.g. the cohort comes with a train/tune/test split and you only want to use the patches belonging to slides in the train partition).

Then, follow these simple steps:

  1. Dump the image filenames (e.g. `patch1.jpg`) of the subset of interest in a `.txt` file (e.g. `{subset}.txt`)

  2. Infer the corresponding auxiliary files `pretrain_entries_{subset}.npy`

      ```shell
      python scripts/infer_entries.py \
        --tarball_path /path/to/pretrain_dataset.tar \
        --output_root /path/to/output/folder \
        --keep /path/to/{subset}.txt \
        --name pretrain \
        --suffix {subset}
      ```

      The `pretrain_entries_{subset}.npy` file will record:
      - a dummy class index (we set it to 0 for all images since we’re not using classes)
      - a unique filename index for each image listed in `{subset}.txt`
      - the start and end offsets of each image within the tarball file

      A generic `pretrain_file_indices.npy` file will be saved the first time you run this command.<br>
      It consists in a dictionnary mapping filename index to corresponding filename for the entire tarball file.

  3. Dump `pretrain_dataset.tar`, `pretrain_entries_{subset}.npy` and `pretrain_file_indices.npy` in a common folder (e.g. `/root/data`)

## Training

:warning: To execute the commands provided in this section, make sure the `dinov2` package is included in the Python module search path:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/dinov2"
```

### Training a ViT-L/14

Update `dinov2/configs/train/vitl14.yaml` if you want to change some parameters, then run:

```shell
python -m torch.distributed.run --nproc_per_node=gpu dinov2/train/train.py \
    --config-file dinov2/configs/train/vitl14.yaml \
    train.dataset_path=Pathology:root={path/to/data/root}:subset={subset}
```

Replace `{path/to/data/root}` with the folder you chose for `--output_root` in data preparation (e.g. `Pathology:root=/root/data`).<br>
Leave out `:subset={subset}` if you didn't restrict the dataset to a specific subset when preparing data.<br>
Otherwise, replace `{subset}` with the suffix you chose for `--suffix` in data preparation (e.g. `Pathology:root=/root/data:subset=train`).
