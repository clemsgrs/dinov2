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

### Pathology Dataset

You need to organize data in a tarball file:

1. ensure images are all in one directory
2. create a single large tarball file that contains all images

    ```shell
    tar -cvf dataset.tar /path/to/image/folder
    ```

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

    The `file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename (as a string).

4. (optional) doublecheck the `entries.npy` file matches the tarball file

    ```shell
    python3 scripts/test_entries.py \
      --image_root /path/to/image/folder \
      --tarball_path /path/to/dataset.tar \
      --entries_path /path/to/entries.npy \
      --file_indices_path /path/to/file_indices.npy
    ```


<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/dinov2"
```

## Training

### Training a ViT-L/14

Update `dinov2/configs/train/vitl14.yaml` and run:

```shell
python dinov2/train/train.py \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR>
```