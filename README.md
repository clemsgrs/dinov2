# DINOv2 for Computational Pathology

## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov2` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate dinov2
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```

For dense tasks (depth estimation and semantic segmentation), there are additional dependencies (specific versions of `mmcv` and `mmsegmentation`) which are captured in the `extras` dependency specifications:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)**:

```shell
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)*:

```shell
pip install -r requirements.txt -r requirements-extras.txt
```

## Data preparation

### Pathology Dataset

You need to organize data in a tarball file:

1. ensure images are all in one directory
2. create a single large tarball file that contains all images

    ```shell
    tar -cvf dataset.tar /path/to/image/folder
    ```

3. infer the auxiliary file `entries.npy`

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/dataset.tar \
      --output_root /path/to/output/folder
    ```

    This file will record:
    - a dummy class index (we set it to 0 for all images since we’re not using classes)
    - the start and end offsets of each image within the tarball file
    - the filename of each image

4. (optional) doublecheck the `entries.npy` file matches the tarball file

    ```shell
    python3 scripts/test_entries.py \
      --image_root /path/to/image/folder \
      --tarball_path /path/to/dataset.tar \
      --entries_path /path/to/entries.npy
    ```


<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.

## Training

### Fast setup: training DINOv2 ViT-L/16 on ImageNet-1k

Run DINOv2 training on 4 A100-80GB nodes (32 GPUs) in a SLURM cluster environment with submitit:

```shell
python dinov2/run/train/train.py \
    --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

Training time is approximately 1 day and the resulting checkpoint should reach 81.6% on k-NN eval and 82.9% on linear eval.

The training code saves the weights of the teacher in the `eval` folder every 12500 iterations for evaluation.

### Long setup: training DINOv2 ViT-L/14 on ImageNet-22k

Run DINOv2 training on 12 A100-80GB nodes (96 GPUs) in a SLURM cluster environment with submitit:

```shell
python dinov2/run/train/train.py \
    --nodes 12 \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

Training time is approximately 3.3 days and the resulting checkpoint should reach 82.0% on k-NN eval and 84.5% on linear eval.

The training code saves the weights of the teacher in the `eval` folder every 12500 iterations for evaluation.


## Evaluation

The training code regularly saves the teacher weights. In order to evaluate the model, run the following evaluation on a single node:

### k-NN classification on ImageNet-1k

```shell
python dinov2/run/eval/knn.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/knn \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Logistic regression classification on ImageNet-1k

```shell
python dinov2/run/eval/log_regression.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/logreg \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Linear classification with data augmentation on ImageNet-1k

```shell
python dinov2/run/eval/linear.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```