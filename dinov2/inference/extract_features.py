import os
import tqdm
import torch
import wandb
import argparse
import datetime
import pandas as pd
import multiprocessing as mp

from pathlib import Path

import dinov2.distributed as distributed

from dinov2.models import PatchEmbedder
from dinov2.utils.config import setup, write_config
from dinov2.utils.utils import initialize_wandb
from dinov2.data import SamplerType, make_data_loader
from dinov2.data.datasets import ImageFolderWithNameDataset
from dinov2.data.transforms import make_classification_eval_transform


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def main(args):
    cfg = setup(args)

    run_distributed = torch.cuda.device_count() > 1
    if run_distributed:
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = -1

    if distributed.is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            run_id = wandb_run.id
    else:
        run_id = ""

    if run_distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(obj, 0, device=torch.device(f"cuda:{gpu_id}"))
        run_id = obj[0]

    output_dir = Path(cfg.train.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    cfg.train.output_dir = str(output_dir)
    features_dir = Path(output_dir, "features")
    features_dir.mkdir(exist_ok=True)

    if distributed.is_main_process():
        write_config(cfg, cfg.train.output_dir)

    model = PatchEmbedder(
        cfg,
        verbose=distributed.is_main_process(),
    )

    transform = make_classification_eval_transform()
    dataset = ImageFolderWithNameDataset(cfg.inference.data_dir, transform)

    if run_distributed:
        sampler_type = SamplerType.DISTRIBUTED
    else:
        sampler_type = SamplerType.RANDOM

    num_workers = min(mp.cpu_count(), cfg.inference.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.inference.batch_size,
        num_workers=num_workers,
        sampler_type=sampler_type,
        drop_last=False,
        shuffle=False,
    )

    if gpu_id == -1:
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")

    model = model.to(device, non_blocking=True)
    model.eval()

    if distributed.is_main_process():
        print()

    filenames, feature_paths = [], []

    with tqdm.tqdm(
        data_loader,
        desc="Feature Extraction",
        unit=" img",
        unit_scale=cfg.inference.batch_size,
        ncols=80,
        position=0,
        leave=True,
        disable=not (gpu_id in [-1, 0]),
    ) as t1:
        with torch.no_grad():
            for i, batch in enumerate(t1):
                imgs, fnames = batch
                imgs = imgs.to(device, non_blocking=True)
                features = model(imgs)
                for k, f in enumerate(features):
                    fname = fnames[k]
                    feature_path = Path(features_dir, f"{fname}.pt")
                    torch.save(f, feature_path)
                    filenames.append(fname)
                    feature_paths.append(feature_path)
                if cfg.wandb.enable and not run_distributed:
                    wandb.log({"processed": i + 1})

    features_df = pd.DataFrame.from_dict(
        {
            "filename": filenames,
            "feature_path": feature_paths,
        }
    )

    if run_distributed:
        features_csv_path = Path(output_dir, f"features_{gpu_id}.csv")
    else:
        features_csv_path = Path(output_dir, "features.csv")
    features_df.to_csv(features_csv_path, index=False)

    if run_distributed:
        torch.distributed.barrier()
        if distributed.is_main_process():
            dfs = []
            for gpu_id in range(torch.cuda.device_count()):
                fp = Path(output_dir, f"features_{gpu_id}.csv")
                df = pd.read_csv(fp)
                dfs.append(df)
                os.remove(fp)
            features_df = pd.concat(dfs, ignore_index=True)
            features_df = features_df.drop_duplicates()
            features_df.to_csv(Path(output_dir, "features.csv"), index=False)

    if cfg.wandb.enable and distributed.is_main_process() and run_distributed:
        wandb.log({"processed": len(features_df)})


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    args = get_args_parser(add_help=True).parse_args()
    main(args)
