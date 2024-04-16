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
from dinov2.data import SamplerType
from dinov2.data.loaders import _make_sampler
from dinov2.data.datasets import SlideIDsDataset, SlideRegionDataset

# from dinov2.data.transforms import make_classification_eval_transform
from dinov2.data.transforms import make_feature_extraction_transform


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

    output_dir = Path(cfg.train.output_dir, cfg.train.experiment_name, run_id)
    slide_features_dir = Path(output_dir, "slide_features")
    region_features_dir = Path(output_dir, "region_features")
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
        slide_features_dir.mkdir(exist_ok=True)
        if cfg.inference.save_region_features:
            region_features_dir.mkdir(exist_ok=True)
    cfg.train.output_dir = str(output_dir)

    if distributed.is_main_process():
        write_config(cfg, cfg.train.output_dir)

    model = PatchEmbedder(
        cfg,
        verbose=distributed.is_main_process(),
    )

    # transform = make_classification_eval_transform()
    transform = make_feature_extraction_transform(cfg.inference.patch_size)

    root_dir = Path(cfg.inference.root_dir)
    slide_ids = sorted([s.name for s in root_dir.iterdir()])
    if distributed.is_main_process():
        print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.inference.slide_list:
        with open(Path(cfg.inference.slide_list), "r") as f:
            slide_ids = sorted([Path(x.strip()).stem for x in f.readlines()])
        if distributed.is_main_process():
            print(f"restricting to {len(slide_ids)} slides from slide list .txt file")

    dataset = SlideIDsDataset(slide_ids=slide_ids)

    if run_distributed:
        sampler_type = SamplerType.DISTRIBUTED
    else:
        sampler_type = SamplerType.RANDOM

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=False,
        verbose=False,
    )

    num_workers = min(mp.cpu_count(), cfg.inference.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    slide_id_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    if gpu_id == -1:
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")

    model = model.to(device, non_blocking=True)

    if distributed.is_main_process():
        print()

    processed_slide_ids, region_slide_ids = [], []
    slide_feature_paths, region_feature_paths = [], []
    x_coords, y_coords = [], []

    with tqdm.tqdm(
        slide_id_loader,
        desc="Slide Encoding",
        unit=" slide",
        ncols=80,
        position=0,
        leave=True,
        disable=not (gpu_id in [-1, 0]),
    ) as t1:
        with torch.no_grad():
            for i, slide_id in enumerate(t1):
                slide_id = slide_id[0]
                processed_slide_ids.append(slide_id)
                region_dataset = SlideRegionDataset(
                    root_dir,
                    slide_id,
                    image_size=cfg.inference.patch_size,
                    transform=transform,
                )
                region_loader = torch.utils.data.DataLoader(
                    region_dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                )

                features = []
                with tqdm.tqdm(
                    region_loader,
                    desc=(f"{slide_id}"),
                    unit=" region",
                    ncols=80 + len(slide_id),
                    position=1,
                    leave=False,
                    disable=not (gpu_id in [-1, 0]),
                ) as t2:
                    for batch in t2:
                        _, region, region_fp = batch
                        region_fp = region_fp[0]
                        region = region.squeeze(0)
                        x_y = Path(region_fp).stem
                        x, y = int(x_y.split("_")[0]), int(x_y.split("_")[1])
                        x_coords.append(x)
                        y_coords.append(y)
                        region = region.to(device, non_blocking=True)
                        feature = model(region)
                        if cfg.inference.save_region_features:
                            save_path = Path(region_features_dir, f"{slide_id}_{x}_{y}.pt")
                            torch.save(feature, save_path)
                            region_feature_paths.append(save_path.resolve())
                            region_slide_ids.append(slide_id)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                save_path = Path(slide_features_dir, f"{slide_id}.pt")
                torch.save(stacked_features, save_path)
                slide_feature_paths.append(save_path.resolve())

                if cfg.wandb.enable and not run_distributed:
                    wandb.log({"processed": i + 1})

    slide_features_df = pd.DataFrame.from_dict(
        {
            "feature_path": slide_feature_paths,
            "slide_id": processed_slide_ids,
            "level": [f"{cfg.inference.level}"] * len(processed_slide_ids),
            "tile_size": [cfg.inference.region_size] * len(processed_slide_ids),
        }
    )

    if run_distributed:
        slide_features_csv_path = Path(output_dir, f"slide_features_{gpu_id}.csv")
    else:
        slide_features_csv_path = Path(output_dir, "slide_features.csv")
    slide_features_df.to_csv(slide_features_csv_path, index=False)

    if cfg.inference.save_region_features:
        region_features_df = pd.DataFrame.from_dict(
            {
                "feature_path": region_feature_paths,
                "slide_id": region_slide_ids,
                "level": [f"{cfg.inference.level}"] * len(region_slide_ids),
                "tile_size": [cfg.inference.region_size] * len(region_slide_ids),
                "x": x_coords,
                "y": y_coords,
            }
        )
        if run_distributed:
            region_features_csv_path = Path(output_dir, f"region_features_{gpu_id}.csv")
        else:
            region_features_csv_path = Path(output_dir, "region_features.csv")
        region_features_df.to_csv(region_features_csv_path, index=False)

    if run_distributed:
        torch.distributed.barrier()
        if distributed.is_main_process():
            slide_dfs = []
            if cfg.inference.save_region_features:
                region_dfs = []
            for gpu_id in range(torch.cuda.device_count()):
                slide_fp = Path(output_dir, f"slide_features_{gpu_id}.csv")
                slide_df = pd.read_csv(slide_fp)
                slide_dfs.append(slide_df)
                os.remove(slide_fp)
                if cfg.inference.save_region_features:
                    region_fp = Path(output_dir, f"region_features_{gpu_id}.csv")
                    region_df = pd.read_csv(region_fp)
                    region_dfs.append(region_df)
                    os.remove(region_fp)
            slide_features_df = pd.concat(slide_dfs, ignore_index=True)
            slide_features_df = slide_features_df.drop_duplicates()
            slide_features_df.to_csv(Path(output_dir, "slide_features.csv"), index=False)
            if cfg.inference.save_region_features:
                region_features_df = pd.concat(region_dfs, ignore_index=True)
                region_features_df = region_features_df.drop_duplicates()
                region_features_df.to_csv(Path(output_dir, "region_features.csv"), index=False)

    if cfg.wandb.enable and distributed.is_main_process() and run_distributed:
        wandb.log({"processed": len(slide_features_df)})


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    args = get_args_parser(add_help=True).parse_args()
    main(args)
