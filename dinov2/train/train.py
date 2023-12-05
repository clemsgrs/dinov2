# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
import wandb
import tqdm
import datetime
from functools import partial
from typing import Optional
from pathlib import Path
from collections import defaultdict

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.data.transforms import make_classification_eval_transform
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup, write_config
from dinov2.utils.utils import CosineScheduler, initialize_wandb, load_weights
from dinov2.models import build_model_from_cfg
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.setup import get_autocast_dtype
from dinov2.eval.metrics import AccuracyAveraging
from dinov2.eval.utils import EarlyStoppingDINO


from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
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


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def update_log_dict(
    log_dict,
    name,
    value,
    step: Optional[str] = "step",
):
    wandb.define_metric(f"{name}", step_metric=step)
    log_dict.update({f"{name}": value})


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_tune(
    cfg,
    epoch,
    model: torch.nn.Module,
    query_dataset,
    test_dataset,
    output_dir,
    gpu_id,
):
    # in DINOv2, they have on SSLMetaArch class
    # first creates student and teacher backbone based on config
    # these are regular ViTs
    # they're added to a dictionnary under the "backbone" key
    # they add a DINOHead to the dictionnary under the "dino_head" key
    # they add a second DINOHead under the "ibot_head" key
    # then they define
    # - self.student = nn.ModuleDict(student_model_dict)
    # - self.teacher = nn.ModuleDict(teacher_model_dict)
    # we can probably skip the DINOHeads and use forward(self, *args, is_training=False)
    # then we get "x_norm_clstoken" = x_norm[:, 0]
    # which should be of dimension embed_dim

    student, teacher, _ = build_model_from_cfg(cfg)
    student.cuda()
    teacher.cuda()
    tqdm.tqdm.write(f"Loading epoch {epoch} weights...")
    student_weights = model.student.state_dict()
    teacher_weights = model.teacher.state_dict()
    load_weights(student, student_weights)
    load_weights(teacher, teacher_weights)
    student.eval()
    teacher.eval()

    autocast_dtype = get_autocast_dtype(cfg)

    student_results = eval_knn_with_model(
        model=student,
        output_dir=output_dir,
        query_dataset=query_dataset,
        test_dataset=test_dataset,
        nb_knn=cfg.tune.knn.nb_knn,
        temperature=cfg.tune.knn.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        gpu_id=gpu_id,
        gather_on_cpu=cfg.tune.knn.gather_on_cpu,
        batch_size=cfg.tune.knn.batch_size,
        num_workers=4,
        n_per_class_list=cfg.tune.knn.n_per_class_list,
        n_tries=cfg.tune.knn.n_tries,
    )

    teacher_results = eval_knn_with_model(
        model=teacher,
        output_dir=output_dir,
        query_dataset=query_dataset,
        test_dataset=test_dataset,
        nb_knn=cfg.tune.knn.nb_knn,
        temperature=cfg.tune.knn.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        gpu_id=gpu_id,
        gather_on_cpu=cfg.tune.knn.gather_on_cpu,
        batch_size=cfg.tune.knn.batch_size,
        num_workers=4,
        n_per_class_list=cfg.tune.knn.n_per_class_list,
        n_tries=cfg.tune.knn.n_tries,
    )

    #######

    results = defaultdict(dict)
    for k in cfg.tune.knn.nb_knn:
        student_acc = student_results[f"{k} Accuracy"]
        student_auc = student_results[f"{k} AUC"]
        teacher_acc = teacher_results[f"{k} Accuracy"]
        teacher_auc = teacher_results[f"{k} AUC"]
        results["student"].update({f"acc_{k}": student_acc, f"auc_{k}": student_auc})
        results["teacher"].update({f"acc_{k}": teacher_acc, f"auc_{k}": teacher_auc})

    return results


def do_train(cfg, model, gpu_id, run_distributed, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    results_save_dir = Path(cfg.train.output_dir, "results")
    if distributed.is_main_process():
        results_save_dir.mkdir(exist_ok=True)

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpoint_save_dir = Path(cfg.train.output_dir, "checkpoints")
    if distributed.is_main_process():
        checkpoint_save_dir.mkdir(exist_ok=True)
    checkpointer = FSDPCheckpointer(model, str(checkpoint_save_dir), optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=cfg.train.save_every * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # setup tuning data

    if cfg.tune.tune_every:
        transform = make_classification_eval_transform()
        query_dataset_str = cfg.tune.query_dataset_path
        test_dataset_str = cfg.tune.test_dataset_path

        query_dataset = make_dataset(
            dataset_str=query_dataset_str,
            transform=transform,
        )
        test_dataset = make_dataset(
            dataset_str=test_dataset_str,
            transform=transform,
        )

    # setup early stopper

    stop = False
    early_stopper = EarlyStoppingDINO(
        cfg.tune.early_stopping.tracking,
        cfg.tune.early_stopping.min_max,
        cfg.tune.early_stopping.patience,
        cfg.tune.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_save_dir,
        verbose=True,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Train"

    for data in metric_logger.log_every(
        data_loader,
        10,
        gpu_id,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        epoch = iteration // OFFICIAL_EPOCH_LENGTH

        # log at the end of each epoch
        if iteration % OFFICIAL_EPOCH_LENGTH == 0:
            if cfg.wandb.enable:
                # log the total loss and each individual loss to wandb
                log_dict = {"epoch": epoch}
                update_log_dict(log_dict, f"{header.lower()}/lr", lr, step="epoch")
                update_log_dict(log_dict, f"{header.lower()}/wd", wd, step="epoch")
                update_log_dict(log_dict, f"{header.lower()}/loss", losses_reduced, step="epoch")
                for loss_name, loss_value in loss_dict.items():
                    update_log_dict(log_dict, f"{header.lower()}/{loss_name}", loss_value, step="epoch")

            # optionally run tuning
            # only run tuning on rank 0, otherwise one has to take care of gathering knn metrics from multiple gpus
            tune_results = None
            if cfg.tune.tune_every and epoch % cfg.tune.tune_every == 0:
                tune_results = do_tune(
                    cfg,
                    epoch + 1,
                    model,
                    query_dataset,
                    test_dataset,
                    results_save_dir,
                    gpu_id,
                )

                if distributed.is_main_process() and cfg.wandb.enable:
                    for model_name, metrics_dict in tune_results.items():
                        for name, value in metrics_dict.items():
                            update_log_dict(log_dict, f"tune/{model_name}.{name}", value, step="epoch")

            if distributed.is_main_process():
                early_stopper(epoch, tune_results, checkpointer, run_distributed, iteration)
                if early_stopper.early_stop and cfg.tune.early_stopping.enable:
                    stop = True

        if stop:
            tqdm.tqdm.write(
                f"Stopping early because best {cfg.tune.early_stopping.tracking} was reached {cfg.tune.early_stopping.patience} epochs ago"
            )
            break

        # save snapshot and log to wandb
        if distributed.is_main_process() and cfg.wandb.enable and iteration % OFFICIAL_EPOCH_LENGTH == 0:
            wandb.log(log_dict, step=epoch)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration, run_distributed=run_distributed)

        iteration = iteration + 1

    # gather stats from all processes
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


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
            wandb_run.define_metric("epoch", summary="max")
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

    if distributed.is_main_process():
        write_config(cfg, cfg.train.output_dir)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, gpu_id, run_distributed, resume=not args.no_resume)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    args = get_args_parser(add_help=True).parse_args()
    main(args)
