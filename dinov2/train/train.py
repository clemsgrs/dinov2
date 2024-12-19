import argparse
import logging
import math
import os
import time
import json
import wandb
import tqdm
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
from dinov2.logging import MetricLogger, SmoothedValue
from dinov2.utils.config import setup, write_config
from dinov2.utils.utils import CosineScheduler, load_weights
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
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg, OFFICIAL_EPOCH_LENGTH):
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=int(round(cfg.optim["warmup_pct"] * cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH, 0)),
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
        total_iters=int(round(cfg.teacher["warmup_teacher_temp_pct"] * cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH, 0)),
        warmup_iters=int(
            round(cfg.teacher["warmup_teacher_temp_pct"] * cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH, 0)
        ),
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : int(round(cfg.optim["freeze_last_layer_pct"] * cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH, 0))
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


def save_checkpoint(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        checkpoint_dir = Path(cfg.train.output_dir, "checkpoints", "teacher")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        # save teacher checkpoint
        teacher_ckp_path = Path(checkpoint_dir, f"teacher_{iteration}.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_tune(
    cfg,
    iteration: int,
    model: torch.nn.Module,
    query_dataset,
    test_dataset,
    output_dir,
    verbose: bool = True,
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

    # from dinov2.models.vision_transformer import BlockChunk
    # from dinov2.fsdp import get_fsdp_wrapper

    # student_model_cfg = cfg.compute_precision.student["backbone"]
    # student = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(student)
    # teacher_model_cfg = cfg.compute_precision.teacher["backbone"]
    # teacher = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(teacher)

    student = student.to(torch.device("cuda"))
    teacher = teacher.to(torch.device("cuda"))
    # student = student.to(torch.device(f"cuda:{distributed.get_global_rank()}"))
    # teacher = teacher.to(torch.device(f"cuda:{distributed.get_global_rank()}"))
    if verbose:
        tqdm.tqdm.write(f"Loading iteration {iteration} weights...")
    student_weights = model.student.state_dict()
    teacher_weights = model.teacher.state_dict()
    student_msg = load_weights(student, student_weights)
    teacher_msg = load_weights(teacher, teacher_weights)

    if len(student_msg.missing_keys) > 0:
        tqdm.tqdm.write(str(student_msg))
    if len(teacher_msg.missing_keys) > 0:
        tqdm.tqdm.write(str(teacher_msg))

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
        gpu_id=distributed.get_global_rank(),
        gather_on_cpu=cfg.tune.knn.gather_on_cpu,
        batch_size=cfg.tune.knn.batch_size,
        num_workers=0,
        persistent_workers=False,
        n_per_class_list=cfg.tune.knn.n_per_class_list,
        n_tries=cfg.tune.knn.n_tries,
        model_name="student",
        verbose=verbose,
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
        gpu_id=distributed.get_global_rank(),
        gather_on_cpu=cfg.tune.knn.gather_on_cpu,
        batch_size=cfg.tune.knn.batch_size,
        num_workers=0,
        persistent_workers=False,
        n_per_class_list=cfg.tune.knn.n_per_class_list,
        n_tries=cfg.tune.knn.n_tries,
        model_name="teacher",
        verbose=verbose,
    )

    #######

    results = defaultdict(dict)
    if distributed.is_main_process():
        for k in cfg.tune.knn.nb_knn:
            student_acc = student_results[f"{k} Accuracy"]
            student_auc = student_results[f"{k} AUC"]
            teacher_acc = teacher_results[f"{k} Accuracy"]
            teacher_auc = teacher_results[f"{k} AUC"]
            results["student"].update({f"acc_{k}": student_acc, f"auc_{k}": student_auc})
            results["teacher"].update({f"acc_{k}": teacher_acc, f"auc_{k}": teacher_auc})

    return results


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    results_save_dir = Path(cfg.train.output_dir, "results")
    if distributed.is_main_process():
        results_save_dir.mkdir(exist_ok=True)

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())

    # checkpointer
    checkpoint_save_dir = Path(cfg.train.output_dir, "checkpoints")
    if distributed.is_main_process():
        checkpoint_save_dir.mkdir(exist_ok=True)
    checkpointer = FSDPCheckpointer(model, str(checkpoint_save_dir), optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

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

    total_batch_size = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    OFFICIAL_EPOCH_LENGTH = len(dataset) // total_batch_size
    save_every = int(round(cfg.train.save_frequency * OFFICIAL_EPOCH_LENGTH, 0))
    if cfg.optim.max_iter is not None:
        max_iter = cfg.optim.max_iter
    else:
        max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=save_every,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup schedulers
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg, OFFICIAL_EPOCH_LENGTH)

    # setup tuning data

    tune_every_iter = None
    if cfg.tune.tune_every_pct:
        tune_every_iter = int(round(cfg.tune.tune_every_pct * OFFICIAL_EPOCH_LENGTH, 0))

    if tune_every_iter:
        transform = make_classification_eval_transform(image_size=cfg.tune.tile_size)
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
        cfg.optim.epochs,
        cfg.tune.early_stopping.patience_pct,
        cfg.tune.early_stopping.min_epoch_pct,
        checkpoint_dir=checkpoint_save_dir,
        verbose=True,
    )

    # training loop

    iteration = start_iter
    run_distributed = distributed.is_enabled() and torch.cuda.device_count() > 1

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    log_freq = 10  # log_freq has to be smaller than the window_size used with instantiating SmoothedValue (here and in MetricLogger)

    forward_backward_time = SmoothedValue(fmt="{avg:.6f}")

    for data in metric_logger.log_every(
        data_loader,
        distributed.get_global_rank(),
        log_freq,
        "Train",
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
        forward_backward_start = time.time()
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        forward_backward_time.update(time.time() - forward_backward_start)

        if metrics_file is not None and distributed.is_main_process():
            if (log_freq is not None and iteration % log_freq == 0) or iteration == max_iter - 1:
                dict_to_dump = dict(
                    iteration=iteration,
                    forward_backward_time=forward_backward_time.avg,
                )
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(dict_to_dump) + "\n")

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

        # logging
        if distributed.is_main_process() and cfg.wandb.enable:
            log_dict = {"iteration": iteration, "epoch": epoch}
            update_log_dict(log_dict, "train/lr", lr, step="iteration")
            update_log_dict(log_dict, "train/wd", wd, step="iteration")
            update_log_dict(log_dict, "train/loss", losses_reduced, step="iteration")
            for loss_name, loss_value in loss_dict.items():
                update_log_dict(log_dict, f"train/{loss_name}", loss_value, step="iteration")

        # optionally run tuning
        tune_results = None
        if distributed.is_main_process() and tune_every_iter and iteration % tune_every_iter == 0:
            # only run tuning on rank 0, otherwise one has to take care of gathering knn metrics from multiple gpus
            tune_results = do_tune(
                cfg,
                iteration,
                model,
                query_dataset,
                test_dataset,
                results_save_dir,
                verbose=False,
            )

            if cfg.wandb.enable:
                for model_name, metrics_dict in tune_results.items():
                    for name, value in metrics_dict.items():
                        update_log_dict(log_dict, f"tune/{model_name}.{name}", value, step="iteration")

            early_stopper(epoch, tune_results, periodic_checkpointer, run_distributed, iteration)
            if early_stopper.early_stop and cfg.tune.early_stopping.enable:
                stop = True

        # log to wandb
        if distributed.is_main_process() and cfg.wandb.enable:
            wandb.log(log_dict, step=iteration)

        # checkpointing and testing

        if cfg.train.save_frequency > 0 and (iteration + 1) % save_every == 0:
            save_checkpoint(cfg, model, iteration + 1)
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration, run_distributed=run_distributed)

        iteration = iteration + 1

        if stop:
            if distributed.is_main_process():
                tqdm.tqdm.write(
                    f"Stopping early because best {early_stopper.tracking} was reached {early_stopper.patience} epochs ago"
                )
            break

    # gather stats from all processes

    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


def main(args):
    cfg = setup(args)
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
        return save_checkpoint(cfg, model, iteration)

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    args = get_args_parser(add_help=True).parse_args()
    main(args)
