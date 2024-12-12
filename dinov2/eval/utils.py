# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional
from pathlib import Path
import tqdm

import torch
from torch import nn
from torchmetrics import MetricCollection
from torch.nn.functional import one_hot

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    num_classes,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    verbose: bool = True,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    header = "Test"

    for samples, targets, *_ in metric_logger.log_every(data_loader, device, 10, header):
        # given model went through ModelWithNormalize, outputs are already normalized
        outputs = model(samples.to(device))
        targets = targets.to(device)
        one_hot_targets = one_hot(targets, num_classes=num_classes)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, one_hot_targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    if verbose:
        logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(
    model, dataset, batch_size, num_workers, gpu_id, gather_on_cpu=False, header: str = "", verbose: bool = True
):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(
        model, data_loader, sample_count, gpu_id, gather_on_cpu, header=header, verbose=verbose
    )


@torch.inference_mode()
def extract_features_with_dataloader(
    model, data_loader, sample_count, gpu_id, gather_on_cpu=False, header: str = "", verbose: bool = True
):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ", verbose=verbose)
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, gpu_id, 10, header):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    if verbose:
        logger.info(f"Features shape: {tuple(features.shape)}")
    if verbose:
        logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


class EarlyStoppingDINO:
    """
    Leverage a downstream classification task to know if teacher still outperforms student
    """

    def __init__(
        self,
        tracking: str,
        min_max: str,
        nepochs: int,
        patience_pct: int = 0.2,
        min_epoch_pct: int = 0.3,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """
        Args:
            tracking (str): Metric to track for early stopping
            min_max (str): Whether to minimize or maximize the tracking metric
            nepochs (int): Total number of epochs
            patience_pct (int): Percentage of epochs to wait before early stopping
            min_epoch_pct (int): Percentage of epochs to wait before enabling early stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = int(round(patience_pct * nepochs, 0))
        self.min_epoch = int(round(min_epoch_pct * nepochs, 0))
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, results, checkpointer, run_distributed, iteration):
        save_best = False
        if results is not None and len(results) > 0:
            teacher_score = results["teacher"][f"{self.tracking}"]
            student_score = results["student"][f"{self.tracking}"]

            if self.min_max == "min":
                teacher_score = -1 * teacher_score
                student_score = -1 * student_score

            if self.best_score is None or (teacher_score > self.best_score and teacher_score > student_score):
                self.best_score = teacher_score
                fname = "best"
                save_best = True
                self.counter = 0

            elif teacher_score <= self.best_score or teacher_score <= student_score:
                self.counter += 1
                if epoch <= self.min_epoch + 1 and self.verbose:
                    tqdm.tqdm.write(f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}")
                elif self.verbose:
                    tqdm.tqdm.write(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience and epoch > self.min_epoch:
                    self.early_stop = True

        torch.distributed.barrier()

        save_best_tensor = torch.tensor(save_best, dtype=torch.bool).cuda()
        # broadcast the tensor from the main process (rank 0) to all others
        torch.distributed.broadcast(save_best_tensor, src=0)
        # convert the tensor back to a boolean
        save_best = save_best_tensor.item()

        if save_best:
            checkpointer.save("best", run_distributed=run_distributed, iteration=iteration)

        torch.distributed.barrier()

        # override latest
        fname = "latest"
        checkpointer.save(fname, run_distributed=run_distributed, iteration=iteration)
