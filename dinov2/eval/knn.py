# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import datetime
import argparse
from functools import partial
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.eval.metrics import AccuracyAveraging, build_metric
from dinov2.utils.utils import initialize_wandb
from dinov2.utils.config import setup, write_config
from dinov2.models import setup_and_build_model
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features
from dinov2.data.transforms import make_classification_eval_transform


logger = logging.getLogger("dinov2")


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
        default="./output",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the query features

    Each rank gets a chunk of the query features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of query features
    then collated back on the original device.
    """

    def __init__(self, query_features, query_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()

        self.device = device
        self.query_features_rank_T = query_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = query_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, query_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(query_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `query_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.query_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        preds = features_dict / features_dict.sum(dim=-1).unsqueeze(-1)
        return {"preds": preds, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, query_features, query_labels):
    modules = {}
    mapping = create_class_indices_mapping(query_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                query_features=query_features,
                query_labels=query_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_query(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                query_features=query_features[final_indices],
                query_labels=query_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_query(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def eval_knn(
    model,
    query_dataset,
    test_dataset,
    accuracy_averaging,
    gpu_id,
    nb_knn,
    temperature,
    batch_size,
    num_workers,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
    persistent_workers: bool = True,
    verbose: bool = True,
):
    model = ModelWithNormalize(model)

    if verbose:
        logger.info("Extracting features for query set...")
    query_features, query_labels = extract_features(
        model,
        query_dataset,
        batch_size,
        num_workers,
        gpu_id,
        gather_on_cpu=gather_on_cpu,
        header="Query",
        verbose=verbose,
    )
    # given model went through ModelWithNormalize, query_features are already normalized
    if verbose:
        logger.info(f"Query features created, shape {tuple(query_features.shape)}.")

    test_dataloader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        verbose=verbose,
    )
    num_classes = query_labels.max() + 1
    metric_collection = build_metric(num_classes=num_classes, average_type=accuracy_averaging)

    device = torch.cuda.current_device()
    partial_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        query_features=query_features,
        query_labels=query_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    if verbose:
        logger.info("Start the k-NN classification.")
    _, results_dict = evaluate(
        model_with_knn, test_dataloader, num_classes, postprocessors, metrics, device, verbose=verbose
    )

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


def eval_knn_with_model(
    model,
    output_dir,
    query_dataset,
    test_dataset,
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
    gpu_id=-1,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=4,
    n_per_class_list=[-1],
    n_tries=1,
    persistent_workers: bool = True,
    model_name: Optional[str] = None,
    verbose: bool = False,
):
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            query_dataset=query_dataset,
            test_dataset=test_dataset,
            accuracy_averaging=accuracy_averaging,
            gpu_id=gpu_id,
            nb_knn=nb_knn,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
            persistent_workers=persistent_workers,
            verbose=verbose,
        )

    results_dict = {}
    if distributed.is_main_process():
        for knn_ in results_dict_knn.keys():
            k = knn_[1]
            acc = results_dict_knn[knn_]["acc"].item() * 100.0
            auc = results_dict_knn[knn_]["auc"].item()
            results_dict[f"{k} Accuracy"] = acc
            results_dict[f"{k} AUC"] = auc
            if model_name and verbose:
                logger.info(f"{model_name.title()} | {k}-NN classifier result: Accuracy: {acc:.2f} | AUC: {auc:.5f}")
            elif verbose:
                logger.info(f"{k}-NN classifier result: Accuracy: {acc:.2f} | AUC: {auc:.5f}")

        metrics_file_path = Path(output_dir, "results_eval_knn.json")
        with open(metrics_file_path, "a") as f:
            for k, v in results_dict.items():
                if model_name:
                    k = f"{model_name.title()} {k}"
                f.write(json.dumps({k: v}) + "\n")

    if distributed.is_enabled():
        torch.distributed.barrier()

    return results_dict


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

    if distributed.is_main_process():
        write_config(cfg, cfg.train.output_dir)

    model, autocast_dtype = setup_and_build_model(cfg)

    transform = make_classification_eval_transform(image_size=cfg.data.image_size)
    query_dataset_str = cfg.data.query_dataset
    test_dataset_str = cfg.data.test_dataset
    query_dataset = make_dataset(
        dataset_str=query_dataset_str,
        transform=transform,
    )
    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=transform,
    )

    eval_knn_with_model(
        model=model,
        output_dir=cfg.train.output_dir,
        query_dataset=query_dataset,
        test_dataset=test_dataset,
        nb_knn=cfg.knn.nb_knn,
        temperature=cfg.knn.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        gpu_id=-1,
        gather_on_cpu=cfg.speed.gather_on_cpu,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.speed.num_workers,
        n_per_class_list=cfg.knn.n_per_class_list,
        n_tries=cfg.knn.n_tries,
        verbose=True,
    )

    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    args = get_args_parser(add_help=True).parse_args()
    main(args)
