# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from typing import Any, Tuple
from urllib.parse import urlparse

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k, v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
            logger.info(f"{k} | ckpt size: {v.size()} | model size: {model_dict[k].size()}")
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f"{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes"
    return updated_state_dict, msg


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict, msg1 = update_state_dict(model.state_dict(), state_dict)
    msg2 = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Pretrained weights found at: {pretrained_weights}")
    logger.info(msg1)
    logger.info(f"Pretrained weights loaded with msg: {msg2}")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, config.student.pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(config) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    model = build_model_for_eval(config)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        cfg,
        verbose: bool = True,
    ):
        super(PatchEmbedder, self).__init__()

        self.vit, self.autocast_dtype = setup_and_build_model(cfg)

        if verbose:
            print("Freezing Vision Transformer")
        for param in self.vit.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

    def forward(self, x):
        # x = [B, 3, img_size, img_size]
        feature = self.vit(x).detach().cpu()  # [B, 384]
        return feature
