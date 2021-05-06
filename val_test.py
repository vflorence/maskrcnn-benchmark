# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import glob
import math

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

output_dir = "./output/SRN_object_orig_output/"
model_files = glob.glob(output_dir + 'model_*')
model_files = sorted(model_files)
config_file = output_dir+"config.yml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["DATASETS.TRAIN", ['srn_objects_orig_val',]])
cfg.freeze()

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

save_to_disk = get_rank() == 0

model_val_loss_dict = {}
for m in model_files:
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )    
    it = iter(data_loader)
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir = cfg.OUTPUT_DIR)
    extra_checkpoint_data = checkpointer.load(m)
    losses = 0
    for i in range(0,math.ceil(len(data_loader.dataset)/data_loader.batch_sampler.batch_sampler.batch_size)):
        with torch.no_grad():
            (images, targets, _) = next(it)
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            loss_dict = model(images, targets)
            losses += sum(loss for loss in loss_dict.values())
    
    print('model: ', m, '  loss: ', losses / math.ceil(len(data_loader.dataset)/data_loader.batch_sampler.batch_sampler.batch_size))
    model_val_loss_dict[m] = losses / math.ceil(len(data_loader.dataset)/data_loader.batch_sampler.batch_sampler.batch_size)

print(model_val_loss_dict)
import IPython; IPython.embed()

