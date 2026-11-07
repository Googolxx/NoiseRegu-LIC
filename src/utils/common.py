# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def set_torch_env():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:
        pass


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "p_frame_net_state_dict" in ckpt:
        ckpt = ckpt["p_frame_net_state_dict"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt
