# Copyright 2022 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from contextlib import nullcontext

import torch

def is_fp16_enabled():
    if not torch.is_autocast_enabled():
        return False

    if hasattr(torch, "get_autocast_dtype"):
        for device_type in ("cuda", "mps", "cpu"):
            try:
                if torch.get_autocast_dtype(device_type) == torch.float16:
                    return True
            except Exception:
                continue
        return False

    # Fallback for older PyTorch
    return torch.get_autocast_gpu_dtype() == torch.float16


def disable_backend_autocast(device: torch.device):
    if device.type in {"cuda", "mps"}:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return nullcontext()
