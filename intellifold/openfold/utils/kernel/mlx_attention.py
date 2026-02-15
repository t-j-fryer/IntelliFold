# Copyright 2024 IntelliGen-AI and/or its affiliates.
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

from typing import List

import mlx.core as mx
import numpy as np
import torch


def _to_mx(x: torch.Tensor) -> mx.array:
    """Convert torch tensor to mlx array, preferring DLPack when available."""
    from_dlpack = getattr(mx, "from_dlpack", None)
    if from_dlpack is not None:
        try:
            dlpack = torch.utils.dlpack.to_dlpack(x.detach().contiguous())
            return from_dlpack(dlpack)
        except Exception:
            pass

    return mx.array(x.detach().to(device="cpu", dtype=torch.float32).numpy())


def _to_torch(x: mx.array, ref: torch.Tensor) -> torch.Tensor:
    """Convert mlx array back to torch tensor on reference device/dtype."""
    to_dlpack = getattr(mx, "to_dlpack", None)
    if to_dlpack is not None:
        try:
            out = torch.utils.dlpack.from_dlpack(to_dlpack(x))
            return out.to(device=ref.device, dtype=ref.dtype)
        except Exception:
            pass

    mx.eval(x)
    output_np = np.array(x)
    return torch.from_numpy(output_np).to(device=ref.device, dtype=ref.dtype)


def mlx_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    """MLX attention implementation with the same contract as _attention().

    Input shapes:
      q: [*, H, Q, C]
      k: [*, H, K, C]
      v: [*, H, K, C]
      biases: broadcastable to [*, H, Q, K]

    Returns:
      [*, H, Q, C]
    """
    q_mx = _to_mx(q)
    k_mx = _to_mx(k)
    v_mx = _to_mx(v)

    logits = mx.matmul(q_mx, mx.swapaxes(k_mx, -1, -2))
    for b in biases:
        logits = logits + _to_mx(b)

    probs = mx.softmax(logits, axis=-1)
    output = mx.matmul(probs, v_mx)

    return _to_torch(output, q)


def mlx_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    """MLX triangle-attention wrapper compatible with cuEq-style call contracts.

    Expected biases order: [mask_bias, triangle_bias].
    """
    if len(biases) != 2:
        raise ValueError("Triangle attention requires two bias terms: mask_bias and triangle_bias")

    mask_bias, triangle_bias = biases

    is_batched_input = False
    original_shape = None

    if len(q.shape) > 5:
        if len(q.shape) != 6:
            raise ValueError("Max number of dimensions for triangle attention is 6")

        is_batched_input = True
        original_shape = q.shape
        batch, n_tmpl = q.shape[:2]

        q = q.view(batch * n_tmpl, *q.shape[2:])
        k = k.view(batch * n_tmpl, *k.shape[2:])
        v = v.view(batch * n_tmpl, *v.shape[2:])
        mask_bias = mask_bias.view(batch * n_tmpl, *mask_bias.shape[2:])
        triangle_bias = triangle_bias.view(batch * n_tmpl, *triangle_bias.shape[2:])

    if mask_bias.dtype != torch.bool:
        mask_bias = mask_bias == 0

    mask_bias_additive = torch.where(
        mask_bias,
        torch.zeros_like(mask_bias, dtype=q.dtype),
        torch.full_like(mask_bias, -float("inf"), dtype=q.dtype),
    )

    output = mlx_attention(q, k, v, [mask_bias_additive, triangle_bias])

    if len(q.shape) == 4 and output.shape[0] == 1:
        output = output.squeeze(0)

    if is_batched_input:
        output = output.view(original_shape[0], original_shape[1], *output.shape[1:])

    return output.transpose(-2, -3)
