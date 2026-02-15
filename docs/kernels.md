### Setting up kernels

- **MLX attention backend (Apple Silicon)** can be enabled to run Evoformer attention through MLX instead of the default PyTorch attention path. To use this feature, run:
  ```bash
  pip install mlx
  export USE_MLX_EVO_ATTENTION=true
  ```
  This backend is intended for macOS Apple Silicon and is useful when running with the `mps` device.
  You can also enable a dedicated MLX triangle-attention wrapper (cuEquivariance-style mask handling) with:
  ```bash
  export USE_MLX_TRIANGLE_ATTENTION=true
  ```

- **Custom CUDA layernorm kernels** modified from [FastFold](https://github.com/hpcaitech/FastFold) and [Oneflow](https://github.com/Oneflow-Inc/oneflow) accelerate about 30%-50% during different training stages. To use this feature, run the following command:
  ```bash
  export LAYERNORM_TYPE=fast_layernorm
  ```
  If the environment variable `LAYERNORM_TYPE` is set to `fast_layernorm`, the model will employ the layernorm we have developed; otherwise, the naive PyTorch layernorm will be adopted. The kernels will be compiled when `fast_layernorm` is called for the first time.
- **[DeepSpeed DS4Sci_EvoformerAttention kernel](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/)** is a memory-efficient attention kernel developed as part of a collaboration between OpenFold and the DeepSpeed4Science initiative. To use this feature, run the following command:
  ```bash
  export USE_DEEPSPEED_EVO_ATTENTION=true
  ```
  DS4Sci_EvoformerAttention is implemented based on [CUTLASS](https://github.com/NVIDIA/cutlass). If you use this feature, You need to clone the CUTLASS repository and specify the path to it in the environment variable CUTLASS_PATH. you can set environment variable `CUTLASS_PATH` as follows:

  ```bash
  git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git  /path/to/cutlass
  export CUTLASS_PATH=/path/to/cutlass
  ```

  The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time.
