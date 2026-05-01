# CS336 Systems Profiling Report

## Environment

- Platform: Ubuntu 24.04.3 LTS on WSL2
- Python: 3.12.3 via `uv run`
- PyTorch: 2.11.0+cu130
- CUDA reported by PyTorch: 13.0
- GPU: NVIDIA GeForce RTX 5070 Ti
- Profiler: NVIDIA Nsight Systems 2025.3.2

## Command

The final profiling run used:

```bash
uv run nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  --output cs336_benchmark \
  --force-overwrite=true \
  -- python -m cs336_basics.benchmarking_script --device cuda --num-layers 4
```

The original `--gpu-metrics-devices=0` option was omitted because Nsight Systems rejected it under the current WSL permissions:

```text
Illegal --gpu-metrics-devices argument: 0.
Insufficient privilege: ERR_NVGPUCTRPERM.
```

## Result

The benchmark completed successfully with CUDA enabled and produced:

```text
cs336_benchmark.nsys-rep
```

Benchmark output:

```text
Mode: optimizer_step
Warmup steps: 5
Measurement steps: 10
Mean step time: 0.086384 seconds
Std step time: 0.013138 seconds
```

## Notes

- The profiling target was `cs336_basics.benchmarking_script`.
- The run used `--num-layers 4`, matching the requested local equivalent of the original `benchmark.py 4` command.
- Nsight Systems was configured to run from a WSL-native path to avoid `LD_PRELOAD` issues caused by spaces in the Windows installation path.
