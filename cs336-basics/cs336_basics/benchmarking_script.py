from __future__ import annotations

import argparse
from dataclasses import dataclass
import statistics
import timeit

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

StepMode = str


@dataclass(frozen=True)
class BenchmarkConfig:
    vocab_size: int = 10_000
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 1_024
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 5
    benchmark_steps: int = 10
    mode: StepMode = "optimizer_step"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    return model.to(config.device)


def build_optimizer(model: BasicsTransformerLM, config: BenchmarkConfig) -> AdamW:
    return AdamW(model.parameters(), lr=config.learning_rate)


def generate_random_batch(config: BenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=config.device,
        dtype=torch.long,
    )
    y = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=config.device,
        dtype=torch.long,
    )
    return x, y


def run_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: StepMode,
) -> None:
    if mode == "forward":
        with torch.no_grad():
            _ = model(x)
        return

    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    if mode == "optimizer_step":
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elif mode == "forward_backward":
        optimizer.zero_grad(set_to_none=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def maybe_sync_cuda(device: str) -> None:
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> list[float]:
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    x, y = generate_random_batch(config)
    model.train()

    # Warm-up (not timed)
    for _ in range(config.warmup_steps):
        run_step(model, optimizer, x, y, config.mode)
        maybe_sync_cuda(config.device)

    timings: list[float] = []
    for _ in range(config.benchmark_steps):
        maybe_sync_cuda(config.device)
        start_time = timeit.default_timer()
        run_step(model, optimizer, x, y, config.mode)
        maybe_sync_cuda(config.device)
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
    return timings


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Basic end-to-end model benchmark skeleton.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("forward", "forward_backward", "optimizer_step"),
        default="optimizer_step",
        help="Which step type to run in benchmark loop.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    return BenchmarkConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        mode=args.mode,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    timings = benchmark_model(config)

    mean_s = statistics.mean(timings)
    std_s = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print(f"Mode: {config.mode}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Measurement steps: {config.benchmark_steps}")
    print(f"Mean step time: {mean_s:.6f} seconds")
    print(f"Std step time: {std_s:.6f} seconds")


if __name__ == "__main__":
    main()
