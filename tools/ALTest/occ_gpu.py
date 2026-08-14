"""Keep GPUs moderately busy while yielding regularly to other workloads."""

import argparse
import subprocess
import time

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0,1", help="comma-separated CUDA device IDs")
parser.add_argument(
    "--mem-gb",
    type=float,
    default=20.0,
    help="approximate memory to occupy on EACH selected GPU (default: 20 GiB)",
)
parser.add_argument(
    "--util",
    type=float,
    default=50.0,
    help="target compute duty cycle in percent (default: 50)",
)
parser.add_argument(
    "--period-ms",
    type=float,
    default=100.0,
    help="compute/sleep control period in milliseconds (default: 100)",
)
parser.add_argument(
    "--matrix-size",
    type=int,
    default=2048,
    help="square FP16 GEMM size; smaller kernels interfere less (default: 2048)",
)
parser.add_argument(
    "--poll-seconds",
    type=float,
    default=2.0,
    help="seconds between utilization checks (default: 2)",
)
parser.add_argument(
    "--no-adaptive",
    action="store_true",
    help="disable nvidia-smi feedback and always use the requested duty cycle",
)
args = parser.parse_args()

gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
if not gpus:
    parser.error("--gpus must contain at least one device ID")
if args.mem_gb < 0:
    parser.error("--mem-gb must be non-negative")
if not 0 < args.util < 100:
    parser.error("--util must be between 0 and 100")
if args.period_ms <= 0:
    parser.error("--period-ms must be positive")
if args.matrix_size <= 0:
    parser.error("--matrix-size must be positive")
if args.poll_seconds <= 0:
    parser.error("--poll-seconds must be positive")


def read_gpu_utilization():
    """Return {physical_gpu_index: utilization_percent}, or None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            int(index.strip()): float(util.strip())
            for index, util in (line.split(",") for line in result.stdout.splitlines())
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


per_gpu_bytes = int(args.mem_gb * 1024**3)
buffers = []
work = []
streams = []

for gpu in gpus:
    with torch.cuda.device(gpu):
        # priority=0 is the lowest/default PyTorch CUDA stream priority.  Using
        # relatively small GEMMs limits the time another process can be delayed
        # by any single non-preemptible kernel.
        stream = torch.cuda.Stream(device=gpu, priority=0)
        streams.append(stream)

        if per_gpu_bytes:
            buffers.append(
                torch.empty(per_gpu_bytes, device=f"cuda:{gpu}", dtype=torch.uint8)
            )
        n = args.matrix_size
        a = torch.randn(n, n, device=f"cuda:{gpu}", dtype=torch.float16)
        b = torch.randn(n, n, device=f"cuda:{gpu}", dtype=torch.float16)
        c = torch.empty(n, n, device=f"cuda:{gpu}", dtype=torch.float16)
        work.append((a, b, c))

for gpu in gpus:
    torch.cuda.synchronize(gpu)

print(
    f"GPU keeper started on {gpus}; memory≈{args.mem_gb:g} GiB/GPU, "
    f"target duty cycle={args.util:g}%, period={args.period_ms:g} ms. "
    "Ctrl+C to stop."
)

period = args.period_ms / 1000.0
# Adaptive mode starts idle and ramps up only after observing spare capacity.
# This avoids adding a full 50% burst when a real job is already running.
duties = [args.util if args.no_adaptive else 0.0] * len(gpus)
next_poll = 0.0
warned_about_poll = False

try:
    while True:
        now = time.monotonic()
        if not args.no_adaptive and now >= next_poll:
            observed = read_gpu_utilization()
            if observed is None:
                if not warned_about_poll:
                    print("Warning: cannot read nvidia-smi; using fixed duty cycle.")
                    warned_about_poll = True
                duties = [args.util] * len(gpus)
            else:
                # Feedback controller: reduce our duty when other work raises the
                # measured total, and restore it gradually when the GPU is idle.
                for i, gpu in enumerate(gpus):
                    if gpu in observed:
                        error = args.util - observed[gpu]
                        # The submit window may need to exceed the utilization
                        # target because launch/synchronization overhead is idle
                        # GPU time.  Feedback is therefore allowed up to 95%.
                        duties[i] = min(95.0, max(0.0, duties[i] + 0.5 * error))
            next_poll = now + args.poll_seconds

        cycle_start = time.perf_counter()
        busy_budgets = [period * duty / 100.0 for duty in duties]

        # Submit one small GEMM per GPU, then synchronize all GPUs together.
        # Each GPU has its own feedback-adjusted budget so a busy card can yield
        # without preventing an idle selected card from maintaining utilization.
        while True:
            elapsed = time.perf_counter() - cycle_start
            active = [i for i, budget in enumerate(busy_budgets) if elapsed < budget]
            if not active:
                break
            for i in active:
                gpu = gpus[i]
                stream = streams[i]
                a, b, c = work[i]
                with torch.cuda.device(gpu), torch.cuda.stream(stream):
                    torch.mm(a, b, out=c)
            for i in active:
                torch.cuda.synchronize(gpus[i])

        remaining = period - (time.perf_counter() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
except KeyboardInterrupt:
    print("Stopped.")
