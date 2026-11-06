"""Benchmark MXDSA with CUDA graphs; input packing is excluded from timing."""
import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from mxfp8_test_utils import (
    pack_dual_decode_kv_pages,
    pack_dual_prefill_kv64,
    pack_dual_q64,
)

ROOT = Path(__file__).resolve().parents[1]


def load_extension():
    path = ROOT / "build/mxdsa_ext.so"
    if not path.is_file():
        raise FileNotFoundError(f"Build the extension first with ./build.sh: {path}")
    spec = importlib.util.spec_from_file_location("mxdsa_ext", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packed_q(shape):
    # Bound temporary FP32 packing storage for the 32k x 64 x 512 Q tensor.
    out = torch.empty((*shape, 64, 528), device="cuda", dtype=torch.uint8)
    flat = out.view(-1, 64, 528)
    for i in range(0, len(flat), 256):
        flat[i:i + 256] = pack_dual_q64(
            torch.randn(len(flat[i:i + 256]), 64, 512, device="cuda")
        )[0]
    return out


def measure(launch, iterations, warmup, flush):
    for _ in range(3):
        launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _captured = launch()  # Keep output storage alive through all replays.
    for _ in range(warmup):
        graph.replay()
    events = []
    for _ in range(iterations):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) * 1000 for start, end in events]
    ordered = sorted(samples)
    return dict(
        median_us=statistics.median(samples),
        p20_us=ordered[len(ordered) // 5],
        p80_us=ordered[4 * len(ordered) // 5],
        samples_us=samples,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--topk", type=int, nargs="+", default=list(range(128, 1025, 128)))
    parser.add_argument("--extra", type=int, nargs="+", default=list(range(0, 1025, 128)))
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "build/benchmark.json")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.flush_mb < 0:
        parser.error("iterations must be positive; warmup and flush-mb must be nonnegative")
    if any(k <= 0 or k % 64 for k in args.topk):
        parser.error("topk must be positive multiples of 64")
    if any(k < 0 for k in args.extra):
        parser.error("extra must be nonnegative")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if props.major != 10:
        parser.error(f"MXDSA requires SM100/SM103; got {props.name} (SM{props.major}{props.minor})")
    torch.manual_seed(20261105)
    ext = load_extension()
    weights = [0x7f7f7f7f] * 8
    flush = (
        torch.empty(args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if args.flush_mb else None
    )
    report = dict(
        gpu=props.name,
        sm=f"{props.major}.{props.minor}",
        sm_count=props.multi_processor_count,
        torch=torch.__version__,
        iterations=args.iterations,
        warmup=args.warmup,
        flush_mb=args.flush_mb,
        method="CUDA graph; decode phase1 + combine with cached scheduler",
        sha256=hashlib.sha256((ROOT / "build/mxdsa_ext.so").read_bytes()).hexdigest(),
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "samples_us"}), flush=True)

    if args.mode in ["prefill", "both"]:
        q = packed_q((32768,))
        kv = pack_dual_prefill_kv64(torch.randn(32768, 1, 512, device="cuda"))[0]
        for topk in args.topk:
            idx = torch.randint(32768, (16384, 1, topk), device="cuda", dtype=torch.int32)

            def launch():
                return ext.dual_mxfp8_head64_sparse_prefill_fwd(
                    q, kv, idx, 512**-.5, weights, weights, None, None
                )

            record(dict(
                mode="prefill", s_q=32768, s_kv=32768, topk=topk,
                **measure(launch, args.iterations, args.warmup, flush),
            ))
            del idx
        del q, kv
    if args.mode in ["decode", "both"]:
        q = packed_q((256, 4))
        pool = pack_dual_decode_kv_pages(torch.randn(512, 64, 1, 512, device="cuda"))[0]
        # Each request has distinct addresses covering 32k KV tokens.
        kv = pool.repeat(256, 1, 1, 1)
        epool = pack_dual_decode_kv_pages(torch.randn(512, 64, 1, 512, device="cuda"))[0]
        ekv = epool.repeat(256, 1, 1, 1)
        offset = torch.arange(256, device="cuda", dtype=torch.int32)[:, None, None] * 32768
        idx = torch.randint(32768, (256, 4, 128), device="cuda", dtype=torch.int32) + offset
        for extra in args.extra:
            ei = (
                torch.randint(32768, (256, 4, extra), device="cuda", dtype=torch.int32) + offset
                if extra else None
            )

            def launch(meta=None, splits=None):
                return ext.dual_mxfp8_sparse_decode_fwd(
                    q, kv, idx, None, None, meta, splits, ekv if extra else None,
                    ei, None, 512, 512, 512**-.5, weights, weights,
                )

            outputs = launch()
            meta, splits = outputs[2:]
            record(dict(
                mode="decode", batch=256, s_q=4, kv_length_per_request=32768,
                topk=128, extra_topk=extra,
                **measure(lambda: launch(meta, splits), args.iterations, args.warmup, flush),
            ))
            del outputs, meta, splits, ei


if __name__ == "__main__":
    main()
