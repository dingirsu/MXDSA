# MXDSA

MXDSA is an **MXFP8 DeepSeek Sparse Attention kernel** optimized for NVIDIA Blackwell GPUs.

## Performance Comparison with FlashMLA

The table below shows the maximum speedups recorded on NVIDIA B30Z (SM103), compared with FlashMLA’s **non-fused, head64, head dim 512** path. Speedup is calculated as `FlashMLA latency / MXDSA latency`.

| Scenario | Test Range | Maximum Speedup | Configuration |
| --- | --- | ---: | --- |
| Prefill, sq = 65,536 | topk = 128–4,096 | **2.03×** | topk = 3,968 |
| Decode, batch = 64, sq = 6 | extra topk = 0–4,096 | **2.05×** | extra topk = 3,968 |
| Decode, batch = 128, sq = 6 | extra topk = 0–4,096 | **2.22×** | extra topk = 4,096 |

![MXDSA vs. FlashMLA: Latency and Speedup](assets/comparison_styled.png)

Solid lines show latency, green dashed lines show speedup (right axis), and shaded regions indicate P20–P80. Annotations mark the **maximum speedup** for each group.

### Test Conditions

- **Hardware and software**: NVIDIA B30Z, 148 SMs; CUDA 13.2, PyTorch 2.13.0+cu132.
- **Prefill**: sq = 65,536, KV length = 16,384; topk ranges from 128 to 4,096 in increments of 128.
- **Decode**: batch = 64 / 128, sq = 6; each request has a main KV length and main topk of 128, with an extra KV length of 16,384; extra topk ranges from 0 to 4,096 in increments of 128. Total topk is `128 + extra topk`.

## Building and Usage

Requires a Blackwell GPU, a CUDA Toolkit that supports `sm_100f` / `sm_103f`, and a compiler with C++20 support. Building the Python extension also requires a compatible version of PyTorch. The benchmarks above used CUDA 13.2.

```bash
# Build the PyTorch extension: build/mxdsa_ext.so
./build.sh

# Or build the standalone CUDA library: build/cuda/libmxdsa.so
./build.sh --cuda-only
```

Use `CUDA_HOME`, `PYTHON`, and `BUILD_DIR` to configure the toolchain and output directory. See [tests/benchmark.py](tests/benchmark.py) for Python usage examples and [tests/mxfp8_test_utils.py](tests/mxfp8_test_utils.py) for input-packing helpers.

The extension provides two entry points:

| Interface | Purpose |
| --- | --- |
| `dual_mxfp8_head64_sparse_prefill_fwd` | Prefill; returns the output, max logits, and log-sum-exp |
| `dual_mxfp8_sparse_decode_fwd` | Decode; returns the output, log-sum-exp, and reusable scheduling information |

Q/KV use the repository’s `uint8` packed format, which contains E4M3 data and UE8M0 scales. KV scales are stored after the data region; ordinary floating-point tensors cannot be used as direct replacements. Within each of the two 256-dimensional segments, V scales must follow the rank-1 factorization of token scale × dimension scale, with the corresponding dimension scales passed through `w1` / `w2`. For prefill, each pair of adjacent queries shares one row of indices, with an index shape of `[sq / 2, 1, topk]`. `sq` must be a positive even integer, and `topk` must be a positive multiple of 64. See the [Python bindings](csrc/bindings/binding.cpp) for detailed parameters and validation checks.

## Acknowledgments

Thanks to [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

Thanks to [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass).

The original code in this project is licensed under the [MIT License](LICENSE), with copyright held by dingirsu. Code derived from DeepSeek and NVIDIA CUTLASS is governed by the license terms in [LICENSE.deepseek](LICENSE.deepseek) and [LICENSE.cutlass](LICENSE.cutlass), respectively, with the original copyright notices retained.
