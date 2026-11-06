#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./build.sh [--cuda-only]

  (default)    Build the Python extension: build/mxdsa_ext.so
  --cuda-only  Build the standalone CUDA library: build/cuda/libmxdsa.so
  -h, --help   Show this help

Environment: CUDA_HOME, PYTHON (Python extension only), BUILD_DIR
EOF
}

mode=python
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "${1:-}" in
    "") ;;
    --cuda-only) mode=cuda ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [[ "$mode" == cuda ]]; then
    BUILD_DIR=${BUILD_DIR:-build/cuda}
    output=libmxdsa.so
else
    BUILD_DIR=${BUILD_DIR:-build}
    output=mxdsa_ext.so
fi
mkdir -p "$BUILD_DIR"

objects=()
link_flags=(-L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -lcuda)
if [[ "$mode" == python ]]; then
    PYTHON=${PYTHON:-python}
    TORCH_ROOT=$("$PYTHON" -c 'import torch; print(torch.__path__[0])')
    PYTHON_INCLUDE=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')
    includes=(-Icsrc -I"$CUDA_HOME/include" -I"$TORCH_ROOT/include"
        -I"$TORCH_ROOT/include/torch/csrc/api/include" -I"$PYTHON_INCLUDE")
    c++ "${includes[@]}" -O3 -std=c++20 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DTORCH_EXTENSION_NAME=mxdsa_ext -fPIC -c csrc/bindings/binding.cpp \
        -o "$BUILD_DIR/binding.o"
    objects+=("$BUILD_DIR/binding.o")
    link_flags=(-L"$TORCH_ROOT/lib" -Wl,-rpath,"$TORCH_ROOT/lib"
        -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 "${link_flags[@]}")
fi

flags=(-O3 -std=c++20 -DNDEBUG -D_USE_MATH_DEFINES -D_GLIBCXX_USE_CXX11_ABI=1
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --compiler-options=-fPIC
    -lineinfo --ptxas-options=-v,--register-usage-level=10,--warn-on-spills
    -gencode arch=compute_100f,code=sm_100f -gencode arch=compute_103f,code=sm_103f)
for source in attention/prefill attention/decode decode/get_decoding_sched_meta decode/combine; do
    unit=${source##*/}
    "$CUDA_HOME/bin/nvcc" -Icsrc "${flags[@]}" -c "csrc/$source.cu" -o "$BUILD_DIR/$unit.o"
    objects+=("$BUILD_DIR/$unit.o")
done
c++ -shared "${objects[@]}" "${link_flags[@]}" -o "$BUILD_DIR/$output"
printf 'Built %s\n' "$BUILD_DIR/$output"
