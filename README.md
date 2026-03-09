# Material Reorder — Vulkan Compute Benchmark

Benchmarks three GPU compute kernels that reorder 16×16 tiles of material IDs into wave-friendly layouts (uniform 32-element blocks front, overflow back).

| Kernel | Algorithm | Barriers | Shared Memory |
|---|---|---|---|
| Baseline | Bitonic sort + serial wave packing | 38 | ~4.1 KB |
| Optimized | Wave-intrinsic sort + prefix scan | 11 | ~2.1 KB |
| Set-Associative | Hash-table counting sort | 3 | ~5.1 KB |

## Prerequisites

- **Vulkan SDK** — [LunarG](https://vulkan.lunarg.com/sdk/home)
- **Slang compiler** (`slangc`) — [Slang releases](https://github.com/shader-slang/slang/releases)
- **CMake 3.20+**
- A Vulkan-capable GPU with subgroup operation support

Make sure `slangc` is on your `PATH`, or set the `SLANG_DIR` environment variable to the directory containing it. CMake also checks `%VULKAN_SDK%\bin` and `%USERPROFILE%\.slang\bin`.

## Build (Windows)

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Build (Linux)

```
mkdir build && cd build
cmake ..
cmake --build .
```

## Run

From the build directory:

```
# Windows
Release\benchmark.exe

# Linux
./benchmark
```

The `.spv` shader files are automatically compiled from `.slang` sources and copied next to the executable.

## Output

The benchmark runs each kernel with 10 warmup dispatches followed by 200 timed dispatches using GPU timestamp queries, verifies correctness against a CPU reference, and prints a summary table with mean/median/min/p5/p95 timings and speedup ratios.

## Files

- `reorder.slang` — Baseline bitonic sort kernel
- `reorder_optimized.slang` — Wave-intrinsic optimized kernel
- `reorder_setassoc.slang` — Hash-table counting sort kernel
- `benchmark.cpp` — Vulkan host application (tile generation, dispatch, verification, timing)
- `CMakeLists.txt` — Build configuration with slangc shader compilation
