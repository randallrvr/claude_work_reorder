# Material Reorder — Vulkan Compute Benchmark

Benchmarks five GPU compute kernels that reorder 16×16 tiles of material IDs into wave-friendly layouts (uniform 32-element blocks front, overflow back).

| Kernel | Algorithm | Barriers | Shared Memory |
|---|---|---|---|
| Baseline | Bitonic sort + serial wave packing | 38 | ~4.1 KB |
| Optimized | Wave-intrinsic sort + prefix scan | 11 | ~2.1 KB |
| Set-Associative | Hash-table counting sort | 3 | ~5.1 KB |
| Set-Associative v2 | Wave-cooperative hash + parallel prefix sum | 3 | ~516 B |
| Set-Associative v3 | Hash-free wave-merge + deterministic scatter | 2 | ~2.6 KB |
| Set-Associative v4 | Parallel merge + precomputed wave bases | 2 | ~3.6 KB |

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

## RenderDoc Capture

If [RenderDoc](https://renderdoc.org/) is installed, CMake will automatically detect it and enable capture support. Use the `--capture` flag to save a RenderDoc frame capture for each kernel:

```
# Capture all kernels
Release\benchmark.exe --capture

# Capture only kernels matching a name substring
Release\benchmark.exe --capture "v2"
```

The capture runs a single extra dispatch per kernel after the timed benchmark, so it doesn't affect timing results. Captures are saved to RenderDoc's default capture directory. You can also launch the benchmark from within RenderDoc's GUI for interactive inspection.

If RenderDoc is not installed, `--capture` prints an error and exits. The benchmark itself always works without RenderDoc.

To point CMake to a non-standard RenderDoc install location:

```
cmake .. -DRENDERDOC_DIR="C:/path/to/RenderDoc"
```

## Files

- `reorder.slang` — Baseline bitonic sort kernel
- `reorder_optimized.slang` — Wave-intrinsic optimized kernel
- `reorder_setassoc.slang` — Hash-table counting sort kernel
- `reorder_setassoc_v2.slang` — Optimized set-associative kernel (wave-cooperative, parallel prefix sum)
- `reorder_setassoc_v3.slang` — Hash-table-free wave-merge kernel (no atomics, 2 barriers)
- `reorder_setassoc_v4.slang` — Parallel merge + precomputed wave bases (2 barriers)
- `benchmark.cpp` — Vulkan host application (tile generation, dispatch, verification, timing, RenderDoc capture)
- `CMakeLists.txt` — Build configuration with slangc shader compilation and optional RenderDoc detection
