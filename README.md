# Parallelizing the Critical Path: OpenMP in Latency-Sensitive Systems

Companion proof-of-concept for the article *Parallelizing the Critical Path: OpenMP in Latency-Sensitive Systems*. Each executable demonstrates a core concept from the article using a realistic trading-system scenario — computing technical indicators across a universe of instruments.

## Prerequisites

- **C++17 compiler** with OpenMP support (GCC 9+ or Clang 11+)
- **CMake** 3.14+

On Ubuntu/Debian:

```bash
sudo apt install build-essential cmake
```

## Build

**With Make** (simplest):

```bash
make
```

**With CMake** (if installed):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Examples

All binaries are placed in `build/`.

### `benchmark`

The headline benchmark from the article. Computes volatility across 1000 instruments (1000 ticks each), reporting median serial vs parallel latency with warmup runs.

```bash
./build/benchmark
```

### `parallel_for_ema`

Demonstrates the basic `#pragma omp parallel for` pattern — computing fast and slow EMA independently across 500 instruments.

```bash
./build/parallel_for_ema
```

### `race_condition_demo`

Walks through the progression from the article:

1. **Data race** — racy `signal_count++` produces inconsistent results across trials
2. **Critical section** — correct but serialized
3. **Atomic** — correct with lower overhead
4. **Reduction** — correct with no contention (the right answer)

```bash
./build/race_condition_demo
```

### `thread_local_storage`

Shows per-thread scratch buffers declared inside a `#pragma omp parallel` block (automatically private), plus the explicit `private()` clause for variables declared outside the region.

```bash
./build/thread_local_storage
```

### `scheduling_strategies`

Compares `static`, `dynamic`, and `guided` scheduling on a heterogeneous instrument universe where tick counts range from 200 to 5000 per symbol — demonstrating why static scheduling leaves performance on the table when per-iteration work varies.

```bash
./build/scheduling_strategies
```

## Tuning

Control thread count and placement via environment variables:

```bash
# Use exactly 4 threads
OMP_NUM_THREADS=4 ./build/benchmark

# Pin threads to cores (important for latency)
OMP_PROC_BIND=true OMP_PLACES=cores ./build/benchmark
```

## Project Structure

```
├── CMakeLists.txt
├── Makefile
├── README.md
├── include/
│   └── market_data.h          # Instrument types, EMA/volatility helpers, universe generators
└── src/
    ├── benchmark.cpp           # Main volatility benchmark
    ├── parallel_for_ema.cpp    # Basic parallel for with EMA
    ├── race_condition_demo.cpp # Race -> critical -> atomic -> reduction
    ├── thread_local_storage.cpp# Per-thread scratch buffers
    └── scheduling_strategies.cpp # static vs dynamic vs guided
```

## License

MIT
