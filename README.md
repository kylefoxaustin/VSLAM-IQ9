# VSLAM-IQ9

**How does Visual SLAM run on the Qualcomm Dragonwing IQ-9075 (SA8775P) — and can its compute engines (A78 CPU, Hexagon v73 DSP, Adreno 663 GPU) accelerate the front end?**

This repo records an ongoing, measurement-first investigation into running Visual SLAM
(ORB-SLAM3 / VINS / OpenVINS) on the IQ-9075 and offloading the vision **front end**
(FAST corners, image pyramid, orientation, descriptors) onto the board's accelerators.

## Why
On edge robots/drones the SLAM front end (feature extraction) is a large slice of the
per-frame budget. If it can move off the CPU onto the DSP or GPU, the freed cores buy
headroom for planning/policy, more cameras, higher frame-rate, and tail-latency clipping.

## The board
- **CPU:** 8× Arm Cortex-A78C (the high-end edge SLAM CPU anchor)
- **DSP:** Hexagon v73 with HVX (two cDSP NSPs), reachable via fastRPC **unsigned PD** (no OEM signing)
- **GPU:** Adreno 663 (open Mesa stack: Turnip Vulkan + rusticl OpenCL)

## What we've measured so far (EuRoC MH_01, 480×752, clocks pinned)
| Result | Value |
|---|---|
| ORB-SLAM3 tracking (A78) | 23.9 ms/frame median (42 fps) |
| ORB feature **extraction** share | **~55%** of tracking (13.1 ms) → offload is worth it |
| OpenCV "FastCV" module | NEON-**CPU**, not Hexagon (no DSP offload on this image) |
| Adreno GPU (rusticl) 3×3 blur | slower than CPU (transfer-bound; unoptimized OpenCL stack) |
| **FAST-9-16 on Hexagon — scalar C** | 76.4 ms on-DSP (1 thread, no HVX) |
| **FAST-9-16 on Hexagon — HVX, 1 thread** | 1.60 ms on-DSP |
| **FAST-9-16 on Hexagon — HVX multi-thread** | **0.555 ms on-DSP** (~137× over scalar; ~3.6× faster than the A78 CPU, offloads it) — bit-exact |
| **5x5 Gaussian blur on Hexagon — HVX** | 8.1 ms (BW-bound: MT doesn't help, LOSES to CPU) |
| **ORB orientation on Hexagon — HVX, 1 thread** | 0.84 ms, ~3.4x (compute-bound but sparse per-keypoint; atan2 is a ~200us floor — this *revises* the earlier 'scalar-bottleneck' roofline) |
| **ORB orientation on Hexagon — HVX multi-thread** | **0.329 ms on-DSP** (8.6× over scalar) — bit-identical; but threading only 2.5× (sparse-compute threads sub-linearly vs dense FAST's ~3×) |
| **Harris corner response on Hexagon — HVX** | 4.90 ms, 9.25× over scalar — corner-ranking identical (top-2000 overlap 100%) |
| **Harris corner response on Hexagon — HVX multi-thread** | **1.97 ms on-DSP** (23.1× over scalar) — threading 2.5×; a load-heavy implementation threads sub-linearly even though the kernel is dense-compute |

See [docs/FINDINGS.md](docs/FINDINGS.md) for the full trail and method.

## Layout
- `src/hexagon/` — our FAST-9-16 kernel for the Hexagon cDSP (scalar + HVX). Drops into a
  Hexagon SDK fastRPC skel; the SDK and its example scaffold are proprietary and **not** included.
- `src/opencl/` — Adreno GPU micro-benchmarks driven via pure `ctypes` → `libOpenCL` (rusticl),
  no compiler/pyopencl needed on the sealed board.
- `src/profiling/` — ORB-SLAM3 front-end vs total profiling.
- `docs/` — findings and method notes.

## What is NOT here (and why)
- **Hexagon SDK** and its example sources — proprietary/licensed (obtain from Qualcomm).
- **EuRoC dataset**, model weights — fetch from their sources.

## Long-term goal
Build a clean, per-kernel (µs + op-count) profile of the SLAM front end on Hexagon v73,
to support an equivalency comparison against other DSPs (e.g. Cadence Tensilica).

## Status
Active / early. Foundation proven: custom code compiles for v73, runs on the cDSP via
unsigned PD, and the first HVX kernel (FAST) is verified and fast. Next: multithread + more kernels.
