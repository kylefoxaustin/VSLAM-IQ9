# Findings — VSLAM on IQ-9075 (chronological)

All timings: EuRoC MH_01, 480×752 8-bit frames, A78 governor pinned to `performance` (2.36 GHz).

## 1. The three engines are real
- CPU: ORB-SLAM3 / VINS / OpenVINS all build & run; EuRoC staged.
- Hexagon v73: reachable via fastRPC **unsigned PD** (`/dev/fastrpc-cdsp`, `libcdsprpc.so`); custom skels load with **no OEM signing**.
- Adreno 663 ("FD663"): OpenCL 3.0 via **rusticl** (`RUSTICL_ENABLE=freedreno`), Vulkan via Turnip.

## 2. FastCV-in-OpenCV is NEON-CPU, not Hexagon
`cv2.fastcv.FAST10` = 2.27 ms / 3641 corners vs OpenCV CPU FAST 1.99 ms / 4554 — FastCV is *slower*, and there is no FastCV DSP skel on the board. The easy "FastCV = Hexagon" assumption is false here.

## 3. Adreno GPU (rusticl) loses to the CPU on small front-end kernels
3×3 blur: CPU `cv2.blur` 0.98 ms vs GPU kernel-only 1.60 ms / e2e-with-transfer 4.996 ms. Transfer dominates e2e; rusticl is unoptimized (~2 GFLOP/s effective — far below the Adreno ceiling). Characterizes the *available open stack*, not the silicon.

## 4. Feature extraction is ~55% of ORB-SLAM3 tracking
Tracking 23.9 ms/frame median (42 fps). ORB extraction (pyramid+FAST+orient+descriptor) ≈ 13.1 ms ≈ 55%; matching+pose-opt ≈ 45% (not offloadable). ORB-SLAM3 already clears 20 fps comfortably, so offload value = **headroom** (co-resident policy, more cameras, tail-clipping).

## 5. FAST-9-16 on the Hexagon cDSP
Custom fastRPC skel, unsigned PD, on-DSP `HAP_perf` timing, corner count gated against a CPU reference of the identical algorithm.
| implementation | corners | on-DSP time |
|---|---|---|
| scalar C (1 HW thread, no HVX) | 19059 | 76.4 ms |
| HVX (128-lane u8, vectorized run-length), 1 thread | 19059 | 1.60 ms |
| **HVX multi-thread (worker_pool across cDSP HW threads)** | **19059** (exact) | **0.555 ms** |
~137× over scalar; HVX multi-thread is **~3.6× faster than the A78 CPU (2 ms)** while **fully offloading it**. DSP pinned to TURBO_PLUS (HAP_power).  This is the first measured per-kernel HVX number for a future Cadence-DSP equivalency.

Method note (HVX): the "≥9 contiguous of 16" arc test is vectorized as a **run-length** over k=0..23 (wrapped) across 128 lanes — `run = brighter_k ? run+1 : 0`, track `maxrun`, corner if `maxrun ≥ 9` (brighter or darker). Right-edge remainder handled by a masked final HVX vector (a scalar tail cost 11 ms and erased the win).

## Next
- Multithread FAST across HVX contexts (dspqueue); add corner-coordinate scatter.
- Port pyramid → orientation → rBRIEF descriptor, each measured per-kernel.
- Compare per-kernel (µs + op-count) to a Cadence DSP.


## 6. Roofline confirmed on silicon: offload compute-bound kernels, not BW-bound
Second kernel (5x5 Gaussian blur) measured the same way, bit-correct (<=1 LSB vs reference):
| kernel (bound) | scalar | HVX 1-thread | HVX multi-thread | CPU |
|---|---|---|---|---|
| FAST (compute-bound) | 76 ms | 1.6 ms | **0.55 ms** (~3x from threads) | 2 ms |
| 5x5 blur (BW-bound) | 45 ms | 8.6 ms | 8.1 ms (threads barely help) | ~sub-ms (cv2 separable) |
- **Multithreading helps compute-bound (FAST ~3x) but NOT BW-bound (blur ~0)** — HVX threads share one DDR path; adding threads adds no bandwidth. Textbook roofline, measured.
- **Hexagon offload wins on FAST, loses on blur.** Not every front-end kernel belongs on the DSP. Build the offload plan (and the cross-DSP comparison) BY BOUND-CLASS: compute-bound (FAST/Harris/orient) -> Hexagon; BW-bound (pyramid/undistort) -> CPU or keep resident.
- HVX gotcha root-caused via micro-test: Q6_Vub_vasr_VuhVuhR_rnd_sat saturates large u16 regardless of shift; do >>8 in u16 domain first. Don't guess HVX widen/narrow semantics -- probe them.
