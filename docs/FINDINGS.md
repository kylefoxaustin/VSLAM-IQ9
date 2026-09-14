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


## 7. ORB orientation: compute-bound (revises the roofline), modest HVX (dense vs sparse)
Third kernel: ORB IC_Angle (intensity centroid over r=15 patch + atan2), 1056 keypoints, on-DSP, gated bit-identical to scalar.
- **atan2 is only ~7% of the scalar kernel** (2841us full vs 2639us centroid-only) -> orient is CENTROID-COMPUTE-bound, NOT the "scalar/atan2 bottleneck" an earlier roofline predicted. atan2 is a ~200us scalar floor, not the wall.
- **HVX = 3.4x (836us vs 2833us)** via vrmpy-accumulate (Q6_Vw_vrmpyacc_VwVubVb). Modest vs FAST's 44x.

| orient implementation | on-DSP | speedup | vs scalar diff |
|---|---|---|---|
| scalar (1 HW thread) | 2833 us | 1x | ref |
| HVX, 1 thread | 836 us | 3.4x | 0 mrad (bit-identical) |
| **HVX multi-thread (worker_pool, keypoint bands)** | **329 us** | **8.6x** | **0 mrad (bit-identical)** |

- **Threading gave only 2.5x** (FAST got ~3-4x on the same silicon). Sparse per-keypoint work, the per-keypoint atan2 float tail, and worker_pool dispatch overhead cap thread scaling — a *quantified* property of the sparse-compute class, not a bug. Keypoint bands are embarrassingly parallel (disjoint ang[] slots, no locks) yet still don't scale like dense FAST.

### The offload taxonomy (3 kernels measured) is 3-way, not 2-way:
| kernel | class | HVX 1t | HVX-MT | thread scaling | vs A78 CPU |
|---|---|---|---|---|---|
| FAST | compute-bound, **dense per-pixel** | 1.6 ms | **0.55 ms** | ~3x | **beats** it, offloads |
| orient | compute-bound, **sparse per-keypoint** | 0.84 ms | **0.33 ms** | 2.5x | competitive |
| blur | **bandwidth-bound** | 8.6 ms | 8.1 ms | ~0 | loses |
**Dense-compute wins big on HVX and threads well; sparse-compute (small per-feature patches) wins modestly and threads *sub-linearly*; bandwidth-bound doesn't win at all and doesn't thread.** That taxonomy — not "offload the front end" — is how to plan a VSLAM offload and the currency for a cross-DSP (e.g. Cadence) equivalency. **Thread-scaling factor is itself a class signature** (dense > sparse > BW).


## 8. Harris corner response: dense wins big on HVX, but thread-scaling is kernel × IMPLEMENTATION
Fourth kernel: dense Harris (Sobel 3x3 -> Ixx/Iyy/Ixy -> 3x3 box-sum -> R=det-0.04*tr^2). Dense Shi-Tomasi/Harris is VINS-Mono's real front-end (goodFeaturesToTrack), so this is both realistic and the clean dense-compute test.
| Harris implementation | on-DSP | speedup |
|---|---|---|
| scalar (1 HW thread) | 45.3 ms | 1x |
| HVX, 1 thread | 4.90 ms | 9.25x |
| **HVX multi-thread (worker_pool, row bands)** | **1.97 ms** | **23.1x** |
- **Gated on the metric a corner-response kernel is FOR**, not raw float: top-500/2000/5000 corner overlap **100.00%**, all 83087 `|R|>1e9` corners match to max_rel **5.4e-5** (float-exact); nonzero-count identical. The larger raw-float diffs (~3e-2) are `det-k*tr^2` catastrophic cancellation in the near-zero *non-corner* noise floor — irrelevant to detection. (Verify the OUTPUT that's used, not the float in the noise.)
- **9.25x single-thread HVX confirms the dense-compute prediction** (big win, FAST-like).
- ⭐ **But HVX-MT threads only 2.49x — like sparse orient, NOT like dense FAST (3x).** De-confounded: HVX-vectorizing the serial zero did *not* change it, so the cause is that *this* implementation is **load-heavy** (108 unaligned u8-vector loads per 64 output pixels, because gradients are recomputed for all 9 box positions), pushing it toward bandwidth-bound. **Thread-scaling is kernel × IMPLEMENTATION arithmetic-intensity, not the abstract kernel class alone.**

**Prediction (published here) → tested → REFUTED, and that's the useful part.** I predicted an optimized Harris — each gradient computed **once** into scratch, then a box-sum reading scratch — would be more compute-bound and thread closer to 3x. Built it (modes 3/4, bit-corner-identical). Result:
| Harris | on-DSP | threading |
|---|---|---|
| HVX naive (9× gradient recompute), 1t | 4.87 ms | — |
| HVX-MT naive | 1.96 ms | 2.48× |
| **OPT single-pass, 1t** | **2.85 ms** (1.71× faster) | — |
| **OPT single-pass, MT** | **1.24 ms** (36.6× over scalar, best Harris) | **2.30×** |
The optimization made it **faster absolute** (fewer redundant computes — OPT-MT is the fastest Harris) but threaded **worse** (2.30× vs 2.48×) — the opposite of the prediction. **Mechanism:** computing gradients once needs a **scratch round-trip through the shared L2/DDR path**, and the cDSP HW threads *share* that path — so caching intermediates converts arithmetic into shared-memory traffic, which is exactly what caps HVX thread-scaling (same root as blur threading ≈0). **Thread-scaling tracks pressure on the SHARED MEMORY PATH, not the kernel's compute/memory ratio in isolation.** For a cross-DSP (Cadence) equivalency this is the sharper currency: compare by *measured arithmetic intensity and shared-memory pressure*, not by kernel name — and beware that a local-compute optimization can lower thread-scaling even as it lowers absolute latency.

## 9. rBRIEF descriptor: "gather-bound" prediction REFUTED — it's rounding/compute-bound
Fifth kernel: rotated 256-bit BRIEF descriptor (chained orient-angles -> rotated sampling), 992 keypoints, bit-exact to scalar (mean Hamming 0.000). Uses a deterministic *synthetic* 256-pair pattern (same shape as ORB's learned bit_pattern_31_, for perf + HVX-vs-scalar characterization; not OpenCV-bit-compatible). Scalar reference independently verified bit-exact vs numpy (Hamming 0 over all 992 kp).
| rBRIEF implementation | on-DSP | speedup |
|---|---|---|
| scalar (1 HW thread) | 166.9 ms | 1x |
| HVX, 1 thread | 6.72 ms | 24.8x |
| **HVX multi-thread (worker_pool, keypoint bands)** | **2.70 ms** | **61.8x** |
- ⭐ **Prediction refuted (the valuable part).** rBRIEF *looks* gather-bound — 512 scattered byte loads per keypoint — which predicts a small HVX win (sparse-class). **Measured, it is ROUNDING/COMPUTE-bound:** the scalar wall is 1024 software-float rounding ops per keypoint (the pattern rotations), which vectorize beautifully (24.8x — *larger* than orient's 3.4x or Harris's 9.25x); the scattered loads are cheap. **Classify a kernel by measuring which op is the scalar bottleneck, not by its scariest-looking operation.**
- HVX gotcha, probed not guessed: `Q6_Vw_equals_Vsf` truncates toward zero but scalar `lrintf` rounds → ~10% of offsets hit the wrong pixel (mean Hamming 26/256). Fix = sign-biased 0.5 in the float domain before the truncating convert → bit-exact.

### Taxonomy, 5 kernels measured:
| kernel | class | HVX 1t | HVX-MT | thread scaling | HVX vs scalar |
|---|---|---|---|---|---|
| FAST | compute-bound, **dense per-pixel** | 1.6 ms | **0.55 ms** | ~3x | 44x |
| Harris | compute-bound, **dense** | 4.90 ms (naive) / 2.85 ms (opt) | **1.24 ms** (opt-MT) | 2.3–2.5x | 9.25x |
| orient | compute-bound, **sparse per-keypoint** | 0.84 ms | **0.33 ms** | 2.5x | 3.4x |
| rBRIEF | **rounding/compute-bound** (not the gather!) | 6.72 ms | **2.70 ms** | 2.5x | 24.8x |
| blur | **bandwidth-bound** | 8.6 ms | 8.1 ms | ~0 | ~5x (loses to CPU) |

**The whole ORB front-end (FAST detect + Harris score + orientation + rBRIEF descriptor) now runs on the Hexagon cDSP via HVX, each kernel gated correct.** Two load-bearing conclusions for planning an offload and for a cross-DSP (e.g. Cadence) equivalency: (1) **measure the scalar bottleneck** — don't classify by the scariest op (rBRIEF's gather was a red herring); (2) **thread-scaling is a signature** — ~3x is the pure-vector dense-row ceiling; any per-keypoint banding or scalar tail caps it near 2.5x; a bandwidth-bound kernel doesn't thread at all. Compare cross-DSP by measured per-kernel µs + arithmetic intensity, not by kernel name.

HVX semantics banked while building this (probed, not guessed): u8->int **widening deinterleaves** (a 0..127 ramp returns stride-4); in-order widen = `Q6_Wuh_vzxt_Vub` then `Q6_W_vshuff_VVR(hi,lo,-2)`. The kernel dodges it — horizontal neighbors come from unaligned u8 loads (in-order), the square's even/odd int32 split *is* the pixel lo/hi partition, reunited with a 32-bit shuff at store.
