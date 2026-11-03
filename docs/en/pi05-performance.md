# PI0.5 on Jetson AGX Thor and RTX 4090

EmbodiInfer's PI0.5 presets against FlashRT, APXinf, Embodied.cpp and the OpenPI reference on two
devices, with one benchmark script and one set of settings for every engine.

![EmbodiInfer speedup on PI0.5 over FlashRT, APXinf, Embodied.cpp and OpenPI on Jetson AGX Thor and RTX 4090](assets/pi05/speedup.svg)

| Device | Precision | EmbodiInfer | Speedup |
|---|---|---:|---|
| Jetson AGX Thor | FP4 | 20.04 ms · 49.9 Hz | 1.05× over FlashRT |
| Jetson AGX Thor | FP8 | 33.60 ms · 29.8 Hz | 1.21× over FlashRT, 1.22× over APXinf |
| Jetson AGX Thor | 16-bit | 61.63 ms · 16.2 Hz | 1.09× over APXinf, 1.12× over FlashRT, 3.63× over Embodied.cpp, 4.24× over OpenPI |
| RTX 4090 | 8-bit | 17.06 ms · 58.6 Hz | 1.29× over FlashRT, 2.38× over APXinf |
| RTX 4090 | 16-bit | 30.08 ms · 33.2 Hz | 1.06× over FlashRT, 1.07× over APXinf, 1.91× over Embodied.cpp |

With OpenPI's own checkpoint and LIBERO evaluation on the RTX 4090, 2 denoising steps on a ρ = 3
schedule take the `fp8-fp8` request to 13.34 ms (75.0 Hz) and solve 99.0% of LIBERO-spatial, where
OpenPI solves 98.8%; see
[LIBERO on RTX 4090](#libero-on-rtx-4090-with-openpis-checkpoint-and-evaluation).

## Method

Every engine runs FlashRT's own PI0.5 benchmark script, `examples/thor/pi05.py` from FlashRT v0.2.0.
Each engine replaces only model loading and the per-call `run()`; the inputs, prompt, warmup, seeded
action pass, timed loop and report are FlashRT's.

| Setting | Value |
|---|---|
| Weights | `RLinf-Pi05-LIBERO-SFT`, the same file for every engine; actions are unnormalized with its quantile `norm_stats.json` |
| Inputs | The 8 LIBERO observations in FlashRT's `pi05_libero.npz`, used in turn: base and wrist cameras, 224×224 uint8 |
| Prompt | "pick up the black bowl and place it on the plate", 13 PaliGemma tokens |
| Model | Batch 1, 2 cameras, 10 flow-matching steps, 10-step action chunk |
| Timing | 200 warmup calls, then 1,000 timed calls (`synchronize → run → synchronize`); P50 per run |
| Runs | The engines of one device interleaved, 3 rounds; the tables give each round's P50 and their mean |

Each timed call covers the image upload, image normalization, noise, the whole request, the action
copy to the host and unnormalization. Weight loading, calibration, prompt binding and graph capture
happen before timing. The actions of the 8 seeded calls use FlashRT's float16 NumPy noise for every
engine, so they can be compared across engines.

## Jetson AGX Thor

Jetson AGX Thor (SM110, 20 SMs, 128 GB LPDDR5X) with L4T R39.2.1, CUDA 13.2, PyTorch 2.13.0 and
cuDNN 9.20, in MAXN with the GPU, memory and CPU clocks locked at 1575 MHz, 4266 MHz and 2.6 GHz.
Each run checks that the GPU is idle and the clocks are locked before it starts.

| Precision | Engine · configuration | Round 1 | Round 2 | Round 3 | Mean P50 (ms) | Max P95 (ms) | Rate | Action cosine vs OpenPI (mean / worst) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FP4 | **EmbodiInfer · `nvfp4-nvfp4`** | 19.938 | 20.157 | 20.033 | **20.04** | 20.34 | **49.9 Hz** | 0.997182 / 0.995833 |
| FP4 | FlashRT · `fp4` (NVFP4 + FA4) | 21.070 | 21.122 | 21.031 | 21.07 | 21.20 | 47.5 Hz | 0.998025 / 0.996257 |
| FP8 | **EmbodiInfer · `fp8-fp8`** | 33.386 | 33.529 | 33.894 | **33.60** | 34.25 | **29.8 Hz** | 0.999785 / 0.999716 |
| FP8 | FlashRT · `fp8` | 40.687 | 40.781 | 40.630 | 40.70 | 40.92 | 24.6 Hz | 0.999773 / 0.999602 |
| FP8 | APXinf · `fp8_static` | 40.875 | 40.980 | 40.796 | 40.88 | 41.06 | 24.5 Hz | 0.997992 / 0.997391 |
| 16-bit | **EmbodiInfer · `bf16`** | 61.360 | 61.867 | 61.655 | **61.63** | 62.06 | **16.2 Hz** | 0.999995 / 0.999993 |
| 16-bit | APXinf · `bf16` | 66.997 | 67.188 | 67.179 | 67.12 | 67.39 | 14.9 Hz | 0.999982 / 0.999977 |
| 16-bit | FlashRT · `fp16` | 69.220 | 69.244 | 69.244 | 69.24 | 69.46 | 14.4 Hz | 0.999891 / 0.999791 |
| 16-bit | Embodied.cpp · BF16 GGUF | 223.895 | 224.080 | 223.939 | 223.97 | 224.50 | 4.5 Hz | 0.999992 / 0.999981 |
| 16-bit | OpenPI · PyTorch BF16 | 258.443 | 265.667 | 259.420 | 261.18 | 271.41 | 3.8 Hz | reference |

The action cosines compare the 8 seeded action chunks of the first round with OpenPI's. Every
engine returns bit-identical actions in all three rounds. The two FP4 engines were measured in a
separate interleaved run of 3 rounds with the same settings.

## RTX 4090

GeForce RTX 4090 (SM89, 128 SMs, 24 GB) with CUDA 12.9, PyTorch 2.13.0 and cuDNN 9.20, at default
clocks (SM at about 2.79 GHz during the runs). SM89 has no FP4 tensor cores, so there is no FP4 tier.

| Precision | Engine · configuration | Round 1 | Round 2 | Round 3 | Mean P50 (ms) | Max P95 (ms) | Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 8-bit | **EmbodiInfer · `fp8-fp8`** | 17.055 | 17.066 | 17.063 | **17.06** | 17.09 | **58.6 Hz** |
| 8-bit | FlashRT · `fp8` | 22.012 | 22.061 | 22.102 | 22.06 | 22.15 | 45.3 Hz |
| 8-bit | APXinf · `int8_dynamic` | 40.612 | 40.615 | 40.643 | 40.62 | 40.85 | 24.6 Hz |
| 16-bit | **EmbodiInfer · `bf16`** | 30.060 | 30.079 | 30.095 | **30.08** | 30.53 | **33.2 Hz** |
| 16-bit | FlashRT · `fp16` | 31.913 | 31.937 | 31.946 | 31.93 | 32.25 | 31.3 Hz |
| 16-bit | FlashRT · `bf16` | 32.881 | 33.314 | 32.933 | 33.04 | 33.42 | 30.3 Hz |
| 16-bit | APXinf · `bf16` | 32.033 | 32.051 | 32.085 | 32.06 | 32.30 | 31.2 Hz |
| 16-bit | Embodied.cpp · BF16 GGUF | 57.452 | 57.479 | 57.500 | 57.48 | 57.68 | 17.4 Hz |

On the RLinf LIBERO validation fixture (10 observations × 5 noise seeds), the 4090 `bf16` and
`fp8-fp8` presets match the strict BF16 preset at action cosine 0.999997 and 0.99996.

## Engines

| Engine | Version | Configuration |
|---|---|---|
| EmbodiInfer | this repository | `Pi05Policy.static_request` with the `thor` or `4090` preset named in the tables; one CUDA graph from uint8 frames to actions |
| FlashRT | v0.2.0 (`f4ddc90`) | Thor: `fp4` is the NVFP4 + FlashAttention-4 configuration of its `docs/thor/pi05.md`, `fp8` its default, `fp16` sets `use_fp16=True, use_fp8=False`. RTX 4090: the `rtx_sm89` route with `fp8`, `bf16` (`use_fp8=False`) and `fp16` (`use_fp16=True`). Calibrated on the same 8 observations |
| APXinf | `4de8f5c` | `ModelRunner.infer_rgb` with the prompt tokenized once and `autotune` on, as in its benchmark commands, since its tactic tables do not cover this prompt length. Thor `fp8_static` is calibrated by its `calibrate_pi05.py` on the same 8 observations; static FP8 needs SM100 or newer, so the 4090 uses its `int8_dynamic` mode |
| Embodied.cpp | `1dad33f2`, llama.cpp `b9016` | BF16 GGUF converted from the same safetensors with a 10-step action chunk, ggml CUDA backend, called in process; its raw actions go through the same unnormalization |
| OpenPI | `pi05_libero` | The reference mode of FlashRT's script: OpenPI's own policy in eager PyTorch. Measured on Thor |

APXinf picks its GEMM kernels for this prompt length when it loads. At this shape one fresh tuning
chose an FP8 GateUp kernel that returns wrong actions, so the APXinf rows reuse a tuning whose
actions match the reference.

## LIBERO closed loop on Thor

LIBERO-spatial task 0 ("pick up the black bowl between the plate and the ramekin and place it on the
plate") from the same initial state, with a new decision at every control step and a fixed
flow-matching sample per engine. The simulator prepares the frames once with OpenPI's Pillow
`resize_with_pad`, and all three engines receive the same frames. The video plays at 2× speed with
its clock in real seconds; each step lasts its policy latency plus the 50 ms control period.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="../assets/pi05/libero-thor-comparison.mp4" type="video/mp4">
</video>

| Engine | Decisions | Policy latency P50 (ms) | Rate | Task completed | Time to complete |
|---|---:|---:|---:|---|---:|
| OpenPI · PyTorch BF16 | 74 | 265.52 | 3.8 Hz | yes | 23.68 s |
| FlashRT · NVFP4 | 71 | 22.79 | 43.9 Hz | yes | 5.17 s |
| **EmbodiInfer · `nvfp4-nvfp4`** | 68 | 20.00 | 50.0 Hz | yes | 4.76 s |

The policy latency is the policy server's model call, from the frames in shared memory to the
unnormalized actions. The time to complete adds each step's policy latency and the 50 ms control
period.

### NVFP4 calibration on Thor

The bundled `nvfp4-nvfp4` calibration of `RLinf-Pi05-LIBERO-SFT` keeps every MLP projection of both
towers in NVFP4 and changes two things relative to round-to-nearest packing of the checkpoint
weights:

- **[SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) gate/up.** Every MLP packs its gate/up weights as W·s, with
  s<sub>j</sub> = (a<sub>j</sub> / mean(a))<sup>0.5</sup> clamped to [0.25, 4], where a<sub>j</sub>
  is input channel j's range over 40 recorded LIBERO-spatial frames (tasks 0–3, initial state 10).
  The preceding norm divides its output by s, through the RMSNorm scale in the prefix and the AdaRMS
  projection in the action expert. Only the packed weights change; the kernels do not.
- **BF16 outlier channels.** The 128 down-projection input channels of prefix layer 16 with the
  largest ranges (up to 14.7k, against a mean of 28) run in BF16 beside the NVFP4 path, whose
  remaining channels then fit a range of 213 instead of 14.1k. These channels carry about three
  times the norm of the rest of that layer's output, which becomes the keys and values of the last
  prefix layer. The side path adds about 0.04 ms.

Over initial states 0–29 of the same task, one episode each with the same frames, compared with
round-to-nearest packing and with FlashRT:

| Engine · configuration | Successes | Bowl on the plate 2 s later | Bowl-to-plate distance |
|---|---:|---:|---:|
| EmbodiInfer · `nvfp4-nvfp4`, round-to-nearest packing | 18/30 | 10/30 | 2.45 cm |
| **EmbodiInfer · `nvfp4-nvfp4`** | **25/30** | **21/30** | **1.85 cm** |
| FlashRT · NVFP4 | 26/30 | 24/30 | 1.91 cm |

A success is LIBERO's own check within 220 policy calls. The bowl is checked again after 40 more
control steps, and the distance between the bowl and plate centres is taken when LIBERO first
reports success, averaged over successful episodes. Thirty episodes resolve differences of about
two successes, so EmbodiInfer and FlashRT are not distinguishable here.

## LIBERO on RTX 4090 with OpenPI's checkpoint and evaluation

These runs use OpenPI's own `pi05_libero` checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`,
converted to PyTorch by OpenPI's `examples/convert_jax_model_to_pytorch.py`) and OpenPI's LIBERO
evaluation `examples/libero/main.py` unchanged: LIBERO-spatial, 10 tasks × 50 initial states,
seed 7, 10 settle steps, then the first 5 actions of every 10-action chunk at the 20 Hz control
rate. OpenPI is served by its own PyTorch policy, EmbodiInfer by
[`libero/serve.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/serve.py)
over the same websocket protocol; both draw fresh flow-matching noise for every call.

| Engine · configuration | Denoise steps | Successes per task | Success rate |
|---|---|---|---:|
| OpenPI · PyTorch BF16 | 10 | 50 50 50 50 47 48 50 50 49 50 | 494/500 · 98.8% |
| **EmbodiInfer · `fp8-fp8`** | **2, ρ = 3** | 50 50 50 50 47 48 50 50 50 50 | **495/500 · 99.0%** |
| EmbodiInfer · `fp8-fp8` | 3, ρ = 1 | 50 50 50 50 48 47 50 50 50 50 | 495/500 · 99.0% |
| EmbodiInfer · `fp8-fp8` | 2, ρ = 1 | 50 50 50 49 47 46 50 50 50 49 | 491/500 · 98.2% |
| EmbodiInfer · `fp8-fp8` | 3, ρ = 2 | 50 50 50 49 47 46 50 50 50 48 | 490/500 · 98.0% |
| EmbodiInfer · `fp8-fp8` | 4, ρ = 1 | 50 50 50 50 47 44 48 50 50 50 | 489/500 · 97.8% |

The fewer-step settings reuse the trained weights. `Pi05Policy.flow_schedule_rho` places the N Euler
steps on the power grid t<sub>i</sub> = (1 − i/N)<sup>ρ</sup>; ρ = 1 is OpenPI's uniform grid, and
ρ > 1 takes large steps near the noise end (t = 1) and small ones near the data end (t = 0). With
2 steps and ρ = 3 the velocity is evaluated at t = 1 and t = 0.125. Over 500 episodes one standard
error of a success rate near 98% is about 0.6 points, so the settings are not distinguishable from
each other or from OpenPI; every one stays within one point of OpenPI's 10-step result.

| EmbodiInfer `fp8-fp8`, RTX 4090 | Latency P50 | Latency P95 | Rate | Speedup |
|---|---:|---:|---:|---:|
| 10 steps | 17.08 ms | 17.19 ms | 58.5 Hz | 1.00× |
| 4 steps, ρ = 1 | 14.21 ms | 14.27 ms | 70.4 Hz | 1.20× |
| 3 steps, ρ = 1 | 13.79 ms | 13.84 ms | 72.5 Hz | 1.24× |
| 3 steps, ρ = 2 | 13.80 ms | 13.86 ms | 72.5 Hz | 1.24× |
| 2 steps, ρ = 1 | 13.33 ms | 13.37 ms | 75.0 Hz | 1.28× |
| **2 steps, ρ = 3** | **13.34 ms** | **13.40 ms** | **75.0 Hz** | **1.28×** |

[`libero/latency.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/latency.py)
times 1,000 calls after 50 warmup calls on an idle GPU, with LIBERO task 0's 20-token prompt and
recorded LIBERO frames: each call copies the frames, draws noise, runs the static request and
unnormalizes the actions, as the server does. The schedule's ρ does not change the latency.

The bundled `fp8-fp8` calibration belongs to `RLinf-Pi05-LIBERO-SFT`, whose SHA256 it is bound to,
so this checkpoint has its own:
[`libero/calibrate.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/calibrate.py)
records every MLP layer's input range on 80 LIBERO observations, the settled first frames of 2
initial states of each of the 40 LIBERO-spatial, -object, -goal and -10 tasks, with 10% headroom.
The down projections of prefix layers 16 and 17 see inputs up to 13k and 20k (1.2k and 9.7k in
the RLinf checkpoint) and stay BF16; everything else in both towers runs FP8.

The film shows one episode for every engine at 10 denoising steps, task 6 ("pick up the black bowl
next to the cookie box and place it on the plate") from initial state 2, each engine alone on the GPU
and played in real time: every control step lasts 50 ms, and a step that requests a new chunk also
lasts that call's inference time. Among 16 candidate episodes (tasks 1, 3, 5 and 6, initial states
0–3) that every engine solved with a single grasp, this one has the closest step counts (96–97), so
the engines differ in speed rather than in behavior. FlashRT runs its `rtx_sm89` FP8 route through
`predict()`, which calibrates on the first observation of each prompt.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="../assets/pi05/libero-4090-comparison.mp4" type="video/mp4">
</video>

| Engine · configuration | Inference latency P50 | Rate | Control steps to success | Time to complete |
|---|---:|---:|---:|---:|
| OpenPI · PyTorch BF16 | 202.0 ms | 4.9 Hz | 97 | 8.89 s |
| FlashRT · FP8 | 23.6 ms | 42.3 Hz | 97 | 5.32 s |
| **EmbodiInfer · `fp8-fp8`** | **18.6 ms** | **53.8 Hz** | 96 | **5.17 s** |

The inference latency is the server's model call inside the closed loop, while the simulator renders
on the same GPU, about 1 ms above the idle-GPU figures. The time to complete adds the 50 ms control
period of every step and the inference time of the steps that requested a chunk.
