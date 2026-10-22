# Third-Party Notices

EmbodiInfer is licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)). It does not distribute model checkpoints or datasets.

## Third-party code and project skills

| File | Origin | License |
|---|---|---|
| `embodiinfer/policies/openvla_oft/vision_prismatic.py` | OpenVLA-OFT / Prismatic `modeling_prismatic.py` | Apache-2.0 |
| `embodiinfer/policies/gr00t/modules_gr00t.py`, `modeling_gr00t.py` | NVIDIA Isaac-GR00T DiT and embodiment-conditioned MLP | Apache-2.0 |
| `embodiinfer/models/video_vae/wan.py` | Cosmos-Policy / Wan2.1 VAE | Apache-2.0 |
| `embodiinfer/backend/cuda/csrc/marlin.cuh`, `backend/cuda/marlin.py` | IST-DASLab/Marlin; BF16/group64 and GeGLU adaptations | Apache-2.0 |
| `embodiinfer/models/video_vae/base.py` | Diffusers VAE conventions | Apache-2.0 |
| `embodiinfer/policies/lingbot_vla/modules_lingbot_vla.py` | LingBot-VLA action expert | Apache-2.0 (see upstream) |
| Pi05 RLinf numerical profile (`embeddings.py`, `inference/vision.py`, operator plans) | RLinf OpenPI/Gemma/SigLIP inference arithmetic | Apache-2.0 |
| `skills/KernelWiki/` | [ZhouBencheng/KernelWiki](https://github.com/ZhouBencheng/KernelWiki), Thor/Orin adaptation of [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) | MIT; embedded sources retain their own notices |
| `skills/ncu-report-skill/` | [ZhouBencheng/ncu-report-skill](https://github.com/ZhouBencheng/ncu-report-skill), Thor/Orin adaptation of [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | MIT |

The skill submodules pin ZhouBencheng's `codex/thor-orin-adaptation` revisions:
KernelWiki `6d1b992bab075df5485359c2b75a49ba0e11717a` and ncu-report-skill
`eaccc5c641969689ef3ae36a3b97ae6aec13bd5c`. Their contents are unmodified; Git
records these exact commits independently of subsequent upstream branch changes.
Initialize them with `git submodule update --init --recursive` before tuning.
Neither skill is an EmbodiInfer runtime dependency. Both original upstream
projects added their MIT notices after these adaptation revisions, so the common
notice is reproduced below without changing the submodules. KernelWiki's embedded
code and cited material retain their original attribution and licenses.

The RLinf-derived numerical profile retains attribution to Copyright 2026 The
RLinf Authors, from `RLinf/RLinf` revision
`d34d4c320d08cb982de034aa9a011f08dc0fa217`. It adapts inference arithmetic and
precision/layout boundaries to EmbodiInfer's existing checkpoint modules and
operator registries. RL/SFT losses, training heads and trainer integrations are
not included. The adapted upstream sources are
`rlinf/models/embodiment/openpi/pi0.py`, `pi0_config.py`,
`modules/gemma.py`, `modules/siglip.py`, and `modules/model.py`.
The CUDA/Triton operator backends are distributed under this repository's
Apache-2.0 license.

The Marlin-derived implementation retains Copyright (C) Marlin.2024 Elias Frantar
(elias.frantar@ist.ac.at) and its original Apache-2.0 header. Weight packing
follows [IST-DASLab/Marlin](https://github.com/IST-DASLab/marlin); calibrated
codes are loaded from external artifacts and never regenerated at deployment.

## KernelWiki and ncu-report-skill MIT notice

```text
MIT License

Copyright (c) 2026 hanlab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime dependencies

Core: numpy (BSD-3-Clause), Pillow (MIT-CMU), torch (BSD-3-Clause).

Optional: lerobot (Apache-2.0), transformers (Apache-2.0), timm (Apache-2.0),
torchvision (BSD-3-Clause), safetensors (Apache-2.0), einops (MIT), websockets
(BSD-3-Clause), msgpack (Apache-2.0), h5py (BSD-3-Clause), PyYAML (MIT), jax
(Apache-2.0), orbax-checkpoint (Apache-2.0), sentencepiece (Apache-2.0), pytest
(MIT), ruff (MIT), wireless-comm (Apache-2.0).

## Models and datasets

Model checkpoints referenced by the engine (pi0.5 / LeRobot, GR00T N1.7,
OpenVLA-OFT, LingBot-VLA, Cosmos Policy, ActiveVLN, StreamVLN, Qwen2.5-VL,
NaVIDA) and datasets (LIBERO, R2R/RxR, Habitat scenes) keep their own licenses
and access conditions. Review each upstream license before redistributing
weights or data.
