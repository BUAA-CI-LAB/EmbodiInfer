# Third-Party Notices

EmbodiInfer is licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)). It does not distribute model checkpoints or datasets.

## Vendored and adapted code

| File | Origin | License |
|---|---|---|
| `embodiinfer/policies/openvla_oft/vision_prismatic.py` | OpenVLA-OFT / Prismatic `modeling_prismatic.py` | Apache-2.0 |
| `embodiinfer/policies/gr00t/modules_gr00t.py`, `modeling_gr00t.py` | NVIDIA Isaac-GR00T DiT and embodiment-conditioned MLP | Apache-2.0 |
| `embodiinfer/models/video_vae/wan.py` | Cosmos-Policy / Wan2.1 VAE | Apache-2.0 |
| `embodiinfer/backend/cuda/csrc/marlin.cuh`, `backend/cuda/marlin.py` | IST-DASLab/Marlin; BF16/group64 and GeGLU adaptations | Apache-2.0 |
| `embodiinfer/models/video_vae/base.py` | Diffusers VAE conventions | Apache-2.0 |
| `embodiinfer/policies/lingbot_vla/modules_lingbot_vla.py` | LingBot-VLA action expert | Apache-2.0 (see upstream) |
| Pi05 RLinf numerical profile (`embeddings.py`, `inference/vision.py`, operator plans) | RLinf OpenPI/Gemma/SigLIP inference arithmetic | Apache-2.0 |

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
