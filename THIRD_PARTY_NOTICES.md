# Third-Party Notices

EmbodiInfer is licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)). It does not distribute model checkpoints or datasets.

## Vendored and adapted code

| File | Origin | License |
|---|---|---|
| `embodiinfer/policies/openvla_oft/vision_prismatic.py` | OpenVLA-OFT / Prismatic `modeling_prismatic.py` | Apache-2.0 |
| `embodiinfer/policies/gr00t/modules_gr00t.py`, `modeling_gr00t.py` | NVIDIA Isaac-GR00T DiT and embodiment-conditioned MLP | Apache-2.0 |
| `embodiinfer/models/video_vae/wan.py` | Cosmos-Policy / Wan2.1 VAE | Apache-2.0 |
| `embodiinfer/models/video_vae/base.py` | Diffusers VAE conventions | Apache-2.0 |
| `embodiinfer/policies/lingbot_vla/modules_lingbot_vla.py` | LingBot-VLA action expert | Apache-2.0 (see upstream) |

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
