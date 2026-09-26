## 最终矩阵（post-fix，RTX 5090）

| dataset | batch | EmbodiInfer sps | EmbodiInfer ms/obs | EmbodiInfer GiB | vLLM sps | vLLM ms/obs | vLLM GiB | 快者 |
|---|---|---|---|---|---|---|---|---|
| R2R | 1 | 17.01 | 58.8 | 11.4 | 11.55 | 86.6 | 26.2 | EmbodiInfer |
| R2R | 2 | 22.89 | 43.7 | 13.9 | 17.25 | 58.0 | 26.0 | EmbodiInfer |
| R2R | 4 | 26.60 | 37.6 | 15.4 | 22.33 | 44.8 | 25.7 | EmbodiInfer |
| R2R | 8 | 27.62 | 36.2 | 22.6 | 27.07 | 36.9 | 24.9 | EmbodiInfer |
| RXR | 1 | 10.81 | 92.5 | 16.2 | 7.19 | 139.1 | 25.9 | EmbodiInfer |
| RXR | 2 | 12.92 | 77.4 | 16.9 | 10.96 | 91.2 | 25.9 | EmbodiInfer |
| RXR | 4 | 13.11 | 76.2 | 20.8 | 13.44 | 74.4 | 25.6 | vLLM |
| RXR | 8 | OOM | — | — | 16.08 | 62.2 | 24.9 | vLLM |
