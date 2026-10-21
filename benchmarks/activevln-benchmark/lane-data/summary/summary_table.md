# ActiveVLN inference benchmark: native vLLM vs EmbodiInfer

Selection: first 8 annotations per dataset (sorted by id), first 12 recorded frames each = 96 requests per repeat; 3 repeats; greedy (temperature=0, top_p=1, repetition_penalty=1.05, max_new_tokens=512, EOS stop); RTX 5090 32GB.

| dataset | batch | vLLM samples/s | EmbodiInfer samples/s | ratio | vLLM lat mean/p50/p95 ms | EmbodiInfer lat mean/p50/p95 ms | vLLM tok/s | EmbodiInfer tok/s | vLLM GPU MiB | EmbodiInfer GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| r2r | 1 | 6.68 ± 0.02 | 2.86 ± 0.01 | 2.33x | 132/135/143 | 346/354/359 | 137 | 58 | 30150 | 11158 |
| r2r | 2 | 11.17 ± 0.37 | 5.40 ± 0.01 | 2.07x | 161/164/178 | 364/365/371 | 228 | 110 | 30150 | 11684 |
| r2r | 4 | 21.91 ± 0.10 | 9.19 ± 0.01 | 2.38x | 161/164/177 | 423/420/445 | 448 | 188 | 30150 | 12738 |
| r2r | 8 | 39.88 ± 0.42 | 13.31 ± 0.03 | 3.00x | 170/172/196 | 576/557/695 | 815 | 272 | 30188 | 14830 |
| rxr | 1 | 6.12 ± 0.18 | 2.70 ± 0.00 | 2.26x | 145/144/159 | 366/360/397 | 132 | 58 | 30246 | 12860 |
| rxr | 2 | 10.92 ± 0.03 | 4.95 ± 0.01 | 2.21x | 164/164/177 | 398/391/460 | 235 | 106 | 30246 | 15192 |
| rxr | 4 | 20.52 ± 0.25 | 7.69 ± 0.01 | 2.67x | 173/173/190 | 507/466/665 | 441 | 165 | 30246 | 20376 |
| rxr | 8 | 37.07 ± 0.97 | 8.62 ± 0.01 | 4.30x | 189/188/219 | 902/888/1220 | 797 | 185 | 30248 | 30732 |
