"""Derived RLinf SigLIP execution over the existing checkpoint parameters."""

import ctypes
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ....backend.cuda.native import build_library
from ....backend.torch.nvfp4 import blocked_scales, quantization_thresholds, quantize_weight


class SiglipPlan:
    """Own BF16 casts and packed MHA weights without changing checkpoint modules.

    Rebuild with the policy controller after refit or device migration. The FP32
    stem and positions, BF16 encoder/projector, exact GELU and native MHA calls
    preserve the RLinf vision precision contract.
    """

    @torch.no_grad()
    def __init__(
        self,
        paligemma: Any,
        *,
        fast_sdpa: bool = False,
        fp8: bool = False,
        nvfp4: bool = False,
        prefetcher: Any = None,
    ) -> None:
        """Prepare immutable weight views/casts before graph capture."""
        self.prefetcher = None
        # cuDNN fused attention runs SigLIP's 72-wide heads directly, where FlashAttention pads them
        # to 96; FlashAttention remains the fallback.
        self.attention_backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]
        vision = paligemma.vision_tower.vision_model
        patch = vision.embeddings.patch_embedding
        self.fast_sdpa = fast_sdpa and patch.weight.device.type == "cuda"
        self.nvfp4 = (
            self.fast_sdpa and nvfp4 and torch.cuda.get_device_capability(patch.weight.device)[0] >= 10
        )
        self.fp8 = self.fast_sdpa and fp8 and hasattr(torch, "_scaled_mm") and not self.nvfp4
        self.patch_weight = patch.weight.detach().float()
        self.patch_bias = None if patch.bias is None else patch.bias.detach().float()
        self.stride, self.padding = patch.stride, patch.padding
        self.positions = vision.embeddings.position_embedding.weight.detach().float()[None]
        if self.fast_sdpa:
            self.patch_weight_bf16 = (
                patch.weight.detach().bfloat16().contiguous(memory_format=torch.channels_last)
            )
            self.patch_weight_lin_t = patch.weight.detach().bfloat16().reshape(1152, -1).t().contiguous()
            self.patch_bias_bf16 = None if patch.bias is None else patch.bias.detach().bfloat16()
            self.positions_bf16 = self.positions.bfloat16()
            self.patch_bias_pos_bf16 = (
                self.positions_bf16
                if self.patch_bias_bf16 is None
                else (self.positions_bf16 + self.patch_bias_bf16.view(1, 1, -1)).contiguous()
            )
            self._bias_positions_by_batch: dict[int, torch.Tensor] = {}
            with torch.cuda.device(patch.weight.device):
                self.nq_lib = build_library(
                    "norm_quant", strict=True, specific=True, native_fp4=False
                ).library
                self.q_lib = build_library("quantize", strict=False, specific=True, native_fp4=False).library
                self.fus_lib = build_library("fusion", strict=False, specific=True, native_fp4=False).library
            ptr = ctypes.c_void_p
            self.nq_lib.cc_layer_norm_fp8.argtypes = [ptr] * 6 + [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
            self.nq_lib.cc_layer_norm_fp8.restype = ctypes.c_int
            self.nq_lib.cc_layer_norm_fp4.argtypes = [ptr] * 7 + [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
            self.nq_lib.cc_layer_norm_fp4.restype = ctypes.c_int
            self.nq_lib.cc_bhsd_fp8.argtypes = [
                ptr,
                ptr,
                ctypes.c_float,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
            self.nq_lib.cc_bhsd_fp8.restype = ctypes.c_int
            self.nq_lib.cc_bhsd_fp4.argtypes = [
                ptr,
                ptr,
                ptr,
                ctypes.c_float,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
            self.nq_lib.cc_bhsd_fp4.restype = ctypes.c_int
            self.q_lib.cc_fp8.argtypes = [ptr, ptr, ptr, ctypes.c_int64, ptr]
            self.q_lib.cc_fp8.restype = ctypes.c_int
            self.fus_lib.cc_geglu.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int, ptr, ptr]
            self.fus_lib.cc_geglu.restype = ctypes.c_int
            codes = torch.arange(65536, device=patch.weight.device, dtype=torch.int32).to(torch.uint16)
            self.gelu_table = F.gelu(codes.view(torch.bfloat16), approximate="tanh").view(torch.uint16)
            thr = quantization_thresholds(patch.weight) if self.nvfp4 else None
            self.fast_layers = []
            for layer in vision.encoder.layers:
                attn = layer.self_attn
                in_w = (
                    torch.cat([p.weight.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)])
                    .bfloat16()
                    .contiguous()
                )
                in_b = (
                    torch.cat([p.bias.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)])
                    .bfloat16()
                    .contiguous()
                )
                out_w, out_b = self._linear(attn.out_proj)
                fc1_w, fc1_b = self._linear(layer.mlp.fc1)
                fc2_w, fc2_b = self._linear(layer.mlp.fc2)
                in_max = max(
                    float(layer.layer_norm1.weight.detach().float().abs().amax().item() * 10.0), 28.0
                )
                fc1_max = max(
                    float(layer.layer_norm2.weight.detach().float().abs().amax().item() * 12.0), 36.0
                )
                entry = (
                    self._norm(layer.layer_norm1),
                    (in_w, in_b),
                    (out_w, out_b),
                    self._norm(layer.layer_norm2),
                    (fc1_w, fc1_b),
                    (fc2_w, fc2_b),
                    attn.num_heads,
                    attn.head_dim,
                    self._pack_fp8(in_w, in_max)
                    if self.fp8
                    else (self._pack_nvfp4(in_w, in_b, in_max, thr) if self.nvfp4 else None),
                    self._pack_fp8(out_w, 12.0)
                    if self.fp8
                    else (self._pack_nvfp4(out_w, out_b, 12.0, thr) if self.nvfp4 else None),
                    self._pack_fp8(fc1_w, fc1_max)
                    if self.fp8
                    else (self._pack_nvfp4(fc1_w, fc1_b, fc1_max, thr, pad_n=48) if self.nvfp4 else None),
                    self._pack_fp8(fc2_w, 48.0)
                    if self.fp8
                    else (self._pack_nvfp4(fc2_w, fc2_b, 48.0, thr, pad_k=48) if self.nvfp4 else None),
                )
                self.fast_layers.append(entry)
            self.final_norm = self._norm(vision.post_layernorm)
            self.projector = self._linear(paligemma.multi_modal_projector.linear)
            proj_max = max(
                float(vision.post_layernorm.weight.detach().float().abs().amax().item() * 12.0), 36.0
            )
            self.proj_fp8 = self._pack_fp8(self.projector[0], proj_max) if self.fp8 else None
            self.proj_fp4 = (
                self._pack_nvfp4(self.projector[0], self.projector[1], proj_max, thr) if self.nvfp4 else None
            )
            if self.nvfp4 and prefetcher is not None:
                # Weights are prefetched just in time: a whole layer issued at layer start is
                # mostly evicted again before its GEMMs run. Entry i holds layer i's packed
                # weights and block scales; the final QKV entry warms the projector.
                self.prefetcher = prefetcher
                self.prefetch_qkv = [prefetcher.segments(entry[8][:2]) for entry in self.fast_layers]
                self.prefetch_qkv.append(prefetcher.segments(self.proj_fp4[:2]))
                self.prefetch_out_fc1 = [
                    prefetcher.segments((*entry[9][:2], *entry[10][:2])) for entry in self.fast_layers
                ]
                self.prefetch_fc2 = [prefetcher.segments(entry[11][:2]) for entry in self.fast_layers]
            return
        self.layers = []
        for layer in vision.encoder.layers:
            attn = layer.self_attn
            width = attn.q_proj.weight.shape[0]
            with torch.device("meta"):
                packed = torch.nn.MultiheadAttention(width, attn.num_heads, batch_first=True)
            state = {
                "in_proj_weight": torch.cat(
                    [p.weight.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
                ).bfloat16(),
                "in_proj_bias": torch.cat(
                    [p.bias.detach() for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
                ).bfloat16(),
                "out_proj.weight": attn.out_proj.weight.detach().bfloat16(),
                "out_proj.bias": attn.out_proj.bias.detach().bfloat16(),
            }
            packed.load_state_dict(state, strict=True, assign=True)
            packed.requires_grad_(False).eval()
            self.layers.append(
                (
                    self._norm(layer.layer_norm1),
                    packed,
                    self._norm(layer.layer_norm2),
                    self._linear(layer.mlp.fc1),
                    self._linear(layer.mlp.fc2),
                )
            )
        self.final_norm = self._norm(vision.post_layernorm)
        self.projector = self._linear(paligemma.multi_modal_projector.linear)

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA SigLIP operator launch failed with error {status}")

    @staticmethod
    def _pack_fp8(weight: torch.Tensor, max_act: float) -> tuple:
        values = weight.detach().contiguous().float()
        w_scale = (values.abs().amax().clamp(min=1e-6) / 448.0).reshape(1)
        w_fp8_t = (values / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).t()
        a_scale = torch.tensor([max_act / 448.0], device=weight.device, dtype=torch.float32)
        a_inv = 448.0 / max_act
        comb_scale = (a_scale * w_scale).contiguous()
        a_inv_t = torch.tensor([a_inv], device=weight.device, dtype=torch.float32)
        return w_fp8_t, w_scale, a_scale, a_inv, comb_scale, a_inv_t

    @staticmethod
    def _pack_nvfp4(
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        max_act: float,
        thr: torch.Tensor,
        *,
        pad_k: int = 0,
        pad_n: int = 0,
    ) -> tuple:
        w = weight.detach().contiguous()
        if pad_k or pad_n:
            w = F.pad(w, (0, pad_k, 0, pad_n))
        b = (
            None
            if bias is None
            else (F.pad(bias.detach().contiguous(), (0, pad_n)) if pad_n else bias.detach().contiguous())
        )
        wq, ws, wg = quantize_weight(w, thr)
        wb = blocked_scales(ws)
        a_global = max_act / (448.0 * 6.0)
        comb_scale = (wg.reshape(1).float() * a_global).contiguous()
        a_inv_t = torch.tensor([(448.0 * 6.0) / max_act], device=weight.device, dtype=torch.float32)
        return wq, wb, comb_scale, a_global, a_inv_t, b

    @staticmethod
    def _linear(module: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (
            module.weight.detach().bfloat16().contiguous(),
            None if module.bias is None else module.bias.detach().bfloat16().contiguous(),
        )

    @staticmethod
    def _norm(module: Any) -> tuple:
        return (
            module.normalized_shape,
            module.weight.detach().bfloat16().contiguous(),
            module.bias.detach().bfloat16().contiguous(),
            module.eps,
        )

    def _call_fast_nvfp4(self, hidden: torch.Tensor) -> torch.Tensor:
        from ....backend.cuda.cudnn_mlp import cudnn_fp4_bias_gelu, cudnn_nvfp4_linear

        b, s, d = hidden.shape
        rows = b * s
        d_mlp = 4352
        with torch.cuda.device(hidden.device):
            stream = torch.cuda.current_stream(hidden.device).cuda_stream
            buf_fp4_d = torch.empty((rows, d // 2), device=hidden.device, dtype=torch.uint8)
            blk_fp4_d = torch.zeros((rows * (d // 16),), device=hidden.device, dtype=torch.uint8)
            buf_fp4_mlp = torch.empty((rows, d_mlp // 2), device=hidden.device, dtype=torch.uint8)
            blk_fp4_mlp = torch.zeros((rows * (d_mlp // 16),), device=hidden.device, dtype=torch.uint8)
            qkv = torch.empty((rows, 3 * d), device=hidden.device, dtype=torch.bfloat16)
            att_out = torch.empty((b, s, d), device=hidden.device, dtype=torch.bfloat16)
            fc2_out = torch.empty((b, s, d), device=hidden.device, dtype=torch.bfloat16)
            res_buf = torch.empty_like(hidden)
            for index, (
                norm1,
                _in_proj,
                _out_proj,
                norm2,
                _first,
                _second,
                heads,
                head_dim,
                in_fp4,
                out_fp4,
                fc1_fp4,
                fc2_fp4,
            ) in enumerate(self.fast_layers):
                if index == 0:
                    self._check(
                        self.nq_lib.cc_layer_norm_fp4(
                            hidden.data_ptr(),
                            None,
                            norm1[1].data_ptr(),
                            norm1[2].data_ptr(),
                            None,
                            buf_fp4_d.data_ptr(),
                            blk_fp4_d.data_ptr(),
                            in_fp4[3],
                            rows,
                            d,
                            stream,
                        )
                    )
                cudnn_nvfp4_linear(buf_fp4_d, blk_fp4_d, in_fp4[0], in_fp4[1], in_fp4[2], in_fp4[5], qkv)
                if self.prefetcher is not None:
                    self.prefetcher.fork(self.prefetch_out_fc1[index])
                q, k, v = qkv.view(b, s, 3, heads, head_dim).unbind(dim=2)
                with sdpa_kernel(self.attention_backends, set_priority=True):
                    attended = F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                    )
                self._check(
                    self.nq_lib.cc_bhsd_fp4(
                        attended.data_ptr(),
                        buf_fp4_d.data_ptr(),
                        blk_fp4_d.data_ptr(),
                        out_fp4[3],
                        attended.stride(0),
                        attended.stride(1),
                        attended.stride(2),
                        b,
                        heads,
                        s,
                        head_dim,
                        stream,
                    )
                )
                cudnn_nvfp4_linear(
                    buf_fp4_d,
                    blk_fp4_d,
                    out_fp4[0],
                    out_fp4[1],
                    out_fp4[2],
                    out_fp4[5],
                    att_out.view(rows, d),
                )
                self._check(
                    self.nq_lib.cc_layer_norm_fp4(
                        hidden.data_ptr(),
                        att_out.data_ptr(),
                        norm2[1].data_ptr(),
                        norm2[2].data_ptr(),
                        res_buf.data_ptr(),
                        buf_fp4_d.data_ptr(),
                        blk_fp4_d.data_ptr(),
                        fc1_fp4[3],
                        rows,
                        d,
                        stream,
                    )
                )
                hidden, res_buf = res_buf, hidden
                if self.prefetcher is not None:
                    self.prefetcher.fork(self.prefetch_fc2[index])
                cudnn_fp4_bias_gelu(
                    buf_fp4_d,
                    blk_fp4_d,
                    fc1_fp4[0],
                    fc1_fp4[1],
                    fc1_fp4[2],
                    fc1_fp4[5],
                    fc2_fp4[4],
                    buf_fp4_mlp,
                    blk_fp4_mlp,
                )
                if self.prefetcher is not None:
                    self.prefetcher.fork(self.prefetch_qkv[index + 1])
                cudnn_nvfp4_linear(
                    buf_fp4_mlp,
                    blk_fp4_mlp,
                    fc2_fp4[0],
                    fc2_fp4[1],
                    fc2_fp4[2],
                    fc2_fp4[5],
                    fc2_out.view(rows, d),
                )
                if index + 1 < len(self.fast_layers):
                    next_norm1 = self.fast_layers[index + 1][0]
                    next_in_fp4 = self.fast_layers[index + 1][8]
                    self._check(
                        self.nq_lib.cc_layer_norm_fp4(
                            hidden.data_ptr(),
                            fc2_out.data_ptr(),
                            next_norm1[1].data_ptr(),
                            next_norm1[2].data_ptr(),
                            res_buf.data_ptr(),
                            buf_fp4_d.data_ptr(),
                            blk_fp4_d.data_ptr(),
                            next_in_fp4[3],
                            rows,
                            d,
                            stream,
                        )
                    )
                    hidden, res_buf = res_buf, hidden
                else:
                    self._check(
                        self.nq_lib.cc_layer_norm_fp4(
                            hidden.data_ptr(),
                            fc2_out.data_ptr(),
                            self.final_norm[1].data_ptr(),
                            self.final_norm[2].data_ptr(),
                            res_buf.data_ptr(),
                            buf_fp4_d.data_ptr(),
                            blk_fp4_d.data_ptr(),
                            self.proj_fp4[3],
                            rows,
                            d,
                            stream,
                        )
                    )
            out_proj_final = torch.empty(
                (rows, self.projector[0].shape[0]), device=hidden.device, dtype=torch.bfloat16
            )
            cudnn_nvfp4_linear(
                buf_fp4_d,
                blk_fp4_d,
                self.proj_fp4[0],
                self.proj_fp4[1],
                self.proj_fp4[2],
                self.proj_fp4[5],
                out_proj_final,
            )
            if self.prefetcher is not None:
                self.prefetcher.join()
            return out_proj_final.view(b, s, -1)

    def _call_fast_fp8(self, hidden: torch.Tensor) -> torch.Tensor:
        from ....backend.cuda.cudnn_mlp import cudnn_fp8_bias_gelu, cudnn_fp8_linear

        b, s, d = hidden.shape
        rows = b * s
        with torch.cuda.device(hidden.device):
            stream = torch.cuda.current_stream(hidden.device).cuda_stream
            buf_fp8_d = torch.empty((rows, d), device=hidden.device, dtype=torch.float8_e4m3fn)
            buf_fp8_mlp = torch.empty((rows, 4304), device=hidden.device, dtype=torch.float8_e4m3fn)
            qkv = torch.empty((rows, 3 * d), device=hidden.device, dtype=torch.bfloat16)
            att_out = torch.empty((b, s, d), device=hidden.device, dtype=torch.bfloat16)
            res_buf = torch.empty_like(hidden)
            for index, (
                norm1,
                in_proj,
                out_proj,
                norm2,
                first,
                second,
                heads,
                head_dim,
                in_fp8,
                out_fp8,
                fc1_fp8,
                fc2_fp8,
            ) in enumerate(self.fast_layers):
                if index == 0:
                    self._check(
                        self.nq_lib.cc_layer_norm_fp8(
                            hidden.data_ptr(),
                            None,
                            norm1[1].data_ptr(),
                            norm1[2].data_ptr(),
                            None,
                            buf_fp8_d.data_ptr(),
                            in_fp8[3],
                            rows,
                            d,
                            stream,
                        )
                    )
                cudnn_fp8_linear(buf_fp8_d, in_fp8[0], in_fp8[4], in_proj[1], qkv)
                q, k, v = qkv.view(b, s, 3, heads, head_dim).unbind(dim=2)
                attended = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
                self._check(
                    self.nq_lib.cc_bhsd_fp8(
                        attended.data_ptr(),
                        buf_fp8_d.data_ptr(),
                        out_fp8[3],
                        attended.stride(0),
                        attended.stride(1),
                        attended.stride(2),
                        b,
                        heads,
                        s,
                        head_dim,
                        stream,
                    )
                )
                cudnn_fp8_linear(buf_fp8_d, out_fp8[0], out_fp8[4], out_proj[1], att_out.view(rows, d))
                self._check(
                    self.nq_lib.cc_layer_norm_fp8(
                        hidden.data_ptr(),
                        att_out.data_ptr(),
                        norm2[1].data_ptr(),
                        norm2[2].data_ptr(),
                        res_buf.data_ptr(),
                        buf_fp8_d.data_ptr(),
                        fc1_fp8[3],
                        rows,
                        d,
                        stream,
                    )
                )
                hidden, res_buf = res_buf, hidden
                cudnn_fp8_bias_gelu(
                    buf_fp8_d,
                    fc1_fp8[0],
                    fc1_fp8[4],
                    first[1],
                    fc2_fp8[5],
                    buf_fp8_mlp,
                )
                fc2 = torch._scaled_mm(
                    buf_fp8_mlp,
                    fc2_fp8[0],
                    scale_a=fc2_fp8[2],
                    scale_b=fc2_fp8[1],
                    bias=second[1],
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                ).view(b, s, d)
                if index + 1 < len(self.fast_layers):
                    next_norm1 = self.fast_layers[index + 1][0]
                    next_in_fp8 = self.fast_layers[index + 1][8]
                    self._check(
                        self.nq_lib.cc_layer_norm_fp8(
                            hidden.data_ptr(),
                            fc2.data_ptr(),
                            next_norm1[1].data_ptr(),
                            next_norm1[2].data_ptr(),
                            res_buf.data_ptr(),
                            buf_fp8_d.data_ptr(),
                            next_in_fp8[3],
                            rows,
                            d,
                            stream,
                        )
                    )
                    hidden, res_buf = res_buf, hidden
                else:
                    self._check(
                        self.nq_lib.cc_layer_norm_fp8(
                            hidden.data_ptr(),
                            fc2.data_ptr(),
                            self.final_norm[1].data_ptr(),
                            self.final_norm[2].data_ptr(),
                            res_buf.data_ptr(),
                            buf_fp8_d.data_ptr(),
                            self.proj_fp8[3],
                            rows,
                            d,
                            stream,
                        )
                    )
            return torch._scaled_mm(
                buf_fp8_d,
                self.proj_fp8[0],
                scale_a=self.proj_fp8[2],
                scale_b=self.proj_fp8[1],
                bias=self.projector[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            ).view(b, s, -1)

    def _bias_positions(self, batch: int) -> torch.Tensor:
        """Patch bias plus positions expanded to ``batch`` images, built once outside graph capture."""
        expanded = self._bias_positions_by_batch.get(batch)
        if expanded is None:
            expanded = self.patch_bias_pos_bf16.expand(batch, -1, -1).contiguous()
            if not torch.cuda.is_current_stream_capturing():
                self._bias_positions_by_batch[batch] = expanded
        return expanded

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized BCHW images using the original MHA fast path."""
        if self.fast_sdpa:
            if self.prefetcher is not None:
                self.prefetcher.fork(self.prefetch_qkv[0])
            b_img, c_img, h_img, w_img = images.shape
            if (c_img, h_img, w_img) == (3, 224, 224) and tuple(self.stride) == (14, 14):
                patches = (
                    images.bfloat16()
                    .view(b_img, 3, 16, 14, 16, 14)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(b_img * 256, 588)
                )
                # Bias and positions are added at full shape: a broadcast add does not vectorize.
                hidden = torch.matmul(patches, self.patch_weight_lin_t).view(b_img, 256, 1152)
                hidden.add_(self._bias_positions(b_img))
            else:
                hidden = F.conv2d(
                    images.bfloat16().contiguous(memory_format=torch.channels_last),
                    self.patch_weight_bf16,
                    self.patch_bias_bf16,
                    stride=self.stride,
                    padding=self.padding,
                )
                hidden = (hidden.flatten(2).transpose(1, 2) + self.positions_bf16).contiguous()
            if self.nvfp4:
                return self._call_fast_nvfp4(hidden)
            if self.fp8:
                return self._call_fast_fp8(hidden)
            b, s, d = hidden.shape
            rows = b * s
            with torch.cuda.device(hidden.device):
                stream = torch.cuda.current_stream(hidden.device).cuda_stream
                norm_buf = torch.empty_like(hidden)
                res_buf = torch.empty_like(hidden)
                mlp_buf = torch.empty((b, s, 4304), device=hidden.device, dtype=torch.bfloat16)
                for index, (norm1, in_proj, out_proj, norm2, first, second, heads, head_dim, *_) in enumerate(
                    self.fast_layers
                ):
                    if index == 0:
                        self._check(
                            self.nq_lib.cc_layer_norm_fp8(
                                hidden.data_ptr(),
                                None,
                                norm1[1].data_ptr(),
                                norm1[2].data_ptr(),
                                None,
                                norm_buf.data_ptr(),
                                -1.0,
                                rows,
                                d,
                                stream,
                            )
                        )
                    qkv = F.linear(norm_buf, *in_proj)
                    q, k, v = qkv.view(b, s, 3, heads, head_dim).unbind(dim=2)
                    attended = (
                        F.scaled_dot_product_attention(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                        )
                        .transpose(1, 2)
                        .reshape(b, s, d)
                    )
                    att_out = F.linear(attended, *out_proj)
                    self._check(
                        self.nq_lib.cc_layer_norm_fp8(
                            hidden.data_ptr(),
                            att_out.data_ptr(),
                            norm2[1].data_ptr(),
                            norm2[2].data_ptr(),
                            res_buf.data_ptr(),
                            norm_buf.data_ptr(),
                            -1.0,
                            rows,
                            d,
                            stream,
                        )
                    )
                    hidden, res_buf = res_buf, hidden
                    fc1 = F.linear(norm_buf, *first)
                    self._check(
                        self.fus_lib.cc_geglu(
                            fc1.data_ptr(),
                            None,
                            mlp_buf.data_ptr(),
                            None,
                            rows * 4304,
                            16,
                            self.gelu_table.data_ptr(),
                            stream,
                        )
                    )
                    fc2 = F.linear(mlp_buf, *second)
                    if index + 1 < len(self.fast_layers):
                        next_norm1 = self.fast_layers[index + 1][0]
                        self._check(
                            self.nq_lib.cc_layer_norm_fp8(
                                hidden.data_ptr(),
                                fc2.data_ptr(),
                                next_norm1[1].data_ptr(),
                                next_norm1[2].data_ptr(),
                                res_buf.data_ptr(),
                                norm_buf.data_ptr(),
                                -1.0,
                                rows,
                                d,
                                stream,
                            )
                        )
                        hidden, res_buf = res_buf, hidden
                    else:
                        self._check(
                            self.nq_lib.cc_layer_norm_fp8(
                                hidden.data_ptr(),
                                fc2.data_ptr(),
                                self.final_norm[1].data_ptr(),
                                self.final_norm[2].data_ptr(),
                                res_buf.data_ptr(),
                                norm_buf.data_ptr(),
                                -1.0,
                                rows,
                                d,
                                stream,
                            )
                        )
                return F.linear(norm_buf, *self.projector)
        hidden = F.conv2d(
            images.float().contiguous(memory_format=torch.channels_last),
            self.patch_weight,
            self.patch_bias,
            stride=self.stride,
            padding=self.padding,
        )
        hidden = (hidden.flatten(2).transpose(1, 2) + self.positions).bfloat16()
        for norm1, attention, norm2, first, second in self.layers:
            normalized = F.layer_norm(hidden, *norm1)
            attended, _ = attention(normalized, normalized, normalized, average_attn_weights=False)
            hidden = hidden + attended
            hidden = hidden + F.linear(F.gelu(F.linear(F.layer_norm(hidden, *norm2), *first)), *second)
        return F.linear(F.layer_norm(hidden, *self.final_norm), *self.projector)
