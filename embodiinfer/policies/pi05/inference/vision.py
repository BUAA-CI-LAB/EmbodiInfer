"""Derived RLinf SigLIP execution over the existing checkpoint parameters."""

import ctypes
from typing import Any

import torch
import torch.nn.functional as F

from ....backend.cuda.native import build_library


class SiglipPlan:
    """Own BF16 casts and packed MHA weights without changing checkpoint modules.

    Rebuild with the policy controller after refit or device migration. The FP32
    stem and positions, BF16 encoder/projector, exact GELU and native MHA calls
    preserve the RLinf vision precision contract.
    """

    @torch.no_grad()
    def __init__(self, paligemma: Any, *, fast_sdpa: bool = False, fp8: bool = False) -> None:
        """Prepare immutable weight views/casts before graph capture."""
        self.fast_sdpa = fast_sdpa
        self.fp8 = fp8 and hasattr(torch, "_scaled_mm")
        vision = paligemma.vision_tower.vision_model
        patch = vision.embeddings.patch_embedding
        self.patch_weight = patch.weight.detach().float()
        self.patch_bias = None if patch.bias is None else patch.bias.detach().float()
        self.stride, self.padding = patch.stride, patch.padding
        self.positions = vision.embeddings.position_embedding.weight.detach().float()[None]
        if fast_sdpa:
            self.patch_weight_bf16 = (
                patch.weight.detach().bfloat16().contiguous(memory_format=torch.channels_last)
            )
            self.patch_bias_bf16 = None if patch.bias is None else patch.bias.detach().bfloat16()
            self.positions_bf16 = self.positions.bfloat16()
            if True:
                with torch.cuda.device(patch.weight.device):
                    self.nq_lib = build_library(
                        "norm_quant", strict=True, specific=True, native_fp4=False
                    ).library
                    self.q_lib = build_library(
                        "quantize", strict=False, specific=True, native_fp4=False
                    ).library
                    self.fus_lib = build_library(
                        "fusion", strict=False, specific=True, native_fp4=False
                    ).library
                ptr = ctypes.c_void_p
                self.nq_lib.cc_layer_norm_fp8.argtypes = [ptr] * 6 + [
                    ctypes.c_float,
                    ctypes.c_int,
                    ctypes.c_int,
                    ptr,
                ]
                self.nq_lib.cc_layer_norm_fp8.restype = ctypes.c_int
                self.nq_lib.cc_bhsd_fp8.argtypes = [
                    ptr,
                    ptr,
                    ctypes.c_float,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ptr,
                ]
                self.nq_lib.cc_bhsd_fp8.restype = ctypes.c_int
                self.q_lib.cc_fp8.argtypes = [ptr, ptr, ptr, ctypes.c_int64, ptr]
                self.q_lib.cc_fp8.restype = ctypes.c_int
                self.fus_lib.cc_geglu.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int, ptr, ptr]
                self.fus_lib.cc_geglu.restype = ctypes.c_int
                codes = torch.arange(65536, device=patch.weight.device, dtype=torch.int32).to(torch.uint16)
                self.gelu_table = F.gelu(codes.view(torch.bfloat16), approximate="tanh").view(torch.uint16)
                self.ones_4304 = torch.ones((1024, 4304), device=patch.weight.device, dtype=torch.bfloat16)
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
                entry = (
                    self._norm(layer.layer_norm1),
                    (in_w, in_b),
                    (out_w, out_b),
                    self._norm(layer.layer_norm2),
                    (fc1_w, fc1_b),
                    (fc2_w, fc2_b),
                    attn.num_heads,
                    attn.head_dim,
                    self._pack_fp8(in_w, 8.0) if self.fp8 else None,
                    self._pack_fp8(out_w, 8.0) if self.fp8 else None,
                    self._pack_fp8(fc1_w, 8.0) if self.fp8 else None,
                    self._pack_fp8(fc2_w, 16.0) if self.fp8 else None,
                )
                self.fast_layers.append(entry)
            self.final_norm = self._norm(vision.post_layernorm)
            self.projector = self._linear(paligemma.multi_modal_projector.linear)
            self.proj_fp8 = self._pack_fp8(self.projector[0], 16.0) if self.fp8 else None
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
            packed.load_state_dict(state, assign=True)
            packed.eval()
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
    def _pack_fp8(weight: torch.Tensor, max_act: float) -> tuple:
        values = weight.detach().contiguous().float()
        w_scale = (values.abs().amax().clamp(min=1e-6) / 448.0).reshape(1)
        w_fp8_t = (values / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).t()
        a_scale = torch.tensor([max_act / 448.0], device=weight.device, dtype=torch.float32)
        a_inv = 448.0 / max_act
        return w_fp8_t, w_scale, a_scale, a_inv

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

    def _call_fast_fp8(self, hidden: torch.Tensor) -> torch.Tensor:
        b, s, d = hidden.shape
        rows = b * s
        stream = torch.cuda.current_stream(hidden.device).cuda_stream
        buf_fp8_d = torch.empty((rows, d), device=hidden.device, dtype=torch.float8_e4m3fn)
        buf_fp8_mlp = torch.empty((rows, 4304), device=hidden.device, dtype=torch.float8_e4m3fn)
        res_buf = torch.empty_like(hidden)
        ones = (
            self.ones_4304[:rows]
            if rows <= self.ones_4304.shape[0]
            else torch.ones((rows, 4304), device=hidden.device, dtype=torch.bfloat16)
        )
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
            qkv = torch._scaled_mm(
                buf_fp8_d,
                in_fp8[0],
                scale_a=in_fp8[2],
                scale_b=in_fp8[1],
                bias=in_proj[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
            q, k, v = qkv.view(b, s, 3, heads, head_dim).unbind(dim=2)
            attended = (
                F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                .transpose(1, 2)
                .reshape(rows, d)
            )
            self.q_lib.cc_fp8(
                attended.data_ptr(), buf_fp8_d.data_ptr(), out_fp8[2].data_ptr(), rows * d, stream
            )
            att_out = torch._scaled_mm(
                buf_fp8_d,
                out_fp8[0],
                scale_a=out_fp8[2],
                scale_b=out_fp8[1],
                bias=out_proj[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            ).view(b, s, d)
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
            hidden, res_buf = res_buf, hidden
            fc1 = torch._scaled_mm(
                buf_fp8_d,
                fc1_fp8[0],
                scale_a=fc1_fp8[2],
                scale_b=fc1_fp8[1],
                bias=first[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
            self.fus_lib.cc_geglu(
                fc1.data_ptr(),
                ones.data_ptr(),
                buf_fp8_mlp.data_ptr(),
                fc2_fp8[2].data_ptr(),
                rows * 4304,
                8,
                self.gelu_table.data_ptr(),
                stream,
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
                hidden, res_buf = res_buf, hidden
            else:
                hidden = hidden + fc2
        return F.linear(F.layer_norm(hidden, *self.final_norm), *self.projector)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized BCHW images using the original MHA fast path."""
        if self.fast_sdpa:
            hidden = F.conv2d(
                images.bfloat16().contiguous(memory_format=torch.channels_last),
                self.patch_weight_bf16,
                self.patch_bias_bf16,
                stride=self.stride,
                padding=self.padding,
            )
            hidden = (hidden.flatten(2).transpose(1, 2) + self.positions_bf16).contiguous()
            if self.fp8:
                return self._call_fast_fp8(hidden)
            b, s, d = hidden.shape
            rows = b * s
            stream = torch.cuda.current_stream(hidden.device).cuda_stream
            norm_buf = torch.empty_like(hidden)
            res_buf = torch.empty_like(hidden)
            mlp_buf = torch.empty((b, s, 4304), device=hidden.device, dtype=torch.bfloat16)
            ones = (
                self.ones_4304[:rows]
                if rows <= self.ones_4304.shape[0]
                else torch.ones((rows, 4304), device=hidden.device, dtype=torch.bfloat16)
            )
            for index, (norm1, in_proj, out_proj, norm2, first, second, heads, head_dim, *_) in enumerate(
                self.fast_layers
            ):
                if index == 0:
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
                qkv = F.linear(norm_buf, *in_proj)
                q, k, v = qkv.view(b, s, 3, heads, head_dim).unbind(dim=2)
                attended = (
                    F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                    .transpose(1, 2)
                    .reshape(b, s, d)
                )
                att_out = F.linear(attended, *out_proj)
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
                hidden, res_buf = res_buf, hidden
                fc1 = F.linear(norm_buf, *first)
                self.fus_lib.cc_geglu(
                    fc1.data_ptr(),
                    ones.data_ptr(),
                    mlp_buf.data_ptr(),
                    None,
                    rows * 4304,
                    16,
                    self.gelu_table.data_ptr(),
                    stream,
                )
                fc2 = F.linear(mlp_buf, *second)
                if index + 1 < len(self.fast_layers):
                    next_norm1 = self.fast_layers[index + 1][0]
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
                    hidden, res_buf = res_buf, hidden
                else:
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
