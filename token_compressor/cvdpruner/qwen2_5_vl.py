from typing import Optional, Union, List
import os
import time
import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

import torch.nn.functional as F
from typing import List, Optional
import math

from typing import List
import torch
import torch.nn.functional as F


def _norm01_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def get_selected_indices_for_images_qwen(
    image_splits: List[torch.Tensor],
    keep_per_image: int = 32,
    beta: float = 3.0,
    w_self: float = 0.35,
    w_dist: float = 0.50,
    w_intra: float = 0.15,
    lambda_red: float = 0.4,
    keep_order: bool = True,
    debug: bool = False,
    ablation_mode: str = "full",
) -> List[torch.LongTensor]:
    """
    Qwen 版 CID：支持不同 n_i

    ablation_mode:
        "only_self"        : Only Self
        "self_dist"        : Self + Dist
        "self_intra"       : Self + Intra
        "full"             : Self + Dist + Intra
        "wo_fixed_greedy"  : 去掉 Fixed Budget + Redundancy-Aware Greedy Search
    """
    assert ablation_mode in ["only_self", "self_dist", "self_intra", "full", "wo_fixed_greedy"]

    I = len(image_splits)
    if I == 0:
        return []

    device = image_splits[0].device
    x_list = [F.normalize(feat.float(), dim=-1) for feat in image_splits]
    g = torch.stack([F.normalize(x.mean(dim=0), dim=-1) for x in x_list], dim=0)

    all_base_scores: List[torch.Tensor] = []

    for i in range(I):
        x = x_list[i]
        n_i = x.shape[0]

        if n_i == 0:
            all_base_scores.append(torch.empty((0,), dtype=torch.float32, device=device))
            continue

        mean = g[i].view(1, -1)

        intra = 1.0 - (x * mean).sum(dim=-1)
        self_sim = (x * mean).sum(dim=-1)

        if I == 1:
            distinct = torch.ones((n_i,), device=device, dtype=x.dtype)
        else:
            sim_to_all_g = x @ g.t()
            sim_to_all_g[:, i] = -1e4
            other_sim = sim_to_all_g.max(dim=-1).values
            distinct = 1.0 - other_sim

        self_hat = _norm01_per_image(self_sim)
        dist_hat = _norm01_per_image(distinct)
        intra_hat = _norm01_per_image(intra)

        # CID scoring ablation
        if ablation_mode == "only_self":
            base = self_hat

        elif ablation_mode == "self_dist":
            denom = w_self + w_dist
            if denom <= 0:
                base = 0.5 * self_hat + 0.5 * dist_hat
            else:
                base = (w_self / denom) * self_hat + (w_dist / denom) * dist_hat

        elif ablation_mode == "self_intra":
            denom = w_self + w_intra
            if denom <= 0:
                base = 0.5 * self_hat + 0.5 * intra_hat
            else:
                base = (w_self / denom) * self_hat + (w_intra / denom) * intra_hat

        else:
            # "full" or "wo_fixed_greedy"
            base = w_self * self_hat + w_dist * dist_hat + w_intra * intra_hat

        all_base_scores.append(base)

        if debug:
            print({
                "img": i,
                "mode": ablation_mode,
                "n_i": n_i,
                "base_mean": float(base.mean().item()),
                "self_mean": float(self_hat.mean().item()),
                "dist_mean": float(dist_hat.mean().item()),
                "intra_mean": float(intra_hat.mean().item()),
            })

    # 去掉 fixed per-image budget + greedy
    if ablation_mode == "wo_fixed_greedy":
        total_budget = I * keep_per_image
        lengths = [x.shape[0] for x in x_list]
        total_tokens = sum(lengths)

        if total_tokens == 0:
            return [torch.empty((0,), dtype=torch.long, device=device) for _ in range(I)]

        total_budget = min(max(1, total_budget), total_tokens)
        concat_scores = torch.cat(all_base_scores, dim=0)
        top_global = torch.topk(concat_scores, k=total_budget, largest=True).indices

        selected = []
        start = 0
        for i, n_i in enumerate(lengths):
            end = start + n_i
            mask = (top_global >= start) & (top_global < end)
            idx_global_i = top_global[mask]
            idx_local_i = idx_global_i - start

            if keep_order and idx_local_i.numel() > 0:
                idx_local_i, _ = torch.sort(idx_local_i)

            selected.append(idx_local_i.to(torch.long))
            start = end

        return selected

    # 保留 fixed per-image budget + greedy
    selected: List[torch.LongTensor] = []

    for i in range(I):
        x = x_list[i]
        base = all_base_scores[i]
        n_i = x.shape[0]

        if n_i == 0:
            selected.append(torch.empty((0,), dtype=torch.long, device=device))
            continue

        K = int(min(max(1, keep_per_image), n_i))
        cand = int(min(n_i, max(K, int(beta * K))))

        cand_idx = torch.topk(base, k=cand, largest=True).indices
        cand_feat = x.index_select(0, cand_idx)
        cand_base = base.index_select(0, cand_idx)

        picked_mask = torch.zeros((cand,), device=device, dtype=torch.bool)
        max_sim_sel = torch.zeros((cand,), device=device, dtype=cand_feat.dtype)
        picked = []

        for _ in range(K):
            gain = cand_base - lambda_red * max_sim_sel
            gain = gain.masked_fill(picked_mask, -1e9)
            j = int(torch.argmax(gain).item())
            picked_mask[j] = True
            picked.append(j)

            new = cand_feat[j]
            sim_new = cand_feat @ new
            max_sim_sel = torch.maximum(max_sim_sel, sim_new)

        idx = cand_idx[torch.tensor(picked, device=device, dtype=torch.long)]
        if keep_order:
            idx, _ = torch.sort(idx)

        selected.append(idx.to(torch.long))

        if debug:
            print({
                "img": i,
                "mode": ablation_mode,
                "K": K,
                "cand": cand,
                "selected_num": int(idx.numel()),
            })

    return selected


def Qwen2_5_VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    """Patched forward that enables RANDOM multi-image token pruning for Qwen2.5-VL."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    compressor = os.getenv("COMPRESSOR", "").lower()
    compression_on = (
        compressor in ("cid_img", "cid_image", "cid")
        and (pixel_values is not None)
        and (image_grid_thw is not None)
        and (past_key_values is None or past_key_values.get_seq_length() == 0)  # 仅 prefill
        and (image_mask is not None)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if compression_on:
        # --- 1) 计算每张图 token 数，并 split 出来 ---
        merge_size = self.visual.spatial_merge_size
        split_sizes = (image_grid_thw.prod(-1) // (merge_size**2)).tolist()  # [t1, t2, ...]
        image_splits = list(torch.split(image_embeds, split_sizes))          # [(t_i,D)]

        # --- 2) 读取超参（不设环境变量就用默认）---
        keep_per_image = int(os.getenv("IMG_KEEP_PER_IMAGE", "64"))
        beta = float(os.getenv("CID_BETA", "3.0"))
        w_self = float(os.getenv("CID_W_SELF", "0.35"))
        w_dist = float(os.getenv("CID_W_DIST", "0.50"))
        w_intra = float(os.getenv("CID_W_INTRA", "0.15"))
        lambda_red = float(os.getenv("CID_LAMBDA_RED", "0.4"))
        debug = os.getenv("CID_DEBUG", "").strip() != ""
        ablation_mode = os.getenv("ablation_mode", "full")
        # --- 3) 对每张图选出要保留的 local indices ---
        selected_local = get_selected_indices_for_images_qwen(
            image_splits=image_splits,
            keep_per_image=keep_per_image,
            beta=beta,
            w_self=w_self,
            w_dist=w_dist,
            w_intra=w_intra,
            lambda_red=lambda_red,
            keep_order=True,
            debug=debug,
            ablation_mode=ablation_mode
        )  # List[(K_i,)]

        kept_indices: List[torch.Tensor] = []
        kept_img_chunks: List[torch.Tensor] = []
        offset = 0
        for feat, idx_local in zip(image_splits, selected_local):
            n = feat.shape[0]
            if n == 0:
                continue
            idx_local = idx_local.clamp_min(0).clamp_max(n - 1)
            kept_indices.append(idx_local + offset)
            kept_img_chunks.append(feat.index_select(0, idx_local))
            offset += n

        kept_indices = torch.sort(torch.cat(kept_indices)).values
        image_embeds = torch.cat(kept_img_chunks, dim=0)

        image_token_positions = image_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_image_positions = image_token_positions[kept_indices]

        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_image_positions = all_positions[~image_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_image_positions, kept_image_positions)).sort().values

        def _prune_attention(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn is None:
                return None
            if attn.dim() == 2:
                return attn[:, keep_token_indices]
            if attn.dim() == 4:
                return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = _prune_attention(attention_mask)
        position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        if debug:
            print("[CID] split_sizes:", split_sizes)
            print("[CID] kept_image_tokens:", int(image_embeds.shape[0]))
            print("[CID] kept_total_tokens:", int(inputs_embeds.shape[1]))
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
