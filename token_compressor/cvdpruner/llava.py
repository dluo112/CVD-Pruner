from abc import ABC, abstractmethod
import math
import re
import time
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image, ImageDraw
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import os
from token_compressor.cvdpruner import *
from typing import List, Optional, Tuple
import torch.nn.functional as F

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def get_selected_indices_for_images(
    image_features: List[torch.Tensor],
    keep_per_image: int = 128, # 每图固定 K
    beta: float = 3.0, # 候选池倍率（先粗筛再精选）
    w_self: float = 0.35, # “像我自己的主体”
    w_dist: float = 0.50, # “不像别人的互补”
    w_intra: float = 0.15,# “本图内多样”
    lambda_red: float = 0.4,# 越大越不允许选相似 token
    seed: Optional[int] = None,
    keep_order: bool = True,
    debug: bool = False,
    ablation_mode: str = "full",
) -> List[torch.LongTensor]:
    """
    ablation_mode:
        "only_self"        : Only Self
        "self_dist"        : Self + Dist
        "self_intra"       : Self + Intra
        "full"             : Self + Dist + Intra
        "wo_fixed_greedy"  : 去掉 Fixed Budget + Redundancy-Aware Greedy Search
    返回：
        List[LongTensor]，每个元素是该图被选 token 的索引
        （相对该图自己的 token 序列）。
    """
    assert ablation_mode in ["only_self", "self_dist", "self_intra", "full", "wo_fixed_greedy"]

    if seed is not None:
        torch.manual_seed(seed)

    I = len(image_features)
    if I == 0:
        return []

    first = None
    for t in image_features:
        if t is not None and t.numel() > 0:
            first = t
            break

    if first is None:
        return [torch.empty((0,), dtype=torch.long) for _ in range(I)]

    device = first.device
    D = None

    x_list: List[torch.Tensor] = []
    N_list: List[int] = []

    for i, feat in enumerate(image_features):
        if feat is None:
            feat = torch.empty((0, first.shape[-1]), device=device, dtype=first.dtype)

        if feat.device != device:
            feat = feat.to(device)

        if feat.ndim != 2:
            raise ValueError(f"image_features[{i}] must be 2D (N,D), got {feat.shape}")

        if D is None:
            D = feat.shape[-1]
        elif feat.shape[-1] != D:
            raise ValueError(
                f"Dim mismatch: image_features[{i}].shape[-1]={feat.shape[-1]} != {D}"
            )

        N_i = int(feat.shape[0])
        N_list.append(N_i)

        if N_i == 0:
            x_list.append(torch.empty((0, D), device=device, dtype=torch.float32))
        else:
            x_list.append(F.normalize(feat.float(), dim=-1))

    g_list: List[torch.Tensor] = []
    for i in range(I):
        if N_list[i] == 0:
            g_list.append(torch.zeros((D,), device=device, dtype=torch.float32))
        else:
            gi = x_list[i].mean(dim=0)
            gi = F.normalize(gi, dim=-1)
            g_list.append(gi)

    g = torch.stack(g_list, dim=0)  # (I, D)

    def norm01_1d(v: torch.Tensor) -> torch.Tensor:
        if v.numel() == 0:
            return v
        vmin = v.min()
        vmax = v.max()
        return (v - vmin) / (vmax - vmin + 1e-6)

    all_base_scores: List[torch.Tensor] = []
    for i in range(I):
        N_i = N_list[i]
        if N_i == 0:
            all_base_scores.append(torch.empty((0,), device=device, dtype=torch.float32))
            continue

        x = x_list[i]            # (N_i, D)
        gi = g[i].view(1, D)     # (1, D)

        self_sim = (x * gi).sum(dim=-1)   # (N_i,)
        intra = 1.0 - self_sim            # (N_i,)
        if I == 1:
            distinct = torch.ones((N_i,), device=device, dtype=x.dtype)
        else:
            sim_to_all = x @ g.t()                # (N_i, I)
            sim_to_all[:, i] = -1e4              # mask self
            other_sim = sim_to_all.max(dim=-1).values
            distinct = 1.0 - other_sim
        self_hat = norm01_1d(self_sim)
        dist_hat = norm01_1d(distinct)
        intra_hat = norm01_1d(intra)

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
                "N": N_i,
                "base_mean": float(base.mean().item()),
                "self_mean": float(self_hat.mean().item()),
                "dist_mean": float(dist_hat.mean().item()),
                "intra_mean": float(intra_hat.mean().item()),
            })
    # ------------------------------------------------------------------
    # 消融：去掉 fixed per-image budget + redundancy-aware greedy
    # 做法：全局 top-(I * keep_per_image)，按图分配，且不做 greedy 去重
    # ------------------------------------------------------------------
    if ablation_mode == "wo_fixed_greedy":
        total_budget = I * keep_per_image
        total_tokens = sum(N_list)

        if total_tokens == 0:
            return [torch.empty((0,), dtype=torch.long, device=device) for _ in range(I)]

        total_budget = min(max(1, total_budget), total_tokens)
        concat_scores = torch.cat(all_base_scores, dim=0)
        top_global = torch.topk(concat_scores, k=total_budget, largest=True).indices

        selected_indices: List[torch.LongTensor] = []
        start = 0
        for i, N_i in enumerate(N_list):
            end = start + N_i
            mask = (top_global >= start) & (top_global < end)
            idx_global_i = top_global[mask]
            idx_local_i = idx_global_i - start

            if keep_order and idx_local_i.numel() > 0:
                idx_local_i, _ = torch.sort(idx_local_i)

            selected_indices.append(idx_local_i.to(torch.long))
            start = end

        return selected_indices
    # ------------------------------------------------------------------
    # 默认：保留 fixed per-image budget + redundancy-aware greedy
    # ------------------------------------------------------------------
    selected_indices: List[torch.LongTensor] = []

    for i in range(I):
        N_i = N_list[i]
        if N_i == 0:
            selected_indices.append(torch.empty((0,), device=device, dtype=torch.long))
            continue

        x = x_list[i]
        base = all_base_scores[i]

        K = int(min(max(1, keep_per_image), N_i))
        cand = int(min(N_i, max(K, int(beta * K))))

        # 候选池：top-(beta*K)
        cand_idx = torch.topk(base, k=cand, largest=True).indices
        cand_feat = x.index_select(0, cand_idx)
        cand_base = base.index_select(0, cand_idx)

        picked = []
        picked_mask = torch.zeros((cand,), device=device, dtype=torch.bool)

        # redundancy tracking：与已选 token 的最大相似度
        max_sim_sel = torch.zeros((cand,), device=device, dtype=cand_feat.dtype)

        for _ in range(K):
            gain = cand_base - lambda_red * max_sim_sel
            gain = gain.masked_fill(picked_mask, -1e9)

            j = int(torch.argmax(gain).item())
            picked_mask[j] = True
            picked.append(j)

            new = cand_feat[j]                # (D,)
            sim_new = cand_feat @ new         # (cand,)
            max_sim_sel = torch.maximum(max_sim_sel, sim_new)

        idx = cand_idx[torch.tensor(picked, device=device, dtype=torch.long)]

        if keep_order:
            idx, _ = torch.sort(idx)

        selected_indices.append(idx.to(torch.long))

        if debug:
            with torch.no_grad():
                self_sim = (x * g[i].view(1, D)).sum(dim=-1)
                intra = 1.0 - self_sim

                if I == 1:
                    distinct = torch.ones((N_i,), device=device, dtype=x.dtype)
                else:
                    sim_to_all = x @ g.t()
                    sim_to_all[:, i] = -1e4
                    other_sim = sim_to_all.max(dim=-1).values
                    distinct = 1.0 - other_sim

                self_hat = norm01_1d(self_sim)
                dist_hat = norm01_1d(distinct)
                intra_hat = norm01_1d(intra)

                avg_self = float(self_hat.index_select(0, idx).mean().item())
                avg_dist = float(dist_hat.index_select(0, idx).mean().item())
                avg_intra = float(intra_hat.index_select(0, idx).mean().item())

            print({
                "img": i,
                "mode": ablation_mode,
                "N": N_i,
                "K": K,
                "cand": cand,
                "selected_num": int(idx.numel()),
                "avg_self": avg_self,
                "avg_dist": avg_dist,
                "avg_intra": avg_intra,
            })

    return selected_indices

def get_selected_indices_for_video_frames(
    video_feature: torch.Tensor,   # (T, N, D)
    keep_per_frame: int = 64,
    beta: float = 3.0,
    w_self: float = 0.35,
    w_dist: float = 0.50,
    w_intra: float = 0.15,
    lambda_red: float = 0.4,
    seed: Optional[int] = None,
    keep_order: bool = True,
    debug: bool = False,
) -> List[torch.LongTensor]:
    """
    对视频逐帧做 CID 选择。
    输入:
        video_feature: (T, N, D)
    返回:
        长度 T 的 list，每个元素是该帧保留的 token 索引
    """
    if video_feature.ndim != 3:
        raise ValueError(f"video_feature must be 3D (T,N,D), got {tuple(video_feature.shape)}")

    T = int(video_feature.shape[0])
    frame_list = [video_feature[t] for t in range(T)]  # list[(N,D)]

    return get_selected_indices_for_images(
        image_features=frame_list,
        keep_per_image=keep_per_frame,
        beta=beta,
        w_self=w_self,
        w_dist=w_dist,
        w_intra=w_intra,
        lambda_red=lambda_red,
        seed=seed,
        keep_order=keep_order,
        debug=debug,
        ablation_mode="self_intra"
    )


def apply_framewise_token_pruning(
    video_feature: torch.Tensor,                 # (T, N, D)
    selected_indices: List[torch.LongTensor],   # len=T
) -> torch.Tensor:
    """
    按逐帧索引裁剪视频 token。
    返回:
        pruned_video_feature: (T, K_t, D)
    """
    if video_feature.ndim != 3:
        raise ValueError(f"video_feature must be 3D (T,N,D), got {tuple(video_feature.shape)}")

    T = int(video_feature.shape[0])
    if len(selected_indices) != T:
        raise ValueError(f"len(selected_indices)={len(selected_indices)} != T={T}")

    pruned_frames = []
    for t in range(T):
        frame_feat = video_feature[t]   # (N,D)
        idx = selected_indices[t]

        if idx.device != frame_feat.device:
            idx = idx.to(frame_feat.device)

        pruned_frames.append(frame_feat.index_select(0, idx))

    return torch.stack(pruned_frames, dim=0)


def cus_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=["image"],
    image_sizes=None,
):
    import os
    import math
    import re
    import random
    import torch
    import torch.nn as nn
    import PIL

    vision_tower = self.get_vision_tower()
    image_feature_is_video = []

    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels
    if isinstance(modalities, str):
        modalities = [modalities]

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]

        encoded_image_features = self.encode_images(concat_images)
        # image_features, all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

        # This is a list, each element is [num_images, patch * patch, dim]
        encoded_image_features = torch.split(encoded_image_features, split_sizes)

        image_features = []
        image_feature_is_video = []

        for idx, image_feat in enumerate(encoded_image_features):
            is_vid = (idx in video_idx_in_batch)
            if is_vid:
                pooled = self.get_2dPool(image_feat)
                image_features.append(pooled)
                image_feature_is_video.append(True)
            else:
                image_features.append(image_feat)
                image_feature_is_video.append(False)

        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            new_image_feature_is_video = []

            for image_idx, image_feature in enumerate(image_features):
                if image_idx in video_idx_in_batch:  # video operations
                    # -------------------------------------------------
                    # NEW: frame-wise pruning BEFORE flatten / newline merge
                    # image_feature: (T, N, D)
                    # -------------------------------------------------
                    keep_per_frame = getattr(self.config, "mm_keep_per_frame", 30)

                    video_selected_indices = get_selected_indices_for_video_frames(
                        video_feature=image_feature,
                        keep_per_frame=keep_per_frame,
                        beta=3.0,
                        w_self=0.35,
                        w_dist=0.50,
                        w_intra=0.15,
                        lambda_red=0.4,
                        seed=321,
                        keep_order=True,
                        debug=False,
                    )

                    image_feature = apply_framewise_token_pruning(
                        video_feature=image_feature,
                        selected_indices=video_selected_indices,
                    )  # (T, K, D)

                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        if getattr(self.config, "add_faster_video", False):
                            if "all_faster_video_features" not in locals():
                                raise RuntimeError(
                                    "add_faster_video=True, but all_faster_video_features is not available. "
                                    "Please use encode_multimodals(...) or disable add_faster_video."
                                )

                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])

                            # Add a token for each frame
                            concat_slow_fater_token = []
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(
                                        torch.cat(
                                            (image_feature[_], self.model.faster_token[None].to(image_feature.device)),
                                            dim=0,
                                        )
                                    )
                                else:
                                    concat_slow_fater_token.append(
                                        torch.cat(
                                            (faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)),
                                            dim=0,
                                        )
                                    )
                            image_feature = torch.cat(concat_slow_fater_token)

                        new_image_features.append(image_feature)
                        new_image_feature_is_video.append(True)

                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature)
                        new_image_features.append(image_feature.flatten(0, 1))
                        new_image_feature_is_video.append(True)

                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1)
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ),
                                dim=0,
                            )
                        new_image_features.append(image_feature)
                        new_image_feature_is_video.append(True)

                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                        new_image_feature_is_video.append(True)

                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    matched_anyres_max_num_patches = None
                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                vision_tower_image_size,
                            )
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(
                                image_feature,
                                [int(h // times), int(w // times)],
                                mode="bilinear",
                            )[0]
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)

                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)

                    new_image_features.append(image_feature)
                    new_image_feature_is_video.append(False)

                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                    new_image_feature_is_video.append(False)

            image_features = new_image_features
            image_feature_is_video = new_image_feature_is_video

        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

    else:
        image_features = self.encode_images(images)
        image_feature_is_video = [False] * int(image_features.shape[0]) if torch.is_tensor(image_features) and image_features.ndim >= 1 else []

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove padding using attention_mask
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    def _count_placeholders(seq):
        return int((seq == IMAGE_TOKEN_INDEX).sum().item())

    total_placeholders = sum(_count_placeholders(seq) for seq in input_ids)

    if torch.is_tensor(image_features):
        avail_feats = int(image_features.shape[0])
    else:
        avail_feats = int(len(image_features))

    if total_placeholders > avail_feats and avail_feats > 0:
        rank0_print(
            f"[WARN] placeholders({total_placeholders}) > image_features({avail_feats}). "
            f"Padding by repeating last feature."
        )
        pad_n = total_placeholders - avail_feats
        if torch.is_tensor(image_features):
            last = image_features[-1:].repeat(pad_n, 1, 1)
            image_features = torch.cat([image_features, last], dim=0)
            if len(image_feature_is_video) > 0:
                image_feature_is_video = list(image_feature_is_video) + [image_feature_is_video[-1]] * pad_n
        else:
            image_features = list(image_features) + [image_features[-1]] * pad_n
            if len(image_feature_is_video) > 0:
                image_feature_is_video = list(image_feature_is_video) + [image_feature_is_video[-1]] * pad_n

    new_input_embeds = []
    new_labels = []
    new_image_masks = []     # [B, seq] bool (pre-pad)
    new_fastv_ranges = []    # [B, list[(s,e),...]] ranges in (pre-pad, after truncation later)

    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = _count_placeholders(cur_input_ids)
        sample_selected_token_feats = []

        try:
            self.get_model().fastv_num_images = int(num_images)
        except Exception:
            pass

        if num_images == 0:
            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            cur_img_mask = torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            new_image_masks.append(cur_img_mask)
            new_fastv_ranges.append([])
            continue

        # split text around image tokens
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_labels = labels[batch_idx]

        cur_input_ids_noim, cur_labels_noim = [], []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])

        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds_all = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds_all, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []
        cur_new_img_mask_parts = []

        fastv_ranges_this = []
        running_len = 0  # IMPORTANT: reset per sample

        if torch.is_tensor(image_features):
            img_feat_list = [image_features[j] for j in range(image_features.shape[0])]
        else:
            img_feat_list = list(image_features)

        selected_indices = []
        keep_per_image = getattr(self.config, "mm_keep_per_image", 128)

        for slot_idx, feat in enumerate(img_feat_list):
            is_video_slot = False
            if slot_idx < len(image_feature_is_video):
                is_video_slot = bool(image_feature_is_video[slot_idx])

            if is_video_slot:
                idx = torch.arange(feat.shape[0], device=feat.device, dtype=torch.long)
            else:
                idx = get_selected_indices_for_images(
                    image_features=[feat],
                    keep_per_image=keep_per_image,
                    beta=3.0,
                    seed=321,
                    keep_order=True,
                    debug=False,
                    ablation_mode="full"
                )[0]
            selected_indices.append(idx)
        for i in range(num_images + 1):
            # ---- text segment ----
            txt_emb = cur_input_embeds_no_im[i]
            cur_new_input_embeds.append(txt_emb)
            cur_new_labels.append(cur_labels_noim[i])
            cur_new_img_mask_parts.append(torch.zeros(txt_emb.shape[0], dtype=torch.bool, device=txt_emb.device))
            running_len += int(txt_emb.shape[0])

            if i < num_images:
                # ---- image segment ----
                img_slot_idx = cur_image_idx
                if img_slot_idx >= len(img_feat_list):
                    img_slot_idx = len(img_feat_list) - 1  # fallback to last

                cur_image_features_full = img_feat_list[img_slot_idx]
                idx = selected_indices[img_slot_idx]

                if idx.device != cur_image_features_full.device:
                    idx = idx.to(cur_image_features_full.device)

                cur_image_features = cur_image_features_full.index_select(0, idx)  # (K_i, D)
                sample_selected_token_feats.append(cur_image_features)
                cur_image_idx += 1

                img_len = int(cur_image_features.shape[0])
                start = running_len
                end = running_len + img_len
                fastv_ranges_this.append((start, end))
                running_len = end

                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (img_len,),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                cur_new_img_mask_parts.append(torch.ones(img_len, dtype=torch.bool, device=cur_image_features.device))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        cur_new_labels = torch.cat(cur_new_labels, dim=0)
        cur_new_img_mask = torch.cat(cur_new_img_mask_parts, dim=0)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        new_image_masks.append(cur_new_img_mask)
        new_fastv_ranges.append(fastv_ranges_this)

    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        trunc_len = int(tokenizer_model_max_length)
        new_input_embeds = [x[:trunc_len] for x in new_input_embeds]
        new_labels = [x[:trunc_len] for x in new_labels]
        new_image_masks = [x[:trunc_len] for x in new_image_masks]

        clipped_ranges = []
        for ranges in new_fastv_ranges:
            r2 = []
            for (s, e) in ranges:
                if s >= trunc_len:
                    continue
                r2.append((s, min(e, trunc_len)))
            clipped_ranges.append(r2)
        new_fastv_ranges = clipped_ranges

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_labels[0].dtype,
        device=new_labels[0].device,
    )
    attention_mask2 = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    image_mask2 = torch.zeros((batch_size, max_len), dtype=torch.bool, device=attention_mask.device)
    position_ids2 = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    pad_left = (getattr(self.config, "tokenizer_padding_side", "right") == "left")

    for i, (cur_embed, cur_lab, cur_img_mask) in enumerate(zip(new_input_embeds, new_labels, new_image_masks)):
        cur_len = cur_embed.shape[0]

        if pad_left:
            pad_len = max_len - cur_len
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        torch.zeros((pad_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device),
                        cur_embed,
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_lab
                attention_mask2[i, -cur_len:] = True
                position_ids2[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids2.dtype, device=position_ids2.device)
                image_mask2[i, -cur_len:] = cur_img_mask

            if len(new_fastv_ranges[i]) > 0:
                new_fastv_ranges[i] = [(s + pad_len, e + pad_len) for (s, e) in new_fastv_ranges[i]]

        else:
            pad_len = max_len - cur_len
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_embed,
                        torch.zeros((pad_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_lab
                attention_mask2[i, :cur_len] = True
                position_ids2[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids2.dtype, device=position_ids2.device)
                image_mask2[i, :cur_len] = cur_img_mask

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels_out = None
    else:
        new_labels_out = new_labels_padded

    if _attention_mask is None:
        attention_mask_out = None
    else:
        attention_mask_out = attention_mask2.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids_out = None
    else:
        position_ids_out = position_ids2

    if getattr(self.config, "use_pos_skipping", False) and self.training and position_ids_out is not None:
        position_ids_out = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids_out[:, :split_position] += left_add
        position_ids_out[:, split_position:] += right_add

    return None, position_ids_out, attention_mask_out, past_key_values, new_input_embeds, new_labels_out
