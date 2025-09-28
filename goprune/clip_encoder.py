#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
# from calendar import c
# from msvcrt import kbhit
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder
import torch.nn.functional as F


from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward


class CLIPVisionTower_GOPrune(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            # attn_weights  = image_forward_outs.attentions[-2]
            # hidden_states = image_forward_outs.hidden_states[-2]
            attentions = image_forward_outs.attentions
            k = self.vision_tower._info["k"]
            c = self.vision_tower._info["c"] 
            r = self.vision_tower._info["r"]
            m = self.vision_tower._info["m"]
            q = self.vision_tower._info["q"]
            # hidden_states_save = self.extract_semantic_tokens_from_vision_outputs(hidden_states,image_forward_outs.attentions,k,c,r,m,q)
            # metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            # dominant_num =  self.vision_tower._info["dominant"]
            # contextual_num = self.vision_tower._info["contextual"]
            """
            每个象限保留得分前 25% 的 token，
            其余通过 2×2 滑窗，只保留每个窗口中得分最高的非重要 token（窗口没 token 则跳过），最终按位置排序输出。
            不固定输出 token 数量。
            """
            # === Step 1: attention score ===
            select_layer_start = m
            select_layer_end = m + q
            attn_mid_layers = attentions[select_layer_start:select_layer_end]  # list of tensors
            avg_attn_mid = torch.stack(attn_mid_layers).mean(dim=0).mean(dim=1)[0]  # [N, N]
            mean_col_score = avg_attn_mid.mean(dim=0)[1:]  # [576]
        
            # === Step 2: patch info ===
            grid_size = 24
            all_tokens = image_forward_outs.hidden_states[-2][0, 1:, :]  # [576, D]
            all_pos_ids = torch.arange(576, device=all_tokens.device)
            rows = torch.arange(grid_size, device=all_tokens.device).unsqueeze(1).repeat(1, grid_size).flatten()
            cols = torch.arange(grid_size, device=all_tokens.device).repeat(grid_size)
        
            important_token_list, important_pos_list = [], []
            fused_token_list, fused_pos_list = [], []
        
            # === Step 3: process each region ===
            for row_range, col_range in [
                ((0, 12), (0, 12)),
                ((0, 12), (12, 24)),
                ((12, 24), (0, 12)),
                ((12, 24), (12, 24)),
            ]:
                # mask for current region
                region_mask = (rows >= row_range[0]) & (rows < row_range[1]) & \
                            (cols >= col_range[0]) & (cols < col_range[1])
                region_indices = torch.nonzero(region_mask, as_tuple=False).squeeze(-1)
                region_scores = mean_col_score[region_indices]
        
                # Top 25% important tokens
                num_top = int(region_indices.shape[0] * 0.25)
                topk_indices = region_indices[torch.topk(region_scores, num_top).indices]
        
                # Non-important tokens
                nonimportant_mask = region_mask.clone()
                nonimportant_mask[topk_indices] = False
                nonimportant_indices = torch.nonzero(nonimportant_mask, as_tuple=False).squeeze(-1)
        
                # === 保留重要 token ===
                important_token_list.append(all_tokens[topk_indices])
                important_pos_list.append(all_pos_ids[topk_indices])
        
                # === 非重要 token滑窗 ===
                # 创建 region_grid 并填充分数（-inf 表示无 token）
                # region_grid = torch.full((row_range[1] - row_range[0], col_range[1] - col_range[0]),
                #                         float('-inf'), device=all_tokens.device)
                region_grid = torch.full(
                    (row_range[1] - row_range[0], col_range[1] - col_range[0]),
                    float("-inf"),
                    device=all_tokens.device,
                    dtype=mean_col_score.dtype,
                )


                region_r = rows[nonimportant_indices] - row_range[0]
                region_c = cols[nonimportant_indices] - col_range[0]
                region_grid[region_r, region_c] = mean_col_score[nonimportant_indices]
        
                # unfold 滑窗
                score_patches = F.unfold(region_grid.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)  # [1, 4, num_win]
                max_vals, max_idx = score_patches.max(dim=1)  # [1, num_win]
        
                # 只保留窗口中有有效 token 的位置
                valid_mask = (max_vals.squeeze(0) != float('-inf'))
        
                # 获取窗口的起始位置
                h_out = (region_grid.shape[0] - 2) // 2 + 1
                w_out = (region_grid.shape[1] - 2) // 2 + 1
                row_base = torch.arange(row_range[0], row_range[1] - 1, 2, device=all_tokens.device).view(-1, 1).repeat(1, w_out).flatten()
                col_base = torch.arange(col_range[0], col_range[1] - 1, 2, device=all_tokens.device).repeat(h_out)
        
                # 计算选中 token 的位置
                dr = (max_idx % 2).squeeze(0)  # 列偏移
                dc = (max_idx // 2).squeeze(0)  # 行偏移
                row_sel = row_base + dc
                col_sel = col_base + dr
        
                row_sel = row_sel[valid_mask]
                col_sel = col_sel[valid_mask]
                pos = row_sel * grid_size + col_sel
        
                fused_token_list.append(all_tokens[pos])
                fused_pos_list.append(pos)
        
            # === Step 4: merge + sort ===
            all_feats = torch.cat(important_token_list + fused_token_list, dim=0)
            all_pos = torch.cat(important_pos_list + fused_pos_list, dim=0)
            sorted_idx = all_pos.argsort()
            sorted_feats = all_feats[sorted_idx]
        
            final = sorted_feats.unsqueeze(0)  # [1, N, D]
            print(f"Final token shape: {final.shape}, token count: {final.shape[1]}")
            hidden_states_save = final

        return hidden_states_save


    # @staticmethod
    # def extract_semantic_tokens_from_vision_outputs(hidden_states, attentions,k,c,r,m,q):
    #     """
    #     每个象限保留得分前 25% 的 token，
    #     其余通过 2×2 滑窗，只保留每个窗口中得分最高的非重要 token（窗口没 token 则跳过），最终按位置排序输出。
    #     不固定输出 token 数量。
    #     """
    #     # === Step 1: attention score ===
    #     select_layer_start = m
    #     select_layer_end = m + q
    #     attn_mid_layers = attentions[select_layer_start:select_layer_end]  # list of tensors
    #     avg_attn_mid = torch.stack(attn_mid_layers).mean(dim=0).mean(dim=1)[0]  # [N, N]
    #     mean_col_score = avg_attn_mid.mean(dim=0)[1:]  # [576]
    
    #     # === Step 2: patch info ===
    #     grid_size = 24
    #     all_tokens = hidden_states[-2][0, 1:, :]  # [576, D]
    #     all_pos_ids = torch.arange(576, device=all_tokens.device)
    #     rows = torch.arange(grid_size, device=all_tokens.device).unsqueeze(1).repeat(1, grid_size).flatten()
    #     cols = torch.arange(grid_size, device=all_tokens.device).repeat(grid_size)
    
    #     important_token_list, important_pos_list = [], []
    #     fused_token_list, fused_pos_list = [], []
    
    #     # === Step 3: process each region ===
    #     for row_range, col_range in [
    #         ((0, 12), (0, 12)),
    #         ((0, 12), (12, 24)),
    #         ((12, 24), (0, 12)),
    #         ((12, 24), (12, 24)),
    #     ]:
    #         # mask for current region
    #         region_mask = (rows >= row_range[0]) & (rows < row_range[1]) & \
    #                       (cols >= col_range[0]) & (cols < col_range[1])
    #         region_indices = torch.nonzero(region_mask, as_tuple=False).squeeze(-1)
    #         region_scores = mean_col_score[region_indices]
    
    #         # Top 25% important tokens
    #         num_top = int(region_indices.shape[0] * 0.25)
    #         topk_indices = region_indices[torch.topk(region_scores, num_top).indices]
    
    #         # Non-important tokens
    #         nonimportant_mask = region_mask.clone()
    #         nonimportant_mask[topk_indices] = False
    #         nonimportant_indices = torch.nonzero(nonimportant_mask, as_tuple=False).squeeze(-1)
    
    #         # === 保留重要 token ===
    #         important_token_list.append(all_tokens[topk_indices])
    #         important_pos_list.append(all_pos_ids[topk_indices])
    
    #         # === 非重要 token滑窗 ===
    #         # 创建 region_grid 并填充分数（-inf 表示无 token）
    #         region_grid = torch.full((row_range[1] - row_range[0], col_range[1] - col_range[0]),
    #                                  float('-inf'), device=all_tokens.device)
    #         region_r = rows[nonimportant_indices] - row_range[0]
    #         region_c = cols[nonimportant_indices] - col_range[0]
    #         region_grid[region_r, region_c] = mean_col_score[nonimportant_indices]
    
    #         # unfold 滑窗
    #         score_patches = F.unfold(region_grid.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)  # [1, 4, num_win]
    #         max_vals, max_idx = score_patches.max(dim=1)  # [1, num_win]
    
    #         # 只保留窗口中有有效 token 的位置
    #         valid_mask = (max_vals.squeeze(0) != float('-inf'))
    
    #         # 获取窗口的起始位置
    #         h_out = (region_grid.shape[0] - 2) // 2 + 1
    #         w_out = (region_grid.shape[1] - 2) // 2 + 1
    #         row_base = torch.arange(row_range[0], row_range[1] - 1, 2, device=all_tokens.device).view(-1, 1).repeat(1, w_out).flatten()
    #         col_base = torch.arange(col_range[0], col_range[1] - 1, 2, device=all_tokens.device).repeat(h_out)
    
    #         # 计算选中 token 的位置
    #         dr = (max_idx % 2).squeeze(0)  # 列偏移
    #         dc = (max_idx // 2).squeeze(0)  # 行偏移
    #         row_sel = row_base + dc
    #         col_sel = col_base + dr
    
    #         row_sel = row_sel[valid_mask]
    #         col_sel = col_sel[valid_mask]
    #         pos = row_sel * grid_size + col_sel
    
    #         fused_token_list.append(all_tokens[pos])
    #         fused_pos_list.append(pos)
    
    #     # === Step 4: merge + sort ===
    #     all_feats = torch.cat(important_token_list + fused_token_list, dim=0)
    #     all_pos = torch.cat(important_pos_list + fused_pos_list, dim=0)
    #     sorted_idx = all_pos.argsort()
    #     sorted_feats = all_feats[sorted_idx]
    
    #     final = sorted_feats.unsqueeze(0)  # [1, N, D]
    #     print(f"Final token shape: {final.shape}, token count: {final.shape[1]}")
    #     return final


# class CLIPVisionTower_VisionZip(nn.Module):


#     @torch.no_grad()
#     def forward(self, images):
        
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
            
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
#             attn_weights  = image_forward_outs.attentions[-2]
#             hidden_states = image_forward_outs.hidden_states[-2]
#             metric = self.vision_tower.vision_model.encoder.layers[-2].metric
#             dominant_num =  self.vision_tower._info["dominant"]
#             contextual_num = self.vision_tower._info["contextual"]

#             ## Dominant Visual Tokens
#             cls_idx = 0
#             cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
#             cls_attention_sum = cls_attention.sum(dim=1)  
#             topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
#             all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
            
#             mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
#             dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
            
#             ### Filter
#             metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

#             hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
            
#             metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

#             ## Contextual Visual Tokens
#             step = max(1, metric_normalized.shape[1] // contextual_num)
#             target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
#             target_tokens = metric_normalized[:, target_indices, :]

#             tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
#             similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
#             assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
#             assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
#             counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
#             hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
#             aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
#             target_hidden = hidden_states_filtered[:, target_indices, :]  
            
#             contextual_tokens = target_hidden + aggregated_hidden

#             # Merge with target hidden states and concatenate
#             hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

#         return hidden_states_save, all_indices

        # return hidden_states_save, hidden_states, all_indices





