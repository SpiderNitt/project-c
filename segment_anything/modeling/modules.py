import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import xformers.ops as xops

class PropagationModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_num_obj = cfg.dataset.max_num_obj
        self.num_frames = cfg.dataset.num_frames
        self.num_tokens = cfg.num_tokens

        self.pos_embed_wt_image = nn.Parameter(torch.zeros((256)))
        
        self.attention = MemoryEfficientAttention(256, 256, 4)
        self.values_mlp = nn.Sequential(
            nn.Linear(256, 256), 
            nn.GELU(), 
            nn.Linear(256, 256)
        )

        self.dense_embedding_linear = nn.Sequential(
            nn.Linear(256, 256), 
            nn.GELU(), 
            nn.Linear(256, 256)
        )

        self.layer_norm_input = nn.LayerNorm(256)
        self.layer_norm_values_0 = nn.LayerNorm(256)
        self.layer_norm_values_1 = nn.LayerNorm(256)

    def forward(self, embeddings: dict, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        image_embeddings: (B, num_frames=3, 256, 64, 64)
        mask_embeddings: # (B, num_frames=2, P=3, 256, 64, 64)
        gt_mask: (3, H, W)
        prev_masks: (B, num_frames=2, num_obj=3, 256, 256)
        selector: (num_obj=3)
        """
        if "image_embeddings" in embeddings:
            curr_embeddings = embeddings["image_embeddings"][:, -1].permute(0, 2, 3, 1) # (B, 64, 64, 256)
            prev_frames_embeddings = embeddings["image_embeddings"][:, :-1].permute(0, 1, 3, 4, 2) # (B, num_frames=2, 64, 64 256)
        else:
            curr_embeddings = embeddings["current_frame_embeddings"].permute(0, 2, 3, 1)
            prev_frames_embeddings = embeddings["prev_frames_embeddings"].permute(0, 1, 3, 4, 2)
        
        mask_embeddings = embeddings["mask_embeddings"].permute(0, 1, 2, 4, 5, 3) # (B, num_frames=2, num_obj=3, 64, 64, 256)

        pos_embed = pos_embed.permute(0, 2, 3, 1)
        curr_embeddings = curr_embeddings + self.pos_embed_wt_image * pos_embed
        prev_frames_embeddings = prev_frames_embeddings + self.pos_embed_wt_image * pos_embed

        curr_embeddings = self.layer_norm_input(curr_embeddings)
        prev_frames_embeddings = self.layer_norm_input(prev_frames_embeddings)

        values = self.attention(curr_embeddings, prev_frames_embeddings, mask_embeddings) # (B, num_objects=3, 64, 64, [num_heads * self.head_dim] = 256)
        values = self.layer_norm_values_0(values)
        values_shortcut = values
        values = self.values_mlp(values)
        values = self.layer_norm_values_1(values + values_shortcut) # (B, num_objects=3, 64, 64, 256)
        dense_embeddings = self.dense_embedding_linear(values) # (B, num_objects=3, 64, 64, 256)
        sparse_embeddings = torch.empty((*dense_embeddings.shape[:2], 0, 256), device=dense_embeddings.device) # (B, num_objects=3, 1, 256)
        return sparse_embeddings, dense_embeddings, [self.pos_embed_wt_image]

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, [num_frames=1 * 64 * 64], [num_frames=2 * 64 * 64])
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v) # (B, num_heads, [num_frames=1 * 64 * 64], head_dim*3)
        return values, attention

    def forward(self, q, k, v, return_attention=False):
        batch_size = q.size(0)
        num_frames = k.size(1)
        num_objects = v.size(2)

        q = self.q_proj(q) # (B, 64, 64, embed_dim=256)
        k = self.k_proj(k) # (B, num_frames=2, 64, 64, embed_dim=256)
        v = self.v_proj(v) # (B, num_frames=2, num_obj=3, 64, 64 embed_dim=256)

        # Reshape to separate heads
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, [num_frames=1 * 64 * 64], head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, [num_frames=2 * 64 * 64], head_dim)
        v = v.permute(0, 1, 3, 4, 5, 2).reshape(batch_size, -1, self.num_heads, self.head_dim*num_objects).transpose(1, 2) # (B, num_heads, [num_frames=2 * 64 * 64], head_dim*num_objects)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        values = values.transpose(1, 2)
        values = values.reshape(batch_size, 64, 64, self.num_heads * self.head_dim, num_objects)
        values = values.permute(0, 4, 1, 2, 3)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class MemoryEfficientAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        num_frames = k.size(1)
        num_objects = v.size(2)

        q = self.q_proj(q) # (B, 64, 64, embed_dim=256)
        k = self.k_proj(k) # (B, num_frames=2, 64, 64, embed_dim=256)
        v = self.v_proj(v) # (B, num_frames=2, num_objects=3, 64, 64 embed_dim=256)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.permute(0, 1, 3, 4, 5, 2).reshape(batch_size, -1, self.num_heads, self.head_dim*num_objects)

        values = xops.memory_efficient_attention(q, k, v) # (B, 64*64, num_heads, self.head_dim*num_obj=3)
        values = values.reshape(batch_size, 64, 64, self.num_heads * self.head_dim, num_objects)
        values = values.permute(0, 4, 1, 2, 3) # (B, num_objects=3, 64, 64, [num_heads * self.head_dim] = 256)

        values = self.o_proj(values)
        return values
    
class MemoryEfficientSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)        
        self.qkv_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        num_objects = x.size(1)

        qkv = self.qkv_proj(x) # (B, num_objects=3, SeqLen, [embed_dim=256]*3)

        qkv = qkv.reshape(batch_size * num_objects, -1, self.num_heads, self.head_dim*3)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values = xops.memory_efficient_attention(q, k, v) # (B*num_obj, 64*64, num_heads, self.head_dim*3)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, num_objects, -1, self.embed_dim)

        values = self.o_proj(values)
        return values # (B, num_objects=3, 64, 64, [num_heads * self.head_dim] = 256)