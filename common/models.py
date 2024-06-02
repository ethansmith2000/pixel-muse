import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as attn_op
from functools import partial
from PIL import Image
from tqdm import tqdm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(dtype)


class AdaNorm(nn.Module):

    def __init__(self, in_dim, ada_dim, rms=False):
        super().__init__()
        self.ada_proj = nn.Linear(ada_dim, 2 * in_dim)
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if not rms else RMSNorm()

    def forward(self, hidden_states, ada_embed):
        scale, shift = self.ada_proj(ada_embed).chunk(2, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, time_dim, num_heads, mlp_ratio=4.0, rms =False):
        super().__init__()
        self.norm_attn1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) if rms else RMSNorm()
        self.attn1 = Attention(hidden_size, heads=num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) if rms else RMSNorm()
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn1(modulate(self.norm_attn1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x


class TransformerBlockNoAda(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_attn1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(hidden_size, heads=num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio)

    def forward(self, x):
        x = x + self.attn1(self.norm_attn1(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AttentionResampler(nn.Module):

    def __init__(self, dim, heads=1, scale_factor = 2.0):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.o = nn.Linear(dim, dim, bias=False) if heads > 1 else nn.Identity()
        self.h = heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        b, s, d = x.shape
        norm_x = self.norm(x)
        norm_x_new_size = F.interpolate(norm_x, scale_factor=self.scale_factor, mode='nearest')
        k, v = self.kv(norm_x).chunk(2, dim=-1)
        q = self.q(norm_x_new_size)
        q, k, v = map(lambda t: t.view(b, -1, self.h, d // self.h).transpose(1, 2), (q, k, v))
        attn_output = self.o(attn_op(q, k, v).transpose(1, 2).reshape(b, -1, d).to(q.dtype))
        return attn_output + x


class AttentionSkipResampler(nn.Module):

    def __init__(self, dim, heads=1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.o = nn.Linear(dim, dim, bias=False) if heads > 1 else nn.Identity()
        self.dh = dim // heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.context_norm = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x, context):
        b, s, d = x.shape
        q = self.q(self.norm(x))
        k, v = self.kv(self.context_norm(context)).chunk(2, dim=-1)
        q, k, v = map(lambda t: t.view(b, -1, self.h, d // self.h).transpose(1, 2), (q, k, v))
        attn_output = self.o(attn_op(q, k, v).transpose(1, 2).reshape(b, -1, d).to(q.dtype))

        return attn_output + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult), bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, 
                dim=768, 
                 heads=8,
                 v_act = False
                 ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.v_act = torch.nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=True)) if v_act else nn.Identity()
        self.o = nn.Linear(dim, dim, bias=False) if heads > 1 else nn.Identity()
        self.h = heads
    
    def forward(self, x):
        b, s, d = x.shape
        q, k, v = map(lambda t: t.view(b, -1, self.h, d // self.h).transpose(1, 2), self.qkv(x).chunk(3, dim=-1))
        v = self.v_act(v)
        return self.o(attn_op(q, k, v).transpose(1, 2).reshape(b, -1, d).to(q.dtype))



class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)


    def get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int,
        downscale_freq_shift: float = 1,
        max_period: int = 10000,
    ):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


    def forward(self, timestep):
        timestep = self.get_timestep_embedding(timestep, self.linear_1.in_features).to(self.linear_1.weight.device).to(self.linear_1.weight.dtype)
        return self.linear_2(self.act(self.linear_1(timestep)))



class PixelTokenizer(torch.nn.Module):
    def __init__(self, vocab_size, image_size, low=0.0, high=1.0, per_channel=False, num_channels=3, most_common_classes=None, dim=768):
        super().__init__()
        self.low = low
        self.high = high
        self.per_channel = per_channel
        self.embed_table = nn.Embedding(vocab_size + 1, dim)
        self.mask_token_id = vocab_size
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.h = self.w = image_size
        self.hw = image_size * image_size
        self.c = num_channels if per_channel else 1
        self.pos_embed = nn.Parameter(torch.randn(1, image_size * image_size * num_channels if per_channel else image_size * image_size,
                                                        dim) * 0.005)
        if per_channel:
            classes = torch.linspace(low, high, vocab_size)[None,None,:]
        else:
            classes = torch.tensor(most_common_classes)[None,None,:,:]

        self.register_buffer('classes', classes)

    def dists(self, x):
        classes = self.classes if len(self.classes.shape) == 4 else self.classes[...,None].repeat(1,1,1, self.num_channels)
        if self.per_channel:
            return (x.unsqueeze(-2) - classes).pow(2)
        else:
            return (x.unsqueeze(-2) - classes).pow(2).sum(-1, keepdim=True)

    def decode(self, indices, h, w):
        c = indices.shape[-1] // (h * w)
        if c > 1:
            indices = indices.reshape(-1, h * w, c)
            indices = indices.reshape(-1, h * w, c).reshape(-1, h, w, c).permute(0, 3, 1, 2)
            pixels = self.classes.squeeze()[indices]
        else:
            indices = indices.reshape(-1, h * w)
            pixels = self.classes.squeeze()[indices, :]
            c = pixels.shape[-1]
            pixels = pixels.reshape(-1, h, w, c).permute(0, 3, 1, 2)

        return pixels


    def encode(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        indices = self.dists(x).argmin(-2)
        indices = indices.reshape(b, h * w * indices.shape[-1])
        return indices

    def to_embs(self, indices):
        b, hwc = indices.shape
        c = hwc // (self.hw)
        embeds = self.embed_table(indices) + self.pos_embed.expand(b, -1, -1)
        # b, hw, c * dim -> b, c * dim, h, w
        # embeds = embeds.permute(0, 2, 1).reshape(b, -1, self.h, self.w * 3)
        return embeds



class ViT(nn.Module):
    def __init__(
        self,
        dim = 1536,
        time_dim = 384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes = 1000,
        image_size = 64,
        per_channel = False,
        most_common_classes=None,
        num_channels=3,
    ):
        super().__init__()
        self.dim = dim
        self.tokenizer = PixelTokenizer(num_classes, image_size, per_channel=per_channel, dim=dim, most_common_classes=most_common_classes, num_channels=num_channels)
        # self.proj_in = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj_in = nn.Linear(dim, dim)

        self.time_embed =  TimestepEmbedding(time_dim, time_dim) if time_dim is not None else None
        
        kwargs = dict(hidden_size=dim, num_heads=num_heads, mlp_ratio=mlp_ratio) if time_dim is None else dict(hidden_size=dim, time_dim=time_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        block = TransformerBlockNoAda if time_dim is None else TransformerBlock
        self.blocks = nn.ModuleList([block(**kwargs) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True)
        self.proj_out = nn.Linear(dim, num_classes)
        self.gradient_checkpointing = False


    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


    @torch.no_grad()
    def sample(self, batch_size=4, num_steps=50, generator=None):
        # ids = torch.randint(0, self.tokenizer.vocab_size, (batch_size, self.tokenizer.hw * self.tokenizer.c))
        ids = torch.full((batch_size, self.tokenizer.hw * self.tokenizer.c), self.tokenizer.mask_token_id, dtype=torch.long).to(next(self.parameters()).device)
        for i in tqdm(range(num_steps)):
            embs = self.tokenizer.to_embs(ids)
            preds = self(embs)
            probs = torch.nn.functional.softmax(preds, dim=-1)
            ids = torch.multinomial(probs.reshape(-1, self.tokenizer.vocab_size), 1, generator=generator).reshape(batch_size, -1)
        pixels = self.tokenizer.decode(ids, self.tokenizer.h, self.tokenizer.w)
        pixels = pixels.permute(0,2,3,1) * 255
        pixels = pixels.to(torch.uint8).cpu().clamp(0, 255).numpy()
        imgs = [Image.fromarray(p) for p in pixels]

        return imgs


    def forward(self, x, t=None, y=None):
        args = (self.time_embed(t)) if t is not None else ()
        x = self.proj_in(x)#.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.dim)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(partial(block, *args), x)
            else:
                x = block(x, *args)
        return self.proj_out(self.norm_out(x))


# class UViT(nn.Module):
#     def __init__(
#         self,
#         dim = 1536,
#         time_dim = 384,
#         depth=12,
#         num_heads=16,
#         mlp_ratio=4.0,
#         num_classes = 1000,
#         image_size = 64,
#         per_channel = False
#     ):
#         super().__init__()
#         self.dim = dim
#         self.tokenizer = PixelTokenizer(num_classes, image_size, per_channel=per_channel, dim=dim)
#         self.proj_in = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)

#         self.time_embed =  TimestepEmbedding(time_dim, time_dim) if time_dim is not None else None
        
#         kwargs = dict(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio) if time_dim is None else dict(dim=dim, time_dim=time_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
#         block = TransformerBlockNoAda if time_dim is None else TransformerBlock
#         self.blocks = nn.ModuleList([block(**kwargs) for _ in range(depth)])
#         self.norm_out = nn.LayerNorm(dim, elementwise_affine=True)
#         self.proj_out = nn.Linear(dim, num_classes)


#     def forward(self, x, t=None, y=None):
#         if self.time_embed is not None:
#             time_emb = self.time_embed(t)

#         x = self.proj_in(x)
#         for block in self.blocks:
#             x = block(x, time_emb) if t is not None else block(x)
#         return self.proj_out(self.norm_out(x))