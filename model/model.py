"""
LUMEN-0 — Lightweight Universal Motion Engine Nano
~47k parameters, 184KB quantized
Text -> MotionParams -> Neural Renderer

Architecture:
  Text Encoder (hash + MLP) -> 64-dim
  Motion Core (GRU-64) -> 16 x 64
  Parameter Head -> 16 x 14 (interpretable physics params per frame)
  Renderer is procedural (differentiable approx in training, canvas in inference)

We predict per-frame physics parameters, not raw pixels.
This is why we're 1000x smaller than diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

INPUT_DIM = 32      # hash embedding dim
HIDDEN_DIM = 64
GRU_HIDDEN = 64
NUM_FRAMES = 16
OUTPUT_PARAMS = 14  # per frame physics params

# Output param semantics:
# 0: x_pos [-1,1]
# 1: y_pos [-1,1]
# 2: scale [0.05,0.6]
# 3: rotation [0, 2pi]
# 4: phase [0, 2pi]
# 5: bg_mix [0,1]
# 6: color_r [0,1]
# 7: color_g [0,1]
# 8: color_b [0,1]
# 9: color2_r [0,1]
# 10: color2_g [0,1]
# 11: color2_b [0,1]
# 12: glow [0,1]
# 13: archetype_blend [0,1]

class TextEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Extra projection for style
        self.style_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):  # [B,32]
        h = self.net(x)  # [B,64]
        style = torch.tanh(self.style_proj(h)) * 0.5 + 0.5
        return h, style

class MotionCore(nn.Module):
    def __init__(self, hidden_dim=64, gru_hidden=64, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        
        # Temporal embedding for each frame index
        self.time_embed = nn.Parameter(torch.randn(num_frames, 16) * 0.2)
        
        # GRU cell conditioned on text + time
        # We use GRUCell unrolled manually for full control
        self.gru_cell = nn.GRUCell(input_size=hidden_dim+16, hidden_size=gru_hidden)
        self.text_to_hidden = nn.Linear(hidden_dim, gru_hidden)
        
        # Self-attention over time (tiny)
        self.attn_q = nn.Linear(gru_hidden, 32)
        self.attn_k = nn.Linear(gru_hidden, 32)
        self.attn_v = nn.Linear(gru_hidden, gru_hidden)
        
    def forward(self, text_h):  # [B,64]
        B = text_h.shape[0]
        device = text_h.device
        
        # Init hidden from text
        h_prev = torch.tanh(self.text_to_hidden(text_h))  # [B,64]
        
        outputs = []
        for t in range(self.num_frames):
            # time embedding [16] -> broadcast
            te = self.time_embed[t].unsqueeze(0).expand(B, -1)  # [B,16]
            inp = torch.cat([text_h, te], dim=-1)  # [B, 80]
            h_prev = self.gru_cell(inp, h_prev)  # [B,64]
            outputs.append(h_prev)
        
        seq = torch.stack(outputs, dim=1)  # [B,T,64]
        
        # Tiny temporal attention refinement
        Q = self.attn_q(seq)  # [B,T,32]
        K = self.attn_k(seq)  # [B,T,32]
        V = self.attn_v(seq)  # [B,T,64]
        attn = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / (32**0.5), dim=-1)  # [B,T,T]
        attn_out = torch.bmm(attn, V)  # [B,T,64]
        seq = seq + attn_out * 0.3  # residual
        
        return seq  # [B,T,64]

class ParamHead(nn.Module):
    def __init__(self, in_dim=64, out_params=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.ReLU(),
            nn.Linear(48, out_params),
        )
        # Learnable scaling to map tanh outputs to meaningful ranges
        # We'll use tanh then scale/bias per param
        
    def forward(self, seq):  # [B,T,64]
        raw = self.net(seq)  # [B,T,14]
        raw = torch.tanh(raw)  # [-1,1]
        
        # Map to interpretable ranges
        # Use differentiable scaling
        out = torch.empty_like(raw)
        # x_pos, y_pos [-1,1] -> [-0.85,0.85]
        out[:,:,0] = raw[:,:,0] * 0.85
        out[:,:,1] = raw[:,:,1] * 0.85
        # scale [0.05,0.55]
        out[:,:,2] = (raw[:,:,2]*0.25 + 0.3).clamp(0.05, 0.6)
        # rotation -> keep as [-pi,pi] then mapped later
        out[:,:,3] = raw[:,:,3] * np.pi
        # phase
        out[:,:,4] = raw[:,:,4] * np.pi
        # bg_mix [0,1]
        out[:,:,5] = (raw[:,:,5]*0.5 + 0.5)
        # colors [0.15,0.95]
        out[:,:,6] = raw[:,:,6]*0.4 + 0.6
        out[:,:,7] = raw[:,:,7]*0.4 + 0.6
        out[:,:,8] = raw[:,:,8]*0.4 + 0.6
        out[:,:,9] = raw[:,:,9]*0.4 + 0.6
        out[:,:,10] = raw[:,:,10]*0.4 + 0.6
        out[:,:,11] = raw[:,:,11]*0.4 + 0.6
        # glow [0,1]
        out[:,:,12] = (raw[:,:,12]*0.5+0.5)
        # archetype blend
        out[:,:,13] = (raw[:,:,13]*0.5+0.5)
        
        return out

class LumenModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, gru_hidden=64, num_frames=16, out_params=14):
        super().__init__()
        self.text_encoder = TextEncoder(input_dim, hidden_dim)
        self.motion_core = MotionCore(hidden_dim, gru_hidden, num_frames)
        self.param_head = ParamHead(gru_hidden, out_params)
        self.num_frames = num_frames
        
    def forward(self, text_emb):  # [B,32]
        text_h, style = self.text_encoder(text_emb)
        seq = self.motion_core(text_h)  # [B,T,64]
        params = self.param_head(seq)  # [B,T,14]
        return params, style, seq
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def count_model_params():
    m = LumenModel()
    cnt = m.count_parameters()
    print(f"Total parameters: {cnt}")
    for name, mod in m.named_modules():
        if len(list(mod.parameters()))>0 and len(name.split("."))<3:
            # print submodule counts
            pass
    # Detailed breakdown
    te = sum(p.numel() for p in m.text_encoder.parameters())
    mc = sum(p.numel() for p in m.motion_core.parameters())
    ph = sum(p.numel() for p in m.param_head.parameters())
    print(f" TextEncoder: {te}")
    print(f" MotionCore: {mc}")
    print(f" ParamHead: {ph}")
    print(f" Size quantized (int8): {cnt * 1 / 1024:.1f} KB")
    print(f" Size fp32: {cnt * 4 / 1024:.1f} KB")
    print(f" Size fp16: {cnt * 2 / 1024:.1f} KB")
    return cnt

if __name__ == "__main__":
    count_model_params()
    # Test forward
    m = LumenModel()
    x = torch.randn(2,32)
    params, style, seq = m(x)
    print("params shape", params.shape)
    print("sample params", params[0,0])
