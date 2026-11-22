"""
LUMEN Dataset - Synthetic Motion Primitives
Generates procedural videos conditioned on text prompts
10,000 samples, 8 motion archetypes, fully deterministic
"""

import torch
import numpy as np
import math
import random
from typing import Tuple, Dict, List

# Motion archetypes - each maps to a visual program
ARCHETYPES = {
    "bounce": {
        "keywords": ["bouncing", "ball", "bounce", "jumping", "elastic"],
        "description": "Elastic bouncing ball with gravity & restitution",
        "color_palette": [[0.9, 0.3, 0.3], [0.2, 0.6, 0.9], [0.3, 0.9, 0.5]],
    },
    "rotate": {
        "keywords": ["rotating", "spinning", "square", "cube", "rotation"],
        "description": "Rigid rotation with perspective skew",
        "color_palette": [[0.9, 0.7, 0.2], [0.8, 0.3, 0.6], [0.4, 0.4, 0.9]],
    },
    "expand": {
        "keywords": ["expanding", "pulse", "circle", "breathing", "growing"],
        "description": "Radial expansion/contraction with easing",
        "color_palette": [[0.3, 0.9, 0.7], [0.9, 0.4, 0.4], [0.5, 0.7, 0.9]],
    },
    "wave": {
        "keywords": ["wave", "ocean", "flowing", "liquid", "water"],
        "description": "Sine wave displacement field",
        "color_palette": [[0.2, 0.5, 0.9], [0.1, 0.8, 0.8], [0.3, 0.4, 0.9]],
    },
    "orbit": {
        "keywords": ["orbit", "circular", "planets", "spiral", "galaxy"],
        "description": "Keplerian orbital mechanics",
        "color_palette": [[0.6, 0.3, 0.9], [0.9, 0.6, 0.2], [0.2, 0.9, 0.8]],
    },
    "particles": {
        "keywords": ["particles", "starfield", "snow", "dust", "sparks"],
        "description": "Stochastic particle swarm dynamics",
        "color_palette": [[0.95, 0.9, 0.4], [0.9, 0.9, 0.95], [0.7, 0.85, 1.0]],
    },
    "gradient": {
        "keywords": ["gradient", "sunset", "aurora", "sky", "nebula"],
        "description": "Temporal color field advection",
        "color_palette": [[0.9, 0.4, 0.6], [0.4, 0.6, 0.9], [0.9, 0.7, 0.3]],
    },
    "orbit2": {  # alias for spiral
        "keywords": ["spiral", "vortex", "whirl", "twist", "helix"],
        "description": "Logarithmic spiral vortex",
        "color_palette": [[0.5, 0.9, 0.4], [0.9, 0.3, 0.5], [0.3, 0.7, 0.9]],
    },
}

# Unified archetype list
ARCHETYPE_NAMES = list(ARCHETYPES.keys())
NUM_ARCHETYPES = len(ARCHETYPE_NAMES)

# Prompt templates for natural language variation
PROMPT_TEMPLATES = {
    "bounce": [
        "a {color} bouncing ball with elastic motion",
        "{color} ball bouncing in slow motion",
        "bouncy {color} sphere with gravity",
        "elastic {color} ball jumping",
        "a playful {color} ball bouncing",
    ],
    "rotate": [
        "a {color} rotating square spinning fast",
        "{color} square rotating in 3d",
        "spinning {color} cube rotation",
        "a {color} square with continuous rotation",
        "rotating {color} geometric shape",
    ],
    "expand": [
        "a {color} expanding circle breathing pulse",
        "{color} circle expanding and contracting",
        "pulsing {color} ring expanding",
        "a {color} circle growing and shrinking",
        "breathing {color} circular pulse",
    ],
    "wave": [
        "a {color} wave flowing like ocean water",
        "{color} liquid wave motion",
        "flowing {color} water waves",
        "ocean waves in {color} tones",
        "smooth {color} wave pattern",
    ],
    "orbit": [
        "a {color} orbiting planets circular motion",
        "{color} planets orbiting in space",
        "circular {color} orbit galaxy",
        "orbiting {color} celestial bodies",
        "a {color} orbital system spinning",
    ],
    "particles": [
        "a {color} starfield particles drifting",
        "{color} particles floating dust",
        "sparkling {color} particles in motion",
        "drifting {color} snow particles",
        "a {color} particle field flowing",
    ],
    "gradient": [
        "a {color} sunset gradient sky animation",
        "{color} gradient aurora flowing",
        "shifting {color} nebula gradient",
        "a {color} sky with gradient motion",
        "soft {color} gradient clouds moving",
    ],
    "orbit2": [
        "a {color} spiral vortex twisting",
        "{color} spiral galaxy rotating",
        "twisting {color} vortex motion",
        "a {color} helical spiral",
        "whirling {color} spiral pattern",
    ],
}

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "amber"]

def generate_prompt(archetype: str) -> str:
    """Generate natural language prompt for archetype"""
    template = random.choice(PROMPT_TEMPLATES[archetype])
    color = random.choice(COLORS)
    prompt = template.format(color=color)
    # Add random suffix variation for richness
    suffixes = ["", ", cinematic", ", 4k smooth", ", ultra smooth", " in zero gravity", " high detail", ""]
    prompt += random.choice(suffixes)
    return prompt, color

def procedural_render_frame(archetype: str, t: float, params: dict, H: int = 64, W: int = 64) -> np.ndarray:
    """
    Procedural ground truth renderer - differentiable-ish via numpy
    t: normalized time [0,1], params: dict of motion params
    Returns HxWx3 frame in [0,1]
    """
    y, x = np.mgrid[0:H, 0:W]
    x_norm = (x - W/2) / (W/2)  # [-1,1]
    y_norm = (y - H/2) / (H/2)
    
    # unpack params with defaults
    speed = params.get("speed", 1.0)
    scale = params.get("scale", 0.3)
    phase = params.get("phase", 0.0)
    color = np.array(params.get("color", [0.9, 0.3, 0.5]))
    color2 = np.array(params.get("color2", [0.2, 0.6, 0.9]))
    
    frame = np.zeros((H, W, 3), dtype=np.float32)
    
    if archetype in ["bounce"]:
        # Bouncing ball - parabolic motion with bounce
        # y = abs(sin(t * pi * freq)) with gravity bias
        freq = 2.0 * speed
        # Elastic bounce: piecewise parabola
        bounce_t = (t * freq + phase) % 1.0
        # gravity fall then bounce
        y_pos = 0.6 * (1 - 4 * (bounce_t - 0.5)**2)  # parabola peak at 0.5
        y_pos = np.clip(y_pos, -0.8, 0.8)
        x_pos = 0.3 * np.sin(t * 1.5 + phase)  # slight horizontal sway
        
        # SDF circle
        r = scale
        dist = np.sqrt((x_norm - x_pos)**2 + (y_norm - y_pos)**2)
        alpha = np.clip(1 - (dist - r) * 8, 0, 1)  # soft edge
        # shadow
        shadow = np.clip(1 - np.abs(x_norm - x_pos) * 2, 0, 1) * np.clip(y_pos + 0.8, 0, 1) * 0.15
        frame = color * alpha[:, :, None] + shadow[:, :, None] * np.array([0,0,0])
        # background gradient
        bg = 0.05 + 0.08 * (1 - y_norm)[:, :, None] * np.array([1,1,1])
        frame = frame + bg * (1 - alpha[:, :, None])
        
    elif archetype == "rotate":
        angle = t * 2 * np.pi * speed + phase
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotate coordinates
        xr = x_norm * cos_a - y_norm * sin_a
        yr = x_norm * sin_a + y_norm * cos_a
        # Square SDF
        s = scale + 0.1 * np.sin(t * 3)
        d = np.maximum(np.abs(xr), np.abs(yr)) - s
        alpha = np.clip(1 - d * 12, 0, 1)
        # inner highlight
        inner = np.clip(1 - d*20, 0, 1) * 0.3
        # checker for depth
        checker = (np.floor(xr*4) + np.floor(yr*4)) % 2 * 0.1
        frame = (color * alpha[:, :, None] + 
                color2 * inner[:, :, None] +
                checker[:, :, None] * alpha[:, :, None])
        bg = 0.08 + 0.05 * np.sin(x_norm*2)[:, :, None]
        frame = frame * alpha[:, :, None] + bg * (1-alpha[:, :, None])
        
    elif archetype == "expand":
        r = scale * (0.5 + 0.5 * np.sin(t * 2 * np.pi * speed + phase) + 0.5)
        r = np.clip(r, 0.05, 0.7)
        dist = np.sqrt(x_norm**2 + y_norm**2)
        # Ring with thickness
        thickness = 0.08
        ring = np.exp(-((dist - r)**2) / (2*thickness**2))
        # fill
        fill = (dist < r).astype(float) * 0.3
        alpha = np.clip(ring + fill, 0, 1)
        # pulsing glow
        glow = np.exp(-dist*2) * 0.15 * (0.5 + 0.5*np.sin(t*4))
        frame = color * alpha[:, :, None] + color2 * fill[:, :, None] + glow[:, :, None]*color
        bg = 0.05 + 0.03 * dist[:, :, None]
        frame = frame + bg * (1 - np.clip(alpha,0,1)[:, :, None])*0.5
        
    elif archetype == "wave":
        # Wave displacement
        wave = 0.3 * np.sin(x_norm * 6 * speed + t * 6 + phase) * np.exp(-y_norm**2 * 2)
        # Create wave field as color bands
        v = y_norm - wave
        bands = 0.5 + 0.5 * np.sin(v*10 + t*4)
        # horizontal gradient
        h_grad = 0.5 + 0.5 * x_norm
        frame[:, :, 0] = color[0]*bands + color2[0]*(1-bands)*0.5 + h_grad*0.1
        frame[:, :, 1] = color[1]*bands + color2[1]*(1-bands)*0.5
        frame[:, :, 2] = color[2]*bands + color2[2]*(1-bands)*0.5 + h_grad*0.2
        # Add shimmer
        shimmer = 0.1 * np.sin(x_norm*12 + t*8) * np.exp(-v**2*10)
        frame += shimmer[:, :, None]
        
    elif archetype == "orbit":
        # Two orbiting bodies
        angle1 = t * 2*np.pi* speed + phase
        angle2 = angle1 + np.pi
        r_orbit = 0.45
        x1, y1 = r_orbit*np.cos(angle1), r_orbit*np.sin(angle1)*0.6
        x2, y2 = r_orbit*np.cos(angle2), r_orbit*np.sin(angle2)*0.6
        # Bodies
        d1 = np.sqrt((x_norm - x1)**2 + (y_norm - y1)**2)
        d2 = np.sqrt((x_norm - x2)**2 + (y_norm - y2)**2)
        a1 = np.clip(1 - (d1- scale*0.7)*18, 0,1)
        a2 = np.clip(1 - (d2- scale*0.5)*18, 0,1)
        # Trails
        trail = np.zeros((H,W))
        for k in range(5):
            tt = max(0, t - k*0.04)
            aa1 = tt*2*np.pi*speed + phase
            xx1, yy1 = r_orbit*np.cos(aa1), r_orbit*np.sin(aa1)*0.6
            dd = np.sqrt((x_norm-xx1)**2 + (y_norm-yy1)**2)
            trail += np.exp(-dd*15) * (0.1 - k*0.015)
        frame = (color*a1[:,:,None] + color2*a2[:,:,None] + 
                trail[:,:,None]*color*0.5)
        # center glow
        center = np.exp(-np.sqrt(x_norm**2 + y_norm**2)*3)*0.2
        frame += center[:,:,None]* np.array([1,1,0.8])
        bg = 0.02 + 0.03*(1+np.sin(x_norm*2-y_norm*2))[:,:,None]
        frame = frame + bg*(1- np.clip(a1+a2,0,1)[:,:,None])
        
    elif archetype == "particles":
        frame = np.ones((H,W,3))*0.03
        # starfield drift
        num_p = 18
        rng = np.random.RandomState(hash(archetype+str(int(phase*100))) % 2**31)
        # Use deterministic pseudo-random based on params
        np.random.seed(int((phase*1000)%100000))
        for i in range(num_p):
            # each particle drifts
            px = (np.sin(t* speed + i*0.7 + phase) *0.9 + np.cos(i*0.3)*0.5)
            py = (np.cos(t* speed*0.7 + i*1.1) *0.8 + np.sin(i*0.5)*0.3)
            # wrap
            px = np.clip(px, -0.9, 0.9)
            py = np.clip(py, -0.9, 0.9)
            d = np.sqrt((x_norm-px)**2 + (y_norm-py)**2)
            size = 0.02 + 0.015*np.sin(t*3+i)
            p_alpha = np.exp(-d/(size)) * (0.6+0.4*np.sin(t*4+i))
            intensity = 0.5+0.5*np.sin(t*2+i)
            col = color if i%2==0 else color2
            frame += p_alpha[:,:,None]* col * intensity
        # subtle bg nebula
        neb = 0.08 * np.sin(x_norm*3 + t) * np.cos(y_norm*2) 
        frame[:,:,0] += np.clip(neb,0,0.08)
        
    elif archetype == "gradient":
        # Shifting gradient field
        shift = t*0.6*speed
        # Perlin-like gradient via sin
        g1 = 0.5+0.5*np.sin(x_norm*2 + shift + phase)
        g2 = 0.5+0.5*np.cos(y_norm*2 + shift*0.7)
        mix = g1*0.5+g2*0.5
        # color lerp between color and color2 based on mix + time
        for c in range(3):
            frame[:,:,c] = color[c]*mix + color2[c]*(1-mix)
        # Add moving clouds
        cloud = 0.15*np.sin(x_norm*4 - shift*2) * np.sin(y_norm*3 + shift)
        frame += np.clip(cloud,0,0.15)[:,:,None]
        # vignette
        vign = 1 - np.sqrt(x_norm**2 + y_norm**2)*0.3
        frame *= np.clip(vign,0.6,1)[:,:,None]
        
    elif archetype == "orbit2":  # spiral
        angle = np.arctan2(y_norm, x_norm)
        rad = np.sqrt(x_norm**2 + y_norm**2)
        spiral = angle + rad*4 - t*5*speed
        spiral_val = 0.5+0.5*np.sin(spiral + phase*3)
        # arms
        arms = np.exp(-np.minimum(np.abs(spiral_val-0.5)*4, 2))*0.8
        # center fade
        fade = 1 - np.exp(-rad*4)
        alpha = arms * fade
        frame = color * alpha[:,:,None] + color2 * (1-alpha)[:,:,None]*0.15
        # rotate colors over time
        hue_shift = 0.05*np.sin(t*2)
        frame = np.clip(frame + hue_shift, 0,1)
        bg = 0.03 + 0.02*np.sin(rad*8 - t*4)[:,:,None]
        frame = frame*0.9 + bg*0.3
        
    frame = np.clip(frame, 0, 1)
    return frame.astype(np.float32)

def generate_video_sample(archetype: str, H=64, W=64, num_frames=16) -> Tuple[np.ndarray, Dict, str]:
    """Generate full video for one prompt"""
    prompt, color_name = generate_prompt(archetype)
    
    # Random motion params within archetype-appropriate range
    palette = ARCHETYPES[archetype]["color_palette"]
    c1 = np.array(random.choice(palette)) + np.random.randn(3)*0.05
    c2 = np.array(random.choice(palette)) + np.random.randn(3)*0.05
    c1 = np.clip(c1, 0.1, 1.0)
    c2 = np.clip(c2, 0.1, 1.0)
    
    params = {
        "speed": np.random.uniform(0.7, 1.8),
        "scale": np.random.uniform(0.15, 0.4),
        "phase": np.random.uniform(0, 2*np.pi),
        "color": c1.tolist(),
        "color2": c2.tolist(),
        "color_name": color_name,
    }
    
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames>1 else 0
        frame = procedural_render_frame(archetype, t, params, H, W)
        frames.append(frame)
    video = np.stack(frames, axis=0)  # T,H,W,3
    return video, params, prompt

class LumenDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=5000, H=64, W=64, num_frames=16, split="train"):
        self.num_samples = num_samples
        self.H, self.W = H, W
        self.num_frames = num_frames
        self.split = split
        # Pre-generate deterministically by seed
        self.samples = []
        base_seed = 42 if split=="train" else 2024
        for i in range(num_samples):
            random.seed(base_seed + i*997)
            np.random.seed(base_seed + i*997)
            arch = random.choice(ARCHETYPE_NAMES)
            video, params, prompt = generate_video_sample(arch, H, W, num_frames)
            # Store as tensors: video [T,3,H,W] for torch
            video_t = torch.from_numpy(video).permute(0,3,1,2).float()  # T,3,H,W
            self.samples.append({
                "video": video_t,
                "prompt": prompt,
                "archetype": arch,
                "archetype_id": ARCHETYPE_NAMES.index(arch),
                "params": params,
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_text_embedding(prompt: str, dim=32) -> np.ndarray:
    """
    Deterministic lightweight text embedding - hash-based random projection
    No transformer needed, fully reproducible, zero params
    Produces 32-dim vector that correlates with archetype semantics
    """
    # Lowercase, simple tokenize
    tokens = prompt.lower().replace(",", " ").replace(".", " ").split()
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        # Hash token to deterministic vector via numpy RandomState
        h = hash(tok) % (2**32)
        rs = np.random.RandomState(h % (2**31-1))
        tok_vec = rs.randn(dim).astype(np.float32) * 0.5
        # Keyword boosting for archetypes
        for arch_id, arch in enumerate(ARCHETYPE_NAMES):
            keywords = ARCHETYPES[arch]["keywords"]
            if tok in keywords:
                # Boost dimension corresponding to archetype
                boost_dim = arch_id % dim
                tok_vec[boost_dim] += 1.5
                tok_vec[(boost_dim+1)%dim] += 0.8
        vec += tok_vec
    if len(tokens) > 0:
        vec /= len(tokens)
    # Add small archetype bias from full prompt matching
    # Check if prompt contains any archetype keyword -> shift embedding
    prompt_low = prompt.lower()
    for arch_id, arch in enumerate(ARCHETYPE_NAMES):
        if any(kw in prompt_low for kw in ARCHETYPES[arch]["keywords"]):
            vec[arch_id % dim] += 0.7
    # L2 norm
    n = np.linalg.norm(vec) + 1e-8
    vec = vec / n
    return vec

if __name__ == "__main__":
    # Smoke test
    ds = LumenDataset(num_samples=2, H=64, W=64, num_frames=4)
    print(f"Dataset len {len(ds)}")
    s = ds[0]
    print(s["prompt"], s["archetype"], s["video"].shape)
    emb = get_text_embedding(s["prompt"])
    print("emb", emb[:5], "norm", np.linalg.norm(emb))
