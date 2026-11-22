# LUMEN Model — Training Details

## Architecture

```python
LUMEN-0: 53,678 params
├── TextEncoder (9,104)
│   ├── Linear 32→48 + LayerNorm + ReLU
│   ├── Linear 48→64 + LayerNorm + ReLU
│   └── style_proj 64→64 (tanh)
├── MotionCore (40,768)
│   ├── time_embed [16,16] learnable
│   ├── text_to_hidden 64→64
│   ├── GRUCell (input 80 → hidden 64)
│   │   ├── weight_ih [192,80], weight_hh [192,64]
│   │   └── bias 384
│   └── Temporal Attention
│       ├── Q 64→32, K 64→32, V 64→64
│       └── softmax(QKᵀ/√32) V
└── ParamHead (3,806)
    ├── Linear 64→48 + ReLU
    └── Linear 48→14 + tanh + scaling
```

Outputs 14 physics params per frame: x, y, scale, rotation, phase, bgMix, color1(3), color2(3), glow, archetypeBlend.

## Dataset

Synthetic procedural generation in `dataset.py`:

- 1,600 train + 200 val
- 8 archetypes × ~200 samples each
- Prompt templates with color variation + suffix noise
- Ground truth via analytic rendering at 64×64, supervised via physics params

## Training

- Optimizer: AdamW lr=1.5e-3, weight_decay 1e-4
- Scheduler: CosineAnnealing T=45, eta_min 1e-5
- Loss: Weighted MSE (pos/scale/color weighted higher) + temporal smoothness 0.15
- Batch 16, 45 epochs, CPU

## Results

```
Epoch 01: train 0.6344 | val 0.6218
Epoch 05: train 0.5353 | val 0.6088
...
Epoch 45: train 0.5177 | val 0.6249
Best val 0.608
```

## Export

`train.py` saves:
- `model/weights/lumen_best.pt` (torch)
- `public/model-weights.json` (lists for JS)
- `public/model-meta.json` (metadata + history)

JS inference in `lib/lumen.ts` mirrors PyTorch exactly, including GRU gating and LayerNorm.
